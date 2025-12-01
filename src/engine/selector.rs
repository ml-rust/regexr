//! Engine selector - chooses the optimal execution strategy.

use crate::hir::{Hir, HirExpr};
use crate::nfa::Nfa;
use crate::vm::is_shift_or_compatible;

/// Recursively checks if an HIR expression contains UnicodeCpClass nodes.
/// These require PikeVM because they use the CodepointClass instruction
/// which does codepoint-level matching instead of byte-level DFA transitions.
fn hir_uses_codepoint_class(expr: &HirExpr) -> bool {
    match expr {
        HirExpr::UnicodeCpClass(_) => true,
        HirExpr::Concat(exprs) | HirExpr::Alt(exprs) => {
            exprs.iter().any(hir_uses_codepoint_class)
        }
        HirExpr::Repeat(r) => hir_uses_codepoint_class(&r.expr),
        HirExpr::Capture(c) => hir_uses_codepoint_class(&c.expr),
        HirExpr::Lookaround(l) => hir_uses_codepoint_class(&l.expr),
        HirExpr::Empty
        | HirExpr::Literal(_)
        | HirExpr::Class(_)
        | HirExpr::Anchor(_)
        | HirExpr::Backref(_) => false,
    }
}

/// The selected engine type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineType {
    /// PikeVM - for patterns with lookarounds or non-greedy quantifiers.
    PikeVm,
    /// BacktrackingVm - for patterns with backreferences (parity with BacktrackingJit).
    BacktrackingVm,
    /// Shift-Or - for small patterns (≤64 character positions).
    ShiftOr,
    /// Lazy DFA - default for most patterns.
    LazyDfa,
    /// JIT-compiled DFA - when available and beneficial.
    #[cfg(feature = "jit")]
    Jit,
}

/// Selects the optimal engine for a given NFA (legacy API).
/// Note: For Shift-Or selection, use `select_engine_from_hir` instead.
pub fn select_engine(nfa: &Nfa) -> EngineType {
    // Backreferences use BacktrackingVm (parity with BacktrackingJit)
    if nfa.has_backrefs {
        return EngineType::BacktrackingVm;
    }
    // Lookarounds require PikeVM
    if nfa.has_lookaround {
        return EngineType::PikeVm;
    }

    // Default to Lazy DFA
    // Shift-Or requires HIR for Glushkov construction
    EngineType::LazyDfa
}

/// Selects the optimal engine for a given HIR.
/// This is the preferred API as it can properly evaluate Shift-Or compatibility
/// using Glushkov construction.
pub fn select_engine_from_hir(hir: &Hir) -> EngineType {
    // Backreferences use BacktrackingVm (parity with BacktrackingJit)
    if hir.props.has_backrefs {
        return EngineType::BacktrackingVm;
    }
    // Lookarounds and non-greedy quantifiers require PikeVM
    // (non-greedy needs the epsilon-ordering semantics of Thompson NFA + PikeVM)
    if hir.props.has_lookaround || hir.props.has_non_greedy {
        return EngineType::PikeVm;
    }

    // Large Unicode classes that use CodepointClass instruction need PikeVM.
    // However, with threshold=500, most Unicode classes now use UTF-8 byte transitions
    // which LazyDFA handles efficiently. Only fall back to PikeVM if CodepointClass
    // is actually used (very rare - only for classes with >500 UTF-8 sequences).
    if hir.props.has_large_unicode_class {
        // Check if the HIR actually uses UnicodeCpClass (CodepointClass instruction)
        // If it's just many UTF-8 byte sequences, LazyDFA handles it fine
        if hir_uses_codepoint_class(&hir.expr) {
            return EngineType::PikeVm;
        }
        // Otherwise, use LazyDFA - it handles UTF-8 byte transitions well
        return EngineType::LazyDfa;
    }

    // Anchors (^, $) are now supported by LazyDFA.
    // Shift-Or doesn't support anchors, so skip it for those patterns.
    if hir.props.has_anchors {
        return EngineType::LazyDfa;
    }

    // Small patterns (≤64 character positions) use Shift-Or
    // Shift-Or uses Glushkov NFA (ε-free) for bit-parallel execution
    // Shift-Or now supports word boundaries (\b) at start and end of pattern
    if is_shift_or_compatible(hir) {
        return EngineType::ShiftOr;
    }

    // Word boundaries are now supported by LazyDFA using character-class augmented states.
    // This is a fallback for patterns too large for Shift-Or.
    if hir.props.has_word_boundary {
        return EngineType::LazyDfa;
    }

    // Default to Lazy DFA
    EngineType::LazyDfa
}

/// Runtime capabilities.
#[derive(Debug, Clone, Copy, Default)]
pub struct Capabilities {
    /// Whether AVX2 SIMD is available.
    #[cfg(feature = "simd")]
    pub has_avx2: bool,
    /// Whether JIT is available.
    #[cfg(feature = "jit")]
    pub has_jit: bool,
}

impl Capabilities {
    /// Detects runtime capabilities.
    pub fn detect() -> Self {
        Self {
            #[cfg(feature = "simd")]
            has_avx2: Self::detect_avx2(),
            #[cfg(feature = "jit")]
            has_jit: Self::detect_jit(),
        }
    }

    #[cfg(feature = "simd")]
    fn detect_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    #[cfg(feature = "jit")]
    fn detect_jit() -> bool {
        // JIT is available on x86-64 Linux/macOS/Windows
        cfg!(target_arch = "x86_64")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::nfa::compile;
    use crate::parser::parse;

    fn get_engine_from_hir(pattern: &str) -> EngineType {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        select_engine_from_hir(&hir)
    }

    fn get_engine_from_nfa(pattern: &str) -> EngineType {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = compile(&hir).unwrap();
        select_engine(&nfa)
    }

    #[test]
    fn test_simple_pattern_uses_shift_or() {
        // Simple patterns should use Shift-Or (via HIR-based selection)
        let engine = get_engine_from_hir("abc");
        assert_eq!(engine, EngineType::ShiftOr);
    }

    #[test]
    fn test_long_pattern_uses_lazy_dfa() {
        // Long patterns (>64 positions) should use Lazy DFA
        let long_pattern = "a".repeat(100);
        let engine = get_engine_from_hir(&long_pattern);
        assert_eq!(engine, EngineType::LazyDfa);
    }

    #[test]
    fn test_backref_uses_backtracking() {
        // Backreferences require BacktrackingVm
        let ast = parse(r"(a)\1").unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = compile(&hir).unwrap();

        // Test both selection APIs
        if nfa.has_backrefs {
            assert_eq!(select_engine(&nfa), EngineType::BacktrackingVm);
        }
        if hir.props.has_backrefs {
            assert_eq!(select_engine_from_hir(&hir), EngineType::BacktrackingVm);
        }
    }

    #[test]
    fn test_nfa_api_defaults_to_lazy_dfa() {
        // NFA-based selection can't check Shift-Or compatibility
        // (would need to rebuild Glushkov), so defaults to LazyDfa
        let engine = get_engine_from_nfa("abc");
        assert_eq!(engine, EngineType::LazyDfa);
    }

    #[test]
    fn test_word_boundary_uses_lazy_dfa() {
        // Word boundary patterns always use LazyDFA
        // ShiftOr does not support word boundaries - they are complex to handle correctly
        assert_eq!(get_engine_from_hir(r"\bthe\b"), EngineType::LazyDfa);
        assert_eq!(get_engine_from_hir(r"\bword\b"), EngineType::LazyDfa);
        assert_eq!(get_engine_from_hir(r"\b\d+\b"), EngineType::LazyDfa);
        assert_eq!(get_engine_from_hir(r"a\Bb"), EngineType::LazyDfa);

        // Long patterns with word boundaries should use LazyDFA
        let long_pattern = format!(r"\b{}\b", "a".repeat(100));
        assert_eq!(get_engine_from_hir(&long_pattern), EngineType::LazyDfa);
    }

    #[test]
    fn test_anchors_use_lazy_dfa() {
        // Anchors now use LazyDFA (implemented in DFA)
        assert_eq!(get_engine_from_hir(r"^hello"), EngineType::LazyDfa);
        assert_eq!(get_engine_from_hir(r"world$"), EngineType::LazyDfa);
        assert_eq!(get_engine_from_hir(r"^hello$"), EngineType::LazyDfa);
        assert_eq!(get_engine_from_hir(r"(?m)^line"), EngineType::LazyDfa);
    }
}
