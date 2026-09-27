//! Engine selector - chooses the optimal execution strategy.

use crate::hir::{Hir, HirExpr};
use crate::nfa::Nfa;
use crate::vm::{is_shift_or_compatible, is_shift_or_wide_compatible};

/// Recursively checks if an HIR expression contains UnicodeCpClass nodes.
/// These rule out the byte-level engines because they use the CodepointClass
/// instruction, which matches a whole codepoint (UTF-8 decode + range check)
/// instead of taking byte-level DFA transitions. Only the tagged NFA and the
/// PikeVM execute that instruction.
fn hir_uses_codepoint_class(expr: &HirExpr) -> bool {
    match expr {
        HirExpr::UnicodeCpClass(_) => true,
        HirExpr::Concat(exprs) | HirExpr::Alt(exprs) => exprs.iter().any(hir_uses_codepoint_class),
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

/// Whether no two branches can begin at the same byte, so none of them competes
/// with another for a match at a given position.
///
/// Branches carrying an assertion are excluded outright. `a|b$` has disjoint
/// first bytes, but the `$` is a zero-width condition these engines resolve for
/// the pattern as a whole rather than per branch, so "only one branch is viable
/// here" stops being the only thing that decides the match.
fn branches_start_disjointly(branches: &[HirExpr]) -> bool {
    let mut claimed = [false; 256];
    for branch in branches {
        if contains_assertion(branch) {
            return false;
        }
        let Some(first) = first_bytes(branch) else {
            return false;
        };
        for (byte, &possible) in first.iter().enumerate() {
            if possible {
                if claimed[byte] {
                    return false;
                }
                claimed[byte] = true;
            }
        }
    }
    true
}

/// Whether an expression contains a zero-width assertion or a backreference.
fn contains_assertion(expr: &HirExpr) -> bool {
    match expr {
        HirExpr::Anchor(_) | HirExpr::Lookaround(_) | HirExpr::Backref(_) => true,
        HirExpr::Concat(exprs) | HirExpr::Alt(exprs) => exprs.iter().any(contains_assertion),
        HirExpr::Repeat(r) => contains_assertion(&r.expr),
        HirExpr::Capture(c) => contains_assertion(&c.expr),
        HirExpr::Empty | HirExpr::Literal(_) | HirExpr::Class(_) | HirExpr::UnicodeCpClass(_) => {
            false
        }
    }
}

/// The bytes an expression can start with, or `None` when that is not a settled
/// question — it may match empty, or it is a construct whose first byte this
/// does not model. `None` always means "assume it competes".
fn first_bytes(expr: &HirExpr) -> Option<[bool; 256]> {
    let mut set = [false; 256];
    match expr {
        HirExpr::Literal(bytes) => {
            set[*bytes.first()? as usize] = true;
        }
        HirExpr::Class(class) => {
            for byte in 0..=255u8 {
                let in_ranges = class
                    .ranges
                    .iter()
                    .any(|&(lo, hi)| lo <= byte && byte <= hi);
                set[byte as usize] = in_ranges != class.negated;
            }
        }
        HirExpr::Concat(exprs) => return first_bytes(exprs.first()?),
        HirExpr::Alt(branches) => {
            for branch in branches {
                let branch_set = first_bytes(branch)?;
                for (byte, possible) in branch_set.iter().enumerate() {
                    set[byte] |= possible;
                }
            }
        }
        HirExpr::Capture(c) => return first_bytes(&c.expr),
        HirExpr::Repeat(r) if r.min >= 1 => return first_bytes(&r.expr),
        // Repetition that may run zero times, anchors, lookaround,
        // backreferences and `UnicodeCpClass` are all either zero-width or not
        // modelled here.
        _ => return None,
    }
    Some(set)
}

/// Whether a pattern combines a word boundary with the ability to match empty
/// (`\b`, `\Ba*`, `\b(?:xy)?`, `a*\B`, …).
///
/// The DFA family resolves `\b`/`\B` while *building* a state's epsilon closure,
/// but the truth of a boundary depends on the byte on **each** side of the
/// position. For a non-empty match that is fine: the assertion is re-resolved
/// when the next byte is consumed, so both sides are known. An empty match
/// consumes nothing, so the DFA would have to decide the assertion at a point
/// where the following byte has not been read — and the start state is built
/// from a guess. That makes empty matches under a boundary unrepresentable in
/// the DFA (and in the DFA JIT), so such patterns are routed to the PikeVM,
/// which evaluates every assertion against the real position.
pub fn needs_boundary_aware_empty_match(hir: &Hir) -> bool {
    hir.props.has_word_boundary && crate::hir::matches_empty(&hir.expr)
}

/// Whether an alternation appears anywhere in the expression, disjoint or not.
///
/// [`hir_has_alternation`] answers a *semantic* question — can branch priority
/// change the answer — and deliberately says no for branches that start on
/// disjoint bytes. This answers a cost question instead, and those branches
/// count: what makes an alternation expensive is that several positions are live
/// at once, which is true however the branches begin.
pub fn hir_contains_alternation(expr: &HirExpr) -> bool {
    match expr {
        HirExpr::Alt(branches) => {
            branches.len() >= 2 || branches.iter().any(hir_contains_alternation)
        }
        HirExpr::Concat(exprs) => exprs.iter().any(hir_contains_alternation),
        HirExpr::Repeat(repeat) => hir_contains_alternation(&repeat.expr),
        HirExpr::Capture(capture) => hir_contains_alternation(&capture.expr),
        _ => false,
    }
}

/// Recursively checks whether an HIR expression contains a user alternation
/// (`a|b`). A DFA/Shift-Or matcher returns at its first/longest accepting state
/// and therefore cannot honour ALTERNATION BRANCH PRIORITY: `ab|a` on "ab" must
/// be `ab` (the first branch) under leftmost-first/PCRE semantics, but a DFA
/// stops at the shorter `a`; `\d+|\w+` on "12ab" must be `12`, but Shift-Or takes
/// the longest `12ab`. Only an ordered NFA simulation (PikeVM) gets these right,
/// so any pattern containing an alternation is routed there.
///
/// Branch priority only decides anything when two branches can match at the
/// same position. When their possible first bytes are disjoint, at most one
/// branch is ever viable and every engine is forced to the same answer — so
/// those do not count. That is what a negated class expands to (`[^a]` becomes
/// "a surviving ASCII byte, or a UTF-8 sequence"), and it keeps such patterns on
/// the DFA instead of dropping them onto the PikeVM.
pub fn hir_has_alternation(expr: &HirExpr) -> bool {
    match expr {
        HirExpr::Alt(branches) => {
            (branches.len() >= 2 && !branches_start_disjointly(branches))
                || branches.iter().any(hir_has_alternation)
        }
        HirExpr::Concat(exprs) => exprs.iter().any(hir_has_alternation),
        HirExpr::Repeat(r) => hir_has_alternation(&r.expr),
        HirExpr::Capture(c) => hir_has_alternation(&c.expr),
        // A lookaround body matches independently (it is a zero-width sub-pattern),
        // so an alternation *inside* it does not affect the outer match's branch
        // priority and need not force PikeVM on its own account.
        HirExpr::Lookaround(_)
        | HirExpr::Empty
        | HirExpr::Literal(_)
        | HirExpr::Class(_)
        | HirExpr::UnicodeCpClass(_)
        | HirExpr::Anchor(_)
        | HirExpr::Backref(_) => false,
    }
}

/// Whether the DFA family and Shift-Or report the PikeVM's match for every
/// haystack.
///
/// Two properties are proven on the Thompson NFA:
/// - Priority cannot move the end ([`crate::nfa::priority_is_irrelevant`]).
///   Branch order is not the only priority: a greedy repeat prefers one more
///   iteration, so `a?(?:ab)?` on "ab" is `a`, while those engines report the
///   longest match `ab`.
/// - Every assertion sits at an edge of the match
///   ([`crate::nfa::assertions_at_edges`]). Those engines check assertions
///   from the start context or at the match end, so `a$a` and `a\ba?` are
///   beyond them.
///
/// A pattern the NFA builder rejects answers `false`, which keeps it on the
/// PikeVM.
pub fn automata_match_like_pikevm(hir: &Hir) -> bool {
    match crate::nfa::compile(hir) {
        Ok(nfa) => nfa_automata_match_like_pikevm(&nfa),
        Err(_) => false,
    }
}

/// [`automata_match_like_pikevm`] for an NFA already built.
fn nfa_automata_match_like_pikevm(nfa: &Nfa) -> bool {
    crate::nfa::priority_is_irrelevant(nfa) && crate::nfa::assertions_at_edges(nfa)
}

/// The selected engine type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineType {
    /// PikeVM - for patterns with lookarounds or non-greedy quantifiers.
    PikeVm,
    /// Tagged NFA interpreter - for patterns using CodepointClass instructions,
    /// which the byte-level engines cannot execute. Falls back internally to the
    /// PikeVM when step extraction declines (see `select_engine_from_hir`).
    TaggedNfa,
    /// BacktrackingVm - for patterns with backreferences (parity with BacktrackingJit).
    BacktrackingVm,
    /// Shift-Or - for small patterns (≤64 character positions).
    ShiftOr,
    /// Wide Shift-Or - for medium patterns (65-256 character positions).
    ShiftOrWide,
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

    // The lazy DFA reports the longest match and checks assertions only at
    // the edges of a match.
    if !nfa_automata_match_like_pikevm(nfa) {
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

    // Large Unicode classes that use the CodepointClass instruction.
    // CodepointClass does codepoint-level matching (UTF-8 decode + range check)
    // instead of byte-level DFA transitions. LazyDFA and Shift-Or cannot run
    // these patterns at all — only the PikeVM and the tagged NFA can.
    //
    // Of those two the tagged NFA is far the faster: `\p{L}+` extracts to a
    // single greedy codepoint step, against the PikeVM's per-byte thread
    // bookkeeping. It is also safe for any shape reaching here. The blow-up the
    // tagged path can suffer is at *extraction* time, not match time — an `Alt`
    // copies everything after it into each branch, so `\X{4}` (a grapheme
    // cluster, which lowers to codepoint classes) would emit an exponential step
    // program. `MAX_EXTRACTED_STEPS` (`nfa::tagged::steps`) caps that: extraction
    // returns `None`, and `TaggedNfaEngine` then runs the PikeVM it constructs
    // regardless. The linear bound is therefore kept exactly where it is needed,
    // instead of being paid for on every codepoint class.
    //
    // Do not restore a blanket PikeVM route here. `compile_with_jit` has taken
    // the tagged path for these patterns all along, and
    // `tests/bounded_execution.rs`'s
    // `repeated_grapheme_cluster_terminates_on_the_tagged_path` holds the
    // grapheme-cluster shape that motivated the old route to a deadline on both
    // build configurations.
    if hir_uses_codepoint_class(&hir.expr) {
        return EngineType::TaggedNfa;
    }

    // A word boundary guarding an empty match cannot be expressed by the DFA
    // family (see `needs_boundary_aware_empty_match`). Checked before Shift-Or,
    // which has its own partial word-boundary handling.
    if needs_boundary_aware_empty_match(hir) {
        return EngineType::PikeVm;
    }

    // Alternations need leftmost-first branch priority, which DFA/Shift-Or cannot
    // express (see `hir_has_alternation`). Route them to the ordered PikeVM.
    if hir_has_alternation(&hir.expr) {
        return EngineType::PikeVm;
    }

    // Every engine below reports the longest match from the leftmost start and
    // checks assertions only at the edges of a match (see
    // `automata_match_like_pikevm`).
    if !automata_match_like_pikevm(hir) {
        return EngineType::PikeVm;
    }

    // An alternation that survived the check above is one no engine can get
    // wrong, because at most one branch is ever viable. It is still the shape
    // Shift-Or handles worst: its step is not a shift but a walk over the live
    // positions, unioning a follow set per position, and an alternation is
    // precisely what keeps several live at once. A DFA spends one table lookup
    // per byte however many positions the pattern has.
    //
    // Negated classes lower to an alternation (an ASCII class beside a UTF-8
    // trie), so this covers `[^>]+` and `[^\s<>]+` as well as a written-out one.
    //
    // Measured against the `regex` crate over nine benchmark patterns plus four
    // class shapes, comparing ratios rather than times because the two runs sit
    // on different machine loads: eight improved and none got worse.
    if hir_contains_alternation(&hir.expr) {
        return EngineType::LazyDfa;
    }

    // Small patterns (≤64 character positions) use Shift-Or
    // Shift-Or uses Glushkov NFA (ε-free) for bit-parallel execution
    // ShiftOr now supports non-multiline anchors (^, $)
    if is_shift_or_compatible(hir) {
        return EngineType::ShiftOr;
    }

    // Multiline anchors ((?m)^, (?m)$) require LazyDFA for proper handling
    if hir.props.has_multiline_anchors {
        return EngineType::LazyDfa;
    }

    // Medium patterns (65-256 character positions) use Wide Shift-Or
    // Uses [u64; 4] for 256-bit state vectors instead of falling back to PikeVM
    if is_shift_or_wide_compatible(hir) {
        return EngineType::ShiftOrWide;
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
        // JIT is available on x86-64 and aarch64 (Linux/macOS/Windows)
        cfg!(any(target_arch = "x86_64", target_arch = "aarch64"))
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
    fn test_medium_pattern_uses_shift_or_wide() {
        // Medium patterns (65-256 positions) should use ShiftOrWide
        let medium_pattern = "a".repeat(100);
        let engine = get_engine_from_hir(&medium_pattern);
        assert_eq!(engine, EngineType::ShiftOrWide);
    }

    #[test]
    fn test_very_long_pattern_uses_lazy_dfa() {
        // Very long patterns (>256 positions) should use Lazy DFA
        let long_pattern = "a".repeat(300);
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
        // A boundary between two consuming parts needs positional evaluation.
        assert_eq!(get_engine_from_hir(r"a\Bb"), EngineType::PikeVm);
        assert_eq!(get_engine_from_hir(r"a{1,2}\Ba"), EngineType::PikeVm);

        // Long patterns with word boundaries should use LazyDFA
        let long_pattern = format!(r"\b{}\b", "a".repeat(100));
        assert_eq!(get_engine_from_hir(&long_pattern), EngineType::LazyDfa);
    }

    #[test]
    fn test_anchors_engine_selection() {
        // Non-multiline anchors use ShiftOr (fast bit-parallel matching)
        assert_eq!(get_engine_from_hir(r"^hello"), EngineType::ShiftOr);
        assert_eq!(get_engine_from_hir(r"world$"), EngineType::ShiftOr);
        assert_eq!(get_engine_from_hir(r"^hello$"), EngineType::ShiftOr);

        // Multiline anchors require LazyDFA (position-aware matching)
        assert_eq!(get_engine_from_hir(r"(?m)^line"), EngineType::LazyDfa);
        assert_eq!(get_engine_from_hir(r"(?m)line$"), EngineType::LazyDfa);
    }
}
