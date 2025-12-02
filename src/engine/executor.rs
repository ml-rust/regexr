//! Unified executor for compiled regex patterns.
//!
//! The executor uses a prefilter (when available) to quickly skip to candidate
//! positions before engaging the full regex engine.

use std::sync::RwLock;

use crate::dfa::{EagerDfa, LazyDfa};
use crate::error::Result;
use crate::hir::Hir;
use crate::literal::{extract_literals, Prefilter};
use crate::nfa::tagged::TaggedNfaEngine;
use crate::nfa::{self, Nfa};
use crate::vm::{
    BacktrackingVm, CodepointClassMatcher, PikeVm, PikeVmContext, ShiftOr, ShiftOrWide,
};

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
use crate::jit;

use super::{select_engine, select_engine_from_hir, EngineType};

/// Returns true if the byte is a word character (alphanumeric or underscore).
#[inline]
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// A compiled regex ready for execution.
pub struct CompiledRegex {
    inner: CompiledInner,
    prefilter: Prefilter,
    /// Fallback NFA for captures when using Shift-Or or LazyDfa.
    /// Lazily compiled on first captures() call.
    capture_nfa: RwLock<Option<Nfa>>,
    /// Cached PikeVM for capture extraction.
    /// Lazily initialized on first captures() call to avoid cloning NFA repeatedly.
    capture_vm: RwLock<Option<PikeVm>>,
    /// Cached execution context for PikeVM.
    /// Provides pre-allocated storage to avoid allocations on each captures() call.
    capture_ctx: RwLock<Option<PikeVmContext>>,
    /// BacktrackingVm for fast single-pass capture extraction.
    /// Used instead of PikeVM for patterns with captures (no lookaround).
    backtracking_vm: Option<BacktrackingVm>,
    /// BacktrackingJit for fast single-pass capture extraction in JIT mode.
    /// Used by JitShiftOr when pattern has captures.
    /// This is the JIT equivalent of backtracking_vm.
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    backtracking_jit: Option<jit::BacktrackingJit>,
}

impl std::fmt::Debug for CompiledRegex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledRegex")
            .field("engine", &self.engine_name())
            .field("prefilter", &self.prefilter)
            .finish_non_exhaustive()
    }
}

enum CompiledInner {
    PikeVm(PikeVm),
    ShiftOr(ShiftOr),
    /// Wide Shift-Or for patterns with 65-256 positions.
    /// Uses [u64; 4] for 256-bit state vectors.
    ShiftOrWide(ShiftOrWide),
    LazyDfa(RwLock<LazyDfa>),
    /// Pre-materialized DFA for fast matching without JIT.
    /// Used for patterns that benefit from eager state computation.
    EagerDfa(EagerDfa),
    /// Fast codepoint-level matching for single character class patterns.
    CodepointClass(CodepointClassMatcher),
    /// Backtracking VM engine for patterns with backreferences.
    /// Uses PCRE-style backtracking (non-JIT version of BacktrackingJit).
    BacktrackingVm(BacktrackingVm),
    /// Tagged NFA interpreter for patterns with lookaround or non-greedy.
    /// Uses liveness analysis for efficient single-pass capture extraction.
    /// Always available (no JIT required).
    TaggedNfaInterp(TaggedNfaEngine),
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    Jit(jit::CompiledRegex),
    /// Tagged NFA JIT engine for patterns with lookaround or non-greedy.
    /// Uses liveness analysis for efficient single-pass capture extraction.
    /// JIT compiles the NFA to native code for better performance.
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    TaggedNfaJit(jit::TaggedNfaJit),
    /// Backtracking JIT engine for patterns with backreferences.
    /// Uses PCRE-style backtracking for fast backreference matching.
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    Backtracking(jit::BacktrackingJit),
    /// JIT-compiled Shift-Or engine for word boundary patterns.
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    JitShiftOr(jit::JitShiftOr),
}

impl CompiledRegex {
    /// Returns the name of the engine being used (for debugging).
    pub fn engine_name(&self) -> &'static str {
        match &self.inner {
            CompiledInner::PikeVm(_) => "PikeVm",
            CompiledInner::ShiftOr(_) => "ShiftOr",
            CompiledInner::ShiftOrWide(_) => "ShiftOrWide",
            CompiledInner::LazyDfa(_) => "LazyDfa",
            CompiledInner::EagerDfa(_) => "EagerDfa",
            CompiledInner::CodepointClass(_) => "CodepointClass",
            CompiledInner::BacktrackingVm(_) => "BacktrackingVm",
            CompiledInner::TaggedNfaInterp(_) => "TaggedNfa",
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::Jit(_) => "Jit",
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::TaggedNfaJit(_) => "TaggedNfaJit",
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::Backtracking(_) => "BacktrackingJit",
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::JitShiftOr(_) => "JitShiftOr",
        }
    }

    /// Gets or creates a cached PikeVM and context for capture extraction.
    /// This avoids cloning the NFA and allocating storage on every captures() call.
    fn get_or_init_capture_vm(&self) {
        if self.capture_vm.read().unwrap().is_some() {
            return;
        }
        if let Some(nfa) = self.capture_nfa.read().unwrap().as_ref() {
            let vm = PikeVm::new(nfa.clone());
            let ctx = vm.create_context();
            *self.capture_vm.write().unwrap() = Some(vm);
            *self.capture_ctx.write().unwrap() = Some(ctx);
        }
    }

    /// Returns true if the pattern matches anywhere in the input.
    pub fn is_match(&self, input: &[u8]) -> bool {
        // Fast path: if prefilter can provide full match bounds (TeddyFull),
        // finding any match means there's a match
        if self.prefilter.is_full_match() {
            return self.prefilter.find_full_match(input, 0).is_some();
        }

        // Use prefilter to skip to candidate positions
        if !self.prefilter.is_none() {
            for candidate in self.prefilter.find_candidates(input) {
                if self.is_match_at(input, candidate) {
                    return true;
                }
            }
            return false;
        }

        // No prefilter - check from start
        match &self.inner {
            CompiledInner::PikeVm(vm) => vm.is_match(input),
            CompiledInner::ShiftOr(so) => so.is_match(input),
            CompiledInner::ShiftOrWide(so) => so.is_match(input),
            CompiledInner::LazyDfa(dfa) => dfa.write().unwrap().find(input).is_some(),
            CompiledInner::EagerDfa(dfa) => dfa.find(input).is_some(),
            CompiledInner::CodepointClass(matcher) => matcher.is_match(input),
            CompiledInner::BacktrackingVm(vm) => vm.find(input).is_some(),
            CompiledInner::TaggedNfaInterp(engine) => engine.is_match(input),
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::Jit(jit) => jit.is_match(input),
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::TaggedNfaJit(engine) => engine.is_match(input),
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::Backtracking(jit) => jit.is_match(input),
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::JitShiftOr(jit) => jit.find(input).is_some(),
        }
    }

    /// Returns true if this regex uses a TeddyFull prefilter.
    /// When true, `find_iter_fast()` can be used for better performance.
    #[inline]
    pub fn is_full_match_prefilter(&self) -> bool {
        self.prefilter.is_full_match()
    }

    /// Returns an optimized iterator for patterns with TeddyFull prefilter.
    /// This returns matches directly from the Teddy SIMD matcher without
    /// going through the NFA/DFA engine.
    ///
    /// Only valid when `is_full_match_prefilter()` returns true.
    #[inline]
    pub fn find_full_matches<'a>(
        &'a self,
        input: &'a [u8],
    ) -> crate::literal::FullMatchIter<'a, 'a> {
        self.prefilter.find_full_matches(input)
    }

    /// Finds the first match, returning (start, end).
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        // Fast path: if prefilter can provide full match bounds (TeddyFull),
        // return directly without running the NFA
        if self.prefilter.is_full_match() {
            return self.prefilter.find_full_match(input, 0);
        }

        // Special handling for InnerByte prefilter
        // InnerByte finds a required byte that appears somewhere in the match,
        // so we need to look back from the found position to find the actual start.
        if self.prefilter.is_inner_byte() {
            let lookback = self.prefilter.inner_byte_lookback();
            let mut search_pos = 0;

            while let Some(inner_pos) = self.prefilter.find_candidate(input, search_pos) {
                // Find the likely start position by looking back for a word boundary
                // (non-word char followed by word char, or start of input)
                let start_pos = inner_pos.saturating_sub(lookback);
                let mut candidate = start_pos;

                // Find the first word boundary in the lookback window
                for i in (start_pos..inner_pos).rev() {
                    if i == 0 || !is_word_byte(input[i - 1]) {
                        if i < input.len() && is_word_byte(input[i]) {
                            candidate = i;
                            break;
                        }
                    }
                }

                // Try starting from the candidate position
                if let Some((start, end)) = self.find_at(input, candidate) {
                    return Some((start, end));
                }

                // No match found around this inner byte, skip past it
                search_pos = inner_pos + 1;
            }
            return None;
        }

        // Use prefilter to skip to candidate positions
        // IMPORTANT: Use find_at_pos (exact position) not find_at (linear search from pos)
        // The prefilter already tells us where candidates are - we only need to verify each one.
        if !self.prefilter.is_none() {
            for candidate in self.prefilter.find_candidates(input) {
                if let Some(result) = self.find_at_pos(input, candidate) {
                    return Some(result);
                }
            }
            return None;
        }

        // No prefilter - search from start
        match &self.inner {
            CompiledInner::PikeVm(vm) => vm.find(input),
            CompiledInner::ShiftOr(so) => so.find(input),
            CompiledInner::ShiftOrWide(so) => so.find(input),
            CompiledInner::LazyDfa(dfa) => dfa.write().unwrap().find(input),
            CompiledInner::EagerDfa(dfa) => dfa.find(input),
            CompiledInner::CodepointClass(matcher) => matcher.find(input),
            CompiledInner::BacktrackingVm(vm) => vm.find(input),
            CompiledInner::TaggedNfaInterp(engine) => engine.find(input),
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::Jit(jit) => jit.find(input),
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::TaggedNfaJit(engine) => engine.find(input),
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::Backtracking(jit) => jit.find(input),
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::JitShiftOr(jit) => jit.find(input),
        }
    }

    /// Returns capture groups for the first match.
    ///
    /// For Shift-Or, LazyDfa, and JIT engines, this uses a two-pass strategy:
    /// 1. Use the fast engine to find match bounds
    /// 2. Use PikeVm on the matched substring to extract captures
    ///
    /// TaggedNfa performs single-pass capture extraction natively.
    pub fn captures(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        match &self.inner {
            CompiledInner::PikeVm(vm) => vm.captures(input),
            CompiledInner::CodepointClass(matcher) => matcher.captures(input),
            CompiledInner::BacktrackingVm(vm) => {
                // BacktrackingVm does single-pass capture extraction
                vm.captures(input)
            }
            CompiledInner::TaggedNfaInterp(engine) => {
                // TaggedNfa interpreter does single-pass capture extraction
                engine.captures(input)
            }
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::TaggedNfaJit(engine) => {
                // TaggedNfa JIT does single-pass capture extraction
                engine.captures(input)
            }
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::Backtracking(jit) => {
                // Backtracking JIT does single-pass capture extraction
                jit.captures(input)
            }
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::Jit(_) => {
                // Fast path: if we have BacktrackingVm, use it for single-pass capture extraction
                if let Some(ref backtracking_vm) = self.backtracking_vm {
                    return backtracking_vm.captures(input);
                }

                // Two-pass capture strategy for DFA JIT (fallback)
                let (start, _end) = self.find(input)?;

                // Use cached PikeVM and context to avoid allocations
                self.get_or_init_capture_vm();
                let vm_ref = self.capture_vm.read().unwrap();
                let vm = vm_ref.as_ref()?;
                let mut ctx_ref = self.capture_ctx.write().unwrap();
                let ctx = ctx_ref.as_mut()?;

                // Use the optimized context-based method
                vm.captures_from_start_with_context(&input[start..], ctx)
                    .map(|mut caps| {
                        for slot in &mut caps {
                            if let Some((s, e)) = slot {
                                *s += start;
                                *e += start;
                            }
                        }
                        caps
                    })
            }
            CompiledInner::ShiftOr(_)
            | CompiledInner::ShiftOrWide(_)
            | CompiledInner::LazyDfa(_)
            | CompiledInner::EagerDfa(_) => {
                // Fast path: if we have BacktrackingVm, use it for single-pass capture extraction
                if let Some(ref backtracking_vm) = self.backtracking_vm {
                    return backtracking_vm.captures(input);
                }

                // Two-pass capture strategy:
                // 1. Find match bounds using fast engine
                let (start, _end) = self.find(input)?;

                // 2. Use cached PikeVM and context for fast capture extraction
                self.get_or_init_capture_vm();
                let vm_ref = self.capture_vm.read().unwrap();
                let vm = vm_ref.as_ref()?;
                let mut ctx_ref = self.capture_ctx.write().unwrap();
                let ctx = ctx_ref.as_mut()?;

                // Use the optimized context-based method
                vm.captures_from_start_with_context(&input[start..], ctx)
                    .map(|mut caps| {
                        // Adjust capture positions to absolute offsets
                        for slot in &mut caps {
                            if let Some((s, e)) = slot {
                                *s += start;
                                *e += start;
                            }
                        }
                        caps
                    })
            }
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::JitShiftOr(_) => {
                // Use BacktrackingJit for capture extraction if available
                // This is the JIT equivalent of BacktrackingVm used by non-JIT ShiftOr
                if let Some(ref backtracking_jit) = self.backtracking_jit {
                    return backtracking_jit.captures(input);
                }

                // Fall back to two-pass strategy if no BacktrackingJit
                let (start, _end) = self.find(input)?;
                self.get_or_init_capture_vm();
                let vm_ref = self.capture_vm.read().unwrap();
                let vm = vm_ref.as_ref()?;
                let mut ctx_ref = self.capture_ctx.write().unwrap();
                let ctx = ctx_ref.as_mut()?;
                vm.captures_from_start_with_context(&input[start..], ctx)
                    .map(|mut caps| {
                        for slot in &mut caps {
                            if let Some((s, e)) = slot {
                                *s += start;
                                *e += start;
                            }
                        }
                        caps
                    })
            }
        }
    }

    /// Check if there's a match starting at `pos`.
    ///
    /// This method passes the full input to allow engines to check context
    /// (e.g., for word boundary assertions).
    fn is_match_at(&self, input: &[u8], pos: usize) -> bool {
        self.find_at_pos(input, pos).is_some()
    }

    /// Find a match starting exactly at `pos`.
    ///
    /// This method passes the full input to allow engines to check context
    /// (e.g., for word boundary assertions).
    fn find_at_pos(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos > input.len() {
            return None;
        }
        match &self.inner {
            CompiledInner::PikeVm(vm) => vm.find_at(input, pos),
            CompiledInner::ShiftOr(so) => so.try_match_at(input, pos),
            CompiledInner::ShiftOrWide(so) => so.try_match_at(input, pos),
            CompiledInner::LazyDfa(dfa) => dfa
                .write()
                .unwrap()
                .find_at(input, pos)
                .map(|end| (pos, end)),
            CompiledInner::EagerDfa(dfa) => dfa.find_at(input, pos).map(|end| (pos, end)),
            CompiledInner::CodepointClass(matcher) => {
                // CodepointClass doesn't support word boundaries, use sliced input
                let slice = &input[pos..];
                matcher.find(slice).map(|(s, e)| (pos + s, pos + e))
            }
            CompiledInner::BacktrackingVm(vm) => vm.find_at(input, pos),
            CompiledInner::TaggedNfaInterp(engine) => engine.find_at(input, pos),
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::Jit(jit) => jit.find_at(input, pos),
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::TaggedNfaJit(engine) => engine.find_at(input, pos),
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::Backtracking(jit) => jit.find_at(input, pos),
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            CompiledInner::JitShiftOr(jit) => jit.try_match_at(input, pos),
        }
    }

    /// Find a match starting at or after `pos`.
    fn find_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        // Try each position starting from pos
        for start in pos..=input.len() {
            if let Some(result) = self.find_at_pos(input, start) {
                return Some(result);
            }
        }
        None
    }
}

/// Compiles an NFA into an executable regex (legacy API).
/// Note: This cannot use Shift-Or as it requires HIR for Glushkov construction.
/// Also cannot use prefilter (requires HIR for literal extraction).
pub fn compile(nfa: Nfa) -> Result<CompiledRegex> {
    let engine = select_engine(&nfa);

    let (inner, capture_nfa) = match engine {
        EngineType::PikeVm => (CompiledInner::PikeVm(PikeVm::new(nfa)), None),
        EngineType::BacktrackingVm => {
            // NFA-based compilation can't use BacktrackingVm (needs HIR)
            // Fall back to PikeVm which also handles backrefs
            (CompiledInner::PikeVm(PikeVm::new(nfa)), None)
        }
        EngineType::ShiftOr | EngineType::ShiftOrWide => {
            // NFA-based compilation can't use Shift-Or (needs Glushkov from HIR)
            // Fall back to LazyDfa, keep NFA for captures
            let capture_nfa = Some(nfa.clone());
            (
                CompiledInner::LazyDfa(RwLock::new(LazyDfa::new(nfa))),
                capture_nfa,
            )
        }
        EngineType::LazyDfa => {
            let capture_nfa = Some(nfa.clone());
            (
                CompiledInner::LazyDfa(RwLock::new(LazyDfa::new(nfa))),
                capture_nfa,
            )
        }
        #[cfg(feature = "jit")]
        EngineType::Jit => {
            // JIT not implemented yet, fall back to LazyDfa
            let capture_nfa = Some(nfa.clone());
            (
                CompiledInner::LazyDfa(RwLock::new(LazyDfa::new(nfa))),
                capture_nfa,
            )
        }
    };

    Ok(CompiledRegex {
        inner,
        prefilter: Prefilter::None, // Can't extract literals from NFA
        capture_nfa: RwLock::new(capture_nfa),
        capture_vm: RwLock::new(None),
        capture_ctx: RwLock::new(None),
        backtracking_vm: None,
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        backtracking_jit: None,
    })
}

/// Compiles an HIR into an executable regex.
/// This is the preferred API as it can use Shift-Or for small patterns
/// and prefilters for SIMD-accelerated candidate detection.
pub fn compile_from_hir(hir: &Hir) -> Result<CompiledRegex> {
    // Fast path: if the pattern is a single character class, use CodepointClassMatcher.
    // This is MUCH faster than byte-level DFA for Unicode patterns like [^α-ω].
    if let Some(ref codepoint_class) = hir.props.codepoint_class {
        return Ok(CompiledRegex {
            inner: CompiledInner::CodepointClass(CodepointClassMatcher::new(
                codepoint_class.clone(),
            )),
            prefilter: Prefilter::None,
            capture_nfa: RwLock::new(None),
            capture_vm: RwLock::new(None),
            capture_ctx: RwLock::new(None),
            backtracking_vm: None,
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            backtracking_jit: None,
        });
    }

    // Patterns with lookaround or non-greedy → TaggedNfa interpreter
    // Both require NFA semantics with proper thread ordering.
    // TaggedNfa uses liveness analysis for efficient capture copying.
    // NOTE: This is always available (no JIT required).
    if hir.props.has_lookaround || hir.props.has_non_greedy {
        let literals = extract_literals(hir);
        let prefilter = Prefilter::from_literals(&literals);
        let nfa = nfa::compile(hir)?;
        let engine = TaggedNfaEngine::new(nfa);
        return Ok(CompiledRegex {
            inner: CompiledInner::TaggedNfaInterp(engine),
            prefilter,
            capture_nfa: RwLock::new(None),
            capture_vm: RwLock::new(None),
            capture_ctx: RwLock::new(None),
            backtracking_vm: None,
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            backtracking_jit: None,
        });
    }

    // Extract literals for prefilter
    // Word boundaries: NOW SUPPORTED! The executor passes full input context
    // to engines via find_at_pos(), allowing proper word boundary checking.
    // Anchors: NOW SUPPORTED! LazyDFA/JIT handle start anchor optimization
    // internally, so prefilter can still be used for patterns with anchors.
    let literals = extract_literals(hir);
    let prefilter = Prefilter::from_literals(&literals);

    // For patterns with captures (but no lookaround), we use a hybrid approach:
    // - find() uses LazyDfa (fast)
    // - captures() uses BacktrackingVm (fast single-pass)
    // The BacktrackingVm is stored in capture_vm for use by captures().
    let has_simple_captures = hir.props.capture_count > 0 && !hir.props.has_lookaround;

    let engine = select_engine_from_hir(hir);

    let (inner, capture_nfa) = match engine {
        EngineType::PikeVm => {
            let nfa = nfa::compile(hir)?;
            (CompiledInner::PikeVm(PikeVm::new(nfa)), None)
        }
        EngineType::BacktrackingVm => {
            // BacktrackingVm for patterns with backreferences
            // This maintains parity with BacktrackingJit for JIT builds
            (
                CompiledInner::BacktrackingVm(BacktrackingVm::new(hir)),
                None,
            )
        }
        EngineType::ShiftOr => {
            // Use Glushkov NFA for Shift-Or
            // Keep Thompson NFA for captures (two-pass strategy)
            // Use from_hir_with_anchors for patterns with non-multiline anchors
            let shift_or = if hir.props.has_anchors {
                ShiftOr::from_hir_with_anchors(hir)
            } else {
                ShiftOr::from_hir(hir)
            };
            match shift_or {
                Some(so) => {
                    let capture_nfa = nfa::compile(hir)?;
                    (CompiledInner::ShiftOr(so), Some(capture_nfa))
                }
                None => {
                    // Fall back to LazyDfa
                    let nfa = nfa::compile(hir)?;
                    let capture_nfa = Some(nfa.clone());
                    (
                        CompiledInner::LazyDfa(RwLock::new(LazyDfa::new(nfa))),
                        capture_nfa,
                    )
                }
            }
        }
        EngineType::ShiftOrWide => {
            // Use Wide Glushkov NFA for ShiftOrWide (65-256 positions)
            // Keep Thompson NFA for captures (two-pass strategy)
            match ShiftOrWide::from_hir(hir) {
                Some(so) => {
                    let capture_nfa = nfa::compile(hir)?;
                    (CompiledInner::ShiftOrWide(so), Some(capture_nfa))
                }
                None => {
                    // Fall back to LazyDfa
                    let nfa = nfa::compile(hir)?;
                    let capture_nfa = Some(nfa.clone());
                    (
                        CompiledInner::LazyDfa(RwLock::new(LazyDfa::new(nfa))),
                        capture_nfa,
                    )
                }
            }
        }
        EngineType::LazyDfa => {
            let nfa = nfa::compile(hir)?;
            let capture_nfa = Some(nfa.clone());

            // Use LazyDfa (not EagerDfa) for:
            // - Large Unicode classes: avoid state explosion during materialization
            // - Patterns with anchors: EagerDfa doesn't handle anchors correctly
            // EagerDfa creates all reachable states upfront, which can be millions
            // for large Unicode classes.
            if hir.props.has_large_unicode_class || hir.props.has_anchors {
                (
                    CompiledInner::LazyDfa(RwLock::new(LazyDfa::new(nfa))),
                    capture_nfa,
                )
            } else {
                // Use EagerDfa for better non-JIT performance on simple patterns.
                // EagerDfa pre-computes all states upfront, eliminating hash lookups.
                let mut lazy = LazyDfa::new(nfa);
                let eager = EagerDfa::from_lazy(&mut lazy);
                (CompiledInner::EagerDfa(eager), capture_nfa)
            }
        }
        #[cfg(feature = "jit")]
        EngineType::Jit => {
            // JIT not implemented yet, fall back to EagerDfa or LazyDfa
            let nfa = nfa::compile(hir)?;
            let capture_nfa = Some(nfa.clone());

            // Use LazyDfa for patterns with large Unicode classes or anchors
            if hir.props.has_large_unicode_class || hir.props.has_anchors {
                (
                    CompiledInner::LazyDfa(RwLock::new(LazyDfa::new(nfa))),
                    capture_nfa,
                )
            } else {
                let mut lazy = LazyDfa::new(nfa);
                let eager = EagerDfa::from_lazy(&mut lazy);
                (CompiledInner::EagerDfa(eager), capture_nfa)
            }
        }
    };

    // Create BacktrackingVm for fast single-pass capture extraction (if applicable)
    let backtracking_vm = if has_simple_captures {
        Some(BacktrackingVm::new(hir))
    } else {
        None
    };

    Ok(CompiledRegex {
        inner,
        prefilter,
        capture_nfa: RwLock::new(capture_nfa),
        capture_vm: RwLock::new(None),
        capture_ctx: RwLock::new(None),
        backtracking_vm,
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        backtracking_jit: None,
    })
}

/// Compiles an HIR using PikeVM (default, no JIT).
///
/// PikeVM is a thread-based NFA simulator that supports all regex features
/// including backreferences, lookarounds, and non-greedy quantifiers.
/// It's slower than JIT but handles all patterns correctly.
pub fn compile_with_pikevm(hir: &Hir) -> Result<CompiledRegex> {
    let literals = extract_literals(hir);
    // Word boundaries and anchors: NOW SUPPORTED via full input context.
    let prefilter = Prefilter::from_literals(&literals);
    let nfa = nfa::compile(hir)?;

    Ok(CompiledRegex {
        inner: CompiledInner::PikeVm(PikeVm::new(nfa)),
        prefilter,
        capture_nfa: RwLock::new(None),
        capture_vm: RwLock::new(None),
        capture_ctx: RwLock::new(None),
        backtracking_vm: None,
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        backtracking_jit: None,
    })
}

/// Compiles an HIR with JIT compilation for maximum performance.
///
/// JIT compiles the pattern to native machine code for fast matching.
/// Ideal for patterns that will be matched many times (e.g., tokenization).
///
/// Engine selection strategy:
/// 0. Single character class → CodepointClassMatcher (fastest for Unicode)
/// 1. Complex Unicode patterns → LazyDfa (skip JIT to avoid state explosion)
/// 2. Patterns with backrefs/lookaround/non-greedy → TaggedNfa (liveness-optimized)
/// 3. Simple patterns → DFA JIT
pub fn compile_with_jit(hir: &Hir) -> Result<CompiledRegex> {
    // 0. Single character class → CodepointClassMatcher (fastest for Unicode)
    if let Some(ref codepoint_class) = hir.props.codepoint_class {
        return Ok(CompiledRegex {
            inner: CompiledInner::CodepointClass(CodepointClassMatcher::new(
                codepoint_class.clone(),
            )),
            prefilter: Prefilter::None,
            capture_nfa: RwLock::new(None),
            capture_vm: RwLock::new(None),
            capture_ctx: RwLock::new(None),
            backtracking_vm: None,
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            backtracking_jit: None,
        });
    }

    // 1. Complex Unicode patterns with large unicode classes → TaggedNfa JIT
    // These patterns use CodepointClass instructions which DFA cannot handle.
    // Route them to TaggedNfa JIT which supports CodepointClass.
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    if hir.props.has_large_unicode_class {
        let literals = extract_literals(hir);
        let prefilter = Prefilter::from_literals(&literals);
        let nfa = nfa::compile(hir)?;
        match jit::compile_tagged_nfa(&nfa) {
            Ok(engine) => {
                return Ok(CompiledRegex {
                    inner: CompiledInner::TaggedNfaJit(engine),
                    prefilter,
                    capture_nfa: RwLock::new(None),
                    capture_vm: RwLock::new(None),
                    capture_ctx: RwLock::new(None),
                    backtracking_vm: None,
                    backtracking_jit: None,
                });
            }
            Err(_e) => {
                // TaggedNfa JIT failed - fall back to TaggedNfa interpreter
                #[cfg(debug_assertions)]
                eprintln!("[regexr] TaggedNfaJit failed for large unicode class, falling back to interpreter: {}", _e);
                let engine = TaggedNfaEngine::new(nfa);
                return Ok(CompiledRegex {
                    inner: CompiledInner::TaggedNfaInterp(engine),
                    prefilter,
                    capture_nfa: RwLock::new(None),
                    capture_vm: RwLock::new(None),
                    capture_ctx: RwLock::new(None),
                    backtracking_vm: None,
                    backtracking_jit: None,
                });
            }
        }
    }

    // Non-JIT: Large unicode classes go to TaggedNfa interpreter
    #[cfg(not(all(feature = "jit", target_arch = "x86_64")))]
    if hir.props.has_large_unicode_class {
        let literals = extract_literals(hir);
        let prefilter = Prefilter::from_literals(&literals);
        let nfa = nfa::compile(hir)?;
        let engine = TaggedNfaEngine::new(nfa);
        return Ok(CompiledRegex {
            inner: CompiledInner::TaggedNfaInterp(engine),
            prefilter,
            capture_nfa: RwLock::new(None),
            capture_vm: RwLock::new(None),
            capture_ctx: RwLock::new(None),
            backtracking_vm: None,
        });
    }

    // 2. Patterns with backreferences → Backtracking JIT (only way to handle backrefs)
    // Backtracking JIT is required for backreferences since DFA cannot handle them.
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    if hir.props.has_backrefs && !hir.props.has_lookaround {
        let literals = extract_literals(hir);
        let prefilter = Prefilter::from_literals(&literals);
        match jit::compile_backtracking(hir) {
            Ok(jit_regex) => {
                return Ok(CompiledRegex {
                    inner: CompiledInner::Backtracking(jit_regex),
                    prefilter,
                    capture_nfa: RwLock::new(None),
                    capture_vm: RwLock::new(None),
                    capture_ctx: RwLock::new(None),
                    backtracking_vm: None,
                    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
                    backtracking_jit: None,
                });
            }
            Err(_) => {
                // Backtracking JIT failed, fall back to PikeVM
                return compile_with_pikevm(hir);
            }
        }
    }

    // 2a. Patterns with lookaround or non-greedy quantifiers → TaggedNfa JIT
    // Both lookaround and non-greedy require NFA semantics for correct matching.
    // TaggedNfa uses single-pass capture extraction with sparse copying and
    // memoized lookaround evaluation.
    //
    // NOTE: Patterns with captures but NO non-greedy/lookaround should use DFA JIT
    // because DFA JIT is much faster. DFA JIT handles captures via two-pass:
    // 1. Fast DFA JIT for find()
    // 2. PikeVM on matched substring for captures() only when needed
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    if hir.props.has_lookaround || hir.props.has_non_greedy {
        let literals = extract_literals(hir);
        let prefilter = Prefilter::from_literals(&literals);
        let nfa = nfa::compile(hir)?;
        match jit::compile_tagged_nfa(&nfa) {
            Ok(engine) => {
                return Ok(CompiledRegex {
                    inner: CompiledInner::TaggedNfaJit(engine),
                    prefilter,
                    capture_nfa: RwLock::new(None),
                    capture_vm: RwLock::new(None),
                    capture_ctx: RwLock::new(None),
                    backtracking_vm: None,
                    backtracking_jit: None,
                });
            }
            Err(_e) => {
                // TaggedNfa JIT failed (e.g., lookahead with captures not yet supported).
                // Fall back to TaggedNfa interpreter which handles all cases correctly.
                #[cfg(debug_assertions)]
                eprintln!(
                    "[regexr] TaggedNfaJit failed, falling back to interpreter: {}",
                    _e
                );
                let engine = TaggedNfaEngine::new(nfa);
                return Ok(CompiledRegex {
                    inner: CompiledInner::TaggedNfaInterp(engine),
                    prefilter,
                    capture_nfa: RwLock::new(None),
                    capture_vm: RwLock::new(None),
                    capture_ctx: RwLock::new(None),
                    backtracking_vm: None,
                    backtracking_jit: None,
                });
            }
        }
    }

    // Fall back to TaggedNfa interpreter when JIT feature is not available
    // Note: TaggedNfa interpreter is now always available (faster than PikeVm for lookaround)
    #[cfg(not(all(feature = "jit", target_arch = "x86_64")))]
    if hir.props.has_lookaround || hir.props.has_non_greedy {
        let literals = extract_literals(hir);
        let prefilter = Prefilter::from_literals(&literals);
        let nfa = nfa::compile(hir)?;
        let engine = TaggedNfaEngine::new(nfa);
        return Ok(CompiledRegex {
            inner: CompiledInner::TaggedNfaInterp(engine),
            prefilter,
            capture_nfa: RwLock::new(None),
            capture_vm: RwLock::new(None),
            capture_ctx: RwLock::new(None),
            backtracking_vm: None,
        });
    }

    // For backrefs without JIT, fall back to PikeVM
    #[cfg(not(all(feature = "jit", target_arch = "x86_64")))]
    if hir.props.has_backrefs {
        return compile_with_pikevm(hir);
    }

    // 3. Small patterns without effective prefilter → JitShiftOr
    // ShiftOr's bit-parallel algorithm is faster than DFA JIT for patterns with
    // many alternations and no common prefix (no effective prefilter).
    // DFA JIT excels when there's a good prefilter to skip non-matching positions.
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    {
        use crate::vm::is_shift_or_compatible;
        let literals = extract_literals(hir);
        let prefilter = Prefilter::from_literals(&literals);

        // Use JitShiftOr when:
        // 1. Pattern is ShiftOr-compatible (≤64 positions, no multiline anchors/word boundaries)
        // 2. No effective prefilter (DFA JIT doesn't benefit as much)
        if !prefilter.is_effective() && is_shift_or_compatible(hir) {
            let shift_or = if hir.props.has_anchors {
                crate::vm::ShiftOr::from_hir_with_anchors(hir)
            } else {
                crate::vm::ShiftOr::from_hir(hir)
            };
            if let Some(shift_or) = shift_or {
                if let Some(jit_shift_or) = jit::JitShiftOr::compile(&shift_or) {
                    let capture_nfa = if hir.props.capture_count > 0 {
                        nfa::compile(hir).ok()
                    } else {
                        None
                    };

                    let backtracking_vm =
                        if hir.props.capture_count > 0 && !hir.props.has_lookaround {
                            Some(BacktrackingVm::new(hir))
                        } else {
                            None
                        };

                    let backtracking_jit =
                        if hir.props.capture_count > 0 && !hir.props.has_lookaround {
                            jit::compile_backtracking(hir).ok()
                        } else {
                            None
                        };

                    return Ok(CompiledRegex {
                        inner: CompiledInner::JitShiftOr(jit_shift_or),
                        prefilter,
                        capture_nfa: RwLock::new(capture_nfa),
                        capture_vm: RwLock::new(None),
                        capture_ctx: RwLock::new(None),
                        backtracking_vm,
                        backtracking_jit,
                    });
                }
            }
        }
    }

    // 4. Simple patterns with effective prefilter → DFA JIT
    // DFA JIT benefits from prefilter to quickly skip non-matching positions.
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    {
        let literals = extract_literals(hir);
        let prefilter = Prefilter::from_literals(&literals);
        let nfa = nfa::compile(hir)?;
        let capture_nfa = Some(nfa.clone());
        let mut dfa = LazyDfa::new(nfa);

        // Create BacktrackingVm for fast capture extraction if pattern has captures
        let backtracking_vm = if hir.props.capture_count > 0 && !hir.props.has_lookaround {
            Some(BacktrackingVm::new(hir))
        } else {
            None
        };

        match jit::compile_dfa(&mut dfa) {
            Ok(jit_regex) => {
                return Ok(CompiledRegex {
                    inner: CompiledInner::Jit(jit_regex),
                    prefilter,
                    capture_nfa: RwLock::new(capture_nfa),
                    capture_vm: RwLock::new(None),
                    capture_ctx: RwLock::new(None),
                    backtracking_vm,
                    backtracking_jit: None,
                });
            }
            Err(_) => {
                // DFA JIT failed, fall back to standard engine selection
            }
        }
    }

    // JIT not available or failed - fall back to standard engine selection
    compile_from_hir(hir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::nfa::compile as nfa_compile;
    use crate::parser::parse;

    fn make_regex(pattern: &str) -> CompiledRegex {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        // Use HIR-based compilation to enable Shift-Or
        compile_from_hir(&hir).unwrap()
    }

    fn make_regex_legacy(pattern: &str) -> CompiledRegex {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = nfa_compile(&hir).unwrap();
        compile(nfa).unwrap()
    }

    #[test]
    fn test_is_match() {
        let re = make_regex("hello");
        assert!(re.is_match(b"hello world"));
        assert!(!re.is_match(b"goodbye"));
    }

    #[test]
    fn test_find() {
        let re = make_regex("world");
        assert_eq!(re.find(b"hello world"), Some((6, 11)));
    }

    #[test]
    fn test_alternation() {
        let re = make_regex("cat|dog");
        assert!(re.is_match(b"I have a cat"));
        assert!(re.is_match(b"I have a dog"));
        assert!(!re.is_match(b"I have a bird"));
    }

    #[test]
    fn test_class() {
        let re = make_regex("[0-9]+");
        assert!(re.is_match(b"abc123def"));
        assert!(!re.is_match(b"abcdef"));
    }

    #[test]
    fn test_legacy_api() {
        // Test NFA-based compilation (uses LazyDfa, not Shift-Or)
        let re = make_regex_legacy("hello");
        assert!(re.is_match(b"hello world"));
        assert!(!re.is_match(b"goodbye"));
    }

    // Prefilter integration tests

    #[test]
    fn test_prefilter_single_literal() {
        // Pattern with literal prefix - simple literal pattern (no . or classes)
        let re = make_regex("hello");
        assert!(re.is_match(b"say hello world"));
        assert!(re.is_match(b"hello"));
        assert!(!re.is_match(b"goodbye"));
    }

    #[test]
    fn test_prefilter_literal_extraction() {
        // Test that literal extraction works
        let ast = parse("needle").unwrap();
        let hir = translate(&ast).unwrap();
        let lits = crate::literal::extract_literals(&hir);
        assert_eq!(lits.prefixes.len(), 1, "Should have 1 prefix");
        assert_eq!(lits.prefixes[0], b"needle", "Prefix should be 'needle'");
    }

    #[test]
    fn test_prefilter_with_dot_star() {
        // Test pattern with .* (uses character class)
        let re = make_regex("hello.*world");
        // Direct matches
        assert!(re.is_match(b"hello world"));
        assert!(re.is_match(b"helloworld"));
        assert!(re.is_match(b"hello to the world"));
        // With prefilter skip
        assert!(re.is_match(b"say hello world"));
        assert!(re.is_match(b"say hello to the world"));
        // Non-matches
        assert!(!re.is_match(b"hello"));
        assert!(!re.is_match(b"world"));
    }

    #[test]
    fn test_prefilter_alternation() {
        // Alternation pattern should extract multiple prefixes for Teddy
        let re = make_regex("cat|dog|bird");
        assert!(re.is_match(b"I have a cat"));
        assert!(re.is_match(b"I have a dog"));
        assert!(re.is_match(b"I have a bird"));
        assert!(!re.is_match(b"I have a fish"));
    }

    #[test]
    fn test_prefilter_find_position() {
        // Verify prefilter returns correct position
        let re = make_regex("needle");
        let haystack = b"xxxxxxxxxxxxxxxxxneedlexxxxxxxx";
        let result = re.find(haystack);
        assert_eq!(result, Some((17, 23)));
    }

    #[test]
    fn test_prefilter_large_input() {
        // Test prefilter with large input to exercise SIMD path
        let re = make_regex("needle");
        let mut haystack = vec![b'x'; 10000];
        haystack[5000..5006].copy_from_slice(b"needle");
        assert_eq!(re.find(&haystack), Some((5000, 5006)));
    }

    #[test]
    fn test_prefilter_no_match() {
        // Prefilter should correctly report no match
        let re = make_regex("needle");
        let haystack = vec![b'x'; 10000];
        assert_eq!(re.find(&haystack), None);
        assert!(!re.is_match(&haystack));
    }

    #[test]
    fn test_prefilter_multiple_matches() {
        // Prefilter should find first match
        let re = make_regex("ab");
        assert_eq!(re.find(b"xxxxabxxxxabxxxx"), Some((4, 6)));
    }

    #[test]
    fn test_no_prefilter_class_start() {
        // Patterns starting with class shouldn't have prefilter
        let re = make_regex("[abc]hello");
        assert!(re.is_match(b"ahello"));
        assert!(re.is_match(b"bhello"));
        assert!(!re.is_match(b"dhello"));
    }

    // TaggedNfa integration tests (backrefs, lookaround, non-greedy)
    // These patterns trigger the TaggedNfaEngine path when JIT is enabled

    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    mod tagged_nfa_integration {
        use super::*;
        use crate::engine::compile_with_jit;

        fn make_jit_regex(pattern: &str) -> CompiledRegex {
            let ast = parse(pattern).unwrap();
            let hir = translate(&ast).unwrap();
            compile_with_jit(&hir).unwrap()
        }

        #[test]
        fn test_backref_simple() {
            // Pattern with backref should use TaggedNfa
            let re = make_jit_regex(r"(a)\1");
            assert!(re.is_match(b"aa"));
            assert!(!re.is_match(b"ab"));
            assert_eq!(re.find(b"aa"), Some((0, 2)));
        }

        #[test]
        fn test_backref_captures() {
            // Verify captures work with backrefs
            let re = make_jit_regex(r"(abc)\1");
            let caps = re.captures(b"abcabc").unwrap();
            assert_eq!(caps.len(), 2); // Group 0 + Group 1
            assert_eq!(caps[0], Some((0, 6))); // Full match
            assert_eq!(caps[1], Some((0, 3))); // Group 1: "abc"
        }

        #[test]
        fn test_positive_lookahead() {
            // Positive lookahead
            let re = make_jit_regex(r"a(?=b)");
            assert!(re.is_match(b"ab"));
            assert!(!re.is_match(b"ac"));
            assert_eq!(re.find(b"ab"), Some((0, 1))); // Only 'a' matched
        }

        #[test]
        fn test_negative_lookahead() {
            // Negative lookahead
            let re = make_jit_regex(r"a(?!b)");
            assert!(re.is_match(b"ac"));
            assert!(!re.is_match(b"ab"));
            assert_eq!(re.find(b"ac"), Some((0, 1)));
        }

        #[test]
        fn test_positive_lookbehind() {
            // Positive lookbehind
            let re = make_jit_regex(r"(?<=a)b");
            assert!(re.is_match(b"ab"));
            assert!(!re.is_match(b"cb"));
            assert_eq!(re.find(b"ab"), Some((1, 2))); // Only 'b' matched
        }

        #[test]
        fn test_negative_lookbehind() {
            // Negative lookbehind
            let re = make_jit_regex(r"(?<!a)b");
            assert!(re.is_match(b"cb"));
            assert!(!re.is_match(b"ab"));
            assert_eq!(re.find(b"cb"), Some((1, 2)));
        }

        #[test]
        fn test_non_greedy_star() {
            // Non-greedy quantifier
            let re = make_jit_regex(r"a*?b");
            assert_eq!(re.find(b"b"), Some((0, 1))); // Zero a's
            assert_eq!(re.find(b"ab"), Some((0, 2))); // One a
            assert_eq!(re.find(b"aaab"), Some((0, 4))); // Multiple a's
        }

        #[test]
        fn test_non_greedy_plus() {
            // Non-greedy plus
            let re = make_jit_regex(r"a+?b");
            assert_eq!(re.find(b"ab"), Some((0, 2))); // One a
            assert_eq!(re.find(b"aaab"), Some((0, 4))); // Multiple a's
            assert_eq!(re.find(b"b"), None); // Need at least one a
        }

        #[test]
        fn test_complex_lookahead_with_capture() {
            // Lookahead with capture group
            let re = make_jit_regex(r"(foo)(?=bar)");
            assert!(re.is_match(b"foobar"));
            assert!(!re.is_match(b"foobaz"));
            let caps = re.captures(b"foobar").unwrap();
            assert_eq!(caps[0], Some((0, 3))); // Full match: "foo"
            assert_eq!(caps[1], Some((0, 3))); // Group 1: "foo"
        }

        #[test]
        fn test_nested_backrefs() {
            // Nested capture with backref
            let re = make_jit_regex(r"((a)(b))\1");
            assert!(re.is_match(b"abab"));
            assert!(!re.is_match(b"abba"));
            assert_eq!(re.find(b"abab"), Some((0, 4)));
        }

        #[test]
        fn test_find_at_with_backref() {
            // Test find_at functionality with backref pattern
            let re = make_jit_regex(r"(x)\1");
            // Input: "axxbxx"
            //        012345
            // First match at position 1: "xx"
            // Second match at position 4: "xx"
            let input = b"axxbxx";
            assert_eq!(re.find(input), Some((1, 3)));
        }
    }
}
