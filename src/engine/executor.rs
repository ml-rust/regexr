//! Unified executor for compiled regex patterns.
//!
//! The executor uses a prefilter (when available) to quickly skip to candidate
//! positions before engaging the full regex engine.

use std::sync::{Arc, OnceLock, RwLock};

use crate::dfa::{CacheCeilingExceeded, EagerDfa, EagerScanBudgetExceeded, LazyDfa};
use crate::error::Result;
use crate::hir::Hir;
use crate::literal::{extract_literals, Prefilter, ReverseSuffixSearch};
use crate::nfa::tagged::TaggedNfaEngine;
use crate::nfa::{self, Nfa};
use crate::vm::backtracking::{BudgetExhausted, CaptureSlots};
use crate::vm::{
    BacktrackingVm, CodepointClassMatcher, OnePass, PikeVm, PikeVmContext, ShiftOr, ShiftOrWide,
};

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
use crate::jit;

use super::dfa_pool::LazyDfaPool;
use super::{needs_boundary_aware_empty_match, select_engine, select_engine_from_hir, EngineType};

/// Runs `search` on `dfa` when the caller already checked one out, or checks
/// one out from `pool` for just this call.
///
/// The `Option<&mut LazyDfa>` dispatch shared by
/// [`CompiledRegex::find_engine_from`] and [`CompiledRegex::find_at_pos`]: both
/// take an optional caller-held instance and fall back to the pool.
fn with_lazy_dfa<T>(
    pool: &LazyDfaPool,
    dfa: Option<&mut LazyDfa>,
    search: impl FnOnce(&mut LazyDfa) -> T,
) -> T {
    match dfa {
        Some(lazy) => search(lazy),
        None => pool.with(search),
    }
}

/// Runs a lazy-DFA search, falling back to the PikeVM when the DFA gives up on
/// its state-cache ceiling.
///
/// Past the ceiling the DFA's transition table is incomplete, so its result
/// would be a false negative rather than a real answer. The same question is
/// re-run on the PikeVM, which caches no states and so has no ceiling to hit.
fn lazy_dfa_or_pikevm<T>(
    dfa: &mut LazyDfa,
    search: impl FnOnce(&mut LazyDfa) -> std::result::Result<T, CacheCeilingExceeded>,
    fallback: impl FnOnce(&PikeVm) -> T,
) -> T {
    match search(dfa) {
        Ok(result) => result,
        Err(_) => fallback(&PikeVm::from_arc(dfa.nfa_arc())),
    }
}

/// Builds the `LazyDfa` used to re-run a search after `EagerDfa::find_from`
/// gives up on its scan budget (see [`EagerScanBudgetExceeded`]).
///
/// Cache limit is unbounded here because this fallback is only reached once
/// per query, on the already-rare metered path, so there is no repeated-flush
/// cost to bound against.
fn eager_scan_fallback(nfa: &Arc<Nfa>) -> LazyDfa {
    let mut fallback = LazyDfa::new((**nfa).clone());
    fallback.set_cache_limit(usize::MAX);
    fallback
}

/// A compiled regex ready for execution.
pub struct CompiledRegex {
    inner: CompiledInner,
    prefilter: Prefilter,
    /// How many bytes past a prefilter candidate the match starts.
    ///
    /// Zero for every ordinary prefilter, whose candidates are match starts.
    /// Non-zero when the literal being searched for is a leading positive
    /// lookbehind's (see [`crate::literal::Literals::prefix_offset`]): the
    /// candidate is then the literal, and the match begins that many bytes
    /// after it.
    prefix_offset: usize,
    /// Whether [`CompiledRegex::find_from_with`] can skip straight to
    /// [`EagerDfa::find_from_simple`].
    ///
    /// Set only where a `CompiledInner::EagerDfa` is built (`compile_from_hir`)
    /// and only when there is no prefilter to consult and the DFA takes the
    /// plain unanchored loop — see [`EagerDfa::is_simple_scan`]. Every other
    /// engine leaves this `false` and keeps the generic dispatch chain, which
    /// remains the only path that can answer for them.
    simple_eager_scan: bool,
    /// The end-first search for patterns shaped like `\w+(?=ing\b)`, when the
    /// gate accepts the pattern.
    ///
    /// Set only where a tagged-NFA engine is built — the shape the gate accepts
    /// always contains a lookaround, so it is the only engine family such a
    /// pattern reaches — and `None` everywhere else. Consulted by
    /// [`CompiledRegex::find_from_inner`], whose anchored confirm is
    /// [`CompiledRegex::find_at_pos`], i.e. the held engine's own `match_at`.
    reverse_suffix: Option<ReverseSuffixSearch>,
    /// Fallback NFA for captures when using Shift-Or or LazyDfa.
    /// Populated at construction (or left permanently `None` for engines that
    /// extract captures themselves) — the `RwLock` exists for interior mutability
    /// of the *derived* engines below, not because this field is built lazily.
    capture_nfa: RwLock<Option<Nfa>>,
    /// Deterministic capture engine, when the capture NFA is one-pass.
    /// Replaces the PikeVM second pass with a single linear scan.
    /// Lazily compiled from `capture_nfa` on first `captures()` call: building
    /// it is a measurable share of `Regex::new` (closure/guard construction),
    /// yet plain `find`/`is_match` callers never read it. See
    /// [`CompiledRegex::one_pass`].
    one_pass: OnceLock<Option<OnePass>>,
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
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    backtracking_jit: Option<jit::BacktrackingJit>,
}

impl std::fmt::Debug for CompiledRegex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledRegex")
            .field("engine", &self.engine_name())
            .field("prefilter", &self.prefilter)
            // The prefilter alone misdescribes how these patterns are searched:
            // the end-first search replaces it outright.
            .field("reverse_suffix", &self.reverse_suffix.is_some())
            .finish_non_exhaustive()
    }
}

#[allow(clippy::large_enum_variant)]
enum CompiledInner {
    PikeVm(PikeVm),
    ShiftOr(ShiftOr),
    /// Wide Shift-Or for patterns with 65-256 positions.
    /// Uses [u64; 4] for 256-bit state vectors.
    ShiftOrWide(ShiftOrWide),
    /// A pool of lazy DFAs rather than one shared instance: a lazy search
    /// mutates its state cache, so sharing would serialize concurrent searches
    /// on a single `Regex` for their whole duration.
    LazyDfa(LazyDfaPool),
    /// Pre-materialized DFA for fast matching without JIT.
    /// Used for patterns that benefit from eager state computation.
    ///
    /// The `Arc<Nfa>` is kept alongside so the unanchored search can hand off
    /// to a fresh `LazyDfa` on [`EagerScanBudgetExceeded`] without rebuilding
    /// the NFA — see [`lazy_dfa_or_pikevm`].
    EagerDfa(EagerDfa, Arc<Nfa>),
    /// Fast codepoint-level matching for single character class patterns.
    CodepointClass(CodepointClassMatcher),
    /// Backtracking VM engine for patterns with backreferences.
    /// Uses PCRE-style backtracking (non-JIT version of BacktrackingJit).
    BacktrackingVm(BacktrackingVm),
    /// Tagged NFA interpreter for patterns with lookaround or non-greedy.
    /// Uses liveness analysis for efficient single-pass capture extraction.
    /// Always available (no JIT required).
    TaggedNfaInterp(TaggedNfaEngine),
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    Jit(jit::CompiledRegex),
    /// Tagged NFA JIT engine for patterns with lookaround or non-greedy.
    /// Uses liveness analysis for efficient single-pass capture extraction.
    /// JIT compiles the NFA to native code for better performance.
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    TaggedNfaJit(jit::TaggedNfaJit),
    /// Backtracking JIT engine for patterns with backreferences.
    /// Uses PCRE-style backtracking for fast backreference matching.
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    Backtracking(jit::BacktrackingJit),
    /// JIT-compiled Shift-Or engine for word boundary patterns.
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    JitShiftOr(jit::JitShiftOr),
}

/// Decides when prefilter-driven verification has stopped paying for itself.
///
/// Verifying one candidate at a time is a win when a failed attempt gives up
/// near the candidate. It is a trap when the pattern consumes a long run before
/// failing and the prefilter keeps most positions: `(?:a|a)+$` over `"aaaa…"`
/// makes every byte a candidate and every attempt scan to the end, so the search
/// costs one engine pass per byte.
///
/// Every engine has a complete linear-time search of its own
/// ([`CompiledRegex::find_engine_from`]), so the escape is to stop verifying and
/// hand the rest of the input to it. This tracks how many attempts have failed
/// and how much input they span: past [`MAX_ATTEMPTS`], a prefilter still
/// keeping more than one position in [`MIN_SELECTIVITY`] is not filtering, and
/// the handoff is taken. A prefilter that is genuinely selective never trips it
/// and keeps the candidate loop for the whole search.
struct PrefilterDrive {
    attempts: usize,
    first_candidate: usize,
}

/// Failed attempts to allow before the selectivity of a prefilter is judged.
/// The handoff costs one engine pass, so this bounds the waste at a constant
/// number of passes rather than one per candidate.
const MAX_ATTEMPTS: usize = 64;

/// One candidate in this many positions is the point below which a prefilter is
/// discarding enough of the input to be worth verifying position by position.
const MIN_SELECTIVITY: usize = 8;

/// Outcome of [`CompiledRegex::captures_one_pass`].
enum OnePassSearch<T> {
    Match(T),
    /// No candidate position matched, so neither will any other.
    NoMatch,
    /// The candidate loop was abandoned; resume the general search here.
    GaveUp(usize),
    /// The prefilter is not a source of anchored start positions.
    NotApplicable,
}

impl PrefilterDrive {
    fn new() -> Self {
        Self {
            attempts: 0,
            first_candidate: 0,
        }
    }

    /// Records a failed attempt at `candidate`. Returns true when the caller
    /// should abandon the loop and search from `candidate` with the engine.
    fn give_up(&mut self, candidate: usize) -> bool {
        if self.attempts == 0 {
            self.first_candidate = candidate;
        }
        self.attempts += 1;
        self.attempts > MAX_ATTEMPTS
            && self.attempts * MIN_SELECTIVITY > candidate - self.first_candidate + 1
    }
}

impl CompiledRegex {
    /// Returns the name of the engine being used (for debugging).
    pub fn engine_name(&self) -> &'static str {
        match &self.inner {
            CompiledInner::PikeVm(_) => "PikeVm",
            CompiledInner::ShiftOr(_) => "ShiftOr",
            CompiledInner::ShiftOrWide(_) => "ShiftOrWide",
            CompiledInner::LazyDfa(_) => "LazyDfa",
            CompiledInner::EagerDfa(_, _) => "EagerDfa",
            CompiledInner::CodepointClass(_) => "CodepointClass",
            CompiledInner::BacktrackingVm(_) => "BacktrackingVm",
            CompiledInner::TaggedNfaInterp(_) => "TaggedNfa",
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::Jit(_) => "Jit",
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::TaggedNfaJit(_) => "TaggedNfaJit",
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::Backtracking(_) => "BacktrackingJit",
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::JitShiftOr(_) => "JitShiftOr",
        }
    }

    /// Gets or compiles the one-pass capture engine from `capture_nfa`, memoizing
    /// the result (including a definitive "not one-pass" answer) behind a
    /// [`OnceLock`] rather than a `RwLock<Option<_>>`: unlike `capture_vm`, a
    /// `None` result here is common (most patterns are not one-pass) and must
    /// still be remembered, or every `captures()` call would re-run
    /// `OnePass::compile` — exactly the cost this laziness exists to avoid.
    ///
    /// `capture_nfa` is itself already fully populated (or permanently `None`)
    /// by the time `CompiledRegex` is constructed, so no ordering concerns arise
    /// between the two: this only ever reads it, never triggers its own lazy
    /// construction.
    fn one_pass(&self) -> Option<&OnePass> {
        self.one_pass
            .get_or_init(|| {
                self.capture_nfa
                    .read()
                    .unwrap()
                    .as_ref()
                    .and_then(OnePass::compile)
            })
            .as_ref()
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

    /// Whether every search already scans the haystack for the pattern's
    /// required literal, making an outside rejection scan for it redundant.
    ///
    /// True exactly when a reverse-suffix search is held: its first act is a
    /// `memmem` scan for the lookahead's leading literal over the same text, and
    /// it reports no match when that literal is absent. The two literals are the
    /// same one by construction — for the shape the gate accepts,
    /// `<class run>(?=<literal>…)`, the run contributes nothing to
    /// [`crate::literal::required_literal`], so the lookahead's leading literal
    /// is the only candidate it has to choose from, and the gate only accepts a
    /// lookahead whose extraction yields exactly one prefix at offset zero,
    /// which is the prefix `required_literal` takes.
    ///
    /// Answering true only ever removes a rejection scan, never a check that
    /// decides which match is reported, so a caller is free to ignore it.
    #[inline]
    pub(crate) fn scans_for_required_literal(&self) -> bool {
        self.reverse_suffix.is_some()
    }

    /// Returns true if the pattern matches anywhere in the input.
    pub fn is_match(&self, input: &[u8]) -> bool {
        // Neither of these searches is expressible as the candidate loop below,
        // which verifies match *starts* one at a time: an offset prefilter's
        // candidates are not match starts, and the reverse-suffix scan seeks the
        // text a match must *end* at. `find_from_inner` owns both translations
        // and answers the same question, so defer to it rather than restating
        // either here — restating them is how the two entry points drift apart.
        if self.prefix_offset != 0 || self.reverse_suffix.is_some() {
            return self.find(input).is_some();
        }

        // Fast path: if prefilter can provide full match bounds (TeddyFull),
        // finding any match means there's a match
        if self.prefilter.is_full_match() {
            return self.prefilter.find_full_match(input, 0).is_some();
        }

        // Use prefilter to skip to candidate positions
        if !self.prefilter.is_none() {
            if self.engine_searches_single_pass() {
                let first = self.prefilter.find_candidates(input).next();
                return match first {
                    Some(first) => self.find_engine_from_boundary(input, first, None).is_some(),
                    None => false,
                };
            }
            let mut drive = PrefilterDrive::new();
            for candidate in self.prefilter.find_candidates(input) {
                if self.is_match_at(input, candidate) {
                    return true;
                }
                if drive.give_up(candidate) {
                    return self
                        .find_engine_from_boundary(input, candidate, None)
                        .is_some();
                }
            }
            return false;
        }

        // No prefilter - check from start
        match &self.inner {
            CompiledInner::PikeVm(vm) => vm.is_match(input),
            CompiledInner::ShiftOr(so) => so.is_match(input),
            CompiledInner::ShiftOrWide(so) => so.is_match(input),
            CompiledInner::LazyDfa(pool) => pool.with(|dfa| {
                lazy_dfa_or_pikevm(
                    dfa,
                    |d| d.find(input).map(|found| found.is_some()),
                    |vm| vm.is_match(input),
                )
            }),
            CompiledInner::EagerDfa(dfa, nfa) => match dfa.find(input) {
                Ok(found) => found.is_some(),
                Err(EagerScanBudgetExceeded) => {
                    let mut fallback = eager_scan_fallback(nfa);
                    lazy_dfa_or_pikevm(
                        &mut fallback,
                        |d| d.find(input).map(|found| found.is_some()),
                        |vm| vm.is_match(input),
                    )
                }
            },
            CompiledInner::CodepointClass(matcher) => matcher.is_match(input),
            CompiledInner::BacktrackingVm(vm) => vm.find(input).is_some(),
            CompiledInner::TaggedNfaInterp(engine) => engine.is_match(input),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::Jit(jit) => jit.is_match(input),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::TaggedNfaJit(engine) => engine.is_match(input),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::Backtracking(jit) => jit.is_match(input),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::JitShiftOr(jit) => jit.find(input).is_some(),
        }
    }

    /// Whether the engine has no way to test one start position in isolation.
    ///
    /// For these, `find_at_pos` is the *same* unanchored search as
    /// `find_engine_from` — it scans to the end of the input rather than
    /// answering about `pos` alone. Handing them candidates one at a time is
    /// then not just wasteful but quadratic: each of k candidates rescans the
    /// remaining n bytes. `a+b` over 100 KB of text with no match took 146 ms
    /// under the DFA JIT against 0.03 ms interpreted, purely from this.
    ///
    /// The prefilter still pays: it skips to the first candidate, and the single
    /// scan from there finds the leftmost match. That is sound because a
    /// prefilter never rules out a real match start.
    ///
    /// The engines with a genuine anchored primitive — Shift-Or's
    /// `try_match_at`, the DFA families' `find_at` — are driven candidate by
    /// candidate as before.
    #[inline]
    fn engine_searches_single_pass(&self) -> bool {
        match &self.inner {
            CompiledInner::PikeVm(_)
            | CompiledInner::BacktrackingVm(_)
            | CompiledInner::TaggedNfaInterp(_)
            | CompiledInner::CodepointClass(_) => true,
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::Jit(_)
            | CompiledInner::TaggedNfaJit(_)
            | CompiledInner::Backtracking(_) => true,
            _ => false,
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
        self.find_from(input, 0)
    }

    /// Finds the leftmost match starting at or after `from`, returning (start, end).
    ///
    /// This is the resume point used by iteration. The engines always get the
    /// *whole* input plus a start offset — never a slice beginning at `from` — so
    /// `^`, `\b`/`\B` and lookbehind see the real text to the left of the resume
    /// position. The prefilter stays on the hot path: it is simply scanned from
    /// `from` instead of from 0.
    pub fn find_from(&self, input: &[u8], from: usize) -> Option<(usize, usize)> {
        self.find_from_with(input, from, None)
    }

    /// [`Self::find_from`] on a lazy DFA the caller already holds.
    ///
    /// Iteration calls this once per match, and for the lazy DFA the pool
    /// round trip around a single search costs more than the search itself.
    /// A caller that will search many times — see [`PooledDfa`] — checks one
    /// instance out up front and passes it here, so the pool is untouched in
    /// between.
    ///
    /// `dfa` is ignored by every other engine, and passing `None` is always
    /// correct: the lazy DFA then takes an instance from the pool per search,
    /// exactly as [`Self::find_from`] does.
    pub fn find_from_with(
        &self,
        input: &[u8],
        from: usize,
        dfa: Option<&mut LazyDfa>,
    ) -> Option<(usize, usize)> {
        if from > input.len() {
            return None;
        }
        // Iteration pays the generic chain below — a prefilter test, a 12-arm
        // engine dispatch and the anchor re-checks inside `EagerDfa::find_from`
        // — once per match, all of it invariant for the whole search. When the
        // flag says those questions are already answered, go straight to the
        // scan loop; the codepoint-boundary rule below still applies, so it is
        // repeated here rather than falling back into the generic path for the
        // retry.
        if self.simple_eager_scan {
            if let CompiledInner::EagerDfa(eager, _) = &self.inner {
                return Self::find_from_simple_scan(eager, input, from);
            }
        }
        self.find_from_generic(input, from, dfa)
    }

    /// The straight-line scan for [`CompiledRegex::find_from_with`]'s
    /// specialized case.
    ///
    /// Kept in its own frame rather than inlined beside the generic loop: the
    /// two share nothing, and merging them makes one frame whose register and
    /// spill pressure is the union of both, which costs the generic path
    /// measurably even though it never executes this.
    fn find_from_simple_scan(
        eager: &EagerDfa,
        input: &[u8],
        from: usize,
    ) -> Option<(usize, usize)> {
        let mut from = from;
        loop {
            let (start, end) = eager.find_from_simple(input, from)?;
            if crate::nfa::is_utf8_boundary(input, start) {
                return Some((start, end));
            }
            from = start + 1;
        }
    }

    /// The engine-agnostic search, for every case
    /// [`CompiledRegex::find_from_with`] does not specialize.
    ///
    /// A match never starts inside a codepoint. Byte-level constructs (`.`,
    /// byte classes) can match a single continuation byte, so an engine
    /// scanning start positions will happily report one; the PikeVM and the
    /// tagged NFA already refuse those starts, and this makes the rest agree.
    /// Rejecting a match and resuming one byte later converges on the leftmost
    /// match that does start at a boundary.
    fn find_from_generic(
        &self,
        input: &[u8],
        from: usize,
        mut dfa: Option<&mut LazyDfa>,
    ) -> Option<(usize, usize)> {
        let mut from = from;
        loop {
            let (start, end) = self.find_from_inner(input, from, dfa.as_deref_mut())?;
            if crate::nfa::is_utf8_boundary(input, start) {
                return Some((start, end));
            }
            from = start + 1;
        }
    }

    fn find_from_inner(
        &self,
        input: &[u8],
        from: usize,
        mut dfa: Option<&mut LazyDfa>,
    ) -> Option<(usize, usize)> {
        // The end-first search, before every prefilter branch. It must come
        // first: the branches below are entered on `engine_searches_single_pass`,
        // which is true for both tagged-NFA engines — the only ones that can
        // hold a reverse-suffix search — so anything placed after them is
        // unreachable for exactly the patterns this exists for.
        //
        // It replaces the search outright rather than filtering it: it visits
        // one start per class run instead of one per position, and the confirm
        // is anchored, so a failed run costs a single attempt.
        if let Some(search) = &self.reverse_suffix {
            return search.find(input, from, |start| self.find_at_pos(input, start, None));
        }

        // Fast path: if prefilter can provide full match bounds (TeddyFull),
        // return directly without running the NFA
        if self.prefilter.is_full_match() {
            return self.prefilter.find_full_match(input, from);
        }

        // Use prefilter to skip to candidate positions
        // IMPORTANT: Use find_at_pos (exact position) not find_at (linear search from pos)
        // The prefilter already tells us where candidates are - we only need to verify each one.
        if !self.prefilter.is_none() {
            // With an offset prefilter the candidate is the literal and the
            // match starts `prefix_offset` bytes later, so the scan itself has
            // to begin BEHIND the resume point: a match starting exactly at
            // `from` has its literal at `from - prefix_offset`, and scanning
            // from `from` would never see it. `match_start` then discards the
            // candidates that translate to a position before `from` (already
            // reported) or past the end of the input.
            let scan_from = from.saturating_sub(self.prefix_offset);
            let match_start = |candidate: usize| {
                candidate
                    .checked_add(self.prefix_offset)
                    .filter(|&start| start >= from && start <= input.len())
            };

            if self.engine_searches_single_pass() {
                // The caller applies the codepoint-boundary rule.
                let first = self
                    .prefilter
                    .find_candidates_from(input, scan_from)
                    .find_map(match_start)?;
                return self.find_engine_from(input, first, dfa);
            }
            let mut drive = PrefilterDrive::new();
            for candidate in self.prefilter.find_candidates_from(input, scan_from) {
                let Some(start) = match_start(candidate) else {
                    continue;
                };
                if let Some(result) = self.find_at_pos(input, start, dfa.as_deref_mut()) {
                    return Some(result);
                }
                if drive.give_up(start) {
                    // The caller applies the codepoint-boundary rule.
                    return self.find_engine_from(input, start, dfa);
                }
            }
            return None;
        }

        // No prefilter - let the engine scan from the resume position
        self.find_engine_from(input, from, dfa)
    }

    /// [`CompiledRegex::find_engine_from`] under the codepoint-boundary rule, for
    /// callers that are not already inside [`CompiledRegex::find_from`]'s loop.
    fn find_engine_from_boundary(
        &self,
        input: &[u8],
        from: usize,
        mut dfa: Option<&mut LazyDfa>,
    ) -> Option<(usize, usize)> {
        let mut from = from;
        loop {
            let (start, end) = self.find_engine_from(input, from, dfa.as_deref_mut())?;
            if crate::nfa::is_utf8_boundary(input, start) {
                return Some((start, end));
            }
            from = start + 1;
        }
    }

    /// Runs the engine's own unanchored search from `from`, with no prefilter.
    ///
    /// Every engine takes the full input plus a start offset. The two engines
    /// whose generated code has no start-offset parameter (the backtracking and
    /// tagged-NFA JITs) fall back internally to their interpreters for patterns
    /// that read left context, so slicing never hides preceding bytes.
    fn find_engine_from(
        &self,
        input: &[u8],
        from: usize,
        dfa: Option<&mut LazyDfa>,
    ) -> Option<(usize, usize)> {
        match &self.inner {
            CompiledInner::PikeVm(vm) => vm.find_from(input, from),
            CompiledInner::ShiftOr(so) => so.find_at(input, from),
            CompiledInner::ShiftOrWide(so) => so.find_at(input, from),
            CompiledInner::LazyDfa(pool) => with_lazy_dfa(pool, dfa, |lazy| {
                lazy_dfa_or_pikevm(
                    lazy,
                    |d| d.find_from(input, from),
                    |vm| vm.find_from(input, from),
                )
            }),
            CompiledInner::EagerDfa(dfa, nfa) => match dfa.find_from(input, from) {
                Ok(result) => result,
                Err(EagerScanBudgetExceeded) => {
                    let mut fallback = eager_scan_fallback(nfa);
                    lazy_dfa_or_pikevm(
                        &mut fallback,
                        |d| d.find_from(input, from),
                        |vm| vm.find_from(input, from),
                    )
                }
            },
            CompiledInner::CodepointClass(matcher) => matcher.find_from(input, from),
            CompiledInner::BacktrackingVm(vm) => vm.find_at(input, from),
            CompiledInner::TaggedNfaInterp(engine) => engine.find_at(input, from),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::Jit(jit) => jit.find_from(input, from),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::TaggedNfaJit(engine) => engine.find_at(input, from),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::Backtracking(jit) => jit.find_from(input, from),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::JitShiftOr(jit) => jit.find_from(input, from),
        }
    }

    /// Returns capture groups for the first match.
    ///
    /// For Shift-Or, LazyDfa, and JIT engines, this uses a two-pass strategy:
    /// 1. Use the fast engine to find match bounds
    /// 2. Re-run PikeVm at that match start to extract captures
    ///
    /// TaggedNfa performs single-pass capture extraction natively.
    pub fn captures(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        self.captures_from(input, 0)
    }

    /// Returns capture groups for the first match starting at or after `from`.
    ///
    /// Like [`CompiledRegex::find_from`], the engines receive the whole input and
    /// an explicit start offset, so a resumed search still sees the text to the
    /// left of `from`. All reported slots are absolute input offsets.
    pub fn captures_from(&self, input: &[u8], from: usize) -> Option<Vec<Option<(usize, usize)>>> {
        self.captures_from_with(input, from, None)
    }

    /// [`Self::captures_from`] on a lazy DFA the caller already holds.
    ///
    /// See [`Self::find_from_with`]: the instance only ever reaches the
    /// two-pass path's bounds search, which is the only part of capture
    /// extraction that runs the lazy DFA.
    pub fn captures_from_with(
        &self,
        input: &[u8],
        from: usize,
        dfa: Option<&mut LazyDfa>,
    ) -> Option<Vec<Option<(usize, usize)>>> {
        if from > input.len() {
            return None;
        }
        // With an end-first search, the forward scan every arm below performs is
        // the cost this whole path exists to avoid. Locating the match first and
        // resuming the capture pass *at* its start keeps the two in agreement
        // for free: every arm searches at or after the position it is given, and
        // the search already confirmed a match beginning exactly there, so the
        // one they find is the one `find` reports.
        let from = if self.reverse_suffix.is_some() {
            // Only the tagged engines hold one of these searches, and neither
            // has a lazy DFA to be handed.
            self.find_from_with(input, from, None)?.0
        } else {
            from
        };
        match &self.inner {
            CompiledInner::PikeVm(vm) => vm.captures_from(input, from),
            CompiledInner::CodepointClass(matcher) => matcher.captures_from(input, from),
            CompiledInner::BacktrackingVm(vm) => {
                // BacktrackingVm does single-pass capture extraction
                vm.captures_from(input, from)
            }
            CompiledInner::TaggedNfaInterp(engine) => {
                // TaggedNfa interpreter does single-pass capture extraction
                engine.captures_from(input, from)
            }
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::TaggedNfaJit(engine) => {
                // TaggedNfa JIT does single-pass capture extraction
                engine.captures_from(input, from)
            }
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::Backtracking(jit) => {
                // Backtracking JIT does single-pass capture extraction
                jit.captures_from(input, from)
            }
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::Jit(_) => {
                // Fast path: if we have BacktrackingVm, use it for single-pass capture extraction
                if let Some(ref backtracking_vm) = self.backtracking_vm {
                    return backtracking_vm.captures_from(input, from);
                }

                // Two-pass capture strategy for DFA JIT (fallback)
                self.captures_two_pass(input, from, dfa)
            }
            CompiledInner::ShiftOr(_)
            | CompiledInner::ShiftOrWide(_)
            | CompiledInner::LazyDfa(_)
            | CompiledInner::EagerDfa(_, _) => {
                // Fast path: if we have BacktrackingVm, use it for single-pass capture extraction
                if let Some(ref backtracking_vm) = self.backtracking_vm {
                    return backtracking_vm.captures_from(input, from);
                }

                self.captures_two_pass(input, from, dfa)
            }
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::JitShiftOr(_) => {
                // Use BacktrackingJit for capture extraction if available
                // This is the JIT equivalent of BacktrackingVm used by non-JIT ShiftOr
                if let Some(ref backtracking_jit) = self.backtracking_jit {
                    return backtracking_jit.captures_from(input, from);
                }

                // Fall back to two-pass strategy if no BacktrackingJit
                self.captures_two_pass(input, from, dfa)
            }
        }
    }

    /// [`Self::captures_from`] under an explicit step budget.
    ///
    /// [`BudgetExhausted`] means the budget ran out before the search finished. Only the
    /// backtracking engines can report that, and only backreference patterns
    /// reach them: every other engine here is linear in the input and always
    /// returns `Ok`.
    pub fn try_captures_from(
        &self,
        input: &[u8],
        from: usize,
        limit: u64,
    ) -> std::result::Result<Option<CaptureSlots>, BudgetExhausted> {
        if from > input.len() {
            return Ok(None);
        }
        if let Some(ref vm) = self.backtracking_vm {
            return vm.try_captures_from(input, from, limit);
        }
        match &self.inner {
            CompiledInner::BacktrackingVm(vm) => vm.try_captures_from(input, from, limit),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::Backtracking(jit) => jit.try_captures_from(input, from, limit),
            _ => Ok(self.captures_from(input, from)),
        }
    }

    /// [`Self::find_from`] under an explicit step budget.
    ///
    /// For a backreference pattern this asks the backtracking engine directly
    /// rather than going through the prefilter. A prefilter only skips start
    /// positions that cannot match, so this finds the same match; it just does
    /// not get the prefilter's head start.
    pub fn try_find_from(
        &self,
        input: &[u8],
        from: usize,
        limit: u64,
    ) -> std::result::Result<Option<(usize, usize)>, BudgetExhausted> {
        if from > input.len() {
            return Ok(None);
        }
        match &self.inner {
            CompiledInner::BacktrackingVm(vm) => vm.try_find_at(input, from, limit),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::Backtracking(jit) => Ok(jit
                .try_captures_from(input, from, limit)?
                .and_then(|c| c[0])),
            _ => Ok(self.find_from(input, from)),
        }
    }

    /// Two-pass capture extraction for engines that only report match bounds:
    /// 1. Find the match bounds with the fast engine (from `from`).
    /// 2. Re-run the cached PikeVM at that exact start position.
    ///
    /// The second pass is given the full input and a start position rather than
    /// a slice starting at the match, so the capture pass evaluates `^`, `\b` and
    /// lookbehind against the same context the first pass used. Slots come back
    /// as absolute offsets.
    ///
    /// A capture NFA is only built when the pattern actually has groups to
    /// extract (see the `capture_nfa` fields set during compilation). With no
    /// groups there is nothing for a second pass to do: the match bounds found in
    /// step 1 *are* the whole capture set, and slot 0 is returned directly.
    ///
    /// Both passes are skipped entirely for a one-pass pattern: [`OnePass`] both
    /// locates the match and writes the slots in a single deterministic scan (see
    /// [`CompiledRegex::captures_one_pass`]), so the match region is walked once
    /// instead of once by the search engine and again by the capture pass.
    fn captures_two_pass(
        &self,
        input: &[u8],
        from: usize,
        dfa: Option<&mut LazyDfa>,
    ) -> Option<Vec<Option<(usize, usize)>>> {
        let mut from = from;
        if let Some(one_pass) = self.one_pass() {
            match self.captures_one_pass(one_pass, input, from) {
                OnePassSearch::Match(slots) => return Some(slots),
                OnePassSearch::NoMatch => return None,
                // The candidate loop stopped paying for itself; every position
                // before `resume` has already been ruled out.
                OnePassSearch::GaveUp(resume) => from = resume,
                OnePassSearch::NotApplicable => {}
            }
        }

        let (match_start, match_end) = self.find_from_with(input, from, dfa)?;

        if let Some(one_pass) = self.one_pass() {
            if let Some(slots) = one_pass.captures_at(input, match_start) {
                // The deterministic scan and the search engine must agree on the
                // match bounds; if they somehow do not, defer to the PikeVM.
                if slots.first().copied().flatten() == Some((match_start, match_end)) {
                    return Some(slots);
                }
            }
        }

        // Use cached PikeVM and context to avoid allocations
        self.get_or_init_capture_vm();
        let vm_ref = self.capture_vm.read().unwrap();
        let vm = match vm_ref.as_ref() {
            Some(vm) => vm,
            // Group-less pattern: the full match is the only slot.
            None => return Some(vec![Some((match_start, match_end))]),
        };
        let mut ctx_ref = self.capture_ctx.write().unwrap();
        let ctx = ctx_ref.as_mut()?;

        vm.captures_with_context(input, ctx, match_start)
    }

    /// Drives [`OnePass`] over candidate start positions, so a match is located
    /// and its slots written by the same scan.
    ///
    /// The two-pass path asks the search engine where the match is and then
    /// re-scans it to recover the groups, which walks the match region twice.
    /// `OnePass` is anchored and deterministic: an attempt at a position that
    /// cannot match stops at the first byte with no transition, so trying the
    /// positions the prefilter keeps replaces the search outright.
    ///
    /// Leftmost-first is preserved because candidates arrive in increasing order
    /// and a prefilter only ever skips positions where no match can begin, so the
    /// first position `OnePass` accepts is the leftmost one.
    ///
    /// A failed attempt is bounded by the pattern, not by the input, but it is not
    /// bounded by a *constant*: `(a{1000})b` over a run of `a`s would scan a
    /// thousand bytes per candidate. [`PrefilterDrive`] watches for exactly that
    /// and hands the rest of the input back to the linear-time search.
    fn captures_one_pass(
        &self,
        one_pass: &OnePass,
        input: &[u8],
        from: usize,
    ) -> OnePassSearch<Vec<Option<(usize, usize)>>> {
        // A full-match prefilter already reports the span without running an
        // engine, and an offset one yields the position of a lookbehind literal
        // rather than a match start. Neither is a source of anchored candidates.
        if self.prefilter.is_full_match() || self.prefix_offset != 0 {
            return OnePassSearch::NotApplicable;
        }

        // A candidate iterator stops one short of the end, because no prefilter
        // byte can live at `input.len()`. A nullable pattern can still match
        // there, so end-of-input is always the last position tried.
        let candidates = self
            .prefilter
            .find_candidates_from(input, from)
            .chain(std::iter::once(input.len()));

        // Allocated once for the whole search rather than once per attempt.
        let mut scratch = vec![None; one_pass.slot_count()];
        let mut slots = vec![None; one_pass.slot_count()];

        let mut drive = PrefilterDrive::new();
        for candidate in candidates {
            // A match never starts inside a codepoint; `find_from` applies the
            // same rule to the engines' answers.
            if !crate::nfa::is_utf8_boundary(input, candidate) {
                continue;
            }
            if one_pass.captures_at_into(input, candidate, &mut scratch, &mut slots) {
                return OnePassSearch::Match(slots);
            }
            if drive.give_up(candidate) {
                return OnePassSearch::GaveUp(candidate);
            }
        }
        OnePassSearch::NoMatch
    }

    /// Check if there's a match starting at `pos`.
    ///
    /// This method passes the full input to allow engines to check context
    /// (e.g., for word boundary assertions).
    fn is_match_at(&self, input: &[u8], pos: usize) -> bool {
        self.find_at_pos(input, pos, None).is_some()
    }

    /// Find a match starting exactly at `pos`.
    ///
    /// This method passes the full input to allow engines to check context
    /// (e.g., for word boundary assertions).
    fn find_at_pos(
        &self,
        input: &[u8],
        pos: usize,
        dfa: Option<&mut LazyDfa>,
    ) -> Option<(usize, usize)> {
        if pos > input.len() {
            return None;
        }
        match &self.inner {
            CompiledInner::PikeVm(vm) => vm.find_at(input, pos),
            CompiledInner::ShiftOr(so) => so.try_match_at(input, pos),
            CompiledInner::ShiftOrWide(so) => so.try_match_at(input, pos),
            CompiledInner::LazyDfa(pool) => with_lazy_dfa(pool, dfa, |lazy| {
                lazy_dfa_or_pikevm(
                    lazy,
                    |d| {
                        d.find_at(input, pos)
                            .map(|found| found.map(|end| (pos, end)))
                    },
                    |vm| vm.find_at(input, pos),
                )
            }),
            CompiledInner::EagerDfa(dfa, _) => dfa.find_at(input, pos).map(|end| (pos, end)),
            CompiledInner::CodepointClass(matcher) => {
                // CodepointClass doesn't support word boundaries, use sliced input
                let slice = &input[pos..];
                matcher.find(slice).map(|(s, e)| (pos + s, pos + e))
            }
            CompiledInner::BacktrackingVm(vm) => vm.find_at(input, pos),
            CompiledInner::TaggedNfaInterp(engine) => engine.match_at(input, pos),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::Jit(jit) => jit.find_at(input, pos),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::TaggedNfaJit(engine) => engine.match_at(input, pos),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::Backtracking(jit) => jit.find_at(input, pos),
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompiledInner::JitShiftOr(jit) => jit.try_match_at(input, pos),
        }
    }

    /// Takes a lazy-DFA instance out of the pool, or `None` when the compiled
    /// engine is not the lazy DFA and there is nothing to hand out.
    pub(crate) fn checkout_lazy_dfa(&self) -> Option<LazyDfa> {
        match &self.inner {
            CompiledInner::LazyDfa(pool) => Some(pool.checkout()),
            _ => None,
        }
    }

    /// Returns an instance taken by [`Self::checkout_lazy_dfa`].
    pub(crate) fn checkin_lazy_dfa(&self, dfa: LazyDfa) {
        if let CompiledInner::LazyDfa(pool) = &self.inner {
            pool.checkin(dfa);
        }
    }
}

/// A lazy DFA held out of a [`CompiledRegex`]'s pool for the lifetime of an
/// iterator, and returned to it on drop.
///
/// Iteration runs one search per match, so a per-search pool round trip is
/// paid once per match — two lock acquisitions around a search that can be a
/// few tens of nanoseconds. Checking one instance out for the whole iteration
/// removes the pool from the hot path entirely, and keeps the state cache warm
/// across matches instead of possibly landing on a different instance each
/// time. That is a cache-warmth difference only: every instance runs the same
/// subset construction over the same NFA, so it cannot change what is matched.
pub struct PooledDfa<'r> {
    regex: &'r CompiledRegex,
    /// `None` when the engine is not the lazy DFA, which is then also what the
    /// searches receive — they fall back to their own engine as usual.
    dfa: Option<LazyDfa>,
}

impl<'r> PooledDfa<'r> {
    /// Takes an instance for `regex`, if its engine has one to give.
    pub fn checkout(regex: &'r CompiledRegex) -> Self {
        Self {
            regex,
            dfa: regex.checkout_lazy_dfa(),
        }
    }

    /// The instance to pass to [`CompiledRegex::find_from_with`] and friends.
    pub fn get(&mut self) -> Option<&mut LazyDfa> {
        self.dfa.as_mut()
    }
}

impl Drop for PooledDfa<'_> {
    fn drop(&mut self) {
        // Dropped while unwinding, the instance goes with the panic rather
        // than being handed to the next search in whatever state it left the
        // cache — the same choice `LazyDfaPool::with` makes. The pool clones
        // its template when it runs dry, so losing one costs only its states.
        if std::thread::panicking() {
            return;
        }
        if let Some(dfa) = self.dfa.take() {
            self.regex.checkin_lazy_dfa(dfa);
        }
    }
}

impl std::fmt::Debug for PooledDfa<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledDfa")
            .field("held", &self.dfa.is_some())
            .finish_non_exhaustive()
    }
}

/// Compiles an NFA into an executable regex (legacy API).
/// Note: This cannot use Shift-Or as it requires HIR for Glushkov construction.
/// Also cannot use prefilter (requires HIR for literal extraction).
pub fn compile(nfa: Nfa) -> Result<CompiledRegex> {
    let engine = select_engine(&nfa);

    let (inner, capture_nfa) = match engine {
        EngineType::PikeVm => (CompiledInner::PikeVm(PikeVm::new(nfa)), None),
        EngineType::TaggedNfa => {
            // `select_engine` works from the NFA alone and never returns this
            // (only `select_engine_from_hir` inspects the HIR for codepoint
            // classes), but the tagged engine needs nothing but an NFA, so this
            // stays a real route rather than a downgrade.
            (
                CompiledInner::TaggedNfaInterp(TaggedNfaEngine::new(nfa)),
                None,
            )
        }
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
                CompiledInner::LazyDfa(LazyDfaPool::new(LazyDfa::new(nfa))),
                capture_nfa,
            )
        }
        EngineType::LazyDfa => {
            let capture_nfa = Some(nfa.clone());
            (
                CompiledInner::LazyDfa(LazyDfaPool::new(LazyDfa::new(nfa))),
                capture_nfa,
            )
        }
        #[cfg(feature = "jit")]
        EngineType::Jit => {
            // JIT not implemented yet, fall back to LazyDfa
            let capture_nfa = Some(nfa.clone());
            (
                CompiledInner::LazyDfa(LazyDfaPool::new(LazyDfa::new(nfa))),
                capture_nfa,
            )
        }
    };

    Ok(CompiledRegex {
        inner,
        prefilter: Prefilter::None, // Can't extract literals from NFA
        prefix_offset: 0,
        simple_eager_scan: false,
        // No HIR here to run the gate on.
        reverse_suffix: None,
        capture_nfa: RwLock::new(capture_nfa),
        one_pass: OnceLock::new(),
        capture_vm: RwLock::new(None),
        capture_ctx: RwLock::new(None),
        backtracking_vm: None,
        #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
        backtracking_jit: None,
    })
}

/// Builds the tagged-NFA interpreter for `hir` from its already-compiled `nfa`.
///
/// Shared by every route that lands on the tagged interpreter — lookaround,
/// codepoint classes, and the JIT paths' fallbacks — so they cannot drift apart
/// in prefilter or capture handling. The NFA is passed in rather than compiled
/// here because the JIT fallback already holds one.
///
/// Safe for any pattern without backreferences: step extraction is bounded by
/// `MAX_EXTRACTED_STEPS`, and when it declines `TaggedNfaEngine` runs the PikeVM
/// it constructs regardless. Captures are single-pass and native to the tagged
/// engine, so no capture NFA is retained.
fn compile_tagged_nfa_interp(hir: &Hir, nfa: Nfa) -> CompiledRegex {
    let literals = extract_literals(hir);
    let prefilter = Prefilter::from_literals(&literals);
    CompiledRegex {
        inner: CompiledInner::TaggedNfaInterp(TaggedNfaEngine::new(nfa)),
        prefilter,
        prefix_offset: literals.prefix_offset,
        simple_eager_scan: false,
        reverse_suffix: ReverseSuffixSearch::new(hir),
        capture_nfa: RwLock::new(None),
        one_pass: OnceLock::new(),
        capture_vm: RwLock::new(None),
        capture_ctx: RwLock::new(None),
        backtracking_vm: None,
        #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
        backtracking_jit: None,
    }
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
            prefix_offset: 0,
            simple_eager_scan: false,
            reverse_suffix: None,
            capture_nfa: RwLock::new(None),
            one_pass: OnceLock::new(),
            capture_vm: RwLock::new(None),
            capture_ctx: RwLock::new(None),
            backtracking_vm: None,
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            backtracking_jit: None,
        });
    }

    // Patterns with lookaround → TaggedNfa interpreter (it handles lookahead via
    // the step model). Non-greedy quantifiers, however, need the precise thread
    // priority of the Pike VM: the step model does not represent non-greedy
    // faithfully (e.g. a bounded repeat followed by `+?` lets the lazy match
    // zero), so any non-greedy pattern falls through to `select_engine_from_hir`,
    // which routes it to PikeVm. (Non-greedy is not on the tiktoken hot path.)
    if hir.props.has_lookaround && !hir.props.has_non_greedy {
        return Ok(compile_tagged_nfa_interp(hir, nfa::compile(hir)?));
    }

    // Extract literals for prefilter
    // Word boundaries: NOW SUPPORTED! The executor passes full input context
    // to engines via find_at_pos(), allowing proper word boundary checking.
    // Anchors: NOW SUPPORTED! LazyDFA/JIT handle start anchor optimization
    // internally, so prefilter can still be used for patterns with anchors.
    let literals = extract_literals(hir);
    let mut prefilter = Prefilter::from_literals(&literals);

    // Capture extraction is a hybrid: `find` uses the fast automaton engine, and
    // the slots come from a second pass (see `captures_two_pass`).
    //
    // That second pass is the PikeVM unless the pattern has backreferences. A
    // backreference makes the match depend on what earlier groups captured,
    // which only a backtracking engine represents, and it is the one construct
    // worth paying for: backtracking explores an unbounded search tree, so it
    // answers under a step budget rather than the linear bound every other
    // engine here meets. Routing capture-only patterns through it — which is
    // what this used to do for *any* pattern with a group — silently gave every
    // `captures()` call that same unbounded cost.
    let needs_backtracking = hir.props.has_backrefs;

    let engine = select_engine_from_hir(hir);

    // The PikeVM owns the leftmost-first match decision, so a full-match literal
    // prefilter must not short-circuit it (see `Prefilter::into_candidate_only`).
    if engine == EngineType::PikeVm {
        prefilter = prefilter.into_candidate_only();
    }

    let (inner, capture_nfa) = match engine {
        EngineType::PikeVm => {
            let nfa = nfa::compile(hir)?;
            (CompiledInner::PikeVm(PikeVm::new(nfa)), None)
        }
        EngineType::TaggedNfa => {
            // Codepoint-class patterns: the same engine `compile_with_jit`
            // builds for them, with the PikeVM kept as the tagged engine's own
            // fallback (see `compile_tagged_nfa_interp`). Returned directly
            // because the tagged engine owns its capture extraction and needs no
            // separate capture NFA.
            return Ok(compile_tagged_nfa_interp(hir, nfa::compile(hir)?));
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
                        CompiledInner::LazyDfa(LazyDfaPool::new(LazyDfa::new(nfa))),
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
                        CompiledInner::LazyDfa(LazyDfaPool::new(LazyDfa::new(nfa))),
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
                    CompiledInner::LazyDfa(LazyDfaPool::new(LazyDfa::new(nfa))),
                    capture_nfa,
                )
            } else {
                // Use EagerDfa for better non-JIT performance on simple patterns.
                // EagerDfa pre-computes all states upfront, eliminating hash lookups.
                let mut lazy = LazyDfa::new(nfa);
                // Already held by `lazy`; cloning the Arc (not the Nfa) so the
                // scan-budget give-up can rebuild a `LazyDfa` without recompiling.
                let nfa_arc = lazy.nfa_arc();
                match EagerDfa::from_lazy(&mut lazy) {
                    Ok(eager) => (CompiledInner::EagerDfa(eager, nfa_arc), capture_nfa),
                    Err(_) => {
                        // Materialization declined (see
                        // EagerMaterializationBudgetExceeded): fall back to a
                        // fresh LazyDfa, which computes the identical states
                        // on demand instead of upfront.
                        (
                            CompiledInner::LazyDfa(LazyDfaPool::new(LazyDfa::new(
                                (*nfa_arc).clone(),
                            ))),
                            capture_nfa,
                        )
                    }
                }
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
                    CompiledInner::LazyDfa(LazyDfaPool::new(LazyDfa::new(nfa))),
                    capture_nfa,
                )
            } else {
                let mut lazy = LazyDfa::new(nfa);
                // Already held by `lazy`; cloning the Arc (not the Nfa) so the
                // scan-budget give-up can rebuild a `LazyDfa` without recompiling.
                let nfa_arc = lazy.nfa_arc();
                match EagerDfa::from_lazy(&mut lazy) {
                    Ok(eager) => (CompiledInner::EagerDfa(eager, nfa_arc), capture_nfa),
                    Err(_) => {
                        // Materialization declined (see
                        // EagerMaterializationBudgetExceeded): fall back to a
                        // fresh LazyDfa, which computes the identical states
                        // on demand instead of upfront.
                        (
                            CompiledInner::LazyDfa(LazyDfaPool::new(LazyDfa::new(
                                (*nfa_arc).clone(),
                            ))),
                            capture_nfa,
                        )
                    }
                }
            }
        }
    };

    // Backreference patterns need single-pass backtracking capture extraction;
    // everything else goes through the PikeVM second pass.
    let backtracking_vm = if needs_backtracking {
        Some(BacktrackingVm::new(hir))
    } else {
        None
    };

    // The only place an `EagerDfa` is built, and so the only place the
    // specialized scan can apply: it needs the prefilter out of the way (no
    // candidates to consult, nothing to translate) and a DFA whose search is
    // the plain unanchored loop.
    let simple_eager_scan = prefilter.is_none()
        && literals.prefix_offset == 0
        && matches!(&inner, CompiledInner::EagerDfa(dfa, _) if dfa.is_simple_scan());

    Ok(CompiledRegex {
        inner,
        prefilter,
        prefix_offset: literals.prefix_offset,
        simple_eager_scan,
        // Not a tagged-NFA engine: the gate only accepts patterns with a
        // lookaround, which never reach this far.
        reverse_suffix: None,
        capture_nfa: RwLock::new(capture_nfa),
        one_pass: OnceLock::new(),
        capture_vm: RwLock::new(None),
        capture_ctx: RwLock::new(None),
        backtracking_vm,
        #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
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
    // Demote any full-match prefilter to candidate-only: the PikeVM is the
    // leftmost-first authority and must determine the match span itself. A
    // full-match literal prefilter resolves overlaps by length/order, not by
    // alternation branch priority (`ab|a` → it returns "a", PikeVM needs "ab"),
    // and all alternations are routed here.
    let prefilter = Prefilter::from_literals(&literals).into_candidate_only();
    let nfa = nfa::compile(hir)?;

    Ok(CompiledRegex {
        inner: CompiledInner::PikeVm(PikeVm::new(nfa)),
        prefilter,
        prefix_offset: literals.prefix_offset,
        simple_eager_scan: false,
        reverse_suffix: None,
        capture_nfa: RwLock::new(None),
        one_pass: OnceLock::new(),
        capture_vm: RwLock::new(None),
        capture_ctx: RwLock::new(None),
        backtracking_vm: None,
        #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
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
            prefix_offset: 0,
            simple_eager_scan: false,
            reverse_suffix: None,
            capture_nfa: RwLock::new(None),
            one_pass: OnceLock::new(),
            capture_vm: RwLock::new(None),
            capture_ctx: RwLock::new(None),
            backtracking_vm: None,
            #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
            backtracking_jit: None,
        });
    }

    // Non-greedy quantifiers → PikeVm, checked BEFORE the large-unicode-class and
    // lookaround routes (which would otherwise grab codepoint lazy patterns like
    // `\S+?`). The step-based TaggedNfa(JIT) does not represent non-greedy
    // faithfully (a bounded repeat followed by `+?` can let the lazy match zero);
    // PikeVm has the precise thread-priority semantics. Backref patterns are left
    // for the backtracking engine below.
    if hir.props.has_non_greedy && !hir.props.has_backrefs {
        // `compile_with_pikevm` demotes any full-match literal prefilter to
        // candidate-only so it cannot short-circuit the leftmost-first match
        // (e.g. `a+?| $`, a non-greedy branch in an alternation).
        return compile_with_pikevm(hir);
    }

    // A word boundary guarding an empty match (`\Ba*`, `\b(?:xy)?`) is not
    // representable in the DFA JIT — see `needs_boundary_aware_empty_match`.
    // Backreference patterns are left for the backtracking engine below, which
    // evaluates assertions positionally and so is unaffected.
    if needs_boundary_aware_empty_match(hir) && !hir.props.has_backrefs {
        return compile_with_pikevm(hir);
    }

    // A multiline anchor guarding an empty match (`\s*(?m)$`) has the same
    // problem in the DFA JIT: the empty match at a line end is only valid
    // because of the byte that follows, which the JIT has already committed to
    // by the time it decides to accept — it reports the final match and drops
    // the interior ones. The interpreted DFA resolves the anchor positionally
    // and gets this right, so hand the pattern back to the ordinary selection
    // rather than to the PikeVM.
    if hir.props.has_multiline_anchors
        && crate::hir::matches_empty(&hir.expr)
        && !hir.props.has_backrefs
    {
        return compile_from_hir(hir);
    }

    // 1. Complex Unicode patterns with large unicode classes → TaggedNfa JIT
    // These patterns use CodepointClass instructions which DFA cannot handle.
    // Route them to TaggedNfa JIT which supports CodepointClass.
    // (Backreference patterns are excluded — TaggedNfa can't handle backrefs;
    // they must reach the backtracking engine below even when they also contain
    // a large unicode class such as Unicode `\s`.)
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    if hir.props.has_large_unicode_class && !hir.props.has_backrefs {
        let literals = extract_literals(hir);
        let prefilter = Prefilter::from_literals(&literals);
        let nfa = nfa::compile(hir)?;
        match jit::compile_tagged_nfa(&nfa) {
            Ok(engine) => {
                return Ok(CompiledRegex {
                    inner: CompiledInner::TaggedNfaJit(engine),
                    prefilter,
                    prefix_offset: literals.prefix_offset,
                    simple_eager_scan: false,
                    reverse_suffix: ReverseSuffixSearch::new(hir),
                    capture_nfa: RwLock::new(None),
                    one_pass: OnceLock::new(),
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
                return Ok(compile_tagged_nfa_interp(hir, nfa));
            }
        }
    }

    // Non-JIT: Large unicode classes go to TaggedNfa interpreter (but not
    // backreference patterns — TaggedNfa can't handle backrefs).
    #[cfg(not(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    if hir.props.has_large_unicode_class && !hir.props.has_backrefs {
        return Ok(compile_tagged_nfa_interp(hir, nfa::compile(hir)?));
    }

    // 2. Patterns with backreferences → Backtracking JIT (only way to handle backrefs)
    // Backtracking JIT is required for backreferences since DFA cannot handle them.
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    if hir.props.has_backrefs && !hir.props.has_lookaround {
        let literals = extract_literals(hir);
        let prefilter = Prefilter::from_literals(&literals);
        match jit::compile_backtracking(hir) {
            Ok(jit_regex) => {
                return Ok(CompiledRegex {
                    inner: CompiledInner::Backtracking(jit_regex),
                    prefilter,
                    prefix_offset: literals.prefix_offset,
                    simple_eager_scan: false,
                    reverse_suffix: None,
                    capture_nfa: RwLock::new(None),
                    one_pass: OnceLock::new(),
                    capture_vm: RwLock::new(None),
                    capture_ctx: RwLock::new(None),
                    backtracking_vm: None,
                    #[cfg(all(
                        feature = "jit",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    ))]
                    backtracking_jit: None,
                });
            }
            Err(_) => {
                // Backtracking JIT failed (e.g. the pattern also contains a large
                // Unicode class like Unicode `\s`). Fall back to the BacktrackingVm
                // interpreter, which handles backreferences — PikeVM does not.
                return Ok(CompiledRegex {
                    inner: CompiledInner::BacktrackingVm(BacktrackingVm::new(hir)),
                    prefilter,
                    prefix_offset: literals.prefix_offset,
                    simple_eager_scan: false,
                    reverse_suffix: None,
                    capture_nfa: RwLock::new(None),
                    one_pass: OnceLock::new(),
                    capture_vm: RwLock::new(None),
                    capture_ctx: RwLock::new(None),
                    backtracking_vm: None,
                    backtracking_jit: None,
                });
            }
        }
    }

    // 2a. Patterns with lookaround → TaggedNfa JIT (handles lookahead via the
    // step model with memoized lookaround evaluation).
    //
    // NOTE: Patterns with captures but NO non-greedy/lookaround should use DFA JIT
    // because DFA JIT is much faster. DFA JIT handles captures via two-pass:
    // 1. Fast DFA JIT for find()
    // 2. PikeVM on matched substring for captures() only when needed
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    if hir.props.has_lookaround {
        let literals = extract_literals(hir);
        let prefilter = Prefilter::from_literals(&literals);
        let nfa = nfa::compile(hir)?;
        match jit::compile_tagged_nfa(&nfa) {
            Ok(engine) => {
                return Ok(CompiledRegex {
                    inner: CompiledInner::TaggedNfaJit(engine),
                    prefilter,
                    prefix_offset: literals.prefix_offset,
                    simple_eager_scan: false,
                    reverse_suffix: ReverseSuffixSearch::new(hir),
                    capture_nfa: RwLock::new(None),
                    one_pass: OnceLock::new(),
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
                return Ok(compile_tagged_nfa_interp(hir, nfa));
            }
        }
    }

    // Fall back to TaggedNfa interpreter when JIT feature is not available
    // Note: TaggedNfa interpreter is now always available (faster than PikeVm for lookaround)
    #[cfg(not(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    if hir.props.has_lookaround || hir.props.has_non_greedy {
        return Ok(compile_tagged_nfa_interp(hir, nfa::compile(hir)?));
    }

    // Backreferences without the JIT go to the same `BacktrackingVm` that
    // `compile_from_hir` picks, NOT the PikeVM.
    //
    // The PikeVM does handle backreferences, but only as a last resort: a
    // backreference breaks its per-position state deduplication (two threads in
    // one state are no longer interchangeable when their captures differ), so it
    // restarts the whole simulation at every start position — quadratic where
    // the backtracking engine is not. Selecting it here made `jit(true)` on a
    // build without the JIT feature *slower* than plain `Regex::new`, which
    // inverts what the flag means.
    #[cfg(not(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    if hir.props.has_backrefs {
        let literals = extract_literals(hir);
        return Ok(CompiledRegex {
            inner: CompiledInner::BacktrackingVm(BacktrackingVm::new(hir)),
            prefilter: Prefilter::from_literals(&literals),
            prefix_offset: literals.prefix_offset,
            simple_eager_scan: false,
            reverse_suffix: None,
            capture_nfa: RwLock::new(None),
            one_pass: OnceLock::new(),
            capture_vm: RwLock::new(None),
            capture_ctx: RwLock::new(None),
            backtracking_vm: None,
        });
    }

    // Alternations require leftmost-first branch priority. The DFA JIT and
    // JitShiftOr below resolve to the first/longest accepting state and so return
    // the wrong branch (`ab|a` → `a`, `\d+|\w+` → the longer `\w+`). Route any
    // alternation to the ordered PikeVM instead. (Patterns with lookaround,
    // backrefs, non-greedy or codepoint classes were already dispatched above.)
    if crate::engine::selector::hir_has_alternation(&hir.expr) {
        return compile_with_pikevm(hir);
    }

    // Repetition priority can end a match early in the same way (`a?(?:ab)?`
    // on "ab" is `a`), and an assertion between consuming parts (`a$a`) needs
    // positional evaluation. Every engine below reports the longest match and
    // checks assertions only at the edges.
    if !crate::engine::selector::automata_match_like_pikevm(hir) {
        return compile_with_pikevm(hir);
    }

    // An alternation — including the one a negated class lowers to — belongs on
    // the DFA rather than Shift-Or, whose step walks the live positions and so
    // costs more the more branches there are. `select_engine_from_hir` already
    // routes it there, and this path has to agree: without it `jit(true)` reached
    // JitShiftOr and ran the tokenizer's alternation ~20% SLOWER than
    // `Regex::new`, which is the one thing asking for the JIT must never do.
    if crate::engine::selector::hir_contains_alternation(&hir.expr) {
        return compile_from_hir(hir);
    }

    // One repeated byte class is answered by a scan in the interpreted ShiftOr,
    // and no engine on this path beats it: JitShiftOr compiles the bit-parallel
    // automaton the scan replaces, and the DFA JIT is slower still. `\w+` — the
    // tokenizer pattern — is ~2x slower under either than interpreted.
    if crate::vm::is_class_run_shape(hir) {
        return compile_from_hir(hir);
    }

    // A word boundary plus an effective prefilter is the DFA JIT's worst shape,
    // and the interpreted engines' best. The prefilter finds a literal, but the
    // boundary is exactly what makes those candidates fail — "the" inside
    // "their" — and the DFA JIT has no anchored entry point, so a failed
    // candidate turns into a scan of everything after it rather than a rejection.
    // The engines `compile_from_hir` picks verify one position and move on, which
    // is why `\bthe\b` runs ~2.5x faster there. Selecting a worse engine than
    // `Regex::new` is the one thing `jit(true)` must never do.
    if hir.props.has_word_boundary && !hir.props.has_backrefs {
        let literals = extract_literals(hir);
        if Prefilter::from_literals(&literals).is_effective() {
            return compile_from_hir(hir);
        }
    }

    // 3. Small patterns without effective prefilter → JitShiftOr
    // ShiftOr's bit-parallel algorithm is faster than DFA JIT for patterns with
    // many alternations and no common prefix (no effective prefilter).
    // DFA JIT excels when there's a good prefilter to skip non-matching positions.
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
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
            // A pattern the interpreter answers with a class-run scan must not
            // be JIT-compiled: the generated code runs the bit-parallel
            // automaton, which is exactly what the scan replaces. `\w+` is 2x
            // slower JIT-compiled than interpreted.
            if let Some(shift_or) = shift_or {
                if let Some(jit_shift_or) = jit::JitShiftOr::compile(&shift_or) {
                    let capture_nfa = if hir.props.capture_count > 0 {
                        nfa::compile(hir).ok()
                    } else {
                        None
                    };

                    // Only backreferences justify backtracking for captures; see
                    // `compile_from_hir`. Everything else takes the PikeVM
                    // second pass, which is linear in the input.
                    let needs_backtracking = hir.props.has_backrefs;
                    let backtracking_vm = if needs_backtracking {
                        Some(BacktrackingVm::new(hir))
                    } else {
                        None
                    };
                    let backtracking_jit = if needs_backtracking {
                        jit::compile_backtracking(hir).ok()
                    } else {
                        None
                    };

                    return Ok(CompiledRegex {
                        inner: CompiledInner::JitShiftOr(jit_shift_or),
                        prefilter,
                        prefix_offset: literals.prefix_offset,
                        simple_eager_scan: false,
                        reverse_suffix: None,
                        capture_nfa: RwLock::new(capture_nfa),
                        one_pass: OnceLock::new(),
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
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let literals = extract_literals(hir);
        let prefilter = Prefilter::from_literals(&literals);
        let nfa = nfa::compile(hir)?;
        let capture_nfa = Some(nfa.clone());
        let mut dfa = LazyDfa::new(nfa);

        // Only backreferences justify backtracking for captures; see
        // `compile_from_hir`.
        let backtracking_vm = if hir.props.has_backrefs {
            Some(BacktrackingVm::new(hir))
        } else {
            None
        };

        match jit::compile_dfa(&mut dfa) {
            Ok(jit_regex) => {
                return Ok(CompiledRegex {
                    inner: CompiledInner::Jit(jit_regex),
                    prefilter,
                    prefix_offset: literals.prefix_offset,
                    simple_eager_scan: false,
                    reverse_suffix: None,
                    capture_nfa: RwLock::new(capture_nfa),
                    one_pass: OnceLock::new(),
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

    /// Patterns the reverse-suffix gate accepts, one per accepted shape: a `+`
    /// run, a `*` run (an empty run is still a match), an explicit class, and a
    /// codepoint run walked one character at a time.
    const END_FIRST_PATTERNS: &[&str] = &[
        r"\w+(?=ing\b)",
        r"\w*(?=ing)",
        r"[a-z]+(?=xy)",
        r"(?u:\w+(?=ing))",
    ];

    /// Haystacks covering every branch of the end-first loop: no literal at all,
    /// a literal with no run before it (skipped, or matched empty), a literal
    /// inside a run whose start is interior, several runs in sequence, a failed
    /// run that must be skipped whole, and multi-byte characters.
    const END_FIRST_INPUTS: &[&str] = &[
        "",
        "a",
        "  ",
        "ing",
        " ing",
        "ings",
        "inging",
        "sing ing",
        "singing",
        "singing ringing",
        "ing ing ing",
        "xy",
        " xyxy",
        "abxyxy",
        "xyzing",
        "naïveing",
        "ï ing",
        "中ing 中",
        "中inging",
    ];

    fn compiled(pattern: &str, jit: bool) -> CompiledRegex {
        let hir = translate(&parse(pattern).unwrap()).unwrap();
        if jit {
            compile_with_jit(&hir).unwrap()
        } else {
            compile_from_hir(&hir).unwrap()
        }
    }

    /// The gate has to be consulted at every site that builds a tagged-NFA
    /// engine, and refused everywhere else. Without this the feature can be
    /// entirely dead — a declined or unwired gate leaves every behavioural test
    /// passing on the forward scan.
    #[test]
    fn end_first_search_is_wired_at_every_tagged_site() {
        for pattern in END_FIRST_PATTERNS {
            for jit in [false, true] {
                assert!(
                    compiled(pattern, jit).reverse_suffix.is_some(),
                    "gate declined {pattern:?} (jit={jit})"
                );
            }
        }

        // Refused shapes reach the same engines and must keep the forward scan.
        for pattern in [
            r"\w+(?=x)",
            r"\w+(?!ing)",
            r"\w+?(?=ing)",
            r"\w+(?=ing|ed)",
            r"(?:abcde|c)(?=d)",
            r"\w+ing",
            r"hello",
        ] {
            for jit in [false, true] {
                assert!(
                    compiled(pattern, jit).reverse_suffix.is_none(),
                    "gate accepted {pattern:?} (jit={jit})"
                );
            }
        }
    }

    /// The end-first search is an optimization, so it must answer exactly what
    /// the forward scan answers — for every resume offset, which is what
    /// `find_iter` walks through and what the reverse walk's floor clamps to.
    ///
    /// The baseline is the *same* compiled regex with the search taken away, so
    /// the only difference between the two sides is the path under test.
    #[test]
    fn end_first_search_agrees_with_the_forward_scan() {
        for pattern in END_FIRST_PATTERNS {
            for jit in [false, true] {
                let fast = compiled(pattern, jit);
                let mut forward = compiled(pattern, jit);
                assert!(fast.reverse_suffix.is_some(), "{pattern:?}");
                forward.reverse_suffix = None;

                for input in END_FIRST_INPUTS {
                    let bytes = input.as_bytes();
                    for from in 0..=bytes.len() {
                        assert_eq!(
                            fast.find_from(bytes, from),
                            forward.find_from(bytes, from),
                            "pattern={pattern:?} jit={jit} input={input:?} from={from}"
                        );
                    }
                }
            }
        }
    }

    /// The other two entry points are routed through the same search, and each
    /// used to reach the engine by a path of its own — a candidate loop in
    /// `is_match`, a forward scan from the resume point in `captures_from`. Both
    /// must now answer exactly what `find_from` answers, at every resume offset:
    /// an entry point that keeps its own scan is not visibly broken, it just
    /// disagrees with `find` on the inputs where the two searches differ.
    #[test]
    fn every_entry_point_agrees_with_the_end_first_search() {
        for pattern in END_FIRST_PATTERNS {
            for jit in [false, true] {
                let fast = compiled(pattern, jit);
                assert!(fast.reverse_suffix.is_some(), "{pattern:?}");

                for input in END_FIRST_INPUTS {
                    let bytes = input.as_bytes();
                    let context = format!("pattern={pattern:?} jit={jit} input={input:?}");
                    assert_eq!(
                        fast.is_match(bytes),
                        fast.find(bytes).is_some(),
                        "is_match disagrees: {context}"
                    );
                    for from in 0..=bytes.len() {
                        let found = fast.find_from(bytes, from);
                        assert_eq!(
                            fast.captures_from(bytes, from)
                                .map(|slots| slots.first().copied().flatten()),
                            found.map(Some),
                            "captures_from disagrees: {context} from={from}"
                        );
                    }
                }
            }
        }
    }

    /// The counterexample the whole design rests on: the match's end is not the
    /// literal occurrence the search found, and its start is not one the forward
    /// scan would have reached first by accident.
    #[test]
    fn end_first_search_finds_the_leftmost_match() {
        for jit in [false, true] {
            let re = compiled(r"\w+(?=ing\b)", jit);
            assert_eq!(re.find(b"singing"), Some((0, 4)), "jit={jit}");
            assert_eq!(re.find_from(b"singing", 4), None, "jit={jit}");
            assert_eq!(re.find(b"ings"), None, "jit={jit}");
            // A start interior to a run, reached only because a later
            // occurrence's run walked back into it.
            assert_eq!(compiled(r"[a-z]+(?=xy)", jit).find(b" xyxy"), Some((1, 3)));
        }
    }

    #[test]
    fn test_is_match() {
        let re = make_regex("hello");
        assert!(re.is_match(b"hello world"));
        assert!(!re.is_match(b"goodbye"));
    }

    /// `one_pass` is built lazily, and a failure to build it is *invisible*
    /// through the public API: `captures_two_pass` just falls back to the PikeVM
    /// and returns the same answers. So this has to assert on `one_pass()`
    /// directly — a test that only checked capture output would pass whether or
    /// not the one-pass engine was ever constructed.
    #[test]
    fn one_pass_is_built_lazily_and_memoized() {
        let re = make_regex(r"(\d+)-(\d+)");

        let first = re.one_pass().expect("pattern is one-pass eligible");
        let second = re.one_pass().expect("second call must still resolve");
        assert!(
            std::ptr::eq(first, second),
            "one_pass must be memoized, not recompiled per call"
        );

        // And the engine it built must actually produce the right captures.
        let caps = re.captures(b"phone: 123-456").expect("must match");
        assert_eq!(caps[0], Some((7, 14)));
        assert_eq!(caps[1], Some((7, 10)));
        assert_eq!(caps[2], Some((11, 14)));
    }

    /// The other half of the contract. Most construction sites carry no capture
    /// NFA and used to hardcode `one_pass: None`; now they resolve lazily, so the
    /// equivalence "no capture NFA implies no one-pass engine" has to be pinned
    /// or a future change could start building one where none existed before.
    #[test]
    fn one_pass_is_none_without_a_capture_nfa() {
        let patterns = [
            "hello",
            r"(\d+)-(\d+)",
            r"(?=foo)bar",
            r"(\w)\1",
            r"\b\w+\b",
            r"a|b|c",
        ];

        let mut saw_without_capture_nfa = false;
        for pattern in patterns {
            let re = make_regex(pattern);
            if re.capture_nfa.read().unwrap().is_none() {
                saw_without_capture_nfa = true;
                assert!(
                    re.one_pass().is_none(),
                    "{pattern}: no capture NFA, so there is nothing to build a \
                     one-pass engine from"
                );
            }
        }

        assert!(
            saw_without_capture_nfa,
            "no pattern exercised the capture-NFA-less path, so this test proved \
             nothing — pick patterns that reach those construction sites"
        );
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

    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
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
