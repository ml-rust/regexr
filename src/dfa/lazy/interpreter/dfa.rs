//! Lazy DFA interpreter implementation.
//!
//! Builds DFA states on-demand using subset construction,
//! caching them for future use.
//!
//! ## Performance Optimizations
//!
//! This implementation uses several optimizations from the regex crate:
//!
//! 1. **Premultiplied State IDs**: State IDs are pre-multiplied by stride (256),
//!    so `transitions[state + byte]` is a simple addition instead of
//!    `transitions[state * 256 + byte]` which requires multiplication.
//!
//! 2. **Tagged State IDs**: High bits encode status (match/dead/unknown), allowing
//!    status checks without memory dereference.
//!
//! 3. **Dense Transition Table**: A flat array of all transitions for cache efficiency.
//!
//! ## Cache Eviction Strategy
//!
//! We use a **full flush** strategy instead of LRU. When the cache reaches
//! its limit, we clear all states and rebuild from the start state.

use std::collections::BTreeSet;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use crate::nfa::{Nfa, NfaInstruction, StateId as NfaStateId};

use super::super::shared::{
    epsilon_closure_subset, epsilon_closure_with_context, flush_cache,
    get_or_create_state_with_class, is_dead_state, is_tagged_match, is_unknown_state,
    match_reachable_without_end_assertion, scan_budget, state_index, tag_state, untag_state,
    CacheCeilingExceeded, CharClass, DfaStateId, LazyDfaContext, NfaSubset, PositionContext,
    DEAD_STATE, UNKNOWN_STATE,
};
use super::linear::ReverseDfa;

/// A lazy DFA that builds states on demand.
#[derive(Debug, Clone)]
pub struct LazyDfa {
    /// Internal context containing state and transition data.
    pub(super) ctx: LazyDfaContext,
    /// The reversed-pattern DFA for [`LazyDfa::find_from_linear`].
    pub(super) reverse: ReverseDfa,
}

/// Holds a [`LazyDfa`] for the duration of one search.
///
/// State IDs are premultiplied indices into the state and transition tables, so
/// a cache flush turns every ID a running search is holding into garbage. While
/// this guard lives, flushing is suppressed and the cache grows instead; the
/// deferred flush happens on the outermost drop, so the growth does not carry
/// into the next search. Being a `Drop` guard, it also restores the depth when a
/// search unwinds, which is what keeps a panic from suppressing flushes forever.
///
/// Searches may nest, hence a depth count rather than a flag.
pub(super) struct SearchGuard<'a> {
    dfa: &'a mut LazyDfa,
}

impl<'a> SearchGuard<'a> {
    pub(super) fn new(dfa: &'a mut LazyDfa) -> Self {
        dfa.ctx.search_depth += 1;
        Self { dfa }
    }

    /// Reports the search's result, unless it gave up on the cache ceiling.
    pub(super) fn finish<T>(self, outcome: T) -> Result<T, CacheCeilingExceeded> {
        if std::mem::take(&mut self.dfa.ctx.ceiling_exceeded) {
            Err(CacheCeilingExceeded)
        } else {
            Ok(outcome)
        }
    }
}

impl Drop for SearchGuard<'_> {
    fn drop(&mut self) {
        self.dfa.ctx.search_depth = self.dfa.ctx.search_depth.saturating_sub(1);
        if self.dfa.ctx.search_depth == 0 && self.dfa.ctx.states.len() >= self.dfa.ctx.cache_limit {
            flush_cache(&mut self.dfa.ctx);
        }
    }
}

impl Deref for SearchGuard<'_> {
    type Target = LazyDfa;

    fn deref(&self) -> &LazyDfa {
        &*self.dfa
    }
}

impl DerefMut for SearchGuard<'_> {
    fn deref_mut(&mut self) -> &mut LazyDfa {
        &mut *self.dfa
    }
}

impl LazyDfa {
    /// Creates a new lazy DFA from an NFA.
    pub fn new(nfa: Nfa) -> Self {
        Self {
            ctx: LazyDfaContext::new(nfa),
            reverse: ReverseDfa::Unbuilt,
        }
    }

    /// Sets the cache size limit.
    pub fn set_cache_limit(&mut self, limit: usize) {
        self.ctx.set_cache_limit(limit);
    }

    /// Returns the number of times the cache has been flushed.
    pub fn flush_count(&self) -> usize {
        self.ctx.flush_count()
    }

    /// Returns the start state.
    pub fn start(&self) -> DfaStateId {
        self.ctx.start()
    }

    /// Returns true if this DFA has word boundary assertions.
    pub fn has_word_boundary(&self) -> bool {
        self.ctx.has_word_boundary()
    }

    /// Returns true if this DFA has anchor assertions (^, $).
    pub fn has_anchors(&self) -> bool {
        self.ctx.has_anchors()
    }

    /// Returns true if this DFA has a start anchor (^).
    pub fn has_start_anchor(&self) -> bool {
        self.ctx.has_start_anchor()
    }

    /// Returns true if this DFA has an end anchor ($).
    pub fn has_end_anchor(&self) -> bool {
        self.ctx.has_end_anchor()
    }

    /// Returns true if any accepting state is reachable without an end assertion.
    pub fn has_clean_accept(&self) -> bool {
        self.ctx.has_clean_accept()
    }

    /// Returns true if this DFA has multiline anchors.
    pub fn has_multiline_anchors(&self) -> bool {
        self.ctx.has_multiline_anchors()
    }

    /// Returns true if the start anchor specifically is line-mode.
    pub fn has_multiline_start_anchor(&self) -> bool {
        self.ctx.has_multiline_start_anchor()
    }

    /// Gets the transition for a state and byte, computing it if needed.
    #[inline]
    pub fn transition(&mut self, state: DfaStateId, byte: u8) -> Option<DfaStateId> {
        let idx = (state + byte as u32) as usize;
        if idx < self.ctx.transitions.len() {
            let tagged = unsafe { *self.ctx.transitions.get_unchecked(idx) };
            if !is_unknown_state(tagged) {
                if is_dead_state(tagged) {
                    return None;
                }
                return Some(untag_state(tagged));
            }
        }

        self.compute_transition(state, byte)
    }

    /// Fast transition lookup returning tagged state ID.
    #[inline(always)]
    pub fn transition_tagged(&self, state: DfaStateId, byte: u8) -> u32 {
        let idx = (state + byte as u32) as usize;
        if idx < self.ctx.transitions.len() {
            unsafe { *self.ctx.transitions.get_unchecked(idx) }
        } else {
            UNKNOWN_STATE
        }
    }

    /// Fast transition lookup for cached states only (immutable).
    #[inline(always)]
    pub fn transition_cached(&self, state: DfaStateId, byte: u8) -> Option<DfaStateId> {
        let idx = (state + byte as u32) as usize;
        if idx < self.ctx.transitions.len() {
            let tagged = unsafe { *self.ctx.transitions.get_unchecked(idx) };
            if is_unknown_state(tagged) || is_dead_state(tagged) {
                None
            } else {
                Some(untag_state(tagged))
            }
        } else {
            None
        }
    }

    /// Computes a transition, handling word boundaries and anchors.
    fn compute_transition(&mut self, state: DfaStateId, byte: u8) -> Option<DfaStateId> {
        #[cfg(test)]
        {
            self.ctx.per_byte_computations += 1;
        }

        let state_idx = state_index(state);
        let dfa_state = self.ctx.states.get(state_idx)?;
        // Refcount bump, and it releases the borrow of `self.ctx.states` so the
        // closure walk below can borrow the scratch mutably.
        let nfa_states = Arc::clone(&dfa_state.nfa_states);
        let prev_class = dfa_state.prev_class;

        let curr_class = CharClass::from_byte(byte);

        let is_at_boundary = if self.ctx.has_word_boundary {
            Some(prev_class != curr_class)
        } else {
            None
        };

        let pos_ctx = if self.ctx.has_anchors {
            Some(PositionContext::middle())
        } else {
            None
        };

        let expanded_nfa_states: NfaSubset = if self.ctx.has_word_boundary || self.ctx.has_anchors {
            epsilon_closure_with_context(
                &self.ctx.nfa,
                &mut self.ctx.scratch,
                &nfa_states[..],
                is_at_boundary,
                pos_ctx,
            )
        } else {
            nfa_states
        };

        let mut next_states = BTreeSet::new();

        for &nfa_state in expanded_nfa_states.iter() {
            if let Some(nfa_s) = self.ctx.nfa.get(nfa_state) {
                for (range, target) in &nfa_s.transitions {
                    if range.contains(byte) {
                        next_states.insert(*target);
                    }
                }
            }
        }

        let cache_idx = (state + byte as u32) as usize;

        if next_states.is_empty() {
            if cache_idx < self.ctx.transitions.len() {
                self.ctx.transitions[cache_idx] = DEAD_STATE;
            }
            return None;
        }

        let target_pos_ctx = if self.ctx.has_anchors {
            if self.ctx.has_multiline_anchors && byte == b'\n' {
                Some(PositionContext::after_newline())
            } else {
                None
            }
        } else {
            None
        };

        let next_closure = if self.ctx.has_word_boundary || self.ctx.has_anchors {
            epsilon_closure_with_context(
                &self.ctx.nfa,
                &mut self.ctx.scratch,
                &next_states,
                None,
                target_pos_ctx,
            )
        } else {
            epsilon_closure_subset(&self.ctx.nfa, &mut self.ctx.scratch, &next_states)
        };

        if next_closure.is_empty() {
            if cache_idx < self.ctx.transitions.len() {
                self.ctx.transitions[cache_idx] = DEAD_STATE;
            }
            return None;
        }

        // Safe to reuse the scratch: `next_closure` is already an owned subset,
        // independent of the buffer it was built in.
        let next_clean = match_reachable_without_end_assertion(
            &self.ctx.nfa,
            &mut self.ctx.scratch,
            &next_states,
            None,
            target_pos_ctx,
        );
        let flushes_before = self.ctx.flush_count;
        let next_id =
            get_or_create_state_with_class(&mut self.ctx, next_closure, curr_class, next_clean);

        let next_idx = state_index(next_id);
        let is_match = self.ctx.states.get(next_idx).is_some_and(|s| s.is_match);

        // `state` is a premultiplied index into the table. Creating the target
        // may have flushed the cache, which renumbers every state, so the row
        // this index names is no longer this state's row — writing there would
        // cache the transition against an unrelated state.
        if self.ctx.flush_count == flushes_before {
            let cache_idx = (state + byte as u32) as usize;
            if cache_idx < self.ctx.transitions.len() {
                self.ctx.transitions[cache_idx] = tag_state(next_id, is_match);
            }
        }

        Some(next_id)
    }

    /// Returns true if the state is a match state.
    pub fn is_match(&self, state: DfaStateId) -> bool {
        let idx = state_index(state);
        self.ctx
            .states
            .get(idx)
            .map(|s| s.is_match)
            .unwrap_or(false)
    }

    /// Returns the size of the state's NFA subset (its live-thread count).
    ///
    /// Used by [`EagerDfa::from_lazy`](crate::dfa::eager::EagerDfa::from_lazy)
    /// as a cheap proxy for the cost of computing this state's transitions:
    /// that cost is dominated by unioning the epsilon closures reachable from
    /// each member of the subset, so it scales with the subset's size.
    pub fn get_state_subset_size(&self, state: DfaStateId) -> usize {
        let idx = state_index(state);
        self.ctx
            .states
            .get(idx)
            .map(|s| s.nfa_states.len())
            .unwrap_or(0)
    }

    /// Returns the prev_class of a state (for JIT compilation).
    pub fn get_state_prev_class(&self, state: DfaStateId) -> CharClass {
        let idx = state_index(state);
        self.ctx
            .states
            .get(idx)
            .map(|s| s.prev_class)
            .unwrap_or(CharClass::NonWord)
    }

    /// Gets or creates a start state with a specific previous character class.
    pub fn get_start_state_for_class(&mut self, prev_class: CharClass) -> DfaStateId {
        self.get_start_state_with_prev_class(prev_class)
    }

    /// Runs `f` with cache flushes suppressed, as a search would.
    ///
    /// A flush renumbers every state, so any caller that holds premultiplied
    /// IDs across several calls — the JIT's materialization BFS, which carries a
    /// queue and a visited set — needs one consistent numbering for the whole
    /// walk, not just within a single call. This is the same guard the search
    /// entry points use, so the cache grows instead of flushing and the deferred
    /// flush happens once `f` returns.
    ///
    /// `Err` means the cache hit its ceiling inside `f`: states are missing, so
    /// whatever `f` computed is incomplete rather than merely negative.
    ///
    /// Only the JIT materializes a DFA this way, so without that feature nothing
    /// calls this.
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    pub(crate) fn with_flushes_suppressed<T>(
        &mut self,
        f: impl FnOnce(&mut LazyDfa) -> T,
    ) -> Result<T, CacheCeilingExceeded> {
        let mut guard = SearchGuard::new(self);
        let outcome = f(&mut guard);
        guard.finish(outcome)
    }

    /// Computes all 256 transitions for a state at once.
    pub fn compute_all_transitions(&mut self, state: DfaStateId) -> [Option<DfaStateId>; 256] {
        let mut result = [None; 256];

        let base_idx = state as usize;
        if base_idx + 255 < self.ctx.transitions.len() {
            let mut all_cached = true;
            for byte in 0..=255u8 {
                let tagged = self.ctx.transitions[base_idx + byte as usize];
                if is_unknown_state(tagged) {
                    all_cached = false;
                } else if !is_dead_state(tagged) {
                    result[byte as usize] = Some(untag_state(tagged));
                }
            }
            if all_cached {
                return result;
            }
        }

        if !self.ctx.has_word_boundary && !self.ctx.has_anchors {
            self.compute_all_transitions_simple(state, &mut result);
        } else {
            self.compute_all_transitions_with_context(state, &mut result);
        }

        result
    }

    /// Computes all transitions for patterns without word boundaries or anchors.
    fn compute_all_transitions_simple(
        &mut self,
        state: DfaStateId,
        result: &mut [Option<DfaStateId>; 256],
    ) {
        let state_idx = state_index(state);
        let nfa_states = match self.ctx.states.get(state_idx) {
            Some(s) => Arc::clone(&s.nfa_states),
            None => return,
        };

        let mut byte_targets: [Option<Vec<NfaStateId>>; 256] = std::array::from_fn(|_| None);

        for &nfa_state in nfa_states.iter() {
            if let Some(nfa_s) = self.ctx.nfa.get(nfa_state) {
                for (range, target) in &nfa_s.transitions {
                    for byte in range.start..=range.end {
                        byte_targets[byte as usize]
                            .get_or_insert_with(Vec::new)
                            .push(*target);
                    }
                }
            }
        }

        // Deliberately NOT canonicalised. See the note in
        // `compute_all_transitions_with_context`.

        for byte in 0..=255u8 {
            let cache_idx = (state + byte as u32) as usize;

            if cache_idx < self.ctx.transitions.len() {
                let tagged = self.ctx.transitions[cache_idx];
                if !is_unknown_state(tagged) {
                    if !is_dead_state(tagged) {
                        result[byte as usize] = Some(untag_state(tagged));
                    }
                    continue;
                }
            }

            if let Some(ref targets) = byte_targets[byte as usize] {
                if targets.is_empty() {
                    if cache_idx < self.ctx.transitions.len() {
                        self.ctx.transitions[cache_idx] = DEAD_STATE;
                    }
                    continue;
                }

                let next_closure =
                    epsilon_closure_subset(&self.ctx.nfa, &mut self.ctx.scratch, targets);
                if next_closure.is_empty() {
                    if cache_idx < self.ctx.transitions.len() {
                        self.ctx.transitions[cache_idx] = DEAD_STATE;
                    }
                    continue;
                }

                let next_clean = match_reachable_without_end_assertion(
                    &self.ctx.nfa,
                    &mut self.ctx.scratch,
                    targets,
                    None,
                    None,
                );
                let flushes_before = self.ctx.flush_count;
                let next_id = get_or_create_state_with_class(
                    &mut self.ctx,
                    next_closure,
                    CharClass::NonWord,
                    next_clean,
                );
                result[byte as usize] = Some(next_id);

                let next_idx = state_index(next_id);
                let is_match = self.ctx.states.get(next_idx).is_some_and(|s| s.is_match);
                // `state` is fixed across this 256-byte loop, but creating a
                // target may flush and renumber the cache underneath it, so the
                // row this index names is only this state's row while no flush
                // has happened.
                if self.ctx.flush_count == flushes_before {
                    let cache_idx = (state + byte as u32) as usize;
                    if cache_idx < self.ctx.transitions.len() {
                        self.ctx.transitions[cache_idx] = tag_state(next_id, is_match);
                    }
                }
            } else if cache_idx < self.ctx.transitions.len() {
                self.ctx.transitions[cache_idx] = DEAD_STATE;
            }
        }
    }

    /// Computes all transitions for patterns with word boundaries or anchors.
    ///
    /// Computing these one byte at a time repeats almost all of the work 256
    /// times: [`LazyDfa::compute_transition`] re-expands this state's subset,
    /// walks the target closure and re-checks match reachability per byte, even
    /// though only two things about the byte are visible to any of it. The first
    /// expansion sees the byte solely through `prev_class != CharClass::from_byte(byte)`,
    /// a two-valued question; the target's position context sees it solely
    /// through `byte == b'\n'`. So the expansion has at most two distinct
    /// answers over the whole alphabet, and the rest is driven by the target
    /// subset, which is constant across runs of bytes that lead to the same
    /// NFA states.
    ///
    /// This computes each distinct answer once and then sweeps the alphabet in
    /// order, doing the closure/reachability/interning work once per run of
    /// adjacent bytes that agree on all three of character class, target
    /// position context, and target subset. The transitions it writes are the
    /// same ones the per-byte path writes, including the state IDs: the sweep
    /// runs in byte order and never merges across a class change, so states are
    /// interned in the same order they would have been.
    fn compute_all_transitions_with_context(
        &mut self,
        state: DfaStateId,
        result: &mut [Option<DfaStateId>; 256],
    ) {
        let state_idx = state_index(state);
        let Some(prev_class) = self.ctx.states.get(state_idx).map(|s| s.prev_class) else {
            // No such state: fall back, so a partially populated row is still
            // reported exactly as the per-byte path reports it.
            self.compute_transitions_per_byte(state, result, 0);
            return;
        };

        // Which target set each byte leads to, taken from the expansion that
        // byte's own character class selects.
        let mut byte_targets: [Option<Vec<NfaStateId>>; 256] = std::array::from_fn(|_| None);
        {
            // Refcount bump rather than a borrow of `self.ctx.states`, so the
            // walks below can take `self.ctx.scratch` mutably.
            let seeds = match self.ctx.states.get(state_idx) {
                Some(s) => Arc::clone(&s.nfa_states),
                None => return,
            };

            let pos_ctx = if self.ctx.has_anchors {
                Some(PositionContext::middle())
            } else {
                None
            };
            // Read out by value: a closure capturing `self.ctx` would collide
            // with the mutable borrow of the scratch.
            let has_word_boundary = self.ctx.has_word_boundary;
            let boundary = |curr_class: CharClass| {
                if has_word_boundary {
                    Some(prev_class != curr_class)
                } else {
                    None
                }
            };

            let expanded_non_word = epsilon_closure_with_context(
                &self.ctx.nfa,
                &mut self.ctx.scratch,
                &seeds[..],
                boundary(CharClass::NonWord),
                pos_ctx,
            );
            // Without `\b`/`\B` the boundary answer is `None` either way, so the
            // two classes share one expansion.
            let expanded_word = if has_word_boundary {
                Some(epsilon_closure_with_context(
                    &self.ctx.nfa,
                    &mut self.ctx.scratch,
                    &seeds[..],
                    boundary(CharClass::Word),
                    pos_ctx,
                ))
            } else {
                None
            };
            let expanded_word_ref = expanded_word.as_ref().unwrap_or(&expanded_non_word);

            for (class, expanded) in [
                (CharClass::NonWord, &expanded_non_word),
                (CharClass::Word, expanded_word_ref),
            ] {
                for &nfa_state in expanded.iter() {
                    if let Some(nfa_s) = self.ctx.nfa.get(nfa_state) {
                        for (range, target) in &nfa_s.transitions {
                            for byte in range.start..=range.end {
                                if CharClass::from_byte(byte) == class {
                                    byte_targets[byte as usize]
                                        .get_or_insert_with(Vec::new)
                                        .push(*target);
                                }
                            }
                        }
                    }
                }
            }
        }

        // `byte_targets` is deliberately left in accumulation order, NOT sorted
        // and deduped. Canonicalising was tried and measured a net loss: it cost
        // 3.9% on a 26-keyword alternation compile and 3.1% on `\b\w+\b`, for no
        // measurable gain — the accumulation walks a sorted subset and appends
        // each state's targets, so in practice the rows come out sorted and
        // duplicate-free already, and the sort merely re-walks all 256 slots of
        // every state to confirm it.
        //
        // It is not needed for correctness either. Un-canonical rows can only
        // make two logically equal target sets compare *unequal* at the
        // `next == targets` check below, which splits a run into smaller ones
        // and recomputes them — never a wrong merge, since merging requires
        // equality. Duplicates are absorbed by the visited set inside
        // `epsilon_closure_with_context`, and the subset that gets interned is
        // that closure's output rather than this row, so state identity does not
        // depend on this ordering at all.

        // Captured by value: the sweep below mutates the transition table, so a
        // closure holding a borrow of `self` could not live across it.
        let newline_starts_a_line = self.ctx.has_anchors && self.ctx.has_multiline_anchors;
        let target_pos_ctx = move |byte: u8| {
            if newline_starts_a_line && byte == b'\n' {
                Some(PositionContext::after_newline())
            } else {
                None
            }
        };

        let mut byte = 0usize;
        while byte < 256 {
            let cache_idx = (state + byte as u32) as usize;
            if cache_idx < self.ctx.transitions.len() {
                let tagged = self.ctx.transitions[cache_idx];
                if !is_unknown_state(tagged) {
                    if !is_dead_state(tagged) {
                        result[byte] = Some(untag_state(tagged));
                    }
                    byte += 1;
                    continue;
                }
            }

            let targets = match &byte_targets[byte] {
                Some(targets) if !targets.is_empty() => targets,
                _ => {
                    if cache_idx < self.ctx.transitions.len() {
                        self.ctx.transitions[cache_idx] = DEAD_STATE;
                    }
                    byte += 1;
                    continue;
                }
            };

            let curr_class = CharClass::from_byte(byte as u8);
            let pos_ctx = target_pos_ctx(byte as u8);

            // Extend the run over adjacent bytes that would compute the same
            // thing. A class change ends it unconditionally: the class is part
            // of the interned state's key, so bytes of different classes are
            // different transitions even when their target subsets coincide.
            let mut end = byte + 1;
            while end < 256 {
                let idx = (state + end as u32) as usize;
                if idx < self.ctx.transitions.len() && !is_unknown_state(self.ctx.transitions[idx])
                {
                    break;
                }
                if CharClass::from_byte(end as u8) != curr_class {
                    break;
                }
                if target_pos_ctx(end as u8) != pos_ctx {
                    break;
                }
                match &byte_targets[end] {
                    Some(next) if next == targets => {}
                    _ => break,
                }
                end += 1;
            }

            #[cfg(test)]
            {
                self.ctx.context_run_computations += 1;
            }
            let next_closure = epsilon_closure_with_context(
                &self.ctx.nfa,
                &mut self.ctx.scratch,
                targets,
                None,
                pos_ctx,
            );
            if next_closure.is_empty() {
                for dead_byte in byte..end {
                    let idx = (state + dead_byte as u32) as usize;
                    if idx < self.ctx.transitions.len() {
                        self.ctx.transitions[idx] = DEAD_STATE;
                    }
                }
                byte = end;
                continue;
            }

            let next_clean = match_reachable_without_end_assertion(
                &self.ctx.nfa,
                &mut self.ctx.scratch,
                targets,
                None,
                pos_ctx,
            );
            let flushes_before = self.ctx.flush_count;
            let next_id =
                get_or_create_state_with_class(&mut self.ctx, next_closure, curr_class, next_clean);
            result[byte] = Some(next_id);

            // `state` is a premultiplied index into the table. Creating the
            // target may have flushed the cache, which renumbers every state, so
            // the row this index names is no longer this state's row — writing
            // there would cache the transition against an unrelated state. The
            // flush also invalidates the run: the per-byte path would rebuild
            // the interned state for each remaining byte, so hand them back to
            // it rather than reusing an ID from before the flush.
            if self.ctx.flush_count != flushes_before {
                self.compute_transitions_per_byte(state, result, byte + 1);
                return;
            }

            let next_idx = state_index(next_id);
            let is_match = self.ctx.states.get(next_idx).is_some_and(|s| s.is_match);
            for (run_byte, slot) in result.iter_mut().enumerate().take(end).skip(byte) {
                *slot = Some(next_id);
                let idx = (state + run_byte as u32) as usize;
                if idx < self.ctx.transitions.len() {
                    self.ctx.transitions[idx] = tag_state(next_id, is_match);
                }
            }

            byte = end;
        }
    }

    /// Fills `result[from..]` by computing one transition per byte.
    ///
    /// The unbatched path, kept for the cases
    /// [`LazyDfa::compute_all_transitions_with_context`] cannot batch.
    fn compute_transitions_per_byte(
        &mut self,
        state: DfaStateId,
        result: &mut [Option<DfaStateId>; 256],
        from: usize,
    ) {
        for (byte, slot) in result.iter_mut().enumerate().skip(from) {
            let cache_idx = (state + byte as u32) as usize;
            if cache_idx < self.ctx.transitions.len() {
                let tagged = self.ctx.transitions[cache_idx];
                if !is_unknown_state(tagged) {
                    if !is_dead_state(tagged) {
                        *slot = Some(untag_state(tagged));
                    }
                    continue;
                }
            }
            if let Some(target) = self.transition(state, byte as u8) {
                *slot = Some(target);
            }
        }
    }

    /// Returns the number of cached states.
    pub fn state_count(&self) -> usize {
        self.ctx.state_count()
    }

    /// Clears the DFA cache (except start state).
    pub fn clear_cache(&mut self) {
        flush_cache(&mut self.ctx);
    }

    /// Executes the DFA on input, returning true if it matches the whole input.
    ///
    /// `Err` means the state cache could not grow far enough to finish the scan;
    /// the answer is unknown, not `false`.
    pub fn is_match_bytes(&mut self, input: &[u8]) -> Result<bool, CacheCeilingExceeded> {
        let mut guard = SearchGuard::new(self);
        let outcome = guard.is_match_bytes_inner(input);
        guard.finish(outcome)
    }

    /// [`LazyDfa::is_match_bytes`] without the search guard.
    fn is_match_bytes_inner(&mut self, input: &[u8]) -> bool {
        let mut state = self.ctx.start;

        for &byte in input {
            match self.transition(state, byte) {
                Some(next) => state = next,
                None => return false,
            }
        }

        self.is_match(state)
    }

    /// Finds the first match in the input.
    ///
    /// `Err` means the search gave up on the cache ceiling; see
    /// [`LazyDfa::find_from`].
    pub fn find(&mut self, input: &[u8]) -> Result<Option<(usize, usize)>, CacheCeilingExceeded> {
        self.find_from(input, 0)
    }

    /// Finds the leftmost match starting at or after `from`.
    ///
    /// Every attempt gets the whole input (see [`LazyDfa::find_at`]), so the
    /// start state carries the real preceding byte for `^` and `\b`.
    ///
    /// `Err` means the search needed more DFA states than the cache is allowed
    /// to hold, so the transition table is incomplete and the scan's result
    /// would be a false negative. Callers re-run such a search on an engine that
    /// does not cache states — the PikeVM — rather than reporting "no match".
    pub fn find_from(
        &mut self,
        input: &[u8],
        from: usize,
    ) -> Result<Option<(usize, usize)>, CacheCeilingExceeded> {
        let mut guard = SearchGuard::new(self);
        let outcome = guard.find_from_inner(input, from);
        guard.finish(outcome)
    }

    /// [`LazyDfa::find_from`] without the search guard.
    fn find_from_inner(&mut self, input: &[u8], from: usize) -> Option<(usize, usize)> {
        if from > input.len() {
            return None;
        }

        // If pattern has both start and end anchors, they may be on different branches
        // of an alternation (e.g., ^X|Y$), so we need to do an unanchored search.
        // Only optimize with line-boundary-only search if we have ONLY a start anchor.
        let start_only = self.ctx.has_start_anchor && !self.ctx.has_end_anchor;

        // Trying one start at a time is right while the attempts are cheap:
        // most give up within a byte or two of where they began, and the first
        // one that matches ends the search. It collapses when they are not — a
        // pattern that consumes a long run before failing costs a full scan per
        // start, and the search turns quadratic.
        //
        // So the attempts are metered. Once they have collectively walked
        // several times the input, the pattern is one of those, and the
        // linear-time search takes over from the next start.
        let budget = scan_budget(input.len(), from, self.ctx.nfa.max_match_len);
        let mut walked = 0usize;

        if start_only {
            if self.ctx.has_multiline_anchors {
                if from == 0 {
                    if let Some(end) = self.find_at_inner(input, 0) {
                        return Some((0, end));
                    }
                }
                // Line starts at or after `from`: the newline itself may sit just
                // before it, hence `from - 1`.
                for (i, &byte) in input.iter().enumerate().skip(from.saturating_sub(1)) {
                    if byte == b'\n' {
                        let line_start = i + 1;
                        if self.ctx.ceiling_exceeded {
                            return None;
                        }
                        self.ctx.last_reach = line_start;
                        if let Some(end) = self.find_at_inner(input, line_start) {
                            return Some((line_start, end));
                        }
                        walked += self.ctx.last_reach.saturating_sub(line_start);
                        if walked > budget {
                            return self.find_from_linear_inner(input, line_start + 1);
                        }
                    }
                }
                None
            } else if from == 0 {
                self.find_at_inner(input, 0).map(|end| (0, end))
            } else {
                None
            }
        } else {
            for start_pos in from..=input.len() {
                // The cache ran out under an earlier start; the remaining ones
                // would scan an incomplete table, so stop and let the entry
                // point report the give-up.
                if self.ctx.ceiling_exceeded {
                    return None;
                }
                self.ctx.last_reach = start_pos;
                if let Some(end) = self.find_at_inner(input, start_pos) {
                    return Some((start_pos, end));
                }
                walked += self.ctx.last_reach.saturating_sub(start_pos);
                if walked > budget {
                    return self.find_from_linear_inner(input, start_pos + 1);
                }
            }
            None
        }
    }

    /// Finds a match starting at the given position, returning its end.
    ///
    /// `Err` means the search gave up on the cache ceiling; see
    /// [`LazyDfa::find_from`].
    pub fn find_at(
        &mut self,
        input: &[u8],
        start: usize,
    ) -> Result<Option<usize>, CacheCeilingExceeded> {
        let mut guard = SearchGuard::new(self);
        let outcome = guard.find_at_inner(input, start);
        guard.finish(outcome)
    }

    /// [`LazyDfa::find_at`], also reporting how far the scan reached, for
    /// callers that meter attempts at many positions.
    pub(crate) fn find_at_reach(
        &mut self,
        input: &[u8],
        start: usize,
    ) -> Result<(Option<usize>, usize), CacheCeilingExceeded> {
        let mut guard = SearchGuard::new(self);
        guard.ctx.last_reach = start;
        let found = guard.find_at_inner(input, start);
        let reach = guard.ctx.last_reach;
        guard.finish((found, reach))
    }

    /// [`LazyDfa::find_at`] without the search guard.
    pub(super) fn find_at_inner(&mut self, input: &[u8], start: usize) -> Option<usize> {
        // A previous attempt in this search already ran out of cache; every
        // further one would scan an incomplete table, so unwind instead.
        if self.ctx.ceiling_exceeded {
            return None;
        }

        // If pattern has ONLY a start anchor (no end anchor), we can skip invalid positions.
        // But if it has both anchors (possibly on different alternation branches), we need
        // to try all positions and let the NFA handle anchor checking per branch.
        let start_only = self.ctx.has_start_anchor && !self.ctx.has_end_anchor;

        if start_only {
            let valid_start = if self.ctx.has_multiline_anchors {
                crate::nfa::at_line_start(input, start)
            } else {
                start == 0
            };
            if !valid_start {
                return None;
            }
        }

        let state = self.get_start_state_for_position(input, start);
        self.find_at_with_state(input, start, state)
    }

    /// Gets the appropriate start state for a position.
    fn get_start_state_for_position(&mut self, input: &[u8], start: usize) -> DfaStateId {
        let prev_class = if start > 0 {
            CharClass::from_byte(input[start - 1])
        } else {
            CharClass::NonWord
        };

        if self.ctx.has_anchors {
            let pos_ctx = if start == 0 {
                PositionContext::start_of_input()
            } else if self.ctx.has_multiline_anchors && input[start - 1] == b'\n' {
                PositionContext::after_newline()
            } else {
                PositionContext::middle()
            };

            let start_seed = [self.ctx.nfa.start];

            let is_at_boundary: Option<bool> = None;

            let start_closure = epsilon_closure_with_context(
                &self.ctx.nfa,
                &mut self.ctx.scratch,
                &start_seed,
                is_at_boundary,
                Some(pos_ctx),
            );
            let start_clean = match_reachable_without_end_assertion(
                &self.ctx.nfa,
                &mut self.ctx.scratch,
                &start_seed,
                is_at_boundary,
                Some(pos_ctx),
            );
            get_or_create_state_with_class(&mut self.ctx, start_closure, prev_class, start_clean)
        } else if self.ctx.has_word_boundary && start > 0 {
            self.get_start_state_with_prev_class(prev_class)
        } else {
            self.ctx.start
        }
    }

    /// Gets a start state with a specific previous character class.
    fn get_start_state_with_prev_class(&mut self, prev_class: CharClass) -> DfaStateId {
        if prev_class == CharClass::NonWord {
            return self.ctx.start;
        }

        let start_seed = [self.ctx.nfa.start];

        let start_closure = epsilon_closure_with_context(
            &self.ctx.nfa,
            &mut self.ctx.scratch,
            &start_seed,
            None,
            None,
        );
        let start_clean = match_reachable_without_end_assertion(
            &self.ctx.nfa,
            &mut self.ctx.scratch,
            &start_seed,
            None,
            None,
        );
        get_or_create_state_with_class(&mut self.ctx, start_closure, prev_class, start_clean)
    }

    /// Internal find implementation with explicit start state.
    fn find_at_with_state(
        &mut self,
        input: &[u8],
        start: usize,
        state: DfaStateId,
    ) -> Option<usize> {
        if !self.ctx.has_word_boundary && !self.ctx.has_anchors {
            return self.find_at_with_state_fast(input, start, state);
        }

        if !self.ctx.has_word_boundary && !self.ctx.has_multiline_anchors {
            return self.find_at_with_state_anchored(input, start, state);
        }

        self.find_at_with_state_checked(input, start, state)
    }

    /// Find that re-checks every candidate end against that state's assertions.
    fn find_at_with_state_checked(
        &mut self,
        input: &[u8],
        start: usize,
        state: DfaStateId,
    ) -> Option<usize> {
        let mut last_match = if self.is_match(state) {
            if self.check_end_assertions(input, start, state) {
                Some(start)
            } else {
                None
            }
        } else {
            None
        };

        let mut current_state = state;
        for (i, &byte) in input[start..].iter().enumerate() {
            match self.transition(current_state, byte) {
                Some(next) => {
                    current_state = next;
                    if self.is_match(current_state) {
                        let match_end = start + i + 1;
                        if self.check_end_assertions(input, match_end, current_state) {
                            last_match = Some(match_end);
                        }
                    }
                }
                None => {
                    self.ctx.last_reach = start + i;
                    return last_match;
                }
            }
        }

        self.ctx.last_reach = input.len();
        last_match
    }

    /// Fast find for patterns with only simple anchors.
    #[inline(never)]
    fn find_at_with_state_anchored(
        &mut self,
        input: &[u8],
        start: usize,
        state: DfaStateId,
    ) -> Option<usize> {
        let potential_match = self.find_at_with_state_fast(input, start, state);

        if let Some(end_pos) = potential_match {
            if !self.ctx.has_end_anchor
                || crate::nfa::at_end_or_before_final_newline(input, end_pos)
            {
                return potential_match;
            }
            // The greedy end fails `$`. If every accepting state requires it,
            // no shorter end can satisfy it either — `$` holds at one position,
            // and a shorter end is further from it — so the start is rejected.
            if !self.ctx.has_clean_accept {
                return None;
            }
            // Otherwise `$` belongs to the branch carrying it: `a|b$` on "aa"
            // reaches a match through `a`, which asks for nothing at the end.
            return self.find_at_with_state_checked(input, start, state);
        }

        None
    }

    /// Fast find implementation for patterns without assertions.
    #[inline(never)]
    fn find_at_with_state_fast(
        &mut self,
        input: &[u8],
        start: usize,
        mut state: DfaStateId,
    ) -> Option<usize> {
        let mut last_match = if self.is_match(state) {
            Some(start)
        } else {
            None
        };

        let bytes = &input[start..];
        let len = bytes.len();
        let mut i = 0;

        // Every exit records how far the scan walked. `find_from` uses it to
        // notice when start attempts are scanning the whole input each time.
        macro_rules! done {
            () => {{
                self.ctx.last_reach = start + i;
                return last_match;
            }};
        }

        while i + 4 <= len {
            let b0 = unsafe { *bytes.get_unchecked(i) };
            let b1 = unsafe { *bytes.get_unchecked(i + 1) };
            let b2 = unsafe { *bytes.get_unchecked(i + 2) };
            let b3 = unsafe { *bytes.get_unchecked(i + 3) };

            let tagged0 = self.transition_or_compute(state, b0);
            if is_dead_state(tagged0) {
                done!();
            }
            state = untag_state(tagged0);
            if is_tagged_match(tagged0) {
                last_match = Some(start + i + 1);
            }

            let tagged1 = self.transition_or_compute(state, b1);
            if is_dead_state(tagged1) {
                done!();
            }
            state = untag_state(tagged1);
            if is_tagged_match(tagged1) {
                last_match = Some(start + i + 2);
            }

            let tagged2 = self.transition_or_compute(state, b2);
            if is_dead_state(tagged2) {
                done!();
            }
            state = untag_state(tagged2);
            if is_tagged_match(tagged2) {
                last_match = Some(start + i + 3);
            }

            let tagged3 = self.transition_or_compute(state, b3);
            if is_dead_state(tagged3) {
                done!();
            }
            state = untag_state(tagged3);
            if is_tagged_match(tagged3) {
                last_match = Some(start + i + 4);
            }

            i += 4;
        }

        while i < len {
            let byte = unsafe { *bytes.get_unchecked(i) };
            let tagged = self.transition_or_compute(state, byte);
            if is_dead_state(tagged) {
                break;
            }
            state = untag_state(tagged);
            if is_tagged_match(tagged) {
                last_match = Some(start + i + 1);
            }
            i += 1;
        }

        self.ctx.last_reach = start + i;
        last_match
    }

    /// Transition in the unanchored automaton: like [`LazyDfa::transition`], but
    /// the target set also contains a match beginning at the position reached.
    ///
    /// Never dead — the start is always live — so this returns a plain state.
    pub(super) fn transition_unanchored(&mut self, state: DfaStateId, byte: u8) -> DfaStateId {
        let idx = (state + byte as u32) as usize;
        if idx < self.ctx.transitions_unanchored.len() {
            let tagged = self.ctx.transitions_unanchored[idx];
            if !is_unknown_state(tagged) {
                return untag_state(tagged);
            }
        }
        self.compute_transition_unanchored(state, byte)
    }

    /// Subset construction for one unanchored transition.
    ///
    /// Identical to [`LazyDfa::compute_transition`] except that the NFA start is
    /// added to the target set before its closure is taken. Seeding through the
    /// closure — rather than by hand — is what makes `^` behave: the closure runs
    /// in mid-input context, which kills a start-anchored branch at every
    /// position but the first.
    fn compute_transition_unanchored(&mut self, state: DfaStateId, byte: u8) -> DfaStateId {
        let state_idx = state_index(state);
        let (nfa_states, prev_class) = match self.ctx.states.get(state_idx) {
            Some(st) => (Arc::clone(&st.nfa_states), st.prev_class),
            None => return self.ctx.start,
        };

        let curr_class = CharClass::from_byte(byte);

        let is_at_boundary = if self.ctx.has_word_boundary {
            Some(prev_class != curr_class)
        } else {
            None
        };
        let pos_ctx = if self.ctx.has_anchors {
            Some(PositionContext::middle())
        } else {
            None
        };

        let expanded: NfaSubset = if self.ctx.has_word_boundary || self.ctx.has_anchors {
            epsilon_closure_with_context(
                &self.ctx.nfa,
                &mut self.ctx.scratch,
                &nfa_states[..],
                is_at_boundary,
                pos_ctx,
            )
        } else {
            nfa_states
        };

        let mut next_states = BTreeSet::new();
        for &nfa_state in expanded.iter() {
            if let Some(nfa_s) = self.ctx.nfa.get(nfa_state) {
                for (range, target) in &nfa_s.transitions {
                    if range.contains(byte) {
                        next_states.insert(*target);
                    }
                }
            }
        }
        next_states.insert(self.ctx.nfa.start);

        let target_pos_ctx = if self.ctx.has_anchors {
            if self.ctx.has_multiline_anchors && byte == b'\n' {
                Some(PositionContext::after_newline())
            } else {
                Some(PositionContext::middle())
            }
        } else {
            None
        };

        let next_closure = if self.ctx.has_word_boundary || self.ctx.has_anchors {
            epsilon_closure_with_context(
                &self.ctx.nfa,
                &mut self.ctx.scratch,
                &next_states,
                None,
                target_pos_ctx,
            )
        } else {
            epsilon_closure_subset(&self.ctx.nfa, &mut self.ctx.scratch, &next_states)
        };
        let next_clean = match_reachable_without_end_assertion(
            &self.ctx.nfa,
            &mut self.ctx.scratch,
            &next_states,
            None,
            target_pos_ctx,
        );
        let next_id =
            get_or_create_state_with_class(&mut self.ctx, next_closure, curr_class, next_clean);

        let is_match = self
            .ctx
            .states
            .get(state_index(next_id))
            .is_some_and(|st| st.is_match);

        if self.ctx.transitions_unanchored.len() < self.ctx.transitions.len() {
            self.ctx
                .transitions_unanchored
                .resize(self.ctx.transitions.len(), UNKNOWN_STATE);
        }
        let idx = (state + byte as u32) as usize;
        if idx < self.ctx.transitions_unanchored.len() {
            self.ctx.transitions_unanchored[idx] = tag_state(next_id, is_match);
        }

        next_id
    }

    /// Get transition, computing if needed, returning tagged state.
    ///
    /// The computed state is returned directly. Re-reading the table at
    /// `state + byte` instead would be reading through an ID that computing the
    /// transition may have invalidated (a cache flush renumbers every state),
    /// and would report the live target as dead.
    #[inline(always)]
    fn transition_or_compute(&mut self, state: DfaStateId, byte: u8) -> u32 {
        let idx = (state + byte as u32) as usize;
        if idx < self.ctx.transitions.len() {
            let tagged = unsafe { *self.ctx.transitions.get_unchecked(idx) };
            if !is_unknown_state(tagged) {
                return tagged;
            }
        }
        match self.compute_transition(state, byte) {
            Some(next_id) => tag_state(next_id, self.is_match(next_id)),
            None => DEAD_STATE,
        }
    }

    /// Checks if the end position satisfies any trailing assertions.
    ///
    /// For patterns like `^A|B$`, we need to check if there's ANY valid path to a match.
    /// - If there's a match state that doesn't require an assertion, the match is valid.
    /// - Only require an assertion if ALL match paths require it.
    fn check_end_assertions(&self, input: &[u8], pos: usize, state: DfaStateId) -> bool {
        if !self.ctx.has_word_boundary && !self.ctx.has_anchors {
            return true;
        }

        let state_idx = state_index(state);
        let dfa_state = match self.ctx.states.get(state_idx) {
            Some(s) => s,
            None => return true,
        };

        // For end anchors: If there's a match path that doesn't require end anchors,
        // the match is valid without satisfying them. This handles patterns like `^A|B$`.
        // Note: This only applies to end anchors, not word boundaries.
        if self.ctx.has_anchors
            && !self.ctx.has_word_boundary
            && dfa_state.match_without_end_assertion
        {
            return true;
        }

        // No clean match path - check if assertions are satisfied
        if self.ctx.has_word_boundary {
            let prev_class = dfa_state.prev_class;

            let next_class = if pos < input.len() {
                CharClass::from_byte(input[pos])
            } else {
                CharClass::NonWord
            };

            let is_at_boundary = prev_class != next_class;

            let needs_word_boundary = self.state_needs_assertion(&dfa_state.nfa_states, |instr| {
                matches!(instr, NfaInstruction::WordBoundary)
            });

            let needs_not_word_boundary = self
                .state_needs_assertion(&dfa_state.nfa_states, |instr| {
                    matches!(instr, NfaInstruction::NotWordBoundary)
                });

            if needs_word_boundary && !is_at_boundary {
                return false;
            }
            if needs_not_word_boundary && is_at_boundary {
                return false;
            }
        }

        if self.ctx.has_anchors {
            let needs_end_of_text = self.state_needs_assertion(&dfa_state.nfa_states, |instr| {
                matches!(instr, NfaInstruction::EndOfText)
            });

            if needs_end_of_text {
                // End of text, or immediately before a trailing newline — the
                // same rule `crate::reference` and the other engines follow, so
                // `a$` matches "a" in "a\n". Patterns without a word boundary
                // take the clean-match-path shortcut above and never get here,
                // which is why this only ever showed up alongside `\b`/`\B`.
                let at_end = pos == input.len()
                    || (pos + 1 == input.len() && input.get(pos) == Some(&b'\n'));
                if !at_end {
                    return false;
                }
            }

            let needs_end_of_line = self.state_needs_assertion(&dfa_state.nfa_states, |instr| {
                matches!(instr, NfaInstruction::EndOfLine)
            });

            if needs_end_of_line && !crate::nfa::at_line_end(input, pos) {
                return false;
            }
        }

        true
    }

    /// Helper to check if any NFA state in the set has a pending assertion.
    ///
    /// Important: We only check states that are actually IN the nfa_states set,
    /// NOT their epsilon targets. The epsilon closure already filtered out assertion
    /// states that don't match the current position context. Checking epsilon targets
    /// would incorrectly require assertions for paths that were already blocked.
    ///
    /// For example, in pattern `(?m)^A|B$`:
    /// - Branch 1 reaches match through ^A (no end anchor)
    /// - Branch 2 reaches match through B$ (EndOfLine)
    ///
    /// After matching, the DFA state may include states from both branches.
    /// If EndOfLine was filtered out during epsilon closure (because we're not at EOL),
    /// we shouldn't require it - branch 1's path is still valid.
    fn state_needs_assertion<F>(&self, nfa_states: &[NfaStateId], pred: F) -> bool
    where
        F: Fn(&NfaInstruction) -> bool,
    {
        nfa_states.iter().any(|&nfa_id| {
            self.ctx
                .nfa
                .get(nfa_id)
                .is_some_and(|nfa_state| nfa_state.instruction.as_ref().is_some_and(&pred))
        })
    }

    /// Returns the boundary requirements for a state.
    pub fn get_state_boundary_requirements(&self, state: DfaStateId) -> (bool, bool) {
        let state_idx = state_index(state);
        let dfa_state = match self.ctx.states.get(state_idx) {
            Some(s) => s,
            None => return (false, false),
        };

        let needs_word_boundary = self.state_needs_assertion(&dfa_state.nfa_states, |instr| {
            matches!(instr, NfaInstruction::WordBoundary)
        });

        let needs_not_word_boundary = self.state_needs_assertion(&dfa_state.nfa_states, |instr| {
            matches!(instr, NfaInstruction::NotWordBoundary)
        });

        (needs_word_boundary, needs_not_word_boundary)
    }

    /// Whether a match is reachable in `state` without passing an end assertion.
    pub fn get_state_match_without_end_assertion(&self, state: DfaStateId) -> bool {
        self.ctx
            .states
            .get(state_index(state))
            .is_some_and(|s| s.match_without_end_assertion)
    }

    /// Returns the anchor requirements for a state.
    pub fn get_state_anchor_requirements(&self, state: DfaStateId) -> (bool, bool) {
        let state_idx = state_index(state);
        let dfa_state = match self.ctx.states.get(state_idx) {
            Some(s) => s,
            None => return (false, false),
        };

        let needs_end_of_text = self.state_needs_assertion(&dfa_state.nfa_states, |instr| {
            matches!(instr, NfaInstruction::EndOfText)
        });

        let needs_end_of_line = self.state_needs_assertion(&dfa_state.nfa_states, |instr| {
            matches!(instr, NfaInstruction::EndOfLine)
        });

        (needs_end_of_text, needs_end_of_line)
    }

    /// Provides access to the internal NFA reference.
    pub fn nfa(&self) -> &Nfa {
        &self.ctx.nfa
    }

    /// The NFA this DFA was built from, shareable without copying it.
    ///
    /// Lets a caller that catches [`CacheCeilingExceeded`] build the fallback
    /// engine for the same pattern.
    pub fn nfa_arc(&self) -> Arc<Nfa> {
        self.ctx.nfa_arc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dfa::lazy::shared::ClosureScratch;
    use crate::dfa::EagerDfa;
    use crate::hir::translate;
    use crate::parser::parse;
    use std::collections::{HashSet, VecDeque};

    fn dfa(pattern: &str) -> LazyDfa {
        let hir = parse(pattern).and_then(|ast| translate(&ast)).unwrap();
        LazyDfa::new(crate::nfa::compile(&hir).unwrap())
    }

    /// REFERENCE IMPLEMENTATION — the unbatched row computation.
    ///
    /// One [`LazyDfa::compute_transition`] per byte, which is what
    /// `compute_all_transitions_with_context` did before it was batched. It is
    /// deliberately written against `transition`, the untouched per-byte entry
    /// point, so comparing the two is not circular.
    fn reference_all_transitions(
        lazy: &mut LazyDfa,
        state: DfaStateId,
    ) -> [Option<DfaStateId>; 256] {
        let mut result = [None; 256];
        for byte in 0..=255u8 {
            let cache_idx = (state + byte as u32) as usize;
            if cache_idx < lazy.ctx.transitions.len() {
                let tagged = lazy.ctx.transitions[cache_idx];
                if !is_unknown_state(tagged) {
                    if !is_dead_state(tagged) {
                        result[byte as usize] = Some(untag_state(tagged));
                    }
                    continue;
                }
            }
            if let Some(target) = lazy.transition(state, byte) {
                result[byte as usize] = Some(target);
            }
        }
        result
    }

    /// The batched row computation must be indistinguishable from the per-byte
    /// one — same targets, same state IDs, for every reachable state and every
    /// one of the 256 bytes.
    ///
    /// Two independent DFAs are walked in lockstep from the same pattern, so
    /// the comparison covers the interned IDs too: batching that merged bytes
    /// across a character-class boundary, or that swept the alphabet
    /// class-by-class instead of in byte order, would intern states in a
    /// different order and show up here as mismatched IDs rather than as a
    /// silently different-but-equivalent automaton.
    ///
    /// The counter assertion is the second half: it fails if the batched path
    /// ever falls back to computing a transition one byte at a time, so this
    /// test cannot pass with the optimization removed.
    #[test]
    fn batched_context_transitions_match_the_per_byte_path() {
        for pattern in [
            r"\b\w+\b",
            r"\bfoo\b",
            r"\B\w+",
            r"(?m)^\w+$",
            r"^\w+$",
            r"(?m)\b\w+\b$",
            // `\t` and `\n` are adjacent bytes of the same class leading to the
            // same NFA states, yet only `\n` opens the `^a` branch — so the run
            // that would otherwise merge them must be split on position
            // context, not just on target subset.
            r"(?m)([\t\n]|^a)+",
            // These pin the class-boundary run terminator, and the boundary has
            // to come *after* the consumed byte for them to bite. Every byte
            // reaches the same NFA states, so target-subset equality alone would
            // merge the whole row into one run — but `/` (0x2F) and `0` (0x30)
            // are adjacent across the Word/NonWord split, and the trailing `\b`
            // means the interned target still carries the consumed byte's class,
            // so they must become *different* DFA states.
            //
            // A leading boundary (`\b.`) does NOT work here: nothing downstream
            // reads the class, so merging across the split is harmless and the
            // test stays green with the terminator deleted. Without `.\b` this
            // guard is caught only by the randomized `resumed_iteration_matches_
            // reference` fuzz — verified by deleting it and watching this fail
            // at exactly byte 0x30.
            r".\b",
            r"(?s).\B",
        ] {
            let mut batched = dfa(pattern);
            let mut per_byte = dfa(pattern);
            assert!(
                batched.has_word_boundary() || batched.has_anchors(),
                "{pattern:?} must reach the context path for this test to mean anything"
            );

            let mut seen: HashSet<DfaStateId> = HashSet::new();
            let mut queue: VecDeque<DfaStateId> = VecDeque::new();
            for class in [CharClass::NonWord, CharClass::Word] {
                let start = batched.get_start_state_for_class(class);
                assert_eq!(
                    start,
                    per_byte.get_start_state_for_class(class),
                    "{pattern:?}: start states diverged before any transition"
                );
                if seen.insert(start) {
                    queue.push_back(start);
                }
            }

            while let Some(state) = queue.pop_front() {
                let before = batched.ctx.per_byte_computations;
                let got = batched.compute_all_transitions(state);
                assert_eq!(
                    batched.ctx.per_byte_computations, before,
                    "{pattern:?} state {state}: the batched path fell back to per-byte work"
                );

                let want = reference_all_transitions(&mut per_byte, state);
                for (byte, (batched_target, per_byte_target)) in
                    got.iter().zip(want.iter()).enumerate()
                {
                    assert_eq!(
                        batched_target, per_byte_target,
                        "{pattern:?} state {state} byte {byte:#04x}"
                    );
                }

                for &target in got.iter().flatten() {
                    if seen.insert(target) {
                        queue.push_back(target);
                    }
                }
            }

            assert!(
                seen.len() > 1,
                "{pattern:?}: expected more than the start state to be reachable"
            );
        }
    }

    /// The probe-key restructure in `get_or_create_state_with_class` — moving
    /// `nfa_states` into the key on the hit path and cloning only on miss —
    /// must not change interning: asking for the same NFA subset twice has to
    /// return the same DFA state id rather than creating a duplicate.
    ///
    /// Note that the two asks here share one allocation (`Arc::clone` is a
    /// refcount bump), so this would still pass if state identity were pointer
    /// based. `independently_built_identical_subsets_intern_to_one_state` is
    /// what covers that.
    #[test]
    fn get_or_create_state_interns_repeated_subset() {
        let lazy = dfa(r"\b\w+\b");
        let mut ctx = lazy.ctx.clone();

        let states: NfaSubset = Arc::from(&[0u32, 2u32][..]);

        let first =
            get_or_create_state_with_class(&mut ctx, Arc::clone(&states), CharClass::Word, true);
        let states_before = ctx.state_count();
        let second = get_or_create_state_with_class(&mut ctx, states, CharClass::Word, true);

        assert_eq!(first, second, "identical subset must intern to the same id");
        assert_eq!(
            ctx.state_count(),
            states_before,
            "asking for an already-cached subset must not grow the cache"
        );
    }

    /// State identity is the subset's *contents*, never its allocation.
    ///
    /// Storing the subset behind an `Arc` makes that a real question: the map
    /// key and the stored `DfaState` deliberately share one allocation, so it
    /// would be easy to slip into treating the pointer as the identity. Two
    /// closure walks that happen to produce the same subset return two separate
    /// allocations, and they must still intern to one DFA state — otherwise the
    /// cache grows a duplicate for every re-derivation of the same state, and
    /// the whole subset construction stops converging.
    ///
    /// `Arc<[u32]>`'s `PartialEq`/`Hash` both go through the pointee, which is
    /// what makes this hold; this test is what stops it from being quietly
    /// traded away.
    #[test]
    fn independently_built_identical_subsets_intern_to_one_state() {
        let lazy = dfa(r"\b\w+\b");
        let mut ctx = lazy.ctx.clone();

        let first_subset: NfaSubset = Arc::from(&[0u32, 2u32][..]);
        let second_subset: NfaSubset = Arc::from(&[0u32, 2u32][..]);
        assert!(
            !Arc::ptr_eq(&first_subset, &second_subset),
            "the two subsets must be separate allocations for this test to mean anything"
        );

        let first = get_or_create_state_with_class(&mut ctx, first_subset, CharClass::Word, true);
        let states_before = ctx.state_count();
        let second = get_or_create_state_with_class(&mut ctx, second_subset, CharClass::Word, true);

        assert_eq!(
            first, second,
            "equal subsets in separate allocations must intern to the same id"
        );
        assert_eq!(
            ctx.state_count(),
            states_before,
            "the second ask must not create a duplicate state"
        );
    }

    /// The shared closure scratch must not carry one walk into the next.
    ///
    /// Two failure modes, both invisible to every other test because they only
    /// show up in the *second* of two consecutive walks:
    ///
    /// - the touched list is not cleared, so walk two's subset also contains
    ///   walk one's states;
    /// - the generation counter is not bumped, so walk one's marks still read
    ///   as current and walk two skips every state they cover.
    ///
    /// The seeds are chosen so both bite. `seed_b` is taken from inside
    /// `seed_a`'s closure and has a strictly smaller closure of its own, so a
    /// stale touched list yields `seed_a`'s (too large) subset and stale marks
    /// yield an empty one — while the correct answer is neither.
    #[test]
    fn closure_scratch_does_not_leak_between_calls() {
        let lazy = dfa(r"(?m)^(foo|bar)+baz$");
        let nfa = lazy.nfa();
        let pos_ctx = Some(PositionContext::start_of_input());

        // Each oracle walk gets its OWN scratch. Sharing one across the oracle
        // calls would corrupt the expected values in exactly the same way the
        // buffer under test is corrupted, so a missing reset would agree with
        // itself and the test would pass — which is precisely what happened
        // before: dropping the generation bump left this green.
        let fresh = |seed: &[NfaStateId]| {
            let mut scratch = ClosureScratch::new(nfa.states.len());
            epsilon_closure_with_context(nfa, &mut scratch, seed, None, pos_ctx)
        };

        let seed_a = [nfa.start];
        let closure_a = fresh(&seed_a);

        // A member of closure(A) whose own closure is a proper subset of it —
        // so it is both reachable from A (stale marks bite) and different from
        // A (a stale touched list bites).
        let (seed_b, closure_b) = closure_a
            .iter()
            .map(|&id| {
                let seed = [id];
                let closure = fresh(&seed);
                (seed, closure)
            })
            .find(|(_, closure)| closure.len() < closure_a.len())
            .expect("some state in the start closure must have a strictly smaller closure");

        // Now the same two walks through one shared buffer.
        let mut shared = ClosureScratch::new(nfa.states.len());
        let first = epsilon_closure_with_context(nfa, &mut shared, &seed_a, None, pos_ctx);
        let second = epsilon_closure_with_context(nfa, &mut shared, &seed_b, None, pos_ctx);

        assert_eq!(first, closure_a, "the first walk must be unaffected");
        assert_eq!(
            second, closure_b,
            "the second walk must equal what a fresh scratch produces"
        );
        assert_ne!(
            first, second,
            "the two walks must differ for this test to mean anything"
        );
    }

    /// Pins that the sweep actually merges, rather than recomputing per byte.
    ///
    /// `batched_context_transitions_match_the_per_byte_path` proves the batched
    /// sweep computes the *same transitions* as the per-byte path regardless of
    /// how many runs it took to get there — `get_or_create_state_with_class`'s
    /// interning makes the final ids converge either way. So it cannot tell
    /// "merged into a few runs" from "recomputed almost every byte". This test
    /// counts runs (`context_run_computations`, bumped once per run rather than
    /// once per byte) to cover that gap: for `\b\w+\b`'s "inside a word" state
    /// every word byte in a contiguous class range reaches the identical target
    /// set, so the 256-byte row must collapse into a handful of runs.
    ///
    /// This is also what shows the rows do not need canonicalising: sorting and
    /// deduping them was tried, left this count unchanged, and cost instructions
    /// on every compile — see the note in `compute_all_transitions_with_context`.
    #[test]
    fn context_transitions_merge_into_few_runs() {
        let mut lazy = dfa(r"\b\w+\b");
        // `\b` holds crossing NonWord -> Word, so the real anchored start
        // (always tagged NonWord) is what needs to be used here: starting
        // from a state already tagged Word would make `\b` fail on the very
        // next word byte and dead-end the transition this test needs.
        let start = lazy.get_start_state_for_class(CharClass::NonWord);
        let inside_word = lazy.compute_all_transitions(start)[b'a' as usize]
            .expect("'a' must transition into the inside-a-word state from the true start");

        let before = lazy.ctx.context_run_computations;
        let _ = lazy.compute_all_transitions(inside_word);
        let runs = lazy.ctx.context_run_computations - before;

        assert!(
            runs <= 10,
            "expected the word-class row to collapse into a handful of runs, got {runs}"
        );
    }

    /// An end anchor belongs to the branch that carries it, not to the pattern.
    ///
    /// `a|b$` can match through `a`, which asks for nothing at the end, so 0..1
    /// of "aa" stands even though `$` does not hold there. `a$|b` must not get
    /// the same treatment, and `world$` — where the anchor sits on the accepting
    /// state itself rather than a predecessor — must still enforce it.
    ///
    /// Driven against the engines directly: engine selection sends these to the
    /// PikeVM today, which would hide a regression here behind the selector.
    /// Every expectation was taken from `regexr::reference` and cross-checked
    /// against the `regex` crate.
    #[test]
    fn end_anchor_in_one_branch_does_not_gate_the_others() {
        for (pattern, input, expected) in [
            ("a|b$", "aa", Some((0, 1))),
            ("a|b$", "ab", Some((0, 1))),
            ("a|b$", "ba", Some((1, 2))),
            ("a|a$", "aa", Some((0, 1))),
            ("a$|b", "ab", Some((1, 2))),
            ("a$|b", "aa", Some((1, 2))),
            ("world$", "world hello", None),
            ("world$", "hello world", Some((6, 11))),
        ] {
            let mut lazy = dfa(pattern);
            assert_eq!(
                lazy.find_from(input.as_bytes(), 0),
                Ok(expected),
                "LazyDfa {pattern:?} on {input:?}"
            );

            let eager = EagerDfa::from_lazy(&mut lazy)
                .expect("pattern is small; must not exceed the budget");
            assert_eq!(
                eager.find_from(input.as_bytes(), 0),
                Ok(expected),
                "EagerDfa {pattern:?} on {input:?}"
            );
        }
    }
}
