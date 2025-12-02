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

use crate::nfa::{Nfa, NfaInstruction, StateId as NfaStateId};

use super::super::shared::{
    epsilon_closure_with_context, flush_cache, get_or_create_state_with_class, is_dead_state,
    is_tagged_match, is_unknown_state, state_index, tag_state, untag_state, CharClass, DfaStateId,
    LazyDfaContext, PositionContext, DEAD_STATE, UNKNOWN_STATE,
};

/// A lazy DFA that builds states on demand.
#[derive(Debug, Clone)]
pub struct LazyDfa {
    /// Internal context containing state and transition data.
    ctx: LazyDfaContext,
}

impl LazyDfa {
    /// Creates a new lazy DFA from an NFA.
    pub fn new(nfa: Nfa) -> Self {
        Self {
            ctx: LazyDfaContext::new(nfa),
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

    /// Returns true if this DFA has multiline anchors.
    pub fn has_multiline_anchors(&self) -> bool {
        self.ctx.has_multiline_anchors()
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
        let state_idx = state_index(state);
        let dfa_state = self.ctx.states.get(state_idx)?;
        let nfa_states = dfa_state.nfa_states.clone();
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

        let expanded_nfa_states = if self.ctx.has_word_boundary || self.ctx.has_anchors {
            epsilon_closure_with_context(&self.ctx.nfa, &nfa_states, is_at_boundary, pos_ctx)
        } else {
            nfa_states
        };

        let mut next_states = BTreeSet::new();

        for &nfa_state in &expanded_nfa_states {
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
            epsilon_closure_with_context(&self.ctx.nfa, &next_states, None, target_pos_ctx)
        } else {
            self.ctx.nfa.epsilon_closure(&next_states)
        };

        if next_closure.is_empty() {
            if cache_idx < self.ctx.transitions.len() {
                self.ctx.transitions[cache_idx] = DEAD_STATE;
            }
            return None;
        }

        let next_id = get_or_create_state_with_class(&mut self.ctx, next_closure, curr_class);

        let next_idx = state_index(next_id);
        let is_match = self.ctx.states.get(next_idx).is_some_and(|s| s.is_match);

        let cache_idx = (state + byte as u32) as usize;
        if cache_idx < self.ctx.transitions.len() {
            self.ctx.transitions[cache_idx] = tag_state(next_id, is_match);
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
            Some(s) => s.nfa_states.clone(),
            None => return,
        };

        let mut byte_targets: [Option<BTreeSet<u32>>; 256] = std::array::from_fn(|_| None);

        for &nfa_state in &nfa_states {
            if let Some(nfa_s) = self.ctx.nfa.get(nfa_state) {
                for (range, target) in &nfa_s.transitions {
                    for byte in range.start..=range.end {
                        byte_targets[byte as usize]
                            .get_or_insert_with(BTreeSet::new)
                            .insert(*target);
                    }
                }
            }
        }

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

                let next_closure = self.ctx.nfa.epsilon_closure(targets);
                if next_closure.is_empty() {
                    if cache_idx < self.ctx.transitions.len() {
                        self.ctx.transitions[cache_idx] = DEAD_STATE;
                    }
                    continue;
                }

                let next_id =
                    get_or_create_state_with_class(&mut self.ctx, next_closure, CharClass::NonWord);
                result[byte as usize] = Some(next_id);

                let next_idx = state_index(next_id);
                let is_match = self.ctx.states.get(next_idx).is_some_and(|s| s.is_match);
                let cache_idx = (state + byte as u32) as usize;
                if cache_idx < self.ctx.transitions.len() {
                    self.ctx.transitions[cache_idx] = tag_state(next_id, is_match);
                }
            } else if cache_idx < self.ctx.transitions.len() {
                self.ctx.transitions[cache_idx] = DEAD_STATE;
            }
        }
    }

    /// Computes all transitions for patterns with word boundaries or anchors.
    fn compute_all_transitions_with_context(
        &mut self,
        state: DfaStateId,
        result: &mut [Option<DfaStateId>; 256],
    ) {
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
            if let Some(target) = self.transition(state, byte) {
                result[byte as usize] = Some(target);
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

    /// Executes the DFA on input, returning true if it matches.
    pub fn is_match_bytes(&mut self, input: &[u8]) -> bool {
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
    pub fn find(&mut self, input: &[u8]) -> Option<(usize, usize)> {
        // If pattern has both start and end anchors, they may be on different branches
        // of an alternation (e.g., ^X|Y$), so we need to do an unanchored search.
        // Only optimize with line-boundary-only search if we have ONLY a start anchor.
        let start_only = self.ctx.has_start_anchor && !self.ctx.has_end_anchor;

        if start_only {
            if self.ctx.has_multiline_anchors {
                if let Some(end) = self.find_at(input, 0) {
                    return Some((0, end));
                }
                for (i, &byte) in input.iter().enumerate() {
                    if byte == b'\n' && i < input.len() {
                        if let Some(end) = self.find_at(input, i + 1) {
                            return Some((i + 1, end));
                        }
                    }
                }
                None
            } else {
                self.find_at(input, 0).map(|end| (0, end))
            }
        } else {
            for start_pos in 0..=input.len() {
                if let Some(end) = self.find_at(input, start_pos) {
                    return Some((start_pos, end));
                }
            }
            None
        }
    }

    /// Finds a match starting at the given position.
    pub fn find_at(&mut self, input: &[u8], start: usize) -> Option<usize> {
        // If pattern has ONLY a start anchor (no end anchor), we can skip invalid positions.
        // But if it has both anchors (possibly on different alternation branches), we need
        // to try all positions and let the NFA handle anchor checking per branch.
        let start_only = self.ctx.has_start_anchor && !self.ctx.has_end_anchor;

        if start_only {
            let valid_start = if self.ctx.has_multiline_anchors {
                start == 0 || (start > 0 && input[start - 1] == b'\n')
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

            let mut start_set = BTreeSet::new();
            start_set.insert(self.ctx.nfa.start);

            let is_at_boundary: Option<bool> = None;

            let start_closure = epsilon_closure_with_context(
                &self.ctx.nfa,
                &start_set,
                is_at_boundary,
                Some(pos_ctx),
            );
            get_or_create_state_with_class(&mut self.ctx, start_closure, prev_class)
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

        let mut start_set = BTreeSet::new();
        start_set.insert(self.ctx.nfa.start);

        let start_closure = epsilon_closure_with_context(&self.ctx.nfa, &start_set, None, None);
        get_or_create_state_with_class(&mut self.ctx, start_closure, prev_class)
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
                None => break,
            }
        }

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
            if self.ctx.has_end_anchor && end_pos != input.len() {
                return None;
            }
        }

        potential_match
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

        while i + 4 <= len {
            let b0 = unsafe { *bytes.get_unchecked(i) };
            let b1 = unsafe { *bytes.get_unchecked(i + 1) };
            let b2 = unsafe { *bytes.get_unchecked(i + 2) };
            let b3 = unsafe { *bytes.get_unchecked(i + 3) };

            let tagged0 = self.transition_or_compute(state, b0);
            if is_dead_state(tagged0) {
                return last_match;
            }
            state = untag_state(tagged0);
            if is_tagged_match(tagged0) {
                last_match = Some(start + i + 1);
            }

            let tagged1 = self.transition_or_compute(state, b1);
            if is_dead_state(tagged1) {
                return last_match;
            }
            state = untag_state(tagged1);
            if is_tagged_match(tagged1) {
                last_match = Some(start + i + 2);
            }

            let tagged2 = self.transition_or_compute(state, b2);
            if is_dead_state(tagged2) {
                return last_match;
            }
            state = untag_state(tagged2);
            if is_tagged_match(tagged2) {
                last_match = Some(start + i + 3);
            }

            let tagged3 = self.transition_or_compute(state, b3);
            if is_dead_state(tagged3) {
                return last_match;
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

        last_match
    }

    /// Get transition, computing if needed, returning tagged state.
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
            Some(_) => {
                let idx = (state + byte as u32) as usize;
                if idx < self.ctx.transitions.len() {
                    unsafe { *self.ctx.transitions.get_unchecked(idx) }
                } else {
                    DEAD_STATE
                }
            }
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
        if self.ctx.has_anchors && !self.ctx.has_word_boundary {
            let has_clean_match_path = self.has_match_without_end_anchors(&dfa_state.nfa_states);
            if has_clean_match_path {
                return true;
            }
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

            if needs_end_of_text && pos != input.len() {
                return false;
            }

            let needs_end_of_line = self.state_needs_assertion(&dfa_state.nfa_states, |instr| {
                matches!(instr, NfaInstruction::EndOfLine)
            });

            if needs_end_of_line {
                let at_end_of_line = pos == input.len() || input.get(pos) == Some(&b'\n');
                if !at_end_of_line {
                    return false;
                }
            }
        }

        true
    }

    /// Checks if there's a match state that doesn't have any pending END anchors.
    /// This handles patterns like `^A|B$` where branch 1 can match without EndOfLine.
    ///
    /// Note: This only checks for END anchors (EndOfLine, EndOfText), not word boundaries.
    /// Word boundaries are handled differently during transition computation.
    fn has_match_without_end_anchors(&self, nfa_states: &BTreeSet<NfaStateId>) -> bool {
        // Find all match states in the closure
        for &nfa_id in nfa_states {
            if let Some(nfa_state) = self.ctx.nfa.get(nfa_id) {
                if nfa_state.is_match {
                    // Check if this match state has any pending END anchor
                    let has_end_anchor = nfa_state.instruction.as_ref().is_some_and(|instr| {
                        matches!(instr, NfaInstruction::EndOfLine | NfaInstruction::EndOfText)
                    });
                    if !has_end_anchor {
                        // Found a match state without pending end anchors
                        return true;
                    }
                }
            }
        }
        false
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
    fn state_needs_assertion<F>(&self, nfa_states: &BTreeSet<NfaStateId>, pred: F) -> bool
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
}
