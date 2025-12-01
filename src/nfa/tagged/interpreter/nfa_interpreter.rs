//! Thompson NFA interpreter with sparse capture copying.
//!
//! This interpreter implements full Tagged NFA simulation using liveness
//! analysis for efficient capture copying during thread spawning.

use std::collections::HashMap;
use crate::nfa::{Nfa, NfaInstruction, StateId};
use crate::nfa::tagged::shared::ThreadWorklist;
use crate::nfa::tagged::liveness::{analyze_liveness, NfaLiveness};

/// Memoization cache for lookaround evaluations.
/// Key: (state_id, position), Value: match result
type LookaroundMemo = HashMap<(StateId, usize), bool>;

/// Interpreter for Tagged NFA with sparse capture copying.
///
/// This interpreter uses liveness analysis to minimize capture copying
/// during thread spawning. Only captures that are "live" (may be read
/// downstream) are copied, reducing overhead for patterns with many
/// capture groups.
pub struct TaggedNfaInterpreter<'a> {
    nfa: &'a Nfa,
    liveness: &'a NfaLiveness,
    stride: usize,
    /// Use sparse copies based on liveness analysis.
    use_sparse_copy: bool,
}

impl<'a> TaggedNfaInterpreter<'a> {
    /// Creates a new interpreter.
    pub fn new(nfa: &'a Nfa, liveness: &'a NfaLiveness) -> Self {
        let stride = (nfa.capture_count as usize + 1) * 2;
        Self { nfa, liveness, stride, use_sparse_copy: true }
    }

    /// Creates a new interpreter with sparse copying disabled (for testing).
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn new_no_sparse(nfa: &'a Nfa, liveness: &'a NfaLiveness) -> Self {
        let stride = (nfa.capture_count as usize + 1) * 2;
        Self { nfa, liveness, stride, use_sparse_copy: false }
    }

    /// Finds the first match.
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        for start in 0..=input.len() {
            if let Some(end) = self.match_at(input, start) {
                return Some((start, end));
            }
        }
        None
    }

    /// Returns captures for the first match.
    pub fn captures(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        for start in 0..=input.len() {
            if let Some(caps) = self.captures_at(input, start) {
                return Some(caps);
            }
        }
        None
    }

    /// Attempts to match at a specific position.
    fn match_at(&self, input: &[u8], start: usize) -> Option<usize> {
        self.captures_at(input, start)
            .and_then(|caps| caps.first().and_then(|c| c.map(|(_, end)| end)))
    }

    /// Returns captures for a match starting at the given position.
    /// This is made public for use by `TaggedNfaEngine`.
    pub fn captures_at(&self, input: &[u8], start: usize) -> Option<Vec<Option<(usize, usize)>>> {
        let state_count = self.nfa.states.len();
        let capture_count = self.nfa.capture_count;

        let mut current = ThreadWorklist::new(capture_count, state_count);
        let mut next = ThreadWorklist::new(capture_count, state_count);

        let mut best_captures: Vec<i64> = vec![-1; self.stride];
        let mut best_end: i64 = -1;
        let mut best_priority: u32 = 0;

        // Memoization cache for lookaround results
        let mut lookaround_memo: LookaroundMemo = HashMap::new();

        // Add initial thread at start state with position = start
        self.add_thread(&mut current, self.nfa.start, 0, input, start, None, &mut lookaround_memo);

        // Set capture 0 start position for all initial threads
        for i in 0..current.count {
            current.set_capture(i, 0, start as i64);
        }

        let mut pos = start;

        while pos <= input.len() {
            if current.count == 0 {
                break;
            }

            let byte = input.get(pos).copied();

            // Check for matches and process transitions
            for thread_idx in 0..current.count {
                let state_id = current.states[thread_idx];
                let state = match self.nfa.states.get(state_id as usize) {
                    Some(s) => s,
                    None => continue,
                };

                // Handle backref instruction (consumes input like a transition)
                if let Some(NfaInstruction::Backref(idx)) = &state.instruction {
                    let idx = *idx as usize;
                    let start_slot = idx * 2;
                    let end_slot = idx * 2 + 1;

                    let cap_start = current.get_capture(thread_idx, start_slot);
                    let cap_end = current.get_capture(thread_idx, end_slot);

                    if cap_start >= 0 && cap_end >= 0 {
                        let cap_start = cap_start as usize;
                        let cap_end = cap_end as usize;
                        let cap_len = cap_end - cap_start;

                        // Check if backref matches at current position
                        if pos + cap_len <= input.len() {
                            let captured = &input[cap_start..cap_end];
                            let current_slice = &input[pos..pos + cap_len];
                            if captured == current_slice {
                                // Backref matches!
                                let match_pos = pos + cap_len;

                                // If this is also a match state, record match at pos + cap_len
                                if state.is_match {
                                    let thread_priority = current.flags[thread_idx] & 0xFFFF;
                                    let is_non_greedy = (current.flags[thread_idx] & 0x10000) != 0;

                                    if is_non_greedy {
                                        return self.extract_captures_at_pos(&current, thread_idx, start, match_pos);
                                    }

                                    if best_end < 0 || thread_priority >= best_priority {
                                        best_end = match_pos as i64;
                                        best_priority = thread_priority;
                                        for slot in 0..self.stride {
                                            best_captures[slot] = current.get_capture(thread_idx, slot);
                                        }
                                        best_captures[0] = start as i64;
                                        best_captures[1] = match_pos as i64;
                                    }
                                }

                                // Follow epsilon transitions from this state
                                // These transitions go to `next` at position `match_pos`
                                for &epsilon_target in &state.epsilon {
                                    self.add_thread_with_captures(
                                        &mut next,
                                        epsilon_target,
                                        current.flags[thread_idx],
                                        input,
                                        match_pos,
                                        &current,
                                        thread_idx,
                                        &mut lookaround_memo,
                                    );
                                }
                            }
                        }
                    }
                    continue; // Skip normal match/transition processing for backref states
                }

                // Check for match (non-backref states)
                if state.is_match {
                    let thread_priority = current.flags[thread_idx] & 0xFFFF;
                    let is_non_greedy = (current.flags[thread_idx] & 0x10000) != 0;

                    // Non-greedy match: return immediately
                    if is_non_greedy {
                        return self.extract_captures(&current, thread_idx, start, pos);
                    }

                    // Greedy: record if better priority
                    if best_end < 0 || thread_priority >= best_priority {
                        best_end = pos as i64;
                        best_priority = thread_priority;
                        // Copy captures from this thread
                        for slot in 0..self.stride {
                            best_captures[slot] = current.get_capture(thread_idx, slot);
                        }
                        // Group 0 (full match) is always (start, pos)
                        best_captures[0] = start as i64; // Group 0 start
                        best_captures[1] = pos as i64;   // Group 0 end
                    }
                }

                // Process byte transitions
                if let Some(b) = byte {
                    for (range, target) in &state.transitions {
                        if b >= range.start && b <= range.end {
                            // Spawn thread in next worklist
                            self.add_thread_with_captures(
                                &mut next,
                                *target,
                                current.flags[thread_idx],
                                input,
                                pos + 1,
                                &current,
                                thread_idx,
                                &mut lookaround_memo,
                            );
                        }
                    }
                }
            }

            // Swap worklists
            std::mem::swap(&mut current, &mut next);
            next.clear();
            pos += 1;
        }

        if best_end >= 0 {
            // Build captures result from best_captures
            let mut result = Vec::with_capacity(capture_count as usize + 1);
            for group in 0..=capture_count as usize {
                let s = best_captures[group * 2];
                let e = best_captures[group * 2 + 1];
                if s >= 0 && e >= 0 {
                    result.push(Some((s as usize, e as usize)));
                } else {
                    result.push(None);
                }
            }
            Some(result)
        } else {
            None
        }
    }

    /// Adds a thread via epsilon closure, tracking captures along the way.
    ///
    /// This follows all epsilon transitions recursively, adding threads
    /// only for states that need to be in the worklist (have byte transitions
    /// or are match states).
    ///
    /// `pending_captures` contains capture updates accumulated during epsilon closure
    /// that need to be applied to any thread we create.
    fn add_thread(
        &self,
        worklist: &mut ThreadWorklist,
        state: StateId,
        flags: u32,
        input: &[u8],
        pos: usize,
        source: Option<(&ThreadWorklist, usize)>,
        lookaround_memo: &mut LookaroundMemo,
    ) {
        // Start with no pending captures
        let mut pending: Vec<(usize, i64)> = Vec::new();
        self.add_thread_inner(worklist, state, flags, input, pos, source, &mut pending, lookaround_memo);
    }

    /// Inner recursive helper that tracks pending captures.
    fn add_thread_inner(
        &self,
        worklist: &mut ThreadWorklist,
        state: StateId,
        flags: u32,
        input: &[u8],
        pos: usize,
        source: Option<(&ThreadWorklist, usize)>,
        pending_captures: &mut Vec<(usize, i64)>,
        lookaround_memo: &mut LookaroundMemo,
    ) {
        // Check visited to avoid infinite loops
        if worklist.is_visited(state) {
            return;
        }
        worklist.mark_visited(state);

        let nfa_state = match self.nfa.states.get(state as usize) {
            Some(s) => s,
            None => return,
        };

        // Handle instruction
        let mut new_flags = flags;
        if let Some(ref instr) = nfa_state.instruction {
            match instr {
                NfaInstruction::CaptureStart(idx) => {
                    pending_captures.push(((*idx as usize) * 2, pos as i64));
                }
                NfaInstruction::CaptureEnd(idx) => {
                    pending_captures.push(((*idx as usize) * 2 + 1, pos as i64));
                }
                NfaInstruction::NonGreedyExit => {
                    new_flags |= 0x10000; // Set non-greedy bit
                }
                NfaInstruction::StartOfText => {
                    if pos != 0 {
                        return; // Kill thread
                    }
                }
                NfaInstruction::EndOfText => {
                    if pos != input.len() {
                        return;
                    }
                }
                NfaInstruction::StartOfLine => {
                    if pos != 0 && (pos == 0 || input[pos - 1] != b'\n') {
                        return;
                    }
                }
                NfaInstruction::EndOfLine => {
                    if pos != input.len() && input.get(pos) != Some(&b'\n') {
                        return;
                    }
                }
                NfaInstruction::WordBoundary => {
                    if !self.is_word_boundary(input, pos) {
                        return;
                    }
                }
                NfaInstruction::NotWordBoundary => {
                    if self.is_word_boundary(input, pos) {
                        return;
                    }
                }
                NfaInstruction::PositiveLookahead(inner_nfa) => {
                    // Check memoization cache first
                    let cache_key = (state, pos);
                    let result = if let Some(&cached) = lookaround_memo.get(&cache_key) {
                        cached
                    } else {
                        // Evaluate lookahead: does inner_nfa match AT current position?
                        // Important: we check that the match starts at position 0 of the slice,
                        // not just anywhere in the remaining input.
                        let inner_liveness = analyze_liveness(inner_nfa);
                        let inner_interp = TaggedNfaInterpreter::new(inner_nfa, &inner_liveness);
                        let result = inner_interp.find(&input[pos..])
                            .map(|(start, _)| start == 0)
                            .unwrap_or(false);
                        lookaround_memo.insert(cache_key, result);
                        result
                    };
                    if !result {
                        return; // Kill thread
                    }
                }
                NfaInstruction::NegativeLookahead(inner_nfa) => {
                    // Check memoization cache first
                    let cache_key = (state, pos);
                    let result = if let Some(&cached) = lookaround_memo.get(&cache_key) {
                        cached
                    } else {
                        // Evaluate lookahead: does inner_nfa NOT match AT current position?
                        // Important: we check that no match starts at position 0.
                        let inner_liveness = analyze_liveness(inner_nfa);
                        let inner_interp = TaggedNfaInterpreter::new(inner_nfa, &inner_liveness);
                        let result = inner_interp.find(&input[pos..])
                            .map(|(start, _)| start != 0)
                            .unwrap_or(true);
                        lookaround_memo.insert(cache_key, result);
                        result
                    };
                    if !result {
                        return; // Kill thread
                    }
                }
                NfaInstruction::PositiveLookbehind(inner_nfa) => {
                    // Check memoization cache first
                    let cache_key = (state, pos);
                    let result = if let Some(&cached) = lookaround_memo.get(&cache_key) {
                        cached
                    } else {
                        // Evaluate lookbehind: does inner_nfa match ending at current position?
                        let inner_liveness = analyze_liveness(inner_nfa);
                        let inner_interp = TaggedNfaInterpreter::new(inner_nfa, &inner_liveness);
                        let mut found = false;
                        for start in 0..=pos {
                            let slice = &input[start..pos];
                            if let Some((s, e)) = inner_interp.find(slice) {
                                if s == 0 && e == slice.len() {
                                    found = true;
                                    break;
                                }
                            }
                        }
                        lookaround_memo.insert(cache_key, found);
                        found
                    };
                    if !result {
                        return; // Kill thread
                    }
                }
                NfaInstruction::NegativeLookbehind(inner_nfa) => {
                    // Check memoization cache first
                    let cache_key = (state, pos);
                    let result = if let Some(&cached) = lookaround_memo.get(&cache_key) {
                        cached
                    } else {
                        // Evaluate lookbehind: does inner_nfa NOT match ending at current position?
                        let inner_liveness = analyze_liveness(inner_nfa);
                        let inner_interp = TaggedNfaInterpreter::new(inner_nfa, &inner_liveness);
                        let mut found = false;
                        for start in 0..=pos {
                            let slice = &input[start..pos];
                            if let Some((s, e)) = inner_interp.find(slice) {
                                if s == 0 && e == slice.len() {
                                    found = true;
                                    break;
                                }
                            }
                        }
                        lookaround_memo.insert(cache_key, !found);
                        !found
                    };
                    if !result {
                        return; // Kill thread
                    }
                }
                NfaInstruction::Backref(idx) => {
                    // Backrefs need to check captured content.
                    // The capture might be in pending_captures (set during this epsilon closure)
                    // or in the source worklist (from previous position).
                    let idx = *idx as usize;
                    let start_slot = idx * 2;
                    let end_slot = idx * 2 + 1;

                    // First check pending_captures for this capture
                    let mut cap_start_val: Option<i64> = None;
                    let mut cap_end_val: Option<i64> = None;
                    for &(slot, val) in pending_captures.iter() {
                        if slot == start_slot {
                            cap_start_val = Some(val);
                        } else if slot == end_slot {
                            cap_end_val = Some(val);
                        }
                    }

                    // Fall back to source worklist if not in pending
                    if let Some((src_wl, src_idx)) = source {
                        if cap_start_val.is_none() {
                            let v = src_wl.get_capture(src_idx, start_slot);
                            if v >= 0 {
                                cap_start_val = Some(v);
                            }
                        }
                        if cap_end_val.is_none() {
                            let v = src_wl.get_capture(src_idx, end_slot);
                            if v >= 0 {
                                cap_end_val = Some(v);
                            }
                        }
                    }

                    if let (Some(cap_start), Some(cap_end)) = (cap_start_val, cap_end_val) {
                        if cap_start >= 0 && cap_end >= 0 {
                            let cap_start = cap_start as usize;
                            let cap_end = cap_end as usize;
                            let cap_len = cap_end - cap_start;

                            // Check if backref matches at current position
                            if pos + cap_len <= input.len() {
                                let captured = &input[cap_start..cap_end];
                                let current_input = &input[pos..pos + cap_len];
                                if captured == current_input {
                                    // Backref matches - continue processing
                                    // The match will be recorded in the main loop
                                } else {
                                    return; // Kill: backref doesn't match
                                }
                            } else {
                                return; // Kill: not enough input for backref
                            }
                        } else {
                            return; // Kill: capture group has invalid values
                        }
                    } else {
                        return; // Kill: capture group not set
                    }
                }
                NfaInstruction::CodepointClass(_, _) => {
                    // TODO: Unicode codepoint class handling
                }
            }
        }

        // If this state has byte transitions, is a match, or has a backref instruction, add the thread
        // Backref states need to be added because backrefs consume input bytes in the main loop
        let has_backref = matches!(nfa_state.instruction, Some(NfaInstruction::Backref(_)));
        if !nfa_state.transitions.is_empty() || nfa_state.is_match || has_backref {
            if let Some(thread_idx) = worklist.add_thread(state, new_flags) {
                // Copy captures from source thread if provided
                if let Some((src_wl, src_idx)) = source {
                    if self.use_sparse_copy {
                        // Sparse copy: only copy captures in the copy_mask
                        let copy_mask = self.liveness.copy_mask(state);
                        for group in copy_mask.iter() {
                            let start_slot = (group as usize) * 2;
                            let end_slot = start_slot + 1;
                            if start_slot < worklist.stride {
                                worklist.set_capture(thread_idx, start_slot, src_wl.get_capture(src_idx, start_slot));
                            }
                            if end_slot < worklist.stride {
                                worklist.set_capture(thread_idx, end_slot, src_wl.get_capture(src_idx, end_slot));
                            }
                        }
                        // Always copy group 0 (full match) - needed for result extraction
                        worklist.set_capture(thread_idx, 0, src_wl.get_capture(src_idx, 0));
                        worklist.set_capture(thread_idx, 1, src_wl.get_capture(src_idx, 1));
                    } else {
                        // Full copy: copy all captures
                        for slot in 0..worklist.stride {
                            let val = src_wl.get_capture(src_idx, slot);
                            worklist.set_capture(thread_idx, slot, val);
                        }
                    }
                }

                // Apply all pending captures accumulated during epsilon closure
                for &(slot, val) in pending_captures.iter() {
                    worklist.set_capture(thread_idx, slot, val);
                }
            }
        }

        // ALWAYS follow epsilon transitions, even if we didn't add a thread.
        // This is critical: states like "1" in "abc" have no byte transitions,
        // but they lead via epsilon to states that DO have byte transitions.
        let pending_len = pending_captures.len();
        for &target in &nfa_state.epsilon {
            self.add_thread_inner(worklist, target, new_flags, input, pos, source, pending_captures, lookaround_memo);
            // Restore pending_captures to state before this branch
            pending_captures.truncate(pending_len);
        }
    }

    /// Adds a thread with captures copied from source thread.
    fn add_thread_with_captures(
        &self,
        worklist: &mut ThreadWorklist,
        state: StateId,
        flags: u32,
        input: &[u8],
        pos: usize,
        src_worklist: &ThreadWorklist,
        src_thread: usize,
        lookaround_memo: &mut LookaroundMemo,
    ) {
        self.add_thread(worklist, state, flags, input, pos, Some((src_worklist, src_thread)), lookaround_memo);
    }

    /// Checks if position is at a word boundary.
    fn is_word_boundary(&self, input: &[u8], pos: usize) -> bool {
        let prev_word = if pos > 0 {
            Self::is_word_char(input[pos - 1])
        } else {
            false
        };
        let curr_word = if pos < input.len() {
            Self::is_word_char(input[pos])
        } else {
            false
        };
        prev_word != curr_word
    }

    /// Checks if a byte is a word character.
    fn is_word_char(b: u8) -> bool {
        matches!(b, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
    }

    /// Extracts captures from a thread into the result format.
    fn extract_captures(
        &self,
        worklist: &ThreadWorklist,
        thread_idx: usize,
        start: usize,
        end: usize,
    ) -> Option<Vec<Option<(usize, usize)>>> {
        self.extract_captures_at_pos(worklist, thread_idx, start, end)
    }

    /// Extracts captures from a thread into the result format with explicit end position.
    fn extract_captures_at_pos(
        &self,
        worklist: &ThreadWorklist,
        thread_idx: usize,
        start: usize,
        end: usize,
    ) -> Option<Vec<Option<(usize, usize)>>> {
        let capture_count = self.nfa.capture_count as usize;
        let mut result = Vec::with_capacity(capture_count + 1);

        // Group 0: full match
        result.push(Some((start, end)));

        // Other groups
        for group in 1..=capture_count {
            let s = worklist.get_capture(thread_idx, group * 2);
            let e = worklist.get_capture(thread_idx, group * 2 + 1);
            if s >= 0 && e >= 0 {
                result.push(Some((s as usize, e as usize)));
            } else {
                result.push(None);
            }
        }

        Some(result)
    }
}
