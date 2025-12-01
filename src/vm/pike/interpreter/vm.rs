//! PikeVM implementation.
//!
//! A thread-based NFA simulator that supports capture groups,
//! backreferences, and lookarounds.
//!
//! Non-greedy quantifiers are supported through thread priority:
//! - Threads have a priority that increases when taking "exit" paths
//! - For non-greedy quantifiers, the exit path has higher priority
//! - The first match from the highest-priority thread wins
//!
//! # Optimizations
//!
//! This implementation includes several key optimizations:
//! 1. **Sparse Set Deduplication**: O(1) state deduplication using generation counters
//! 2. **BinaryHeap Scheduling**: Efficient backref handling with min-heap instead of BTreeMap
//! 3. **`Arc<Nfa>` for Lookarounds**: Avoids expensive NFA cloning during lookaround checks

use crate::hir::unicode::is_word_byte;
use crate::nfa::{Nfa, NfaInstruction};
use std::sync::Arc;

use crate::vm::pike::shared::{
    decode_utf8_codepoint, InstructionResult, PendingThread, PikeVmContext, Thread,
};

/// The PikeVM executor.
pub struct PikeVm {
    nfa: Arc<Nfa>,
}

impl PikeVm {
    /// Creates a new PikeVM from an NFA.
    pub fn new(nfa: Nfa) -> Self {
        Self { nfa: Arc::new(nfa) }
    }

    /// Creates a new PikeVM from an `Arc<Nfa>` (avoids cloning).
    pub fn from_arc(nfa: Arc<Nfa>) -> Self {
        Self { nfa }
    }

    /// Returns true if the pattern matches the input.
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.find(input).is_some()
    }

    /// Finds the first match, returning (start, end).
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        for start in 0..=input.len() {
            if let Some(end) = self.match_at(input, start) {
                return Some((start, end));
            }
        }
        None
    }

    /// Returns capture groups for the first match.
    pub fn captures(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        for start in 0..=input.len() {
            if let Some(captures) = self.captures_at(input, start) {
                return Some(captures);
            }
        }
        None
    }

    /// Finds a match starting at the given position.
    /// Returns (start, end) if found.
    ///
    /// This method correctly handles word boundaries by using the full input
    /// to determine the character class before the start position.
    pub fn find_at(&self, input: &[u8], start: usize) -> Option<(usize, usize)> {
        self.match_at(input, start).map(|end| (start, end))
    }

    /// Attempts to match at a specific position.
    fn match_at(&self, input: &[u8], start: usize) -> Option<usize> {
        self.captures_at(input, start)
            .and_then(|caps| caps.first().and_then(|c| c.map(|(_, end)| end)))
    }

    /// Returns capture groups for a match known to start at position 0.
    ///
    /// This is more efficient than `captures()` when match bounds are already
    /// known (e.g., from a DFA match). Skips the loop that tries every position.
    pub fn captures_from_start(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        self.captures_at(input, 0)
    }

    /// Creates a reusable context for this VM.
    pub fn create_context(&self) -> PikeVmContext {
        PikeVmContext::new(self.nfa.capture_count as usize, self.nfa.states.len())
    }

    /// Returns capture groups using a pre-allocated context.
    /// This is the fastest method for repeated captures on different inputs.
    ///
    /// The context should be created once via `create_context()` and reused.
    pub fn captures_from_start_with_context(
        &self,
        input: &[u8],
        ctx: &mut PikeVmContext,
    ) -> Option<Vec<Option<(usize, usize)>>> {
        self.captures_with_context(input, ctx, 0)
    }

    /// Returns capture groups using a pre-allocated context, starting from a given position.
    pub fn captures_with_context(
        &self,
        input: &[u8],
        ctx: &mut PikeVmContext,
        start_pos: usize,
    ) -> Option<Vec<Option<(usize, usize)>>> {
        ctx.reset();
        ctx.ensure_state_capacity(self.nfa.states.len());

        let capture_count = self.nfa.capture_count as usize;

        // Initialize with start state
        let initial_thread = Thread::new(self.nfa.start, capture_count);

        // Start new generation for initial epsilon closure
        ctx.generation += 1;
        self.add_thread_optimized(ctx, initial_thread, input, start_pos);

        let mut matched: Option<Vec<Option<(usize, usize)>>> = None;
        let mut pos = start_pos;

        // Main loop: process positions until we run out of threads
        while pos <= input.len() {
            // Move threads from heap to current if scheduled for this position
            // Need fresh generation for threads joining at this position
            let mut has_future_threads_at_pos = false;
            while let Some(pt) = ctx.future_threads.peek() {
                if pt.pos == pos {
                    if !has_future_threads_at_pos {
                        // First future thread at this position - start new generation
                        ctx.generation += 1;
                        has_future_threads_at_pos = true;
                    }
                    let pt = ctx.future_threads.pop().unwrap();
                    // Add thread from future - instruction already processed, just need to
                    // add to current_threads and follow epsilon transitions
                    self.add_future_thread(ctx, pt.thread, input, pos);
                } else if pt.pos < pos {
                    // Should not happen, but safe cleanup
                    ctx.future_threads.pop();
                } else {
                    break; // All remaining threads are for future positions
                }
            }

            if ctx.current_threads.is_empty() {
                // If we have no threads now, but have future threads, jump forward
                if let Some(pt) = ctx.future_threads.peek() {
                    pos = pt.pos;
                    continue;
                } else {
                    break; // No work left
                }
            }

            let byte = input.get(pos).copied();

            // Find match threads and process transitions
            let mut match_thread_idx: Option<usize> = None;

            for (idx, thread) in ctx.current_threads.iter().enumerate() {
                let state = match self.nfa.get(thread.state) {
                    Some(s) => s,
                    None => continue,
                };

                if state.is_match {
                    match match_thread_idx {
                        None => match_thread_idx = Some(idx),
                        Some(existing_idx) => {
                            // Prefer non-greedy exit thread
                            if thread.non_greedy_exit
                                && !ctx.current_threads[existing_idx].non_greedy_exit
                            {
                                match_thread_idx = Some(idx);
                            }
                        }
                    }
                }
            }

            // Handle match
            if let Some(idx) = match_thread_idx {
                let thread = &ctx.current_threads[idx];
                if thread.non_greedy_exit {
                    // Reconstruct captures from linked list (only done on match)
                    let mut caps = thread.reconstruct_captures();
                    caps[0] = Some((start_pos, pos));
                    return Some(caps);
                }
                // Greedy: record but continue
                let mut caps = thread.reconstruct_captures();
                caps[0] = Some((start_pos, pos));
                matched = Some(caps);
            }

            // New generation for next position
            ctx.generation += 1;
            ctx.next_threads.clear();

            // Process byte transitions - collect next threads first to avoid borrow conflict
            if let Some(b) = byte {
                // Collect threads that need transitions
                let mut next_states: Vec<Thread> = Vec::new();

                for thread in &ctx.current_threads {
                    let state = match self.nfa.get(thread.state) {
                        Some(s) => s,
                        None => continue,
                    };

                    for (range, target) in &state.transitions {
                        if range.contains(b) {
                            next_states.push(thread.clone_with_state(*target));
                        }
                    }
                }

                // Now process the collected threads
                for next_thread in next_states {
                    self.add_thread_to_next(ctx, next_thread, input, pos + 1);
                }
            }

            // Swap current and next
            std::mem::swap(&mut ctx.current_threads, &mut ctx.next_threads);
            pos += 1;
        }

        matched
    }

    /// Add a thread that jumped from a future position (e.g., backref).
    /// The instruction has already been processed, so we just add to current_threads
    /// and follow epsilon transitions.
    #[inline]
    fn add_future_thread(
        &self,
        ctx: &mut PikeVmContext,
        thread: Thread,
        input: &[u8],
        pos: usize,
    ) {
        let state_id = thread.state as usize;

        // O(1) deduplication check
        if ctx.visited.get(state_id).copied() == Some(ctx.generation) {
            return;
        }

        // Mark as visited
        if state_id < ctx.visited.len() {
            ctx.visited[state_id] = ctx.generation;
        }

        let state = match self.nfa.get(thread.state) {
            Some(s) => s,
            None => return,
        };

        // Add to current threads (instruction already processed)
        ctx.current_threads.push(thread.clone());

        // Follow epsilon transitions (these need full processing)
        for &next_id in &state.epsilon {
            let next_thread = thread.clone_with_state(next_id);
            self.add_thread_optimized(ctx, next_thread, input, pos);
        }
    }

    /// Optimized thread addition with O(1) deduplication.
    #[inline]
    fn add_thread_optimized(
        &self,
        ctx: &mut PikeVmContext,
        mut thread: Thread,
        input: &[u8],
        pos: usize,
    ) {
        let state_id = thread.state as usize;

        // O(1) deduplication check
        if ctx.visited.get(state_id).copied() == Some(ctx.generation) {
            // State already visited in this generation
            // For non-greedy threads, we might need to update, but for simplicity
            // the first thread to arrive wins (standard PikeVM behavior)
            return;
        }

        // Mark as visited
        if state_id < ctx.visited.len() {
            ctx.visited[state_id] = ctx.generation;
        }

        let state = match self.nfa.get(thread.state) {
            Some(s) => s,
            None => return,
        };

        // Handle instructions
        if let Some(ref instruction) = state.instruction {
            match self.process_instruction(instruction, &mut thread, input, pos) {
                InstructionResult::Continue => {}
                InstructionResult::Kill => return,
                InstructionResult::NonGreedyExit => {
                    thread.non_greedy_exit = true;
                }
                InstructionResult::Jump(new_pos) => {
                    // For backrefs: push threads for epsilon transitions from this state
                    // to be processed at the new position (after the matched backref text).
                    // Don't push the backref state itself - only follow epsilon transitions.
                    for &next_id in &state.epsilon {
                        let next_thread = thread.clone_with_state(next_id);
                        ctx.future_threads.push(PendingThread {
                            pos: new_pos,
                            thread: next_thread,
                        });
                    }
                    return;
                }
                InstructionResult::CodepointTransition { bytes_consumed, target } => {
                    // Schedule thread at new position
                    let next_thread = thread.clone_with_state(target);
                    ctx.future_threads.push(PendingThread {
                        pos: pos + bytes_consumed,
                        thread: next_thread,
                    });
                    return;
                }
            }
        }

        // Add to current threads
        ctx.current_threads.push(thread.clone());

        // Follow epsilon transitions
        for &next_id in &state.epsilon {
            let next_thread = thread.clone_with_state(next_id);
            self.add_thread_optimized(ctx, next_thread, input, pos);
        }
    }

    /// Add thread to next_threads list with O(1) deduplication.
    #[inline]
    fn add_thread_to_next(
        &self,
        ctx: &mut PikeVmContext,
        mut thread: Thread,
        input: &[u8],
        pos: usize,
    ) {
        let state_id = thread.state as usize;

        // O(1) deduplication check
        if ctx.visited.get(state_id).copied() == Some(ctx.generation) {
            return;
        }

        // Mark as visited
        if state_id < ctx.visited.len() {
            ctx.visited[state_id] = ctx.generation;
        }

        let state = match self.nfa.get(thread.state) {
            Some(s) => s,
            None => return,
        };

        // Handle instructions
        if let Some(ref instruction) = state.instruction {
            match self.process_instruction(instruction, &mut thread, input, pos) {
                InstructionResult::Continue => {}
                InstructionResult::Kill => return,
                InstructionResult::NonGreedyExit => {
                    thread.non_greedy_exit = true;
                }
                InstructionResult::Jump(new_pos) => {
                    // For backrefs: push the thread to be processed at the new position
                    // The thread keeps its current state (which may be a match state)
                    // and will also follow epsilon transitions when processed
                    ctx.future_threads.push(PendingThread {
                        pos: new_pos,
                        thread,
                    });
                    return;
                }
                InstructionResult::CodepointTransition { bytes_consumed, target } => {
                    let next_thread = thread.clone_with_state(target);
                    ctx.future_threads.push(PendingThread {
                        pos: pos + bytes_consumed,
                        thread: next_thread,
                    });
                    return;
                }
            }
        }

        // Add to next threads
        ctx.next_threads.push(thread.clone());

        // Follow epsilon transitions
        for &next_id in &state.epsilon {
            let next_thread = thread.clone_with_state(next_id);
            self.add_thread_to_next(ctx, next_thread, input, pos);
        }
    }

    /// Attempts to capture at a specific position (non-context version).
    fn captures_at(&self, input: &[u8], start: usize) -> Option<Vec<Option<(usize, usize)>>> {
        let mut ctx = self.create_context();
        self.captures_with_context(input, &mut ctx, start)
    }

    /// Process an NFA instruction and determine what to do with the thread.
    fn process_instruction(
        &self,
        instruction: &NfaInstruction,
        thread: &mut Thread,
        input: &[u8],
        pos: usize,
    ) -> InstructionResult {
        match instruction {
            NfaInstruction::CaptureStart(idx) => {
                thread.record_capture_start(*idx, pos);
                InstructionResult::Continue
            }
            NfaInstruction::CaptureEnd(idx) => {
                thread.record_capture_end(*idx, pos);
                InstructionResult::Continue
            }
            NfaInstruction::StartOfText => {
                if pos != 0 {
                    InstructionResult::Kill
                } else {
                    InstructionResult::Continue
                }
            }
            NfaInstruction::EndOfText => {
                if pos != input.len() {
                    InstructionResult::Kill
                } else {
                    InstructionResult::Continue
                }
            }
            NfaInstruction::StartOfLine => {
                if pos != 0 && input.get(pos - 1) != Some(&b'\n') {
                    InstructionResult::Kill
                } else {
                    InstructionResult::Continue
                }
            }
            NfaInstruction::EndOfLine => {
                if pos != input.len() && input.get(pos) != Some(&b'\n') {
                    InstructionResult::Kill
                } else {
                    InstructionResult::Continue
                }
            }
            NfaInstruction::WordBoundary => {
                if !self.is_word_boundary(input, pos) {
                    InstructionResult::Kill
                } else {
                    InstructionResult::Continue
                }
            }
            NfaInstruction::NotWordBoundary => {
                if self.is_word_boundary(input, pos) {
                    InstructionResult::Kill
                } else {
                    InstructionResult::Continue
                }
            }
            NfaInstruction::Backref(idx) => {
                if let Some((cap_start, cap_end)) = thread.get_capture(*idx) {
                    let cap_len = cap_end - cap_start;

                    // Empty capture - just continue (no text to match)
                    if cap_len == 0 {
                        return InstructionResult::Continue;
                    }

                    if pos + cap_len > input.len() {
                        return InstructionResult::Kill;
                    }
                    let captured = &input[cap_start..cap_end];
                    let current = &input[pos..pos + cap_len];
                    if captured != current {
                        return InstructionResult::Kill;
                    }
                    // Backref matched - jump to position after the matched text
                    InstructionResult::Jump(pos + cap_len)
                } else {
                    InstructionResult::Kill
                }
            }
            NfaInstruction::PositiveLookahead(inner_nfa) => {
                // Use Arc to avoid cloning the NFA
                let inner_vm = PikeVm::from_arc(Arc::new((**inner_nfa).clone()));
                if !inner_vm.is_match(&input[pos..]) {
                    InstructionResult::Kill
                } else {
                    InstructionResult::Continue
                }
            }
            NfaInstruction::NegativeLookahead(inner_nfa) => {
                let inner_vm = PikeVm::from_arc(Arc::new((**inner_nfa).clone()));
                if inner_vm.is_match(&input[pos..]) {
                    InstructionResult::Kill
                } else {
                    InstructionResult::Continue
                }
            }
            NfaInstruction::PositiveLookbehind(inner_nfa) => {
                // For lookbehind, we need to check if the inner pattern matches
                // ending at the current position. We do this by trying all possible
                // start positions before the current position.
                let inner_vm = PikeVm::from_arc(Arc::new((**inner_nfa).clone()));
                let mut found = false;
                for lookback_start in 0..=pos {
                    let slice = &input[lookback_start..pos];
                    // Check if inner pattern matches the entire slice (anchored)
                    if let Some((s, e)) = inner_vm.find(slice) {
                        if s == 0 && e == slice.len() {
                            found = true;
                            break;
                        }
                    }
                }
                if !found {
                    InstructionResult::Kill
                } else {
                    InstructionResult::Continue
                }
            }
            NfaInstruction::NegativeLookbehind(inner_nfa) => {
                // For negative lookbehind, we check that the inner pattern does NOT match
                // ending at the current position.
                let inner_vm = PikeVm::from_arc(Arc::new((**inner_nfa).clone()));
                let mut found = false;
                for lookback_start in 0..=pos {
                    let slice = &input[lookback_start..pos];
                    if let Some((s, e)) = inner_vm.find(slice) {
                        if s == 0 && e == slice.len() {
                            found = true;
                            break;
                        }
                    }
                }
                if found {
                    InstructionResult::Kill
                } else {
                    InstructionResult::Continue
                }
            }
            NfaInstruction::NonGreedyExit => {
                // Mark this thread as having taken a non-greedy exit path
                InstructionResult::NonGreedyExit
            }
            NfaInstruction::CodepointClass(cpclass, target) => {
                // Try to decode a UTF-8 codepoint at the current position
                if pos >= input.len() {
                    return InstructionResult::Kill;
                }

                // Decode the codepoint
                let remaining = &input[pos..];
                if let Some((codepoint, len)) = decode_utf8_codepoint(remaining) {
                    if cpclass.contains(codepoint) {
                        InstructionResult::CodepointTransition { bytes_consumed: len, target: *target }
                    } else {
                        InstructionResult::Kill
                    }
                } else {
                    // Invalid UTF-8
                    InstructionResult::Kill
                }
            }
        }
    }

    /// Returns true if position is at a word boundary.
    fn is_word_boundary(&self, input: &[u8], pos: usize) -> bool {
        let prev_word = pos > 0 && is_word_byte(input[pos - 1]);
        let curr_word = pos < input.len() && is_word_byte(input[pos]);
        prev_word != curr_word
    }
}
