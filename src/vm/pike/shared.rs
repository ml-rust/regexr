//! Shared types for PikeVM execution.
//!
//! Contains thread management structures and utilities used by the interpreter.
//!
//! # Copy-On-Write Captures
//!
//! Thread captures use a linked-list based Copy-On-Write (COW) strategy to avoid
//! expensive Vec cloning on every thread fork. Instead of storing a full Vec of
//! captures in each thread, we store:
//! - A reference-counted pointer to the parent thread's capture history
//! - The specific capture action this thread took (if any)
//!
//! This makes thread creation O(1) instead of O(G) where G is the number of capture groups.
//! The full capture Vec is only reconstructed when a match is found.

use crate::nfa::StateId;
use std::collections::BinaryHeap;
use std::rc::Rc;

/// Thread scheduled for a future position (used for backrefs).
#[derive(Debug)]
pub struct PendingThread {
    pub pos: usize,
    pub thread: Thread,
}

impl PartialEq for PendingThread {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
    }
}

impl Eq for PendingThread {}

impl PartialOrd for PendingThread {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingThread {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse because BinaryHeap is a max-heap, we want min position first
        other.pos.cmp(&self.pos)
    }
}

/// Pre-allocated execution context for PikeVM.
/// Reusing this context across multiple captures() calls avoids repeated allocations.
#[derive(Debug)]
pub struct PikeVmContext {
    /// Thread storage for current position
    pub current_threads: Vec<Thread>,
    /// Thread storage for next position
    pub next_threads: Vec<Thread>,
    /// Threads waiting for future positions (for backrefs)
    pub future_threads: BinaryHeap<PendingThread>,
    /// O(1) deduplication: visited[state_id] == generation means state already visited
    pub visited: Vec<usize>,
    /// Current generation counter (incremented per position/step)
    pub generation: usize,
    /// Capture slot storage (reused across calls)
    #[allow(dead_code)]
    pub capture_slots: Vec<Option<(usize, usize)>>,
    /// Stack for iterative epsilon closure (avoids recursion stack overflow)
    pub epsilon_stack: Vec<Thread>,
}

impl PikeVmContext {
    /// Create a new context for the given capture count and state count.
    pub fn new(capture_count: usize, state_count: usize) -> Self {
        Self {
            current_threads: Vec::with_capacity(32),
            next_threads: Vec::with_capacity(32),
            future_threads: BinaryHeap::new(),
            visited: vec![0; state_count],
            generation: 0,
            capture_slots: vec![None; capture_count + 1],
            epsilon_stack: Vec::with_capacity(32),
        }
    }

    /// Reset the context for a new match attempt.
    #[inline]
    pub fn reset(&mut self) {
        self.current_threads.clear();
        self.next_threads.clear();
        self.future_threads.clear();
        self.epsilon_stack.clear();
        // Don't clear visited or reset generation - the sparse set approach
        // relies on keeping generation incrementing to invalidate old entries.
        // Just increment once to ensure fresh start
        self.generation = self.generation.wrapping_add(1);
        for slot in &mut self.capture_slots {
            *slot = None;
        }
    }

    /// Ensure visited array is large enough for the given state count.
    #[inline]
    pub fn ensure_state_capacity(&mut self, state_count: usize) {
        if self.visited.len() < state_count {
            self.visited.resize(state_count, 0);
        }
    }
}

/// A capture action in the linked list.
/// Each action records a single capture start or end event.
#[derive(Debug, Clone)]
pub enum CaptureAction {
    /// Start of capture group at position
    Start(u32, usize),
    /// End of capture group at position
    End(u32, usize),
}

/// A node in the capture history linked list.
/// Uses Rc for O(1) thread forking - just increment reference count.
#[derive(Debug, Clone)]
pub struct CaptureNode {
    /// The capture action at this node
    pub action: CaptureAction,
    /// Link to parent node (previous capture action)
    pub parent: Option<Rc<CaptureNode>>,
}

impl CaptureNode {
    /// Create a new capture node with the given action and parent.
    #[inline]
    pub fn new(action: CaptureAction, parent: Option<Rc<CaptureNode>>) -> Rc<Self> {
        Rc::new(Self { action, parent })
    }
}

/// A thread in the PikeVM.
///
/// Uses Copy-On-Write (COW) captures via a linked list to avoid expensive
/// Vec cloning on every thread fork. Thread creation is O(1) - just increment
/// an Rc counter. The full capture Vec is reconstructed only when a match is found.
#[derive(Debug, Clone)]
pub struct Thread {
    /// Current NFA state.
    pub state: StateId,
    /// Head of the capture history linked list.
    /// None means no captures have been recorded yet.
    pub capture_head: Option<Rc<CaptureNode>>,
    /// Number of capture groups (needed for reconstruction).
    pub capture_count: usize,
    /// Whether this thread passed through a non-greedy exit.
    /// If true, a match found by this thread should be returned immediately.
    pub non_greedy_exit: bool,
}

impl Thread {
    /// Create a new thread with no capture history.
    #[inline]
    pub fn new(state: StateId, capture_count: usize) -> Self {
        Self {
            state,
            capture_head: None,
            capture_count,
            non_greedy_exit: false,
        }
    }

    /// Clone this thread with a new state. O(1) operation - just increments Rc counter.
    #[inline]
    pub fn clone_with_state(&self, state: StateId) -> Self {
        Self {
            state,
            capture_head: self.capture_head.clone(), // Rc::clone is O(1)
            capture_count: self.capture_count,
            non_greedy_exit: self.non_greedy_exit,
        }
    }

    /// Record a capture start event. O(1) operation.
    #[inline]
    pub fn record_capture_start(&mut self, group_idx: u32, pos: usize) {
        let node = CaptureNode::new(
            CaptureAction::Start(group_idx, pos),
            self.capture_head.take(),
        );
        self.capture_head = Some(node);
    }

    /// Record a capture end event. O(1) operation.
    #[inline]
    pub fn record_capture_end(&mut self, group_idx: u32, pos: usize) {
        let node = CaptureNode::new(
            CaptureAction::End(group_idx, pos),
            self.capture_head.take(),
        );
        self.capture_head = Some(node);
    }

    /// Reconstruct the full capture Vec from the linked list.
    /// Called only when a match is found. O(depth) where depth is number of capture actions.
    pub fn reconstruct_captures(&self) -> Vec<Option<(usize, usize)>> {
        let mut captures = vec![None; self.capture_count + 1];

        // Walk the linked list backwards to collect all actions
        let mut actions = Vec::new();
        let mut current = self.capture_head.as_ref();
        while let Some(node) = current {
            actions.push(&node.action);
            current = node.parent.as_ref();
        }

        // Process actions in reverse order (oldest first) to build final capture state
        // For each group, we want the LAST (most recent) start and end positions
        // But we process oldest-first, so each new value overwrites the old
        for action in actions.into_iter().rev() {
            match action {
                CaptureAction::Start(idx, pos) => {
                    let idx = *idx as usize;
                    if idx < captures.len() {
                        // Start a new capture - set start position, end will be set later
                        captures[idx] = Some((*pos, *pos));
                    }
                }
                CaptureAction::End(idx, pos) => {
                    let idx = *idx as usize;
                    if idx < captures.len() {
                        if let Some((start, _)) = captures[idx] {
                            captures[idx] = Some((start, *pos));
                        }
                    }
                }
            }
        }

        captures
    }

    /// Get a capture group value by walking the linked list.
    /// Used for backref matching - more efficient than full reconstruction.
    /// Returns None if capture group is not set or incomplete.
    pub fn get_capture(&self, group_idx: u32) -> Option<(usize, usize)> {
        let mut start: Option<usize> = None;
        let mut end: Option<usize> = None;

        // Walk backwards to find the most recent start and end for this group
        let mut current = self.capture_head.as_ref();
        while let Some(node) = current {
            match &node.action {
                CaptureAction::Start(idx, pos) if *idx == group_idx => {
                    if start.is_none() {
                        start = Some(*pos);
                    }
                }
                CaptureAction::End(idx, pos) if *idx == group_idx => {
                    if end.is_none() {
                        end = Some(*pos);
                    }
                }
                _ => {}
            }
            // Early exit if we found both
            if start.is_some() && end.is_some() {
                break;
            }
            current = node.parent.as_ref();
        }

        match (start, end) {
            (Some(s), Some(e)) => Some((s, e)),
            _ => None,
        }
    }
}

/// Result of processing an instruction during epsilon closure.
pub enum InstructionResult {
    /// Continue with epsilon transitions at current position
    Continue,
    /// Thread should be killed (assertion failed)
    Kill,
    /// Thread should jump to a different position (for backrefs)
    Jump(usize),
    /// Mark thread as having passed through a non-greedy exit
    NonGreedyExit,
    /// Transition to target state after consuming `bytes_consumed` bytes (for CodepointClass)
    CodepointTransition { bytes_consumed: usize, target: StateId },
}

/// Decodes a single UTF-8 codepoint from a byte slice.
/// Returns the codepoint value and its length in bytes, or None if invalid UTF-8.
#[inline]
pub fn decode_utf8_codepoint(bytes: &[u8]) -> Option<(u32, usize)> {
    if bytes.is_empty() {
        return None;
    }

    let first = bytes[0];
    if first < 0x80 {
        // ASCII: 1 byte
        return Some((first as u32, 1));
    }

    if first < 0xC0 {
        // Invalid: continuation byte as first byte
        return None;
    }

    if first < 0xE0 {
        // 2-byte sequence: 110xxxxx 10xxxxxx
        if bytes.len() < 2 {
            return None;
        }
        let b1 = bytes[1];
        if (b1 & 0xC0) != 0x80 {
            return None;
        }
        let cp = ((first as u32 & 0x1F) << 6) | (b1 as u32 & 0x3F);
        return Some((cp, 2));
    }

    if first < 0xF0 {
        // 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx
        if bytes.len() < 3 {
            return None;
        }
        let b1 = bytes[1];
        let b2 = bytes[2];
        if (b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 {
            return None;
        }
        let cp = ((first as u32 & 0x0F) << 12) | ((b1 as u32 & 0x3F) << 6) | (b2 as u32 & 0x3F);
        return Some((cp, 3));
    }

    if first < 0xF8 {
        // 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        if bytes.len() < 4 {
            return None;
        }
        let b1 = bytes[1];
        let b2 = bytes[2];
        let b3 = bytes[3];
        if (b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80 {
            return None;
        }
        let cp = ((first as u32 & 0x07) << 18)
            | ((b1 as u32 & 0x3F) << 12)
            | ((b2 as u32 & 0x3F) << 6)
            | (b3 as u32 & 0x3F);
        return Some((cp, 4));
    }

    // Invalid: first byte > 0xF7
    None
}
