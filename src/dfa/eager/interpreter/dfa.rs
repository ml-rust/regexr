//! Eager (pre-materialized) DFA implementation.
//!
//! Unlike LazyDfa which computes states on-demand, EagerDfa pre-computes
//! all reachable states upfront for faster matching. This trades compilation
//! time for matching speed.
//!
//! ## Performance
//!
//! EagerDfa uses a flat transition table where each state has 256 entries.
//! This enables O(1) transition lookups without hash map overhead.
//!
//! The matching loop is also simpler since no state computation is needed:
//! ```text
//! state = transitions[state * 256 + byte]
//! if state & MATCH_FLAG: record match
//! if state == DEAD: stop
//! ```

use std::collections::{HashMap, VecDeque};

use super::super::super::lazy::{CharClass, LazyDfa};
use super::super::shared::{
    is_word_byte, StateMetadata, DEAD_STATE, STATE_MASK, TAG_DEAD, TAG_MATCH,
};

/// A pre-materialized DFA with flat transition table.
///
/// All states and transitions are computed upfront, enabling
/// fast O(1) transition lookups during matching.
pub struct EagerDfa {
    /// Flat transition table: transitions[state_idx * 256 + byte] = next_state
    /// Uses tagged state IDs (high bits encode match/dead status)
    transitions: Vec<u32>,
    /// Number of states
    state_count: usize,
    /// Start state index (for NonWord context)
    start: u32,
    /// Start state index (for Word context), if pattern has word boundaries
    start_word: Option<u32>,
    /// Whether pattern has word boundaries
    has_word_boundary: bool,
    /// Whether pattern has anchors
    has_anchors: bool,
    /// Whether pattern has start anchor (^)
    has_start_anchor: bool,
    /// Whether pattern has end anchor ($)
    has_end_anchor: bool,
    /// Whether pattern has multiline anchors
    has_multiline_anchors: bool,
    /// Per-state metadata for end assertion checking.
    /// Only populated when has_word_boundary or has_anchors is true.
    state_metadata: Vec<StateMetadata>,
}

impl EagerDfa {
    /// Creates an EagerDfa by materializing all states from a LazyDfa.
    pub fn from_lazy(lazy: &mut LazyDfa) -> Self {
        // Disable cache flushing during materialization to prevent state loss.
        // When LazyDfa flushes its cache, state IDs become invalid, which would
        // corrupt the state mapping we're building during BFS.
        lazy.set_cache_limit(usize::MAX);

        let has_word_boundary = lazy.has_word_boundary();
        let has_anchors = lazy.has_anchors();
        let has_start_anchor = lazy.has_start_anchor();
        let has_end_anchor = lazy.has_end_anchor();
        let has_multiline_anchors = lazy.has_multiline_anchors();
        let needs_metadata = has_word_boundary || has_anchors;

        // Get start states
        let lazy_start = lazy.get_start_state_for_class(CharClass::NonWord);
        let lazy_start_word = if has_word_boundary {
            Some(lazy.get_start_state_for_class(CharClass::Word))
        } else {
            None
        };

        // Map from lazy state ID to eager state index
        let mut state_map: HashMap<u32, u32> = HashMap::new();
        let mut queue: VecDeque<u32> = VecDeque::new();

        // Temporary storage for transitions before we know final state count
        let mut all_transitions: Vec<[Option<u32>; 256]> = Vec::new();
        let mut match_flags: Vec<bool> = Vec::new();
        // Track lazy state IDs in order for metadata extraction
        let mut lazy_state_order: Vec<u32> = Vec::new();

        // Add start state(s)
        let start_idx = 0u32;
        state_map.insert(lazy_start, start_idx);
        queue.push_back(lazy_start);

        let start_word_idx = if let Some(sw) = lazy_start_word {
            if sw != lazy_start {
                let idx = 1u32;
                state_map.insert(sw, idx);
                queue.push_back(sw);
                Some(idx)
            } else {
                Some(start_idx)
            }
        } else {
            None
        };

        // BFS to materialize all reachable states
        while let Some(lazy_state) = queue.pop_front() {
            let eager_idx = *state_map.get(&lazy_state).unwrap();

            // Ensure we have space for this state
            while all_transitions.len() <= eager_idx as usize {
                all_transitions.push([None; 256]);
                match_flags.push(false);
                lazy_state_order.push(0); // Placeholder
            }

            // Record the lazy state ID for this index
            lazy_state_order[eager_idx as usize] = lazy_state;

            // Record if this is a match state
            match_flags[eager_idx as usize] = lazy.is_match(lazy_state);

            // Compute all 256 transitions
            let lazy_transitions = lazy.compute_all_transitions(lazy_state);

            for byte in 0..=255u8 {
                if let Some(next_lazy) = lazy_transitions[byte as usize] {
                    // Get or create eager index for the target state
                    let next_idx = if let Some(&idx) = state_map.get(&next_lazy) {
                        idx
                    } else {
                        let idx = state_map.len() as u32;
                        state_map.insert(next_lazy, idx);
                        queue.push_back(next_lazy);
                        idx
                    };
                    all_transitions[eager_idx as usize][byte as usize] = Some(next_idx);
                }
            }
        }

        let state_count = all_transitions.len();

        // Build flat transition table with tagged states
        let mut transitions = vec![DEAD_STATE; state_count * 256];

        for (state_idx, trans) in all_transitions.iter().enumerate() {
            let base = state_idx * 256;

            for byte in 0..256 {
                if let Some(next_idx) = trans[byte] {
                    let next_is_match = match_flags.get(next_idx as usize).copied().unwrap_or(false);
                    let mut tagged = next_idx;
                    if next_is_match {
                        tagged |= TAG_MATCH;
                    }
                    transitions[base + byte] = tagged;
                }
            }
        }

        // Build per-state metadata for end assertion checking
        let state_metadata = if needs_metadata {
            lazy_state_order
                .iter()
                .map(|&lazy_state| {
                    let prev_class = lazy.get_state_prev_class(lazy_state);
                    let (needs_word_boundary, needs_not_word_boundary) =
                        lazy.get_state_boundary_requirements(lazy_state);
                    let (needs_end_of_text, needs_end_of_line) =
                        lazy.get_state_anchor_requirements(lazy_state);
                    StateMetadata {
                        prev_class,
                        needs_word_boundary,
                        needs_not_word_boundary,
                        needs_end_of_text,
                        needs_end_of_line,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        // Tag the start state if it's a match state
        let start = if match_flags.get(start_idx as usize).copied().unwrap_or(false) {
            start_idx | TAG_MATCH
        } else {
            start_idx
        };

        let start_word = start_word_idx.map(|idx| {
            if match_flags.get(idx as usize).copied().unwrap_or(false) {
                idx | TAG_MATCH
            } else {
                idx
            }
        });

        Self {
            transitions,
            state_count,
            start,
            start_word,
            has_word_boundary,
            has_anchors,
            has_start_anchor,
            has_end_anchor,
            has_multiline_anchors,
            state_metadata,
        }
    }

    /// Returns the number of DFA states.
    #[inline]
    pub fn state_count(&self) -> usize {
        self.state_count
    }

    /// Returns whether this DFA has word boundary assertions.
    pub fn has_word_boundary(&self) -> bool {
        self.has_word_boundary
    }

    /// Returns whether this DFA has anchor assertions.
    pub fn has_anchors(&self) -> bool {
        self.has_anchors
    }

    /// Returns whether this DFA has a start anchor.
    pub fn has_start_anchor(&self) -> bool {
        self.has_start_anchor
    }

    /// Returns whether this DFA has an end anchor.
    pub fn has_end_anchor(&self) -> bool {
        self.has_end_anchor
    }

    /// Returns whether this DFA has multiline anchors.
    pub fn has_multiline_anchors(&self) -> bool {
        self.has_multiline_anchors
    }

    /// Finds the first match in the input, returning (start, end).
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        if self.has_start_anchor {
            if self.has_multiline_anchors {
                // Multiline: try position 0 and after each newline
                if let Some(end) = self.find_at(input, 0) {
                    return Some((0, end));
                }
                for (i, &byte) in input.iter().enumerate() {
                    if byte == b'\n' && i + 1 <= input.len() {
                        if let Some(end) = self.find_at(input, i + 1) {
                            return Some((i + 1, end));
                        }
                    }
                }
                None
            } else {
                // Non-multiline: only try position 0
                self.find_at(input, 0).map(|end| (0, end))
            }
        } else {
            // No start anchor: try starting at each position
            for start_pos in 0..=input.len() {
                if let Some(end) = self.find_at(input, start_pos) {
                    return Some((start_pos, end));
                }
            }
            None
        }
    }

    /// Finds a match starting at the given position.
    #[inline]
    pub fn find_at(&self, input: &[u8], start: usize) -> Option<usize> {
        if start > input.len() {
            return None;
        }

        // For anchored patterns, verify start position is valid
        if self.has_start_anchor && !self.has_multiline_anchors && start != 0 {
            return None;
        }

        // Get appropriate start state
        let state = if self.has_word_boundary && start > 0 {
            let prev_byte = input[start - 1];
            if is_word_byte(prev_byte) {
                self.start_word.unwrap_or(self.start)
            } else {
                self.start
            }
        } else {
            self.start
        };

        // Use fast path for simple patterns (no word boundary, simple anchors)
        if !self.has_word_boundary && !self.has_multiline_anchors {
            return self.find_at_fast(input, start, state);
        }

        self.find_at_slow(input, start, state)
    }

    /// Fast matching loop for patterns without complex assertions.
    #[inline(never)]
    fn find_at_fast(&self, input: &[u8], start: usize, mut state: u32) -> Option<usize> {
        let mut last_match = if state & TAG_MATCH != 0 {
            Some(start)
        } else {
            None
        };

        let bytes = &input[start..];

        for (i, &byte) in bytes.iter().enumerate() {
            let state_idx = (state & STATE_MASK) as usize;
            let next = self.transitions[state_idx * 256 + byte as usize];

            if next & TAG_DEAD != 0 {
                break;
            }

            state = next;

            if state & TAG_MATCH != 0 {
                last_match = Some(start + i + 1);
            }
        }

        // For patterns with end anchor, verify match ends at end of input
        if let Some(end_pos) = last_match {
            if self.has_end_anchor && end_pos != input.len() {
                return None;
            }
        }

        last_match
    }

    /// Slow matching loop for patterns with word boundaries or multiline anchors.
    fn find_at_slow(&self, input: &[u8], start: usize, mut state: u32) -> Option<usize> {
        let mut last_match = if state & TAG_MATCH != 0 {
            if self.check_end_assertions(input, start, (state & STATE_MASK) as usize) {
                Some(start)
            } else {
                None
            }
        } else {
            None
        };

        for (i, &byte) in input[start..].iter().enumerate() {
            let state_idx = (state & STATE_MASK) as usize;
            let next = self.transitions[state_idx * 256 + byte as usize];

            if next & TAG_DEAD != 0 {
                break;
            }

            state = next;

            if state & TAG_MATCH != 0 {
                let match_end = start + i + 1;
                let next_state_idx = (state & STATE_MASK) as usize;
                if self.check_end_assertions(input, match_end, next_state_idx) {
                    last_match = Some(match_end);
                }
            }
        }

        last_match
    }

    /// Checks end assertions (word boundary and anchors) for a match at the given position.
    #[inline]
    fn check_end_assertions(&self, input: &[u8], pos: usize, state_idx: usize) -> bool {
        // Fast path: no assertions to check
        if self.state_metadata.is_empty() {
            return true;
        }

        let metadata = match self.state_metadata.get(state_idx) {
            Some(m) => m,
            None => return true,
        };

        // Check word boundary assertions
        if self.has_word_boundary {
            let prev_class = metadata.prev_class;
            let next_class = if pos < input.len() {
                CharClass::from_byte(input[pos])
            } else {
                CharClass::NonWord
            };

            let is_at_boundary = prev_class != next_class;

            if metadata.needs_word_boundary && !is_at_boundary {
                return false;
            }
            if metadata.needs_not_word_boundary && is_at_boundary {
                return false;
            }
        }

        // Check anchor assertions
        if self.has_anchors {
            if metadata.needs_end_of_text && pos != input.len() {
                return false;
            }

            if metadata.needs_end_of_line {
                let at_end_of_line = pos == input.len() || input.get(pos) == Some(&b'\n');
                if !at_end_of_line {
                    return false;
                }
            }
        }

        true
    }
}

impl std::fmt::Debug for EagerDfa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EagerDfa")
            .field("state_count", &self.state_count)
            .field("has_word_boundary", &self.has_word_boundary)
            .field("has_anchors", &self.has_anchors)
            .finish()
    }
}
