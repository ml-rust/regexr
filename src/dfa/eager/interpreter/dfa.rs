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

use std::collections::VecDeque;

use super::super::super::lazy::{scan_budget, CharClass, LazyDfa};
use super::super::shared::{
    is_word_byte, EagerMaterializationBudgetExceeded, EagerScanBudgetExceeded, StateMetadata,
    DEAD_STATE, MATERIALIZATION_WORK_BUDGET, STATE_MASK, TAG_DEAD, TAG_MATCH,
};
use crate::hash::FxHashMap;

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
    /// The most bytes a match can span, when bounded; sizes the scan budget
    /// (see [`scan_budget`]).
    max_match_len: Option<usize>,
}

impl EagerDfa {
    /// Creates an EagerDfa by materializing all states from a LazyDfa.
    ///
    /// Declines with [`EagerMaterializationBudgetExceeded`] if the BFS's
    /// cumulative work crosses `MATERIALIZATION_WORK_BUDGET` before
    /// finishing — the caller should fall back to `LazyDfa`,
    /// which computes the identical states/transitions on demand instead of
    /// paying for all of them upfront.
    pub fn from_lazy(lazy: &mut LazyDfa) -> Result<Self, EagerMaterializationBudgetExceeded> {
        // Disable cache flushing during materialization to prevent state loss.
        // When LazyDfa flushes its cache, state IDs become invalid, which would
        // corrupt the state mapping we're building during BFS.
        lazy.set_cache_limit(usize::MAX);

        let has_word_boundary = lazy.has_word_boundary();
        let has_anchors = lazy.has_anchors();
        let has_start_anchor = lazy.has_start_anchor();
        let has_end_anchor = lazy.has_end_anchor();
        let has_multiline_anchors = lazy.has_multiline_anchors();
        let max_match_len = lazy.nfa().max_match_len;
        let needs_metadata = has_word_boundary || has_anchors;

        // Get start states
        let lazy_start = lazy.get_start_state_for_class(CharClass::NonWord);
        let lazy_start_word = if has_word_boundary {
            Some(lazy.get_start_state_for_class(CharClass::Word))
        } else {
            None
        };

        // Map from lazy state ID to eager state index
        let mut state_map: FxHashMap<u32, u32> = FxHashMap::default();
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

        // BFS to materialize all reachable states, metered by cumulative
        // NFA-subset size (see MATERIALIZATION_WORK_BUDGET) so a state whose
        // transition computation would be expensive gets charged for it
        // before that cost is paid, not after.
        let mut work_budget_used: usize = 0;

        while let Some(lazy_state) = queue.pop_front() {
            let eager_idx = *state_map.get(&lazy_state).unwrap();

            work_budget_used =
                work_budget_used.saturating_add(lazy.get_state_subset_size(lazy_state));
            if work_budget_used > MATERIALIZATION_WORK_BUDGET {
                return Err(EagerMaterializationBudgetExceeded);
            }

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
                    let next_is_match =
                        match_flags.get(next_idx as usize).copied().unwrap_or(false);
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
                        match_without_end_assertion: lazy
                            .get_state_match_without_end_assertion(lazy_state),
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        // Tag the start state if it's a match state
        let start = if match_flags
            .get(start_idx as usize)
            .copied()
            .unwrap_or(false)
        {
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

        Ok(Self {
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
            max_match_len,
        })
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

    /// Returns whether searches on this DFA take the plain unanchored loop —
    /// the branch of [`EagerDfa::find_from`] that needs neither a start-state
    /// choice nor a per-attempt anchor check.
    ///
    /// This is the precondition of [`EagerDfa::find_from_simple`]. An end
    /// anchor is allowed: it is handled inside the attempt itself.
    #[inline]
    pub fn is_simple_scan(&self) -> bool {
        !self.has_start_anchor && !self.has_word_boundary && !self.has_multiline_anchors
    }

    /// Finds the first match in the input, returning (start, end).
    ///
    /// `Err` means the unanchored search gave up on its scan budget; see
    /// [`EagerDfa::find_from`].
    pub fn find(&self, input: &[u8]) -> Result<Option<(usize, usize)>, EagerScanBudgetExceeded> {
        self.find_from(input, 0)
    }

    /// Finds the leftmost match starting at or after `from`.
    ///
    /// Attempts see the whole input (see [`EagerDfa::find_at`]), so the start
    /// state is picked from the real preceding byte rather than from a slice
    /// boundary.
    ///
    /// The loops below try one start position at a time, which is right while
    /// a failed attempt gives up near where it began, and quadratic when it
    /// does not: `(?s)a.*b` over a run of `a` scans to the end from every
    /// start. So every loop is metered. Once the attempts have collectively
    /// walked past [`scan_budget`] — a few times the input, or the longest
    /// match times the input when that is bounded — `Err` hands the search to
    /// a caller that runs the linear-time search instead (see
    /// [`EagerScanBudgetExceeded`]).
    pub fn find_from(
        &self,
        input: &[u8],
        from: usize,
    ) -> Result<Option<(usize, usize)>, EagerScanBudgetExceeded> {
        if from > input.len() {
            return Ok(None);
        }

        if self.has_start_anchor {
            if self.has_multiline_anchors {
                // Multiline: try position 0 and after each newline
                if from == 0 {
                    if let Some(end) = self.find_at(input, 0) {
                        return Ok(Some((0, end)));
                    }
                }
                // Line starts at or after `from`: the newline itself may sit just
                // before it, hence `from - 1`.
                let mut meter = ScanMeter::new(input, from, self.max_match_len);
                for (i, &byte) in input.iter().enumerate().skip(from.saturating_sub(1)) {
                    if byte == b'\n' {
                        let line_start = i + 1;
                        let (result, reach) = self.find_at_with_reach(input, line_start);
                        if let Some(end) = result {
                            return Ok(Some((line_start, end)));
                        }
                        meter.charge(line_start, reach)?;
                    }
                }
                Ok(None)
            } else if from == 0 {
                // Non-multiline: only try position 0
                Ok(self.find_at(input, 0).map(|end| (0, end)))
            } else {
                Ok(None)
            }
        } else {
            let mut meter = ScanMeter::new(input, from, self.max_match_len);
            for start_pos in from..=input.len() {
                let (result, reach) = self.find_at_with_reach(input, start_pos);
                if let Some(end) = result {
                    return Ok(Some((start_pos, end)));
                }
                meter.charge(start_pos, reach)?;
            }
            Ok(None)
        }
    }

    /// [`EagerDfa::find_from`]'s unanchored branch, specialized.
    ///
    /// Only valid when [`EagerDfa::is_simple_scan`] holds, which is exactly the
    /// last branch above: no start anchor to re-check per attempt and no word
    /// boundary, so `start_state_at` is `self.start` unconditionally and every
    /// attempt is `find_at_fast`. It is metered the same way.
    ///
    /// The caller guarantees `from <= input.len()`.
    #[inline]
    pub fn find_from_simple(
        &self,
        input: &[u8],
        from: usize,
    ) -> Result<Option<(usize, usize)>, EagerScanBudgetExceeded> {
        let mut meter = ScanMeter::new(input, from, self.max_match_len);
        for start_pos in from..=input.len() {
            let (result, reach) = self.find_at_fast_with_reach(input, start_pos, self.start);
            if let Some(end) = result {
                return Ok(Some((start_pos, end)));
            }
            meter.charge(start_pos, reach)?;
        }
        Ok(None)
    }

    /// Picks the start state for an attempt beginning at `start`, from the
    /// character class of the preceding byte.
    #[inline]
    fn start_state_at(&self, input: &[u8], start: usize) -> u32 {
        if self.has_word_boundary && start > 0 {
            let prev_byte = input[start - 1];
            if is_word_byte(prev_byte) {
                self.start_word.unwrap_or(self.start)
            } else {
                self.start
            }
        } else {
            self.start
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

        let state = self.start_state_at(input, start);

        // Use fast path for simple patterns (no word boundary, simple anchors)
        if !self.has_word_boundary && !self.has_multiline_anchors {
            return self.find_at_fast(input, start, state);
        }

        self.find_at_slow(input, start, state)
    }

    /// [`EagerDfa::find_at`], also reporting how far the scan reached, for
    /// callers that meter attempts at many positions.
    pub(crate) fn find_at_reach(&self, input: &[u8], start: usize) -> (Option<usize>, usize) {
        if start > input.len()
            || (self.has_start_anchor && !self.has_multiline_anchors && start != 0)
        {
            return (None, start);
        }
        self.find_at_with_reach(input, start)
    }

    /// [`EagerDfa::find_at`] that also reports how far the scan reached
    /// before giving up, for the metered start-position loops in
    /// [`EagerDfa::find_from`].
    ///
    /// Mirrors `find_at`'s dispatch without its start-anchor short-circuit:
    /// the loops only try positions where a start anchor can hold.
    #[inline]
    fn find_at_with_reach(&self, input: &[u8], start: usize) -> (Option<usize>, usize) {
        let state = self.start_state_at(input, start);
        if !self.has_word_boundary && !self.has_multiline_anchors {
            return self.find_at_fast_with_reach(input, start, state);
        }
        self.find_at_slow_with_reach(input, start, state)
    }

    /// Fast matching loop for patterns without complex assertions.
    #[inline(never)]
    fn find_at_fast(&self, input: &[u8], start: usize, state: u32) -> Option<usize> {
        self.find_at_fast_with_reach(input, start, state).0
    }

    /// [`EagerDfa::find_at_fast`] that also reports the offset the scan
    /// reached: the dead transition it stopped on, or the end of the input.
    #[inline(always)]
    fn find_at_fast_with_reach(
        &self,
        input: &[u8],
        start: usize,
        mut state: u32,
    ) -> (Option<usize>, usize) {
        // Kept for the hand-off below: the loop overwrites `state`.
        let start_state = state;
        let mut last_match = if state & TAG_MATCH != 0 {
            Some(start)
        } else {
            None
        };

        let bytes = &input[start..];
        let mut reach = input.len();

        for (i, &byte) in bytes.iter().enumerate() {
            let state_idx = (state & STATE_MASK) as usize;
            let next = self.transitions[state_idx * 256 + byte as usize];

            if next & TAG_DEAD != 0 {
                reach = start + i;
                break;
            }

            state = next;

            if state & TAG_MATCH != 0 {
                last_match = Some(start + i + 1);
            }
        }

        // For patterns with end anchor (`$`/`\Z`), verify the match ends at end of
        // input or just before a final newline (PCRE/Python semantics).
        if let Some(end_pos) = last_match {
            if self.has_end_anchor && !crate::nfa::at_end_or_before_final_newline(input, end_pos) {
                // See the lazy DFA: `$` belongs to the branch carrying it, so a
                // greedy end that fails it does not condemn the start position.
                return self.find_at_slow_with_reach(input, start, start_state);
            }
        }

        (last_match, reach)
    }

    /// Slow matching loop for patterns with word boundaries or multiline anchors.
    fn find_at_slow(&self, input: &[u8], start: usize, state: u32) -> Option<usize> {
        self.find_at_slow_with_reach(input, start, state).0
    }

    /// [`EagerDfa::find_at_slow`] that also reports the byte offset the scan
    /// reached before stopping — the position of the dead transition it broke
    /// on, or `input.len()` if it ran to the end.
    ///
    /// Reported once per call, at the exit point, not per byte: the metered
    /// loop in `find_from` only needs to know how far *this* attempt got, not
    /// a running per-byte count, so nothing is added to the loop body below.
    fn find_at_slow_with_reach(
        &self,
        input: &[u8],
        start: usize,
        mut state: u32,
    ) -> (Option<usize>, usize) {
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
                return (last_match, start + i);
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

        (last_match, input.len())
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
            // The anchor belongs to the branch carrying it; if some other branch
            // reaches a match without one, the match stands wherever it ends.
            if metadata.match_without_end_assertion {
                return true;
            }

            if metadata.needs_end_of_text && !crate::nfa::at_end_or_before_final_newline(input, pos)
            {
                return false;
            }

            if metadata.needs_end_of_line && !crate::nfa::at_line_end(input, pos) {
                return false;
            }
        }

        true
    }
}

/// Walked-byte accounting for the start-position loops in
/// [`EagerDfa::find_from`].
///
/// The budget covers the input from the search's first start onward (see
/// [`scan_budget`]), so a search that gives up has already spent work
/// proportional to what the linear-time search it hands over to will cost.
struct ScanMeter {
    budget: usize,
    walked: usize,
}

impl ScanMeter {
    fn new(input: &[u8], from: usize, max_match_len: Option<usize>) -> Self {
        Self {
            budget: scan_budget(input.len(), from, max_match_len),
            walked: 0,
        }
    }

    /// Records a failed attempt at `start` that reached `reach`.
    #[inline(always)]
    fn charge(&mut self, start: usize, reach: usize) -> Result<(), EagerScanBudgetExceeded> {
        self.walked += reach.saturating_sub(start);
        if self.walked > self.budget {
            Err(EagerScanBudgetExceeded)
        } else {
            Ok(())
        }
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
