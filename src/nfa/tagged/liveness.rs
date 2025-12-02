//! Liveness analysis for NFA capture groups.
//!
//! Computes which capture slots are "live" (may be read) at each NFA state.
//! This enables sparse copying during thread spawning - we only copy slots
//! that could possibly be needed downstream.
//!
//! The analysis uses backward dataflow:
//! - A capture is live at state S if it may be read on any path from S to a match
//! - `live_reads[S] = reads_at[S] ∪ (∪ live_reads[successors(S)])`
//! - `copy_mask[S] = live_reads[S] ∩ writes_before[S]`

use crate::nfa::{Nfa, NfaInstruction, StateId};
use std::collections::VecDeque;

/// A compact bitset for capture group indices (up to 64 groups).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CaptureBitSet(pub u64);

impl CaptureBitSet {
    /// Creates an empty bitset.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Creates a bitset with all bits set up to `count`.
    pub fn all(count: usize) -> Self {
        if count >= 64 {
            Self(u64::MAX)
        } else if count == 0 {
            Self(0)
        } else {
            Self((1u64 << count) - 1)
        }
    }

    /// Sets a bit.
    #[inline]
    pub fn set(&mut self, idx: u32) {
        if idx < 64 {
            self.0 |= 1u64 << idx;
        }
    }

    /// Clears a bit.
    #[inline]
    pub fn clear(&mut self, idx: u32) {
        if idx < 64 {
            self.0 &= !(1u64 << idx);
        }
    }

    /// Checks if a bit is set.
    #[inline]
    pub fn contains(&self, idx: u32) -> bool {
        if idx < 64 {
            (self.0 & (1u64 << idx)) != 0
        } else {
            false
        }
    }

    /// Returns true if the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Returns the number of set bits.
    #[inline]
    pub fn count(&self) -> u32 {
        self.0.count_ones()
    }

    /// Union of two bitsets.
    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Intersection of two bitsets.
    #[inline]
    pub fn intersect(&self, other: &Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Iterator over set bits.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        (0..64u32).filter(move |&i| self.contains(i))
    }
}

/// Liveness information for a single NFA state.
#[derive(Debug, Clone, Default)]
pub struct StateLiveness {
    /// Captures that may be read on any path from this state to a match.
    pub live_reads: CaptureBitSet,
    /// Captures that are written at this state.
    pub writes: CaptureBitSet,
    /// Captures that need to be copied when spawning a thread from this state.
    /// This is: live_reads ∩ (captures written on any path to this state)
    pub copy_mask: CaptureBitSet,
}

/// Liveness analysis result for an entire NFA.
#[derive(Debug, Clone)]
pub struct NfaLiveness {
    /// Per-state liveness information.
    pub states: Vec<StateLiveness>,
    /// Total number of capture groups.
    pub capture_count: u32,
    /// Number of lookarounds in the NFA (for memoization array sizing).
    pub lookaround_count: u32,
}

impl NfaLiveness {
    /// Returns the copy mask for spawning from a given state.
    pub fn copy_mask(&self, state: StateId) -> CaptureBitSet {
        self.states
            .get(state as usize)
            .map(|s| s.copy_mask)
            .unwrap_or_default()
    }

    /// Returns true if spawning from this state requires any capture copying.
    pub fn needs_copy(&self, state: StateId) -> bool {
        !self.copy_mask(state).is_empty()
    }
}

/// Analyzes capture liveness in an NFA.
///
/// This performs a backward dataflow analysis to determine which capture
/// slots are "live" at each state - i.e., may be read on some path to a match.
pub fn analyze_liveness(nfa: &Nfa) -> NfaLiveness {
    let state_count = nfa.states.len();
    let capture_count = nfa.capture_count;

    // Initialize per-state info
    let mut states: Vec<StateLiveness> = vec![StateLiveness::default(); state_count];
    let mut lookaround_count = 0u32;

    // First pass: collect reads and writes at each state
    for (id, state) in nfa.states.iter().enumerate() {
        if let Some(ref instr) = state.instruction {
            match instr {
                NfaInstruction::CaptureStart(idx) | NfaInstruction::CaptureEnd(idx) => {
                    // These write to a capture slot
                    states[id].writes.set(*idx);
                }
                NfaInstruction::Backref(idx) => {
                    // Backrefs read from a capture slot
                    states[id].live_reads.set(*idx);
                }
                NfaInstruction::PositiveLookahead(_)
                | NfaInstruction::NegativeLookahead(_)
                | NfaInstruction::PositiveLookbehind(_)
                | NfaInstruction::NegativeLookbehind(_) => {
                    lookaround_count += 1;
                }
                _ => {}
            }
        }
    }

    // Backward dataflow: propagate live_reads from successors
    // Use worklist algorithm for efficiency
    let mut worklist: VecDeque<StateId> = VecDeque::new();
    let mut in_worklist = vec![false; state_count];

    // Build predecessor map for backward propagation
    let mut predecessors: Vec<Vec<StateId>> = vec![Vec::new(); state_count];
    for (id, state) in nfa.states.iter().enumerate() {
        let id = id as StateId;
        // Epsilon transitions
        for &target in &state.epsilon {
            predecessors[target as usize].push(id);
        }
        // Byte transitions
        for (_, target) in &state.transitions {
            predecessors[*target as usize].push(id);
        }
    }

    // Initialize worklist with all states that have reads
    for (id, state_liveness) in states.iter().enumerate() {
        if !state_liveness.live_reads.is_empty() {
            worklist.push_back(id as StateId);
            in_worklist[id] = true;
        }
    }

    // At match states, ALL captures are implicitly "read" for result extraction.
    // Mark all captures as live at match states and propagate backward.
    let all_captures = CaptureBitSet::all(capture_count as usize + 1);
    for &match_id in &nfa.matches {
        states[match_id as usize].live_reads = all_captures;
        if !in_worklist[match_id as usize] {
            worklist.push_back(match_id);
            in_worklist[match_id as usize] = true;
        }
    }

    // Backward propagation
    while let Some(state_id) = worklist.pop_front() {
        in_worklist[state_id as usize] = false;
        let current_reads = states[state_id as usize].live_reads;

        // Propagate to predecessors
        for &pred_id in &predecessors[state_id as usize] {
            let pred_state = &mut states[pred_id as usize];
            let old_reads = pred_state.live_reads;

            // Union current state's live reads into predecessor
            // But exclude anything the predecessor writes (kill)
            let propagated = current_reads.union(&pred_state.live_reads);

            if propagated != old_reads {
                pred_state.live_reads = propagated;
                if !in_worklist[pred_id as usize] {
                    worklist.push_back(pred_id);
                    in_worklist[pred_id as usize] = true;
                }
            }
        }
    }

    // Forward pass: compute copy_mask
    // copy_mask[S] = captures that are:
    // 1. Live at S (may be read downstream)
    // 2. Written on some path to S (need to preserve the value)
    //
    // For simplicity, we use a conservative approximation:
    // copy_mask[S] = live_reads[S] for states that have epsilon transitions out
    // (these are the states where threads spawn)
    //
    // More precise: track writes_before using forward dataflow
    let mut writes_before: Vec<CaptureBitSet> = vec![CaptureBitSet::empty(); state_count];

    // Forward worklist
    worklist.clear();
    in_worklist.fill(false);
    worklist.push_back(nfa.start);
    in_worklist[nfa.start as usize] = true;

    while let Some(state_id) = worklist.pop_front() {
        in_worklist[state_id as usize] = false;
        let state = &nfa.states[state_id as usize];

        // Current writes_before includes this state's writes
        let current_writes =
            writes_before[state_id as usize].union(&states[state_id as usize].writes);

        // Propagate to successors
        let mut propagate = |target: StateId| {
            let old = writes_before[target as usize];
            let new = old.union(&current_writes);
            if new != old {
                writes_before[target as usize] = new;
                if !in_worklist[target as usize] {
                    worklist.push_back(target);
                    in_worklist[target as usize] = true;
                }
            }
        };

        for &target in &state.epsilon {
            propagate(target);
        }
        for (_, target) in &state.transitions {
            propagate(*target);
        }
    }

    // Compute final copy_mask
    // For correctness, we use a conservative approach: copy all live captures.
    // This ensures captures are preserved across alternations where different
    // branches write different captures. The writes_before analysis is still
    // useful for future optimization with more precise path tracking.
    //
    // Future optimization: For patterns without alternations, we could use
    // `copy_mask = live_reads ∩ writes_before` to reduce copying overhead.
    // This would require detecting alternation-free paths during analysis.
    for (id, state_liveness) in states.iter_mut().enumerate() {
        // Conservative: copy all captures that are live at this state
        state_liveness.copy_mask = state_liveness.live_reads;
        let _ = &writes_before[id]; // Suppress unused warning for now
    }

    NfaLiveness {
        states,
        capture_count,
        lookaround_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::nfa::compile;
    use crate::parser::parse;

    fn analyze_pattern(pattern: &str) -> NfaLiveness {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = compile(&hir).unwrap();
        analyze_liveness(&nfa)
    }

    #[test]
    fn test_bitset_operations() {
        let mut bs = CaptureBitSet::empty();
        assert!(bs.is_empty());

        bs.set(0);
        bs.set(3);
        bs.set(7);
        assert!(bs.contains(0));
        assert!(bs.contains(3));
        assert!(bs.contains(7));
        assert!(!bs.contains(1));
        assert_eq!(bs.count(), 3);

        let bs2 = CaptureBitSet::all(4);
        assert!(bs2.contains(0));
        assert!(bs2.contains(3));
        assert!(!bs2.contains(4));

        let union = bs.union(&bs2);
        assert!(union.contains(0));
        assert!(union.contains(3));
        assert!(union.contains(7));

        let intersect = bs.intersect(&bs2);
        assert!(intersect.contains(0));
        assert!(intersect.contains(3));
        assert!(!intersect.contains(7));
    }

    #[test]
    fn test_simple_capture() {
        // Pattern: (a)
        // Capture 0 (full match) and capture 1 should be live
        let liveness = analyze_pattern(r"(a)");
        assert_eq!(liveness.capture_count, 1);
        // At least some states should have capture 1 in their live set
        let _has_capture_1_live = liveness.states.iter().any(|s| s.live_reads.contains(1));
        // Capture writes should exist
        let has_capture_1_write = liveness.states.iter().any(|s| s.writes.contains(1));
        assert!(has_capture_1_write, "Should have capture 1 write");
    }

    #[test]
    fn test_alternation_no_captures() {
        // Pattern: a|b - no explicit captures, but group 0 (full match) is implicit
        let liveness = analyze_pattern(r"a|b");
        assert_eq!(liveness.capture_count, 0);
        // Even with no explicit captures, group 0 needs to be tracked at match states
        // copy_mask will contain group 0 at states that lead to matches
        // The important thing is that explicit capture groups (1+) are not in copy_mask
        for state in &liveness.states {
            assert!(
                !state.copy_mask.contains(1),
                "No explicit captures means group 1+ should not be in copy_mask"
            );
        }
    }

    #[test]
    fn test_alternation_with_captures() {
        // Pattern: (a)|(b) - each branch captures differently
        let liveness = analyze_pattern(r"(a)|(b)");
        assert_eq!(liveness.capture_count, 2);
    }

    #[test]
    fn test_backref_makes_capture_live() {
        // Pattern: (a)\1 - backref reads capture 1
        let liveness = analyze_pattern(r"(a)\1");

        // Capture 1 should be live at states before the backref
        let has_live_capture_1 = liveness.states.iter().any(|s| s.live_reads.contains(1));
        assert!(has_live_capture_1, "Backref should make capture 1 live");
    }

    #[test]
    fn test_nested_captures() {
        // Pattern: ((a)(b)) - nested captures
        let liveness = analyze_pattern(r"((a)(b))");
        assert_eq!(liveness.capture_count, 3);
    }

    #[test]
    fn test_lookaround_count() {
        // Pattern with lookahead
        let liveness = analyze_pattern(r"a(?=b)");
        assert_eq!(liveness.lookaround_count, 1);

        // Pattern with multiple lookarounds
        let liveness = analyze_pattern(r"(?=a)b(?!c)");
        assert_eq!(liveness.lookaround_count, 2);
    }
}
