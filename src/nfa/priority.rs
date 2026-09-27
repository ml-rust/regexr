//! Whether match priority can change where a match ends.
//!
//! The PikeVM reports the leftmost-first match: threads are ordered by
//! priority, and the first thread to reach a match state cuts off every thread
//! ranked below it. The DFA family and Shift-Or keep no order and report the
//! longest match from the leftmost start instead. The two agree on the start.
//! They disagree on the end only when a cut removes a thread that could still
//! consume input and match later.
//!
//! `a?(?:ab)?` on "ab" shows the cut: the preferred run takes `a` and
//! matches at 1, which cuts the run that skipped `a` and is part-way through
//! `ab`. Leftmost-first ends at 1, the longest match ends at 2.
//!
//! [`priority_is_irrelevant`] proves that no such cut exists. It tracks pairs
//! of runs over the same input, where the first run outranks the second. A cut
//! that loses input needs a pair where the first run sits on a match state and
//! the second on a consuming state that can still reach one. When no reachable
//! pair has that shape, the leftmost-first end is the longest end for every
//! haystack.
//!
//! Assertions are treated as always true. That admits runs the real search
//! rejects, so it can only find more pairs, never fewer.

use super::{Nfa, NfaInstruction, StateId};
use crate::hash::FxHashSet;

/// Reachable pairs explored before the analysis gives up. Giving up answers
/// "not proven", which routes the pattern to the PikeVM.
const MAX_PAIRS: usize = 1 << 17;

/// One run in a pair: the NFA state it is at, and whether it may still take
/// that state's epsilon edges.
///
/// A run that chose to consume at a state (`resting`) ranks above the runs
/// that leave it by epsilon, so it must not also take those edges itself.
#[derive(Clone, Copy)]
struct Run {
    state: StateId,
    resting: bool,
}

impl Run {
    fn at(state: StateId) -> Self {
        Self {
            state,
            resting: false,
        }
    }
}

/// Pair key: the higher-priority run's state and resting flag, then the
/// lower-priority run's state.
fn key(high: Run, low: StateId) -> u64 {
    (u64::from(high.state) << 33) | (u64::from(high.resting) << 32) | u64::from(low)
}

/// Whether every haystack gets the same match end under leftmost-first and
/// leftmost-longest semantics. `false` means "not proven", never "differs".
pub fn priority_is_irrelevant(nfa: &Nfa) -> bool {
    let supported = nfa.states.iter().all(|s| {
        matches!(
            s.instruction,
            None | Some(
                NfaInstruction::CaptureStart(_)
                    | NfaInstruction::CaptureEnd(_)
                    | NfaInstruction::WordBoundary
                    | NfaInstruction::NotWordBoundary
                    | NfaInstruction::StartOfText
                    | NfaInstruction::EndOfText
                    | NfaInstruction::StartOfLine
                    | NfaInstruction::EndOfLine
            )
        )
    });
    if !supported {
        return false;
    }

    let live = can_reach_match(nfa);
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    let mut stack: Vec<(Run, StateId)> = Vec::new();

    // Every point where one run can outrank another: a state's own consuming
    // edges rank first, then its epsilon edges in order; overlapping byte
    // edges rank in the order they are listed.
    for (id, state) in nfa.states.iter().enumerate() {
        let id = id as StateId;
        let mut options: Vec<Run> = Vec::with_capacity(state.epsilon.len() + 1);
        if !state.transitions.is_empty() {
            options.push(Run {
                state: id,
                resting: true,
            });
        }
        options.extend(state.epsilon.iter().map(|&e| Run::at(e)));
        for (i, &high) in options.iter().enumerate() {
            for low in &options[i + 1..] {
                stack.push((high, low.state));
            }
        }
        for (i, (range_high, target_high)) in state.transitions.iter().enumerate() {
            for (range_low, target_low) in &state.transitions[i + 1..] {
                if range_high.start <= range_low.end && range_low.start <= range_high.end {
                    stack.push((Run::at(*target_high), *target_low));
                }
            }
        }
    }

    while let Some((high, low)) = stack.pop() {
        // Two runs on one state at one position are one thread: the search
        // keeps the higher-ranked arrival. Whatever the lower run could do
        // next, the kept thread does at its own rank, and the divergence
        // points seeded above already pair those continuations.
        if high.state == low {
            continue;
        }
        if !seen.insert(key(high, low)) {
            continue;
        }
        if seen.len() > MAX_PAIRS {
            return false;
        }
        let (Some(h), Some(l)) = (nfa.get(high.state), nfa.get(low)) else {
            return false;
        };

        if h.is_match
            && l.transitions
                .iter()
                .any(|&(_, target)| live.get(target as usize).copied().unwrap_or(true))
        {
            return false;
        }

        if !high.resting {
            for &e in &h.epsilon {
                stack.push((Run::at(e), low));
            }
        }
        for &e in &l.epsilon {
            stack.push((high, e));
        }
        for (range_high, target_high) in &h.transitions {
            for (range_low, target_low) in &l.transitions {
                if range_high.start <= range_low.end && range_low.start <= range_high.end {
                    stack.push((Run::at(*target_high), *target_low));
                }
            }
        }
    }
    true
}

/// For each state, whether some path of epsilon and byte edges leads from it
/// to a match state.
fn can_reach_match(nfa: &Nfa) -> Vec<bool> {
    let n = nfa.states.len();
    let mut preds: Vec<Vec<StateId>> = vec![Vec::new(); n];
    for (id, state) in nfa.states.iter().enumerate() {
        let targets = state
            .epsilon
            .iter()
            .copied()
            .chain(state.transitions.iter().map(|&(_, t)| t));
        for target in targets {
            if let Some(p) = preds.get_mut(target as usize) {
                p.push(id as StateId);
            }
        }
    }
    let mut live = vec![false; n];
    let mut stack: Vec<StateId> = Vec::new();
    for (id, state) in nfa.states.iter().enumerate() {
        if state.is_match {
            live[id] = true;
            stack.push(id as StateId);
        }
    }
    while let Some(id) = stack.pop() {
        for &p in &preds[id as usize] {
            if !live[p as usize] {
                live[p as usize] = true;
                stack.push(p);
            }
        }
    }
    live
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn irrelevant(pattern: &str) -> bool {
        let hir = translate(&parse(pattern).unwrap()).unwrap();
        priority_is_irrelevant(&super::super::compile(&hir).unwrap())
    }

    #[test]
    fn single_position_repeats_are_proven() {
        for pattern in [
            "abc",
            "a*ab",
            "[ab]*b?a+",
            r"\d+(?:\.\d+)?",
            ".*foo",
            r"\w+\s*=",
            r"^\d+$",
            r"\bthe\b",
            r"(?:\d{1,3}\.){3}\d{1,3}",
            "(?:[a-z]+\\.)*[a-z]+",
        ] {
            assert!(irrelevant(pattern), "{pattern}");
        }
    }

    #[test]
    fn a_cut_that_loses_input_is_found() {
        for pattern in [
            "a?(?:ab)?",
            "a*(?:ab)*",
            "a?(?:abcd|b)",
            "(?:.{1,2}[^s])*",
            "ab|abc",
            "[sS]|[sS][eE][cC]",
        ] {
            assert!(!irrelevant(pattern), "{pattern}");
        }
    }
}
