//! Reversal of a Thompson NFA.
//!
//! The reversed NFA accepts exactly the reversed strings of the original. Fed
//! the input from right to left, it recognises the spans the original matches
//! left to right, which is what lets one backward pass find where the leftmost
//! match starts (see `LazyDfa::find_from_linear`).

use super::{Nfa, NfaInstruction, NfaState};

/// Builds the NFA that recognises the reverse of `nfa`'s language.
///
/// Every byte and epsilon edge is flipped, the old start becomes the only
/// match state, and a fresh start reaches every old match state by epsilon.
/// Capture markers are dropped: they only record positions, and a reversed
/// search reports none.
///
/// `\b` and `\B` are kept. They read the bytes on both sides of a position and
/// nothing else, so they hold in a right-to-left scan exactly where they held
/// in the left-to-right one. One that the old start is reached from by epsilon
/// alone — a `\b` the original match began with — is marked a match state
/// itself, the convention the forward DFA follows for a trailing assertion: a
/// state that stops at it still reports the match, and the caller settles the
/// assertion against the byte beyond.
///
/// Returns `None` when the NFA holds anything else that is not symmetric: a
/// text or line anchor (which side of the position it inspects flips with the
/// direction), a lookaround, a backreference, a codepoint class (a whole-codepoint
/// step with no reversed byte form), or a non-greedy marker.
pub(crate) fn reverse(nfa: &Nfa) -> Option<Nfa> {
    let count = nfa.states.len();
    let mut states: Vec<NfaState> = (0..count).map(|_| NfaState::new()).collect();

    for (id, state) in nfa.states.iter().enumerate() {
        match &state.instruction {
            None | Some(NfaInstruction::CaptureStart(_) | NfaInstruction::CaptureEnd(_)) => {}
            Some(NfaInstruction::WordBoundary) => {
                states[id].instruction = Some(NfaInstruction::WordBoundary);
            }
            Some(NfaInstruction::NotWordBoundary) => {
                states[id].instruction = Some(NfaInstruction::NotWordBoundary);
            }
            Some(_) => return None,
        }
        let from = u32::try_from(id).ok()?;
        for &(range, target) in &state.transitions {
            states.get_mut(target as usize)?.add_transition(range, from);
        }
        for &target in &state.epsilon {
            states.get_mut(target as usize)?.add_epsilon(from);
        }
    }

    states.get_mut(nfa.start as usize)?.is_match = true;
    for id in 0..count {
        let is_boundary = matches!(
            states[id].instruction,
            Some(NfaInstruction::WordBoundary | NfaInstruction::NotWordBoundary)
        );
        if is_boundary && reaches_by_epsilon(&states, id, nfa.start as usize) {
            states[id].is_match = true;
        }
    }

    let mut start = NfaState::new();
    for (id, state) in nfa.states.iter().enumerate() {
        if state.is_match {
            start.add_epsilon(u32::try_from(id).ok()?);
        }
    }

    let mut reversed = Nfa::new();
    reversed.states = states;
    reversed.start = reversed.add_state(start);
    reversed.matches = vec![nfa.start];
    reversed.splits_codepoints = reversed.compute_splits_codepoints();
    reversed.max_match_len = nfa.max_match_len;
    Some(reversed)
}

/// Whether `target` is reachable from `from` by epsilon edges that pass through
/// no assertion.
fn reaches_by_epsilon(states: &[NfaState], from: usize, target: usize) -> bool {
    let mut seen = vec![false; states.len()];
    let mut stack = vec![from];
    while let Some(id) = stack.pop() {
        if id == target {
            return true;
        }
        let Some(state) = states.get(id) else {
            continue;
        };
        if std::mem::replace(&mut seen[id], true) {
            continue;
        }
        if id != from && state.instruction.is_some() {
            continue;
        }
        stack.extend(state.epsilon.iter().map(|&next| next as usize));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn nfa(pattern: &str) -> Nfa {
        let hir = translate(&parse(pattern).expect("pattern parses")).expect("pattern translates");
        crate::nfa::compile(&hir).expect("pattern compiles")
    }

    /// Whether `nfa` accepts all of `input`, by plain subset simulation.
    fn accepts(nfa: &Nfa, input: &[u8]) -> bool {
        let mut current = nfa.epsilon_closure(&[nfa.start]);
        for &byte in input {
            let mut next = std::collections::BTreeSet::new();
            for &id in &current {
                if let Some(state) = nfa.get(id) {
                    for (range, target) in &state.transitions {
                        if range.contains(byte) {
                            next.insert(*target);
                        }
                    }
                }
            }
            current = nfa.epsilon_closure(&next);
        }
        current
            .iter()
            .any(|&id| nfa.get(id).is_some_and(|state| state.is_match))
    }

    #[test]
    fn reversed_nfa_accepts_the_reversed_strings() {
        let cases: &[(&str, &[&str])] = &[
            ("abc", &["abc", "cba", "ab", ""]),
            ("a.*b", &["ab", "axxb", "ba", "a"]),
            ("(a|bc)+d", &["ad", "bcad", "abcd", "d", "cbd"]),
            ("é+x", &["éx", "ééx", "x"]),
            ("(?:ab)?c{2,3}", &["cc", "abccc", "abc", "ccab"]),
        ];
        for (pattern, inputs) in cases {
            let forward = nfa(pattern);
            let backward = reverse(&forward).expect("pattern is reversible");
            for input in *inputs {
                let reversed: Vec<u8> = input.bytes().rev().collect();
                assert_eq!(
                    accepts(&forward, input.as_bytes()),
                    accepts(&backward, &reversed),
                    "{pattern:?} on {input:?}"
                );
            }
        }
    }

    #[test]
    fn anchors_and_lookaround_are_not_reversed() {
        for pattern in ["^a", "a$", "(?m)^a", "a(?=b)", "(a)\\1"] {
            assert!(reverse(&nfa(pattern)).is_none(), "{pattern:?}");
        }
        assert!(reverse(&nfa(r"\ba+\b")).is_some());
    }
}
