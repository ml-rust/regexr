//! The linear-time search a lazy DFA runs once trying one start at a time has
//! proven expensive.
//!
//! Every pattern the DFA family runs reports the longest match from the
//! leftmost start (see `automata_match_like_pikevm`). So the search splits in
//! two passes. A backward pass of the reversed pattern, seeded at every
//! position, marks each position where some match begins, and the smallest
//! one is the leftmost start. An anchored forward pass from that start finds
//! the longest end. Each pass reads every byte once.

use super::dfa::{LazyDfa, SearchGuard};
use crate::dfa::lazy::shared::{state_index, CacheCeilingExceeded, CharClass, DfaStateId};
use crate::nfa::NfaInstruction;
use crate::vm::PikeVm;

/// The reversed-pattern DFA behind [`LazyDfa::find_from_linear`], built on
/// first use.
#[derive(Debug, Clone, Default)]
pub(crate) enum ReverseDfa {
    /// Not needed yet.
    #[default]
    Unbuilt,
    /// The pattern holds a construct the reversal declines; see
    /// [`crate::nfa::reverse`].
    Unavailable,
    /// The DFA of the reversed NFA.
    Built(Box<LazyDfa>),
}

impl LazyDfa {
    /// The leftmost match starting at or after `from`, in time linear in the
    /// input.
    ///
    /// The reversed-pattern pass and the anchored forward pass cost one read
    /// of the input each. A pattern whose reversal is declined (text and line
    /// anchors), or a pass that outgrows the state cache, runs on the PikeVM
    /// instead, which is O(m·n) as well.
    pub(crate) fn find_from_linear(&mut self, input: &[u8], from: usize) -> Option<(usize, usize)> {
        let mut guard = SearchGuard::new(self);
        let outcome = guard.find_from_linear_inner(input, from);
        match guard.finish(outcome) {
            Ok(found) => found,
            Err(CacheCeilingExceeded) => PikeVm::from_arc(self.nfa_arc()).find_from(input, from),
        }
    }

    /// [`LazyDfa::find_from_linear`] inside a search already guarded.
    ///
    /// A forward pass that hits the cache ceiling leaves the flag set, so the
    /// guarded entry point reports the give-up and its caller re-runs the
    /// search on the PikeVM.
    pub(super) fn find_from_linear_inner(
        &mut self,
        input: &[u8],
        from: usize,
    ) -> Option<(usize, usize)> {
        if from > input.len() {
            return None;
        }
        let start = match self.reverse_dfa() {
            Some(reverse) => match reverse.leftmost_start(input, from) {
                Ok(start) => start?,
                Err(CacheCeilingExceeded) => return self.pikevm_find_from(input, from),
            },
            None => return self.pikevm_find_from(input, from),
        };
        self.find_at_inner(input, start).map(|end| (start, end))
    }

    /// The PikeVM search for the same pattern.
    fn pikevm_find_from(&self, input: &[u8], from: usize) -> Option<(usize, usize)> {
        PikeVm::from_arc(self.nfa_arc()).find_from(input, from)
    }

    /// The reversed-pattern DFA, built the first time it is asked for.
    fn reverse_dfa(&mut self) -> Option<&mut LazyDfa> {
        if matches!(self.reverse, ReverseDfa::Unbuilt) {
            self.reverse = match crate::nfa::reverse(self.nfa()) {
                Some(nfa) => ReverseDfa::Built(Box::new(LazyDfa::new(nfa))),
                None => ReverseDfa::Unavailable,
            };
        }
        match &mut self.reverse {
            ReverseDfa::Built(dfa) => Some(dfa),
            ReverseDfa::Unbuilt | ReverseDfa::Unavailable => None,
        }
    }

    /// On the DFA of a reversed NFA: the smallest position at or after `from`
    /// where a match of the original pattern begins.
    ///
    /// Walks `input[from..]` right to left. A fresh thread joins at every
    /// position, standing for a match that ends there, so the state at `pos`
    /// is accepting exactly when some match spans `pos..end` for an `end` at
    /// or after `pos`. The last accepting position seen is the leftmost start.
    fn leftmost_start(
        &mut self,
        input: &[u8],
        from: usize,
    ) -> Result<Option<usize>, CacheCeilingExceeded> {
        let mut guard = SearchGuard::new(self);
        let outcome = guard.leftmost_start_inner(input, from);
        guard.finish(outcome)
    }

    /// [`LazyDfa::leftmost_start`] without the search guard.
    fn leftmost_start_inner(&mut self, input: &[u8], from: usize) -> Option<usize> {
        let mut state = self.start();
        let mut leftmost = None;
        if self.is_match(state) && self.match_start_holds(input, input.len(), state) {
            leftmost = Some(input.len());
        }
        for pos in (from..input.len()).rev() {
            if self.ctx.ceiling_exceeded {
                return None;
            }
            state = self.transition_unanchored(state, input[pos]);
            if self.is_match(state) && self.match_start_holds(input, pos, state) {
                leftmost = Some(pos);
            }
        }
        leftmost
    }

    /// Whether a reversed match stands at `pos`, where it ends in the backward
    /// walk.
    ///
    /// The mirror of the forward check at a match end: the state's class is
    /// that of the byte at `pos` (read last), and the byte on the far side is
    /// the one before `pos`. A match state with no assertion stands outright.
    /// One carrying `\b` or `\B` — the assertion the original match began
    /// with, left pending because the byte beyond was not yet read — stands if
    /// its assertion holds here. Assertion states that are not match states
    /// belong to other threads and decide nothing. The reversal keeps no
    /// anchors, so these are the only assertions to settle.
    fn match_start_holds(&self, input: &[u8], pos: usize, state: DfaStateId) -> bool {
        if !self.ctx.has_word_boundary {
            return true;
        }
        let Some(dfa_state) = self.ctx.states.get(state_index(state)) else {
            return true;
        };
        let before = pos
            .checked_sub(1)
            .and_then(|i| input.get(i))
            .map_or(CharClass::NonWord, |&byte| CharClass::from_byte(byte));
        let at_boundary = dfa_state.prev_class != before;
        dfa_state.nfa_states.iter().any(|&id| {
            let Some(nfa_state) = self.ctx.nfa.get(id) else {
                return false;
            };
            nfa_state.is_match
                && match nfa_state.instruction {
                    Some(NfaInstruction::WordBoundary) => at_boundary,
                    Some(NfaInstruction::NotWordBoundary) => !at_boundary,
                    _ => true,
                }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn dfa(pattern: &str) -> LazyDfa {
        let hir = translate(&parse(pattern).expect("pattern parses")).expect("pattern translates");
        LazyDfa::new(crate::nfa::compile(&hir).expect("pattern compiles"))
    }

    /// The leftmost match by trying every start, the answer the linear search
    /// must reproduce.
    fn by_every_start(dfa: &mut LazyDfa, input: &[u8], from: usize) -> Option<(usize, usize)> {
        (from..=input.len()).find_map(|start| {
            dfa.find_at(input, start)
                .expect("cache limit is not reached")
                .map(|end| (start, end))
        })
    }

    #[test]
    fn linear_search_agrees_with_every_start() {
        let patterns = [
            r"a.*b",
            r"(?s)a.*b",
            r"[^b]*b",
            r"\w+z",
            r"x[a-z]*y",
            r"é+x?",
            r"\ba[a-z]*\b",
            r"\Ba+",
            r"a*",
            r"^a+",
            r"a+$",
            r"(?m)^a+$",
            r"(?:ab)+c?",
            r"(?i)[:sS]*.\b",
            r"[a-z]*.\b",
            r"\b.[a-z]*",
        ];
        let inputs = [
            "",
            "aaaa",
            "aaab",
            "baaa",
            "xaaay",
            "ab ab",
            "a b aab zz",
            "ééx é",
            "cabab\nabab",
            "zzz a",
            "abababc abab",
            "ssssEc11e1c",
            "ab cd",
        ];
        for pattern in patterns {
            let mut linear = dfa(pattern);
            let mut reference = dfa(pattern);
            for input in inputs {
                let input = input.as_bytes();
                for from in 0..=input.len() {
                    assert_eq!(
                        linear.find_from_linear(input, from),
                        by_every_start(&mut reference, input, from),
                        "{pattern:?} on {:?} from {from}",
                        String::from_utf8_lossy(input)
                    );
                }
            }
        }
    }

    /// The same agreement over generated patterns and haystacks, weighted
    /// toward word boundaries: the reversal keeps them, and a pending one at
    /// either end of a reversed match is where the backward pass can go wrong.
    #[test]
    fn linear_search_agrees_on_generated_patterns() {
        const ATOMS: &[&str] = &["a", "b", " ", "[ab]", "[^a]", ".", r"\w", r"\d", "é"];
        const SUFFIXES: &[&str] = &["", "", "*", "+", "?", "{1,2}"];
        const EDGES: &[&str] = &["", "", r"\b", r"\B"];
        const BYTES: &[&str] = &["a", "b", " ", "1", "é", "a"];

        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };

        for _ in 0..2000 {
            let mut pattern = String::from(EDGES[next(EDGES.len())]);
            for _ in 0..1 + next(3) {
                pattern.push_str(ATOMS[next(ATOMS.len())]);
                pattern.push_str(SUFFIXES[next(SUFFIXES.len())]);
            }
            pattern.push_str(EDGES[next(EDGES.len())]);
            let Ok(hir) = parse(&pattern).and_then(|ast| translate(&ast)) else {
                continue;
            };
            // Only the patterns engine selection lets a DFA run.
            if crate::engine::needs_boundary_aware_empty_match(&hir)
                || !crate::engine::automata_match_like_pikevm(&hir)
            {
                continue;
            }
            let Ok(nfa) = crate::nfa::compile(&hir) else {
                continue;
            };
            let mut linear = LazyDfa::new(nfa.clone());
            let mut reference = LazyDfa::new(nfa);
            for _ in 0..6 {
                let input: String = (0..next(9)).map(|_| BYTES[next(BYTES.len())]).collect();
                let input = input.as_bytes();
                for from in 0..=input.len() {
                    assert_eq!(
                        linear.find_from_linear(input, from),
                        by_every_start(&mut reference, input, from),
                        "{pattern:?} on {:?} from {from}",
                        String::from_utf8_lossy(input)
                    );
                }
            }
        }
    }
}
