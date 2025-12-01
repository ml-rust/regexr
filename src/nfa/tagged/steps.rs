//! Pattern step extraction from NFA.
//!
//! Extracts pattern steps from NFA for fast step-based matching.

use crate::nfa::{ByteRange, Nfa, NfaInstruction, StateId};
use super::shared::PatternStep;

/// Combines greedy quantifiers followed by lookahead into combined variants.
/// This is needed for both JIT and interpreter to handle backtracking correctly.
pub fn combine_greedy_with_lookahead(steps: Vec<PatternStep>) -> Vec<PatternStep> {
    let mut result = Vec::with_capacity(steps.len());
    let mut i = 0;

    while i < steps.len() {
        match &steps[i] {
            PatternStep::GreedyPlus(ranges) if i + 1 < steps.len() => {
                match &steps[i + 1] {
                    PatternStep::PositiveLookahead(inner) => {
                        result.push(PatternStep::GreedyPlusLookahead(
                            ranges.clone(),
                            inner.clone(),
                            true,
                        ));
                        i += 2;
                        continue;
                    }
                    PatternStep::NegativeLookahead(inner) => {
                        result.push(PatternStep::GreedyPlusLookahead(
                            ranges.clone(),
                            inner.clone(),
                            false,
                        ));
                        i += 2;
                        continue;
                    }
                    _ => {}
                }
            }
            PatternStep::GreedyStar(ranges) if i + 1 < steps.len() => {
                match &steps[i + 1] {
                    PatternStep::PositiveLookahead(inner) => {
                        result.push(PatternStep::GreedyStarLookahead(
                            ranges.clone(),
                            inner.clone(),
                            true,
                        ));
                        i += 2;
                        continue;
                    }
                    PatternStep::NegativeLookahead(inner) => {
                        result.push(PatternStep::GreedyStarLookahead(
                            ranges.clone(),
                            inner.clone(),
                            false,
                        ));
                        i += 2;
                        continue;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        result.push(steps[i].clone());
        i += 1;
    }

    result
}

/// Extracts pattern steps from an NFA for fast matching.
pub struct StepExtractor<'a> {
    nfa: &'a Nfa,
}

impl<'a> StepExtractor<'a> {
    /// Creates a new step extractor for the given NFA.
    pub fn new(nfa: &'a Nfa) -> Self {
        Self { nfa }
    }

    /// Extracts pattern steps, returning None if pattern is too complex.
    pub fn extract(&self) -> Option<Vec<PatternStep>> {
        let mut visited = vec![false; self.nfa.states.len()];
        let steps = self.extract_from_state(self.nfa.start, &mut visited);
        if steps.is_empty() {
            return None;
        }
        // Combine greedy quantifiers with following lookahead
        Some(combine_greedy_with_lookahead(steps))
    }

    fn extract_from_state(&self, start: StateId, visited: &mut [bool]) -> Vec<PatternStep> {
        let mut steps = Vec::new();
        let mut current = start;

        loop {
            if current as usize >= self.nfa.states.len() {
                return Vec::new();
            }

            let state = &self.nfa.states[current as usize];

            // Handle instructions BEFORE checking match state
            // (lookahead instructions can be on the match state itself)
            if let Some(ref instr) = state.instruction {
                match instr {
                    NfaInstruction::CaptureStart(_) | NfaInstruction::CaptureEnd(_) => {
                        // Skip capture markers for find (they don't affect matching)
                    }
                    NfaInstruction::WordBoundary => {
                        steps.push(PatternStep::WordBoundary);
                    }
                    NfaInstruction::NotWordBoundary => {
                        steps.push(PatternStep::NotWordBoundary);
                    }
                    NfaInstruction::StartOfText => {
                        steps.push(PatternStep::StartOfText);
                    }
                    NfaInstruction::EndOfText => {
                        steps.push(PatternStep::EndOfText);
                    }
                    NfaInstruction::PositiveLookahead(inner_nfa) => {
                        let inner_steps = self.extract_lookaround_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        steps.push(PatternStep::PositiveLookahead(inner_steps));
                    }
                    NfaInstruction::NegativeLookahead(inner_nfa) => {
                        let inner_steps = self.extract_lookaround_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        steps.push(PatternStep::NegativeLookahead(inner_steps));
                    }
                    NfaInstruction::PositiveLookbehind(inner_nfa) => {
                        let inner_steps = self.extract_lookaround_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        let min_len = Self::calc_min_len(&inner_steps);
                        steps.push(PatternStep::PositiveLookbehind(inner_steps, min_len));
                    }
                    NfaInstruction::NegativeLookbehind(inner_nfa) => {
                        let inner_steps = self.extract_lookaround_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        let min_len = Self::calc_min_len(&inner_steps);
                        steps.push(PatternStep::NegativeLookbehind(inner_steps, min_len));
                    }
                    _ => {
                        // Unsupported instruction
                        return Vec::new();
                    }
                }
            }

            // Handle byte transitions
            if !state.transitions.is_empty() {
                let target = state.transitions[0].1;
                if !state.transitions.iter().all(|(_, t)| *t == target) {
                    return Vec::new();
                }

                let ranges: Vec<ByteRange> = state.transitions.iter()
                    .map(|(r, _)| r.clone())
                    .collect();

                // Check for greedy loop
                let target_state = &self.nfa.states[target as usize];
                if target_state.epsilon.len() == 2 && target_state.transitions.is_empty() {
                    let eps0 = target_state.epsilon[0];
                    let eps1 = target_state.epsilon[1];

                    if eps0 == current {
                        // Greedy plus: loop back
                        steps.push(PatternStep::GreedyPlus(ranges));
                        if visited[target as usize] {
                            return Vec::new();
                        }
                        visited[target as usize] = true;
                        current = eps1;
                        continue;
                    }
                }

                // Regular transition
                if visited[current as usize] {
                    return Vec::new();
                }
                visited[current as usize] = true;

                if ranges.len() == 1 && ranges[0].start == ranges[0].end {
                    steps.push(PatternStep::Byte(ranges[0].start));
                } else {
                    steps.push(PatternStep::Ranges(ranges));
                }
                current = target;
                continue;
            }

            // Handle epsilon transitions
            if state.epsilon.len() == 1 {
                if visited[current as usize] {
                    return Vec::new();
                }
                visited[current as usize] = true;
                current = state.epsilon[0];
                continue;
            }

            // Multiple epsilon = unsupported for now
            if !state.epsilon.is_empty() {
                return Vec::new();
            }

            break;
        }

        steps
    }

    fn extract_lookaround_steps(&self, inner_nfa: &Nfa) -> Vec<PatternStep> {
        let mut visited = vec![false; inner_nfa.states.len()];
        let mut steps = Vec::new();
        let mut current = inner_nfa.start;

        loop {
            if current as usize >= inner_nfa.states.len() {
                return Vec::new();
            }

            let state = &inner_nfa.states[current as usize];

            if state.is_match {
                break;
            }

            // Handle instructions in lookaround
            if let Some(ref instr) = state.instruction {
                match instr {
                    NfaInstruction::WordBoundary => {
                        steps.push(PatternStep::WordBoundary);
                    }
                    NfaInstruction::EndOfText => {
                        steps.push(PatternStep::EndOfText);
                    }
                    _ => return Vec::new(),
                }
            }

            if !state.transitions.is_empty() {
                let target = state.transitions[0].1;
                if !state.transitions.iter().all(|(_, t)| *t == target) {
                    return Vec::new();
                }

                let ranges: Vec<ByteRange> = state.transitions.iter()
                    .map(|(r, _)| r.clone())
                    .collect();

                if visited[current as usize] {
                    return Vec::new();
                }
                visited[current as usize] = true;

                if ranges.len() == 1 && ranges[0].start == ranges[0].end {
                    steps.push(PatternStep::Byte(ranges[0].start));
                } else {
                    steps.push(PatternStep::Ranges(ranges));
                }
                current = target;
                continue;
            }

            if state.epsilon.len() == 1 {
                if visited[current as usize] {
                    return Vec::new();
                }
                visited[current as usize] = true;
                current = state.epsilon[0];
                continue;
            }

            if !state.epsilon.is_empty() {
                return Vec::new();
            }

            break;
        }

        steps
    }

    /// Calculates the minimum length (in bytes) of input that a sequence of steps can match.
    pub fn calc_min_len(steps: &[PatternStep]) -> usize {
        let mut len = 0;
        for step in steps {
            match step {
                PatternStep::Byte(_) => len += 1,
                PatternStep::Ranges(_) => len += 1,
                PatternStep::GreedyPlus(_) => len += 1, // At least one
                PatternStep::GreedyStar(_) => {}, // Zero or more
                PatternStep::GreedyPlusLookahead(_, _, _) => len += 1,
                PatternStep::GreedyStarLookahead(_, _, _) => {},
                PatternStep::PositiveLookahead(_) |
                PatternStep::NegativeLookahead(_) |
                PatternStep::PositiveLookbehind(_, _) |
                PatternStep::NegativeLookbehind(_, _) => {}, // Zero-width
                PatternStep::WordBoundary |
                PatternStep::NotWordBoundary |
                PatternStep::StartOfText |
                PatternStep::EndOfText => {}, // Zero-width
                _ => {}, // Other steps - conservatively assume 0
            }
        }
        len
    }
}
