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
                        // Lookbehind uses fixed-length matching, so don't allow GreedyStar/Plus
                        let inner_steps = self.extract_lookbehind_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        let min_len = Self::calc_min_len(&inner_steps);
                        steps.push(PatternStep::PositiveLookbehind(inner_steps, min_len));
                    }
                    NfaInstruction::NegativeLookbehind(inner_nfa) => {
                        // Lookbehind uses fixed-length matching, so don't allow GreedyStar/Plus
                        let inner_steps = self.extract_lookbehind_steps(inner_nfa);
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
                    NfaInstruction::StartOfText => {
                        steps.push(PatternStep::StartOfText);
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

                // Check for greedy star/plus pattern: state has transitions to target,
                // and target has epsilon transitions where one leads back to current state
                let target_state = &inner_nfa.states[target as usize];

                // Pattern for greedy plus: current -[byte]-> target -[eps]-> current (loop back)
                //                                              |-> next (exit)
                if target_state.transitions.is_empty() && target_state.epsilon.len() == 2 {
                    let eps0 = target_state.epsilon[0];
                    let eps1 = target_state.epsilon[1];

                    // Check if one epsilon leads back to current (greedy loop)
                    if eps0 == current {
                        // Greedy plus: must match at least one
                        steps.push(PatternStep::GreedyPlus(ranges));
                        if visited[target as usize] {
                            return Vec::new();
                        }
                        visited[target as usize] = true;
                        current = eps1; // Continue from exit path
                        continue;
                    } else if eps1 == current {
                        // Greedy plus: loop back is second epsilon
                        steps.push(PatternStep::GreedyPlus(ranges));
                        if visited[target as usize] {
                            return Vec::new();
                        }
                        visited[target as usize] = true;
                        current = eps0; // Continue from exit path
                        continue;
                    }
                }

                // Check for greedy star pattern: current has epsilon to both:
                // - a state with byte transitions (the loop body)
                // - a state that continues (the exit)
                // This is handled below in epsilon section

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

            if state.epsilon.len() == 1 && state.transitions.is_empty() {
                if visited[current as usize] {
                    return Vec::new();
                }
                visited[current as usize] = true;
                current = state.epsilon[0];
                continue;
            }

            // Handle greedy star: two epsilons where one leads to byte transitions
            // that loop back, and one leads to exit
            if state.epsilon.len() == 2 && state.transitions.is_empty() {
                let eps0 = state.epsilon[0];
                let eps1 = state.epsilon[1];

                // Try to detect: eps0 has transitions that loop, eps1 exits
                // or vice versa
                if let Some((ranges, exit_state)) = self.detect_greedy_star_in_lookaround(inner_nfa, current, eps0, eps1, &visited) {
                    steps.push(PatternStep::GreedyStar(ranges));
                    visited[current as usize] = true;
                    current = exit_state;
                    continue;
                }
                if let Some((ranges, exit_state)) = self.detect_greedy_star_in_lookaround(inner_nfa, current, eps1, eps0, &visited) {
                    steps.push(PatternStep::GreedyStar(ranges));
                    visited[current as usize] = true;
                    current = exit_state;
                    continue;
                }

                // Not a recognized greedy star pattern
                return Vec::new();
            }

            if !state.epsilon.is_empty() {
                return Vec::new();
            }

            break;
        }

        steps
    }

    /// Extracts pattern steps from a lookbehind inner NFA.
    /// This is simpler than lookahead extraction - it doesn't recognize GreedyStar/Plus
    /// because check_lookbehind uses fixed-length matching.
    /// Patterns with repetitions in lookbehind will return empty, causing fallback to PikeVM.
    fn extract_lookbehind_steps(&self, inner_nfa: &Nfa) -> Vec<PatternStep> {
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

            // Handle instructions
            if let Some(ref instr) = state.instruction {
                match instr {
                    NfaInstruction::WordBoundary => {
                        steps.push(PatternStep::WordBoundary);
                    }
                    NfaInstruction::EndOfText => {
                        steps.push(PatternStep::EndOfText);
                    }
                    NfaInstruction::StartOfText => {
                        steps.push(PatternStep::StartOfText);
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

                // Check for repetition patterns - we can't handle these in lookbehind
                let target_state = &inner_nfa.states[target as usize];
                if target_state.transitions.is_empty() && target_state.epsilon.len() == 2 {
                    let eps0 = target_state.epsilon[0];
                    let eps1 = target_state.epsilon[1];
                    // If any epsilon leads back to current, it's a loop - reject
                    if eps0 == current || eps1 == current {
                        return Vec::new();
                    }
                }

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

            if state.epsilon.len() == 1 && state.transitions.is_empty() {
                if visited[current as usize] {
                    return Vec::new();
                }
                visited[current as usize] = true;
                current = state.epsilon[0];
                continue;
            }

            // Any other epsilon patterns (including repetitions) - reject
            if !state.epsilon.is_empty() {
                return Vec::new();
            }

            break;
        }

        steps
    }

    /// Detects a greedy star pattern where `loop_start` has transitions that loop back
    /// to `loop_start` itself (or to a state that leads back), and `exit_state` is the continuation.
    /// Returns (ranges, exit_state) if detected, None otherwise.
    fn detect_greedy_star_in_lookaround(
        &self,
        inner_nfa: &Nfa,
        _branch_state: StateId,
        loop_start: StateId,
        exit_state: StateId,
        visited: &[bool],
    ) -> Option<(Vec<ByteRange>, StateId)> {
        if loop_start as usize >= inner_nfa.states.len() {
            return None;
        }

        let loop_state = &inner_nfa.states[loop_start as usize];

        // The loop state should have byte transitions
        if loop_state.transitions.is_empty() {
            return None;
        }

        // All transitions should go to the same target
        let target = loop_state.transitions[0].1;
        if !loop_state.transitions.iter().all(|(_, t)| *t == target) {
            return None;
        }

        let ranges: Vec<ByteRange> = loop_state.transitions.iter()
            .map(|(r, _)| r.clone())
            .collect();

        // The target should have epsilon back to loop_start (completing the loop)
        let target_state = &inner_nfa.states[target as usize];

        // Check if target has two epsilons where one leads back to loop_start
        if target_state.epsilon.len() == 2 {
            let eps0 = target_state.epsilon[0];
            let eps1 = target_state.epsilon[1];

            // Check if one epsilon leads back to loop_start
            if eps0 == loop_start {
                // eps0 loops back, eps1 exits
                if !visited[loop_start as usize] {
                    // The exit should eventually lead to exit_state or be it
                    return Some((ranges, exit_state));
                }
            } else if eps1 == loop_start {
                // eps1 loops back, eps0 exits
                if !visited[loop_start as usize] {
                    return Some((ranges, exit_state));
                }
            }
        }

        // Simple case: target has single epsilon back to loop_start
        if target_state.epsilon.len() == 1 && target_state.epsilon[0] == loop_start {
            if !visited[loop_start as usize] {
                return Some((ranges, exit_state));
            }
        }

        None
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
