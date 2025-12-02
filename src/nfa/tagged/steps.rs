//! Pattern step extraction from NFA.
//!
//! Extracts pattern steps from NFA for fast step-based matching.

use super::shared::PatternStep;
use crate::nfa::{ByteClass, ByteRange, Nfa, NfaInstruction, StateId};

/// Combines greedy quantifiers followed by lookahead into combined variants.
/// This is needed for both JIT and interpreter to handle backtracking correctly.
pub fn combine_greedy_with_lookahead(steps: Vec<PatternStep>) -> Vec<PatternStep> {
    let mut result = Vec::with_capacity(steps.len());
    let mut i = 0;

    while i < steps.len() {
        match &steps[i] {
            PatternStep::GreedyPlus(ranges) if i + 1 < steps.len() => match &steps[i + 1] {
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
            },
            PatternStep::GreedyStar(ranges) if i + 1 < steps.len() => match &steps[i + 1] {
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
            },
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
        // #[cfg(debug_assertions)]
        {
            if steps.is_empty() {
                // eprintln!("DEBUG extract: extract_from_state returned empty");
            } else {
                // eprintln!("DEBUG extract: got {} steps", steps.len());
            }
        }
        if steps.is_empty() {
            return None;
        }
        // Combine greedy quantifiers with following lookahead
        Some(combine_greedy_with_lookahead(steps))
    }

    fn extract_from_state(&self, start: StateId, visited: &mut [bool]) -> Vec<PatternStep> {
        let mut steps = Vec::new();
        let mut current = start;
        // #[cfg(debug_assertions)]
        let mut iteration = 0;

        loop {
            // #[cfg(debug_assertions)]
            {
                iteration += 1;
                if iteration > 1000 {
                    // eprintln!("DEBUG: too many iterations (>1000) at state {}", current);
                    return Vec::new();
                }
            }

            if current as usize >= self.nfa.states.len() {
                // #[cfg(debug_assertions)]
                // eprintln!("DEBUG: state {} out of bounds", current);
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
                    NfaInstruction::CodepointClass(cpclass, target) => {
                        // Unicode codepoint class - check for greedy loop pattern
                        if (*target as usize) < self.nfa.states.len() {
                            let target_state = &self.nfa.states[*target as usize];
                            if target_state.epsilon.len() == 2
                                && target_state.transitions.is_empty()
                            {
                                let eps0 = target_state.epsilon[0];
                                let eps1 = target_state.epsilon[1];

                                // Check if this is a greedy plus (X+) pattern
                                if eps0 == current {
                                    // Found greedy plus: emit GreedyCodepointPlus and continue to eps1
                                    steps.push(PatternStep::GreedyCodepointPlus(cpclass.clone()));
                                    visited[current as usize] = true;
                                    visited[*target as usize] = true;
                                    current = eps1;
                                    continue;
                                } else if eps1 == current {
                                    // Found greedy plus: emit GreedyCodepointPlus and continue to eps0
                                    steps.push(PatternStep::GreedyCodepointPlus(cpclass.clone()));
                                    visited[current as usize] = true;
                                    visited[*target as usize] = true;
                                    current = eps0;
                                    continue;
                                }
                            }
                        }
                        // Not a greedy loop - emit single CodepointClass and continue
                        steps.push(PatternStep::CodepointClass(cpclass.clone(), *target));
                        if visited[current as usize] {
                            return Vec::new();
                        }
                        visited[current as usize] = true;
                        current = *target;
                        continue;
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

                let ranges: Vec<ByteRange> =
                    state.transitions.iter().map(|(r, _)| r.clone()).collect();

                // Check for greedy loop
                let target_state = &self.nfa.states[target as usize];
                if target_state.epsilon.len() == 2 && target_state.transitions.is_empty() {
                    let eps0 = target_state.epsilon[0];
                    let eps1 = target_state.epsilon[1];

                    if eps0 == current {
                        // Greedy plus: loop back
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
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
                    steps.push(PatternStep::ByteClass(ByteClass::new(ranges)));
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

            // Multiple epsilon = alternation
            if state.epsilon.len() >= 2 {
                // #[cfg(debug_assertions)]
                // eprintln!("DEBUG: state {} has {} epsilon transitions (alternation)", current, state.epsilon.len());

                // Extract each alternative branch
                let mut alternatives: Vec<Vec<PatternStep>> = Vec::new();
                for (_i, &target) in state.epsilon.iter().enumerate() {
                    let mut branch_visited = visited.to_vec();
                    branch_visited[current as usize] = true;
                    let branch_steps = self.extract_branch(target, &mut branch_visited);
                    if branch_steps.is_empty() {
                        // #[cfg(debug_assertions)]
                        // eprintln!("DEBUG: alternation branch {} (target state {}) returned empty", i, target);
                        // If any branch fails to extract, fall back
                        return Vec::new();
                    }
                    alternatives.push(branch_steps);
                }
                steps.push(PatternStep::Alt(alternatives));
                break; // Alternation consumes the rest of the pattern
            }

            break;
        }

        steps
    }

    /// Extracts steps from a single branch of an alternation.
    fn extract_branch(&self, start: StateId, visited: &mut [bool]) -> Vec<PatternStep> {
        let mut steps = Vec::new();
        let mut current = start;
        // #[cfg(debug_assertions)]
        let mut iteration = 0;

        loop {
            // #[cfg(debug_assertions)]
            {
                iteration += 1;
                if iteration > 10000 {
                    return Vec::new();
                }
            }

            if current as usize >= self.nfa.states.len() {
                // #[cfg(debug_assertions)]
                // eprintln!("DEBUG extract_branch: state {} out of bounds", current);
                return Vec::new();
            }

            if visited[current as usize] {
                // #[cfg(debug_assertions)]
                // eprintln!("DEBUG extract_branch: state {} already visited (cycle detected)", current);
                return Vec::new();
            }

            let state = &self.nfa.states[current as usize];

            // Match state = end of this branch
            if state.is_match {
                break;
            }

            // Handle instructions
            if let Some(ref instr) = state.instruction {
                match instr {
                    NfaInstruction::CaptureStart(_) | NfaInstruction::CaptureEnd(_) => {
                        // Skip capture markers
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
                            // #[cfg(debug_assertions)]
                            // eprintln!("DEBUG extract_branch: lookahead extraction failed at state {}", current);
                            return Vec::new();
                        }
                        steps.push(PatternStep::PositiveLookahead(inner_steps));
                    }
                    NfaInstruction::NegativeLookahead(inner_nfa) => {
                        let inner_steps = self.extract_lookaround_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            // #[cfg(debug_assertions)]
                            // eprintln!("DEBUG extract_branch: neg lookahead extraction failed at state {}", current);
                            return Vec::new();
                        }
                        steps.push(PatternStep::NegativeLookahead(inner_steps));
                    }
                    NfaInstruction::CodepointClass(cpclass, target) => {
                        // #[cfg(debug_assertions)]
                        // eprintln!("DEBUG: CodepointClass at state {}, target={}", current, target);

                        // Check for greedy loop pattern: CodepointClass -> epsilon state -> back to current
                        if (*target as usize) >= self.nfa.states.len() {
                            // #[cfg(debug_assertions)]
                            // eprintln!("DEBUG: CodepointClass target {} out of bounds", target);
                            return Vec::new();
                        }

                        let target_state = &self.nfa.states[*target as usize];
                        if target_state.epsilon.len() == 2 && target_state.transitions.is_empty() {
                            let eps0 = target_state.epsilon[0];
                            let eps1 = target_state.epsilon[1];

                            // Check if this is a greedy plus (X+) pattern
                            if eps0 == current {
                                // Found greedy plus: CodepointClass+ - emit GreedyCodepointPlus and continue to eps1
                                steps.push(PatternStep::GreedyCodepointPlus(cpclass.clone()));
                                visited[current as usize] = true;
                                visited[*target as usize] = true;
                                current = eps1;
                                continue;
                            } else if eps1 == current {
                                // Found greedy plus: CodepointClass+ - emit GreedyCodepointPlus and continue to eps0
                                steps.push(PatternStep::GreedyCodepointPlus(cpclass.clone()));
                                visited[current as usize] = true;
                                visited[*target as usize] = true;
                                current = eps0;
                                continue;
                            }
                        }

                        // Not a greedy loop - just emit and continue
                        // #[cfg(debug_assertions)]
                        // eprintln!("DEBUG: CodepointClass not a greedy loop, continuing to target {}", target);
                        steps.push(PatternStep::CodepointClass(cpclass.clone(), *target));
                        visited[current as usize] = true;
                        current = *target;
                        continue;
                    }
                    _ => {
                        // #[cfg(debug_assertions)]
                        // eprintln!("DEBUG extract_branch: unsupported instruction {:?} at state {}", instr, current);
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

                let ranges: Vec<ByteRange> =
                    state.transitions.iter().map(|(r, _)| r.clone()).collect();

                // Check for greedy loop
                let target_state = &self.nfa.states[target as usize];
                if target_state.epsilon.len() == 2 && target_state.transitions.is_empty() {
                    let eps0 = target_state.epsilon[0];
                    let eps1 = target_state.epsilon[1];

                    if eps0 == current {
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        visited[current as usize] = true;
                        visited[target as usize] = true;
                        current = eps1;
                        continue;
                    } else if eps1 == current {
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        visited[current as usize] = true;
                        visited[target as usize] = true;
                        current = eps0;
                        continue;
                    }
                }

                // Simple transition
                steps.push(PatternStep::ByteClass(ByteClass::new(ranges)));
                visited[current as usize] = true;
                current = target;
                continue;
            }

            // Handle epsilon transitions
            if state.epsilon.len() == 1 {
                visited[current as usize] = true;
                current = state.epsilon[0];
                continue;
            }

            // Two epsilons - could be greedy star or alternation
            if state.epsilon.len() == 2 {
                let eps0 = state.epsilon[0];
                let eps1 = state.epsilon[1];

                // Check for greedy star pattern: one epsilon leads to loop body with transitions,
                // other epsilon leads to exit (continuation)
                if let Some((ranges, exit_state)) =
                    self.detect_greedy_star(current, eps0, eps1, visited)
                {
                    // #[cfg(debug_assertions)]
                    // eprintln!("DEBUG extract_branch: detected greedy star at state {}", current);
                    steps.push(PatternStep::GreedyStar(ByteClass::new(ranges)));
                    visited[current as usize] = true;
                    current = exit_state;
                    continue;
                }
                if let Some((ranges, exit_state)) =
                    self.detect_greedy_star(current, eps1, eps0, visited)
                {
                    // #[cfg(debug_assertions)]
                    // eprintln!("DEBUG extract_branch: detected greedy star at state {} (swapped)", current);
                    steps.push(PatternStep::GreedyStar(ByteClass::new(ranges)));
                    visited[current as usize] = true;
                    current = exit_state;
                    continue;
                }

                // Check for greedy loop: if one epsilon leads to an already-visited state,
                // and we have a CodepointClass step, this is a greedy plus/star pattern
                let eps0_visited = visited[eps0 as usize];
                let eps1_visited = visited[eps1 as usize];

                if eps0_visited && !eps1_visited {
                    // eps0 is the back-edge of a greedy loop, eps1 is the exit
                    // Continue with the exit branch
                    // #[cfg(debug_assertions)]
                    // eprintln!("DEBUG extract_branch: greedy loop detected at state {} (back to {}, exit to {})", current, eps0, eps1);
                    visited[current as usize] = true;
                    current = eps1;
                    continue;
                }
                if eps1_visited && !eps0_visited {
                    // eps1 is the back-edge of a greedy loop, eps0 is the exit
                    // #[cfg(debug_assertions)]
                    // eprintln!("DEBUG extract_branch: greedy loop detected at state {} (back to {}, exit to {})", current, eps1, eps0);
                    visited[current as usize] = true;
                    current = eps0;
                    continue;
                }

                // Not a greedy star, treat as alternation
                // #[cfg(debug_assertions)]
                // eprintln!("DEBUG extract_branch: actual alternation at state {} with 2 epsilons", current);
                let mut alternatives: Vec<Vec<PatternStep>> = Vec::new();
                let mut any_valid = false;
                for (_i, &target) in state.epsilon.iter().enumerate() {
                    let mut branch_visited = visited.to_vec();
                    branch_visited[current as usize] = true;
                    // Check if this branch can reach the match state
                    let target_state = &self.nfa.states[target as usize];
                    if target_state.is_match {
                        // Empty branch directly to match (like in X? patterns)
                        alternatives.push(Vec::new());
                        any_valid = true;
                        continue;
                    }
                    let branch_steps = self.extract_branch(target, &mut branch_visited);
                    // Empty steps means either valid empty branch or extraction failure
                    // We'll accept it as valid since we're dealing with alternations
                    alternatives.push(branch_steps);
                    any_valid = true;
                }
                if !any_valid {
                    // #[cfg(debug_assertions)]
                    // eprintln!("DEBUG extract_branch: no valid alternatives at state {}", current);
                    return Vec::new();
                }
                steps.push(PatternStep::Alt(alternatives));
                break;
            }

            // More than 2 epsilons - must be alternation
            if state.epsilon.len() > 2 {
                // #[cfg(debug_assertions)]
                // eprintln!("DEBUG extract_branch: multi-alternation at state {} with {} epsilons", current, state.epsilon.len());
                let mut alternatives: Vec<Vec<PatternStep>> = Vec::new();
                let mut any_valid = false;
                for (_i, &target) in state.epsilon.iter().enumerate() {
                    let mut branch_visited = visited.to_vec();
                    branch_visited[current as usize] = true;
                    // Check if this branch directly reaches match state
                    let target_state = &self.nfa.states[target as usize];
                    if target_state.is_match {
                        alternatives.push(Vec::new());
                        any_valid = true;
                        continue;
                    }
                    let branch_steps = self.extract_branch(target, &mut branch_visited);
                    // Accept empty branches as valid for alternations
                    alternatives.push(branch_steps);
                    any_valid = true;
                }
                if !any_valid {
                    // #[cfg(debug_assertions)]
                    // eprintln!("DEBUG extract_branch: no valid alternatives in multi-alternation at state {}", current);
                    return Vec::new();
                }
                steps.push(PatternStep::Alt(alternatives));
                break;
            }

            // Dead end - no transitions, no epsilon, not a match state
            return Vec::new();
        }

        steps
    }

    fn extract_lookaround_steps(&self, inner_nfa: &Nfa) -> Vec<PatternStep> {
        let mut visited = vec![false; inner_nfa.states.len()];
        let mut steps = Vec::new();
        let mut current = inner_nfa.start;
        // #[cfg(debug_assertions)]
        let mut iteration = 0;

        loop {
            // #[cfg(debug_assertions)]
            {
                iteration += 1;
                if iteration > 10000 {
                    // eprintln!("DEBUG extract_lookaround: too many iterations (>10000) at state {}", current);
                    return Vec::new();
                }
            }

            if current as usize >= inner_nfa.states.len() {
                // #[cfg(debug_assertions)]
                // eprintln!("DEBUG extract_lookaround: state {} out of bounds", current);
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
                    NfaInstruction::CaptureStart(_) | NfaInstruction::CaptureEnd(_) => {
                        // Skip capture markers in lookaround
                    }
                    NfaInstruction::CodepointClass(cpclass, target) => {
                        // Unicode codepoint class in lookaround
                        steps.push(PatternStep::CodepointClass(cpclass.clone(), *target));
                        if visited[current as usize] {
                            return Vec::new();
                        }
                        visited[current as usize] = true;
                        current = *target;
                        continue;
                    }
                    _ => {
                        // #[cfg(debug_assertions)]
                        // eprintln!("DEBUG extract_lookaround: unsupported instruction {:?} at state {}", instr, current);
                        return Vec::new();
                    }
                }
            }

            if !state.transitions.is_empty() {
                let target = state.transitions[0].1;
                if !state.transitions.iter().all(|(_, t)| *t == target) {
                    return Vec::new();
                }

                let ranges: Vec<ByteRange> =
                    state.transitions.iter().map(|(r, _)| r.clone()).collect();

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
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        if visited[target as usize] {
                            return Vec::new();
                        }
                        visited[target as usize] = true;
                        current = eps1; // Continue from exit path
                        continue;
                    } else if eps1 == current {
                        // Greedy plus: loop back is second epsilon
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
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
                    steps.push(PatternStep::ByteClass(ByteClass::new(ranges)));
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
                if let Some((ranges, exit_state)) =
                    self.detect_greedy_star_in_lookaround(inner_nfa, current, eps0, eps1, &visited)
                {
                    steps.push(PatternStep::GreedyStar(ByteClass::new(ranges)));
                    visited[current as usize] = true;
                    current = exit_state;
                    continue;
                }
                if let Some((ranges, exit_state)) =
                    self.detect_greedy_star_in_lookaround(inner_nfa, current, eps1, eps0, &visited)
                {
                    steps.push(PatternStep::GreedyStar(ByteClass::new(ranges)));
                    visited[current as usize] = true;
                    current = exit_state;
                    continue;
                }

                // Not a recognized greedy star pattern
                // #[cfg(debug_assertions)]
                // eprintln!("DEBUG extract_lookaround: state {} has 2 epsilons but not a greedy star pattern", current);
                return Vec::new();
            }

            if !state.epsilon.is_empty() {
                // #[cfg(debug_assertions)]
                // eprintln!("DEBUG extract_lookaround: state {} has {} epsilons (not handled)", current, state.epsilon.len());
                return Vec::new();
            }

            // #[cfg(debug_assertions)]
            {
                // eprintln!("DEBUG extract_lookaround: dead end at state {} (no transitions, no epsilon, not match)", current);
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

                let ranges: Vec<ByteRange> =
                    state.transitions.iter().map(|(r, _)| r.clone()).collect();

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
                    steps.push(PatternStep::ByteClass(ByteClass::new(ranges)));
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

    /// Detects a greedy star pattern (X*) in the main NFA.
    ///
    /// This handles complex structures where the loop body is a nested alternation
    /// (common for Unicode character classes like \S which expand to UTF-8 byte sequences).
    ///
    /// Pattern structure:
    ///   branch_state --eps--> loop_body_start --...-> eventually back to branch_state
    ///                  --eps--> exit_state (skip)
    fn detect_greedy_star(
        &self,
        branch_state: StateId,
        loop_start: StateId,
        exit_state: StateId,
        _visited: &[bool],
    ) -> Option<(Vec<ByteRange>, StateId)> {
        if loop_start as usize >= self.nfa.states.len() {
            return None;
        }

        // Check if exit_state eventually leads to match or continuation
        // (not back to branch_state - that would be the loop path)
        if self.state_eventually_reaches(exit_state, branch_state, 10) {
            // exit_state loops back - this might be the wrong interpretation
            return None;
        }

        // Check if loop_start eventually loops back to branch_state
        if self.state_eventually_reaches(loop_start, branch_state, 50) {
            // This is a greedy star/plus pattern!
            // We can't easily extract the byte ranges without fully traversing the sub-NFA,
            // so we'll use a placeholder approach for now
            // #[cfg(debug_assertions)]
            // eprintln!("DEBUG detect_greedy_star: FOUND loop {} -> {} -> back to {}", branch_state, loop_start, branch_state);

            // For now, signal that we found a greedy pattern but can't extract ranges
            // This will cause us to fall through to alternation handling
            // TODO: Properly extract ranges from the loop body
            return None;
        }

        None
    }

    /// Checks if following epsilon/byte transitions from `start` eventually reaches `target`.
    /// Uses BFS with a depth limit to avoid infinite loops.
    fn state_eventually_reaches(&self, start: StateId, target: StateId, max_depth: usize) -> bool {
        use std::collections::VecDeque;

        let mut queue = VecDeque::new();
        let mut seen = vec![false; self.nfa.states.len()];
        queue.push_back((start, 0));

        while let Some((state_id, depth)) = queue.pop_front() {
            if state_id == target {
                return true;
            }
            if depth >= max_depth {
                continue;
            }
            if state_id as usize >= self.nfa.states.len() || seen[state_id as usize] {
                continue;
            }
            seen[state_id as usize] = true;

            let state = &self.nfa.states[state_id as usize];

            // Follow epsilon transitions
            for &eps in &state.epsilon {
                queue.push_back((eps, depth + 1));
            }

            // Follow byte transitions
            for (_, next) in &state.transitions {
                queue.push_back((*next, depth + 1));
            }
        }

        false
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

        let ranges: Vec<ByteRange> = loop_state
            .transitions
            .iter()
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
                PatternStep::ByteClass(_) => len += 1,
                PatternStep::GreedyPlus(_) => len += 1, // At least one
                PatternStep::GreedyStar(_) => {}        // Zero or more
                PatternStep::GreedyPlusLookahead(_, _, _) => len += 1,
                PatternStep::GreedyStarLookahead(_, _, _) => {}
                PatternStep::PositiveLookahead(_)
                | PatternStep::NegativeLookahead(_)
                | PatternStep::PositiveLookbehind(_, _)
                | PatternStep::NegativeLookbehind(_, _) => {} // Zero-width
                PatternStep::WordBoundary
                | PatternStep::NotWordBoundary
                | PatternStep::StartOfText
                | PatternStep::EndOfText => {} // Zero-width
                _ => {} // Other steps - conservatively assume 0
            }
        }
        len
    }
}
