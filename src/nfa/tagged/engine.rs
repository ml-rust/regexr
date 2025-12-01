//! Tagged NFA Engine - Facade for interpreter and JIT execution.
//!
//! This module provides `TaggedNfaEngine`, the primary interface for Tagged NFA
//! execution from `executor.rs`. It automatically selects between:
//! - `StepInterpreter` - Fast step-based matching for simple patterns
//! - `TaggedNfaInterpreter` - Full Thompson NFA simulation for complex patterns
//! - `TaggedNfaJit` - JIT-compiled execution (when `jit` feature is enabled)

use crate::nfa::Nfa;
use super::liveness::{analyze_liveness, NfaLiveness};
use super::steps::StepExtractor;
use super::shared::PatternStep;
use super::interpreter::{StepInterpreter, TaggedNfaInterpreter};

/// An owning wrapper for Tagged NFA execution that stores the NFA and liveness data.
///
/// This is the primary interface for using the Tagged NFA engine from `executor.rs`.
/// Unlike `TaggedNfaInterpreter` which borrows the NFA, this struct owns all the data
/// needed for execution.
pub struct TaggedNfaEngine {
    nfa: Nfa,
    liveness: NfaLiveness,
    /// Pre-extracted pattern steps for fast matching (same algorithm as JIT).
    steps: Option<Vec<PatternStep>>,
}

impl TaggedNfaEngine {
    /// Creates a new Tagged NFA engine from an NFA.
    pub fn new(nfa: Nfa) -> Self {
        let liveness = analyze_liveness(&nfa);
        // Try to extract pattern steps for fast step-based matching
        let steps = StepExtractor::new(&nfa).extract();
        Self { nfa, liveness, steps }
    }

    /// Returns whether the pattern matches the input.
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.find(input).is_some()
    }

    /// Finds the first match, returning (start, end).
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        // Use fast step-based interpreter if pattern steps were extracted
        if let Some(ref steps) = self.steps {
            return StepInterpreter::find(steps, input);
        }
        // Fall back to Thompson NFA simulation
        let interp = TaggedNfaInterpreter::new(&self.nfa, &self.liveness);
        interp.find(input)
    }

    /// Finds a match starting at or after the given position.
    pub fn find_at(&self, input: &[u8], start: usize) -> Option<(usize, usize)> {
        // Use fast step-based interpreter if pattern steps were extracted
        if let Some(ref steps) = self.steps {
            return StepInterpreter::find_at(steps, input, start);
        }
        // Fall back to Thompson NFA simulation
        let interp = TaggedNfaInterpreter::new(&self.nfa, &self.liveness);
        for pos in start..=input.len() {
            if let Some(caps) = interp.captures_at(input, pos) {
                if let Some(full_match) = caps.first().and_then(|c| *c) {
                    return Some(full_match);
                }
            }
        }
        None
    }

    /// Returns capture groups for the first match.
    pub fn captures(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        // For captures, always use Thompson NFA (steps don't track captures yet)
        let interp = TaggedNfaInterpreter::new(&self.nfa, &self.liveness);
        interp.captures(input)
    }
}
