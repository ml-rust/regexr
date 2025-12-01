//! Eager DFA engine facade.
//!
//! Provides a unified interface for the Eager DFA executor.

use crate::nfa::Nfa;

use super::super::lazy::LazyDfa;
use super::interpreter::EagerDfa;

/// Eager DFA engine that wraps the interpreter.
///
/// This is a pre-materialized DFA that computes all states upfront
/// for O(1) transition lookups during matching.
pub struct EagerDfaEngine {
    dfa: EagerDfa,
}

impl std::fmt::Debug for EagerDfaEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EagerDfaEngine")
            .field("state_count", &self.dfa.state_count())
            .finish()
    }
}

impl EagerDfaEngine {
    /// Creates a new Eager DFA engine from an NFA.
    ///
    /// This first creates a LazyDfa and then materializes all states.
    pub fn new(nfa: Nfa) -> Self {
        let mut lazy = LazyDfa::new(nfa);
        Self {
            dfa: EagerDfa::from_lazy(&mut lazy),
        }
    }

    /// Creates a new Eager DFA engine from a LazyDfa.
    pub fn from_lazy(lazy: &mut LazyDfa) -> Self {
        Self {
            dfa: EagerDfa::from_lazy(lazy),
        }
    }

    /// Finds the first match, returning (start, end).
    #[inline]
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        self.dfa.find(input)
    }

    /// Finds a match starting at or after the given position.
    #[inline]
    pub fn find_at(&self, input: &[u8], pos: usize) -> Option<usize> {
        self.dfa.find_at(input, pos)
    }

    /// Returns the number of DFA states.
    pub fn state_count(&self) -> usize {
        self.dfa.state_count()
    }

    /// Returns a reference to the underlying EagerDfa.
    pub fn dfa(&self) -> &EagerDfa {
        &self.dfa
    }

    /// Returns whether JIT is being used (always false for EagerDfa currently).
    pub fn is_jit(&self) -> bool {
        false
    }

    /// Returns whether this DFA has word boundary assertions.
    pub fn has_word_boundary(&self) -> bool {
        self.dfa.has_word_boundary()
    }

    /// Returns whether this DFA has anchor assertions.
    pub fn has_anchors(&self) -> bool {
        self.dfa.has_anchors()
    }

    /// Returns whether this DFA has a start anchor.
    pub fn has_start_anchor(&self) -> bool {
        self.dfa.has_start_anchor()
    }

    /// Returns whether this DFA has an end anchor.
    pub fn has_end_anchor(&self) -> bool {
        self.dfa.has_end_anchor()
    }

    /// Returns whether this DFA has multiline anchors.
    pub fn has_multiline_anchors(&self) -> bool {
        self.dfa.has_multiline_anchors()
    }
}
