//! Lazy DFA engine facade.
//!
//! Provides a unified interface for the Lazy DFA executor.
//! Since LazyDfa doesn't have a JIT backend currently, this facade wraps the interpreter.

use crate::nfa::Nfa;

use super::interpreter::LazyDfa;

/// Lazy DFA engine that wraps the interpreter.
///
/// This is an on-demand DFA that builds states lazily during matching.
/// It provides O(1) state transitions once states are cached.
pub struct LazyDfaEngine {
    dfa: LazyDfa,
}

impl std::fmt::Debug for LazyDfaEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyDfaEngine")
            .field("state_count", &self.dfa.state_count())
            .finish()
    }
}

impl LazyDfaEngine {
    /// Creates a new Lazy DFA engine from an NFA.
    pub fn new(nfa: Nfa) -> Self {
        Self {
            dfa: LazyDfa::new(nfa),
        }
    }

    /// Returns true if the pattern matches the entire input.
    #[inline]
    pub fn is_match_bytes(&mut self, input: &[u8]) -> bool {
        self.dfa.is_match_bytes(input)
    }

    /// Finds the first match, returning (start, end).
    #[inline]
    pub fn find(&mut self, input: &[u8]) -> Option<(usize, usize)> {
        self.dfa.find(input)
    }

    /// Finds a match starting at or after the given position.
    #[inline]
    pub fn find_at(&mut self, input: &[u8], pos: usize) -> Option<usize> {
        self.dfa.find_at(input, pos)
    }

    /// Returns the number of cached DFA states.
    pub fn state_count(&self) -> usize {
        self.dfa.state_count()
    }

    /// Sets the cache size limit.
    pub fn set_cache_limit(&mut self, limit: usize) {
        self.dfa.set_cache_limit(limit);
    }

    /// Returns the number of cache flushes.
    pub fn flush_count(&self) -> usize {
        self.dfa.flush_count()
    }

    /// Clears the DFA cache.
    pub fn clear_cache(&mut self) {
        self.dfa.clear_cache();
    }

    /// Returns a reference to the underlying LazyDfa.
    pub fn dfa(&self) -> &LazyDfa {
        &self.dfa
    }

    /// Returns a mutable reference to the underlying LazyDfa.
    pub fn dfa_mut(&mut self) -> &mut LazyDfa {
        &mut self.dfa
    }

    /// Returns whether JIT is being used (always false for LazyDfa currently).
    pub fn is_jit(&self) -> bool {
        false
    }

    /// Returns true if this DFA has word boundary assertions.
    pub fn has_word_boundary(&self) -> bool {
        self.dfa.has_word_boundary()
    }

    /// Returns true if this DFA has anchor assertions.
    pub fn has_anchors(&self) -> bool {
        self.dfa.has_anchors()
    }

    /// Returns true if this DFA has a start anchor.
    pub fn has_start_anchor(&self) -> bool {
        self.dfa.has_start_anchor()
    }

    /// Returns true if this DFA has an end anchor.
    pub fn has_end_anchor(&self) -> bool {
        self.dfa.has_end_anchor()
    }

    /// Returns true if this DFA has multiline anchors.
    pub fn has_multiline_anchors(&self) -> bool {
        self.dfa.has_multiline_anchors()
    }
}
