//! The linear-time search an engine hands a query to once trying one start
//! position at a time has proven expensive.

use std::sync::{Arc, OnceLock};

use super::dfa_pool::LazyDfaPool;
use crate::dfa::LazyDfa;
use crate::nfa::Nfa;

/// A lazy DFA for the pattern, built on first use and driven through
/// [`LazyDfa::find_from_linear`].
///
/// Engines whose own search has no single-pass form (the eager DFA) keep one
/// of these. It is built lazily because it is only reached after a metered
/// search gives up, which ordinary inputs never cause.
pub(crate) struct LinearFallback {
    nfa: Arc<Nfa>,
    pool: OnceLock<LazyDfaPool>,
}

impl LinearFallback {
    /// Creates the fallback for the pattern `nfa` was compiled from.
    pub(crate) fn new(nfa: Arc<Nfa>) -> Self {
        Self {
            nfa,
            pool: OnceLock::new(),
        }
    }

    /// The leftmost match starting at or after `from`, in time linear in the
    /// input.
    pub(crate) fn find_from(&self, input: &[u8], from: usize) -> Option<(usize, usize)> {
        self.pool
            .get_or_init(|| LazyDfaPool::new(LazyDfa::new(Nfa::clone(&self.nfa))))
            .with(|dfa| dfa.find_from_linear(input, from))
    }
}
