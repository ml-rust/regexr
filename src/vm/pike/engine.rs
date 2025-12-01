//! PikeVM engine facade.
//!
//! Provides a unified interface for the PikeVM executor.
//! Since PikeVM doesn't have a JIT backend, this facade simply wraps the interpreter.

use crate::nfa::Nfa;

use super::interpreter::PikeVm;
use super::shared::PikeVmContext;

/// PikeVM engine that wraps the interpreter.
///
/// This is a thread-based NFA simulator that supports capture groups,
/// backreferences, and lookarounds.
pub struct PikeVmEngine {
    vm: PikeVm,
}

impl std::fmt::Debug for PikeVmEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PikeVmEngine").finish()
    }
}

impl PikeVmEngine {
    /// Creates a new PikeVM engine from an NFA.
    pub fn new(nfa: Nfa) -> Self {
        Self {
            vm: PikeVm::new(nfa),
        }
    }

    /// Returns true if the pattern matches anywhere in the input.
    #[inline]
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.vm.is_match(input)
    }

    /// Finds the first match, returning (start, end).
    #[inline]
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        self.vm.find(input)
    }

    /// Finds a match starting at or after the given position.
    #[inline]
    pub fn find_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        self.vm.find_at(input, pos)
    }

    /// Returns capture groups for the first match.
    #[inline]
    pub fn captures(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        self.vm.captures(input)
    }

    /// Returns capture groups for a match known to start at position 0.
    #[inline]
    pub fn captures_from_start(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        self.vm.captures_from_start(input)
    }

    /// Creates a reusable context for this VM.
    pub fn create_context(&self) -> PikeVmContext {
        self.vm.create_context()
    }

    /// Returns capture groups using a pre-allocated context.
    #[inline]
    pub fn captures_from_start_with_context(
        &self,
        input: &[u8],
        ctx: &mut PikeVmContext,
    ) -> Option<Vec<Option<(usize, usize)>>> {
        self.vm.captures_from_start_with_context(input, ctx)
    }

    /// Returns capture groups using a pre-allocated context, starting from a given position.
    #[inline]
    pub fn captures_with_context(
        &self,
        input: &[u8],
        ctx: &mut PikeVmContext,
        start_pos: usize,
    ) -> Option<Vec<Option<(usize, usize)>>> {
        self.vm.captures_with_context(input, ctx, start_pos)
    }

    /// Returns a reference to the underlying PikeVm.
    pub fn vm(&self) -> &PikeVm {
        &self.vm
    }

    /// Returns whether JIT is being used (always false for PikeVM).
    pub fn is_jit(&self) -> bool {
        false
    }
}
