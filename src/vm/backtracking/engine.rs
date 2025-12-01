//! Backtracking engine facade.
//!
//! Provides a unified interface that selects between interpreter and JIT backends.

use crate::hir::Hir;

use super::interpreter::BacktrackingVm;

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
use super::jit::{compile_backtracking, BacktrackingJit};

/// Backtracking engine that automatically selects the best backend.
///
/// In JIT mode, uses native x86-64 code. Otherwise, uses the interpreter.
pub struct BacktrackingEngine {
    /// The compiled backtracking VM (interpreter).
    vm: BacktrackingVm,
    /// JIT-compiled version (if available).
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    jit: Option<BacktrackingJit>,
}

impl std::fmt::Debug for BacktrackingEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BacktrackingEngine")
            .field("capture_count", &self.vm.capture_count())
            .finish()
    }
}

impl BacktrackingEngine {
    /// Creates a new backtracking engine from HIR.
    pub fn new(hir: &Hir) -> Self {
        let vm = BacktrackingVm::new(hir);

        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        let jit = compile_backtracking(hir).ok();

        Self {
            vm,
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            jit,
        }
    }

    /// Returns true if the pattern matches anywhere in the input.
    #[inline]
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.find(input).is_some()
    }

    /// Finds the first match, returning (start, end).
    #[inline]
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        if let Some(ref jit) = self.jit {
            return jit.find(input);
        }

        self.vm.find(input)
    }

    /// Finds a match starting at or after the given position.
    #[inline]
    pub fn find_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        if let Some(ref jit) = self.jit {
            return jit.find_at(input, pos);
        }

        self.vm.find_at(input, pos)
    }

    /// Returns capture groups for the first match.
    #[inline]
    pub fn captures(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        if let Some(ref jit) = self.jit {
            return jit.captures(input);
        }

        self.vm.captures(input)
    }

    /// Returns the number of capture groups.
    pub fn capture_count(&self) -> u32 {
        self.vm.capture_count()
    }

    /// Returns a reference to the underlying BacktrackingVm.
    pub fn vm(&self) -> &BacktrackingVm {
        &self.vm
    }

    /// Returns whether JIT is being used.
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    pub fn is_jit(&self) -> bool {
        self.jit.is_some()
    }

    /// Returns whether JIT is being used (always false without JIT feature).
    #[cfg(not(all(feature = "jit", target_arch = "x86_64")))]
    pub fn is_jit(&self) -> bool {
        false
    }
}
