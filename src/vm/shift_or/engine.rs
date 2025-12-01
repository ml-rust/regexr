//! Shift-Or engine facade.
//!
//! Provides a unified interface that selects between interpreter and JIT backends.

use crate::hir::Hir;

use super::{ShiftOr, ShiftOrInterpreter};

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
use super::jit::JitShiftOr;

/// Shift-Or engine that automatically selects the best backend.
///
/// In JIT mode, uses native x86-64 code. Otherwise, uses the interpreter.
pub struct ShiftOrEngine {
    /// The compiled Shift-Or data structure.
    shift_or: ShiftOr,
    /// JIT-compiled version (if available).
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    jit: Option<JitShiftOr>,
}

impl std::fmt::Debug for ShiftOrEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShiftOrEngine")
            .field("position_count", &self.shift_or.state_count())
            .finish()
    }
}

impl ShiftOrEngine {
    /// Creates a new Shift-Or engine from HIR.
    /// Returns None if the pattern is not suitable for Shift-Or.
    pub fn from_hir(hir: &Hir) -> Option<Self> {
        let shift_or = ShiftOr::from_hir(hir)?;

        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        let jit = JitShiftOr::compile(&shift_or);

        Some(Self {
            shift_or,
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            jit,
        })
    }

    /// Creates a new Shift-Or engine from a pre-compiled ShiftOr.
    pub fn new(shift_or: ShiftOr) -> Self {
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        let jit = JitShiftOr::compile(&shift_or);

        Self {
            shift_or,
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

        ShiftOrInterpreter::new(&self.shift_or).find(input)
    }

    /// Finds a match starting at or after the given position.
    #[inline]
    pub fn find_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        if let Some(ref jit) = self.jit {
            return jit.find_at(input, pos);
        }

        ShiftOrInterpreter::new(&self.shift_or).find_at(input, pos)
    }

    /// Tries to match at exactly the given position.
    #[inline]
    pub fn try_match_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        if let Some(ref jit) = self.jit {
            return jit.try_match_at(input, pos);
        }

        ShiftOrInterpreter::new(&self.shift_or).try_match_at(input, pos)
    }

    /// Returns the number of positions in the pattern.
    pub fn state_count(&self) -> usize {
        self.shift_or.state_count()
    }

    /// Returns a reference to the underlying ShiftOr data.
    pub fn shift_or(&self) -> &ShiftOr {
        &self.shift_or
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
