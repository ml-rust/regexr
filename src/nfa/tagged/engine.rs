//! Tagged NFA Engine - Facade for interpreter and JIT execution.
//!
//! This module provides `TaggedNfaEngine`, the primary interface for Tagged NFA
//! execution from `executor.rs`. It automatically selects between:
//! - `TaggedNfa` - Fast step-based matching for simple patterns
//! - `PikeVm` - Fallback for captures and complex patterns
//! - `TaggedNfaJit` - JIT-compiled execution (when `jit` feature is enabled)

use super::interpreter::TaggedNfa;
use super::shared::PatternStep;
use super::steps::StepExtractor;
use crate::nfa::Nfa;
use crate::vm::{PikeVm, PikeVmContext};

use std::sync::RwLock;

/// An owning wrapper for Tagged NFA execution that stores the NFA and execution engines.
///
/// This is the primary interface for using the Tagged NFA engine from `executor.rs`.
/// Uses TaggedNfa for fast find() and PikeVm for captures().
pub struct TaggedNfaEngine {
    /// Pre-extracted pattern steps for fast matching (same algorithm as JIT).
    steps: Option<Vec<PatternStep>>,
    /// Cached PikeVm for capture extraction.
    pike_vm: PikeVm,
    /// Cached execution context for PikeVM (avoids allocations).
    pike_ctx: RwLock<PikeVmContext>,
}

impl TaggedNfaEngine {
    /// Creates a new Tagged NFA engine from an NFA.
    pub fn new(nfa: Nfa) -> Self {
        // Try to extract pattern steps for fast step-based matching
        let steps = StepExtractor::new(&nfa).extract();
        // Create PikeVm for capture extraction and fallback
        let pike_vm = PikeVm::new(nfa);
        let pike_ctx = RwLock::new(pike_vm.create_context());
        Self {
            steps,
            pike_vm,
            pike_ctx,
        }
    }

    /// Returns whether the pattern matches the input.
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.find(input).is_some()
    }

    /// Finds the first match, returning (start, end).
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        // Use fast step-based interpreter if pattern steps were extracted
        if let Some(ref steps) = self.steps {
            return TaggedNfa::find(steps, input);
        }
        // Fall back to PikeVm
        self.pike_vm.find(input)
    }

    /// Finds a match starting at or after the given position.
    pub fn find_at(&self, input: &[u8], start: usize) -> Option<(usize, usize)> {
        // Use fast step-based interpreter if pattern steps were extracted
        if let Some(ref steps) = self.steps {
            return TaggedNfa::find_at(steps, input, start);
        }
        // Fall back to PikeVm
        self.pike_vm.find_at(input, start)
    }

    /// Returns capture groups for the first match.
    pub fn captures(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        // Use PikeVm with cached context for capture extraction
        let mut ctx = self.pike_ctx.write().unwrap();
        self.pike_vm.captures_with_context(input, &mut ctx, 0)
    }
}
