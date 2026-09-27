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
        self.find_at(input, 0)
    }

    /// Finds a match starting at or after the given position.
    ///
    /// The full input is passed with an explicit start position, so the search
    /// keeps the left context of `start` (`^`, `\b`, lookbehind).
    pub fn find_at(&self, input: &[u8], start: usize) -> Option<(usize, usize)> {
        // Use fast step-based interpreter if pattern steps were extracted
        if let Some(ref steps) = self.steps {
            return TaggedNfa::find_at_bounded(steps, &self.pike_vm, input, start);
        }
        // Fall back to PikeVm. `find_from`, not `find_at`: this method searches
        // at or after `start`, while `PikeVm::find_at` tries only `start` itself.
        self.pike_vm.find_from(input, start)
    }

    /// Finds a match beginning exactly at `pos`, returning (pos, end).
    ///
    /// The anchored counterpart of [`TaggedNfaEngine::find_at`]: no later start
    /// is tried, so a caller verifying prefilter candidates one at a time gets
    /// an answer about `pos` alone instead of a scan to the end of the input.
    pub(crate) fn match_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos > input.len() {
            return None;
        }
        // Only start at UTF-8 codepoint boundaries (see `is_utf8_boundary`);
        // `find_at`'s loop applies the same rule to every start it tries.
        if !crate::nfa::is_utf8_boundary(input, pos) {
            return None;
        }
        // Use fast step-based interpreter if pattern steps were extracted
        if let Some(ref steps) = self.steps {
            return TaggedNfa::match_at_bounded(steps, &self.pike_vm, input, pos);
        }
        // Fall back to PikeVm. `find_at`, not `find_from`: this method requires
        // the match to begin at `pos`, which is what `PikeVm::find_at` tries.
        self.pike_vm.find_at(input, pos)
    }

    /// Returns capture groups for the first match.
    pub fn captures(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        self.captures_from(input, 0)
    }

    /// Returns capture groups for the first match starting at or after `start`.
    ///
    /// The full input is passed with an explicit start position rather than a
    /// slice beginning at `start`, so `^`, `\b` and lookbehind still see the
    /// preceding bytes.
    pub fn captures_from(&self, input: &[u8], start: usize) -> Option<Vec<Option<(usize, usize)>>> {
        // Searches for the match (like `find_at`) rather than requiring one at
        // `start`; the unanchored entry point does that in a single pass, so the
        // cost stays linear in the input rather than one full run per position.
        let mut ctx = self.pike_ctx.write().unwrap();
        self.pike_vm
            .captures_unanchored_with_context(input, &mut ctx, start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn engine(pattern: &str) -> TaggedNfaEngine {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = crate::nfa::compile(&hir).unwrap();
        TaggedNfaEngine::new(nfa)
    }

    /// `match_at` answers about `pos` alone. `find_at` searches forward from it,
    /// which is why a prefilter driven through `find_at` never filters: the
    /// first candidate scans to the end of the input.
    #[test]
    fn match_at_is_anchored_to_the_position() {
        let engine = engine(r"\d+");
        let input: &[u8] = b"abc 42";
        assert_eq!(engine.match_at(input, 0), None);
        assert_eq!(engine.match_at(input, 4), Some((4, 6)));
        assert_eq!(engine.find_at(input, 0), Some((4, 6)));
        assert_eq!(engine.match_at(input, input.len() + 1), None);
    }

    /// The same, for a lookbehind pattern — the shape that reaches this engine
    /// with a literal prefilter in front of it.
    #[test]
    fn match_at_is_anchored_with_lookbehind() {
        let engine = engine(r"(?<=@)\w+");
        let input: &[u8] = b"user @name";
        // `\w+` alone would match at 0; the lookbehind is what rejects it, and
        // no later start is tried.
        assert_eq!(engine.match_at(input, 0), None);
        assert_eq!(engine.match_at(input, 6), Some((6, 10)));
        assert_eq!(engine.find_at(input, 0), Some((6, 10)));
    }
}
