//! Interpreter implementations for Tagged NFA execution.
//!
//! This module provides the `TaggedNfa` interpreter for fast step-based matching.
//! For captures and complex patterns, use PikeVm instead.

mod tagged_nfa;

pub use tagged_nfa::TaggedNfa;
