//! Tagged NFA module for single-pass capture extraction.
//!
//! Provides interpreters and data structures for patterns with:
//! - Lookaround assertions
//! - Non-greedy quantifiers
//! - Complex capture groups
//!
//! # Module Organization
//!
//! - `shared` - Common data structures (ThreadWorklist, PatternStep, etc.)
//! - `liveness` - Liveness analysis for sparse capture copying
//! - `steps` - Pattern step extraction from NFA
//! - `interpreter/` - Interpreter implementations (always available)
//! - `jit/` - JIT compilation (feature-gated)

pub mod shared;
pub mod liveness;
pub mod steps;
pub mod interpreter;

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub mod jit;

// Re-export commonly used types
pub use shared::{
    ThreadWorklist, LookaroundCache, TaggedNfaContext, PatternStep,
    MAX_THREADS, is_word_char,
};
pub use liveness::{analyze_liveness, CaptureBitSet, NfaLiveness, StateLiveness};
pub use steps::{StepExtractor, combine_greedy_with_lookahead};
pub use interpreter::TaggedNfa;

// Engine facade
mod engine;
pub use engine::TaggedNfaEngine;
