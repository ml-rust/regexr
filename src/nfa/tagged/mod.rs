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

pub mod interpreter;
pub mod liveness;
pub mod shared;
pub mod steps;

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod jit;

// Re-export commonly used types
pub use interpreter::TaggedNfa;
pub use liveness::{analyze_liveness, CaptureBitSet, NfaLiveness, StateLiveness};
pub use shared::{
    is_word_char, LookaroundCache, PatternStep, TaggedNfaContext, ThreadWorklist, MAX_THREADS,
};
pub use steps::{combine_greedy_with_lookahead, StepExtractor};

// Engine facade
mod engine;
pub use engine::TaggedNfaEngine;
