//! Integration tests for regex features.
//!
//! This module tests core regex features including:
//! - Backreferences
//! - Case-insensitive matching
//! - Lookahead and lookbehind assertions
//! - Named captures
//! - Syntax validation

use regexr::Regex;
#[cfg(feature = "jit")]
use regexr::RegexBuilder;

/// Creates a Regex with JIT enabled when the `jit` feature is available.
#[allow(dead_code)]
pub fn regex(pattern: &str) -> Regex {
    #[cfg(feature = "jit")]
    {
        RegexBuilder::new(pattern)
            .jit(true)
            .build()
            .expect("failed to compile pattern")
    }
    #[cfg(not(feature = "jit"))]
    {
        Regex::new(pattern).expect("failed to compile pattern")
    }
}

mod backreference;
mod case_insensitive;
mod lookaround;
mod named_capture;
mod syntax;
mod word_boundary;
