//! Unicode support integration tests.
//!
//! This module tests comprehensive Unicode functionality including:
//! - Basic Unicode character matching
//! - Unicode mode flags
//! - Negated Unicode properties
//! - Unicode property escapes (\p{...}, \P{...})
//! - Unicode script properties

#[cfg(feature = "jit")]
use regexr::RegexBuilder;
use regexr::Regex;

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

mod basic;
mod mode;
mod negated;
mod property;
mod script;
