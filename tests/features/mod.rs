//! Integration tests for regex features.
//!
//! This module tests core regex features including:
//! - Backreferences
//! - Case-insensitive matching
//! - Lookahead and lookbehind assertions
//! - Character escapes
//! - Extended (`x`) mode
//! - Grapheme clusters (`\X`)
//! - Literal quoting (`\Q…\E`)
//! - Named captures
//! - One-pass (deterministic) capture extraction
//! - Rejection of malformed patterns
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
mod class_set_ops;
mod escape_sequences;
mod extended_mode;
mod grapheme_cluster;
mod inline_flags;
mod lookaround;
mod named_capture;
mod onepass_captures;
mod pattern_rejection;
mod posix_class;
mod quoting;
mod regex_crate_differential;
mod syntax;
mod word_boundary;
