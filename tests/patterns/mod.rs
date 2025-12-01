//! Real-world pattern tests.
//!
//! This module contains integration tests for common real-world regex patterns
//! used in production applications, including:
//! - Email validation
//! - URL parsing
//! - Phone number extraction
//! - Date/time parsing
//! - IP address validation
//! - Tokenization for LLM/NLP contexts

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

mod datetime;
mod email;
mod ip_address;
mod markup;
mod phone;
mod text_search;
mod tokenization;
mod url;
