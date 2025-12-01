//! Case-insensitive matching tests.
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

// Using local mod.rs

use super::regex;

#[test]
fn test_case_insensitive_ascii() {
    let re = regex(r"(?i)hello");
    assert!(re.is_match("hello"));
    assert!(re.is_match("HELLO"));
    assert!(re.is_match("Hello"));
    assert!(re.is_match("HeLLo"));
    assert!(!re.is_match("helo"));
}

#[test]
fn test_case_insensitive_single_char() {
    let re = regex(r"(?i)a");
    assert!(re.is_match("a"));
    assert!(re.is_match("A"));
    assert!(!re.is_match("b"));
}

#[test]
fn test_case_insensitive_mixed() {
    let re = regex(r"(?i)test123");
    assert!(re.is_match("test123"));
    assert!(re.is_match("TEST123"));
    assert!(re.is_match("Test123"));
    assert!(!re.is_match("test12"));
}

#[test]
fn test_case_insensitive_unicode() {
    let re = regex(r"(?i)α");
    assert!(re.is_match("α"));
    assert!(re.is_match("Α"));
}

#[test]
fn test_case_insensitive_german() {
    let re = regex(r"(?i)straße");
    assert!(re.is_match("Straße"));
    assert!(!re.is_match("STRASSE"));
}

#[test]
fn test_case_sensitive_default() {
    let re = regex(r"Hello");
    assert!(re.is_match("Hello"));
    assert!(!re.is_match("hello"));
    assert!(!re.is_match("HELLO"));
}
