//! Lookahead and lookbehind tests.
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.
//! Lookarounds require PikeVM fallback for JIT (DFA can't handle them).

// Using local mod.rs

use super::regex;

use regexr::Regex;

// =============================================================================
// Positive Lookahead (?=...)
// =============================================================================

#[test]
fn test_positive_lookahead_basic() {
    let re = regex(r"a(?=b)");
    assert!(re.is_match("ab"));
    assert!(!re.is_match("ac"));
    assert!(!re.is_match("a"));
}

#[test]
fn test_positive_lookahead_find() {
    let re = regex(r"a(?=b)");
    let m = re.find("ab").unwrap();
    assert_eq!(m.as_str(), "a");
    assert_eq!(m.start(), 0);
    assert_eq!(m.end(), 1);
}

#[test]
fn test_positive_lookahead_complex() {
    let re = regex(r"\w+(?=\.)");
    let m = re.find("hello. world").unwrap();
    assert_eq!(m.as_str(), "hello");
}

// =============================================================================
// Negative Lookahead (?!...)
// =============================================================================

#[test]
fn test_negative_lookahead_basic() {
    let re = regex(r"a(?!b)");
    assert!(re.is_match("ac"));
    assert!(re.is_match("ad"));
    assert!(!re.is_match("ab"));
}

#[test]
fn test_negative_lookahead_end() {
    let re = regex(r"a(?!b)");
    assert!(re.is_match("a"));
}

#[test]
fn test_negative_lookahead_find() {
    let re = regex(r"a(?!b)");
    let m = re.find("ab ac").unwrap();
    assert_eq!(m.as_str(), "a");
    assert_eq!(m.start(), 3);
}

#[test]
fn test_lookahead_in_pattern() {
    let re = regex(r"(?=a)ab");
    assert!(re.is_match("ab"));
    assert!(!re.is_match("cb"));
}

#[test]
fn test_multiple_lookaheads() {
    let re = regex(r"(?=a)(?=.b)..");
    assert!(re.is_match("ab"));
    assert!(!re.is_match("ac"));
}

// =============================================================================
// Positive Lookbehind (?<=...)
// =============================================================================

#[test]
fn test_positive_lookbehind_basic() {
    let re = regex(r"(?<=a)b");
    assert!(re.is_match("ab"));
    assert!(!re.is_match("cb"));
    assert!(!re.is_match("b"));
}

#[test]
fn test_positive_lookbehind_find() {
    let re = regex(r"(?<=a)b");
    let m = re.find("ab").unwrap();
    assert_eq!(m.as_str(), "b");
    assert_eq!(m.start(), 1);
    assert_eq!(m.end(), 2);
}

#[test]
fn test_positive_lookbehind_complex() {
    let re = regex(r"(?<=hello )\w+");
    let m = re.find("hello world").unwrap();
    assert_eq!(m.as_str(), "world");
}

// =============================================================================
// Negative Lookbehind (?<!...)
// =============================================================================

#[test]
fn test_negative_lookbehind_basic() {
    let re = regex(r"(?<!a)b");
    assert!(re.is_match("cb"));
    assert!(re.is_match("xb"));
    assert!(!re.is_match("ab"));
}

#[test]
fn test_negative_lookbehind_start() {
    let re = regex(r"(?<!a)b");
    assert!(re.is_match("b"));
}

#[test]
fn test_negative_lookbehind_find() {
    let re = regex(r"(?<!a)b");
    let m = re.find("ab cb").unwrap();
    assert_eq!(m.as_str(), "b");
    assert_eq!(m.start(), 4);
}

// =============================================================================
// Combined Lookahead and Lookbehind
// =============================================================================

#[test]
fn test_lookbehind_and_lookahead() {
    let re = regex(r"(?<=a).(?=c)");
    assert!(re.is_match("abc"));
    assert!(re.is_match("adc"));
    assert!(!re.is_match("axyz"));
    assert!(!re.is_match("xbc"));
}

#[test]
fn test_lookbehind_variable_length() {
    let re = Regex::new(r"(?<=a+)b");
    if let Ok(re) = re {
        assert!(re.is_match("ab"));
        assert!(re.is_match("aab"));
        assert!(re.is_match("aaab"));
    }
}
