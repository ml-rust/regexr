//! Regex syntax feature integration tests.
//!
//! Tests for alternation, character classes, quantifiers, anchors, and shorthand classes.
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

// Using local mod.rs

use super::regex;

// =============================================================================
// Alternation
// =============================================================================

#[test]
fn test_alternation() {
    let re = regex("cat|dog");
    assert!(re.is_match("I have a cat"));
    assert!(re.is_match("I have a dog"));
    assert!(!re.is_match("I have a bird"));
}

#[test]
fn test_dot_star() {
    let re = regex("a.*b");
    assert!(re.is_match("ab"));
    assert!(re.is_match("axb"));
    assert!(re.is_match("axxxb"));
    assert!(!re.is_match("a"));
    assert!(!re.is_match("b"));
}

// =============================================================================
// Character Classes
// =============================================================================

#[test]
fn test_character_class() {
    let re = regex("[0-9]+");
    let m = re.find("abc123def").unwrap();
    assert_eq!(m.as_str(), "123");
}

/// Test caret inside character class (not at start) is literal
#[test]
fn test_caret_in_character_class() {
    // Caret not at start should match literal ^
    let re = regex("[a^b]");
    assert!(re.is_match("a"));
    assert!(re.is_match("^"));
    assert!(re.is_match("b"));
    assert!(!re.is_match("c"));

    // Caret at end
    let re2 = regex("[ab^]");
    assert!(re2.is_match("^"));

    // Multiple special chars including caret (benchmark tokenization pattern)
    let re3 = regex(r"[+\-*/=<>!&|^%]+");
    assert!(re3.is_match("^"));
    assert!(re3.is_match("+"));
    assert!(re3.is_match("^%"));
    assert!(re3.is_match("!="));
}

/// Test the full tokenization pattern from benchmarks
#[test]
fn test_tokenization_pattern() {
    let pattern = r#"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+(?:\.[0-9]+)?|[+\-*/=<>!&|^%]+|[(){}\[\];,.]|"[^"]*"|'[^']*'"#;
    let re = regex(pattern);

    // Identifiers
    assert!(re.is_match("foo"));
    assert!(re.is_match("_bar"));

    // Numbers
    assert!(re.is_match("123"));
    assert!(re.is_match("3.14"));

    // Operators
    assert!(re.is_match("+"));
    assert!(re.is_match("^"));
    assert!(re.is_match("!="));

    // Punctuation
    assert!(re.is_match("("));
    assert!(re.is_match(";"));

    // Strings
    assert!(re.is_match(r#""hello""#));
    assert!(re.is_match("'world'"));
}

// =============================================================================
// Shorthand Classes (\d, \w, \s)
// =============================================================================

#[test]
fn test_shorthand_digit() {
    let re = regex("\\d+");
    let m = re.find("abc123def").unwrap();
    assert_eq!(m.as_str(), "123");
}

#[test]
fn test_shorthand_word() {
    let re = regex("\\w+");
    let m = re.find("hello world").unwrap();
    assert_eq!(m.as_str(), "hello");
}

#[test]
fn test_shorthand_whitespace() {
    let re = regex("\\s+");
    let m = re.find("hello world").unwrap();
    assert_eq!(m.as_str(), " ");
}

// =============================================================================
// Quantifiers
// =============================================================================

#[test]
fn test_plus() {
    let re = regex("a+");
    assert!(re.is_match("a"));
    assert!(re.is_match("aaa"));
    assert!(!re.is_match("b"));
}

#[test]
fn test_optional() {
    let re = regex("a?");
    assert!(re.is_match("a"));
    assert!(re.is_match(""));
}

#[test]
fn test_star() {
    let re = regex("a*");
    assert!(re.is_match(""));
    assert!(re.is_match("aaa"));
}

// =============================================================================
// Anchors
// =============================================================================

#[test]
fn test_start_anchor() {
    let re = regex("^hello");
    assert!(re.is_match("hello world"));
    assert!(!re.is_match("say hello")); // Not at start
    assert!(!re.is_match("  hello")); // Not at start
}

#[test]
fn test_end_anchor() {
    let re = regex("world$");
    assert!(re.is_match("hello world"));
    assert!(!re.is_match("world is big")); // Not at end
    assert!(!re.is_match("world  ")); // Not at end
}

#[test]
fn test_both_anchors() {
    let re = regex("^hello$");
    assert!(re.is_match("hello")); // Exact match
    assert!(!re.is_match("hello world")); // Has suffix
    assert!(!re.is_match("say hello")); // Has prefix
    assert!(!re.is_match(" hello ")); // Has both
}

#[test]
fn test_anchored_pattern() {
    let re = regex("^[a-z]+$");
    assert!(re.is_match("hello"));
    assert!(re.is_match("world"));
    assert!(!re.is_match("hello world")); // Has space
    assert!(!re.is_match("Hello")); // Has uppercase
    assert!(!re.is_match("123")); // Has digits
}

#[test]
fn test_multiline_start_anchor() {
    let re = regex("(?m)^hello");
    assert!(re.is_match("hello world")); // At start
    assert!(re.is_match("first\nhello")); // After newline
    assert!(re.is_match("line1\nline2\nhello")); // After multiple newlines
    assert!(!re.is_match("say hello")); // Not at start of line
}

#[test]
fn test_multiline_end_anchor() {
    let re = regex("(?m)world$");
    assert!(re.is_match("hello world")); // At end
    assert!(re.is_match("world\nnext")); // Before newline
    assert!(!re.is_match("world hello")); // Not at end of line
}

#[test]
fn test_empty_with_anchors() {
    let re = regex("^$");
    assert!(re.is_match("")); // Empty string
    assert!(!re.is_match("x")); // Non-empty
}
