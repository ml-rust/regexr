//! Text search pattern tests.
//!
//! Common patterns for searching text content including word boundaries,
//! alternation patterns, and word matching.
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

use super::regex;

// =============================================================================
// Word Boundary Patterns
// =============================================================================

/// Word boundary search pattern from benchmark
/// Pattern: `\bthe\b`
#[test]
fn test_word_boundary_the() {
    let re = regex(r"\bthe\b");

    // Should match "the" as a whole word
    assert!(re.is_match("the cat"));
    assert!(re.is_match("in the house"));
    assert!(re.is_match("the"));

    // Should NOT match "the" within other words
    assert!(!re.is_match("there"));
    assert!(!re.is_match("other"));
    assert!(!re.is_match("them"));
    assert!(!re.is_match("bathe"));
}

#[test]
fn test_word_boundary_count() {
    let re = regex(r"\bthe\b");

    let text = "The quick brown fox jumps over the lazy dog. The end.";
    // Note: case-sensitive, so "The" won't match "\bthe\b"
    let count = re.find_iter(text).count();
    assert_eq!(count, 1); // Only "the" in "over the lazy"
}

#[test]
fn test_word_boundary_case_insensitive() {
    let re = regex(r"(?i)\bthe\b");

    let text = "The quick brown fox jumps over the lazy dog. The end.";
    let count = re.find_iter(text).count();
    assert_eq!(count, 3); // "The", "the", "The"
}

#[test]
fn test_word_boundary_various_words() {
    let re = regex(r"\bword\b");

    assert!(re.is_match("word"));
    assert!(re.is_match("a word here"));
    assert!(re.is_match("word!"));
    assert!(re.is_match("(word)"));

    assert!(!re.is_match("words"));
    assert!(!re.is_match("keyword"));
    assert!(!re.is_match("wording"));
}

#[test]
fn test_word_boundary_numbers() {
    let re = regex(r"\b\d+\b");

    assert!(re.is_match("123"));
    assert!(re.is_match("value: 42"));
    assert!(re.is_match("test 100 here"));

    // Numbers within words shouldn't match as standalone
    let text = "abc123def";
    assert!(!re.is_match(text));
}

// =============================================================================
// Alternation Patterns (log level keywords)
// =============================================================================

/// Log level alternation pattern from benchmark
/// Pattern: `error|warning|critical|fatal`
#[test]
fn test_log_level_alternation() {
    let re = regex(r"error|warning|critical|fatal");

    assert!(re.is_match("error"));
    assert!(re.is_match("warning"));
    assert!(re.is_match("critical"));
    assert!(re.is_match("fatal"));

    assert!(!re.is_match("info"));
    assert!(!re.is_match("debug"));
}

#[test]
fn test_log_level_in_text() {
    let re = regex(r"error|warning|critical|fatal");

    let logs = r#"2024-01-15 10:30:00 [info] Server starting
2024-01-15 10:30:01 [warning] deprecated config option found
2024-01-15 10:30:02 [error] Failed to connect to database
2024-01-15 10:30:03 [critical] System shutdown initiated
2024-01-15 10:30:04 [fatal] Unrecoverable issue"#;

    let matches: Vec<_> = re.find_iter(logs).map(|m| m.as_str()).collect();
    assert_eq!(matches.len(), 4);
    assert_eq!(matches[0], "warning");
    assert_eq!(matches[1], "error");
    assert_eq!(matches[2], "critical");
    assert_eq!(matches[3], "fatal");
}

#[test]
fn test_extended_log_level_alternation() {
    let re = regex(r"error|warning|critical|fatal|info|debug|trace");

    assert!(re.is_match("info"));
    assert!(re.is_match("debug"));
    assert!(re.is_match("trace"));
    assert!(re.is_match("error"));
}

#[test]
fn test_alternation_with_word_boundary() {
    let re = regex(r"\b(?:error|warning|critical|fatal)\b");

    // Should match whole words only (case-sensitive)
    assert!(re.is_match("[error]"));
    assert!(re.is_match("warning: something"));

    // Should not match within words
    assert!(!re.is_match("errors"));
    assert!(!re.is_match("prewarning"));
}

#[test]
fn test_alternation_with_word_boundary_case_insensitive() {
    let re = regex(r"(?i)\b(?:error|warning|critical|fatal)\b");

    // Case-insensitive matching
    assert!(re.is_match("ERROR"));
    assert!(re.is_match("Warning"));
    assert!(re.is_match("CRITICAL"));
}

// =============================================================================
// Word Matching Patterns
// =============================================================================

/// Word character sequence pattern from benchmark
/// Pattern: `\w+`
#[test]
fn test_word_sequence() {
    let re = regex(r"\w+");

    let text = "Hello, World! This is a test.";
    let words: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(words, vec!["Hello", "World", "This", "is", "a", "test"]);
}

#[test]
fn test_word_with_numbers() {
    let re = regex(r"\w+");

    let text = "user123 test_var item2";
    let words: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(words, vec!["user123", "test_var", "item2"]);
}

#[test]
fn test_word_count() {
    let re = regex(r"\w+");

    let text = "The quick brown fox jumps over the lazy dog.";
    let count = re.find_iter(text).count();
    assert_eq!(count, 9);
}

// =============================================================================
// Common Text Search Patterns
// =============================================================================

#[test]
fn test_search_pattern_with_context() {
    // Find word with surrounding context
    let re = regex(r"\w+\s+error\s+\w+");

    let text = "A critical error occurred in the system";
    let m = re.find(text).unwrap();
    assert_eq!(m.as_str(), "critical error occurred");
}

#[test]
fn test_search_multiple_keywords() {
    let re = regex(r"TODO|FIXME|HACK|XXX");

    let code = r#"
        // TODO: implement this
        let x = 42; // FIXME: magic number
        // HACK: workaround for bug
        // XXX: remove before release
    "#;

    let matches: Vec<_> = re.find_iter(code).map(|m| m.as_str()).collect();
    assert_eq!(matches, vec!["TODO", "FIXME", "HACK", "XXX"]);
}

#[test]
fn test_identifier_pattern() {
    let re = regex(r"[a-zA-Z_][a-zA-Z0-9_]*");

    let code = "int main() { return 0; }";
    let ids: Vec<_> = re.find_iter(code).map(|m| m.as_str()).collect();

    assert!(ids.contains(&"int"));
    assert!(ids.contains(&"main"));
    assert!(ids.contains(&"return"));
}
