//! Integration tests for word boundary assertions (\b and \B).
//!
//! Tests verify correct behavior of word boundaries in both interpreted
//! and JIT modes.

use super::regex;

// =============================================================================
// Basic \b word boundary tests
// =============================================================================

#[test]
fn test_word_boundary_at_start() {
    let re = regex(r"\bword");
    assert!(re.is_match("word"));
    assert!(re.is_match("word is here"));
    assert!(!re.is_match("sword"));  // 's' is a word char, no boundary
}

#[test]
fn test_word_boundary_at_end() {
    let re = regex(r"word\b");
    assert!(re.is_match("word"));
    assert!(re.is_match("this is a word"));
    assert!(!re.is_match("words"));  // 's' is a word char, no boundary
}

#[test]
fn test_word_boundary_both_sides() {
    let re = regex(r"\bword\b");
    assert!(re.is_match("word"));
    assert!(re.is_match("a word here"));
    assert!(re.is_match("word "));
    assert!(re.is_match(" word"));
    assert!(!re.is_match("words"));
    assert!(!re.is_match("sword"));
    assert!(!re.is_match("swords"));
}

#[test]
fn test_word_boundary_with_punctuation() {
    let re = regex(r"\bfoo\b");
    // Punctuation is not a word character
    assert!(re.is_match("foo."));
    assert!(re.is_match(".foo"));
    assert!(re.is_match("foo!"));
    assert!(re.is_match("(foo)"));
    assert!(re.is_match("foo,bar"));  // matches "foo"
}

#[test]
fn test_word_boundary_find_position() {
    let re = regex(r"\btest\b");

    let m = re.find("a test case").unwrap();
    assert_eq!(m.start(), 2);
    assert_eq!(m.end(), 6);
    assert_eq!(m.as_str(), "test");

    // At the beginning
    let m = re.find("test case").unwrap();
    assert_eq!(m.start(), 0);

    // At the end
    let m = re.find("this is a test").unwrap();
    assert_eq!(m.start(), 10);
}

// =============================================================================
// Basic \B non-word boundary tests
// =============================================================================

#[test]
fn test_non_word_boundary_basic() {
    let re = regex(r"\Bword");
    // \B matches where there is NOT a word boundary
    assert!(re.is_match("sword"));  // 's' is word char, 'w' is word char - no boundary
    assert!(!re.is_match("word"));  // start of string is a boundary
    assert!(!re.is_match(" word")); // space to word is a boundary
}

#[test]
fn test_non_word_boundary_end() {
    let re = regex(r"word\B");
    assert!(re.is_match("words"));
    assert!(!re.is_match("word"));
    assert!(!re.is_match("word "));
}

#[test]
fn test_non_word_boundary_both_sides() {
    let re = regex(r"\Bor\B");
    assert!(re.is_match("word"));    // w-or-d: both sides are word chars
    assert!(re.is_match("world"));
    assert!(!re.is_match("or"));     // start/end boundaries
    assert!(!re.is_match("for"));    // 'f' to 'o' is not boundary, but 'r' to end is
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn test_word_boundary_empty_string() {
    let re = regex(r"\b");
    // Empty string has no word boundaries (no chars to create boundary)
    assert!(!re.is_match(""));
}

#[test]
fn test_non_word_boundary_empty_string() {
    let re = regex(r"\B");
    // Empty string: at pos=0, prev is non-word, curr is non-word
    // \B matches when both are same (both non-word), so it should match
    assert!(re.is_match(""));
}

#[test]
fn test_word_boundary_with_numbers() {
    let re = regex(r"\b\d+\b");
    assert!(re.is_match("123"));
    assert!(re.is_match("foo 123 bar"));
    assert!(!re.is_match("foo123"));  // no boundary before '1'
    assert!(!re.is_match("123bar"));  // no boundary after '3'
}

#[test]
fn test_word_boundary_underscore() {
    // Underscore is a word character
    let re = regex(r"\bvar_name\b");
    assert!(re.is_match("var_name"));
    assert!(re.is_match(" var_name "));
    assert!(!re.is_match("my_var_name"));
}

// =============================================================================
// Complex patterns with word boundaries
// =============================================================================

#[test]
fn test_word_boundary_alternation() {
    let re = regex(r"\b(foo|bar)\b");
    assert!(re.is_match("foo"));
    assert!(re.is_match("bar"));
    assert!(re.is_match("a foo b"));
    assert!(re.is_match("a bar b"));
    assert!(!re.is_match("foobar"));  // word boundary between foo and bar fails
    assert!(!re.is_match("afoo"));
}

#[test]
fn test_word_boundary_with_quantifiers() {
    let re = regex(r"\b[a-z]+\b");
    assert!(re.is_match("hello"));
    assert!(re.is_match("word world"));

    let m = re.find("hello world").unwrap();
    assert_eq!(m.as_str(), "hello");
}

#[test]
fn test_word_boundary_find_iter() {
    let re = regex(r"\b\w+\b");
    let text = "hello world foo";
    let words: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();
    assert_eq!(words, vec!["hello", "world", "foo"]);
}

// =============================================================================
// JIT-specific tests (explicit JIT usage)
// =============================================================================

#[cfg(feature = "jit")]
mod jit_word_boundary {
    use regexr::RegexBuilder;

    #[test]
    fn test_jit_word_boundary_basic() {
        let re = RegexBuilder::new(r"\btest\b")
            .jit(true)
            .build()
            .unwrap();

        assert!(re.is_match("test"));
        assert!(re.is_match("a test b"));
        assert!(!re.is_match("testing"));
        assert!(!re.is_match("attest"));
    }

    #[test]
    fn test_jit_non_word_boundary() {
        let re = RegexBuilder::new(r"\Btest")
            .jit(true)
            .build()
            .unwrap();

        assert!(re.is_match("attest"));  // 'a' to 't' - both word chars
        assert!(!re.is_match("test"));   // start is a boundary
    }

    #[test]
    fn test_jit_word_boundary_captures() {
        let re = RegexBuilder::new(r"\b(\w+)\b")
            .jit(true)
            .build()
            .unwrap();

        let caps = re.captures("hello world").unwrap();
        assert_eq!(&caps[1], "hello");
    }

    #[test]
    fn test_jit_word_boundary_find() {
        let re = RegexBuilder::new(r"\bfoo\b")
            .jit(true)
            .build()
            .unwrap();

        let m = re.find("say foo bar").unwrap();
        assert_eq!(m.start(), 4);
        assert_eq!(m.end(), 7);
        assert_eq!(m.as_str(), "foo");
    }
}
