//! Basic Unicode matching and character class tests.
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

use super::regex;

// =============================================================================
// Basic Unicode
// =============================================================================

#[test]
fn test_unicode_basic() {
    let re = regex("héllo");
    assert!(re.is_match("héllo world"));
    assert!(!re.is_match("hello world"));
}

// =============================================================================
// Unicode Character Classes (multi-byte UTF-8)
// =============================================================================

#[test]
fn test_unicode_character_class_greek() {
    let re = regex("[α-ω]+");
    assert!(re.is_match("αβγδ"));
    assert!(re.is_match("ωψχ"));
    assert!(!re.is_match("abc"));

    let m = re.find("hello αβγ world").unwrap();
    assert_eq!(m.as_str(), "αβγ");
}

#[test]
fn test_unicode_character_class_mixed() {
    let re = regex("[a-zα-ω]+");
    assert!(re.is_match("hello"));
    assert!(re.is_match("αβγ"));
    assert!(re.is_match("helloαβγ"));

    let m = re.find("123 abcαβγ 456").unwrap();
    assert_eq!(m.as_str(), "abcαβγ");
}

#[test]
fn test_unicode_character_class_cjk() {
    let re = regex("[一-龥]+");
    assert!(re.is_match("中文"));
    assert!(!re.is_match("abc"));
}

#[test]
fn test_unicode_character_class_emoji() {
    let re = regex("[😀-😂]+");
    assert!(re.is_match("😀"));
    assert!(re.is_match("😁"));
    assert!(!re.is_match("abc"));
}
