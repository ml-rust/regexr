//! Unicode script property tests.
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

use super::regex;

#[test]
fn test_unicode_script_greek() {
    let re = regex(r"\p{Greek}+");
    assert!(re.is_match("αβγ"));
    assert!(re.is_match("ΑΒΓΔ"));
    assert!(!re.is_match("abc"));
    assert!(!re.is_match("中文"));
}

#[test]
fn test_unicode_script_han() {
    let re = regex(r"\p{Han}+");
    assert!(re.is_match("中文"));
    assert!(re.is_match("漢字"));
    assert!(!re.is_match("abc"));
    assert!(!re.is_match("αβγ"));
}

#[test]
fn test_unicode_script_latin() {
    let re = regex(r"\p{Latin}+");
    assert!(re.is_match("abc"));
    assert!(re.is_match("XYZ"));
    assert!(re.is_match("àéîõü"));
    assert!(!re.is_match("αβγ"));
    assert!(!re.is_match("中文"));
}

#[test]
fn test_unicode_script_cyrillic() {
    let re = regex(r"\p{Cyrillic}+");
    assert!(re.is_match("абв"));
    assert!(re.is_match("АБВ"));
    assert!(!re.is_match("abc"));
}

#[test]
fn test_unicode_script_arabic() {
    let re = regex(r"\p{Arabic}+");
    assert!(re.is_match("مرحبا"));
    assert!(!re.is_match("abc"));
}

#[test]
fn test_unicode_script_hiragana() {
    let re = regex(r"\p{Hiragana}+");
    assert!(re.is_match("ひらがな"));
    assert!(!re.is_match("カタカナ"));
    assert!(!re.is_match("abc"));
}

#[test]
fn test_unicode_script_katakana() {
    let re = regex(r"\p{Katakana}+");
    assert!(re.is_match("カタカナ"));
    assert!(!re.is_match("ひらがな"));
    assert!(!re.is_match("abc"));
}

#[test]
fn test_unicode_script_hangul() {
    let re = regex(r"\p{Hangul}+");
    assert!(re.is_match("한글"));
    assert!(!re.is_match("abc"));
    assert!(!re.is_match("中文"));
}

#[test]
fn test_negated_script_property() {
    let re = regex(r"\P{Greek}+");
    assert!(re.is_match("abc"));
    assert!(re.is_match("中文"));
    assert!(re.is_match("123"));
    assert!(!re.is_match("αβγ"));

    let m = re.find("αβγabcδεζ").unwrap();
    assert_eq!(m.as_str(), "abc");
}

#[test]
fn test_combined_script_patterns() {
    let re = regex(r"(?:\p{Han}|\p{Hiragana}|\p{Katakana})+");
    assert!(re.is_match("日本語"));
    assert!(re.is_match("中文"));
    assert!(re.is_match("ひらがな"));
    assert!(re.is_match("カタカナ"));
    assert!(!re.is_match("abc"));
}
