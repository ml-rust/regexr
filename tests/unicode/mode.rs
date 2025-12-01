//! Unicode mode tests ((?u) flag for \w, \d, \s).
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

use super::regex;

#[test]
fn test_unicode_mode_word_class() {
    let re_ascii = regex(r"\w+");
    assert!(re_ascii.is_match("hello"));

    let re_unicode = regex(r"(?u)\w+");
    assert!(re_unicode.is_match("hello"));
    assert!(re_unicode.is_match("αβγ"));
    assert!(re_unicode.is_match("中文"));
}

#[test]
fn test_unicode_mode_digit_class() {
    let re_ascii = regex(r"\d+");
    assert!(re_ascii.is_match("123"));

    let re_unicode = regex(r"(?u)\d+");
    assert!(re_unicode.is_match("123"));
    assert!(re_unicode.is_match("٠١٢"));
}

#[test]
fn test_unicode_mode_whitespace_class() {
    let re_ascii = regex(r"\s+");
    assert!(re_ascii.is_match(" \t\n"));

    let re_unicode = regex(r"(?u)\s+");
    assert!(re_unicode.is_match(" \t\n"));
    assert!(re_unicode.is_match("\u{00A0}"));
    assert!(re_unicode.is_match("\u{3000}"));
}

#[test]
fn test_unicode_mode_negated_classes() {
    let re_unicode = regex(r"(?u)\D+");
    assert!(re_unicode.is_match("abc"));
    assert!(re_unicode.is_match("αβγ"));
    assert!(!re_unicode.is_match("123"));
    assert!(!re_unicode.is_match("٠١٢"));

    let re_unicode_w = regex(r"(?u)\W+");
    assert!(re_unicode_w.is_match("!@#"));
    assert!(re_unicode_w.is_match(" \t\n"));
    assert!(!re_unicode_w.is_match("abc"));
    assert!(!re_unicode_w.is_match("αβγ"));
}
