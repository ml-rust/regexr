//! Negated Unicode character class tests ([^...]).
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

use super::regex;

#[test]
fn test_negated_unicode_class_greek() {
    let re = regex("[^α-ω]+");

    assert!(re.is_match("abc"));
    assert!(re.is_match("XYZ"));
    assert!(re.is_match("123"));
    assert!(re.is_match("中文"));
    assert!(re.is_match("😀"));

    assert!(!re.is_match("αβγ"));
    assert!(!re.is_match("ωψχ"));

    let m = re.find("αβγabcδεζ").unwrap();
    assert_eq!(m.as_str(), "abc");
}

#[test]
fn test_negated_unicode_class_cjk() {
    let re = regex("[^一-龥]+");

    assert!(re.is_match("hello"));
    assert!(re.is_match("world"));
    assert!(re.is_match("αβγ"));

    assert!(!re.is_match("中文"));
    assert!(!re.is_match("汉字"));

    let m = re.find("中文hello世界").unwrap();
    assert_eq!(m.as_str(), "hello");
}

#[test]
fn test_negated_unicode_class_emoji() {
    let re = regex("[^😀-😂]+");

    assert!(re.is_match("hello"));
    assert!(re.is_match("αβγ"));
    assert!(re.is_match("中文"));
    assert!(re.is_match("🎉"));

    assert!(!re.is_match("😀"));
    assert!(!re.is_match("😁"));
    assert!(!re.is_match("😂"));
}

#[test]
fn test_negated_unicode_class_ascii() {
    let re = regex("[^a-z]+");

    assert!(re.is_match("ABC"));
    assert!(re.is_match("123"));
    assert!(re.is_match("!@#"));
    assert!(re.is_match("αβγ"));
    assert!(re.is_match("中文"));
    assert!(re.is_match("😀"));

    assert!(!re.is_match("abc"));
    assert!(!re.is_match("xyz"));

    let m = re.find("abc123xyz").unwrap();
    assert_eq!(m.as_str(), "123");
}

#[test]
fn test_negated_unicode_class_single_char() {
    let re = regex("[^α]+");

    assert!(re.is_match("abc"));
    assert!(re.is_match("βγδ"));
    assert!(re.is_match("中文"));

    assert!(!re.is_match("α"));

    let m = re.find("ααβγδαα").unwrap();
    assert_eq!(m.as_str(), "βγδ");
}

#[test]
fn test_negated_unicode_class_multiple_ranges() {
    let re = regex("[^a-zA-Z]+");

    assert!(re.is_match("123"));
    assert!(re.is_match("!@#$"));
    assert!(re.is_match("αβγ"));
    assert!(re.is_match("中文"));

    assert!(!re.is_match("abc"));
    assert!(!re.is_match("XYZ"));
    assert!(!re.is_match("Hello"));

    let m = re.find("hello123world").unwrap();
    assert_eq!(m.as_str(), "123");
}

#[test]
fn test_negated_unicode_class_find_iter() {
    let re = regex("[^α-ω]+");

    let matches: Vec<_> = re.find_iter("αβγ123δεζabc").collect();
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].as_str(), "123");
    assert_eq!(matches[1].as_str(), "abc");
}

#[test]
fn test_negated_unicode_mixed_with_quantifiers() {
    let re = regex("[^α-ω]*");

    assert!(re.is_match(""));
    assert!(re.is_match("abc"));
    assert!(re.is_match("123"));

    let m = re.find("abc αβγ").unwrap();
    assert_eq!(m.as_str(), "abc ");
}

#[test]
fn test_negated_unicode_class_complex() {
    let re = regex("[^α-ω]+[0-9]+");

    assert!(re.is_match("abc123"));
    assert!(re.is_match("中文456"));
    assert!(re.is_match("αβγ123"));
    assert!(re.is_match("αβγabc123"));

    let re2 = regex("^[^α-ω]+$");
    assert!(!re2.is_match("αβγ"));
    assert!(re2.is_match("abc"));
    assert!(re2.is_match("123"));
}
