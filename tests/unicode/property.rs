//! Unicode property escape tests (\p{...}, \P{...}).
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

use super::regex;

use regexr::Regex;

// =============================================================================
// General Category Properties
// =============================================================================

#[test]
fn test_unicode_property_letter() {
    let re = regex(r"\p{Letter}+");
    assert!(re.is_match("hello"));
    assert!(re.is_match("αβγ"));
    assert!(re.is_match("中文"));
    assert!(!re.is_match("123"));
    assert!(!re.is_match("!@#"));

    let m = re.find("123abc456").unwrap();
    assert_eq!(m.as_str(), "abc");
}

#[test]
fn test_unicode_property_letter_short() {
    let re = regex(r"\p{L}+");
    assert!(re.is_match("hello"));
    assert!(re.is_match("αβγ"));
    assert!(!re.is_match("123"));
}

#[test]
fn test_unicode_property_number() {
    let re = regex(r"\p{Number}+");
    assert!(re.is_match("123"));
    assert!(!re.is_match("abc"));
    assert!(!re.is_match("αβγ"));
}

#[test]
fn test_unicode_property_decimal_number() {
    let re = regex(r"\p{Nd}+");
    assert!(re.is_match("123"));
    assert!(!re.is_match("abc"));
}

#[test]
fn test_unicode_property_whitespace() {
    let re = regex(r"\p{Whitespace}+");
    assert!(re.is_match(" "));
    assert!(re.is_match("\t"));
    assert!(re.is_match("\n"));
    assert!(!re.is_match("abc"));
}

#[test]
fn test_unicode_property_punctuation() {
    let re = regex(r"\p{Punctuation}+");
    assert!(re.is_match(".,;:"));
    assert!(re.is_match("!?"));
    assert!(!re.is_match("abc"));
    assert!(!re.is_match("123"));
}

// =============================================================================
// Negated Properties
// =============================================================================

#[test]
fn test_negated_unicode_property() {
    let re = regex(r"\P{Letter}+");
    assert!(re.is_match("123"));
    assert!(re.is_match("!@#"));
    assert!(re.is_match("   "));
    assert!(!re.is_match("abc"));
    assert!(!re.is_match("αβγ"));

    let m = re.find("abc123def").unwrap();
    assert_eq!(m.as_str(), "123");
}

// =============================================================================
// Property Name Variations
// =============================================================================

#[test]
fn test_unicode_property_case_insensitive() {
    let re1 = regex(r"\p{letter}+");
    let re2 = regex(r"\p{LETTER}+");
    let re3 = regex(r"\p{Letter}+");

    assert!(re1.is_match("abc"));
    assert!(re2.is_match("abc"));
    assert!(re3.is_match("abc"));
}

#[test]
fn test_unicode_property_normalized_names() {
    let re1 = regex(r"\p{decimal_number}+");
    let re2 = regex(r"\p{decimalnumber}+");

    assert!(re1.is_match("123"));
    assert!(re2.is_match("123"));
}

#[test]
fn test_unknown_unicode_property() {
    let result = Regex::new(r"\p{NotARealProperty}");
    assert!(result.is_err());
}

// =============================================================================
// Combined with Other Patterns
// =============================================================================

#[test]
fn test_unicode_property_in_pattern() {
    let re = regex(r"\p{L}+\s+\p{N}+");
    assert!(re.is_match("hello 123"));
    assert!(re.is_match("αβγ 456"));
    assert!(!re.is_match("123 456"));
}

// =============================================================================
// Sub-category Properties
// =============================================================================

#[test]
fn test_unicode_general_category_subcategories() {
    let re_lu = regex(r"\p{Lu}+");
    assert!(re_lu.is_match("ABC"));
    assert!(!re_lu.is_match("abc"));

    let re_ll = regex(r"\p{Ll}+");
    assert!(re_ll.is_match("abc"));
    assert!(!re_ll.is_match("ABC"));

    let re_nd = regex(r"\p{Nd}+");
    assert!(re_nd.is_match("123"));
    assert!(re_nd.is_match("٠١٢"));
}

// =============================================================================
// Derived Properties
// =============================================================================

#[test]
fn test_unicode_emoji_property() {
    let re = regex(r"\p{Emoji}+");
    assert!(re.is_match("😀"));
    assert!(re.is_match("🎉🎊🎈"));
}

#[test]
fn test_unicode_alphabetic_property() {
    let re = regex(r"\p{Alphabetic}+");
    assert!(re.is_match("hello"));
    assert!(re.is_match("αβγ"));
    assert!(re.is_match("中文"));
    assert!(!re.is_match("123"));
    assert!(!re.is_match("!@#"));
}

#[test]
fn test_unicode_xid_properties() {
    let re_start = regex(r"\p{XID_Start}");
    assert!(re_start.is_match("a"));
    assert!(re_start.is_match("α"));
    assert!(!re_start.is_match("1"));
    assert!(!re_start.is_match("_"));

    let re_continue = regex(r"\p{XID_Continue}+");
    assert!(re_continue.is_match("abc123"));
    assert!(re_continue.is_match("αβγ_δ"));
}
