//! Phone number pattern tests.
//!
//! Real-world phone number validation and extraction.
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

// Using local mod.rs

use super::regex;

#[test]
fn test_us_phone_basic() {
    let re = regex(r"^\d{3}-\d{3}-\d{4}$");

    assert!(re.is_match("123-456-7890"));
    assert!(re.is_match("555-123-4567"));

    assert!(!re.is_match("1234567890"));
    assert!(!re.is_match("123-45-6789"));
    assert!(!re.is_match("12-345-6789"));
}

#[test]
fn test_us_phone_with_parentheses() {
    let re = regex(r"^\(\d{3}\)\s*\d{3}-\d{4}$");

    assert!(re.is_match("(123) 456-7890"));
    assert!(re.is_match("(555)123-4567"));
    assert!(re.is_match("(800) 555-1234"));

    assert!(!re.is_match("123-456-7890"));
}

#[test]
fn test_phone_with_country_code() {
    let re = regex(r"^\+1\s*\d{3}-\d{3}-\d{4}$");

    assert!(re.is_match("+1 123-456-7890"));
    assert!(re.is_match("+1123-456-7890"));

    assert!(!re.is_match("123-456-7890"));
    assert!(!re.is_match("+44 123-456-7890"));
}

#[test]
fn test_phone_flexible_format() {
    let re = regex(r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}");

    assert!(re.is_match("123-456-7890"));
    assert!(re.is_match("123.456.7890"));
    assert!(re.is_match("123 456 7890"));
    assert!(re.is_match("1234567890"));
}

#[test]
fn test_phone_extraction_from_text() {
    let re = regex(r"\d{3}-\d{3}-\d{4}");

    let text = "Call me at 555-123-4567 or 555-987-6543 anytime.";
    let phones: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(phones.len(), 2);
    assert_eq!(phones[0], "555-123-4567");
    assert_eq!(phones[1], "555-987-6543");
}

#[test]
fn test_international_phone_basic() {
    let re = regex(r"^\+\d{1,3}\s*\d{1,14}$");

    assert!(re.is_match("+1 1234567890"));
    assert!(re.is_match("+44 1234567890"));
    assert!(re.is_match("+86 1234567890"));
}

#[test]
fn test_phone_area_code_capture() {
    let re = regex(r"(\d{3})-(\d{3})-(\d{4})");

    let caps = re.captures("555-123-4567").unwrap();
    assert_eq!(&caps[1], "555");
    assert_eq!(&caps[2], "123");
    assert_eq!(&caps[3], "4567");
}

#[test]
fn test_phone_with_extension() {
    let re = regex(r"\d{3}-\d{3}-\d{4}\s*(?:ext\.?\s*\d+)?");

    assert!(re.is_match("555-123-4567"));
    assert!(re.is_match("555-123-4567 ext. 123"));
    assert!(re.is_match("555-123-4567 ext 456"));
}
