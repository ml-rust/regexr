//! Email validation pattern tests.
//!
//! Real-world email validation scenarios used in production systems.
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

// Using local mod.rs

use super::regex;

#[test]
fn test_simple_email() {
    let re = regex(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$");

    assert!(re.is_match("user@example.com"));
    assert!(re.is_match("test.user@example.com"));
    assert!(re.is_match("user+tag@example.co.uk"));
    assert!(re.is_match("user_name@example.com"));

    assert!(!re.is_match("invalid@"));
    assert!(!re.is_match("@example.com"));
    assert!(!re.is_match("user@example"));
    assert!(!re.is_match("user example.com"));
}

#[test]
fn test_email_with_subdomains() {
    let re = regex(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$");

    assert!(re.is_match("admin@mail.example.com"));
    assert!(re.is_match("support@subdomain.example.org"));
    assert!(re.is_match("user@a.b.c.example.com"));
}

#[test]
fn test_email_extraction() {
    let re = regex(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}");

    let text = "Contact us at support@example.com or sales@example.org for more info.";
    let emails: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(emails.len(), 2);
    assert_eq!(emails[0], "support@example.com");
    assert_eq!(emails[1], "sales@example.org");
}

#[test]
fn test_email_with_dots_and_plus() {
    let re = regex(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$");

    assert!(re.is_match("first.last@example.com"));
    assert!(re.is_match("user+filter@gmail.com"));
    assert!(re.is_match("user_123@example.org"));
}

#[test]
fn test_email_edge_cases() {
    let re = regex(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$");

    // Valid edge cases
    assert!(re.is_match("a@b.co"));
    assert!(re.is_match("test@example.museum"));

    // Invalid cases
    assert!(!re.is_match("user@.com"));
    assert!(!re.is_match("user@example."));
    assert!(!re.is_match("user @example.com"));
    assert!(!re.is_match("user@exam ple.com"));
}

#[test]
fn test_multiple_emails_in_text() {
    let re = regex(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}");

    let text = "Email bob@example.com, alice@test.org, and charlie@sample.net";
    let count = re.find_iter(text).count();

    assert_eq!(count, 3);
}

#[test]
fn test_email_capture_groups() {
    let re = regex(r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})");

    let caps = re.captures("user@example.com").unwrap();
    assert_eq!(&caps[1], "user");
    assert_eq!(&caps[2], "example.com");
}
