//! URL parsing pattern tests.
//!
//! Real-world URL validation and extraction scenarios.
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

// Using local mod.rs

use super::regex;

#[test]
fn test_http_urls() {
    let re = regex(r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[^\s]*)?");

    assert!(re.is_match("http://example.com"));
    assert!(re.is_match("https://example.com"));
    assert!(re.is_match("http://www.example.com"));
    assert!(re.is_match("https://sub.example.org"));

    assert!(!re.is_match("ftp://example.com"));
    assert!(!re.is_match("example.com"));
}

#[test]
fn test_urls_with_paths() {
    let re = regex(r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[^\s]*)?");

    assert!(re.is_match("http://example.com/path"));
    assert!(re.is_match("https://example.com/path/to/resource"));
    assert!(re.is_match("http://example.com/page.html"));
}

#[test]
fn test_urls_with_query_params() {
    let re = regex(r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[^\s]*)?");

    assert!(re.is_match("http://example.com?query=value"));
    assert!(re.is_match("https://example.com/page?a=1&b=2"));
    assert!(re.is_match("http://example.com/search?q=test"));
}

#[test]
fn test_url_extraction_from_text() {
    let re = regex(r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[^\s]*)?");

    let text = "Visit https://example.com or http://test.org for more info.";
    let urls: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(urls.len(), 2);
    assert_eq!(urls[0], "https://example.com");
    assert_eq!(urls[1], "http://test.org");
}

#[test]
fn test_url_with_port() {
    let re = regex(r"https?://[a-zA-Z0-9.-]+(:[0-9]+)?(/[^\s]*)?");

    assert!(re.is_match("http://localhost:8080"));
    assert!(re.is_match("https://example.com:443/path"));
    assert!(re.is_match("http://192.168.1.1:3000"));
}

/// This test works - no optional groups needed
#[test]
fn test_url_protocol_capture() {
    let re = regex(r"(https?)://([a-zA-Z0-9.-]+)");

    let caps = re.captures("https://example.com").unwrap();
    assert_eq!(&caps[1], "https");
    assert_eq!(&caps[2], "example.com");

    let caps2 = re.captures("http://test.org").unwrap();
    assert_eq!(&caps2[1], "http");
    assert_eq!(&caps2[2], "test.org");
}

#[test]
fn test_url_in_markdown() {
    let re = regex(r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[^\s\)]*)?");

    let text = "Check out [this link](https://example.com/page) for details.";
    let m = re.find(text).unwrap();
    assert_eq!(m.as_str(), "https://example.com/page");
}

#[test]
fn test_multiple_urls_in_sentence() {
    let re = regex(r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[^\s]*)?");

    let text = "Compare http://site1.com and https://site2.org and http://site3.net today.";
    let count = re.find_iter(text).count();

    assert_eq!(count, 3);
}

// =============================================================================
// Simplified URL Patterns (benchmark patterns)
// =============================================================================

/// Simplified URL extraction pattern from benchmark
/// Pattern: `https?://[^\s<>]+`
#[test]
fn test_url_simplified_extraction() {
    let re = regex(r"https?://[^\s<>]+");

    let text = "Visit https://example.com/path?query=value and http://test.org for info.";
    let urls: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(urls.len(), 2);
    assert_eq!(urls[0], "https://example.com/path?query=value");
    assert_eq!(urls[1], "http://test.org");
}

#[test]
fn test_url_simplified_with_special_chars() {
    let re = regex(r"https?://[^\s<>]+");

    // URLs with various special characters
    assert!(re.is_match("https://example.com/path/to/page"));
    assert!(re.is_match("http://example.com?q=hello&lang=en"));
    assert!(re.is_match("https://example.com/page#section"));
    assert!(re.is_match("http://user:pass@example.com/"));
}

#[test]
fn test_url_simplified_in_html() {
    let re = regex(r"https?://[^\s<>]+");

    // Should stop at < and >
    let html = r#"<a href=https://example.com/page>Link</a>"#;
    let m = re.find(html).unwrap();
    assert_eq!(m.as_str(), "https://example.com/page");
}

#[test]
fn test_url_in_html_with_quotes() {
    // Pattern that also excludes quotes for cleaner HTML parsing
    let re = regex(r#"https?://[^\s<>"']+"#);

    let html = r#"<a href="https://example.com/page">Link</a>"#;
    let m = re.find(html).unwrap();
    assert_eq!(m.as_str(), "https://example.com/page");
}

#[test]
fn test_url_with_alternation_prefix() {
    // Pattern using alternation for protocol (from earlier bug fix)
    let re = regex(r"(?:https|http)://[^\s<>]+");

    let text = "Visit https://example.com and http://test.org for info.";
    let urls: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(urls.len(), 2);
    assert!(urls[0].contains("example.com"));
    assert!(urls[1].contains("test.org"));
}
