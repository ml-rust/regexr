//! Markup and structured text pattern tests.
//!
//! Patterns for parsing HTML tags, JSON strings, and other structured text formats.
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

use super::regex;

// =============================================================================
// HTML Tag Patterns
// =============================================================================

/// HTML tag stripping pattern from benchmark
/// Pattern: `<[^>]+>`
#[test]
fn test_html_tag_basic() {
    let re = regex(r"<[^>]+>");

    assert!(re.is_match("<p>"));
    assert!(re.is_match("</p>"));
    assert!(re.is_match("<div class=\"test\">"));
    assert!(re.is_match("<br/>"));
    assert!(re.is_match("<img src=\"image.png\" alt=\"test\">"));

    assert!(!re.is_match("plain text"));
    assert!(!re.is_match("<>"));  // Empty tag
}

#[test]
fn test_html_tag_extraction() {
    let re = regex(r"<[^>]+>");

    let html = "<html><head><title>Test</title></head><body><p>Hello</p></body></html>";
    let tags: Vec<_> = re.find_iter(html).map(|m| m.as_str()).collect();

    assert_eq!(tags.len(), 10);
    assert_eq!(tags[0], "<html>");
    assert_eq!(tags[1], "<head>");
    assert_eq!(tags[2], "<title>");
}

#[test]
fn test_html_tag_stripping() {
    let re = regex(r"<[^>]+>");

    let html = "<p>Hello <b>World</b>!</p>";
    let stripped = re.replace_all(html, "");
    assert_eq!(stripped, "Hello World!");
}

#[test]
fn test_html_self_closing_tags() {
    let re = regex(r"<[^>]+>");

    assert!(re.is_match("<br>"));
    assert!(re.is_match("<br/>"));
    assert!(re.is_match("<br />"));
    assert!(re.is_match("<hr>"));
    assert!(re.is_match("<img src=\"x\">"));
}

#[test]
fn test_html_tag_with_attributes() {
    let re = regex(r"<[^>]+>");

    let html = r#"<div id="main" class="container" data-value="123">"#;
    let m = re.find(html).unwrap();
    assert_eq!(m.as_str(), html);
}

// =============================================================================
// JSON String Patterns
// =============================================================================

/// JSON string extraction pattern from benchmark
/// Pattern: `"([^"\\]|\\.)*"`
#[test]
fn test_json_string_basic() {
    let re = regex(r#""([^"\\]|\\.)*""#);

    assert!(re.is_match(r#""hello""#));
    assert!(re.is_match(r#""hello world""#));
    assert!(re.is_match(r#""""#));  // Empty string

    assert!(!re.is_match("hello"));  // No quotes
    assert!(!re.is_match(r#""unclosed"#));  // Unclosed
}

#[test]
fn test_json_string_with_escapes() {
    let re = regex(r#""([^"\\]|\\.)*""#);

    // Escaped characters
    assert!(re.is_match(r#""hello\"world""#));  // Escaped quote
    assert!(re.is_match(r#""path\\to\\file""#));  // Escaped backslash
    assert!(re.is_match(r#""line1\nline2""#));  // Escaped newline
    assert!(re.is_match(r#""tab\there""#));  // Escaped tab
}

#[test]
fn test_json_string_extraction() {
    let re = regex(r#""([^"\\]|\\.)*""#);

    let json = r#"{"name": "John", "city": "New York"}"#;
    let strings: Vec<_> = re.find_iter(json).map(|m| m.as_str()).collect();

    assert_eq!(strings.len(), 4);
    assert_eq!(strings[0], r#""name""#);
    assert_eq!(strings[1], r#""John""#);
    assert_eq!(strings[2], r#""city""#);
    assert_eq!(strings[3], r#""New York""#);
}

#[test]
fn test_json_string_in_array() {
    let re = regex(r#""([^"\\]|\\.)*""#);

    let json = r#"["apple", "banana", "cherry"]"#;
    let strings: Vec<_> = re.find_iter(json).map(|m| m.as_str()).collect();

    assert_eq!(strings.len(), 3);
    assert_eq!(strings[0], r#""apple""#);
    assert_eq!(strings[1], r#""banana""#);
    assert_eq!(strings[2], r#""cherry""#);
}

#[test]
fn test_json_string_unicode() {
    let re = regex(r#""([^"\\]|\\.)*""#);

    assert!(re.is_match(r#""hello \u0041""#));  // Unicode escape
    assert!(re.is_match(r#""emoji: 😀""#));  // Direct emoji
    assert!(re.is_match(r#""日本語""#));  // Japanese
}

// =============================================================================
// Backreference Patterns (matching quotes)
// =============================================================================

/// Quote matching pattern with backreference from benchmark
/// Pattern: `(['"])[^'"]*\1`
#[test]
fn test_matching_quotes() {
    let re = regex(r#"(['"])[^'"]*\1"#);

    // Matching quotes
    assert!(re.is_match(r#"'hello'"#));
    assert!(re.is_match(r#""world""#));
    assert!(re.is_match(r#"'test string'"#));
    assert!(re.is_match(r#""another test""#));

    // Mismatched quotes should not match
    assert!(!re.is_match(r#"'hello""#));
    assert!(!re.is_match(r#""world'"#));
}

#[test]
fn test_matching_quotes_extraction() {
    let re = regex(r#"(['"])[^'"]*\1"#);

    let code = r#"let x = 'hello'; let y = "world";"#;
    let strings: Vec<_> = re.find_iter(code).map(|m| m.as_str()).collect();

    assert_eq!(strings.len(), 2);
    assert_eq!(strings[0], "'hello'");
    assert_eq!(strings[1], "\"world\"");
}

#[test]
fn test_matching_quotes_empty() {
    let re = regex(r#"(['"])[^'"]*\1"#);

    assert!(re.is_match(r#"''"#));
    assert!(re.is_match(r#""""#));
}

// =============================================================================
// XML/Markup Patterns
// =============================================================================

#[test]
fn test_xml_comment() {
    let re = regex(r"<!--[^-]*(?:-[^-]+)*-->");

    assert!(re.is_match("<!-- comment -->"));
    assert!(re.is_match("<!---->"));
    // Note: for multiline, a different approach would be needed
}

#[test]
fn test_xml_cdata() {
    let re = regex(r"<!\[CDATA\[.*?\]\]>");

    assert!(re.is_match("<![CDATA[content]]>"));
    assert!(re.is_match("<![CDATA[<html>code</html>]]>"));
}

#[test]
fn test_xml_processing_instruction() {
    let re = regex(r"<\?[^?]+\?>");

    assert!(re.is_match("<?xml version=\"1.0\"?>"));
    assert!(re.is_match("<?php echo 'hello'; ?>"));
}

// =============================================================================
// Markdown Patterns
// =============================================================================

#[test]
fn test_markdown_header() {
    let re = regex(r"^#{1,6}\s+.+$");

    assert!(re.is_match("# Header 1"));
    assert!(re.is_match("## Header 2"));
    assert!(re.is_match("###### Header 6"));

    assert!(!re.is_match("####### Too many"));
    assert!(!re.is_match("#NoSpace"));
}

#[test]
fn test_markdown_link() {
    let re = regex(r"\[([^\]]+)\]\(([^)]+)\)");

    let md = "[Click here](https://example.com)";
    let caps = re.captures(md).unwrap();

    assert_eq!(&caps[1], "Click here");
    assert_eq!(&caps[2], "https://example.com");
}

#[test]
fn test_markdown_code_block() {
    let re = regex(r"```\w*");

    assert!(re.is_match("```"));
    assert!(re.is_match("```python"));
    assert!(re.is_match("```javascript"));
    assert!(re.is_match("```rust"));
}
