//! Date and time pattern tests.
//!
//! Real-world date/time validation and extraction scenarios.
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

// Using local mod.rs

use super::regex;

#[test]
fn test_date_yyyy_mm_dd() {
    let re = regex(r"^\d{4}-\d{2}-\d{2}$");

    assert!(re.is_match("2024-01-15"));
    assert!(re.is_match("2023-12-31"));
    assert!(re.is_match("2000-02-29"));

    assert!(!re.is_match("24-01-15"));
    assert!(!re.is_match("2024-1-15"));
    assert!(!re.is_match("2024/01/15"));
}

#[test]
fn test_date_mm_dd_yyyy() {
    let re = regex(r"^\d{2}/\d{2}/\d{4}$");

    assert!(re.is_match("01/15/2024"));
    assert!(re.is_match("12/31/2023"));

    assert!(!re.is_match("2024/01/15"));
    assert!(!re.is_match("1/15/2024"));
}

#[test]
fn test_time_24hour() {
    let re = regex(r"^\d{2}:\d{2}:\d{2}$");

    assert!(re.is_match("14:30:00"));
    assert!(re.is_match("00:00:00"));
    assert!(re.is_match("23:59:59"));

    // Note: Regex doesn't validate semantic correctness
    // 24:00:00 matches the format, just not semantically valid
    assert!(re.is_match("24:00:00")); // Format matches
    assert!(!re.is_match("14:30")); // Missing seconds
    assert!(!re.is_match("2:30:00")); // Single digit hour
}

#[test]
fn test_time_12hour() {
    let re = regex(r"^\d{1,2}:\d{2}\s*(?:AM|PM)$");

    assert!(re.is_match("2:30 PM"));
    assert!(re.is_match("12:00 AM"));
    assert!(re.is_match("11:59 PM"));

    // Note: Regex doesn't validate semantic correctness
    // 13:30 PM matches the format, just not semantically valid for 12-hour time
    assert!(re.is_match("13:30 PM")); // Format matches
    assert!(!re.is_match("2:30")); // Missing AM/PM
}

#[test]
fn test_datetime_iso8601() {
    let re = regex(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}");

    assert!(re.is_match("2024-01-15T14:30:00"));
    assert!(re.is_match("2023-12-31T23:59:59"));
    assert!(re.is_match("2024-01-15T14:30:00Z"));
    assert!(re.is_match("2024-01-15T14:30:00+00:00"));
}

#[test]
fn test_date_extraction_from_text() {
    let re = regex(r"\d{4}-\d{2}-\d{2}");

    let text = "The event is on 2024-03-15 and ends on 2024-03-20.";
    let dates: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(dates.len(), 2);
    assert_eq!(dates[0], "2024-03-15");
    assert_eq!(dates[1], "2024-03-20");
}

#[test]
fn test_date_with_month_names() {
    let re = regex(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}");

    assert!(re.is_match("Jan 15, 2024"));
    assert!(re.is_match("Dec 31, 2023"));
    assert!(re.is_match("May 1, 2000"));

    assert!(!re.is_match("January 15, 2024"));
}

#[test]
fn test_datetime_capture_groups() {
    let re = regex(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})");

    let caps = re.captures("2024-03-15 14:30:00").unwrap();
    assert_eq!(&caps[1], "2024"); // year
    assert_eq!(&caps[2], "03"); // month
    assert_eq!(&caps[3], "15"); // day
    assert_eq!(&caps[4], "14"); // hour
    assert_eq!(&caps[5], "30"); // minute
    assert_eq!(&caps[6], "00"); // second
}

#[test]
fn test_relative_date_patterns() {
    let re = regex(r"\d+\s+(?:days?|weeks?|months?|years?)\s+ago");

    assert!(re.is_match("3 days ago"));
    assert!(re.is_match("1 week ago"));
    assert!(re.is_match("5 months ago"));
    assert!(re.is_match("2 years ago"));
}

#[test]
fn test_timestamp_unix() {
    let re = regex(r"^\d{10}$");

    assert!(re.is_match("1706198400")); // Unix timestamp
    assert!(re.is_match("1234567890"));

    assert!(!re.is_match("123456789")); // Too short
    assert!(!re.is_match("12345678901")); // Too long
}

// =============================================================================
// Log Parsing Patterns (benchmark patterns)
// =============================================================================

/// Log line parsing pattern with capture groups
/// Pattern: `(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)`
#[test]
fn test_log_line_parsing() {
    let re = regex(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)");

    let log = "2024-01-15 14:30:45 [INFO] Application started successfully";
    let caps = re.captures(log).unwrap();

    assert_eq!(&caps[1], "2024-01-15");
    assert_eq!(&caps[2], "14:30:45");
    assert_eq!(&caps[3], "INFO");
    assert_eq!(&caps[4], "Application started successfully");
}

#[test]
fn test_log_line_parsing_various_levels() {
    let re = regex(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)");

    // Test different log levels
    assert!(re.is_match("2024-01-15 10:00:00 [DEBUG] Debug message"));
    assert!(re.is_match("2024-01-15 10:00:01 [INFO] Info message"));
    assert!(re.is_match("2024-01-15 10:00:02 [WARNING] Warning message"));
    assert!(re.is_match("2024-01-15 10:00:03 [ERROR] Error message"));
    assert!(re.is_match("2024-01-15 10:00:04 [CRITICAL] Critical message"));
}

#[test]
fn test_log_line_extraction() {
    let re = regex(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)");

    let logs = r#"2024-01-15 10:30:00 [INFO] Server starting
2024-01-15 10:30:01 [DEBUG] Loading configuration
2024-01-15 10:30:02 [ERROR] Failed to connect to database
2024-01-15 10:30:03 [INFO] Retrying connection"#;

    let count = re.find_iter(logs).count();
    assert_eq!(count, 4);
}

#[test]
fn test_log_level_extraction() {
    let re = regex(r"\[(\w+)\]");

    let log = "2024-01-15 14:30:45 [ERROR] Something went wrong";
    let caps = re.captures(log).unwrap();
    assert_eq!(&caps[1], "ERROR");
}
