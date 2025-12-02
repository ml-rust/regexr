//! Tokenization pattern tests for LLM and NLP contexts.
//!
//! This module contains comprehensive tests for tokenization patterns commonly
//! used in Large Language Models (LLMs) and Natural Language Processing (NLP).
//! These patterns are critical for handling diverse text inputs including:
//! - ASCII control characters
//! - Unicode whitespace variations
//! - Word boundaries and segmentation
//! - Context-aware lookahead/lookbehind
//! - Emoji and special characters
//! - Code/programming language tokens
//! - Punctuation and delimiters
//! - Mixed-script text handling
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

// Using local mod.rs

use super::regex;

// =============================================================================
// ASCII Control Character Handling
// =============================================================================

#[test]
fn test_ascii_control_chars() {
    let re = regex(r"[\x00-\x1F\x7F]");

    // Control characters should match
    assert!(re.is_match("\x00")); // NUL
    assert!(re.is_match("\x01")); // SOH
    assert!(re.is_match("\x1B")); // ESC
    assert!(re.is_match("\x7F")); // DEL

    // Regular characters should not match
    assert!(!re.is_match("a"));
    assert!(!re.is_match(" "));
}

#[test]
fn test_strip_control_chars() {
    let re = regex(r"[\x00-\x1F\x7F]+");

    let text = "Hello\x00World\x1B";
    let result = re.replace_all(text, "");
    assert_eq!(result, "HelloWorld");
}

#[test]
fn test_preserve_important_control_chars() {
    // Pattern that matches control chars except tab, newline, carriage return
    let re = regex(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]");

    assert!(!re.is_match("\t")); // Tab (0x09)
    assert!(!re.is_match("\n")); // Newline (0x0A)
    assert!(!re.is_match("\r")); // Carriage return (0x0D)

    assert!(re.is_match("\x00")); // NUL
    assert!(re.is_match("\x1F")); // Unit separator
}

// =============================================================================
// Whitespace Tokenization (Unicode-aware)
// =============================================================================

#[test]
fn test_unicode_whitespace() {
    let re = regex(r"\p{Whitespace}+");

    // ASCII whitespace
    assert!(re.is_match(" "));
    assert!(re.is_match("\t"));
    assert!(re.is_match("\n"));
    assert!(re.is_match("\r"));

    // Unicode whitespace
    assert!(re.is_match("\u{00A0}")); // Non-breaking space
    assert!(re.is_match("\u{2000}")); // En quad
    assert!(re.is_match("\u{2003}")); // Em space
    assert!(re.is_match("\u{3000}")); // Ideographic space
}

#[test]
fn test_split_on_whitespace() {
    let re = regex(r"\p{Whitespace}+");

    let text = "Hello\u{2003}World\u{00A0}Test";
    let parts: Vec<&str> = text
        .split(|c: char| re.is_match(&c.to_string()))
        .filter(|s| !s.is_empty())
        .collect();

    assert!(!parts.is_empty());
}

#[test]
fn test_normalize_whitespace() {
    let re = regex(r"\p{Whitespace}+");

    let text = "Hello\u{2003}\u{2003}World\u{00A0}\u{00A0}Test";
    let normalized = re.replace_all(text, " ");

    assert_eq!(normalized, "Hello World Test");
}

// =============================================================================
// Word Boundary Detection
// =============================================================================

#[test]
fn test_word_boundaries_basic() {
    let re = regex(r"\b\w+\b");

    let matches: Vec<_> = re
        .find_iter("hello world test")
        .map(|m| m.as_str())
        .collect();
    assert_eq!(matches, vec!["hello", "world", "test"]);
}

#[test]
fn test_word_boundaries_with_punctuation() {
    let re = regex(r"\b\w+\b");

    let matches: Vec<_> = re
        .find_iter("Hello, world! How's it?")
        .map(|m| m.as_str())
        .collect();
    assert!(matches.contains(&"Hello"));
    assert!(matches.contains(&"world"));
    assert!(matches.contains(&"How"));
    assert!(matches.contains(&"s"));
    assert!(matches.contains(&"it"));
}

#[test]
fn test_word_boundaries_unicode() {
    let re = regex(r"\b\w+\b");

    let text = "café naïve résumé";
    let count = re.find_iter(text).count();
    assert!(count >= 3);
}

#[test]
fn test_alphanumeric_tokens() {
    let re = regex(r"[a-zA-Z0-9]+");

    let matches: Vec<_> = re
        .find_iter("test123 hello456 world")
        .map(|m| m.as_str())
        .collect();
    assert_eq!(matches, vec!["test123", "hello456", "world"]);
}

// =============================================================================
// Lookahead/Lookbehind for Context-Aware Tokenization
// =============================================================================

#[test]
fn test_word_followed_by_punctuation() {
    let re = regex(r"\w+(?=[.,!?])");

    let matches: Vec<_> = re
        .find_iter("Hello, world! How are you?")
        .map(|m| m.as_str())
        .collect();
    assert!(matches.contains(&"Hello"));
    assert!(matches.contains(&"world"));
    assert!(matches.contains(&"you"));
}

#[test]
fn test_word_after_whitespace() {
    let re = regex(r"(?<=\s)\w+");

    let matches: Vec<_> = re
        .find_iter("hello world test")
        .map(|m| m.as_str())
        .collect();
    assert!(matches.contains(&"world"));
    assert!(matches.contains(&"test"));
    assert!(!matches.contains(&"hello")); // First word has no preceding space
}

#[test]
fn test_quoted_strings() {
    let re = regex(r#""[^"]+""#);

    let matches: Vec<_> = re
        .find_iter(r#"He said "hello" and "goodbye"."#)
        .map(|m| m.as_str())
        .collect();
    assert_eq!(matches, vec![r#""hello""#, r#""goodbye""#]);
}

#[test]
fn test_context_aware_apostrophe() {
    // Match word contractions (don't, can't, it's)
    let re = regex(r"\b\w+'\w+\b");

    let matches: Vec<_> = re
        .find_iter("Don't can't won't it's")
        .map(|m| m.as_str())
        .collect();
    assert!(matches.contains(&"Don't") || matches.contains(&"don't"));
    assert!(matches.contains(&"can't"));
}

// =============================================================================
// Emoji and Special Character Handling
// =============================================================================

#[test]
fn test_emoji_detection() {
    let re = regex(r"\p{Emoji}+");

    assert!(re.is_match("😀"));
    assert!(re.is_match("🎉"));
    assert!(re.is_match("👍"));
    assert!(re.is_match("❤️"));

    assert!(!re.is_match("hello"));
    // Note: In Unicode, \p{Emoji} includes ASCII digits 0-9 (#, *, 0-9)
    // because they can be emoji when followed by U+FE0F (emoji variation selector)
    // This is correct Unicode behavior per UAX #44
}

#[test]
fn test_emoji_extraction() {
    let re = regex(r"\p{Emoji}+");

    let text = "Hello 😀 World 🎉 Test 👍";
    let emojis: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert!(emojis.len() >= 3);
}

#[test]
fn test_emoji_with_modifiers() {
    let re = regex(r"\p{Emoji}+");

    assert!(re.is_match("👋🏻")); // Waving hand with skin tone
    assert!(re.is_match("👨‍👩‍👧‍👦")); // Family emoji with ZWJ
}

#[test]
fn test_special_symbols() {
    let re = regex(r"[\p{Symbol}\p{Punctuation}]+");

    assert!(re.is_match("©"));
    assert!(re.is_match("®"));
    assert!(re.is_match("™"));
    assert!(re.is_match("€"));
    assert!(re.is_match("¥"));
    assert!(re.is_match("@#$%"));
}

// =============================================================================
// Code/Programming Language Token Patterns
// =============================================================================

#[test]
fn test_identifier_tokens() {
    let re = regex(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b");

    let code = "var foo_bar = my_function(arg1, arg2);";
    let matches: Vec<_> = re.find_iter(code).map(|m| m.as_str()).collect();

    assert!(matches.contains(&"var"));
    assert!(matches.contains(&"foo_bar"));
    assert!(matches.contains(&"my_function"));
    assert!(matches.contains(&"arg1"));
    assert!(matches.contains(&"arg2"));
}

#[test]
fn test_number_literals() {
    let re = regex(r"\b\d+\.?\d*\b");

    let code = "x = 42, y = 3.14, z = 100";
    let matches: Vec<_> = re.find_iter(code).map(|m| m.as_str()).collect();

    assert!(matches.contains(&"42"));
    assert!(matches.contains(&"3.14"));
    assert!(matches.contains(&"100"));
}

#[test]
fn test_hex_literals() {
    let re = regex(r"\b0x[0-9a-fA-F]+\b");

    let code = "color = 0xFF00FF, value = 0xDEADBEEF";
    let matches: Vec<_> = re.find_iter(code).map(|m| m.as_str()).collect();

    assert_eq!(matches, vec!["0xFF00FF", "0xDEADBEEF"]);
}

#[test]
fn test_string_literals_double() {
    let re = regex(r#""(?:[^"\\]|\\.)*""#);

    let code = r#"msg = "Hello \"World\"", path = "C:\\Users\\test""#;
    let matches: Vec<_> = re.find_iter(code).map(|m| m.as_str()).collect();

    assert_eq!(matches.len(), 2);
}

#[test]
fn test_string_literals_single() {
    let re = regex(r"'(?:[^'\\]|\\.)*'");

    let code = r"msg = 'Hello \'World\'', char = 'a'";
    let matches: Vec<_> = re.find_iter(code).map(|m| m.as_str()).collect();

    assert_eq!(matches.len(), 2);
}

#[test]
fn test_comment_single_line() {
    let re = regex(r"//[^\n]*");

    let code = "x = 5; // This is a comment\ny = 10; // Another comment";
    let matches: Vec<_> = re.find_iter(code).map(|m| m.as_str()).collect();

    assert_eq!(matches.len(), 2);
    assert!(matches[0].contains("This is a comment"));
    assert!(matches[1].contains("Another comment"));
}

#[test]
fn test_operators() {
    let re = regex(r"[+\-*/=<>!&|]+");

    let code = "x = a + b - c * d / e";
    let matches: Vec<_> = re.find_iter(code).map(|m| m.as_str()).collect();

    assert!(matches.contains(&"+"));
    assert!(matches.contains(&"-"));
    assert!(matches.contains(&"*"));
    assert!(matches.contains(&"/"));
    assert!(matches.contains(&"="));
}

// =============================================================================
// Punctuation and Delimiter Handling
// =============================================================================

#[test]
fn test_sentence_boundaries() {
    let re = regex(r"[.!?]+");

    let text = "Hello world! How are you? I'm fine.";
    let matches: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(matches, vec!["!", "?", "."]);
}

#[test]
fn test_comma_separated() {
    let re = regex(r"[^,]+");

    let text = "apple,banana,cherry,date";
    let matches: Vec<_> = re.find_iter(text).map(|m| m.as_str().trim()).collect();

    assert_eq!(matches, vec!["apple", "banana", "cherry", "date"]);
}

#[test]
fn test_bracket_matching() {
    let re = regex(r"[(\[{].*?[)\]}]");

    let text = "array[0] func(x) dict{key}";
    let count = re.find_iter(text).count();

    // Non-greedy would give 3 matches, greedy gives 1
    assert!(count >= 3);
}

#[test]
fn test_delimiter_types() {
    let re = regex(r"[\p{Punctuation}]");

    let text = "Hello, world! How's it? (fine) {great} [awesome]";
    let delimiters: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert!(delimiters.contains(&","));
    assert!(delimiters.contains(&"!"));
    assert!(delimiters.contains(&"?"));
    assert!(delimiters.contains(&"("));
    assert!(delimiters.contains(&")"));
}

// =============================================================================
// Number and Digit Patterns (including Unicode)
// =============================================================================

#[test]
fn test_ascii_digits() {
    let re = regex(r"\d+");

    let text = "There are 42 apples and 100 oranges";
    let matches: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(matches, vec!["42", "100"]);
}

#[test]
fn test_unicode_digits() {
    let re = regex(r"\p{Nd}+");

    // Arabic-Indic digits
    assert!(re.is_match("٠١٢٣٤٥٦٧٨٩"));

    // Devanagari digits
    assert!(re.is_match("०१२३४५६७८९"));

    // ASCII digits
    assert!(re.is_match("0123456789"));
}

#[test]
fn test_decimal_numbers() {
    let re = regex(r"\d+\.\d+");

    let text = "pi is 3.14159 and e is 2.71828";
    let matches: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(matches, vec!["3.14159", "2.71828"]);
}

#[test]
fn test_scientific_notation() {
    let re = regex(r"\d+\.?\d*[eE][+-]?\d+");

    assert!(re.is_match("1.23e10"));
    assert!(re.is_match("4.56E-5"));
    assert!(re.is_match("7e8"));
}

#[test]
fn test_number_with_separators() {
    let re = regex(r"\d{1,3}(?:,\d{3})+");

    assert!(re.is_match("1,000"));
    assert!(re.is_match("1,000,000"));
    assert!(re.is_match("123,456,789"));
}

// =============================================================================
// Mixed Script Text Handling
// =============================================================================

#[test]
fn test_latin_cjk_mixed() {
    let re_latin = regex(r"\p{Latin}+");
    let re_han = regex(r"\p{Han}+");

    let text = "Hello 你好 World 世界";

    let latin: Vec<_> = re_latin.find_iter(text).map(|m| m.as_str()).collect();
    let han: Vec<_> = re_han.find_iter(text).map(|m| m.as_str()).collect();

    assert!(latin.contains(&"Hello"));
    assert!(latin.contains(&"World"));
    assert!(han.contains(&"你好"));
    assert!(han.contains(&"世界"));
}

#[test]
fn test_latin_arabic_mixed() {
    let re_latin = regex(r"\p{Latin}+");
    let re_arabic = regex(r"\p{Arabic}+");

    let text = "Hello مرحبا World";

    assert!(re_latin.is_match(text));
    assert!(re_arabic.is_match(text));
}

#[test]
fn test_latin_cyrillic_mixed() {
    let re_latin = regex(r"\p{Latin}+");
    let re_cyrillic = regex(r"\p{Cyrillic}+");

    let text = "Hello Привет World";

    let latin: Vec<_> = re_latin.find_iter(text).map(|m| m.as_str()).collect();
    let cyrillic: Vec<_> = re_cyrillic.find_iter(text).map(|m| m.as_str()).collect();

    assert!(latin.contains(&"Hello"));
    assert!(latin.contains(&"World"));
    assert!(cyrillic.contains(&"Привет"));
}

#[test]
fn test_script_detection() {
    // Note: "日本語" is Han (Kanji) + Hiragana mix, but "語" is actually Han
    // Use proper test strings for each script
    let re_latin = regex(r"\p{Latin}+");
    let re_han = regex(r"\p{Han}+");
    let re_hiragana = regex(r"\p{Hiragana}+");
    let re_hangul = regex(r"\p{Hangul}+");
    let re_arabic = regex(r"\p{Arabic}+");

    assert!(re_latin.is_match("English"));
    assert!(re_han.is_match("中文"));
    assert!(re_hiragana.is_match("ひらがな")); // Pure Hiragana
    assert!(re_hangul.is_match("한글"));
    assert!(re_arabic.is_match("العربية"));
}

#[test]
fn test_mixed_script_word_segmentation() {
    let re = regex(r"\p{L}+");

    let text = "Hello世界Test中文Mix";
    let segments: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    // Should segment based on script boundaries
    assert!(!segments.is_empty());
}

// =============================================================================
// Unicode Normalization Patterns
// =============================================================================

#[test]
fn test_combining_marks() {
    let re = regex(r"\p{M}+");

    // Combining diacritical marks
    assert!(re.is_match("\u{0301}")); // Combining acute accent
    assert!(re.is_match("\u{0308}")); // Combining diaeresis

    assert!(!re.is_match("a"));
}

#[test]
fn test_grapheme_clusters() {
    let re = regex(r"\X");

    // Should match grapheme clusters
    assert!(re.is_match("é")); // e + combining acute
    assert!(re.is_match("ñ")); // n + combining tilde
}

#[test]
fn test_zero_width_joiners() {
    let text = "👨‍👩‍👧‍👦"; // Family emoji with ZWJ

    let re = regex(r"\p{Emoji}");
    assert!(re.is_match(text));
}

// =============================================================================
// Edge Cases and Real-World Scenarios
// =============================================================================

#[test]
fn test_html_tags_removal() {
    let re = regex(r"<[^>]+>");

    let html = "Hello <b>world</b> <a href='test'>link</a>";
    let cleaned = re.replace_all(html, "");

    assert_eq!(cleaned, "Hello world link");
}

#[test]
fn test_markdown_code_blocks() {
    let re = regex(r"`[^`]+`");

    let text = "Use `print()` and `input()` functions";
    let matches: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(matches, vec!["`print()`", "`input()`"]);
}

#[test]
fn test_camelcase_split() {
    let re = regex(r"[A-Z][a-z]+");

    let text = "CamelCaseExample";
    let matches: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(matches, vec!["Camel", "Case", "Example"]);
}

#[test]
fn test_hashtag_extraction() {
    let re = regex(r"#\w+");

    let text = "Check out #regex #programming #rust";
    let hashtags: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(hashtags, vec!["#regex", "#programming", "#rust"]);
}

#[test]
fn test_mention_extraction() {
    let re = regex(r"@\w+");

    let text = "Hey @user123 and @developer, check this out!";
    let mentions: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(mentions, vec!["@user123", "@developer"]);
}

/// Test URL detection in text - using simpler pattern
#[test]
fn test_url_in_text() {
    // Use alternation and explicit whitespace chars (no \s in char class)
    let re = regex(r"(?:https|http)://[^ \t\n\r]+");

    let text = "Visit https://example.com and http://test.org for info.";
    let urls: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert!(urls.len() >= 2);
    // URLs may include trailing punctuation, so check contains
    assert!(urls.iter().any(|u| u.contains("example.com")));
    assert!(urls.iter().any(|u| u.contains("test.org")));
}

#[test]
fn test_newline_types() {
    let re = regex(r"\r?\n|\r");

    assert!(re.is_match("\n")); // Unix
    assert!(re.is_match("\r\n")); // Windows
    assert!(re.is_match("\r")); // Old Mac
}

#[test]
fn test_consecutive_duplicates() {
    let re = regex(r"(\w)\1+");

    assert!(re.is_match("hello")); // 'll'
    assert!(re.is_match("book")); // 'oo'
    assert!(re.is_match("aaa")); // 'aa' or 'aaa'
}

#[test]
fn test_acronym_detection() {
    let re = regex(r"\b[A-Z]{2,}\b");

    let text = "NASA and FBI work with USA on AI";
    let acronyms: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert!(acronyms.contains(&"NASA"));
    assert!(acronyms.contains(&"FBI"));
    assert!(acronyms.contains(&"USA"));
    assert!(acronyms.contains(&"AI"));
}

// =============================================================================
// CL100K_BASE Tokenizer Pattern Tests (OpenAI GPT-4/3.5)
// =============================================================================

/// The actual cl100k_base pattern used by OpenAI's tiktoken tokenizer.
/// This is a complex pattern with alternations, lookahead, and Unicode properties.
const CL100K_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Regression test: Negative lookahead with greedy quantifier backtracking.
/// Issue: `\s+(?!\S)` was not backtracking properly, causing incorrect tokenization.
///
/// For "hello   world", the correct tokenization is:
/// - "hello" (word)
/// - "  " (2 spaces - matched by `\s+(?!\S)` which stops before the last space)
/// - " world" (space + word - matched by `[^\r\n\p{L}\p{N}]?\p{L}+`)
///
/// The bug was that `\s+` consumed all 3 spaces without backtracking when `(?!\S)` failed.
#[test]
fn test_cl100k_whitespace_tokenization() {
    let re = regex(CL100K_PATTERN);

    // Test: multiple spaces before a word
    let text = "hello   world";
    let tokens: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    // Expected: ["hello", "  ", " world"]
    // NOT: ["hello", "   ", "world"]
    assert_eq!(tokens.len(), 3, "Should have 3 tokens");
    assert_eq!(tokens[0], "hello", "First token should be 'hello'");
    assert_eq!(tokens[1], "  ", "Second token should be 2 spaces (not 3)");
    assert_eq!(
        tokens[2], " world",
        "Third token should be ' world' (space + word)"
    );
}

/// Test whitespace-heavy text tokenization matches expected behavior.
#[test]
fn test_cl100k_mixed_whitespace() {
    let re = regex(CL100K_PATTERN);

    let text = "hello   world\t\ttest\n\n\nmore   text";
    let tokens: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    // Expected tokenization:
    // "hello", "  ", " world", "\t", "\ttest", "\n\n\n", "more", "  ", " text"
    assert_eq!(tokens.len(), 9, "Should have 9 tokens");
    assert_eq!(tokens[0], "hello");
    assert_eq!(tokens[1], "  "); // 2 spaces
    assert_eq!(tokens[2], " world"); // space + word
    assert_eq!(tokens[3], "\t"); // 1 tab
    assert_eq!(tokens[4], "\ttest"); // tab + word
    assert_eq!(tokens[5], "\n\n\n"); // 3 newlines
    assert_eq!(tokens[6], "more");
    assert_eq!(tokens[7], "  "); // 2 spaces
    assert_eq!(tokens[8], " text"); // space + word
}

/// Test the isolated negative lookahead pattern for whitespace.
#[test]
fn test_whitespace_negative_lookahead_backtracking() {
    let re = regex(r"\s+(?!\S)");

    // "  w" - 2 spaces followed by 'w' (non-whitespace)
    // Should match only 1 space (backtrack until (?!\S) succeeds)
    let text = "  w";
    let m = re.find(text);
    assert!(m.is_some(), "Should find a match");
    assert_eq!(m.unwrap().as_str(), " ", "Should match only 1 space");

    // "   word" - 3 spaces followed by word
    // Should match 2 spaces (position 2 is still space, so (?!\S) succeeds)
    let text2 = "   word";
    let m2 = re.find(text2);
    assert!(m2.is_some(), "Should find a match");
    assert_eq!(m2.unwrap().as_str(), "  ", "Should match 2 spaces");

    // "   " - 3 spaces at end (no following character)
    // Should match all 3 spaces (end of string is not \S)
    let text3 = "   ";
    let m3 = re.find(text3);
    assert!(m3.is_some(), "Should find a match");
    assert_eq!(m3.unwrap().as_str(), "   ", "Should match all 3 spaces");
}

/// Test contractions in cl100k pattern.
#[test]
fn test_cl100k_contractions() {
    let re = regex(CL100K_PATTERN);

    let text = "I'm you're they've we'll";
    let tokens: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    // Contractions should be split: word + contraction suffix
    assert!(tokens.contains(&"'m"), "Should contain 'm");
    assert!(tokens.contains(&"'re"), "Should contain 're");
    assert!(tokens.contains(&"'ve"), "Should contain 've");
    assert!(tokens.contains(&"'ll"), "Should contain 'll");
}

/// Test number chunking in cl100k pattern.
#[test]
fn test_cl100k_numbers() {
    let re = regex(CL100K_PATTERN);

    // Numbers are split into chunks of 1-3 digits
    let text = "123456789";
    let tokens: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    // Should be split into 3-digit chunks
    assert_eq!(tokens, vec!["123", "456", "789"]);
}

// =============================================================================
// O200K_BASE Tokenizer Pattern Tests (OpenAI GPT-4o)
// =============================================================================

/// The actual o200k_base pattern used by OpenAI's GPT-4o tokenizer.
/// This pattern is more complex than cl100k_base, with additional Unicode
/// property matching for better handling of international text.
const O200K_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Basic o200k tokenization test.
#[test]
fn test_o200k_basic_tokenization() {
    let re = regex(O200K_PATTERN);

    let text = "Hello world";
    let tokens: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0], "Hello");
    assert_eq!(tokens[1], " world");
}

/// Test o200k with whitespace (same as cl100k behavior).
#[test]
fn test_o200k_whitespace_tokenization() {
    let re = regex(O200K_PATTERN);

    let text = "hello   world";
    let tokens: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert_eq!(tokens.len(), 3, "Should have 3 tokens");
    assert_eq!(tokens[0], "hello", "First token should be 'hello'");
    assert_eq!(tokens[1], "  ", "Second token should be 2 spaces");
    assert_eq!(tokens[2], " world", "Third token should be ' world'");
}

/// Test o200k contractions.
#[test]
fn test_o200k_contractions() {
    let re = regex(O200K_PATTERN);

    let text = "I'm you're they've we'll";
    let tokens: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    // o200k keeps contractions attached to words
    assert!(tokens.iter().any(|t| t.contains("'m")));
    assert!(tokens.iter().any(|t| t.contains("'re")));
    assert!(tokens.iter().any(|t| t.contains("'ve")));
    assert!(tokens.iter().any(|t| t.contains("'ll")));
}

/// Test o200k number chunking.
#[test]
fn test_o200k_numbers() {
    let re = regex(O200K_PATTERN);

    let text = "123456789";
    let tokens: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    // Should be split into 3-digit chunks
    assert_eq!(tokens, vec!["123", "456", "789"]);
}

/// Regression test: UTF-8 boundary handling with em-dash.
/// This test catches bugs where regex match positions fall inside
/// multi-byte UTF-8 characters.
#[test]
fn test_o200k_utf8_em_dash() {
    let re = regex(O200K_PATTERN);

    // Em-dash (—) is 3 bytes in UTF-8
    let text = "word—word";
    let tokens: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    // Should not panic and should produce valid tokens
    assert!(!tokens.is_empty());

    // Verify we can reconstruct the original text
    let reconstructed: String = tokens.concat();
    assert_eq!(reconstructed, text);
}

/// Regression test: UTF-8 boundary with curly quotes.
#[test]
fn test_o200k_utf8_curly_quotes() {
    let re = regex(O200K_PATTERN);

    // Curly quotes are 3 bytes each in UTF-8
    let text = "He said, \u{2018}Hello\u{2019} and \u{201c}Goodbye\u{201d}.";
    let tokens: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert!(!tokens.is_empty());

    let reconstructed: String = tokens.concat();
    assert_eq!(reconstructed, text);
}

/// Regression test: Multiple em-dashes in sequence.
#[test]
fn test_o200k_utf8_multiple_em_dashes() {
    let re = regex(O200K_PATTERN);

    let texts = [
        "word—word",
        "a—b",
        "test—",
        "—start",
        "one—two—three",
        "Check your brake pads or rotors—they might be worn out.",
        "I'm sorry you're hurting—breakups suck, but you'll get through it.",
        "Check if you're using valid credentials—API key, token—in headers.",
    ];

    for text in texts {
        let tokens: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();
        assert!(!tokens.is_empty(), "Should find tokens in: {}", text);

        let reconstructed: String = tokens.concat();
        assert_eq!(
            reconstructed, text,
            "Should reconstruct original text: {}",
            text
        );
    }
}

/// Test o200k with mixed Unicode scripts.
#[test]
fn test_o200k_mixed_unicode() {
    let re = regex(O200K_PATTERN);

    let text = "Hello 你好 World 世界";
    let tokens: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert!(!tokens.is_empty());

    let reconstructed: String = tokens.concat();
    assert_eq!(reconstructed, text);
}

/// Test o200k with emoji.
#[test]
fn test_o200k_emoji() {
    let re = regex(O200K_PATTERN);

    let text = "Hello 😀 World 🎉";
    let tokens: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();

    assert!(!tokens.is_empty());

    let reconstructed: String = tokens.concat();
    assert_eq!(reconstructed, text);
}
