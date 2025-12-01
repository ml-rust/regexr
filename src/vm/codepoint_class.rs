//! Fast codepoint-level character class matcher.
//!
//! This matcher operates directly on Unicode codepoints rather than bytes,
//! making it much faster for patterns that are just single character classes
//! like `[α-ω]`, `\p{Greek}`, or `[^α-ω]`.
//!
//! Instead of expanding Unicode ranges into thousands of UTF-8 byte sequences
//! and running an NFA/DFA, we simply iterate through codepoints and check
//! class membership with binary search.

use crate::hir::CodepointClass;

/// A matcher for single codepoint class patterns.
#[derive(Debug, Clone)]
pub struct CodepointClassMatcher {
    class: CodepointClass,
}

impl CodepointClassMatcher {
    /// Creates a new codepoint class matcher.
    pub fn new(class: CodepointClass) -> Self {
        Self { class }
    }

    /// Returns true if any character in the input matches the class.
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.find(input).is_some()
    }

    /// Finds the first character matching the class, returning (start, end) byte positions.
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        let s = std::str::from_utf8(input).ok()?;

        for (byte_idx, ch) in s.char_indices() {
            let cp = ch as u32;
            if self.class.contains(cp) {
                return Some((byte_idx, byte_idx + ch.len_utf8()));
            }
        }
        None
    }

    /// Returns captures for the first match (just the full match for a char class).
    pub fn captures(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        self.find(input).map(|m| vec![Some(m)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_class() {
        // [a-z]
        let class = CodepointClass::new(vec![(0x61, 0x7A)], false);
        let matcher = CodepointClassMatcher::new(class);

        assert!(matcher.is_match(b"hello"));
        assert!(matcher.is_match(b"123abc"));
        assert!(!matcher.is_match(b"123"));
        assert_eq!(matcher.find(b"123abc"), Some((3, 4)));
    }

    #[test]
    fn test_greek_class() {
        // [α-ω] (Greek lowercase)
        let class = CodepointClass::new(vec![(0x03B1, 0x03C9)], false);
        let matcher = CodepointClassMatcher::new(class);

        assert!(matcher.is_match("αβγ".as_bytes()));
        assert!(matcher.is_match("hello α world".as_bytes()));
        assert!(!matcher.is_match(b"hello world"));

        // α is at byte position 6 in "hello α", takes 2 bytes
        let result = matcher.find("hello α".as_bytes());
        assert_eq!(result, Some((6, 8)));
    }

    #[test]
    fn test_negated_greek_class() {
        // [^α-ω]
        let class = CodepointClass::new(vec![(0x03B1, 0x03C9)], true);
        let matcher = CodepointClassMatcher::new(class);

        // Should match any non-Greek character
        assert!(matcher.is_match(b"hello"));
        assert!(matcher.is_match("αβγhello".as_bytes()));
        assert!(!matcher.is_match("αβγ".as_bytes()));

        // First non-Greek char in "αβγhello" is 'h' at byte 6
        let result = matcher.find("αβγhello".as_bytes());
        assert_eq!(result, Some((6, 7)));
    }

    #[test]
    fn test_empty_input() {
        let class = CodepointClass::new(vec![(0x61, 0x7A)], false);
        let matcher = CodepointClassMatcher::new(class);

        assert!(!matcher.is_match(b""));
        assert_eq!(matcher.find(b""), None);
    }

    #[test]
    fn test_invalid_utf8() {
        let class = CodepointClass::new(vec![(0x61, 0x7A)], false);
        let matcher = CodepointClassMatcher::new(class);

        // Invalid UTF-8 should return no match
        assert!(!matcher.is_match(&[0xFF, 0xFE]));
        assert_eq!(matcher.find(&[0xFF, 0xFE]), None);
    }
}
