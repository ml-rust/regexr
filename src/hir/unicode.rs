//! Unicode tables and utilities.
//!
//! Contains basic character class definitions.
//! Full Unicode property support will be added with the `unicode-full` feature.

/// Digit characters (ASCII).
pub const DIGITS: &[(u8, u8)] = &[(b'0', b'9')];

/// Word characters (ASCII).
pub const WORD: &[(u8, u8)] = &[
    (b'0', b'9'),
    (b'A', b'Z'),
    (b'a', b'z'),
    (b'_', b'_'),
];

/// Whitespace characters (ASCII).
pub const WHITESPACE: &[(u8, u8)] = &[
    (b'\t', b'\t'),       // Tab
    (b'\n', b'\n'),       // Newline
    (b'\x0B', b'\x0B'),   // Vertical tab
    (b'\x0C', b'\x0C'),   // Form feed
    (b'\r', b'\r'),       // Carriage return
    (b' ', b' '),         // Space
];

/// Returns true if the byte is a word character.
#[inline]
pub fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Returns true if the byte is a whitespace character.
#[inline]
pub fn is_whitespace_byte(b: u8) -> bool {
    matches!(b, b'\t' | b'\n' | b'\x0B' | b'\x0C' | b'\r' | b' ')
}

/// Returns true if the byte is a digit.
#[inline]
pub fn is_digit_byte(b: u8) -> bool {
    b.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_word_byte() {
        assert!(is_word_byte(b'a'));
        assert!(is_word_byte(b'Z'));
        assert!(is_word_byte(b'5'));
        assert!(is_word_byte(b'_'));
        assert!(!is_word_byte(b'-'));
        assert!(!is_word_byte(b' '));
    }

    #[test]
    fn test_is_whitespace_byte() {
        assert!(is_whitespace_byte(b' '));
        assert!(is_whitespace_byte(b'\t'));
        assert!(is_whitespace_byte(b'\n'));
        assert!(!is_whitespace_byte(b'a'));
    }
}
