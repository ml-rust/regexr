//! Scalar fallback implementations.
//!
//! Used when SIMD is not available.

#![allow(dead_code)]

/// Scalar byte search.
pub fn find_byte(needle: u8, haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| b == needle)
}

/// Scalar multi-byte search.
pub fn find_bytes(needles: &[u8], haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|b| needles.contains(b))
}

/// Scalar substring search.
pub fn find_substring(needle: &[u8], haystack: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > haystack.len() {
        return None;
    }

    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_byte() {
        assert_eq!(find_byte(b'l', b"hello"), Some(2));
        assert_eq!(find_byte(b'x', b"hello"), None);
    }

    #[test]
    fn test_find_substring() {
        assert_eq!(find_substring(b"llo", b"hello"), Some(2));
        assert_eq!(find_substring(b"xyz", b"hello"), None);
        assert_eq!(find_substring(b"", b"hello"), Some(0));
    }
}
