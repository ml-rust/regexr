//! Shared types for backtracking engine.
//!
//! Contains the bytecode instructions and helper functions used by both
//! interpreter and JIT backends.

/// Bytecode instructions for the backtracking VM.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub enum Op {
    /// Match a single byte.
    Byte(u8),
    /// Match a byte range [lo, hi].
    ByteRange(u8, u8),
    /// Match any byte except these (negated class with up to 4 ranges inline).
    NotByteRanges { count: u8, ranges: [(u8, u8); 4] },
    /// Match any byte in these ranges (up to 4 ranges inline).
    ByteRanges { count: u8, ranges: [(u8, u8); 4] },
    /// Match any byte in a large class (index into byte_classes table).
    ByteClassRef { index: u16, negated: bool },
    /// Match a Unicode codepoint range.
    CpRange(u32, u32),
    /// Negated Unicode codepoint range.
    NotCpRange(u32, u32),
    /// Match a Unicode codepoint class (index into cp_classes table).
    CpClassRef { index: u16, negated: bool },
    /// Match any byte.
    Any,
    /// Split: try pc+1 first, on backtrack try target.
    Split(u32),
    /// Jump to target.
    Jump(u32),
    /// Save position to capture slot.
    Save(u16),
    /// Match (success).
    Match,
    /// Start anchor (^).
    StartAnchor,
    /// End anchor ($).
    EndAnchor,
    /// Word boundary.
    WordBoundary,
    /// Not word boundary.
    NotWordBoundary,
    /// Backreference to group N.
    Backref(u16),
}

/// Decode UTF-8 codepoint from bytes.
/// Returns (codepoint, byte_length) if valid.
#[inline]
pub fn decode_utf8(bytes: &[u8]) -> Option<(u32, usize)> {
    if bytes.is_empty() {
        return None;
    }
    let b0 = bytes[0];
    if b0 < 0x80 {
        return Some((b0 as u32, 1));
    }
    if bytes.len() < 2 {
        return None;
    }
    let b1 = bytes[1];
    if (b0 & 0xE0) == 0xC0 {
        return Some((((b0 as u32 & 0x1F) << 6) | (b1 as u32 & 0x3F), 2));
    }
    if bytes.len() < 3 {
        return None;
    }
    let b2 = bytes[2];
    if (b0 & 0xF0) == 0xE0 {
        return Some((
            ((b0 as u32 & 0x0F) << 12) | ((b1 as u32 & 0x3F) << 6) | (b2 as u32 & 0x3F),
            3,
        ));
    }
    if bytes.len() < 4 {
        return None;
    }
    let b3 = bytes[3];
    if (b0 & 0xF8) == 0xF0 {
        return Some((
            ((b0 as u32 & 0x07) << 18)
                | ((b1 as u32 & 0x3F) << 12)
                | ((b2 as u32 & 0x3F) << 6)
                | (b3 as u32 & 0x3F),
            4,
        ));
    }
    None
}

/// Returns true if the byte is a word character (alphanumeric or underscore).
#[inline]
pub fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
