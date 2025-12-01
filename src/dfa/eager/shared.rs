//! Shared types for the Eager DFA engine.
//!
//! Contains types used by both the interpreter and potentially a JIT backend.

use super::super::lazy::CharClass;

/// Tagged state encoding constants.
pub const TAG_MATCH: u32 = 1 << 30;
pub const TAG_DEAD: u32 = 1 << 31;
pub const STATE_MASK: u32 = !(TAG_MATCH | TAG_DEAD);
pub const DEAD_STATE: u32 = TAG_DEAD | STATE_MASK;

/// Per-state metadata for end assertion checking.
#[derive(Clone, Copy, Default)]
pub struct StateMetadata {
    /// Character class of the last byte consumed to reach this state.
    pub prev_class: CharClass,
    /// Whether this match state requires a word boundary assertion.
    pub needs_word_boundary: bool,
    /// Whether this match state requires a NOT word boundary assertion.
    pub needs_not_word_boundary: bool,
    /// Whether this match state requires end of text ($) assertion.
    pub needs_end_of_text: bool,
    /// Whether this match state requires end of line (multiline $) assertion.
    pub needs_end_of_line: bool,
}

/// Check if a byte is a word character.
#[inline(always)]
pub fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
