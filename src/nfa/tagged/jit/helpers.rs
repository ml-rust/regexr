//! JIT helper functions and context structures.
//!
//! This module provides:
//! - `JitContext` - Runtime context passed to JIT-compiled code
//! - Helper functions callable from JIT code (lookaround evaluation, etc.)

use crate::hir::CodepointClass;
use crate::nfa::Nfa;

/// Runtime context for JIT-compiled Tagged NFA execution.
///
/// This struct is passed to JIT code and provides access to working memory
/// for execution. The layout must match what the JIT code expects.
///
/// Note: Currently unused - will be used when full JIT codegen is implemented.
#[repr(C)]
#[allow(dead_code)]
pub struct JitContext {
    /// Input pointer.
    pub input_ptr: *const u8,
    /// Input length.
    pub input_len: usize,
    /// Best match end position (-1 if no match).
    pub best_match_end: i64,
    /// Pointer to best captures array.
    pub best_captures: *mut i64,
    /// Number of capture slots per thread.
    pub stride: usize,
    /// Current worklist count.
    pub current_count: usize,
    /// Current worklist states.
    pub current_states: *mut u32,
    /// Current worklist captures (flat array, indexed by thread_idx * stride + slot).
    pub current_captures: *mut i64,
    /// Next worklist count.
    pub next_count: usize,
    /// Next worklist states.
    pub next_states: *mut u32,
    /// Next worklist captures.
    pub next_captures: *mut i64,
    /// Visited bitmap.
    pub visited: *mut u64,
    /// Number of bitmap words.
    pub visited_words: usize,
    /// Maximum threads.
    pub max_threads: usize,
}

/// Helper function callable from JIT code to check if a UTF-8 character matches a CodepointClass.
///
/// Arguments:
/// - `input_ptr`: Pointer to the start of the input string
/// - `pos`: Current position in the input
/// - `input_len`: Total length of the input
/// - `cpclass_ptr`: Pointer to the CodepointClass struct
///
/// Returns:
/// - Positive value: The length of the UTF-8 character that matched (1-4 bytes)
/// - 0 or negative: No match (or position out of bounds)
#[allow(dead_code)]
pub unsafe extern "sysv64" fn check_codepoint_class(
    input_ptr: *const u8,
    pos: usize,
    input_len: usize,
    cpclass_ptr: *const CodepointClass,
) -> i64 {
    if pos >= input_len {
        return -1;
    }

    let cpclass = &*cpclass_ptr;
    let input = std::slice::from_raw_parts(input_ptr, input_len);

    // Decode UTF-8 character at position
    let first_byte = input[pos];
    let (codepoint, char_len) = if first_byte < 0x80 {
        // ASCII (1 byte)
        (first_byte as u32, 1)
    } else if first_byte < 0xC0 {
        // Invalid UTF-8 continuation byte
        return -1;
    } else if first_byte < 0xE0 {
        // 2-byte sequence
        if pos + 1 >= input_len {
            return -1;
        }
        let b1 = input[pos + 1];
        if (b1 & 0xC0) != 0x80 {
            return -1;
        }
        let cp = ((first_byte as u32 & 0x1F) << 6) | (b1 as u32 & 0x3F);
        (cp, 2)
    } else if first_byte < 0xF0 {
        // 3-byte sequence
        if pos + 2 >= input_len {
            return -1;
        }
        let b1 = input[pos + 1];
        let b2 = input[pos + 2];
        if (b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 {
            return -1;
        }
        let cp = ((first_byte as u32 & 0x0F) << 12) | ((b1 as u32 & 0x3F) << 6) | (b2 as u32 & 0x3F);
        (cp, 3)
    } else if first_byte < 0xF8 {
        // 4-byte sequence
        if pos + 3 >= input_len {
            return -1;
        }
        let b1 = input[pos + 1];
        let b2 = input[pos + 2];
        let b3 = input[pos + 3];
        if (b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80 {
            return -1;
        }
        let cp = ((first_byte as u32 & 0x07) << 18) | ((b1 as u32 & 0x3F) << 12)
            | ((b2 as u32 & 0x3F) << 6) | (b3 as u32 & 0x3F);
        (cp, 4)
    } else {
        // Invalid UTF-8
        return -1;
    };

    // Check if codepoint is in the class
    if cpclass.contains(codepoint) {
        char_len as i64
    } else {
        -1
    }
}

/// Helper function callable from JIT code to evaluate a positive lookahead assertion.
///
/// Arguments:
/// - `input_ptr`: Pointer to the start of the input string
/// - `pos`: Current position in the input
/// - `input_len`: Total length of the input
/// - `nfa_ptr`: Pointer to the inner NFA for the lookahead
///
/// Returns:
/// - 1 if the lookahead matches (pattern found at position)
/// - 0 if the lookahead does not match
#[allow(dead_code)]
pub unsafe extern "sysv64" fn check_positive_lookahead(
    input_ptr: *const u8,
    pos: usize,
    input_len: usize,
    nfa_ptr: *const Nfa,
) -> i64 {
    if pos > input_len {
        return 0;
    }

    let nfa = &*nfa_ptr;
    let input = std::slice::from_raw_parts(input_ptr, input_len);
    let remaining = &input[pos..];

    // Use PikeVM to check if the pattern matches at the current position
    let vm = crate::vm::PikeVm::new(nfa.clone());
    if vm.is_match(remaining) {
        1
    } else {
        0
    }
}

/// Helper function callable from JIT code to evaluate a negative lookahead assertion.
///
/// Arguments:
/// - `input_ptr`: Pointer to the start of the input string
/// - `pos`: Current position in the input
/// - `input_len`: Total length of the input
/// - `nfa_ptr`: Pointer to the inner NFA for the lookahead
///
/// Returns:
/// - 1 if the lookahead succeeds (pattern NOT found at position)
/// - 0 if the lookahead fails (pattern was found)
#[allow(dead_code)]
pub unsafe extern "sysv64" fn check_negative_lookahead(
    input_ptr: *const u8,
    pos: usize,
    input_len: usize,
    nfa_ptr: *const Nfa,
) -> i64 {
    if pos > input_len {
        return 1; // At invalid position, negative lookahead succeeds
    }

    let nfa = &*nfa_ptr;
    let input = std::slice::from_raw_parts(input_ptr, input_len);
    let remaining = &input[pos..];

    // Use PikeVM to check if the pattern matches at the current position
    let vm = crate::vm::PikeVm::new(nfa.clone());
    if vm.is_match(remaining) {
        0 // Pattern matched, negative lookahead fails
    } else {
        1 // Pattern didn't match, negative lookahead succeeds
    }
}

/// Helper function callable from JIT code to evaluate a positive lookbehind assertion.
///
/// Arguments:
/// - `input_ptr`: Pointer to the start of the input string
/// - `pos`: Current position in the input
/// - `input_len`: Total length of the input (unused but kept for ABI consistency)
/// - `nfa_ptr`: Pointer to the inner NFA for the lookbehind
///
/// Returns:
/// - 1 if the lookbehind matches (pattern found ending at position)
/// - 0 if the lookbehind does not match
#[allow(dead_code)]
pub unsafe extern "sysv64" fn check_positive_lookbehind(
    input_ptr: *const u8,
    pos: usize,
    _input_len: usize,
    nfa_ptr: *const Nfa,
) -> i64 {
    let nfa = &*nfa_ptr;
    let input = std::slice::from_raw_parts(input_ptr, pos);

    // Use PikeVM to check if the pattern matches ending at the current position
    let vm = crate::vm::PikeVm::new(nfa.clone());

    // Try all possible start positions before current position
    for lookback_start in 0..=pos {
        let slice = &input[lookback_start..];
        // Check if pattern matches the entire slice (anchored match)
        if let Some((s, e)) = vm.find(slice) {
            if s == 0 && e == slice.len() {
                return 1;
            }
        }
    }
    0
}

/// Helper function callable from JIT code to evaluate a negative lookbehind assertion.
///
/// Arguments:
/// - `input_ptr`: Pointer to the start of the input string
/// - `pos`: Current position in the input
/// - `input_len`: Total length of the input (unused but kept for ABI consistency)
/// - `nfa_ptr`: Pointer to the inner NFA for the lookbehind
///
/// Returns:
/// - 1 if the lookbehind succeeds (pattern NOT found ending at position)
/// - 0 if the lookbehind fails (pattern was found ending at position)
#[allow(dead_code)]
pub unsafe extern "sysv64" fn check_negative_lookbehind(
    input_ptr: *const u8,
    pos: usize,
    _input_len: usize,
    nfa_ptr: *const Nfa,
) -> i64 {
    let nfa = &*nfa_ptr;
    let input = std::slice::from_raw_parts(input_ptr, pos);

    // Use PikeVM to check if the pattern matches ending at the current position
    let vm = crate::vm::PikeVm::new(nfa.clone());

    // Try all possible start positions before current position
    for lookback_start in 0..=pos {
        let slice = &input[lookback_start..];
        // Check if pattern matches the entire slice (anchored match)
        if let Some((s, e)) = vm.find(slice) {
            if s == 0 && e == slice.len() {
                return 0; // Pattern found, negative lookbehind fails
            }
        }
    }
    1 // Pattern not found, negative lookbehind succeeds
}
