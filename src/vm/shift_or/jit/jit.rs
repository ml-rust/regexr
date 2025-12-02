//! JitShiftOr - JIT-compiled Shift-Or matcher.
//!
//! This module provides the public API for JIT-compiled Shift-Or matching.

use dynasmrt::ExecutableBuffer;

use super::super::ShiftOr;

#[cfg(target_arch = "x86_64")]
use super::x86_64::ShiftOrJitCompiler;

#[cfg(target_arch = "aarch64")]
use super::aarch64::ShiftOrJitCompiler;

/// JIT-compiled Shift-Or matcher with Glushkov follow sets.
pub struct JitShiftOr {
    /// The compiled code buffer.
    code: ExecutableBuffer,
    /// Offset to the find function.
    find_offset: dynasmrt::AssemblyOffset,
    /// The mask table (256 entries of u64).
    masks: Box<[u64; 256]>,
    /// Follow sets for each position (up to 64 positions).
    /// follow[i] contains a bitmask of positions that can follow position i.
    follow: Box<[u64; 64]>,
    /// Accept mask.
    accept: u64,
    /// First set mask.
    first: u64,
    /// Number of positions in the pattern.
    position_count: usize,
    /// Whether the pattern can match empty string.
    nullable: bool,
    /// Whether pattern has leading word boundary.
    has_leading_wb: bool,
    /// Whether pattern has trailing word boundary.
    has_trailing_wb: bool,
    /// Whether the pattern has a start anchor (^).
    has_start_anchor: bool,
    /// Whether the pattern has an end anchor ($).
    has_end_anchor: bool,
}

impl std::fmt::Debug for JitShiftOr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitShiftOr")
            .field("code_size", &self.code.len())
            .field("position_count", &self.position_count)
            .field("has_leading_wb", &self.has_leading_wb)
            .field("has_trailing_wb", &self.has_trailing_wb)
            .field("has_start_anchor", &self.has_start_anchor)
            .field("has_end_anchor", &self.has_end_anchor)
            .finish()
    }
}

/// Check if a byte is a word character.
#[inline(always)]
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

impl JitShiftOr {
    /// Creates a new JitShiftOr from compiled components.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        code: ExecutableBuffer,
        find_offset: dynasmrt::AssemblyOffset,
        masks: Box<[u64; 256]>,
        follow: Box<[u64; 64]>,
        accept: u64,
        first: u64,
        position_count: usize,
        nullable: bool,
        has_leading_wb: bool,
        has_trailing_wb: bool,
        has_start_anchor: bool,
        has_end_anchor: bool,
    ) -> Self {
        Self {
            code,
            find_offset,
            masks,
            follow,
            accept,
            first,
            position_count,
            nullable,
            has_leading_wb,
            has_trailing_wb,
            has_start_anchor,
            has_end_anchor,
        }
    }

    /// Compiles a ShiftOr matcher to native code.
    pub fn compile(shift_or: &ShiftOr) -> Option<Self> {
        ShiftOrJitCompiler::compile(shift_or)
    }

    /// Finds the first match in the input, returning (start, end).
    #[inline]
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        // For patterns with word boundaries, fall back to Rust implementation
        // Note: This path should never be taken since engine selection
        // doesn't create JitShiftOr for patterns with word boundaries.
        if self.has_leading_wb || self.has_trailing_wb {
            return self.find_with_word_boundaries(input);
        }

        // Handle start anchor: only try matching at position 0
        if self.has_start_anchor {
            if let Some(end) = self.match_at(input) {
                // For end anchor: must match entire input
                if self.has_end_anchor && end != input.len() {
                    return None;
                }
                return Some((0, end));
            }
            // If pattern is nullable with start anchor, return empty match at 0
            if self.nullable {
                if self.has_end_anchor {
                    if input.is_empty() {
                        return Some((0, 0));
                    }
                    return None;
                }
                return Some((0, 0));
            }
            return None;
        }

        // Try to find a non-empty match first (greedy)
        if !input.is_empty() {
            let result = self.call_find(input);
            if result >= 0 {
                // JIT returns packed (start << 32 | end)
                let packed = result as u64;
                let start = (packed >> 32) as usize;
                let end = (packed & 0xFFFF_FFFF) as usize;

                // For end anchor: only accept matches that end at input end
                if self.has_end_anchor && end != input.len() {
                    // Need to search for a match that ends at input.len()
                    return self.find_with_end_anchor(input);
                }
                return Some((start, end));
            }
        }

        // If pattern is nullable and no non-empty match found, return empty match
        if self.nullable {
            if self.has_end_anchor {
                return Some((input.len(), input.len()));
            }
            return Some((0, 0));
        }

        None
    }

    /// Find a match that ends at input.len() (for end anchor).
    fn find_with_end_anchor(&self, input: &[u8]) -> Option<(usize, usize)> {
        for start in 0..=input.len() {
            if let Some(end) = self.match_at(&input[start..]) {
                if start + end == input.len() {
                    return Some((start, start + end));
                }
            }
        }
        if self.nullable {
            return Some((input.len(), input.len()));
        }
        None
    }

    #[inline(always)]
    fn call_find(&self, input: &[u8]) -> i64 {
        // OPTIMIZED: Only 4 parameters (masks/follow are embedded in JIT code)
        // Function signature: fn(input, len, accept, first) -> i64

        // x86_64 Windows uses Microsoft x64 ABI
        #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
        let func: extern "win64" fn(*const u8, usize, u64, u64) -> i64 =
            unsafe { std::mem::transmute(self.code.ptr(self.find_offset)) };

        // x86_64 Unix uses System V AMD64 ABI
        #[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
        let func: extern "sysv64" fn(*const u8, usize, u64, u64) -> i64 =
            unsafe { std::mem::transmute(self.code.ptr(self.find_offset)) };

        // ARM64 uses AAPCS64 on all platforms (extern "C")
        #[cfg(target_arch = "aarch64")]
        let func: extern "C" fn(*const u8, usize, u64, u64) -> i64 =
            unsafe { std::mem::transmute(self.code.ptr(self.find_offset)) };

        func(input.as_ptr(), input.len(), self.accept, self.first)
    }

    /// Find with word boundary handling (fallback to Rust).
    fn find_with_word_boundaries(&self, input: &[u8]) -> Option<(usize, usize)> {
        // Use similar logic to ShiftOr::find but with JIT for inner matching
        for start in 0..input.len() {
            let prev_is_word = if start > 0 {
                is_word_byte(input[start - 1])
            } else {
                false
            };
            let curr_is_word = is_word_byte(input[start]);

            // Check leading word boundary
            if self.has_leading_wb && prev_is_word == curr_is_word {
                continue;
            }

            // Try to match from this position
            if let Some(end) = self.match_at(&input[start..]) {
                let abs_end = start + end;

                // Check trailing word boundary
                if self.has_trailing_wb {
                    let last_is_word = if abs_end > 0 {
                        is_word_byte(input[abs_end - 1])
                    } else {
                        false
                    };
                    let next_is_word = if abs_end < input.len() {
                        is_word_byte(input[abs_end])
                    } else {
                        false
                    };
                    if last_is_word == next_is_word {
                        continue;
                    }
                }

                return Some((start, abs_end));
            }
        }
        None
    }

    fn match_at(&self, input: &[u8]) -> Option<usize> {
        if input.is_empty() {
            return None;
        }

        // Glushkov Shift-Or matching with follow sets
        let mut last_match = None;

        // First byte: can only start at positions in First set
        let mut state = (!self.first) | self.masks[input[0] as usize];
        if (state | self.accept) != !0u64 {
            last_match = Some(1);
        }

        // Remaining bytes: use follow sets for transitions
        for (i, &byte) in input[1..].iter().enumerate() {
            let active = !state; // 1 = active position

            // Compute union of follow sets for all active positions
            let mut reachable = 0u64;
            let mut remaining = active;
            while remaining != 0 {
                let pos = remaining.trailing_zeros() as usize;
                if pos < self.follow.len() {
                    reachable |= self.follow[pos];
                }
                remaining &= remaining - 1; // Clear lowest set bit
            }

            // state = !reachable | mask[byte]
            state = (!reachable) | self.masks[byte as usize];

            if (state | self.accept) != !0u64 {
                last_match = Some(i + 2);
            }

            if state == !0u64 {
                break;
            }
        }

        last_match
    }

    /// Finds a match at or after position.
    pub fn find_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos > input.len() {
            return None;
        }

        // For start anchor, can only match at position 0
        if self.has_start_anchor {
            if pos > 0 {
                return None;
            }
            if let Some(end) = self.match_at(input) {
                if self.has_end_anchor && end != input.len() {
                    return None;
                }
                return Some((0, end));
            }
            return None;
        }

        // For end anchor, only accept matches that end at input.len()
        if self.has_end_anchor {
            for start in pos..=input.len() {
                if let Some(end) = self.match_at(&input[start..]) {
                    if start + end == input.len() {
                        return Some((start, start + end));
                    }
                }
            }
            return None;
        }

        // No anchors: delegate to find on slice and adjust positions
        if pos >= input.len() {
            return None;
        }
        self.find(&input[pos..]).map(|(s, e)| (pos + s, pos + e))
    }

    /// Tries to match at exactly the given position.
    /// Returns (start, end) if matched, None otherwise.
    /// Use this when you know the match should start at exactly `pos` (e.g., from a prefilter).
    pub fn try_match_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        // For start anchor: only position 0 can match
        if self.has_start_anchor && pos != 0 {
            return None;
        }

        if pos > input.len() {
            return None;
        }

        // Check if the pattern matches at exactly position 0 of the slice
        let slice = &input[pos..];
        if let Some(end) = self.match_at(slice) {
            // For end anchor: must match to end of input
            if self.has_end_anchor && pos + end != input.len() {
                return None;
            }
            return Some((pos, pos + end));
        }
        None
    }
}
