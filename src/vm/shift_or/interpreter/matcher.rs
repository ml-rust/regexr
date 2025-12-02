//! Shift-Or interpreter matching implementation.

use super::super::ShiftOr;

/// Interpreter for Shift-Or matching.
///
/// This provides the matching logic for the Shift-Or algorithm,
/// operating on a pre-compiled ShiftOr data structure.
pub struct ShiftOrInterpreter<'a> {
    shift_or: &'a ShiftOr,
}

impl<'a> ShiftOrInterpreter<'a> {
    /// Creates a new interpreter for the given ShiftOr.
    pub fn new(shift_or: &'a ShiftOr) -> Self {
        Self { shift_or }
    }

    /// Returns true if the pattern matches anywhere in the input.
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.find(input).is_some()
    }

    /// Finds the first match, returning (start, end).
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        // If start anchor, only try matching at position 0
        if self.shift_or.has_start_anchor {
            if let Some(end) = self.match_at(input, 0) {
                // If end anchor, must match entire input
                if self.shift_or.has_end_anchor {
                    if end == input.len() {
                        return Some((0, end));
                    }
                    return None;
                }
                return Some((0, end));
            }
            // If pattern is nullable with start anchor, return empty match at 0
            if self.shift_or.nullable {
                if self.shift_or.has_end_anchor {
                    if input.is_empty() {
                        return Some((0, 0));
                    }
                    return None;
                }
                return Some((0, 0));
            }
            return None;
        }

        // If end anchor but no start anchor, try all positions but only accept end matches
        if self.shift_or.has_end_anchor {
            for start in 0..=input.len() {
                if let Some(end) = self.match_at(input, start) {
                    if end == input.len() {
                        return Some((start, end));
                    }
                }
            }
            // If pattern is nullable with end anchor, return empty match at end
            if self.shift_or.nullable {
                return Some((input.len(), input.len()));
            }
            return None;
        }

        // No anchors: try matching at each position, preferring longest match (greedy)
        for start in 0..=input.len() {
            if let Some(end) = self.match_at(input, start) {
                return Some((start, end));
            }
        }

        // If pattern is nullable and no non-empty match found, return empty match at 0
        if self.shift_or.nullable {
            return Some((0, 0));
        }

        None
    }

    /// Finds a match starting at or after the given position.
    /// Returns (start, end) if found.
    pub fn find_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos > input.len() {
            return None;
        }

        // If start anchor, can only match at position 0
        if self.shift_or.has_start_anchor {
            if pos > 0 {
                return None; // Can't find a match after position 0 with start anchor
            }
            if let Some(end) = self.match_at(input, 0) {
                if self.shift_or.has_end_anchor {
                    if end == input.len() {
                        return Some((0, end));
                    }
                    return None;
                }
                return Some((0, end));
            }
            return None;
        }

        // If end anchor, only accept matches that end at input.len()
        if self.shift_or.has_end_anchor {
            for start in pos..=input.len() {
                if let Some(end) = self.match_at(input, start) {
                    if end == input.len() {
                        return Some((start, end));
                    }
                }
            }
            return None;
        }

        // No anchors: try matching at each position from pos
        for start in pos..=input.len() {
            if let Some(end) = self.match_at(input, start) {
                return Some((start, end));
            }
        }
        None
    }

    /// Tries to match at exactly the given position.
    /// Returns (start, end) if matched, None otherwise.
    /// Use this when you know the match should start at exactly `pos` (e.g., from a prefilter).
    pub fn try_match_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        self.match_at(input, pos).map(|end| (pos, end))
    }

    /// Attempts to match at a specific position.
    fn match_at(&self, input: &[u8], start: usize) -> Option<usize> {
        if start > input.len() {
            return None;
        }

        // Track the last match position found
        let mut last_match = None;

        // Check if nullable (empty match)
        if self.shift_or.nullable {
            last_match = Some(start);
        }

        // State tracking using inverted logic:
        // - bit i = 0 means we've reached position i (active)
        // - bit i = 1 means we haven't reached position i (inactive)
        //
        // Initial state: all 1s (no positions reached yet)
        let mut state = !0u64;

        for (i, &byte) in input[start..].iter().enumerate() {
            let byte_mask = self.shift_or.masks[byte as usize];

            if i == 0 {
                // First byte: can only start at positions in First set
                // ~first gives us 0s at First positions, 1s elsewhere
                // Then apply byte mask to filter positions that don't accept this byte
                state = (!self.shift_or.first) | byte_mask;
            } else {
                // Subsequent bytes: use follow sets for transitions
                let mut active = !state; // Flip: 1 = active, 0 = inactive

                // Compute union of follow sets for all active positions
                let mut reachable = 0u64;
                while active != 0 {
                    let pos = active.trailing_zeros() as usize;
                    reachable |= self.shift_or.follow[pos];
                    active &= active - 1; // Clear lowest set bit
                }

                // Invert back to Shift-Or convention (0 = active)
                // Then apply byte mask (positions that don't accept byte become 1)
                state = (!reachable) | byte_mask;
            }

            // Check for match: if any accepting position is reached (bit is 0)
            if (state | self.shift_or.accept) != !0u64 {
                last_match = Some(start + i + 1);
            }

            // If all bits are 1, no possible match from this starting point
            if state == !0u64 {
                break;
            }
        }

        last_match
    }
}
