//! Shared types for Shift-Or engine.
//!
//! Contains the ShiftOr data structure used by both interpreter and JIT.

use crate::hir::Hir;
use crate::nfa::{
    compile_glushkov, compile_glushkov_wide, BitSet256, GlushkovNfa, GlushkovWideNfa,
    MAX_POSITIONS, MAX_POSITIONS_WIDE,
};

/// A compiled Shift-Or pattern.
///
/// This is a data structure that holds the precomputed masks and follow sets
/// for the Shift-Or (Bitap) algorithm. The actual matching is performed by
/// either the interpreter or JIT.
///
/// **CRITICAL**: This implementation uses Glushkov NFA (epsilon-free), NOT Thompson NFA.
/// Thompson's epsilon-transitions break the 1-shift = 1-byte invariant.
///
/// Unlike classic Shift-Or which assumes linear position progression (i -> i+1),
/// this implementation uses explicit follow sets from Glushkov construction to
/// handle patterns like `a.*b` where nullable subexpressions create non-linear
/// transitions.
///
/// ## Limitations
///
/// ShiftOr does NOT support:
/// - Anchors (`^`, `$`)
/// - Word boundaries (`\b`, `\B`) - use LazyDFA instead
/// - Backreferences - use PikeVM or BacktrackingVM instead
/// - Lookaround - use PikeVM instead
/// - Non-greedy quantifiers (`.*?`, `.+?`) - Glushkov doesn't preserve match preference
/// - Patterns with more than 64 positions
#[derive(Debug)]
pub struct ShiftOr {
    /// Bit masks for each byte value.
    /// mask[b] has bit i cleared (0) if position i can transition on byte b.
    /// (Shift-Or uses inverted logic: 0 = "can be in this state")
    pub(crate) masks: [u64; 256],
    /// Accept state mask (inverted: 0 bits = accepting positions).
    pub(crate) accept: u64,
    /// First set: positions that can start a match.
    pub(crate) first: u64,
    /// Follow sets: follow[i] indicates which positions can follow position i.
    pub(crate) follow: Vec<u64>,
    /// Whether the pattern can match empty string.
    pub(crate) nullable: bool,
    /// Number of positions.
    pub(crate) position_count: usize,
    /// Whether the pattern has a leading word boundary (\b at start).
    pub(crate) has_leading_word_boundary: bool,
    /// Whether the pattern has a trailing word boundary (\b at end).
    pub(crate) has_trailing_word_boundary: bool,
}

impl ShiftOr {
    /// Tries to compile an HIR into a Shift-Or matcher.
    /// Returns None if the pattern is not suitable for Shift-Or.
    pub fn from_hir(hir: &Hir) -> Option<Self> {
        // Skip patterns with special features that can't be handled
        // Anchors (^, $), backrefs, lookarounds, and word boundaries require different engines.
        // Word boundaries (\b, \B) are complex to handle correctly in shift-or;
        // LazyDFA handles them properly with character-class augmented states.
        // Non-greedy quantifiers (.*?, .+?) require tracking match preference which
        // Glushkov NFA doesn't preserve - use TaggedNFA or PikeVM instead.
        if hir.props.has_backrefs
            || hir.props.has_lookaround
            || hir.props.has_anchors
            || hir.props.has_word_boundary
            || hir.props.has_non_greedy
        {
            return None;
        }

        // Build Glushkov NFA (epsilon-free)
        let glushkov = compile_glushkov(hir)?;

        Self::from_glushkov_with_boundaries(&glushkov, false, false)
    }

    /// Creates a Shift-Or matcher from a Glushkov NFA.
    pub fn from_glushkov(nfa: &GlushkovNfa) -> Option<Self> {
        Self::from_glushkov_with_boundaries(nfa, false, false)
    }

    /// Creates a Shift-Or matcher from a Glushkov NFA with word boundary info.
    fn from_glushkov_with_boundaries(
        nfa: &GlushkovNfa,
        has_leading_word_boundary: bool,
        has_trailing_word_boundary: bool,
    ) -> Option<Self> {
        if nfa.position_count > MAX_POSITIONS || nfa.position_count == 0 {
            return None;
        }

        let masks = nfa.build_shift_or_masks();
        let accept = nfa.build_accept_mask();

        Some(Self {
            masks,
            accept,
            first: nfa.first,
            follow: nfa.follow.clone(),
            nullable: nfa.nullable,
            position_count: nfa.position_count,
            has_leading_word_boundary,
            has_trailing_word_boundary,
        })
    }

    /// Returns true if this pattern has word boundaries.
    /// Note: ShiftOr no longer accepts patterns with word boundaries,
    /// so this always returns false for valid ShiftOr instances.
    #[inline]
    pub fn has_word_boundary(&self) -> bool {
        self.has_leading_word_boundary || self.has_trailing_word_boundary
    }

    /// Returns the number of positions.
    pub fn state_count(&self) -> usize {
        self.position_count
    }

    /// Returns the masks table.
    pub fn masks(&self) -> &[u64; 256] {
        &self.masks
    }

    /// Returns the accept mask.
    pub fn accept(&self) -> u64 {
        self.accept
    }

    /// Returns the first set.
    pub fn first(&self) -> u64 {
        self.first
    }

    /// Returns the follow sets.
    pub fn follow(&self) -> &[u64] {
        &self.follow
    }

    /// Returns whether the pattern is nullable.
    pub fn is_nullable(&self) -> bool {
        self.nullable
    }

    /// Returns whether there's a leading word boundary.
    pub fn has_leading_word_boundary(&self) -> bool {
        self.has_leading_word_boundary
    }

    /// Returns whether there's a trailing word boundary.
    pub fn has_trailing_word_boundary(&self) -> bool {
        self.has_trailing_word_boundary
    }

    // ========================================================================
    // Convenience matching methods (delegate to interpreter)
    // ========================================================================

    /// Returns true if the pattern matches anywhere in the input.
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.find(input).is_some()
    }

    /// Finds the first match, returning (start, end).
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        // Handle nullable patterns (can match empty string)
        if self.nullable {
            return Some((0, 0));
        }

        // Try matching at each position
        for start in 0..=input.len() {
            if let Some(end) = self.match_at(input, start) {
                return Some((start, end));
            }
        }
        None
    }

    /// Finds a match starting at or after the given position.
    /// Returns (start, end) if found.
    pub fn find_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos > input.len() {
            return None;
        }

        // Try matching at each position from pos
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
        if self.nullable {
            last_match = Some(start);
        }

        // State tracking using inverted logic:
        // - bit i = 0 means we've reached position i (active)
        // - bit i = 1 means we haven't reached position i (inactive)
        //
        // Initial state: all 1s (no positions reached yet)
        let mut state = !0u64;

        for (i, &byte) in input[start..].iter().enumerate() {
            let byte_mask = self.masks[byte as usize];

            if i == 0 {
                // First byte: can only start at positions in First set
                // ~first gives us 0s at First positions, 1s elsewhere
                // Then apply byte mask to filter positions that don't accept this byte
                state = (!self.first) | byte_mask;
            } else {
                // Subsequent bytes: use follow sets for transitions
                let mut active = !state; // Flip: 1 = active, 0 = inactive

                // Compute union of follow sets for all active positions
                let mut reachable = 0u64;
                while active != 0 {
                    let pos = active.trailing_zeros() as usize;
                    reachable |= self.follow[pos];
                    active &= active - 1; // Clear lowest set bit
                }

                // Invert back to Shift-Or convention (0 = active)
                // Then apply byte mask (positions that don't accept byte become 1)
                state = (!reachable) | byte_mask;
            }

            // Check for match: if any accepting position is reached (bit is 0)
            if (state | self.accept) != !0u64 {
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

/// Checks if an HIR is suitable for Shift-Or.
pub fn is_shift_or_compatible(hir: &Hir) -> bool {
    // Anchors (^, $), backrefs, lookarounds, and word boundaries require different engines.
    // Word boundaries (\b, \B) are complex to handle correctly in shift-or;
    // LazyDFA handles them properly with character-class augmented states.
    // Non-greedy quantifiers (.*?, .+?) require tracking match preference which
    // Glushkov NFA doesn't preserve - use TaggedNFA or PikeVM instead.
    if hir.props.has_backrefs
        || hir.props.has_lookaround
        || hir.props.has_anchors
        || hir.props.has_word_boundary
        || hir.props.has_non_greedy
    {
        return false;
    }

    // Try to build Glushkov NFA to check position count
    compile_glushkov(hir)
        .map(|nfa| nfa.position_count <= MAX_POSITIONS && nfa.position_count > 0)
        .unwrap_or(false)
}

// ============================================================================
// Wide Shift-Or (supports up to 256 positions)
// ============================================================================

/// A compiled Wide Shift-Or pattern supporting up to 256 positions.
///
/// Uses `[u64; 4]` (BitSet256) for state vectors instead of `u64`,
/// allowing patterns with 65-256 character positions to use the efficient
/// bit-parallel Shift-Or algorithm instead of falling back to PikeVM.
///
/// Performance notes:
/// - For patterns with ≤64 positions, use `ShiftOr` (faster due to single u64)
/// - For patterns with 65-256 positions, use `ShiftOrWide`
/// - For patterns with >256 positions, use LazyDFA or PikeVM
#[derive(Debug)]
pub struct ShiftOrWide {
    /// Bit masks for each byte value (256-bit wide).
    pub(crate) masks: Box<[BitSet256; 256]>,
    /// Accept state mask (inverted: 0 bits = accepting positions).
    pub(crate) accept: BitSet256,
    /// First set: positions that can start a match.
    pub(crate) first: BitSet256,
    /// Follow sets: follow[i] indicates which positions can follow position i.
    pub(crate) follow: Vec<BitSet256>,
    /// Whether the pattern can match empty string.
    pub(crate) nullable: bool,
    /// Number of positions.
    pub(crate) position_count: usize,
}

impl ShiftOrWide {
    /// Tries to compile an HIR into a Wide Shift-Or matcher.
    /// Returns None if the pattern is not suitable.
    pub fn from_hir(hir: &Hir) -> Option<Self> {
        // Skip patterns with special features that can't be handled
        if hir.props.has_backrefs
            || hir.props.has_lookaround
            || hir.props.has_anchors
            || hir.props.has_word_boundary
            || hir.props.has_non_greedy
        {
            return None;
        }

        // Build Wide Glushkov NFA
        let glushkov = compile_glushkov_wide(hir)?;

        Self::from_glushkov(&glushkov)
    }

    /// Creates a Wide Shift-Or matcher from a Wide Glushkov NFA.
    pub fn from_glushkov(nfa: &GlushkovWideNfa) -> Option<Self> {
        if nfa.position_count > MAX_POSITIONS_WIDE || nfa.position_count == 0 {
            return None;
        }

        let masks = Box::new(nfa.build_shift_or_masks());
        let accept = nfa.build_accept_mask();

        Some(Self {
            masks,
            accept,
            first: nfa.first,
            follow: nfa.follow.clone(),
            nullable: nfa.nullable,
            position_count: nfa.position_count,
        })
    }

    /// Returns the number of positions.
    pub fn state_count(&self) -> usize {
        self.position_count
    }

    /// Returns whether the pattern is nullable.
    pub fn is_nullable(&self) -> bool {
        self.nullable
    }

    // ========================================================================
    // Convenience matching methods (delegate to interpreter)
    // ========================================================================

    /// Returns true if the pattern matches anywhere in the input.
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.find(input).is_some()
    }

    /// Finds the first match, returning (start, end).
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        if self.nullable {
            return Some((0, 0));
        }

        for start in 0..=input.len() {
            if let Some(end) = self.match_at(input, start) {
                return Some((start, end));
            }
        }
        None
    }

    /// Finds a match starting at or after the given position.
    pub fn find_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos > input.len() {
            return None;
        }

        for start in pos..=input.len() {
            if let Some(end) = self.match_at(input, start) {
                return Some((start, end));
            }
        }
        None
    }

    /// Tries to match at exactly the given position.
    pub fn try_match_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        self.match_at(input, pos).map(|end| (pos, end))
    }

    /// Core matching logic using 256-bit state vectors.
    fn match_at(&self, input: &[u8], start: usize) -> Option<usize> {
        if start > input.len() {
            return None;
        }

        let mut last_match = None;

        if self.nullable {
            last_match = Some(start);
        }

        // State tracking using inverted logic (same as u64 version):
        // - bit i = 0 means we've reached position i (active)
        // - bit i = 1 means we haven't reached position i (inactive)
        let mut state = BitSet256::all_ones();

        for (i, &byte) in input[start..].iter().enumerate() {
            let byte_mask = self.masks[byte as usize];

            if i == 0 {
                // First byte: can only start at positions in First set
                state = self.first.complement().union(byte_mask);
            } else {
                // Subsequent bytes: use follow sets for transitions
                // Flip state: 1 = active, 0 = inactive
                let active = state.complement();

                // Compute union of follow sets for all active positions
                let mut reachable = BitSet256::empty();

                // Iterate over all 4 words to find active positions
                for word_idx in 0..4 {
                    let mut word = active.parts[word_idx];
                    while word != 0 {
                        let bit_idx = word.trailing_zeros() as usize;
                        let pos = word_idx * 64 + bit_idx;
                        if pos < self.follow.len() {
                            reachable.union_assign(self.follow[pos]);
                        }
                        word &= word - 1; // Clear lowest set bit
                    }
                }

                // Invert back to Shift-Or convention (0 = active)
                // Then apply byte mask
                state = reachable.complement().union(byte_mask);
            }

            // Check for match: if any accepting position is reached (bit is 0)
            if !state.union(self.accept).is_all_ones() {
                last_match = Some(start + i + 1);
            }

            // If all bits are 1, no possible match from this starting point
            if state.is_all_ones() {
                break;
            }
        }

        last_match
    }
}

/// Checks if an HIR is suitable for Wide Shift-Or (65-256 positions).
pub fn is_shift_or_wide_compatible(hir: &Hir) -> bool {
    if hir.props.has_backrefs
        || hir.props.has_lookaround
        || hir.props.has_anchors
        || hir.props.has_word_boundary
        || hir.props.has_non_greedy
    {
        return false;
    }

    // Try to build Wide Glushkov NFA to check position count
    compile_glushkov_wide(hir)
        .map(|nfa| {
            nfa.position_count > MAX_POSITIONS
                && nfa.position_count <= MAX_POSITIONS_WIDE
                && nfa.position_count > 0
        })
        .unwrap_or(false)
}
