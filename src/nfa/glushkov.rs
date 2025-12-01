//! Glushkov Construction (Position Automaton) for Shift-Or.
//!
//! Unlike Thompson's construction, Glushkov produces an ε-free NFA where:
//! - Each state corresponds to exactly one character position in the pattern
//! - 1 state transition = 1 byte consumed
//!
//! This makes it ideal for the Shift-Or (Bitap) algorithm where:
//! `state = ((state << 1) | 1) & mask[byte]`

use crate::hir::{Hir, HirExpr};

/// Maximum positions supported by standard Shift-Or (limited by u64 bit width).
pub const MAX_POSITIONS: usize = 64;

/// Maximum positions supported by Wide Shift-Or (using [u64; 4]).
pub const MAX_POSITIONS_WIDE: usize = 256;

/// A 256-bit set for wide state vectors.
///
/// Used by ShiftOrWide to support patterns with 65-256 positions.
/// Operations are implemented to work efficiently with the Shift-Or algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BitSet256 {
    /// 4 x u64 for 256 bits. parts[0] holds bits 0-63, parts[1] holds bits 64-127, etc.
    pub parts: [u64; 4],
}

impl BitSet256 {
    /// Creates a new empty bit set (all zeros).
    #[inline]
    pub const fn empty() -> Self {
        Self { parts: [0; 4] }
    }

    /// Creates a new bit set with all bits set to 1.
    #[inline]
    pub const fn all_ones() -> Self {
        Self { parts: [!0u64; 4] }
    }

    /// Creates a bit set with a single bit set.
    #[inline]
    pub fn singleton(pos: usize) -> Self {
        let mut set = Self::empty();
        set.set(pos);
        set
    }

    /// Sets the bit at the given position.
    #[inline]
    pub fn set(&mut self, pos: usize) {
        let word = pos / 64;
        let bit = pos % 64;
        if word < 4 {
            self.parts[word] |= 1u64 << bit;
        }
    }

    /// Clears the bit at the given position.
    #[inline]
    pub fn clear(&mut self, pos: usize) {
        let word = pos / 64;
        let bit = pos % 64;
        if word < 4 {
            self.parts[word] &= !(1u64 << bit);
        }
    }

    /// Returns true if the bit at the given position is set.
    #[inline]
    pub fn get(&self, pos: usize) -> bool {
        let word = pos / 64;
        let bit = pos % 64;
        if word < 4 {
            (self.parts[word] >> bit) & 1 != 0
        } else {
            false
        }
    }

    /// Returns true if all bits are zero.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.parts[0] == 0 && self.parts[1] == 0 && self.parts[2] == 0 && self.parts[3] == 0
    }

    /// Returns true if all bits are one.
    #[inline]
    pub fn is_all_ones(&self) -> bool {
        self.parts[0] == !0u64 && self.parts[1] == !0u64 && self.parts[2] == !0u64 && self.parts[3] == !0u64
    }

    /// Computes the bitwise OR of two bit sets.
    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self {
            parts: [
                self.parts[0] | other.parts[0],
                self.parts[1] | other.parts[1],
                self.parts[2] | other.parts[2],
                self.parts[3] | other.parts[3],
            ],
        }
    }

    /// Computes the bitwise AND of two bit sets.
    #[inline]
    pub fn intersection(self, other: Self) -> Self {
        Self {
            parts: [
                self.parts[0] & other.parts[0],
                self.parts[1] & other.parts[1],
                self.parts[2] & other.parts[2],
                self.parts[3] & other.parts[3],
            ],
        }
    }

    /// Computes the bitwise NOT of this bit set.
    #[inline]
    pub fn complement(self) -> Self {
        Self {
            parts: [
                !self.parts[0],
                !self.parts[1],
                !self.parts[2],
                !self.parts[3],
            ],
        }
    }

    /// Computes self OR other, storing result in self.
    #[inline]
    pub fn union_assign(&mut self, other: Self) {
        self.parts[0] |= other.parts[0];
        self.parts[1] |= other.parts[1];
        self.parts[2] |= other.parts[2];
        self.parts[3] |= other.parts[3];
    }

    /// Iterator over all set bit positions.
    /// Yields positions in ascending order.
    #[inline]
    pub fn iter_ones(&self) -> BitSet256Iter {
        BitSet256Iter {
            set: *self,
            word_idx: 0,
        }
    }
}

/// Iterator over set bits in a BitSet256.
pub struct BitSet256Iter {
    set: BitSet256,
    word_idx: usize,
}

impl Iterator for BitSet256Iter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Find next word with set bits
        while self.word_idx < 4 {
            let word = self.set.parts[self.word_idx];
            if word != 0 {
                let bit_idx = word.trailing_zeros() as usize;
                let pos = self.word_idx * 64 + bit_idx;
                // Clear this bit
                self.set.parts[self.word_idx] &= word - 1;
                return Some(pos);
            }
            self.word_idx += 1;
        }
        None
    }
}

/// A Glushkov NFA (Position Automaton).
///
/// Unlike Thompson NFA, this has no ε-transitions.
/// Each position corresponds to one character in the pattern.
#[derive(Debug, Clone)]
pub struct GlushkovNfa {
    /// What bytes each position accepts.
    /// `positions[i]` is a 256-bit set indicating which bytes position i accepts.
    pub positions: Vec<ByteSet>,

    /// Follow sets: `follow[i]` contains positions that can follow position i.
    pub follow: Vec<u64>,

    /// Positions that can start a match (First set).
    pub first: u64,

    /// Positions that can end a match (Last set).
    pub last: u64,

    /// Whether the pattern can match the empty string.
    pub nullable: bool,

    /// Number of positions (excluding the implicit start state).
    pub position_count: usize,
}

/// A set of bytes (256 bits).
#[derive(Debug, Clone, Default)]
pub struct ByteSet {
    /// 4 x u64 for 256 bits.
    bits: [u64; 4],
}

impl ByteSet {
    /// Creates an empty byte set.
    pub fn new() -> Self {
        Self { bits: [0; 4] }
    }

    /// Creates a byte set containing a single byte.
    pub fn singleton(byte: u8) -> Self {
        let mut set = Self::new();
        set.insert(byte);
        set
    }

    /// Creates a byte set from a range of bytes (inclusive).
    pub fn from_range(start: u8, end: u8) -> Self {
        let mut set = Self::new();
        for b in start..=end {
            set.insert(b);
        }
        set
    }

    /// Creates a byte set containing all bytes.
    pub fn all() -> Self {
        Self {
            bits: [!0u64; 4],
        }
    }

    /// Inserts a byte into the set.
    pub fn insert(&mut self, byte: u8) {
        let idx = byte as usize / 64;
        let bit = byte as usize % 64;
        self.bits[idx] |= 1u64 << bit;
    }

    /// Checks if the set contains a byte.
    pub fn contains(&self, byte: u8) -> bool {
        let idx = byte as usize / 64;
        let bit = byte as usize % 64;
        (self.bits[idx] >> bit) & 1 != 0
    }

    /// Returns true if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&b| b == 0)
    }

    /// Computes the union of two byte sets.
    pub fn union(&self, other: &ByteSet) -> ByteSet {
        ByteSet {
            bits: [
                self.bits[0] | other.bits[0],
                self.bits[1] | other.bits[1],
                self.bits[2] | other.bits[2],
                self.bits[3] | other.bits[3],
            ],
        }
    }

    /// Computes the complement of the byte set.
    pub fn complement(&self) -> ByteSet {
        ByteSet {
            bits: [
                !self.bits[0],
                !self.bits[1],
                !self.bits[2],
                !self.bits[3],
            ],
        }
    }
}

/// Result of analyzing an HIR expression for Glushkov construction.
#[derive(Debug, Clone)]
struct ExprInfo {
    /// First set: positions that can start matching this expression.
    first: u64,
    /// Last set: positions that can end matching this expression.
    last: u64,
    /// Whether this expression can match the empty string.
    nullable: bool,
}

impl ExprInfo {
    fn empty() -> Self {
        Self {
            first: 0,
            last: 0,
            nullable: true,
        }
    }
}

/// Builds a Glushkov NFA from HIR.
pub struct GlushkovBuilder {
    /// Byte sets for each position.
    positions: Vec<ByteSet>,
    /// Follow sets for each position.
    follow: Vec<u64>,
}

impl GlushkovBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            follow: Vec::new(),
        }
    }

    /// Builds a Glushkov NFA from HIR.
    ///
    /// This uses a single-pass algorithm that allocates positions and computes
    /// follow sets simultaneously, ensuring correct position tracking.
    pub fn build(&mut self, hir: &Hir) -> Option<GlushkovNfa> {
        // Check for features that Glushkov/Shift-Or can't handle
        if hir.props.has_backrefs || hir.props.has_lookaround {
            return None;
        }

        // Clear state
        self.positions.clear();
        self.follow.clear();

        // Single-pass: allocate positions AND compute follow sets
        let info = self.build_expr(&hir.expr)?;

        // Check if we have too many positions
        if self.positions.len() > MAX_POSITIONS {
            return None;
        }

        Some(GlushkovNfa {
            positions: self.positions.clone(),
            follow: self.follow.clone(),
            first: info.first,
            last: info.last,
            nullable: info.nullable,
            position_count: self.positions.len(),
        })
    }

    /// Single-pass construction: allocate positions and compute follow sets.
    fn build_expr(&mut self, expr: &HirExpr) -> Option<ExprInfo> {
        match expr {
            HirExpr::Empty => Some(ExprInfo::empty()),

            HirExpr::Literal(bytes) => {
                if bytes.is_empty() {
                    return Some(ExprInfo::empty());
                }

                let first_pos = self.positions.len();

                // Each byte gets its own position
                for &b in bytes {
                    if self.positions.len() >= MAX_POSITIONS {
                        return None;
                    }
                    self.positions.push(ByteSet::singleton(b));
                    self.follow.push(0); // Initialize follow set
                }

                let last_pos = self.positions.len() - 1;

                // Set follow: each position follows the previous within the literal
                for i in first_pos..last_pos {
                    self.follow[i] |= 1u64 << (i + 1);
                }

                Some(ExprInfo {
                    first: 1u64 << first_pos,
                    last: 1u64 << last_pos,
                    nullable: false,
                })
            }

            HirExpr::Class(class) => {
                if self.positions.len() >= MAX_POSITIONS {
                    return None;
                }

                let pos = self.positions.len();
                let mut byte_set = ByteSet::new();

                for &(start, end) in &class.ranges {
                    for b in start..=end {
                        byte_set.insert(b);
                    }
                }

                if class.negated {
                    byte_set = byte_set.complement();
                }

                self.positions.push(byte_set);
                self.follow.push(0); // Initialize follow set

                Some(ExprInfo {
                    first: 1u64 << pos,
                    last: 1u64 << pos,
                    nullable: false,
                })
            }

            HirExpr::Concat(exprs) => {
                if exprs.is_empty() {
                    return Some(ExprInfo::empty());
                }

                let mut result = self.build_expr(&exprs[0])?;

                for expr in &exprs[1..] {
                    let next = self.build_expr(expr)?;

                    // Follow(last(A)) includes First(B)
                    for pos in 0..MAX_POSITIONS {
                        if (result.last >> pos) & 1 != 0 && pos < self.follow.len() {
                            self.follow[pos] |= next.first;
                        }
                    }

                    // Update result
                    let new_first = if result.nullable {
                        result.first | next.first
                    } else {
                        result.first
                    };

                    let new_last = if next.nullable {
                        next.last | result.last
                    } else {
                        next.last
                    };

                    result = ExprInfo {
                        first: new_first,
                        last: new_last,
                        nullable: result.nullable && next.nullable,
                    };
                }

                Some(result)
            }

            HirExpr::Alt(exprs) => {
                if exprs.is_empty() {
                    return Some(ExprInfo::empty());
                }

                let mut result = self.build_expr(&exprs[0])?;

                for expr in &exprs[1..] {
                    let alt = self.build_expr(expr)?;
                    result = ExprInfo {
                        first: result.first | alt.first,
                        last: result.last | alt.last,
                        nullable: result.nullable || alt.nullable,
                    };
                }

                Some(result)
            }

            HirExpr::Repeat(rep) => {
                let inner = self.build_expr(&rep.expr)?;

                // For repetition with unbounded max (*, +, {n,}), Last positions
                // can go back to First positions. This enables loop behavior.
                // But for bounded repetition like ? ({0,1}), we don't add a loop.
                let allows_repetition = rep.max.is_none() || rep.max > Some(1);

                if allows_repetition {
                    for pos in 0..MAX_POSITIONS {
                        if (inner.last >> pos) & 1 != 0 && pos < self.follow.len() {
                            self.follow[pos] |= inner.first;
                        }
                    }
                }

                match (rep.min, rep.max) {
                    // E? or E* - zero or more (nullable)
                    (0, _) => Some(ExprInfo {
                        first: inner.first,
                        last: inner.last,
                        nullable: true,
                    }),
                    // E+ or E{n,} with n >= 1
                    _ => Some(ExprInfo {
                        first: inner.first,
                        last: inner.last,
                        nullable: inner.nullable,
                    }),
                }
            }

            HirExpr::Capture(cap) => self.build_expr(&cap.expr),

            HirExpr::Anchor(_) => Some(ExprInfo::empty()),

            HirExpr::Lookaround(_) | HirExpr::Backref(_) => None,

            // Unicode codepoint classes are not supported in Glushkov/Shift-Or
            // They use a special instruction that consumes variable-length UTF-8
            HirExpr::UnicodeCpClass(_) => None,
        }
    }

}

impl Default for GlushkovBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GlushkovNfa {
    /// Checks if this NFA is compatible with Shift-Or.
    pub fn is_shift_or_compatible(&self) -> bool {
        self.position_count <= MAX_POSITIONS
    }

    /// Builds the Shift-Or mask table from this Glushkov NFA.
    ///
    /// Returns masks where `mask[byte]` has bit `i` as 0 if position `i` can accept `byte`,
    /// 1 otherwise (Shift-Or uses inverted logic).
    pub fn build_shift_or_masks(&self) -> [u64; 256] {
        let mut masks = [!0u64; 256];

        for (pos_idx, byte_set) in self.positions.iter().enumerate() {
            for byte in 0..=255u8 {
                if byte_set.contains(byte) {
                    // Clear bit to indicate this position can accept this byte
                    masks[byte as usize] &= !(1u64 << pos_idx);
                }
            }
        }

        masks
    }

    /// Builds the initial state for Shift-Or.
    ///
    /// In Shift-Or, we track "not yet reached" with 1 bits.
    /// Initially, positions in First set can be reached.
    pub fn build_initial_state(&self) -> u64 {
        // All bits set to 1 (not reached), except we'll handle First set during matching
        !0u64
    }

    /// Builds the accept mask for Shift-Or.
    ///
    /// Positions in Last set are accepting positions.
    pub fn build_accept_mask(&self) -> u64 {
        // Inverted: 0 bits indicate accepting positions
        !self.last
    }
}

/// Compiles an HIR to a Glushkov NFA.
pub fn compile_glushkov(hir: &Hir) -> Option<GlushkovNfa> {
    let mut builder = GlushkovBuilder::new();
    builder.build(hir)
}

// ============================================================================
// Wide Glushkov NFA (supports up to 256 positions)
// ============================================================================

/// A Wide Glushkov NFA supporting up to 256 positions.
///
/// Uses BitSet256 instead of u64 for state vectors, allowing patterns
/// with 65-256 character positions to use the efficient Shift-Or algorithm
/// instead of falling back to the slower PikeVM.
#[derive(Debug, Clone)]
pub struct GlushkovWideNfa {
    /// What bytes each position accepts.
    pub positions: Vec<ByteSet>,

    /// Follow sets: `follow[i]` contains positions that can follow position i.
    pub follow: Vec<BitSet256>,

    /// Positions that can start a match (First set).
    pub first: BitSet256,

    /// Positions that can end a match (Last set).
    pub last: BitSet256,

    /// Whether the pattern can match the empty string.
    pub nullable: bool,

    /// Number of positions.
    pub position_count: usize,
}

impl GlushkovWideNfa {
    /// Checks if this NFA is compatible with Wide Shift-Or.
    pub fn is_shift_or_wide_compatible(&self) -> bool {
        self.position_count <= MAX_POSITIONS_WIDE && self.position_count > 0
    }

    /// Builds the Wide Shift-Or mask table from this Glushkov NFA.
    ///
    /// Returns masks where `mask[byte]` has bit `i` as 0 if position `i` can accept `byte`,
    /// 1 otherwise (Shift-Or uses inverted logic).
    pub fn build_shift_or_masks(&self) -> [BitSet256; 256] {
        let mut masks = [BitSet256::all_ones(); 256];

        for (pos_idx, byte_set) in self.positions.iter().enumerate() {
            for byte in 0..=255u8 {
                if byte_set.contains(byte) {
                    // Clear bit to indicate this position can accept this byte
                    masks[byte as usize].clear(pos_idx);
                }
            }
        }

        masks
    }

    /// Builds the accept mask for Wide Shift-Or.
    ///
    /// Positions in Last set are accepting positions.
    pub fn build_accept_mask(&self) -> BitSet256 {
        // Inverted: 0 bits indicate accepting positions
        self.last.complement()
    }
}

/// Result of analyzing an HIR expression for Wide Glushkov construction.
#[derive(Debug, Clone)]
struct WideExprInfo {
    first: BitSet256,
    last: BitSet256,
    nullable: bool,
}

impl WideExprInfo {
    fn empty() -> Self {
        Self {
            first: BitSet256::empty(),
            last: BitSet256::empty(),
            nullable: true,
        }
    }
}

/// Builds a Wide Glushkov NFA from HIR.
pub struct GlushkovWideBuilder {
    positions: Vec<ByteSet>,
    follow: Vec<BitSet256>,
}

impl GlushkovWideBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            follow: Vec::new(),
        }
    }

    /// Builds a Wide Glushkov NFA from HIR.
    pub fn build(&mut self, hir: &Hir) -> Option<GlushkovWideNfa> {
        // Check for features that Glushkov/Shift-Or can't handle
        if hir.props.has_backrefs || hir.props.has_lookaround {
            return None;
        }

        // Clear state
        self.positions.clear();
        self.follow.clear();

        // Single-pass: allocate positions AND compute follow sets
        let info = self.build_expr(&hir.expr)?;

        // Check if we have too many positions
        if self.positions.len() > MAX_POSITIONS_WIDE {
            return None;
        }

        Some(GlushkovWideNfa {
            positions: self.positions.clone(),
            follow: self.follow.clone(),
            first: info.first,
            last: info.last,
            nullable: info.nullable,
            position_count: self.positions.len(),
        })
    }

    /// Single-pass construction for wide NFA.
    fn build_expr(&mut self, expr: &HirExpr) -> Option<WideExprInfo> {
        match expr {
            HirExpr::Empty => Some(WideExprInfo::empty()),

            HirExpr::Literal(bytes) => {
                if bytes.is_empty() {
                    return Some(WideExprInfo::empty());
                }

                let first_pos = self.positions.len();

                // Each byte gets its own position
                for &b in bytes {
                    if self.positions.len() >= MAX_POSITIONS_WIDE {
                        return None;
                    }
                    self.positions.push(ByteSet::singleton(b));
                    self.follow.push(BitSet256::empty());
                }

                let last_pos = self.positions.len() - 1;

                // Set follow: each position follows the previous within the literal
                for i in first_pos..last_pos {
                    self.follow[i].set(i + 1);
                }

                Some(WideExprInfo {
                    first: BitSet256::singleton(first_pos),
                    last: BitSet256::singleton(last_pos),
                    nullable: false,
                })
            }

            HirExpr::Class(class) => {
                if self.positions.len() >= MAX_POSITIONS_WIDE {
                    return None;
                }

                let pos = self.positions.len();
                let mut byte_set = ByteSet::new();

                for &(start, end) in &class.ranges {
                    for b in start..=end {
                        byte_set.insert(b);
                    }
                }

                if class.negated {
                    byte_set = byte_set.complement();
                }

                self.positions.push(byte_set);
                self.follow.push(BitSet256::empty());

                Some(WideExprInfo {
                    first: BitSet256::singleton(pos),
                    last: BitSet256::singleton(pos),
                    nullable: false,
                })
            }

            HirExpr::Concat(exprs) => {
                if exprs.is_empty() {
                    return Some(WideExprInfo::empty());
                }

                let mut result = self.build_expr(&exprs[0])?;

                for expr in &exprs[1..] {
                    let next = self.build_expr(expr)?;

                    // Follow(last(A)) includes First(B)
                    for pos in result.last.iter_ones() {
                        if pos < self.follow.len() {
                            self.follow[pos].union_assign(next.first);
                        }
                    }

                    // Update result
                    let new_first = if result.nullable {
                        result.first.union(next.first)
                    } else {
                        result.first
                    };

                    let new_last = if next.nullable {
                        next.last.union(result.last)
                    } else {
                        next.last
                    };

                    result = WideExprInfo {
                        first: new_first,
                        last: new_last,
                        nullable: result.nullable && next.nullable,
                    };
                }

                Some(result)
            }

            HirExpr::Alt(exprs) => {
                if exprs.is_empty() {
                    return Some(WideExprInfo::empty());
                }

                let mut result = self.build_expr(&exprs[0])?;

                for expr in &exprs[1..] {
                    let alt = self.build_expr(expr)?;
                    result = WideExprInfo {
                        first: result.first.union(alt.first),
                        last: result.last.union(alt.last),
                        nullable: result.nullable || alt.nullable,
                    };
                }

                Some(result)
            }

            HirExpr::Repeat(rep) => {
                let inner = self.build_expr(&rep.expr)?;

                // For repetition with unbounded max (*, +, {n,}), Last positions
                // can go back to First positions.
                let allows_repetition = rep.max.is_none() || rep.max > Some(1);

                if allows_repetition {
                    for pos in inner.last.iter_ones() {
                        if pos < self.follow.len() {
                            self.follow[pos].union_assign(inner.first);
                        }
                    }
                }

                match (rep.min, rep.max) {
                    (0, _) => Some(WideExprInfo {
                        first: inner.first,
                        last: inner.last,
                        nullable: true,
                    }),
                    _ => Some(WideExprInfo {
                        first: inner.first,
                        last: inner.last,
                        nullable: inner.nullable,
                    }),
                }
            }

            HirExpr::Capture(cap) => self.build_expr(&cap.expr),

            HirExpr::Anchor(_) => Some(WideExprInfo::empty()),

            HirExpr::Lookaround(_) | HirExpr::Backref(_) => None,

            HirExpr::UnicodeCpClass(_) => None,
        }
    }
}

impl Default for GlushkovWideBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiles an HIR to a Wide Glushkov NFA (supports up to 256 positions).
pub fn compile_glushkov_wide(hir: &Hir) -> Option<GlushkovWideNfa> {
    let mut builder = GlushkovWideBuilder::new();
    builder.build(hir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn make_glushkov(pattern: &str) -> Option<GlushkovNfa> {
        let ast = parse(pattern).ok()?;
        let hir = translate(&ast).ok()?;
        compile_glushkov(&hir)
    }

    #[test]
    fn test_simple_literal() {
        let nfa = make_glushkov("abc").unwrap();
        assert_eq!(nfa.position_count, 3);
        assert!(!nfa.nullable);

        // Position 0 accepts 'a', position 1 accepts 'b', position 2 accepts 'c'
        assert!(nfa.positions[0].contains(b'a'));
        assert!(nfa.positions[1].contains(b'b'));
        assert!(nfa.positions[2].contains(b'c'));
    }

    #[test]
    fn test_alternation() {
        let nfa = make_glushkov("a|b").unwrap();
        assert_eq!(nfa.position_count, 2);
        assert!(!nfa.nullable);

        // First set should include both positions
        assert_eq!(nfa.first, 0b11);
        // Last set should include both positions
        assert_eq!(nfa.last, 0b11);
    }

    #[test]
    fn test_optional() {
        let nfa = make_glushkov("a?").unwrap();
        assert_eq!(nfa.position_count, 1);
        assert!(nfa.nullable); // Can match empty string
    }

    #[test]
    fn test_star() {
        let nfa = make_glushkov("a*").unwrap();
        assert_eq!(nfa.position_count, 1);
        assert!(nfa.nullable); // Can match empty string
    }

    #[test]
    fn test_plus() {
        let nfa = make_glushkov("a+").unwrap();
        assert_eq!(nfa.position_count, 1);
        assert!(!nfa.nullable); // Must match at least one 'a'
    }

    #[test]
    fn test_class() {
        let nfa = make_glushkov("[a-z]").unwrap();
        assert_eq!(nfa.position_count, 1);

        // Should accept all lowercase letters
        for c in b'a'..=b'z' {
            assert!(nfa.positions[0].contains(c));
        }

        // Should not accept uppercase
        assert!(!nfa.positions[0].contains(b'A'));
    }

    #[test]
    fn test_too_many_positions() {
        // Pattern with more than 64 character positions
        let pattern = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnop";
        assert!(pattern.len() > MAX_POSITIONS);

        let result = make_glushkov(pattern);
        assert!(result.is_none());
    }

    #[test]
    fn test_shift_or_masks() {
        let nfa = make_glushkov("ab").unwrap();
        let masks = nfa.build_shift_or_masks();

        // For 'a' at position 0: bit 0 should be 0 (cleared)
        assert_eq!(masks[b'a' as usize] & 1, 0);

        // For 'b' at position 1: bit 1 should be 0 (cleared)
        assert_eq!(masks[b'b' as usize] & 2, 0);

        // For 'c' (not in pattern): all bits should be 1
        assert_eq!(masks[b'c' as usize], !0u64);
    }

    #[test]
    fn test_dot_star_glushkov() {
        // a.*b pattern: position 0 = 'a', position 1 = '.', position 2 = 'b'
        let nfa = make_glushkov("a.*b").unwrap();

        // Should have 3 positions: 'a', '.', 'b'
        assert_eq!(nfa.position_count, 3);

        // First set should only include position 0 ('a')
        assert_eq!(nfa.first, 0b001);

        // Last set should only include position 2 ('b')
        assert_eq!(nfa.last, 0b100);

        // Pattern is not nullable (can't match empty string)
        assert!(!nfa.nullable);

        // Position 0 accepts 'a'
        assert!(nfa.positions[0].contains(b'a'));
        assert!(!nfa.positions[0].contains(b'b'));

        // Position 1 accepts any byte (.)
        assert!(nfa.positions[1].contains(b'a'));
        assert!(nfa.positions[1].contains(b'b'));
        assert!(nfa.positions[1].contains(b'x'));

        // Position 2 accepts 'b'
        assert!(nfa.positions[2].contains(b'b'));
        assert!(!nfa.positions[2].contains(b'a'));

        // Follow sets:
        // Follow(0) = {1, 2} (after 'a', can go to '.' or 'b' since .* is nullable)
        assert_eq!(nfa.follow[0] & 0b110, 0b110, "Follow(0) should include positions 1 and 2");

        // Follow(1) = {1, 2} (after '.', can stay at '.' or go to 'b')
        assert_eq!(nfa.follow[1] & 0b110, 0b110, "Follow(1) should include positions 1 and 2");

        // Follow(2) = {} (after 'b', nothing follows - it's the end)
        assert_eq!(nfa.follow[2], 0, "Follow(2) should be empty");
    }
}
