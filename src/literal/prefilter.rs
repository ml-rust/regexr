//! Prefilter - SIMD-accelerated candidate position filtering.
//!
//! A prefilter quickly skips to positions in the input where a match might occur,
//! avoiding full regex execution on positions that can't possibly match.
//!
//! # Algorithm Selection
//!
//! - **Single literal prefix**: Use memchr for single-byte, or memmem for substring
//! - **Multiple literal prefixes**: Use aho-corasick packed searcher (SIMD-accelerated)
//! - **No common literals**: No prefilter (full regex on every position)

#[cfg(feature = "simd")]
use crate::simd::Teddy;

use super::Literals;

/// A prefilter for fast candidate position detection.
pub enum Prefilter {
    /// No prefilter available - scan all positions.
    None,
    /// Single-byte prefix search using memchr.
    SingleByte(u8),
    /// Inner byte search - finds a required byte that appears somewhere in the match.
    /// Unlike SingleByte (prefix), this requires backtracking to find the actual match start.
    /// NOTE: Currently disabled in from_literals() due to performance issues.
    #[allow(dead_code)]
    InnerByte {
        /// The byte to search for.
        byte: u8,
        /// How far back to search for the actual match start.
        max_lookback: usize,
    },
    /// Multi-byte literal prefix search using memmem.
    Literal(memchr::memmem::Finder<'static>),
    /// Pattern starts with a digit (0-9). Uses memchr to find digit positions.
    StartsWithDigit,
    /// Multi-pattern search using aho-corasick packed searcher (for prefix candidates).
    AhoCorasick {
        /// The aho-corasick automaton.
        ac: aho_corasick::AhoCorasick,
    },
    /// Complete literal alternation with full match bounds using aho-corasick.
    /// Used when the pattern is just `literal1|literal2|...` with no suffix.
    AhoCorasickFull {
        /// The aho-corasick automaton.
        ac: aho_corasick::AhoCorasick,
    },
    /// Multi-pattern search using Teddy algorithm (for prefix candidates).
    #[cfg(feature = "simd")]
    Teddy(Teddy),
    /// Complete literal alternation: Teddy + pattern lengths for full match bounds.
    #[cfg(feature = "simd")]
    TeddyFull {
        /// The Teddy multi-pattern matcher.
        teddy: Teddy,
        /// Length of each pattern for computing match end positions.
        lengths: Vec<usize>,
    },
}

impl std::fmt::Debug for Prefilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "Prefilter::None"),
            Self::SingleByte(b) => write!(f, "Prefilter::SingleByte({:#04x})", b),
            Self::InnerByte { byte, max_lookback } => {
                write!(
                    f,
                    "Prefilter::InnerByte({:#04x}, lookback={})",
                    byte, max_lookback
                )
            }
            Self::Literal(_) => write!(f, "Prefilter::Literal"),
            Self::StartsWithDigit => write!(f, "Prefilter::StartsWithDigit"),
            Self::AhoCorasick { ac } => {
                write!(f, "Prefilter::AhoCorasick({} patterns)", ac.patterns_len())
            }
            Self::AhoCorasickFull { ac } => write!(
                f,
                "Prefilter::AhoCorasickFull({} patterns)",
                ac.patterns_len()
            ),
            #[cfg(feature = "simd")]
            Self::Teddy(t) => write!(f, "Prefilter::Teddy({} patterns)", t.pattern_count()),
            #[cfg(feature = "simd")]
            Self::TeddyFull { teddy, .. } => write!(
                f,
                "Prefilter::TeddyFull({} patterns)",
                teddy.pattern_count()
            ),
        }
    }
}

impl Prefilter {
    /// Creates a prefilter from extracted literals.
    pub fn from_literals(literals: &Literals) -> Self {
        if literals.prefixes.is_empty() {
            // No literal prefix - check for specialized prefilters
            if literals.starts_with_digit {
                return Self::StartsWithDigit;
            }
            return Self::None;
        }

        // Single prefix case
        if literals.prefixes.len() == 1 {
            let prefix = &literals.prefixes[0];
            if prefix.len() == 1 {
                return Self::SingleByte(prefix[0]);
            }
            if !prefix.is_empty() {
                // Use memchr's memmem for single literal search
                let finder = memchr::memmem::Finder::new(prefix).into_owned();
                return Self::Literal(finder);
            }
            return Self::None;
        }

        // Multiple prefixes - use aho-corasick with leftmost-first semantics
        let ac = aho_corasick::AhoCorasick::builder()
            .match_kind(aho_corasick::MatchKind::LeftmostFirst)
            .build(&literals.prefixes)
            .expect("aho-corasick build should not fail");

        if literals.prefix_complete {
            return Self::AhoCorasickFull { ac };
        }
        Self::AhoCorasick { ac }
    }

    /// Returns the next candidate position in the haystack starting from `pos`.
    /// Returns None if no candidate is found.
    pub fn find_candidate(&self, haystack: &[u8], pos: usize) -> Option<usize> {
        if pos >= haystack.len() {
            return None;
        }

        match self {
            Self::None => {
                // No prefilter - every position is a candidate
                if pos < haystack.len() {
                    Some(pos)
                } else {
                    None
                }
            }
            Self::SingleByte(needle) => {
                let slice = &haystack[pos..];
                memchr::memchr(*needle, slice).map(|i| pos + i)
            }
            Self::InnerByte { byte, .. } => {
                // For InnerByte, we just find the next occurrence of the byte.
                // The executor will handle the lookback logic.
                let slice = &haystack[pos..];
                memchr::memchr(*byte, slice).map(|i| pos + i)
            }
            Self::Literal(finder) => {
                let slice = &haystack[pos..];
                finder.find(slice).map(|i| pos + i)
            }
            Self::StartsWithDigit => {
                // Find next digit (0-9) position using SIMD range search
                let slice = &haystack[pos..];
                #[cfg(feature = "simd")]
                {
                    // Use AVX2-accelerated range search for digits
                    crate::simd::memchr_range(b'0', b'9', slice).map(|i| pos + i)
                }
                #[cfg(not(feature = "simd"))]
                {
                    // Scalar fallback: search for any digit
                    slice
                        .iter()
                        .position(|&b| b >= b'0' && b <= b'9')
                        .map(|i| pos + i)
                }
            }
            Self::AhoCorasick { ac } => {
                let input = aho_corasick::Input::new(haystack).span(pos..haystack.len());
                ac.find(input).map(|m| m.start())
            }
            Self::AhoCorasickFull { ac } => {
                let input = aho_corasick::Input::new(haystack).span(pos..haystack.len());
                ac.find(input).map(|m| m.start())
            }
            #[cfg(feature = "simd")]
            Self::Teddy(teddy) => {
                let slice = &haystack[pos..];
                teddy.find(slice).map(|(_, i)| pos + i)
            }
            #[cfg(feature = "simd")]
            Self::TeddyFull { teddy, .. } => {
                let slice = &haystack[pos..];
                teddy.find(slice).map(|(_, i)| pos + i)
            }
        }
    }

    /// Finds the next complete match for AhoCorasickFull or TeddyFull patterns.
    /// Returns (start, end) byte offsets for the match.
    pub fn find_full_match(&self, haystack: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos >= haystack.len() {
            return None;
        }

        match self {
            Self::AhoCorasickFull { ac } => {
                let input = aho_corasick::Input::new(haystack).span(pos..haystack.len());
                ac.find(input).map(|m| (m.start(), m.end()))
            }
            #[cfg(feature = "simd")]
            Self::TeddyFull { teddy, lengths } => {
                let slice = &haystack[pos..];
                if let Some((pattern_id, match_pos)) = teddy.find(slice) {
                    let abs_pos = pos + match_pos;
                    let len = lengths[pattern_id];
                    return Some((abs_pos, abs_pos + len));
                }
                None
            }
            _ => None,
        }
    }

    /// Returns true if this prefilter can provide complete matches.
    pub fn is_full_match(&self) -> bool {
        matches!(self, Self::AhoCorasickFull { .. }) || {
            #[cfg(feature = "simd")]
            {
                matches!(self, Self::TeddyFull { .. })
            }
            #[cfg(not(feature = "simd"))]
            {
                false
            }
        }
    }

    /// Returns an iterator over all candidate positions.
    pub fn find_candidates<'a, 'h>(&'a self, haystack: &'h [u8]) -> CandidateIter<'a, 'h> {
        CandidateIter {
            prefilter: self,
            haystack,
            pos: 0,
        }
    }

    /// Returns an iterator over all complete matches for TeddyFull patterns.
    /// For non-TeddyFull prefilters, this iterator will be empty.
    pub fn find_full_matches<'a, 'h>(&'a self, haystack: &'h [u8]) -> FullMatchIter<'a, 'h> {
        FullMatchIter::new(self, haystack)
    }

    /// Returns true if this prefilter is trivial (matches every position).
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Returns true if this prefilter has good selectivity.
    /// An effective prefilter can significantly reduce the number of candidate positions.
    /// This is used for engine selection: DFA JIT benefits more from effective prefilters,
    /// while ShiftOr/JitShiftOr is often better without one.
    pub fn is_effective(&self) -> bool {
        match self {
            // Effective prefilters - good selectivity
            Self::SingleByte(_) => true,
            Self::Literal(_) => true,
            Self::AhoCorasick { .. } => true,
            Self::AhoCorasickFull { .. } => true,
            #[cfg(feature = "simd")]
            Self::Teddy(_) => true,
            #[cfg(feature = "simd")]
            Self::TeddyFull { .. } => true,
            // Not effective - weak or no selectivity
            Self::None => false,
            Self::StartsWithDigit => false, // Too many candidates in typical text
            Self::InnerByte { .. } => false, // Requires lookback, complex handling
        }
    }

    /// Returns true if this is an InnerByte prefilter.
    pub fn is_inner_byte(&self) -> bool {
        matches!(self, Self::InnerByte { .. })
    }

    /// Returns the max_lookback for InnerByte prefilter.
    pub fn inner_byte_lookback(&self) -> usize {
        match self {
            Self::InnerByte { max_lookback, .. } => *max_lookback,
            _ => 0,
        }
    }
}

/// Iterator over candidate positions.
pub struct CandidateIter<'a, 'h> {
    prefilter: &'a Prefilter,
    haystack: &'h [u8],
    pos: usize,
}

impl<'a, 'h> Iterator for CandidateIter<'a, 'h> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let candidate = self.prefilter.find_candidate(self.haystack, self.pos)?;
        self.pos = candidate + 1;
        Some(candidate)
    }
}

/// Iterator over complete matches for full-match prefilters.
pub struct FullMatchIter<'a, 'h> {
    inner: FullMatchIterInner<'a, 'h>,
}

enum FullMatchIterInner<'a, 'h> {
    /// Multi-literal using aho-corasick.
    AhoCorasickFull {
        ac_iter: aho_corasick::FindIter<'a, 'h>,
        last_end: usize,
    },
    /// Direct Teddy iterator for TeddyFull patterns.
    #[cfg(feature = "simd")]
    TeddyFull {
        teddy_iter: crate::simd::TeddyIter<'a, 'h>,
        lengths: &'a [usize],
        last_end: usize,
    },
    /// Empty iterator for non-full-match prefilters.
    Empty,
}

impl<'a, 'h> FullMatchIter<'a, 'h> {
    pub(crate) fn new(prefilter: &'a Prefilter, haystack: &'h [u8]) -> Self {
        let inner = match prefilter {
            Prefilter::AhoCorasickFull { ac } => FullMatchIterInner::AhoCorasickFull {
                ac_iter: ac.find_iter(haystack),
                last_end: 0,
            },
            #[cfg(feature = "simd")]
            Prefilter::TeddyFull { teddy, lengths } => FullMatchIterInner::TeddyFull {
                teddy_iter: teddy.find_iter(haystack),
                lengths,
                last_end: 0,
            },
            _ => FullMatchIterInner::Empty,
        };
        Self { inner }
    }
}

impl<'a, 'h> Iterator for FullMatchIter<'a, 'h> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            FullMatchIterInner::AhoCorasickFull { ac_iter, last_end } => {
                // Skip matches that overlap with the previous match
                loop {
                    let m = ac_iter.next()?;
                    if m.start() >= *last_end {
                        *last_end = m.end();
                        return Some((m.start(), m.end()));
                    }
                }
            }
            #[cfg(feature = "simd")]
            FullMatchIterInner::TeddyFull {
                teddy_iter,
                lengths,
                last_end,
            } => {
                // Skip matches that overlap with the previous match
                loop {
                    let (pattern_id, pos) = teddy_iter.next()?;
                    if pos >= *last_end {
                        let len = lengths[pattern_id];
                        let end = pos + len;
                        *last_end = end;
                        return Some((pos, end));
                    }
                }
            }
            FullMatchIterInner::Empty => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_byte_prefilter() {
        let pf = Prefilter::SingleByte(b'x');
        assert_eq!(pf.find_candidate(b"hello x world", 0), Some(6));
        assert_eq!(pf.find_candidate(b"hello x world", 7), None);
    }

    #[test]
    fn test_literal_prefilter() {
        let finder = memchr::memmem::Finder::new(b"world").into_owned();
        let pf = Prefilter::Literal(finder);
        assert_eq!(pf.find_candidate(b"hello world", 0), Some(6));
        assert_eq!(pf.find_candidate(b"hello world", 7), None);
    }

    #[test]
    fn test_none_prefilter() {
        let pf = Prefilter::None;
        assert_eq!(pf.find_candidate(b"hello", 0), Some(0));
        assert_eq!(pf.find_candidate(b"hello", 3), Some(3));
        assert_eq!(pf.find_candidate(b"hello", 5), None);
    }

    #[test]
    fn test_from_literals_single() {
        let literals = Literals {
            prefixes: vec![b"hello".to_vec()],
            suffixes: vec![],
            prefix_complete: true,
            starts_with_digit: false,
        };
        let pf = Prefilter::from_literals(&literals);
        assert!(matches!(pf, Prefilter::Literal(_)));
    }

    #[test]
    fn test_from_literals_single_byte() {
        let literals = Literals {
            prefixes: vec![b"h".to_vec()],
            suffixes: vec![],
            prefix_complete: true,
            starts_with_digit: false,
        };
        let pf = Prefilter::from_literals(&literals);
        assert!(matches!(pf, Prefilter::SingleByte(b'h')));
    }

    #[test]
    fn test_from_literals_empty() {
        let literals = Literals {
            prefixes: vec![],
            suffixes: vec![],
            prefix_complete: false,
            starts_with_digit: false,
        };
        let pf = Prefilter::from_literals(&literals);
        assert!(pf.is_none());
    }

    #[test]
    fn test_multi_literal_prefilter() {
        let literals = Literals {
            prefixes: vec![b"hello".to_vec(), b"world".to_vec()],
            suffixes: vec![],
            prefix_complete: true,
            starts_with_digit: false,
        };
        let pf = Prefilter::from_literals(&literals);
        // With prefix_complete=true, we get AhoCorasickFull for complete literal matching
        assert!(matches!(pf, Prefilter::AhoCorasickFull { .. }));

        // Should find "hello" at position 4
        assert_eq!(pf.find_candidate(b"say hello there", 0), Some(4));
        // Should find "world" at position 6
        assert_eq!(pf.find_candidate(b"hello world", 6), Some(6));

        // Test full match functionality
        assert_eq!(pf.find_full_match(b"say hello there", 0), Some((4, 9)));
        assert_eq!(pf.find_full_match(b"hello world", 0), Some((0, 5)));
    }

    #[test]
    fn test_multi_literal_prefilter_incomplete() {
        let literals = Literals {
            prefixes: vec![b"hello".to_vec(), b"world".to_vec()],
            suffixes: vec![],
            prefix_complete: false, // Not a complete literal match
            starts_with_digit: false,
        };
        let pf = Prefilter::from_literals(&literals);
        // With prefix_complete=false, we get regular AhoCorasick
        assert!(matches!(pf, Prefilter::AhoCorasick { .. }));

        // Should find "hello" at position 4
        assert_eq!(pf.find_candidate(b"say hello there", 0), Some(4));
    }

    #[test]
    fn test_candidate_iter() {
        let pf = Prefilter::SingleByte(b'o');
        let candidates: Vec<_> = pf.find_candidates(b"hello world").collect();
        assert_eq!(candidates, vec![4, 7]);
    }
}
