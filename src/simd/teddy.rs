//! Teddy multi-literal matcher.
//!
//! SIMD-accelerated algorithm for matching multiple literal patterns.
//! Based on the Teddy algorithm from Hyperscan/ripgrep.
//!
//! # Algorithm Overview
//!
//! Teddy uses SIMD nibble-based hashing to quickly filter candidate positions.
//! For each position, it checks if the low and high nibbles of the first byte
//! match any of the patterns. Only positions that pass this filter are verified.
//!
//! # Limitations
//!
//! - Works best with 1-8 patterns
//! - Pattern length should be 1-8 bytes
//! - Falls back to scalar search when SIMD is not available

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Maximum number of patterns Teddy can handle efficiently.
pub const MAX_PATTERNS: usize = 8;

/// Maximum pattern length for Teddy.
pub const MAX_PATTERN_LEN: usize = 8;

/// A compiled Teddy matcher.
pub struct Teddy {
    /// The patterns to match.
    patterns: Vec<Vec<u8>>,
    /// Nibble lookup table for low nibbles of first byte.
    /// Each bit position corresponds to a pattern ID.
    #[cfg(target_arch = "x86_64")]
    lo_nibble_table: [u8; 16],
    /// Nibble lookup table for high nibbles of first byte.
    #[cfg(target_arch = "x86_64")]
    hi_nibble_table: [u8; 16],
    /// Pre-computed SIMD lookup table for low nibbles (cached to avoid rebuilding).
    #[cfg(target_arch = "x86_64")]
    lo_simd_table: std::sync::OnceLock<[u8; 32]>,
    /// Pre-computed SIMD lookup table for high nibbles (cached to avoid rebuilding).
    #[cfg(target_arch = "x86_64")]
    hi_simd_table: std::sync::OnceLock<[u8; 32]>,
    /// Cached AVX2 detection result.
    #[cfg(target_arch = "x86_64")]
    use_avx2: bool,
}

impl Teddy {
    /// Creates a new Teddy matcher from patterns.
    /// Returns None if there are too many patterns or they're too long.
    pub fn new(patterns: Vec<Vec<u8>>) -> Option<Self> {
        // Teddy works best with 1-8 patterns, each 1-8 bytes
        if patterns.is_empty() || patterns.len() > MAX_PATTERNS {
            return None;
        }

        if patterns
            .iter()
            .any(|p| p.is_empty() || p.len() > MAX_PATTERN_LEN)
        {
            return None;
        }

        #[cfg(target_arch = "x86_64")]
        {
            // Build nibble tables
            let mut lo_nibble_table = [0u8; 16];
            let mut hi_nibble_table = [0u8; 16];

            for (i, pattern) in patterns.iter().enumerate() {
                let first_byte = pattern[0];
                let lo_nibble = (first_byte & 0x0F) as usize;
                let hi_nibble = (first_byte >> 4) as usize;

                // Set bit i in the corresponding nibble entry
                lo_nibble_table[lo_nibble] |= 1 << i;
                hi_nibble_table[hi_nibble] |= 1 << i;
            }

            Some(Self {
                patterns,
                lo_nibble_table,
                hi_nibble_table,
                lo_simd_table: std::sync::OnceLock::new(),
                hi_simd_table: std::sync::OnceLock::new(),
                use_avx2: is_x86_feature_detected!("avx2"),
            })
        }

        #[cfg(not(target_arch = "x86_64"))]
        Some(Self { patterns })
    }

    /// Returns the patterns this matcher is searching for.
    pub fn patterns(&self) -> &[Vec<u8>] {
        &self.patterns
    }

    /// Returns the number of patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Finds the first match in the haystack.
    /// Returns (pattern_index, position).
    pub fn find(&self, haystack: &[u8]) -> Option<(usize, usize)> {
        self.find_from(haystack, 0)
    }

    /// Finds the first match in the haystack starting from `pos`.
    /// Returns (pattern_index, absolute_position).
    #[inline]
    pub fn find_from(&self, haystack: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos >= haystack.len() {
            return None;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if self.use_avx2 {
                // SAFETY: use_avx2 was set at construction time after checking for AVX2 support
                return unsafe { self.find_avx2_from(haystack, pos) };
            }
        }

        // Scalar fallback
        self.find_scalar_from(haystack, pos)
    }

    /// Finds all matches in the haystack.
    /// Returns an iterator of (pattern_index, position) pairs.
    #[inline]
    pub fn find_iter<'a, 'h>(&'a self, haystack: &'h [u8]) -> TeddyIter<'a, 'h> {
        TeddyIter {
            teddy: self,
            haystack,
            pos: 0,
        }
    }

    /// AVX2-accelerated Teddy search.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[allow(dead_code)]
    unsafe fn find_avx2(&self, haystack: &[u8]) -> Option<(usize, usize)> {
        self.find_avx2_from(haystack, 0)
    }

    /// AVX2-accelerated Teddy search starting from a position.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn find_avx2_from(&self, haystack: &[u8], start_pos: usize) -> Option<(usize, usize)> {
        let len = haystack.len();
        if start_pos >= len {
            return None;
        }

        // Get or build cached SIMD lookup tables
        let lo_bytes = self
            .lo_simd_table
            .get_or_init(|| self.build_simd_table_cached(&self.lo_nibble_table));
        let hi_bytes = self
            .hi_simd_table
            .get_or_init(|| self.build_simd_table_cached(&self.hi_nibble_table));

        // Load cached tables into SIMD registers
        let lo_table = _mm256_loadu_si256(lo_bytes.as_ptr() as *const __m256i);
        let hi_table = _mm256_loadu_si256(hi_bytes.as_ptr() as *const __m256i);
        let lo_mask = _mm256_set1_epi8(0x0F);

        let ptr = haystack.as_ptr();
        let mut offset = start_pos;

        // Process 32 bytes at a time
        while offset + 32 <= len {
            let data = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);

            // Extract low and high nibbles
            let lo_nibbles = _mm256_and_si256(data, lo_mask);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(data, 4), lo_mask);

            // Look up pattern masks for each nibble
            let lo_matches = _mm256_shuffle_epi8(lo_table, lo_nibbles);
            let hi_matches = _mm256_shuffle_epi8(hi_table, hi_nibbles);

            // Both nibbles must match for a candidate
            let candidates = _mm256_and_si256(lo_matches, hi_matches);

            // Check if any position has candidates
            let mask =
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(candidates, _mm256_setzero_si256())) as u32;

            // Invert: we want positions that are NOT zero
            let candidate_mask = !mask;

            if candidate_mask != 0 {
                // Verify candidates
                let mut remaining = candidate_mask;

                while remaining != 0 {
                    let bit_pos = remaining.trailing_zeros() as usize;
                    remaining &= remaining - 1; // Clear lowest set bit

                    let pos = offset + bit_pos;

                    // Get the pattern mask for this position
                    let pattern_bits = *haystack.get_unchecked(pos);
                    let pattern_mask = self.lo_nibble_table[(pattern_bits & 0x0F) as usize]
                        & self.hi_nibble_table[(pattern_bits >> 4) as usize];

                    // Verify each matching pattern
                    for (pat_idx, pattern) in self.patterns.iter().enumerate() {
                        if (pattern_mask & (1 << pat_idx)) != 0 {
                            // First byte matches, verify the rest
                            if pos + pattern.len() <= len {
                                if haystack[pos..pos + pattern.len()] == *pattern {
                                    return Some((pat_idx, pos));
                                }
                            }
                        }
                    }
                }
            }

            offset += 32;
        }

        // Handle remaining bytes with scalar
        self.find_scalar_from(&haystack[offset..], offset)
    }

    /// Builds a 256-bit SIMD lookup table from a 16-byte table and returns it as a byte array.
    /// This is cached in the struct to avoid rebuilding on each find() call.
    #[cfg(target_arch = "x86_64")]
    fn build_simd_table_cached(&self, table: &[u8; 16]) -> [u8; 32] {
        // For vpshufb to work correctly in AVX2, we need the same 16-byte
        // table in both lanes of the 256-bit register
        let mut result = [0u8; 32];
        result[0..16].copy_from_slice(table);
        result[16..32].copy_from_slice(table);
        result
    }

    /// Scalar fallback for Teddy.
    #[allow(dead_code)]
    fn find_scalar(&self, haystack: &[u8]) -> Option<(usize, usize)> {
        self.find_scalar_from(haystack, 0)
    }

    /// Scalar search starting from a base offset.
    fn find_scalar_from(&self, haystack: &[u8], base_offset: usize) -> Option<(usize, usize)> {
        for (i, window) in haystack.windows(1).enumerate() {
            let first_byte = window[0];
            let pos = base_offset + i;

            // Quick nibble check
            #[cfg(target_arch = "x86_64")]
            let pattern_mask = self.lo_nibble_table[(first_byte & 0x0F) as usize]
                & self.hi_nibble_table[(first_byte >> 4) as usize];

            #[cfg(not(target_arch = "x86_64"))]
            let pattern_mask = 0xFFu8; // Check all patterns

            if pattern_mask != 0 {
                for (pat_idx, pattern) in self.patterns.iter().enumerate() {
                    #[cfg(target_arch = "x86_64")]
                    if (pattern_mask & (1 << pat_idx)) == 0 {
                        continue;
                    }

                    if i + pattern.len() <= haystack.len() {
                        if &haystack[i..i + pattern.len()] == pattern.as_slice() {
                            return Some((pat_idx, pos));
                        }
                    }
                }
            }
        }
        None
    }
}

/// Iterator over Teddy matches.
pub struct TeddyIter<'a, 'h> {
    teddy: &'a Teddy,
    haystack: &'h [u8],
    pos: usize,
}

impl<'a, 'h> Iterator for TeddyIter<'a, 'h> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.haystack.len() {
            return None;
        }

        // Use find_from which uses cached AVX2 detection
        let result = self.teddy.find_from(self.haystack, self.pos);

        if let Some((pat_idx, abs_pos)) = result {
            // Move past this match (position is already absolute)
            self.pos = abs_pos + 1;
            Some((pat_idx, abs_pos))
        } else {
            self.pos = self.haystack.len();
            None
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn test_teddy_single() {
        let teddy = Teddy::new(vec![b"hello".to_vec()]).unwrap();
        assert_eq!(teddy.find(b"say hello world"), Some((0, 4)));
    }

    #[test]
    fn test_teddy_multiple() {
        let teddy = Teddy::new(vec![b"cat".to_vec(), b"dog".to_vec()]).unwrap();
        assert_eq!(teddy.find(b"I have a dog"), Some((1, 9)));
    }

    #[test]
    fn test_teddy_no_match() {
        let teddy = Teddy::new(vec![b"xyz".to_vec()]).unwrap();
        assert_eq!(teddy.find(b"hello world"), None);
    }

    #[test]
    fn test_teddy_first_pattern_wins() {
        let teddy = Teddy::new(vec![b"abc".to_vec(), b"abc".to_vec()]).unwrap();
        let result = teddy.find(b"xxxabcxxx");
        assert_eq!(result, Some((0, 3))); // First pattern should match
    }

    #[test]
    fn test_teddy_overlapping() {
        let teddy = Teddy::new(vec![b"aa".to_vec(), b"aaa".to_vec()]).unwrap();
        let result = teddy.find(b"xaaaax");
        // First match should be "aa" at position 1
        assert_eq!(result, Some((0, 1)));
    }

    #[test]
    fn test_teddy_at_start() {
        let teddy = Teddy::new(vec![b"hello".to_vec()]).unwrap();
        assert_eq!(teddy.find(b"hello world"), Some((0, 0)));
    }

    #[test]
    fn test_teddy_at_end() {
        let teddy = Teddy::new(vec![b"world".to_vec()]).unwrap();
        assert_eq!(teddy.find(b"hello world"), Some((0, 6)));
    }

    #[test]
    fn test_teddy_iter() {
        let teddy = Teddy::new(vec![b"a".to_vec()]).unwrap();
        let matches: Vec<_> = teddy.find_iter(b"abacada").collect();
        assert_eq!(matches, vec![(0, 0), (0, 2), (0, 4), (0, 6)]);
    }

    #[test]
    fn test_teddy_iter_multiple_patterns() {
        let teddy = Teddy::new(vec![b"a".to_vec(), b"b".to_vec()]).unwrap();
        let matches: Vec<_> = teddy.find_iter(b"abba").collect();
        assert_eq!(matches, vec![(0, 0), (1, 1), (1, 2), (0, 3)]);
    }

    #[test]
    fn test_teddy_empty_haystack() {
        let teddy = Teddy::new(vec![b"hello".to_vec()]).unwrap();
        assert_eq!(teddy.find(b""), None);
    }

    #[test]
    fn test_teddy_large_input() {
        // Test with input larger than 32 bytes to exercise SIMD path
        let teddy = Teddy::new(vec![b"needle".to_vec()]).unwrap();
        let mut haystack = vec![b'x'; 100];
        haystack[50..56].copy_from_slice(b"needle");
        assert_eq!(teddy.find(&haystack), Some((0, 50)));
    }

    #[test]
    fn test_teddy_too_many_patterns() {
        let patterns: Vec<Vec<u8>> = (0..10).map(|i| vec![b'a' + i]).collect();
        assert!(Teddy::new(patterns).is_none());
    }

    #[test]
    fn test_teddy_empty_pattern() {
        assert!(Teddy::new(vec![vec![]]).is_none());
    }
}
