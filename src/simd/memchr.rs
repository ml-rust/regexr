//! Single-byte search (memchr).
//!
//! AVX2-accelerated single byte search with scalar fallback.
//! Processes 32 bytes at a time using SIMD when available.

/// Finds the first occurrence of a byte in a slice.
///
/// Uses AVX2 SIMD when available, falls back to scalar otherwise.
pub fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked for AVX2 support
            return unsafe { memchr_avx2(needle, haystack) };
        }
    }

    // Scalar fallback
    memchr_scalar(needle, haystack)
}

/// AVX2-accelerated memchr implementation.
///
/// Processes 32 bytes at a time using SIMD comparison.
///
/// # Safety
/// - Requires AVX2 support
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn memchr_avx2(needle: u8, haystack: &[u8]) -> Option<usize> {
    use std::arch::x86_64::*;

    let len = haystack.len();
    if len == 0 {
        return None;
    }

    let ptr = haystack.as_ptr();
    let needle_vec = _mm256_set1_epi8(needle as i8);

    let mut offset = 0;

    // Process 32 bytes at a time
    while offset + 32 <= len {
        let data = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(data, needle_vec);
        let mask = _mm256_movemask_epi8(cmp) as u32;

        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }

        offset += 32;
    }

    // Handle remaining bytes (scalar)
    (offset..len).find(|&i| *haystack.get_unchecked(i) == needle)
}

/// Scalar implementation of memchr.
#[inline]
fn memchr_scalar(needle: u8, haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| b == needle)
}

/// Finds the first occurrence of any of 2 bytes.
///
/// Uses AVX2 SIMD when available.
pub fn memchr2(needle1: u8, needle2: u8, haystack: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked for AVX2 support
            return unsafe { memchr2_avx2(needle1, needle2, haystack) };
        }
    }

    // Scalar fallback
    haystack.iter().position(|&b| b == needle1 || b == needle2)
}

/// AVX2-accelerated memchr2 implementation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn memchr2_avx2(needle1: u8, needle2: u8, haystack: &[u8]) -> Option<usize> {
    use std::arch::x86_64::*;

    let len = haystack.len();
    if len == 0 {
        return None;
    }

    let ptr = haystack.as_ptr();
    let needle1_vec = _mm256_set1_epi8(needle1 as i8);
    let needle2_vec = _mm256_set1_epi8(needle2 as i8);

    let mut offset = 0;

    // Process 32 bytes at a time
    while offset + 32 <= len {
        let data = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);
        let cmp1 = _mm256_cmpeq_epi8(data, needle1_vec);
        let cmp2 = _mm256_cmpeq_epi8(data, needle2_vec);
        let combined = _mm256_or_si256(cmp1, cmp2);
        let mask = _mm256_movemask_epi8(combined) as u32;

        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }

        offset += 32;
    }

    // Handle remaining bytes (scalar)
    for i in offset..len {
        let b = *haystack.get_unchecked(i);
        if b == needle1 || b == needle2 {
            return Some(i);
        }
    }

    None
}

/// Finds the first occurrence of any of 3 bytes.
///
/// Uses AVX2 SIMD when available.
pub fn memchr3(needle1: u8, needle2: u8, needle3: u8, haystack: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked for AVX2 support
            return unsafe { memchr3_avx2(needle1, needle2, needle3, haystack) };
        }
    }

    // Scalar fallback
    haystack
        .iter()
        .position(|&b| b == needle1 || b == needle2 || b == needle3)
}

/// AVX2-accelerated memchr3 implementation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn memchr3_avx2(needle1: u8, needle2: u8, needle3: u8, haystack: &[u8]) -> Option<usize> {
    use std::arch::x86_64::*;

    let len = haystack.len();
    if len == 0 {
        return None;
    }

    let ptr = haystack.as_ptr();
    let needle1_vec = _mm256_set1_epi8(needle1 as i8);
    let needle2_vec = _mm256_set1_epi8(needle2 as i8);
    let needle3_vec = _mm256_set1_epi8(needle3 as i8);

    let mut offset = 0;

    // Process 32 bytes at a time
    while offset + 32 <= len {
        let data = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);
        let cmp1 = _mm256_cmpeq_epi8(data, needle1_vec);
        let cmp2 = _mm256_cmpeq_epi8(data, needle2_vec);
        let cmp3 = _mm256_cmpeq_epi8(data, needle3_vec);
        let combined = _mm256_or_si256(_mm256_or_si256(cmp1, cmp2), cmp3);
        let mask = _mm256_movemask_epi8(combined) as u32;

        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }

        offset += 32;
    }

    // Handle remaining bytes (scalar)
    for i in offset..len {
        let b = *haystack.get_unchecked(i);
        if b == needle1 || b == needle2 || b == needle3 {
            return Some(i);
        }
    }

    None
}

/// Reverse memchr - finds the last occurrence of a byte.
///
/// Uses AVX2 SIMD when available.
pub fn memrchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked for AVX2 support
            return unsafe { memrchr_avx2(needle, haystack) };
        }
    }

    // Scalar fallback
    haystack.iter().rposition(|&b| b == needle)
}

/// AVX2-accelerated reverse memchr implementation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn memrchr_avx2(needle: u8, haystack: &[u8]) -> Option<usize> {
    use std::arch::x86_64::*;

    let len = haystack.len();
    if len == 0 {
        return None;
    }

    let ptr = haystack.as_ptr();
    let needle_vec = _mm256_set1_epi8(needle as i8);

    // Start from the end, rounding down to 32-byte boundary
    let mut offset = (len / 32) * 32;

    // Handle trailing bytes first (scalar)
    for i in (offset..len).rev() {
        if *haystack.get_unchecked(i) == needle {
            return Some(i);
        }
    }

    // Process 32 bytes at a time, backwards
    while offset >= 32 {
        offset -= 32;
        let data = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(data, needle_vec);
        let mask = _mm256_movemask_epi8(cmp) as u32;

        if mask != 0 {
            // Find the highest set bit (last occurrence in this chunk)
            return Some(offset + 31 - mask.leading_zeros() as usize);
        }
    }

    None
}

/// Finds the first byte in a contiguous range [lo, hi].
///
/// Uses AVX2 SIMD range comparison when available.
/// This is much faster than calling memchr 10 times for digits (0-9).
pub fn memchr_range(lo: u8, hi: u8, haystack: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked for AVX2 support
            return unsafe { memchr_range_avx2(lo, hi, haystack) };
        }
    }

    // Scalar fallback
    haystack.iter().position(|&b| b >= lo && b <= hi)
}

/// AVX2-accelerated range search.
///
/// Uses two comparisons: x >= lo AND x <= hi.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn memchr_range_avx2(lo: u8, hi: u8, haystack: &[u8]) -> Option<usize> {
    use std::arch::x86_64::*;

    let len = haystack.len();
    if len == 0 {
        return None;
    }

    let ptr = haystack.as_ptr();

    // For unsigned range check [lo, hi], we use:
    // x >= lo AND x <= hi
    //
    // For unsigned comparison with signed instructions, we add 128 to convert
    // the unsigned range [0, 255] to signed range [-128, 127].
    // Then x >= lo becomes (x + 128) >= (lo + 128) using signed comparison.
    let bias = _mm256_set1_epi8(-128i8); // Add 128 as signed = -128 unsigned
    let lo_biased = _mm256_set1_epi8((lo as i8).wrapping_add(-128i8));
    let hi_biased = _mm256_set1_epi8((hi as i8).wrapping_add(-128i8));

    let mut offset = 0;

    // Process 32 bytes at a time
    while offset + 32 <= len {
        let data = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);

        // Convert to signed by adding bias (XOR with 0x80 = add 128)
        let data_biased = _mm256_add_epi8(data, bias);

        // Check x >= lo: data_biased > lo_biased - 1, or equivalently NOT(data_biased < lo_biased) AND NOT(data_biased == lo_biased - 1)
        // Simpler: use _mm256_cmpgt_epi8 for > and _mm256_cmpeq_epi8 for ==
        // x >= lo means x > lo - 1, but we need to handle underflow.
        // Instead: x >= lo is equivalent to NOT(x < lo) = NOT(lo > x)
        let lt_lo = _mm256_cmpgt_epi8(lo_biased, data_biased); // data < lo
        let gt_hi = _mm256_cmpgt_epi8(data_biased, hi_biased); // data > hi

        // In range if NOT(lt_lo) AND NOT(gt_hi)
        let out_of_range = _mm256_or_si256(lt_lo, gt_hi);
        let in_range = _mm256_andnot_si256(out_of_range, _mm256_set1_epi8(-1i8));
        let mask = _mm256_movemask_epi8(in_range) as u32;

        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }

        offset += 32;
    }

    // Handle remaining bytes (scalar)
    for i in offset..len {
        let b = *haystack.get_unchecked(i);
        if b >= lo && b <= hi {
            return Some(i);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memchr() {
        assert_eq!(memchr(b'o', b"hello"), Some(4));
        assert_eq!(memchr(b'x', b"hello"), None);
        assert_eq!(memchr(b'h', b"hello"), Some(0));
        assert_eq!(memchr(b'o', b"hello world"), Some(4));
    }

    #[test]
    fn test_memchr_empty() {
        assert_eq!(memchr(b'x', b""), None);
    }

    #[test]
    fn test_memchr_large() {
        // Test with data larger than 32 bytes to exercise SIMD path
        let data = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaax";
        assert_eq!(memchr(b'x', data), Some(64));

        let data = b"xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        assert_eq!(memchr(b'x', data), Some(0));
    }

    #[test]
    fn test_memchr2() {
        assert_eq!(memchr2(b'e', b'o', b"hello"), Some(1));
        assert_eq!(memchr2(b'x', b'y', b"hello"), None);
        assert_eq!(memchr2(b'o', b'h', b"hello"), Some(0)); // h comes first
    }

    #[test]
    fn test_memchr3() {
        assert_eq!(memchr3(b'x', b'y', b'e', b"hello"), Some(1));
        assert_eq!(memchr3(b'x', b'y', b'z', b"hello"), None);
    }

    #[test]
    fn test_memrchr() {
        assert_eq!(memrchr(b'l', b"hello"), Some(3));
        assert_eq!(memrchr(b'x', b"hello"), None);
        assert_eq!(memrchr(b'o', b"hello world"), Some(7));
    }

    #[test]
    fn test_memrchr_large() {
        // Test with data larger than 32 bytes
        let data = b"xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaax";
        assert_eq!(memrchr(b'x', data), Some(64));
    }

    #[test]
    fn test_memchr_at_chunk_boundaries() {
        // Test finding bytes at various positions around 32-byte boundaries
        for pos in [0, 1, 15, 16, 30, 31, 32, 33, 63, 64, 65] {
            if pos < 70 {
                let mut data = vec![b'a'; 70];
                data[pos] = b'x';
                assert_eq!(memchr(b'x', &data), Some(pos), "Failed at position {}", pos);
            }
        }
    }

    #[test]
    fn test_memchr_range_digits() {
        // Find digits in a string
        assert_eq!(memchr_range(b'0', b'9', b"hello 123 world"), Some(6));
        assert_eq!(memchr_range(b'0', b'9', b"no digits here"), None);
        assert_eq!(memchr_range(b'0', b'9', b"0 at start"), Some(0));
        assert_eq!(memchr_range(b'0', b'9', b"end is 9"), Some(7));
    }

    #[test]
    fn test_memchr_range_letters() {
        // Find lowercase letters
        assert_eq!(memchr_range(b'a', b'z', b"123 abc"), Some(4));
        assert_eq!(memchr_range(b'a', b'z', b"123 456"), None);
    }

    #[test]
    fn test_memchr_range_large() {
        // Test with data larger than 32 bytes
        let data = b"............................................5....................";
        assert_eq!(memchr_range(b'0', b'9', data), Some(44));
    }

    #[test]
    fn test_memchr_range_empty() {
        assert_eq!(memchr_range(b'0', b'9', b""), None);
    }

    #[test]
    fn test_memchr_range_all_match() {
        // All bytes are in range - should find first
        assert_eq!(memchr_range(b'0', b'9', b"123456"), Some(0));
    }
}
