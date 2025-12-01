//! AVX2 SIMD utilities.
//!
//! Low-level AVX2 intrinsics wrappers for byte-level search operations.
//! All functions are marked with `#[target_feature(enable = "avx2")]` and
//! must only be called when AVX2 is available.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Broadcasts a byte to all lanes of a 256-bit vector.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub unsafe fn broadcast_byte(byte: u8) -> __m256i {
    _mm256_set1_epi8(byte as i8)
}

/// Loads 32 bytes from memory (unaligned).
///
/// # Safety
/// - Requires AVX2 support
/// - `ptr` must point to at least 32 readable bytes
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub unsafe fn load_unaligned(ptr: *const u8) -> __m256i {
    _mm256_loadu_si256(ptr as *const __m256i)
}

/// Compares two vectors for equality and returns a mask.
///
/// Each byte in the result is 0xFF if equal, 0x00 if not.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub unsafe fn compare_eq(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpeq_epi8(a, b)
}

/// Extracts a 32-bit mask from the high bits of each byte.
///
/// Bit i is set if byte i has its high bit set.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub unsafe fn movemask(v: __m256i) -> u32 {
    _mm256_movemask_epi8(v) as u32
}

/// Finds the first occurrence of a byte in a 32-byte chunk.
///
/// Returns the index within the chunk (0-31) or None if not found.
///
/// # Safety
/// - Requires AVX2 support
/// - `chunk` must have exactly 32 bytes
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub unsafe fn find_byte_in_chunk(chunk: &[u8; 32], needle: u8) -> Option<usize> {
    let needle_vec = broadcast_byte(needle);
    let data = load_unaligned(chunk.as_ptr());
    let cmp = compare_eq(data, needle_vec);
    let mask = movemask(cmp);

    if mask != 0 {
        Some(mask.trailing_zeros() as usize)
    } else {
        None
    }
}

/// Finds any of two bytes in a 32-byte chunk.
///
/// # Safety
/// - Requires AVX2 support
/// - `chunk` must have exactly 32 bytes
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub unsafe fn find_byte2_in_chunk(chunk: &[u8; 32], needle1: u8, needle2: u8) -> Option<usize> {
    let needle1_vec = broadcast_byte(needle1);
    let needle2_vec = broadcast_byte(needle2);
    let data = load_unaligned(chunk.as_ptr());

    let cmp1 = compare_eq(data, needle1_vec);
    let cmp2 = compare_eq(data, needle2_vec);
    let combined = _mm256_or_si256(cmp1, cmp2);
    let mask = movemask(combined);

    if mask != 0 {
        Some(mask.trailing_zeros() as usize)
    } else {
        None
    }
}

/// Finds any of three bytes in a 32-byte chunk.
///
/// # Safety
/// - Requires AVX2 support
/// - `chunk` must have exactly 32 bytes
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub unsafe fn find_byte3_in_chunk(
    chunk: &[u8; 32],
    needle1: u8,
    needle2: u8,
    needle3: u8,
) -> Option<usize> {
    let needle1_vec = broadcast_byte(needle1);
    let needle2_vec = broadcast_byte(needle2);
    let needle3_vec = broadcast_byte(needle3);
    let data = load_unaligned(chunk.as_ptr());

    let cmp1 = compare_eq(data, needle1_vec);
    let cmp2 = compare_eq(data, needle2_vec);
    let cmp3 = compare_eq(data, needle3_vec);
    let combined = _mm256_or_si256(_mm256_or_si256(cmp1, cmp2), cmp3);
    let mask = movemask(combined);

    if mask != 0 {
        Some(mask.trailing_zeros() as usize)
    } else {
        None
    }
}

/// Creates a vector with each byte set to its low nibble's lookup result.
///
/// Used for Teddy algorithm's nibble-based filtering.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub unsafe fn shuffle_nibble(table: __m256i, indices: __m256i) -> __m256i {
    // Extract low 4 bits of each byte
    let low_nibble = _mm256_and_si256(indices, _mm256_set1_epi8(0x0F));
    _mm256_shuffle_epi8(table, low_nibble)
}

/// Bitwise AND of two vectors.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub unsafe fn and(a: __m256i, b: __m256i) -> __m256i {
    _mm256_and_si256(a, b)
}

/// Bitwise OR of two vectors.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub unsafe fn or(a: __m256i, b: __m256i) -> __m256i {
    _mm256_or_si256(a, b)
}

/// Creates a vector with all bytes set to zero.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub unsafe fn zero() -> __m256i {
    _mm256_setzero_si256()
}

/// Creates a vector with all bytes set to 0xFF.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub unsafe fn all_ones() -> __m256i {
    _mm256_set1_epi8(-1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_byte_in_chunk() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut chunk = [0u8; 32];
        chunk[15] = b'x';

        unsafe {
            assert_eq!(find_byte_in_chunk(&chunk, b'x'), Some(15));
            assert_eq!(find_byte_in_chunk(&chunk, b'y'), None);
        }
    }

    #[test]
    fn test_find_byte2_in_chunk() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut chunk = [0u8; 32];
        chunk[10] = b'a';
        chunk[20] = b'b';

        unsafe {
            // Should find 'a' first
            assert_eq!(find_byte2_in_chunk(&chunk, b'a', b'b'), Some(10));
            // Should find 'b' when 'a' is not searched
            assert_eq!(find_byte2_in_chunk(&chunk, b'x', b'b'), Some(20));
            // Neither found
            assert_eq!(find_byte2_in_chunk(&chunk, b'x', b'y'), None);
        }
    }

    #[test]
    fn test_find_byte3_in_chunk() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut chunk = [0u8; 32];
        chunk[5] = b'c';
        chunk[15] = b'a';
        chunk[25] = b'b';

        unsafe {
            // Should find 'c' first
            assert_eq!(find_byte3_in_chunk(&chunk, b'a', b'b', b'c'), Some(5));
        }
    }
}
