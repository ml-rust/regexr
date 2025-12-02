//! SIMD acceleration module for high-performance pattern matching.
//!
//! This module provides AVX2-accelerated string search routines with automatic
//! fallback to scalar implementations when AVX2 is not available.
//!
//! # Features
//!
//! - **memchr family**: Single-byte and multi-byte search (memchr, memchr2, memchr3, memrchr)
//! - **Teddy**: Multi-literal matcher using SIMD nibble hashing (up to 8 patterns)
//!
//! # Performance
//!
//! When AVX2 is available, these routines process 32 bytes per iteration, providing
//! significant speedup over scalar implementations for long haystacks.
//!
//! # Example
//!
//! ```
//! use regexr::simd::{memchr, Teddy};
//!
//! // Single byte search
//! let pos = memchr(b'x', b"hello world");
//! assert_eq!(pos, None);
//!
//! // Multi-literal search
//! let teddy = Teddy::new(vec![b"hello".to_vec(), b"world".to_vec()]).unwrap();
//! let (pattern_id, pos) = teddy.find(b"say hello there").unwrap();
//! assert_eq!(pattern_id, 0);
//! assert_eq!(pos, 4);
//! ```

mod avx2;
mod fallback;
mod memchr;
mod teddy;

#[cfg(test)]
mod tests;

pub use self::memchr::{memchr, memchr2, memchr3, memchr_range, memrchr};
pub use self::teddy::{Teddy, TeddyIter, MAX_PATTERNS, MAX_PATTERN_LEN};

/// Returns true if AVX2 SIMD instructions are available at runtime.
///
/// This checks the CPU features at runtime and returns true if AVX2 is supported.
/// When AVX2 is not available, all SIMD functions fall back to scalar implementations.
#[inline]
pub fn is_avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
