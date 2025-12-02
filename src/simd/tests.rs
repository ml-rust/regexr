//! Integration tests for SIMD module.
//!
//! Tests the interaction between different SIMD components and validates
//! behavior across various edge cases.

#[cfg(test)]
mod integration_tests {
    use crate::simd::*;

    #[test]
    fn test_avx2_detection() {
        // Should not panic regardless of CPU features
        let has_avx2 = is_avx2_available();
        println!("AVX2 available: {}", has_avx2);
    }

    #[test]
    fn test_memchr_vs_teddy_single_byte() {
        // For single byte search, both should give same result
        let haystack = b"The quick brown fox jumps over the lazy dog";

        let memchr_result = memchr(b'q', haystack);
        let teddy = Teddy::new(vec![b"q".to_vec()]).unwrap();
        let teddy_result = teddy.find(haystack).map(|(_, pos)| pos);

        assert_eq!(memchr_result, teddy_result);
        assert_eq!(memchr_result, Some(4));
    }

    #[test]
    fn test_memchr2_vs_teddy_two_bytes() {
        let haystack = b"hello world";

        let memchr2_result = memchr2(b'w', b'x', haystack);
        let teddy = Teddy::new(vec![b"w".to_vec(), b"x".to_vec()]).unwrap();
        let teddy_result = teddy.find(haystack).map(|(_, pos)| pos);

        assert_eq!(memchr2_result, teddy_result);
        assert_eq!(memchr2_result, Some(6));
    }

    #[test]
    fn test_unicode_handling() {
        // Invalid UTF-8 should be treated as literal bytes
        let haystack = b"hello\xFF\xFEworld";

        assert_eq!(memchr(0xFF, haystack), Some(5));
        assert_eq!(memchr(0xFE, haystack), Some(6));

        let teddy = Teddy::new(vec![vec![0xFF]]).unwrap();
        assert_eq!(teddy.find(haystack), Some((0, 5)));
    }

    #[test]
    fn test_large_haystack_performance() {
        // Test with a large haystack to ensure SIMD path is exercised
        let mut haystack = vec![b'a'; 10000];
        haystack[9999] = b'x';

        assert_eq!(memchr(b'x', &haystack), Some(9999));

        let teddy = Teddy::new(vec![b"x".to_vec()]).unwrap();
        assert_eq!(teddy.find(&haystack), Some((0, 9999)));
    }

    #[test]
    fn test_pattern_at_various_alignments() {
        // Test finding patterns at various alignments (0, 1, 15, 16, 31, 32, 33, etc.)
        for offset in [0, 1, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128] {
            let mut data = vec![b'a'; 200];
            if offset < 200 {
                data[offset] = b'X';

                assert_eq!(
                    memchr(b'X', &data),
                    Some(offset),
                    "memchr failed at offset {}",
                    offset
                );

                let teddy = Teddy::new(vec![b"X".to_vec()]).unwrap();
                assert_eq!(
                    teddy.find(&data),
                    Some((0, offset)),
                    "teddy failed at offset {}",
                    offset
                );
            }
        }
    }

    #[test]
    fn test_multiple_pattern_precedence() {
        // When multiple patterns match, should return first match position
        let haystack = b"abcdefghijklmnop";
        let teddy = Teddy::new(vec![b"def".to_vec(), b"abc".to_vec()]).unwrap();

        // "abc" is at position 0, "def" is at position 3
        // Should find "abc" first
        let result = teddy.find(haystack);
        assert_eq!(result, Some((1, 0))); // pattern 1 (abc) at position 0
    }

    #[test]
    fn test_sherlock_holmes_watson() {
        // Realistic use case from requirements
        let text = b"In the case, Sherlock Holmes and Watson investigated thoroughly";

        let teddy = Teddy::new(vec![
            b"Sherlock".to_vec(),
            b"Holmes".to_vec(),
            b"Watson".to_vec(),
        ])
        .unwrap();

        // Should find "Sherlock" first at position 13
        assert_eq!(teddy.find(text), Some((0, 13)));

        // Find all matches
        let matches: Vec<_> = teddy.find_iter(text).collect();
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0], (0, 13)); // Sherlock
        assert_eq!(matches[1], (1, 22)); // Holmes
        assert_eq!(matches[2], (2, 33)); // Watson
    }

    #[test]
    fn test_overlapping_patterns() {
        let haystack = b"aaaaa";
        let teddy = Teddy::new(vec![b"aa".to_vec(), b"aaa".to_vec()]).unwrap();

        // Should find all overlapping occurrences when iterating
        let matches: Vec<_> = teddy.find_iter(haystack).collect();
        // At positions 0, 1, 2, 3 for "aa" or "aaa"
        assert!(matches.len() >= 4);
    }

    #[test]
    fn test_memrchr_reverse_search() {
        let haystack = b"abcabcabc";

        // Forward search finds first 'a'
        assert_eq!(memchr(b'a', haystack), Some(0));

        // Reverse search finds last 'a'
        assert_eq!(memrchr(b'a', haystack), Some(6));

        // Middle character
        assert_eq!(memrchr(b'b', haystack), Some(7));
    }

    #[test]
    fn test_empty_and_short_inputs() {
        // Empty haystack
        assert_eq!(memchr(b'x', b""), None);
        assert_eq!(memchr2(b'x', b'y', b""), None);
        assert_eq!(memchr3(b'x', b'y', b'z', b""), None);
        assert_eq!(memrchr(b'x', b""), None);

        let teddy = Teddy::new(vec![b"x".to_vec()]).unwrap();
        assert_eq!(teddy.find(b""), None);

        // Single byte haystack
        assert_eq!(memchr(b'x', b"x"), Some(0));
        assert_eq!(memchr(b'y', b"x"), None);

        // Short haystack (less than 32 bytes)
        let short = b"short";
        assert_eq!(memchr(b's', short), Some(0));
        assert_eq!(memchr(b'o', short), Some(2));
        assert_eq!(memrchr(b'o', short), Some(2));
    }

    #[test]
    fn test_all_bytes_search() {
        // Test searching for each possible byte value
        for byte in 0u8..=255 {
            let mut haystack = vec![0u8; 100];
            haystack[50] = byte;

            let result = memchr(byte, &haystack);
            // byte 0 is everywhere, so it will be found at position 0
            // all other bytes should be found at position 50
            let expected = if byte == 0 { 0 } else { 50 };
            assert_eq!(result, Some(expected), "Failed to find byte 0x{:02X}", byte);
        }
    }

    #[test]
    fn test_memchr3_priority() {
        let haystack = b"abcdef";

        // When multiple needles match, should return first occurrence
        assert_eq!(memchr3(b'a', b'b', b'c', haystack), Some(0)); // 'a' comes first
        assert_eq!(memchr3(b'x', b'y', b'c', haystack), Some(2)); // only 'c' matches
        assert_eq!(memchr3(b'd', b'e', b'f', haystack), Some(3)); // 'd' comes first
    }

    #[test]
    fn test_teddy_max_patterns() {
        // Exactly MAX_PATTERNS should work
        let patterns: Vec<Vec<u8>> = (0..MAX_PATTERNS).map(|i| vec![b'a' + i as u8]).collect();
        assert!(Teddy::new(patterns).is_some());

        // MAX_PATTERNS + 1 should fail
        let too_many: Vec<Vec<u8>> = (0..MAX_PATTERNS + 1)
            .map(|i| vec![b'a' + (i % 26) as u8])
            .collect();
        assert!(Teddy::new(too_many).is_none());
    }

    #[test]
    fn test_teddy_pattern_length_limits() {
        // Empty pattern should fail
        assert!(Teddy::new(vec![vec![]]).is_none());

        // Maximum length pattern should work
        let max_pattern = vec![b'a'; MAX_PATTERN_LEN];
        assert!(Teddy::new(vec![max_pattern]).is_some());

        // Too long pattern should fail
        let too_long = vec![b'a'; MAX_PATTERN_LEN + 1];
        assert!(Teddy::new(vec![too_long]).is_none());
    }

    #[test]
    fn test_consecutive_searches() {
        // Ensure SIMD state doesn't leak between searches
        let teddy = Teddy::new(vec![b"test".to_vec()]).unwrap();

        assert_eq!(teddy.find(b"this is a test"), Some((0, 10)));
        assert_eq!(teddy.find(b"no match here"), None);
        assert_eq!(teddy.find(b"test again"), Some((0, 0)));
        assert_eq!(teddy.find(b"another test"), Some((0, 8)));
    }

    #[test]
    fn test_binary_data() {
        // Test with binary data (not text)
        let binary = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F";

        assert_eq!(memchr(0x00, binary), Some(0));
        assert_eq!(memchr(0x0F, binary), Some(15));
        assert_eq!(memchr(0xFF, binary), None);

        let teddy = Teddy::new(vec![vec![0x03, 0x04, 0x05]]).unwrap();
        assert_eq!(teddy.find(binary), Some((0, 3)));
    }

    #[test]
    fn test_repeated_pattern() {
        let haystack = b"test test test";
        let teddy = Teddy::new(vec![b"test".to_vec()]).unwrap();

        let matches: Vec<_> = teddy.find_iter(haystack).collect();
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0], (0, 0));
        assert_eq!(matches[1], (0, 5));
        assert_eq!(matches[2], (0, 10));
    }

    #[test]
    fn test_memchr_family_consistency() {
        // All memchr variants should be consistent with each other
        let haystack = b"The quick brown fox jumps over the lazy dog";

        // memchr should find first 'o'
        let pos_o = memchr(b'o', haystack).unwrap();

        // memchr2 with 'o' and a non-existent char should find same 'o'
        assert_eq!(memchr2(b'o', b'X', haystack), Some(pos_o));

        // memchr3 with 'o' and two non-existent chars should find same 'o'
        assert_eq!(memchr3(b'o', b'X', b'Y', haystack), Some(pos_o));
    }
}
