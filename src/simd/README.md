# SIMD Module - High-Performance String Search

This module provides AVX2-accelerated string search routines for the regexr regex engine. It implements the prefilter layer that quickly skips to candidate match positions before running the full DFA/NFA.

## Architecture

```
src/simd/
├── mod.rs          - Public API and runtime CPU feature detection
├── avx2.rs         - Low-level AVX2 intrinsics wrappers
├── memchr.rs       - Single-byte and multi-byte search (memchr family)
├── teddy.rs        - Multi-literal matcher using SIMD nibble hashing
├── fallback.rs     - Scalar fallback implementations
└── tests.rs        - Integration tests
```

## Features

### memchr Family

Fast single-byte and multi-byte search functions:

- `memchr(needle, haystack)` - Find first occurrence of a single byte
- `memchr2(n1, n2, haystack)` - Find first occurrence of either of 2 bytes
- `memchr3(n1, n2, n3, haystack)` - Find first occurrence of any of 3 bytes
- `memrchr(needle, haystack)` - Find last occurrence of a byte (reverse search)

**Algorithm**: Broadcasts the needle to all 32 lanes of an AVX2 register, compares 32 bytes at once using `_mm256_cmpeq_epi8`, and extracts a bitmask to find the first match.

**Performance**: Processes 32 bytes per iteration when AVX2 is available.

### Teddy Multi-Literal Matcher

SIMD-accelerated algorithm for matching multiple literal patterns simultaneously.

**Capabilities**:
- Match 1-8 patterns simultaneously
- Pattern length: 1-8 bytes
- Returns first match position and pattern ID

**Algorithm**:
1. Build nibble lookup tables for low/high nibbles of each pattern's first byte
2. Extract low and high nibbles from 32 input bytes in parallel
3. Use `_mm256_shuffle_epi8` for parallel table lookup
4. AND the results to find candidate positions
5. Verify candidates with full pattern comparison

**Use Cases**:
- Regex literal prefixes: `Sherlock|Holmes|Watson`
- Multiple keyword search
- DNA/protein sequence matching

## Performance Characteristics

### SIMD Path (AVX2)

- Throughput: 32 bytes per iteration
- Best case: ~10-30x faster than scalar for large haystacks
- Overhead: ~10-20 cycles for setup

### Scalar Fallback

- Automatically used when AVX2 is not available
- Uses optimized iterator-based search
- No performance penalty when SIMD is not needed

### When SIMD Helps Most

- Large haystacks (>100 bytes)
- Needle is rare in haystack
- Multiple patterns (Teddy)

### When Scalar is Better

- Very small haystacks (<32 bytes)
- Needle is very common (found in first few bytes)
- Single-byte search in short strings

## Runtime CPU Detection

The module uses runtime feature detection (`is_x86_feature_detected!("avx2")`) to automatically select the best implementation:

```rust
pub fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    if is_x86_feature_detected!("avx2") {
        unsafe { memchr_avx2(needle, haystack) }
    } else {
        memchr_scalar(needle, haystack)
    }
}
```

You can check AVX2 availability:

```rust
use regexr::simd::is_avx2_available;

if is_avx2_available() {
    println!("Using AVX2 SIMD acceleration");
}
```

## Safety

All AVX2 intrinsic code is marked with:
- `#[target_feature(enable = "avx2")]` - Ensures correct code generation
- `unsafe` blocks with `// SAFETY:` comments explaining invariants
- Runtime feature detection before calling SIMD code

### Safety Invariants

1. **AVX2 availability**: Always checked via `is_x86_feature_detected!` before calling AVX2 code
2. **Bounds checking**: Loops ensure we never read past the haystack end
3. **Tail bytes**: Remaining bytes (<32) are handled with scalar loops
4. **Unaligned loads**: Use `_mm256_loadu_si256` for unaligned memory access

## Edge Cases Handled

### Empty Inputs
- Empty haystack returns `None` (memchr) or appropriate result
- Empty pattern rejected by Teddy constructor

### Short Haystacks
- Haystacks shorter than 32 bytes handled correctly
- No SIMD overhead for very short inputs

### Alignment
- All loads use unaligned intrinsics (`_mm256_loadu_si256`)
- No alignment requirements on input data

### Invalid UTF-8
- All functions operate on bytes, not characters
- Invalid UTF-8 bytes (0x80-0xFF) treated as literal bytes
- Never panics on invalid input

### Boundary Conditions
- Needle at position 0: Correctly found
- Needle at last position: Correctly found
- Needle spanning chunk boundary: Handled by tail processing

## Examples

### Basic Single-Byte Search

```rust
use regexr::simd::memchr;

let haystack = b"The quick brown fox";
assert_eq!(memchr(b'q', haystack), Some(4));
assert_eq!(memchr(b'z', haystack), None);
```

### Multi-Byte Search

```rust
use regexr::simd::memchr3;

let haystack = b"hello world";
// Find first vowel
assert_eq!(memchr3(b'a', b'e', b'i', haystack), Some(1)); // 'e' in "hello"
```

### Reverse Search

```rust
use regexr::simd::memrchr;

let haystack = b"abcabcabc";
assert_eq!(memrchr(b'a', haystack), Some(6)); // Last 'a'
```

### Multi-Literal Matching (Teddy)

```rust
use regexr::simd::Teddy;

let patterns = vec![
    b"Sherlock".to_vec(),
    b"Holmes".to_vec(),
    b"Watson".to_vec(),
];
let teddy = Teddy::new(patterns).unwrap();

let text = b"Sherlock Holmes and Watson investigated";
let (pattern_id, position) = teddy.find(text).unwrap();
assert_eq!(pattern_id, 0); // "Sherlock"
assert_eq!(position, 0);
```

### Finding All Matches

```rust
use regexr::simd::Teddy;

let teddy = Teddy::new(vec![b"cat".to_vec(), b"dog".to_vec()]).unwrap();
let text = b"I have a cat and a dog";

for (pattern_id, pos) in teddy.find_iter(text) {
    println!("Found pattern {} at position {}", pattern_id, pos);
}
// Output:
// Found pattern 0 at position 9  (cat)
// Found pattern 1 at position 19 (dog)
```

## Testing

The module includes comprehensive tests (45 test functions across modules):

- **Unit tests**: In each module file (avx2.rs, memchr.rs, teddy.rs, fallback.rs)
- **Integration tests**: In tests.rs covering cross-module behavior
- **Edge case tests**: Empty inputs, boundaries, alignment, all byte values
- **Property tests**: Consistency between SIMD and scalar implementations

Run tests:
```bash
# All SIMD tests
cargo test --lib simd --features simd

# Specific test
cargo test --lib simd::tests::test_sherlock_holmes_watson --features simd

# With AVX2 disabled (forces scalar path)
RUSTFLAGS='-C target-feature=-avx2' cargo test --lib simd --features simd
```

## Implementation Notes

### AVX2 Register Layout

AVX2 registers are 256 bits (32 bytes):
```
|-------- 256 bits (32 bytes) --------|
| byte 0 | byte 1 | ... | byte 31 |
```

### vpshufb Behavior in AVX2

`_mm256_shuffle_epi8` operates on two 128-bit lanes independently. For Teddy, we duplicate the 16-byte lookup table in both lanes.

### Bitmask Extraction

`_mm256_movemask_epi8` extracts the MSB of each byte into a 32-bit mask:
```
Bytes:  [0xFF, 0x00, 0xFF, 0x00, ...]
Mask:   0b1010...
        └─ bit 0 set (byte 0 had MSB set)
           └─ bit 2 set (byte 2 had MSB set)
```

We use `trailing_zeros()` to find the first set bit (first match).

### Tail Processing

When haystack length is not a multiple of 32, the last <32 bytes are processed with a scalar loop. Alternative approaches (overlapping reads) could reduce branches but add complexity.

## Future Optimizations

Potential improvements not yet implemented:

1. **AVX-512**: Support for 64-byte vectors on newer CPUs
2. **SSE2/SSSE3**: Fallback to 16-byte SIMD for older CPUs
3. **Overlapping reads**: Avoid scalar tail by reading last 32 bytes with overlap
4. **Aligned loads**: Branch on alignment for `_mm256_load_si256` (faster but complex)
5. **Huge page support**: For very large haystacks
6. **Multi-threaded search**: Parallel search for multi-gigabyte inputs

## References

- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- Hyperscan Teddy algorithm: https://github.com/intel/hyperscan
- ripgrep SIMD implementation: https://github.com/BurntSushi/ripgrep/tree/master/crates/grep
- "SIMD-friendly algorithms for substring searching": https://arxiv.org/abs/1612.01506
