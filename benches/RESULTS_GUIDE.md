# Benchmark Results Interpretation Guide

This guide helps you understand and interpret the benchmark results from the regexr competitive analysis.

## Understanding the Competitors

### 1. regex (Baseline)
- **Type**: Pure Rust implementation
- **Strengths**: Safe, no FFI overhead, excellent for simple patterns
- **Limitations**: No backreferences or lookaround support
- **Use as**: Industry standard baseline for comparison

### 2. fancy-regex
- **Type**: Rust wrapper around regex + backtracking engine
- **Strengths**: Supports backreferences and lookaround
- **Limitations**: Slower due to hybrid approach
- **Use as**: Feature comparison baseline

### 3. pcre2 (Interpreted)
- **Type**: C library binding, interpreted mode
- **Strengths**: Mature, well-tested, full feature set
- **Limitations**: FFI overhead, no JIT optimization
- **Use as**: PCRE2 baseline performance

### 4. pcre2-jit
- **Type**: PCRE2 with JIT compilation enabled
- **Strengths**: Very fast execution, mature JIT compiler
- **Limitations**: Compilation overhead, FFI calls
- **Use as**: Best-case PCRE2 performance

### 5. regexr-base (SIMD only)
- **Type**: regexr with SIMD acceleration, no JIT
- **Strengths**: Fast compilation, SIMD-accelerated literal search
- **Limitations**: No JIT optimization for DFA
- **Use as**: Low-overhead regexr configuration

### 6. regexr-full (SIMD + JIT)
- **Type**: regexr with both SIMD and JIT enabled
- **Strengths**: Maximum performance, native code generation
- **Limitations**: Higher compilation overhead
- **Use as**: Maximum performance regexr configuration

## Key Metrics Explained

### Throughput (Operations/Second)

**What it measures**: How many operations the engine can perform per second.

**Higher is better**: More operations = faster processing.

**Typical ranges**:
- Simple patterns: 1M+ ops/sec
- Complex patterns: 100K-500K ops/sec
- Backreferences: 10K-100K ops/sec

**When to focus on this**: Batch processing, log analysis, data pipelines

### Latency (Microseconds)

**What it measures**: Time taken for a single operation.

**Lower is better**: Less time = faster response.

**Typical ranges**:
- Small inputs (1KB): 0.1-10 μs
- Medium inputs (10KB): 1-100 μs
- Large inputs (100KB): 10-1000 μs

**When to focus on this**: Interactive applications, request processing

### Compilation Time

**What it measures**: Time to compile the regex pattern.

**Lower is better**: Less compilation time = faster startup.

**Typical ranges**:
- Simple patterns: 1-10 μs
- Complex patterns: 10-100 μs
- With JIT: 100-1000 μs

**When to focus on this**: One-time pattern matching, dynamic patterns

## Expected Performance Characteristics

### Pattern Type Impact

#### Literal Search (e.g., "error")
**Expected ranking**: regexr-full ≈ pcre2-jit > regex > regexr-base

**Why**: SIMD acceleration helps, but all engines are optimized for literals.

#### Alternation (e.g., "error|warning|critical")
**Expected ranking**: regexr-full > pcre2-jit > regex > fancy-regex

**Why**: DFA-based engines excel at alternation; JIT helps DFA compilation.

#### Character Classes (e.g., `[a-zA-Z0-9]+`)
**Expected ranking**: regexr-full > pcre2-jit ≈ regex > pcre2

**Why**: JIT-compiled DFA is optimal for character classes.

#### Capture Groups (e.g., `(\d+)-(\d+)`)
**Expected ranking**: regex ≈ regexr-full > pcre2-jit > fancy-regex

**Why**: Capture group overhead is similar across engines.

#### Backreferences (e.g., `(['"'])[^'"']*\1`)
**Expected ranking**: pcre2-jit > regexr-full > fancy-regex

**Why**: All use backtracking; PCRE2 JIT is most optimized. Note: regex crate doesn't support this.

#### Word Boundaries (e.g., `\bword\b`)
**Expected ranking**: regexr-full > pcre2-jit > regex

**Why**: Simple boundary checks; JIT helps eliminate overhead.

### Input Size Scaling

#### Small Inputs (1KB)
- Compilation overhead matters more
- Cache effects minimal
- regexr-base may be competitive due to low compilation overhead

#### Medium Inputs (10KB)
- Sweet spot for most engines
- JIT benefits become visible
- Cache effects start appearing

#### Large Inputs (100KB)
- JIT/SIMD benefits are maximum
- Cache effects significant
- regexr-full should show clear advantages

## Interpreting Speedup

### Speedup > 2.0 (Green)
**Interpretation**: Significantly faster than baseline

**Possible reasons**:
- Effective JIT compilation
- SIMD acceleration kicking in
- Better algorithm for this pattern type

**Action**: This configuration is excellent for this use case

### Speedup 0.9-1.1 (Yellow)
**Interpretation**: Comparable to baseline

**Possible reasons**:
- Similar algorithmic approach
- Overheads balance out optimizations
- Pattern type doesn't benefit from special features

**Action**: Choose based on other factors (features, compilation time)

### Speedup < 0.9 (Red)
**Interpretation**: Slower than baseline

**Possible reasons**:
- FFI overhead (for pcre2)
- Compilation overhead not amortized
- Suboptimal algorithm for this pattern
- Debug build vs release build

**Action**: Investigate why; may not be suitable for this use case

## Common Patterns in Results

### Pattern 1: JIT Shines on Repetitive Matching

**Observation**: pcre2-jit and regexr-full much faster on large inputs

**Explanation**: JIT compilation cost amortized over many matches

**Takeaway**: Use JIT for batch processing, avoid for one-time patterns

### Pattern 2: SIMD Accelerates Literal Search

**Observation**: Significant speedup on patterns with literal prefixes

**Explanation**: AVX2 can scan 32 bytes at once

**Takeaway**: Effective for log parsing, keyword search

### Pattern 3: Backreferences Are Expensive

**Observation**: All engines slower on backreference patterns

**Explanation**: Requires backtracking, can't use pure DFA

**Takeaway**: Avoid backreferences if possible; use alternatives

### Pattern 4: Compilation Overhead Varies

**Observation**: pcre2-jit and regexr-full have higher compilation times

**Explanation**: JIT code generation takes time

**Takeaway**: Compile once, reuse; cache compiled patterns

## Red Flags in Results

### Unexpected Slowdowns

**If regexr-full is slower than regexr-base**:
- Check if JIT is actually enabled (`cfg!(feature = "jit")`)
- Verify x86_64 architecture
- Check for debug vs release build

**If all engines are slow**:
- Verify input data is realistic (not pathological)
- Check for catastrophic backtracking
- Review pattern complexity

### Inconsistent Results

**If results vary widely between runs**:
- CPU frequency scaling may be active
- Other processes interfering
- Insufficient warmup iterations

**Action**: Run on quiet system, increase sample size

### Compilation Failures

**If pcre2 benchmarks fail**:
- PCRE2 library not installed
- ABI mismatch between library versions
- Platform not supported

**Action**: Check prerequisites, update libraries

## Using Results for Decision Making

### For Maximum Throughput
1. Run the full benchmark suite
2. Identify your typical use case (pattern type + input size)
3. Choose engine with highest ops/sec for that case
4. Verify compilation overhead is acceptable

### For Lowest Latency
1. Focus on latency metrics, not throughput
2. Consider cold-start scenarios (compilation + first match)
3. Test with realistic input sizes
4. Account for tail latencies (p95, p99)

### For Best Balance
1. Define acceptable latency threshold
2. Filter engines meeting threshold
3. Choose simplest/most maintainable option
4. Consider feature requirements (backreferences, etc.)

### For Production Deployment
1. Run benchmarks on production-like hardware
2. Test with production-like data
3. Measure memory usage (not just CPU)
4. Consider compilation overhead for dynamic patterns
5. Verify thread-safety requirements

## Reporting Results

When sharing benchmark results, always include:

```
## Environment
- CPU: [model name and frequency]
- RAM: [amount and type]
- OS: [operating system and version]
- Rust: [rustc version]
- PCRE2: [library version]

## Configuration
- regexr features: [default / simd / full]
- Build type: [release / debug]
- Compiler flags: [any special flags]

## Results
[Include charts, tables, or raw data]

## Analysis
[Your interpretation and conclusions]
```

## Further Reading

- [Russ Cox: Regular Expression Matching Can Be Simple And Fast](https://swtch.com/~rsc/regexp/regexp1.html)
- [Criterion.rs Statistical Analysis](https://bheisler.github.io/criterion.rs/book/analysis.html)
- [PCRE2 Performance Tips](https://www.pcre.org/current/doc/html/pcre2jit.html)
- [Regex Performance Best Practices](https://www.regular-expressions.info/performance.html)
