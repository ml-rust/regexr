# regexr Benchmark Suite

Comprehensive benchmarks comparing regexr against major regex engines using real-world test cases.

## Overview

This benchmark suite measures the performance of **6 different regex engine configurations**:

1. **regex** - The standard Rust regex crate (industry baseline)
2. **fancy-regex** - Rust regex with backreference and lookaround support
3. **pcre2** - PCRE2 library in interpreted mode (no JIT)
4. **pcre2-jit** - PCRE2 with JIT compilation enabled
5. **regexr-base** - regexr with SIMD acceleration only (no JIT)
6. **regexr-full** - regexr with both SIMD and JIT enabled

## Benchmark Categories

### 1. Email Validation
- **Pattern**: `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
- **Use Case**: Validating email addresses in forms, user input
- **Operations**: `is_match()` on individual email strings
- **Input Sizes**: 1KB, 10KB, 100KB

### 2. URL Extraction
- **Pattern**: `https?://[^\s<>"{}|\\^` + "`" + `\[\]]+`
- **Use Case**: Finding URLs in text, web scraping, link extraction
- **Operations**: `find_iter()` to extract all URLs
- **Input Sizes**: 1KB, 10KB, 100KB

### 3. Log Line Parsing
- **Pattern**: `(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)`
- **Use Case**: Parsing structured server logs, extracting timestamp/level/message
- **Operations**: `captures()` to extract all capture groups
- **Input Sizes**: 1KB, 10KB, 100KB

### 4. JSON String Extraction
- **Pattern**: `"([^"\\]|\\.)*"`
- **Use Case**: Extracting string literals from JSON-like text
- **Operations**: `find_iter()` to find all quoted strings
- **Input Sizes**: 1KB, 10KB, 100KB

### 5. IP Address Matching
- **Pattern**: `\b(?:\d{1,3}\.){3}\d{1,3}\b`
- **Use Case**: Finding IPv4 addresses in network logs, configuration files
- **Operations**: `find_iter()` to extract all IP addresses
- **Input Sizes**: 1KB, 10KB, 100KB

### 6. HTML Tag Stripping
- **Pattern**: `<[^>]+>`
- **Use Case**: Removing HTML tags from text, sanitization
- **Operations**: `find_iter()` to find all HTML tags
- **Input Sizes**: 1KB, 10KB, 100KB

### 7. Word Boundary Search
- **Pattern**: `\bthe\b`
- **Use Case**: Simple literal search with word boundaries, text analysis
- **Operations**: `find_iter()` to count occurrences
- **Input Sizes**: 1KB, 10KB, 100KB

### 8. Alternation (Multiple Literals)
- **Pattern**: `error|warning|critical|fatal`
- **Use Case**: Log level filtering, keyword search
- **Operations**: `find_iter()` to find all matches
- **Input Sizes**: 1KB, 10KB, 100KB

### 9. Unicode Letters
- **Pattern**: `\w+` (word characters including Unicode)
- **Use Case**: Tokenization, multilingual text processing
- **Operations**: `find_iter()` on multilingual text
- **Input Sizes**: 1KB, 10KB

### 10. Backreferences
- **Pattern**: `(['"])[^'"]*\1`
- **Use Case**: Matching paired quotes in code, string literal extraction
- **Operations**: `find_iter()` to find quoted strings
- **Input Sizes**: 1KB, 10KB, 100KB
- **Note**: Not supported by the standard `regex` crate

### 11. Compilation Time
- **Patterns**: Simple, email, complex, alternation
- **Use Case**: Measuring regex compilation overhead
- **Operations**: Regex creation/compilation

## Running the Benchmarks

### Prerequisites

```bash
# Install PCRE2 library (required for pcre2 crate)
# Ubuntu/Debian:
sudo apt-get install libpcre2-dev

# macOS:
brew install pcre2

# Fedora/RHEL:
sudo dnf install pcre2-devel
```

### Run All Benchmarks

```bash
# Run with SIMD only (default features)
cargo bench --bench competitors

# Run with SIMD + JIT (regexr-full)
cargo bench --bench competitors --features full

# Run with no acceleration (minimal build)
cargo bench --bench competitors --no-default-features
```

### Run Specific Benchmark Groups

```bash
# Only email validation benchmarks
cargo bench --bench competitors -- email

# Only URL extraction benchmarks
cargo bench --bench competitors -- url

# Only compilation time benchmarks
cargo bench --bench competitors -- compilation

# Specific size
cargo bench --bench competitors -- 10KB
```

### View Results

Criterion saves detailed results in `target/criterion/`:

```bash
# View HTML reports
open target/criterion/report/index.html

# Or on Linux:
xdg-open target/criterion/report/index.html
```

## Generating Visualizations

After running benchmarks, generate publication-quality charts:

```bash
# Install Python dependencies
pip install matplotlib pandas seaborn

# Generate visualizations
python benches/visualize_results.py
```

This creates:

1. **throughput_comparison.png** - Bar charts comparing operations/second
2. **latency_comparison.png** - Bar charts comparing latency across engines
3. **speedup_heatmap.png** - Heatmap showing speedup relative to baseline
4. **compilation_time.png** - Compilation time comparison
5. **detailed_results.csv** - Complete results in CSV format
6. **detailed_results_summary.txt** - Human-readable summary

Output location: `benches/results/visualizations/`

## Benchmark Methodology

### Statistical Rigor

- **Iterations**: Criterion automatically determines sample size for statistical significance
- **Warmup**: Multiple warmup iterations before measurements to account for:
  - JIT compilation (V8, LLVM, etc.)
  - CPU cache warming
  - Operating system scheduler stabilization
- **Outlier Detection**: Criterion identifies and reports outliers
- **Confidence Intervals**: 95% confidence intervals reported for all measurements

### Fair Comparison Principles

1. **Same Input Data**: All engines tested on identical input
2. **Equivalent Operations**: Comparing `is_match()` to `is_match()`, `find_iter()` to `find_iter()`, etc.
3. **Realistic Workloads**: Test data mimics real-world usage patterns
4. **Transparent Configuration**:
   - PCRE2 JIT explicitly enabled/disabled
   - regexr feature flags clearly documented
5. **Cold vs Warm**: Regex compilation happens once, matching is measured separately

### Test Data Generation

Test data is realistic and cached to avoid regeneration overhead:

- **Email data**: Mix of valid and invalid email addresses
- **URL data**: Real-looking URLs embedded in text
- **Log data**: Structured log lines with timestamps and levels
- **JSON data**: Valid JSON with escaped strings
- **IP data**: IPv4 addresses in various contexts
- **HTML data**: Mix of HTML tags and content
- **Text data**: Natural language text
- **Code data**: Source code with string literals
- **Unicode data**: Multilingual text (9+ languages)

## Interpreting Results

### Throughput (Higher is Better)

Measures operations per second. A 2x improvement means the engine can process twice as many operations in the same time.

### Latency (Lower is Better)

Measures time per operation in microseconds. Lower latency means faster individual operations.

### Speedup Heatmap

Shows relative performance compared to the baseline (standard `regex` crate):
- **Green (>1.0)**: Faster than baseline
- **Yellow (~1.0)**: Similar to baseline
- **Red (<1.0)**: Slower than baseline

### What to Look For

1. **Pattern Type Impact**: Some engines excel at certain pattern types
2. **Size Scaling**: How performance changes with input size
3. **JIT Benefits**: Comparing pcre2 vs pcre2-jit, regexr-base vs regexr-full
4. **Compilation Overhead**: Trade-off between compilation time and matching speed
5. **Feature Cost**: Impact of supporting backreferences/lookaround

## System Information

Benchmark results are highly dependent on hardware and software environment. When sharing results, include:

```bash
# CPU information
lscpu | grep "Model name"

# Rust version
rustc --version

# PCRE2 version
pcre2-config --version

# OS information
uname -a
```

## Benchmark Structure

```
benches/
├── README.md                    # This file
├── competitors.rs               # Main benchmark suite
├── utils/
│   ├── mod.rs
│   └── test_data.rs            # Test data generators
├── visualize_results.py        # Visualization script
├── results/
│   ├── detailed_results.csv    # Generated: CSV results
│   ├── detailed_results_summary.txt  # Generated: Text summary
│   └── visualizations/
│       ├── throughput_comparison.png
│       ├── latency_comparison.png
│       ├── speedup_heatmap.png
│       └── compilation_time.png
└── fixtures/                   # Additional test data (if needed)
```

## Extending the Benchmarks

### Adding a New Test Case

1. Add test data generator to `utils/test_data.rs`
2. Add benchmark function to `competitors.rs`
3. Add to `criterion_group!` macro
4. Update this README

Example:

```rust
fn bench_my_pattern(c: &mut Criterion) {
    let pattern = r"your-pattern-here";

    let regex_re = regex::Regex::new(pattern).unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    // ... other engines

    let data = get_test_data();

    for (size_name, _) in SIZES {
        let text = /* select appropriate test data */;

        let mut group = c.benchmark_group(format!("my_pattern/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("regex", |b| {
            b.iter(|| {
                // Your benchmark code
            })
        });

        // ... other engines

        group.finish();
    }
}
```

### Adding a New Competitor

1. Add dependency to `Cargo.toml` under `[dev-dependencies]`
2. Import in `competitors.rs`
3. Add benchmark functions for the new engine
4. Update this README

## Known Limitations

1. **PCRE2 Dependency**: Requires system installation of libpcre2
2. **Platform-Specific**: JIT may not be available on all platforms
3. **Memory Usage**: Not currently benchmarked (focus on throughput/latency)
4. **Pattern Coverage**: Not all regex features tested (e.g., no lookahead/lookbehind benchmarks yet)
5. **Unicode Normalization**: Not tested separately

## Contributing

When contributing new benchmarks:

1. Ensure fair comparison (same input, equivalent operations)
2. Use realistic test data
3. Document the use case and pattern
4. Update this README
5. Test on multiple platforms if possible

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Rust regex Crate](https://docs.rs/regex/)
- [PCRE2 Documentation](https://www.pcre.org/current/doc/html/)
- [Regex Performance Best Practices](https://www.regular-expressions.info/performance.html)

## License

Same as the regexr project (MIT).
