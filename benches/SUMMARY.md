# regexr Benchmark Suite - Complete Summary

## Overview

This benchmark suite provides a comprehensive, scientifically rigorous comparison of the **regexr** regex engine against major competitors using real-world test cases.

## What's Included

### 📊 Benchmarks

**11 benchmark categories** covering real-world use cases:

1. **Email Validation** - Form validation, user input checking
2. **URL Extraction** - Link finding, web scraping
3. **Log Line Parsing** - Structured log analysis with capture groups
4. **JSON String Extraction** - Parsing quoted strings with escapes
5. **IP Address Matching** - Network log analysis
6. **HTML Tag Stripping** - Text sanitization
7. **Word Boundary Search** - Simple literal matching
8. **Alternation** - Multi-keyword search (error|warning|critical|fatal)
9. **Unicode Letters** - Multilingual text processing
10. **Backreferences** - Paired quote matching
11. **Compilation Time** - Regex creation overhead

### 🏁 Competitors

**6 configurations** tested:

| Configuration | Description | Use Case |
|--------------|-------------|----------|
| **regex** | Standard Rust regex | Industry baseline |
| **fancy-regex** | Backreferences + lookaround | Advanced features |
| **pcre2** | PCRE2 interpreted | Mature C library |
| **pcre2-jit** | PCRE2 with JIT | Maximum PCRE2 speed |
| **regexr-base** | SIMD only, no JIT | Fast compilation |
| **regexr-full** | SIMD + JIT | Maximum performance |

### 📏 Test Sizes

Each benchmark runs on **3 input sizes**:
- **1KB** - Small inputs, compilation overhead matters
- **10KB** - Medium inputs, sweet spot for most engines
- **100KB** - Large inputs, JIT/SIMD benefits shine

### 📈 Visualizations

**4 publication-quality charts** (PNG, 150+ DPI):

1. **Throughput Comparison** - Operations/second across all tests
2. **Latency Comparison** - Time per operation
3. **Speedup Heatmap** - Relative performance vs baseline
4. **Compilation Time** - Pattern compilation overhead

Plus **CSV export** and **human-readable summary**.

## File Structure

```
benches/
├── README.md                    # Detailed documentation
├── QUICKSTART.md               # Quick start guide
├── RESULTS_GUIDE.md            # How to interpret results
├── SUMMARY.md                  # This file
│
├── competitors.rs              # Main benchmark suite (790+ lines)
├── run_benchmarks.sh          # Automated runner script
├── visualize_results.py       # Visualization generator
├── requirements.txt           # Python dependencies
│
├── utils/
│   ├── mod.rs
│   └── test_data.rs           # Realistic test data generators
│
├── fixtures/                  # (Reserved for additional test data)
│
└── results/                   # (Generated)
    ├── detailed_results.csv
    ├── detailed_results_summary.txt
    └── visualizations/
        ├── throughput_comparison.png
        ├── latency_comparison.png
        ├── speedup_heatmap.png
        └── compilation_time.png
```

## Quick Start

### 1. Install Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install libpcre2-dev

# macOS
brew install pcre2
```

### 2. Run Benchmarks

```bash
# Easy way - using helper script
./benches/run_benchmarks.sh --full --visualize

# Or direct cargo commands
cargo bench --bench competitors                    # SIMD only (default)
cargo bench --bench competitors --features full   # SIMD + JIT
```

### 3. View Results

```bash
# HTML reports
xdg-open target/criterion/report/index.html

# Visualizations (after running visualization script)
ls benches/results/visualizations/
```

## Key Features

### ✅ Scientifically Rigorous

- **Statistical Analysis**: Criterion.rs provides confidence intervals, outlier detection
- **Sufficient Iterations**: Automatic sample size determination for significance
- **Warm-up Runs**: Accounts for JIT compilation, CPU cache effects
- **Reproducible**: Cached test data, documented environment

### ✅ Fair Comparisons

- **Same Input**: All engines tested on identical data
- **Equivalent Operations**: `is_match()` vs `is_match()`, `find_iter()` vs `find_iter()`
- **Realistic Workloads**: Test data mimics real-world usage
- **Transparent Configuration**: Feature flags clearly documented

### ✅ Comprehensive Coverage

- **10 pattern types**: Literals, alternation, character classes, captures, backreferences
- **3 input sizes**: Small, medium, large
- **3 operation types**: `is_match()`, `find()`, `captures()`
- **5 competitors**: regex, fancy-regex, pcre2, pcre2-jit, regexr

### ✅ Production-Ready

- **Automated Execution**: Helper scripts for easy running
- **Publication-Quality Output**: Charts suitable for papers, presentations
- **Extensible Design**: Easy to add new patterns or competitors
- **Well-Documented**: Multiple guides for different audiences

## Performance Insights

### Expected Strengths of regexr-full

1. **Literal Search**: SIMD acceleration provides 2-4x speedup on patterns with literal prefixes
2. **Large Inputs**: JIT compilation shines on 100KB+ inputs
3. **Alternation**: DFA-based approach excels at `error|warning|critical|fatal` patterns
4. **Character Classes**: JIT-compiled DFA optimal for `[a-zA-Z0-9]+` patterns

### Expected Trade-offs

1. **Compilation Time**: JIT has higher upfront cost (100-1000μs vs 1-10μs)
2. **Backreferences**: All engines slower (requires backtracking, not pure DFA)
3. **Small Inputs**: regexr-base may be competitive due to lower compilation overhead
4. **Unicode**: Full Unicode support adds complexity

## What Makes This Benchmark Suite Excellent

### 1. Real-World Focus

❌ **Not tested**: Contrived patterns, pathological cases, microbenchmarks
✅ **Tested**: Actual use cases from production systems

### 2. Multiple Perspectives

- **Throughput**: For batch processing
- **Latency**: For interactive applications
- **Compilation**: For dynamic patterns
- **Scalability**: Across input sizes

### 3. Actionable Results

Not just "which is faster" but:
- When is it faster?
- Why is it faster?
- What are the trade-offs?
- How to interpret results?

### 4. Extensibility

Easy to add:
- New patterns (add function + test data)
- New competitors (add dependency + benchmark calls)
- New metrics (modify visualization script)
- New input sizes (update SIZES constant)

## Usage Scenarios

### Scenario 1: "Should I use regexr in my project?"

1. Identify your typical patterns (logs? emails? URLs?)
2. Run: `./benches/run_benchmarks.sh --full --visualize`
3. Check heatmap for your pattern type
4. Read RESULTS_GUIDE.md for interpretation
5. Decide based on performance + feature requirements

### Scenario 2: "JIT vs no JIT?"

1. Run both configurations:
   ```bash
   ./benches/run_benchmarks.sh --base --visualize
   ./benches/run_benchmarks.sh --full --visualize
   ```
2. Compare compilation time charts
3. Compare throughput on your target input size
4. Choose based on usage pattern (one-time vs repeated)

### Scenario 3: "Performance regression testing"

1. Run benchmarks on main branch
2. Save results: `cp -r target/criterion target/criterion-baseline`
3. Make changes
4. Run again: `cargo bench --bench competitors`
5. Criterion automatically compares to baseline

### Scenario 4: "Academic/Blog Post"

1. Run full suite with visualization
2. Include generated charts
3. Reference RESULTS_GUIDE.md for interpretation
4. Include environment details (see RESULTS_GUIDE.md)
5. Cite this implementation

## Technical Highlights

### Test Data Generation

- **Cached**: Generated once, reused across benchmarks (zero overhead)
- **Realistic**: Mimics actual production data
- **Sized**: Precise control over input sizes
- **Varied**: Mix of matching and non-matching cases

### Statistical Rigor

- **Outlier Detection**: Criterion identifies and reports outliers
- **Confidence Intervals**: 95% CI for all measurements
- **Multiple Samples**: Sufficient iterations for significance
- **Warm-up**: Separate warm-up phase before measurement

### Visualization Quality

- **Colorblind-Friendly**: Uses seaborn colorblind palette
- **High DPI**: 150+ DPI for publication quality
- **Clear Labels**: Descriptive titles, axis labels, legends
- **Multiple Formats**: PNG images + CSV + text summary

## Limitations & Future Work

### Current Limitations

1. **Memory Usage**: Not currently benchmarked (focus on CPU performance)
2. **Unicode Normalization**: Not tested separately
3. **Lookahead/Lookbehind**: Not included (not all engines support)
4. **Streaming**: Only full-input matching tested
5. **Platform**: Tested on x86_64 Linux (JIT may vary on other platforms)

### Potential Additions

1. Memory profiling benchmarks
2. Concurrent matching benchmarks
3. Incremental matching scenarios
4. More Unicode-heavy patterns
5. Fuzzing-derived test cases
6. Real-world corpus (Apache logs, source code, etc.)

## Credits & References

### Tools Used

- [Criterion.rs](https://github.com/bheisler/criterion.rs) - Statistical benchmarking
- [regex](https://github.com/rust-lang/regex) - Baseline comparison
- [fancy-regex](https://github.com/fancy-regex/fancy-regex) - Feature comparison
- [pcre2](https://github.com/rust-pcre2/pcre2) - Industry standard comparison

### Methodology Inspired By

- [Russ Cox: Regular Expression Matching](https://swtch.com/~rsc/regexp/)
- [Rust regex benchmarks](https://github.com/rust-lang/regex/tree/master/bench)
- [Hyperscan performance analysis](https://www.hyperscan.io/performance/)

## License

Same as regexr project (MIT).

## Questions?

- **How to run**: See QUICKSTART.md
- **How to interpret**: See RESULTS_GUIDE.md
- **How to extend**: See README.md
- **Implementation details**: Read competitors.rs (heavily commented)

---

**Last Updated**: 2025-11-30
**Benchmark Suite Version**: 1.0
**regexr Version**: 0.1.0
