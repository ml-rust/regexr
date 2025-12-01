# Quick Start Guide

## Prerequisites

```bash
# Install PCRE2 (required for competitor benchmarks)
# Ubuntu/Debian:
sudo apt-get install libpcre2-dev

# macOS:
brew install pcre2

# Fedora/RHEL:
sudo dnf install pcre2-devel
```

## Run Benchmarks

### Option 1: Using the helper script (recommended)

```bash
# Run all benchmarks with default configuration
./benches/run_benchmarks.sh

# Run with SIMD only (no JIT) and generate visualizations
./benches/run_benchmarks.sh --base --visualize

# Run with SIMD + JIT and generate visualizations
./benches/run_benchmarks.sh --full --visualize
```

### Option 2: Direct cargo commands

```bash
# regexr-base (SIMD only - this is the default)
cargo bench --bench competitors

# regexr-full (SIMD + JIT)
cargo bench --bench competitors --features full

# Run specific benchmark group
cargo bench --bench competitors -- email_validation

# Run specific size
cargo bench --bench competitors -- 10KB
```

## View Results

### HTML Reports (automatically generated)

```bash
# Open in browser
xdg-open target/criterion/report/index.html   # Linux
open target/criterion/report/index.html        # macOS
```

### Generate Visualizations

```bash
# Install Python dependencies
pip install -r benches/requirements.txt

# Generate charts
python3 benches/visualize_results.py

# View results
ls benches/results/visualizations/
```

## Output Files

After running benchmarks:

- `target/criterion/` - Criterion HTML reports and raw data
- `benches/results/visualizations/` - Generated charts (PNG)
- `benches/results/detailed_results.csv` - Complete results in CSV
- `benches/results/detailed_results_summary.txt` - Human-readable summary

## Understanding Results

### Key Metrics

- **Throughput** (ops/sec): Higher is better
- **Latency** (μs): Lower is better
- **Speedup**: >1.0 means faster than baseline (regex crate)

### Comparison Matrix

| Engine | Description | Use Case |
|--------|-------------|----------|
| regex | Standard Rust regex | Baseline comparison |
| fancy-regex | Backreferences/lookaround | Advanced features |
| pcre2 | Interpreted PCRE2 | No JIT overhead |
| pcre2-jit | JIT-compiled PCRE2 | Maximum PCRE2 speed |
| regexr-base | SIMD only | Low compilation overhead |
| regexr-full | SIMD + JIT | Maximum performance |

## Common Issues

### PCRE2 not found

```
error: failed to run custom build command for `pcre2-sys`
```

**Solution**: Install libpcre2-dev (see Prerequisites above)

### Python dependencies missing

```
ModuleNotFoundError: No module named 'matplotlib'
```

**Solution**: `pip install -r benches/requirements.txt`

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Examine individual benchmark results in `target/criterion/`
- Compare configurations by running with different feature flags
- Contribute new benchmark cases for additional patterns

## Tips

1. Run benchmarks on a quiet system (close other applications)
2. Results vary by CPU - document your hardware when sharing results
3. Use `--visualize` to automatically generate charts
4. Compare `--base` vs `--full` to see JIT impact
5. Check HTML reports for statistical analysis and confidence intervals
