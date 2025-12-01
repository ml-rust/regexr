//! Comprehensive benchmark suite comparing regexr against major regex engines.
//!
//! This benchmark compares 6 configurations:
//! 1. regex - Standard Rust regex crate
//! 2. fancy-regex - Rust regex with backreferences and lookaround
//! 3. pcre2 - PCRE2 without JIT
//! 4. pcre2-jit - PCRE2 with JIT enabled
//! 5. regexr-base - regexr with SIMD only (no JIT)
//! 6. regexr-full - regexr with SIMD + JIT (when available)
//!
//! Run with:
//!   cargo bench --bench competitors                    # regexr-full (default features)
//!   cargo bench --bench competitors --no-default-features --features simd  # regexr-base

mod utils;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::time::Duration;
use utils::test_data::{generate_unicode_data, get_test_data};

/// Test configuration with different input sizes
const SIZES: &[(&str, usize)] = &[("1KB", 1024), ("10KB", 10 * 1024), ("100KB", 100 * 1024)];

/// Configure criterion for faster benchmarks
fn fast_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(1))
        .warm_up_time(Duration::from_millis(200))
}

// ============================================================================
// EMAIL VALIDATION BENCHMARKS
// ============================================================================

fn bench_email_validation(c: &mut Criterion) {
    let pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$";

    let regex_re = regex::Regex::new(pattern).unwrap();
    let fancy_re = fancy_regex::Regex::new(pattern).unwrap();
    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    let data = get_test_data();

    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.emails_1kb,
            "10KB" => &data.emails_10kb,
            "100KB" => &data.emails_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("email_validation/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("regex", |b| {
            b.iter(|| {
                for line in text.lines() {
                    black_box(regex_re.is_match(black_box(line)));
                }
            })
        });

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                for line in text.lines() {
                    black_box(fancy_re.is_match(black_box(line)).unwrap());
                }
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                for line in text.lines() {
                    black_box(pcre2_re.is_match(black_box(line.as_bytes())).unwrap());
                }
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                for line in text.lines() {
                    black_box(pcre2_jit.is_match(black_box(line.as_bytes())).unwrap());
                }
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                for line in text.lines() {
                    black_box(regexr_re.is_match(black_box(line)));
                }
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                for line in text.lines() {
                    black_box(regexr_jit.is_match(black_box(line)));
                }
            })
        });

        group.finish();
    }
}

// ============================================================================
// URL EXTRACTION BENCHMARKS
// ============================================================================

fn bench_url_extraction(c: &mut Criterion) {
    // Simplified URL pattern that works across all engines
    let pattern = r"https?://[^\s<>]+";

    let regex_re = regex::Regex::new(pattern).unwrap();
    let fancy_re = fancy_regex::Regex::new(pattern).unwrap();
    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    let data = get_test_data();

    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.urls_1kb,
            "10KB" => &data.urls_10kb,
            "100KB" => &data.urls_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("url_extraction/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("regex", |b| {
            b.iter(|| {
                let count = regex_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let count = fancy_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let count = pcre2_re.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let count = pcre2_jit.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let count = regexr_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let count = regexr_jit.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.finish();
    }
}

// ============================================================================
// LOG LINE PARSING BENCHMARKS
// ============================================================================

fn bench_log_parsing(c: &mut Criterion) {
    let pattern = r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)";

    let regex_re = regex::Regex::new(pattern).unwrap();
    let fancy_re = fancy_regex::Regex::new(pattern).unwrap();
    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    let data = get_test_data();

    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.logs_1kb,
            "10KB" => &data.logs_10kb,
            "100KB" => &data.logs_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("log_parsing/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("regex", |b| {
            b.iter(|| {
                for line in text.lines() {
                    if let Some(caps) = regex_re.captures(black_box(line)) {
                        black_box(&caps[1]);
                        black_box(&caps[2]);
                        black_box(&caps[3]);
                        black_box(&caps[4]);
                    }
                }
            })
        });

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                for line in text.lines() {
                    if let Ok(Some(caps)) = fancy_re.captures(black_box(line)) {
                        black_box(caps.get(1));
                        black_box(caps.get(2));
                        black_box(caps.get(3));
                        black_box(caps.get(4));
                    }
                }
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                for line in text.lines() {
                    if let Ok(Some(caps)) = pcre2_re.captures(black_box(line.as_bytes())) {
                        black_box(caps.get(1));
                        black_box(caps.get(2));
                        black_box(caps.get(3));
                        black_box(caps.get(4));
                    }
                }
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                for line in text.lines() {
                    if let Ok(Some(caps)) = pcre2_jit.captures(black_box(line.as_bytes())) {
                        black_box(caps.get(1));
                        black_box(caps.get(2));
                        black_box(caps.get(3));
                        black_box(caps.get(4));
                    }
                }
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                for line in text.lines() {
                    if let Some(caps) = regexr_re.captures(black_box(line)) {
                        black_box(caps.get(1));
                        black_box(caps.get(2));
                        black_box(caps.get(3));
                        black_box(caps.get(4));
                    }
                }
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                for line in text.lines() {
                    if let Some(caps) = regexr_jit.captures(black_box(line)) {
                        black_box(caps.get(1));
                        black_box(caps.get(2));
                        black_box(caps.get(3));
                        black_box(caps.get(4));
                    }
                }
            })
        });

        group.finish();
    }
}

// ============================================================================
// JSON STRING EXTRACTION BENCHMARKS
// ============================================================================

fn bench_json_strings(c: &mut Criterion) {
    let pattern = r#""([^"\\]|\\.)*""#;

    let regex_re = regex::Regex::new(pattern).unwrap();
    let fancy_re = fancy_regex::Regex::new(pattern).unwrap();
    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    let data = get_test_data();

    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.json_1kb,
            "10KB" => &data.json_10kb,
            "100KB" => &data.json_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("json_strings/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("regex", |b| {
            b.iter(|| {
                let count = regex_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let count = fancy_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let count = pcre2_re.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let count = pcre2_jit.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let count = regexr_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let count = regexr_jit.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.finish();
    }
}

// ============================================================================
// IP ADDRESS MATCHING BENCHMARKS
// ============================================================================

fn bench_ip_addresses(c: &mut Criterion) {
    let pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b";

    let regex_re = regex::Regex::new(pattern).unwrap();
    let fancy_re = fancy_regex::Regex::new(pattern).unwrap();
    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    let data = get_test_data();

    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.ips_1kb,
            "10KB" => &data.ips_10kb,
            "100KB" => &data.ips_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("ip_addresses/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("regex", |b| {
            b.iter(|| {
                let count = regex_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let count = fancy_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let count = pcre2_re.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let count = pcre2_jit.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let count = regexr_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let count = regexr_jit.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.finish();
    }
}

// ============================================================================
// HTML TAG STRIPPING BENCHMARKS
// ============================================================================

fn bench_html_tags(c: &mut Criterion) {
    let pattern = r"<[^>]+>";

    let regex_re = regex::Regex::new(pattern).unwrap();
    let fancy_re = fancy_regex::Regex::new(pattern).unwrap();
    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    let data = get_test_data();

    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.html_1kb,
            "10KB" => &data.html_10kb,
            "100KB" => &data.html_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("html_tags/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("regex", |b| {
            b.iter(|| {
                let count = regex_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let count = fancy_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let count = pcre2_re.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let count = pcre2_jit.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let count = regexr_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let count = regexr_jit.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.finish();
    }
}

// ============================================================================
// WORD BOUNDARY SEARCH BENCHMARKS
// ============================================================================

fn bench_word_boundary(c: &mut Criterion) {
    let pattern = r"\bthe\b";

    let regex_re = regex::Regex::new(pattern).unwrap();
    let fancy_re = fancy_regex::Regex::new(pattern).unwrap();
    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    let data = get_test_data();

    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.text_1kb,
            "10KB" => &data.text_10kb,
            "100KB" => &data.text_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("word_boundary/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("regex", |b| {
            b.iter(|| {
                let count = regex_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let count = fancy_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let count = pcre2_re.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let count = pcre2_jit.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let count = regexr_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let count = regexr_jit.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.finish();
    }
}

// ============================================================================
// ALTERNATION BENCHMARKS
// ============================================================================

fn bench_alternation(c: &mut Criterion) {
    let pattern = r"error|warning|critical|fatal";

    let regex_re = regex::Regex::new(pattern).unwrap();
    let fancy_re = fancy_regex::Regex::new(pattern).unwrap();
    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    let data = get_test_data();

    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.logs_1kb,
            "10KB" => &data.logs_10kb,
            "100KB" => &data.logs_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("alternation/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("regex", |b| {
            b.iter(|| {
                let count = regex_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let count = fancy_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let count = pcre2_re.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let count = pcre2_jit.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let count = regexr_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let count = regexr_jit.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.finish();
    }
}

// ============================================================================
// UNICODE PROPERTY BENCHMARKS (if supported)
// ============================================================================

fn bench_unicode_letters(c: &mut Criterion) {
    // Using \w+ as a proxy since \p{L}+ may not be supported by all engines
    let pattern = r"\w+";

    let regex_re = regex::Regex::new(pattern).unwrap();
    let fancy_re = fancy_regex::Regex::new(pattern).unwrap();
    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    let sizes = &[("1KB", 1024), ("10KB", 10 * 1024)];

    for (size_name, size) in sizes {
        let text = generate_unicode_data(*size);

        let mut group = c.benchmark_group(format!("unicode_letters/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("regex", |b| {
            b.iter(|| {
                let count = regex_re.find_iter(black_box(&text)).count();
                black_box(count);
            })
        });

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let count = fancy_re.find_iter(black_box(&text)).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let count = pcre2_re.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let count = pcre2_jit.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let count = regexr_re.find_iter(black_box(&text)).count();
                black_box(count);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let count = regexr_jit.find_iter(black_box(&text)).count();
                black_box(count);
            })
        });

        group.finish();
    }
}

// ============================================================================
// BACKREFERENCE BENCHMARKS (fancy-regex, pcre2, regexr only)
// ============================================================================

fn bench_backreferences(c: &mut Criterion) {
    let pattern = r#"(['"])[^'"]*\1"#;

    // regex crate doesn't support backreferences, so we skip it
    let fancy_re = fancy_regex::Regex::new(pattern).unwrap();
    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    let data = get_test_data();

    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.code_1kb,
            "10KB" => &data.code_10kb,
            "100KB" => &data.code_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("backreferences/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let count = fancy_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let count = pcre2_re.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let count = pcre2_jit.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let count = regexr_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let count = regexr_jit.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.finish();
    }
}

// ============================================================================
// LOOKAHEAD BENCHMARKS (fancy-regex, pcre2, regexr only - regex crate doesn't support)
// ============================================================================

fn bench_lookahead(c: &mut Criterion) {
    // Positive lookahead: match word followed by specific suffix
    let pattern_positive = r"\w+(?=ing\b)";
    // Negative lookahead: match word NOT followed by 's'
    let pattern_negative = r"\w+(?![s]\b)";

    // regex crate doesn't support lookahead, so we skip it
    let fancy_pos = fancy_regex::Regex::new(pattern_positive).unwrap();
    let fancy_neg = fancy_regex::Regex::new(pattern_negative).unwrap();
    let pcre2_pos = pcre2::bytes::Regex::new(pattern_positive).unwrap();
    let pcre2_neg = pcre2::bytes::Regex::new(pattern_negative).unwrap();
    let pcre2_jit_pos = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern_positive)
        .unwrap();
    let pcre2_jit_neg = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern_negative)
        .unwrap();
    let regexr_pos = regexr::Regex::new(pattern_positive).unwrap();
    let regexr_neg = regexr::Regex::new(pattern_negative).unwrap();
    let regexr_jit_pos = regexr::RegexBuilder::new(pattern_positive)
        .jit(true)
        .build()
        .unwrap();
    let regexr_jit_neg = regexr::RegexBuilder::new(pattern_negative)
        .jit(true)
        .build()
        .unwrap();

    let data = get_test_data();

    // Positive lookahead benchmark
    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.text_1kb,
            "10KB" => &data.text_10kb,
            "100KB" => &data.text_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("lookahead_positive/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let count = fancy_pos.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let count = pcre2_pos.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let count = pcre2_jit_pos.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let count = regexr_pos.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let count = regexr_jit_pos.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.finish();
    }

    // Negative lookahead benchmark
    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.text_1kb,
            "10KB" => &data.text_10kb,
            "100KB" => &data.text_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("lookahead_negative/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let count = fancy_neg.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let count = pcre2_neg.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let count = pcre2_jit_neg.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let count = regexr_neg.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let count = regexr_jit_neg.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.finish();
    }
}

// ============================================================================
// LOOKBEHIND BENCHMARKS (fancy-regex, pcre2, regexr only - regex crate doesn't support)
// ============================================================================

fn bench_lookbehind(c: &mut Criterion) {
    // Positive lookbehind: match word preceded by '@' (like email usernames)
    let pattern_positive = r"(?<=@)\w+";
    // Negative lookbehind: match digits NOT preceded by '$'
    let pattern_negative = r"(?<!\$)\d+";

    // regex crate doesn't support lookbehind, so we skip it
    let fancy_pos = fancy_regex::Regex::new(pattern_positive).unwrap();
    let fancy_neg = fancy_regex::Regex::new(pattern_negative).unwrap();
    let pcre2_pos = pcre2::bytes::Regex::new(pattern_positive).unwrap();
    let pcre2_neg = pcre2::bytes::Regex::new(pattern_negative).unwrap();
    let pcre2_jit_pos = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern_positive)
        .unwrap();
    let pcre2_jit_neg = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern_negative)
        .unwrap();
    let regexr_pos = regexr::Regex::new(pattern_positive).unwrap();
    let regexr_neg = regexr::Regex::new(pattern_negative).unwrap();
    let regexr_jit_pos = regexr::RegexBuilder::new(pattern_positive)
        .jit(true)
        .build()
        .unwrap();
    let regexr_jit_neg = regexr::RegexBuilder::new(pattern_negative)
        .jit(true)
        .build()
        .unwrap();

    let data = get_test_data();

    // Positive lookbehind benchmark
    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.emails_1kb,
            "10KB" => &data.emails_10kb,
            "100KB" => &data.emails_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("lookbehind_positive/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let count = fancy_pos.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let count = pcre2_pos.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let count = pcre2_jit_pos.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let count = regexr_pos.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let count = regexr_jit_pos.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.finish();
    }

    // Negative lookbehind benchmark
    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.code_1kb,
            "10KB" => &data.code_10kb,
            "100KB" => &data.code_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("lookbehind_negative/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let count = fancy_neg.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let count = pcre2_neg.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let count = pcre2_jit_neg.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let count = regexr_neg.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let count = regexr_jit_neg.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.finish();
    }
}

// ============================================================================
// COMPILATION TIME BENCHMARKS
// ============================================================================

fn bench_compilation_time(c: &mut Criterion) {
    let patterns = vec![
        ("simple", r"hello"),
        ("email", r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        (
            "complex",
            r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)",
        ),
        (
            "alternation",
            r"error|warning|critical|fatal|info|debug|trace",
        ),
    ];

    for (name, pattern) in patterns {
        let mut group = c.benchmark_group(format!("compilation/{}", name));

        group.bench_function("regex", |b| {
            b.iter(|| {
                let re = regex::Regex::new(black_box(pattern)).unwrap();
                black_box(re);
            })
        });

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let re = fancy_regex::Regex::new(black_box(pattern)).unwrap();
                black_box(re);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let re = pcre2::bytes::Regex::new(black_box(pattern)).unwrap();
                black_box(re);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let re = pcre2::bytes::RegexBuilder::new()
                    .jit_if_available(true)
                    .build(black_box(pattern))
                    .unwrap();
                black_box(re);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let re = regexr::Regex::new(black_box(pattern)).unwrap();
                black_box(re);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let re = regexr::RegexBuilder::new(black_box(pattern))
                    .jit(true)
                    .build()
                    .unwrap();
                black_box(re);
            })
        });

        group.finish();
    }
}

// ============================================================================
// TOKENIZATION BENCHMARKS
// ============================================================================

fn bench_tokenization(c: &mut Criterion) {
    // Pattern to tokenize programming language constructs:
    // identifiers, numbers, operators, punctuation, strings
    let pattern = r#"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+(?:\.[0-9]+)?|[+\-*/=<>!&|^%]+|[(){}\[\];,.]|"[^"]*"|'[^']*'"#;

    let regex_re = regex::Regex::new(pattern).unwrap();
    let fancy_re = fancy_regex::Regex::new(pattern).unwrap();
    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    let data = get_test_data();

    for (size_name, _) in SIZES {
        let text = match *size_name {
            "1KB" => &data.code_1kb,
            "10KB" => &data.code_10kb,
            "100KB" => &data.code_100kb,
            _ => unreachable!(),
        };

        let mut group = c.benchmark_group(format!("tokenization/{}", size_name));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("regex", |b| {
            b.iter(|| {
                let count = regex_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("fancy-regex", |b| {
            b.iter(|| {
                let count = fancy_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2", |b| {
            b.iter(|| {
                let count = pcre2_re.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("pcre2-jit", |b| {
            b.iter(|| {
                let count = pcre2_jit.find_iter(black_box(text.as_bytes())).count();
                black_box(count);
            })
        });

        group.bench_function("regexr", |b| {
            b.iter(|| {
                let count = regexr_re.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.bench_function("regexr-jit", |b| {
            b.iter(|| {
                let count = regexr_jit.find_iter(black_box(text)).count();
                black_box(count);
            })
        });

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = fast_config();
    targets =
        bench_email_validation,
        bench_url_extraction,
        bench_log_parsing,
        bench_json_strings,
        bench_ip_addresses,
        bench_html_tags,
        bench_word_boundary,
        bench_alternation,
        bench_unicode_letters,
        bench_backreferences,
        bench_lookahead,
        bench_lookbehind,
        bench_tokenization,
        bench_compilation_time,
}
criterion_main!(benches);
