//! Profile cl100k_base pattern to find bottlenecks.
//!
//! Run with: cargo run --example profile_cl100k --release --features jit

use std::time::{Duration, Instant};
use pcre2::bytes::RegexBuilder as Pcre2Builder;

// cl100k_base pattern from OpenAI's tiktoken
const CL100K_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// Sample text with mixed content
const SAMPLE_TEXT: &str = r#"
Hello, world! This is a test of the tokenization system. It's designed to handle
various text patterns including contractions like I'm, you're, we've, they'll.

Numbers: 123, 45.67, 1000000
Special chars: @#$%^&*()
Unicode: こんにちは 你好 مرحبا Привет

The quick brown fox jumps over the lazy dog. The lazy dog was not amused.
We've been working on this for a while now. It'll be ready soon, I'm sure.

    Indented text with leading spaces.
    More indented content here.

Line breaks and whitespace handling:


Multiple blank lines above.

Code-like content: fn main() { println!("Hello"); }
URLs and paths: https://example.com/path?query=value

End of sample text.
"#;

fn generate_large_text(base: &str, target_size: usize) -> String {
    let mut result = String::with_capacity(target_size);
    while result.len() < target_size {
        result.push_str(base);
        result.push('\n');
    }
    result
}

fn bench_iterations<F>(name: &str, iterations: u32, mut f: F) -> Duration
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..5 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations;

    println!("{}: {:?} total, {:?}/iter ({} iters)", name, elapsed, per_iter, iterations);
    elapsed
}

fn main() {
    println!("=== cl100k_base Pattern Profiling ===\n");
    println!("Pattern: {}\n", CL100K_PATTERN);

    // Generate test data
    let text_1kb = generate_large_text(SAMPLE_TEXT, 1024);
    let text_10kb = generate_large_text(SAMPLE_TEXT, 10 * 1024);
    let text_100kb = generate_large_text(SAMPLE_TEXT, 100 * 1024);

    println!("Test data sizes: 1KB={}, 10KB={}, 100KB={}\n",
             text_1kb.len(), text_10kb.len(), text_100kb.len());

    // Compile patterns
    println!("--- Compilation Time ---");

    let start = Instant::now();
    let re_base = regexr::Regex::new(CL100K_PATTERN).unwrap();
    println!("regexr (no JIT): {:?}", start.elapsed());

    let start = Instant::now();
    let re_jit = regexr::RegexBuilder::new(CL100K_PATTERN)
        .jit(true)
        .build()
        .unwrap();
    println!("regexr (JIT):    {:?}", start.elapsed());

    let start = Instant::now();
    let pcre2_jit = Pcre2Builder::new()
        .utf(true)
        .ucp(true)
        .jit_if_available(true)
        .build(CL100K_PATTERN)
        .unwrap();
    println!("pcre2-jit:       {:?}", start.elapsed());

    // Check what engine is being used
    println!("\n--- Engine Info ---");
    println!("Pattern has lookaround: yes (negative lookahead (?!\\S))");
    println!("Pattern has Unicode: yes (\\p{{L}}, \\p{{N}})");

    // Count matches to verify correctness
    println!("\n--- Match Counts (verification) ---");
    let count_base: usize = re_base.find_iter(&text_1kb).count();
    let count_jit: usize = re_jit.find_iter(&text_1kb).count();
    println!("1KB - base: {}, jit: {}", count_base, count_jit);
    assert_eq!(count_base, count_jit, "Match counts differ!");

    // Benchmark find_iter (main tokenization operation)
    println!("\n--- find_iter Performance ---");

    println!("\n1KB text:");
    bench_iterations("regexr-base", 1000, || {
        let _: usize = re_base.find_iter(&text_1kb).count();
    });
    bench_iterations("regexr-jit ", 1000, || {
        let _: usize = re_jit.find_iter(&text_1kb).count();
    });
    bench_iterations("pcre2-jit  ", 1000, || {
        let _: usize = pcre2_jit.find_iter(text_1kb.as_bytes()).count();
    });

    println!("\n10KB text:");
    bench_iterations("regexr-base", 100, || {
        let _: usize = re_base.find_iter(&text_10kb).count();
    });
    bench_iterations("regexr-jit ", 100, || {
        let _: usize = re_jit.find_iter(&text_10kb).count();
    });
    bench_iterations("pcre2-jit  ", 100, || {
        let _: usize = pcre2_jit.find_iter(text_10kb.as_bytes()).count();
    });

    println!("\n100KB text:");
    bench_iterations("regexr-base", 10, || {
        let _: usize = re_base.find_iter(&text_100kb).count();
    });
    bench_iterations("regexr-jit ", 10, || {
        let _: usize = re_jit.find_iter(&text_100kb).count();
    });
    bench_iterations("pcre2-jit  ", 10, || {
        let _: usize = pcre2_jit.find_iter(text_100kb.as_bytes()).count();
    });

    // Benchmark individual operations
    println!("\n--- Operation Breakdown (100KB, 10 iterations) ---");

    // Test find() which returns first match only
    println!("\nfind() - first match only:");
    bench_iterations("regexr-jit", 1000, || {
        let _ = re_jit.find(&text_100kb);
    });

    // Test is_match()
    println!("\nis_match():");
    bench_iterations("regexr-jit", 1000, || {
        let _ = re_jit.is_match(&text_100kb);
    });

    // Test individual pattern components to find bottleneck
    println!("\n--- Component Breakdown (isolate bottleneck) ---");
    println!("Testing isolated pattern components on 10KB text:\n");

    // Component 1: Contractions (case-insensitive alternation)
    let p1 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)";
    let re1 = regexr::RegexBuilder::new(p1).jit(true).build().unwrap();
    let pc1 = Pcre2Builder::new().utf(true).ucp(true).jit_if_available(true).build(p1).unwrap();
    bench_iterations("contractions regexr", 100, || { let _: usize = re1.find_iter(&text_10kb).count(); });
    bench_iterations("contractions pcre2 ", 100, || { let _: usize = pc1.find_iter(text_10kb.as_bytes()).count(); });

    // Component 2: Unicode letter sequence
    let p2 = r"\p{L}+";
    let re2 = regexr::RegexBuilder::new(p2).jit(true).build().unwrap();
    let pc2 = Pcre2Builder::new().utf(true).ucp(true).jit_if_available(true).build(p2).unwrap();
    println!();
    bench_iterations("\\p{L}+ regexr     ", 100, || { let _: usize = re2.find_iter(&text_10kb).count(); });
    bench_iterations("\\p{L}+ pcre2      ", 100, || { let _: usize = pc2.find_iter(text_10kb.as_bytes()).count(); });

    // Component 3: Unicode number sequence
    let p3 = r"\p{N}{1,3}";
    let re3 = regexr::RegexBuilder::new(p3).jit(true).build().unwrap();
    let pc3 = Pcre2Builder::new().utf(true).ucp(true).jit_if_available(true).build(p3).unwrap();
    println!();
    bench_iterations("\\p{N}{1,3} regexr ", 100, || { let _: usize = re3.find_iter(&text_10kb).count(); });
    bench_iterations("\\p{N}{1,3} pcre2  ", 100, || { let _: usize = pc3.find_iter(text_10kb.as_bytes()).count(); });

    // Component 4: Whitespace with negative lookahead
    let p4 = r"\s+(?!\S)";
    let re4 = regexr::RegexBuilder::new(p4).jit(true).build().unwrap();
    let pc4 = Pcre2Builder::new().utf(true).ucp(true).jit_if_available(true).build(p4).unwrap();
    println!();
    bench_iterations("\\s+(?!\\S) regexr ", 100, || { let _: usize = re4.find_iter(&text_10kb).count(); });
    bench_iterations("\\s+(?!\\S) pcre2  ", 100, || { let _: usize = pc4.find_iter(text_10kb.as_bytes()).count(); });

    // Component 5: Simple whitespace (no lookahead)
    let p5 = r"\s+";
    let re5 = regexr::RegexBuilder::new(p5).jit(true).build().unwrap();
    let pc5 = Pcre2Builder::new().utf(true).ucp(true).jit_if_available(true).build(p5).unwrap();
    println!();
    bench_iterations("\\s+ regexr        ", 100, || { let _: usize = re5.find_iter(&text_10kb).count(); });
    bench_iterations("\\s+ pcre2         ", 100, || { let _: usize = pc5.find_iter(text_10kb.as_bytes()).count(); });

    // Component 6: Newlines
    let p6 = r"\s*[\r\n]+";
    let re6 = regexr::RegexBuilder::new(p6).jit(true).build().unwrap();
    let pc6 = Pcre2Builder::new().utf(true).ucp(true).jit_if_available(true).build(p6).unwrap();
    println!();
    bench_iterations("\\s*[\\r\\n]+ regexr", 100, || { let _: usize = re6.find_iter(&text_10kb).count(); });
    bench_iterations("\\s*[\\r\\n]+ pcre2 ", 100, || { let _: usize = pc6.find_iter(text_10kb.as_bytes()).count(); });

    // Component 7: Optional + Unicode letter
    let p7 = r"[^\r\n\p{L}\p{N}]?\p{L}+";
    let re7 = regexr::RegexBuilder::new(p7).jit(true).build().unwrap();
    let pc7 = Pcre2Builder::new().utf(true).ucp(true).jit_if_available(true).build(p7).unwrap();
    println!();
    bench_iterations("[^...]?\\p{L}+ regexr", 100, || { let _: usize = re7.find_iter(&text_10kb).count(); });
    bench_iterations("[^...]?\\p{L}+ pcre2 ", 100, || { let _: usize = pc7.find_iter(text_10kb.as_bytes()).count(); });

    // Throughput calculation
    println!("\n--- Throughput Summary ---");
    let iterations = 10u32;
    let start = Instant::now();
    for _ in 0..iterations {
        let _: usize = re_jit.find_iter(&text_100kb).count();
    }
    let elapsed = start.elapsed();
    let bytes_processed = text_100kb.len() as u64 * iterations as u64;
    let throughput_mb = bytes_processed as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    println!("JIT Throughput: {:.2} MB/s on 100KB text", throughput_mb);

    println!("\n=== Profiling Complete ===");
}
