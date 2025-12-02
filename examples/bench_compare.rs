//  to run cargo run --release --features jit --example bench_compare

use std::time::Instant;

fn bench<F: Fn() -> usize>(iterations: u32, f: F) -> f64 {
    for _ in 0..100 {
        let _ = f();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = f();
    }
    start.elapsed().as_nanos() as f64 / iterations as f64
}

struct BenchResult {
    name: String,
    regex: f64,
    regexr: f64,
    regexr_jit: f64,
    pcre2: f64,
    pcre2_jit: f64,
    matches: usize,
}

fn run_find_bench(name: &str, pattern: &str, text: &str, iterations: u32) -> BenchResult {
    let regex_re = regex::Regex::new(pattern).unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    // Debug: print engine names
    eprintln!(
        "{}: non-jit={}, jit={}",
        name,
        regexr_re.engine_name(),
        regexr_jit.engine_name()
    );

    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();

    let matches = regex_re.find_iter(text).count();

    BenchResult {
        name: name.to_string(),
        regex: bench(iterations, || regex_re.find_iter(text).count()),
        regexr: bench(iterations, || regexr_re.find_iter(text).count()),
        regexr_jit: bench(iterations, || regexr_jit.find_iter(text).count()),
        pcre2: bench(iterations, || pcre2_re.find_iter(text.as_bytes()).count()),
        pcre2_jit: bench(iterations, || pcre2_jit.find_iter(text.as_bytes()).count()),
        matches,
    }
}

fn run_captures_bench(name: &str, pattern: &str, text: &str, iterations: u32) -> BenchResult {
    let regex_re = regex::Regex::new(pattern).unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    // Debug: print engine names
    eprintln!(
        "{}: non-jit={}, jit={}",
        name,
        regexr_re.engine_name(),
        regexr_jit.engine_name()
    );

    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();

    let matches: usize = text.lines().filter_map(|l| regex_re.captures(l)).count();

    BenchResult {
        name: name.to_string(),
        regex: bench(iterations, || {
            text.lines().filter_map(|l| regex_re.captures(l)).count()
        }),
        regexr: bench(iterations, || {
            text.lines().filter_map(|l| regexr_re.captures(l)).count()
        }),
        regexr_jit: bench(iterations, || {
            text.lines().filter_map(|l| regexr_jit.captures(l)).count()
        }),
        pcre2: bench(iterations, || {
            text.lines()
                .filter_map(|l| pcre2_re.captures(l.as_bytes()).ok().flatten())
                .count()
        }),
        pcre2_jit: bench(iterations, || {
            text.lines()
                .filter_map(|l| pcre2_jit.captures(l.as_bytes()).ok().flatten())
                .count()
        }),
        matches,
    }
}

fn run_is_match_bench(name: &str, pattern: &str, lines: &[&str], iterations: u32) -> BenchResult {
    let regex_re = regex::Regex::new(pattern).unwrap();
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    // Debug: print engine names
    eprintln!(
        "{}: non-jit={}, jit={}",
        name,
        regexr_re.engine_name(),
        regexr_jit.engine_name()
    );

    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();

    let matches: usize = lines.iter().filter(|l| regex_re.is_match(l)).count();

    BenchResult {
        name: name.to_string(),
        regex: bench(iterations, || {
            lines.iter().filter(|l| regex_re.is_match(l)).count()
        }),
        regexr: bench(iterations, || {
            lines.iter().filter(|l| regexr_re.is_match(l)).count()
        }),
        regexr_jit: bench(iterations, || {
            lines.iter().filter(|l| regexr_jit.is_match(l)).count()
        }),
        pcre2: bench(iterations, || {
            lines
                .iter()
                .filter(|l| pcre2_re.is_match(l.as_bytes()).unwrap_or(false))
                .count()
        }),
        pcre2_jit: bench(iterations, || {
            lines
                .iter()
                .filter(|l| pcre2_jit.is_match(l.as_bytes()).unwrap_or(false))
                .count()
        }),
        matches,
    }
}

fn run_backref_bench(name: &str, pattern: &str, text: &str, iterations: u32) -> BenchResult {
    // regex crate doesn't support backreferences - use regexr as baseline
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    // Debug: print engine names
    eprintln!(
        "{}: non-jit={}, jit={}",
        name,
        regexr_re.engine_name(),
        regexr_jit.engine_name()
    );

    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();

    let matches = regexr_re.find_iter(text).count();

    BenchResult {
        name: format!("{} (backref)", name),
        regex: 0.0, // Not supported
        regexr: bench(iterations, || regexr_re.find_iter(text).count()),
        regexr_jit: bench(iterations, || regexr_jit.find_iter(text).count()),
        pcre2: bench(iterations, || pcre2_re.find_iter(text.as_bytes()).count()),
        pcre2_jit: bench(iterations, || pcre2_jit.find_iter(text.as_bytes()).count()),
        matches,
    }
}

fn run_lookaround_bench(name: &str, pattern: &str, text: &str, iterations: u32) -> BenchResult {
    // regex crate doesn't support lookaround - use regexr as baseline
    let regexr_re = regexr::Regex::new(pattern).unwrap();
    let regexr_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    // Debug: print engine names
    eprintln!(
        "{}: non-jit={}, jit={}",
        name,
        regexr_re.engine_name(),
        regexr_jit.engine_name()
    );

    let pcre2_re = pcre2::bytes::Regex::new(pattern).unwrap();
    let pcre2_jit = pcre2::bytes::RegexBuilder::new()
        .jit_if_available(true)
        .build(pattern)
        .unwrap();

    let matches = regexr_re.find_iter(text).count();

    BenchResult {
        name: format!("{} (lookaround)", name),
        regex: 0.0, // Not supported
        regexr: bench(iterations, || regexr_re.find_iter(text).count()),
        regexr_jit: bench(iterations, || regexr_jit.find_iter(text).count()),
        pcre2: bench(iterations, || pcre2_re.find_iter(text.as_bytes()).count()),
        pcre2_jit: bench(iterations, || pcre2_jit.find_iter(text.as_bytes()).count()),
        matches,
    }
}

fn print_table(results: &[BenchResult]) {
    println!("\n{:=<110}", "");
    println!(
        "{:^110}",
        "BENCHMARK RESULTS (times in ns, lower is better)"
    );
    println!("{:=<110}", "");
    println!(
        "{:<30} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Test", "Matches", "regex", "regexr", "regexr-jit", "pcre2", "pcre2-jit"
    );
    println!("{:-<110}", "");

    for r in results {
        if r.regex > 0.0 {
            println!(
                "{:<30} {:>10} {:>12.0} {:>12.0} {:>12.0} {:>12.0} {:>12.0}",
                r.name, r.matches, r.regex, r.regexr, r.regexr_jit, r.pcre2, r.pcre2_jit
            );
        } else {
            println!(
                "{:<30} {:>10} {:>12} {:>12.0} {:>12.0} {:>12.0} {:>12.0}",
                r.name, r.matches, "N/A", r.regexr, r.regexr_jit, r.pcre2, r.pcre2_jit
            );
        }
    }
    println!("{:-<110}", "");

    // Print relative performance (vs regex crate, or vs regexr for backrefs)
    println!("\n{:=<110}", "");
    println!(
        "{:^110}",
        "RELATIVE PERFORMANCE (vs regex crate, <1.0 = faster)"
    );
    println!("{:=<110}", "");
    println!(
        "{:<30} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Test", "regex", "regexr", "regexr-jit", "pcre2", "pcre2-jit"
    );
    println!("{:-<110}", "");

    for r in results {
        if r.regex > 0.0 {
            println!(
                "{:<30} {:>12} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
                r.name,
                "1.00",
                r.regexr / r.regex,
                r.regexr_jit / r.regex,
                r.pcre2 / r.regex,
                r.pcre2_jit / r.regex
            );
        } else {
            // For backrefs, use regexr as baseline
            println!(
                "{:<30} {:>12} {:>12} {:>12.2} {:>12.2} {:>12.2}",
                r.name,
                "N/A",
                "1.00",
                r.regexr_jit / r.regexr,
                r.pcre2 / r.regexr,
                r.pcre2_jit / r.regexr
            );
        }
    }
    println!("{:=<110}", "");
}

fn main() {
    const ITERATIONS: u32 = 10000;

    println!("\n{:=<110}", "");
    println!("{:^110}", "QUICK REGRESSION TEST - regexr vs competitors");
    println!("{:=<110}", "");

    let mut results = Vec::new();

    // 1. JSON String extraction
    let json_pattern = r#""([^"\\]|\\.)*""#;
    let json_text = r#"{"name": "test", "value": "hello", "desc": "world"}"#.repeat(100);
    results.push(run_find_bench(
        "JSON STRING",
        json_pattern,
        &json_text,
        ITERATIONS,
    ));

    // 2. IP Address matching
    let ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b";
    let ip_text =
        "Server 192.168.1.1 connected. Client 10.0.0.5 via 172.16.0.1 gateway.\n".repeat(100);
    results.push(run_find_bench(
        "IP ADDRESS",
        ip_pattern,
        &ip_text,
        ITERATIONS,
    ));

    // 3. Log parsing with captures
    let log_pattern = r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)";
    let log_text = "2024-01-15 10:30:45 [INFO] Application started successfully\n2024-01-15 10:30:46 [DEBUG] Loading config\n".repeat(50);
    results.push(run_captures_bench(
        "LOG PARSING",
        log_pattern,
        &log_text,
        ITERATIONS,
    ));

    // 4. URL extraction
    let url_pattern = r"https?://[^\s<>]+";
    let url_text =
        "Visit https://example.com or http://test.org/path?q=1 for more info.\n".repeat(100);
    results.push(run_find_bench(
        "URL EXTRACTION",
        url_pattern,
        &url_text,
        ITERATIONS,
    ));

    // 5. Email validation
    let email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$";
    let emails: Vec<&str> = vec![
        "user@example.com",
        "test.email@domain.org",
        "invalid-email",
        "another@test.co.uk",
        "not-an-email",
        "valid@email.io",
    ]
    .into_iter()
    .cycle()
    .take(100)
    .collect();
    results.push(run_is_match_bench(
        "EMAIL VALIDATION",
        email_pattern,
        &emails,
        ITERATIONS,
    ));

    // 6. HTML tag stripping
    let html_pattern = r"<[^>]+>";
    let html_text = "<html><head><title>Test</title></head><body><p>Hello</p><div class=\"test\">World</div></body></html>\n".repeat(50);
    results.push(run_find_bench(
        "HTML TAGS",
        html_pattern,
        &html_text,
        ITERATIONS,
    ));

    // 7. Word boundary search
    let word_pattern = r"\bthe\b";
    let word_text =
        "The quick brown fox jumps over the lazy dog. Then the fox ran away from the dog.\n"
            .repeat(100);
    results.push(run_find_bench(
        "WORD BOUNDARY",
        word_pattern,
        &word_text,
        ITERATIONS,
    ));

    // 8. Alternation
    let alt_pattern = r"error|warning|critical|fatal";
    let alt_text = "2024-01-15 [INFO] Starting...\n2024-01-15 [ERROR] Failed!\n2024-01-15 [WARNING] Check this\n2024-01-15 [CRITICAL] System down\n".repeat(50);
    results.push(run_find_bench(
        "ALTERNATION",
        alt_pattern,
        &alt_text,
        ITERATIONS,
    ));

    // 9. Unicode/word characters
    let unicode_pattern = r"\w+";
    let unicode_text =
        "Hello world! This is a test with some numbers 12345 and symbols @#$%.\n".repeat(100);
    results.push(run_find_bench(
        "UNICODE WORDS",
        unicode_pattern,
        &unicode_text,
        ITERATIONS,
    ));

    // 10. Backreferences (quoted strings)
    let backref_pattern = r#"(['"])[^'"]*\1"#;
    let backref_text = r#"let x = "hello"; let y = 'world'; let z = "test's quote";"#.repeat(50);
    results.push(run_backref_bench(
        "QUOTED STRINGS",
        backref_pattern,
        &backref_text,
        ITERATIONS,
    ));

    // 11. Tokenization (complex alternation)
    let token_pattern = r#"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+(?:\.[0-9]+)?|[+\-*/=<>!&|^%]+|[(){}\[\];,.]|"[^"]*"|'[^']*'"#;
    let token_text =
        r#"fn main() { let x = 123; let y = "hello"; if x > 0 { println!("{}", y); } }"#.repeat(50);
    results.push(run_find_bench(
        "TOKENIZATION",
        token_pattern,
        &token_text,
        ITERATIONS,
    ));

    // Lookaround benchmarks use fewer iterations due to inherent cost of NFA simulation
    const LOOKAROUND_ITERATIONS: u32 = 100;

    // 12. Positive lookahead - match words followed by "ing"
    let lookahead_pos_pattern = r"\w+(?=ing\b)";
    let lookahead_text = "Running jumping swimming walking talking reading writing coding testing debugging building shipping\n".repeat(10);
    results.push(run_lookaround_bench(
        "POS LOOKAHEAD",
        lookahead_pos_pattern,
        &lookahead_text,
        LOOKAROUND_ITERATIONS,
    ));

    // 13. Negative lookahead - match "foo" NOT followed by "bar"
    let lookahead_neg_pattern = r"foo(?!bar)";
    let lookahead_neg_text = "foobar foobaz foo foobar fooqux foo foobar foo fooxyz\n".repeat(10);
    results.push(run_lookaround_bench(
        "NEG LOOKAHEAD",
        lookahead_neg_pattern,
        &lookahead_neg_text,
        LOOKAROUND_ITERATIONS,
    ));

    // 14. Positive lookbehind - match word after "@" (like email domains)
    let lookbehind_pos_pattern = r"(?<=@)\w+";
    let lookbehind_text =
        "user@example.com admin@test.org info@domain.co.uk support@company.io\n".repeat(10);
    results.push(run_lookaround_bench(
        "POS LOOKBEHIND",
        lookbehind_pos_pattern,
        &lookbehind_text,
        LOOKAROUND_ITERATIONS,
    ));

    // 15. Negative lookbehind - match digits NOT preceded by "$"
    let lookbehind_neg_pattern = r"(?<!\$)\d+";
    let lookbehind_neg_text =
        "Price: $100 Quantity: 50 Total: $500 Count: 25 Value: $1000 Items: 10\n".repeat(10);
    results.push(run_lookaround_bench(
        "NEG LOOKBEHIND",
        lookbehind_neg_pattern,
        &lookbehind_neg_text,
        LOOKAROUND_ITERATIONS,
    ));

    // 16. Complex lookahead - password validation style (word with digit ahead)
    let complex_lookahead_pattern = r"\w+(?=.*\d)";
    let complex_lookahead_text =
        "password123 secret456 admin user guest root test99 demo hello world\n".repeat(10);
    results.push(run_lookaround_bench(
        "COMPLEX LOOKAHEAD",
        complex_lookahead_pattern,
        &complex_lookahead_text,
        LOOKAROUND_ITERATIONS,
    ));

    // 17. CL100K_BASE - OpenAI GPT-4/GPT-3.5-turbo tokenizer pattern (the real target)
    // Uses negative lookahead (?!\S) so regex crate doesn't support it
    // Start with 1 iteration to test - this is a complex pattern
    const CL100K_ITERATIONS: u32 = 1;
    let cl100k_pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let cl100k_text = "Hello, world! This is a test of the cl100k_base tokenizer pattern. It's designed to handle contractions like I'm, you're, they've, and we'll. Numbers like 123 and 456789 are split into chunks. Special characters @#$%^&*() are handled too.\n".repeat(20);
    results.push(run_lookaround_bench(
        "CL100K_BASE",
        cl100k_pattern,
        &cl100k_text,
        CL100K_ITERATIONS,
    ));

    print_table(&results);
}
