//! Performance test for backreferences - comparing JIT vs interpreter
//!
//! Run with JIT:    cargo test --features jit --test perf_backref -- --nocapture
//! Run without JIT: cargo test --test perf_backref -- --nocapture

use std::time::Instant;

fn generate_code_data(target_size: usize) -> String {
    let code_snippets = [
        r#"let x = "hello world";"#,
        r#"const y = 'single quoted string';"#,
        r#"var z = "escaped \"quotes\" inside";"#,
        r#"String s = "Java string literal";"#,
        r#"str = 'Python string with "nested" quotes';"#,
    ];

    let mut result = String::with_capacity(target_size);

    while result.len() < target_size {
        result.push_str(code_snippets[result.len() % code_snippets.len()]);
        result.push('\n');
    }

    result.truncate(target_size);
    result
}

#[test]
fn test_backref_performance() {
    let pattern = r#"(['"])[^'"]*\1"#;

    // Test data sizes
    let sizes = [("1KB", 1024), ("10KB", 10 * 1024)];

    println!("\n=== Backreference Performance Test ===");
    println!("Pattern: {}", pattern);

    #[cfg(feature = "jit")]
    println!("Mode: JIT enabled");
    #[cfg(not(feature = "jit"))]
    println!("Mode: Interpreter only (no JIT)");
    println!();

    let re = regexr::Regex::new(pattern).unwrap();

    #[cfg(feature = "jit")]
    let re_jit = regexr::RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap();

    for (name, size) in sizes {
        let data = generate_code_data(size);

        // Warm up
        let _ = re.find_iter(&data).count();
        #[cfg(feature = "jit")]
        let _ = re_jit.find_iter(&data).count();

        // Benchmark default regex
        let iterations = 10;
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..iterations {
            count = re.find_iter(&data).count();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iterations;
        println!(
            "{} - Default regex: {:?} per iteration, {} matches",
            name, per_iter, count
        );

        // Benchmark JIT regex
        #[cfg(feature = "jit")]
        {
            let start = Instant::now();
            let mut count_jit = 0;
            for _ in 0..iterations {
                count_jit = re_jit.find_iter(&data).count();
            }
            let elapsed = start.elapsed();
            let per_iter = elapsed / iterations;
            println!(
                "{} - JIT regex:     {:?} per iteration, {} matches",
                name, per_iter, count_jit
            );

            // Verify same results
            assert_eq!(
                count, count_jit,
                "JIT and default should find same number of matches"
            );
        }

        println!();
    }
}

#[test]
fn test_backref_simple() {
    let pattern = r#"(['"])[^'"]*\1"#;
    let re = regexr::Regex::new(pattern).unwrap();

    // Simple test cases
    let test_cases = [
        (r#""hello""#, true, r#""hello""#),
        (r#"'world'"#, true, r#"'world'"#),
        (r#""mixed'"#, false, ""),
        (r#"'mixed""#, false, ""),
        (r#"let x = "test";"#, true, r#""test""#),
    ];

    println!("\n=== Simple Backreference Tests ===");
    for (input, should_match, expected) in test_cases {
        let result = re.find(input);
        println!("Input: {:?}", input);
        if should_match {
            let m = result.expect("Expected match");
            println!("  Match: {:?} (expected {:?})", m.as_str(), expected);
            assert_eq!(m.as_str(), expected);
        } else {
            assert!(result.is_none(), "Expected no match for {:?}", input);
            println!("  No match (expected)");
        }
    }
}

#[test]
fn test_backref_captures() {
    let pattern = r#"(['"])[^'"]*\1"#;
    let re = regexr::Regex::new(pattern).unwrap();

    println!("\n=== Backreference Captures Test ===");

    let input = r#"let x = "hello"; let y = 'world';"#;
    println!("Input: {:?}", input);

    // Test find_iter
    let matches: Vec<_> = re.find_iter(input).collect();
    println!("Matches found: {}", matches.len());
    for m in &matches {
        println!("  {:?} at {}..{}", m.as_str(), m.start(), m.end());
    }

    // Test captures
    if let Some(caps) = re.captures(input) {
        println!("Captures:");
        println!("  Full match: {:?}", caps.get(0).map(|m| m.as_str()));
        println!("  Group 1 (quote): {:?}", caps.get(1).map(|m| m.as_str()));
    }
}
