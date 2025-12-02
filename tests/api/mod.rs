//! Public API integration tests.
//!
//! Tests for the core `Regex` type and its methods: `new`, `is_match`, `find`,
//! `find_iter`, `captures`, `captures_iter`, `replace`, `replace_all`.
//!
//! When the `jit` feature is enabled, these tests use JIT compilation.

use regexr::Regex;
#[cfg(feature = "jit")]
use regexr::RegexBuilder;

/// Creates a Regex with JIT enabled when the `jit` feature is available.
#[allow(dead_code)]
fn regex(pattern: &str) -> Regex {
    #[cfg(feature = "jit")]
    {
        RegexBuilder::new(pattern)
            .jit(true)
            .build()
            .expect("failed to compile pattern")
    }
    #[cfg(not(feature = "jit"))]
    {
        Regex::new(pattern).expect("failed to compile pattern")
    }
}

// =============================================================================
// Regex::new and as_str
// =============================================================================

#[test]
fn test_regex_new() {
    let re = Regex::new("hello").unwrap();
    assert_eq!(re.as_str(), "hello");
}

// =============================================================================
// is_match
// =============================================================================

#[test]
fn test_is_match() {
    let re = regex("hello");
    assert!(re.is_match("hello world"));
    assert!(re.is_match("say hello"));
    assert!(!re.is_match("goodbye"));
}

// =============================================================================
// find
// =============================================================================

#[test]
fn test_find() {
    let re = regex("world");
    let m = re.find("hello world").unwrap();
    assert_eq!(m.start(), 6);
    assert_eq!(m.end(), 11);
    assert_eq!(m.as_str(), "world");
    assert_eq!(m.len(), 5);
    assert!(!m.is_empty());
}

#[test]
fn test_find_none() {
    let re = regex("xyz");
    assert!(re.find("hello world").is_none());
}

#[test]
fn test_match_range() {
    let re = regex("test");
    let m = re.find("this is a test").unwrap();
    assert_eq!(m.range(), 10..14);
}

#[test]
fn test_empty_match() {
    let re = regex("a*");
    let m = re.find("bbb").unwrap();
    assert!(m.is_empty());
    assert_eq!(m.len(), 0);
}

// =============================================================================
// find_iter
// =============================================================================

#[test]
fn test_find_iter() {
    let re = regex("a");
    let matches: Vec<_> = re.find_iter("abracadabra").collect();
    assert_eq!(matches.len(), 5);
    assert_eq!(matches[0].start(), 0);
    assert_eq!(matches[1].start(), 3);
    assert_eq!(matches[2].start(), 5);
    assert_eq!(matches[3].start(), 7);
    assert_eq!(matches[4].start(), 10);
}

#[test]
fn test_find_iter_empty() {
    let re = regex("xyz");
    let matches: Vec<_> = re.find_iter("hello world").collect();
    assert!(matches.is_empty());
}

// =============================================================================
// captures
// =============================================================================

#[test]
fn test_captures() {
    let re = regex("(\\d+)-(\\d+)");
    let caps = re.captures("phone: 123-456").unwrap();
    assert!(caps.len() >= 3);
    assert_eq!(&caps[0], "123-456");
    assert_eq!(&caps[1], "123");
    assert_eq!(&caps[2], "456");
}

// =============================================================================
// captures_iter
// =============================================================================

#[test]
fn test_captures_iter_basic() {
    let re = regex(r"(\w+)");
    let text = "hello world foo";
    let caps: Vec<_> = re.captures_iter(text).collect();
    assert_eq!(caps.len(), 3);
    assert_eq!(&caps[0][0], "hello");
    assert_eq!(&caps[1][0], "world");
    assert_eq!(&caps[2][0], "foo");
}

#[test]
fn test_captures_iter_with_groups() {
    let re = regex(r"(\w+)=(\d+)");
    let text = "a=1 b=2 c=3";
    let caps: Vec<_> = re.captures_iter(text).collect();
    assert_eq!(caps.len(), 3);
    assert_eq!(&caps[0][1], "a");
    assert_eq!(&caps[0][2], "1");
    assert_eq!(&caps[1][1], "b");
    assert_eq!(&caps[1][2], "2");
    assert_eq!(&caps[2][1], "c");
    assert_eq!(&caps[2][2], "3");
}

#[test]
fn test_captures_iter_named() {
    let re = regex(r"(?<key>\w+)=(?<value>\d+)");
    let text = "x=10 y=20";
    let caps: Vec<_> = re.captures_iter(text).collect();
    assert_eq!(caps.len(), 2);
    assert_eq!(&caps[0]["key"], "x");
    assert_eq!(&caps[0]["value"], "10");
    assert_eq!(&caps[1]["key"], "y");
    assert_eq!(&caps[1]["value"], "20");
}

#[test]
fn test_captures_iter_positions() {
    let re = regex(r"(\d+)");
    let text = "a1b22c333";
    let caps: Vec<_> = re.captures_iter(text).collect();
    assert_eq!(caps.len(), 3);
    assert_eq!(caps[0].get(0).unwrap().start(), 1);
    assert_eq!(caps[0].get(0).unwrap().end(), 2);
    assert_eq!(caps[1].get(0).unwrap().start(), 3);
    assert_eq!(caps[1].get(0).unwrap().end(), 5);
    assert_eq!(caps[2].get(0).unwrap().start(), 6);
    assert_eq!(caps[2].get(0).unwrap().end(), 9);
}

// =============================================================================
// replace
// =============================================================================

#[test]
fn test_replace() {
    let re = regex("world");
    let result = re.replace("hello world", "rust");
    assert_eq!(result, "hello rust");
}

#[test]
fn test_replace_no_match() {
    let re = regex("xyz");
    let result = re.replace("hello world", "rust");
    assert_eq!(result, "hello world");
}

// =============================================================================
// replace_all
// =============================================================================

#[test]
fn test_replace_all() {
    let re = regex("o");
    let result = re.replace_all("hello world", "0");
    assert_eq!(result, "hell0 w0rld");
}

#[test]
fn test_replace_all_no_match() {
    let re = regex("xyz");
    let result = re.replace_all("hello world", "!");
    assert_eq!(result, "hello world");
}

// =============================================================================
// Prefix Optimization
// =============================================================================

#[cfg(feature = "jit")]
mod prefix_opt {
    use regexr::RegexBuilder;

    #[test]
    fn test_prefix_optimized_basic() {
        // Pattern with many tokens sharing common prefixes
        let re = RegexBuilder::new(r"the|that|them|they|this")
            .optimize_prefixes(true)
            .build()
            .unwrap();

        assert!(re.is_match("the"));
        assert!(re.is_match("that"));
        assert!(re.is_match("them"));
        assert!(re.is_match("they"));
        assert!(re.is_match("this"));
        assert!(!re.is_match("those"));
    }

    #[test]
    fn test_prefix_optimized_find() {
        let re = RegexBuilder::new(r"apple|application|apply|apt")
            .optimize_prefixes(true)
            .build()
            .unwrap();

        let m = re.find("the application was running").unwrap();
        assert_eq!(m.as_str(), "application");
    }

    #[test]
    fn test_prefix_optimized_multiple_branches() {
        // Words that share some prefixes but not all
        let re = RegexBuilder::new(r"test|testing|tested|tester|apple|application")
            .optimize_prefixes(true)
            .build()
            .unwrap();

        assert!(re.is_match("test"));
        assert!(re.is_match("testing"));
        assert!(re.is_match("tested"));
        assert!(re.is_match("tester"));
        assert!(re.is_match("apple"));
        assert!(re.is_match("application"));
        // "tests" matches because it contains "test"
        assert!(re.is_match("tests"));
        // But "xyz" doesn't match
        assert!(!re.is_match("xyz"));
    }

    #[test]
    fn test_prefix_optimized_with_jit() {
        // Combine prefix optimization with JIT
        let re = RegexBuilder::new(r"the|that|them|they|this")
            .optimize_prefixes(true)
            .jit(true)
            .build()
            .unwrap();

        assert!(re.is_match("the"));
        assert!(re.is_match("that"));
        assert!(re.is_match("them"));
        assert!(re.is_match("they"));
        assert!(re.is_match("this"));
        assert!(!re.is_match("those"));
    }

    #[test]
    fn test_prefix_optimized_find_iter() {
        let re = RegexBuilder::new(r"the|that|them|they")
            .optimize_prefixes(true)
            .build()
            .unwrap();

        let text = "the cat that sat on them made they jump";
        let matches: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();
        assert_eq!(matches, vec!["the", "that", "them", "they"]);
    }
}

// =============================================================================
// JIT Alternation Tests
// =============================================================================

#[cfg(feature = "jit")]
mod jit_alternation {
    use regexr::RegexBuilder;

    #[test]
    fn test_jit_simple_alternation() {
        let re = RegexBuilder::new(r"foo|bar").jit(true).build().unwrap();

        assert!(re.is_match("foo"));
        assert!(re.is_match("bar"));
        assert!(!re.is_match("baz"));
    }

    #[test]
    fn test_jit_alternation_find() {
        let re = RegexBuilder::new(r"foo|bar").jit(true).build().unwrap();

        let m = re.find("xyzfoo123").unwrap();
        assert_eq!(m.start(), 3);
        assert_eq!(m.end(), 6);
        assert_eq!(m.as_str(), "foo");

        let m = re.find("xyzbar123").unwrap();
        assert_eq!(m.start(), 3);
        assert_eq!(m.end(), 6);
        assert_eq!(m.as_str(), "bar");
    }

    #[test]
    fn test_jit_alternation_multi() {
        let re = RegexBuilder::new(r"hello|world|test")
            .jit(true)
            .build()
            .unwrap();

        assert!(re.is_match("hello"));
        assert!(re.is_match("world"));
        assert!(re.is_match("test"));
        assert!(!re.is_match("other"));

        let m = re.find("say hello there").unwrap();
        assert_eq!(m.as_str(), "hello");
    }

    #[test]
    fn test_jit_alternation_with_char_class() {
        let re = RegexBuilder::new(r"[a-z]+|[0-9]+")
            .jit(true)
            .build()
            .unwrap();

        assert!(re.is_match("abc"));
        assert!(re.is_match("123"));

        let m = re.find("...abc...").unwrap();
        assert_eq!(m.as_str(), "abc");

        let m = re.find("...123...").unwrap();
        assert_eq!(m.as_str(), "123");
    }

    #[test]
    fn test_jit_alternation_find_iter() {
        let re = RegexBuilder::new(r"foo|bar").jit(true).build().unwrap();

        let text = "foo bar foo bar baz";
        let matches: Vec<_> = re.find_iter(text).map(|m| m.as_str()).collect();
        assert_eq!(matches, vec!["foo", "bar", "foo", "bar"]);
    }

    #[test]
    fn test_jit_alternation_different_lengths() {
        let re = RegexBuilder::new(r"a|bb|ccc").jit(true).build().unwrap();

        assert!(re.is_match("a"));
        assert!(re.is_match("bb"));
        assert!(re.is_match("ccc"));

        let m = re.find("xxaxx").unwrap();
        assert_eq!(m.as_str(), "a");

        let m = re.find("xxbbxx").unwrap();
        assert_eq!(m.as_str(), "bb");

        let m = re.find("xxcccxx").unwrap();
        assert_eq!(m.as_str(), "ccc");
    }

    #[test]
    fn test_jit_captures_basic() {
        // Test that JIT captures work correctly
        let re = RegexBuilder::new(r"([a-z]+)=([0-9]+)")
            .jit(true)
            .build()
            .unwrap();

        let caps = re.captures("key=123").unwrap();
        assert_eq!(&caps[0], "key=123");
        assert_eq!(&caps[1], "key");
        assert_eq!(&caps[2], "123");
    }

    #[test]
    fn test_jit_captures_find_in_text() {
        let re = RegexBuilder::new(r"([a-z]+):([0-9]+)")
            .jit(true)
            .build()
            .unwrap();

        let caps = re.captures("data is foo:42 and bar:99").unwrap();
        assert_eq!(&caps[0], "foo:42");
        assert_eq!(&caps[1], "foo");
        assert_eq!(&caps[2], "42");
    }

    #[test]
    fn test_jit_captures_iter() {
        let re = RegexBuilder::new(r"([a-z]+)=([0-9]+)")
            .jit(true)
            .build()
            .unwrap();

        let text = "a=1 b=2 c=3";
        let caps: Vec<_> = re.captures_iter(text).collect();
        assert_eq!(caps.len(), 3);
        assert_eq!(&caps[0][1], "a");
        assert_eq!(&caps[0][2], "1");
        assert_eq!(&caps[1][1], "b");
        assert_eq!(&caps[1][2], "2");
        assert_eq!(&caps[2][1], "c");
        assert_eq!(&caps[2][2], "3");
    }
}
