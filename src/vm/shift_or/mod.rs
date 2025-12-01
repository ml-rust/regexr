//! Shift-Or (Bitap) engine module.
//!
//! A bit-parallel NFA simulation that runs entirely in CPU registers.
//! Only works for patterns with ≤64 character positions.
//!
//! ## Structure
//!
//! - `shared.rs` - ShiftOr data structure (masks, follow sets)
//! - `interpreter/` - Pure Rust execution
//! - `jit/` - JIT-compiled execution (x86-64 only, feature-gated)
//! - `engine.rs` - Engine facade that selects the best backend
//!
//! ## Algorithm
//!
//! This implementation uses Glushkov NFA (epsilon-free), NOT Thompson NFA.
//! Thompson's epsilon-transitions break the 1-shift = 1-byte invariant.
//!
//! Unlike classic Shift-Or which assumes linear position progression,
//! this implementation uses explicit follow sets from Glushkov construction
//! to handle patterns with nullable subexpressions.

mod shared;
mod engine;
pub mod interpreter;

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub mod jit;

// Re-exports
pub use shared::{ShiftOr, is_shift_or_compatible};
pub use interpreter::ShiftOrInterpreter;
pub use engine::ShiftOrEngine;

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub use jit::JitShiftOr;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn make_shift_or(pattern: &str) -> Option<ShiftOr> {
        let ast = parse(pattern).ok()?;
        let hir = translate(&ast).ok()?;
        ShiftOr::from_hir(&hir)
    }

    fn make_engine(pattern: &str) -> Option<ShiftOrEngine> {
        let ast = parse(pattern).ok()?;
        let hir = translate(&ast).ok()?;
        ShiftOrEngine::from_hir(&hir)
    }

    #[test]
    fn test_simple_literal() {
        let so = make_shift_or("abc").unwrap();
        let interp = ShiftOrInterpreter::new(&so);
        assert!(interp.is_match(b"abc"));
        assert!(interp.is_match(b"xyzabc"));
        assert!(interp.is_match(b"abcdef"));
        assert!(!interp.is_match(b"ab"));
        assert!(!interp.is_match(b"abd"));
    }

    #[test]
    fn test_find() {
        let engine = make_engine("abc").unwrap();
        assert_eq!(engine.find(b"xyzabc123"), Some((3, 6)));
        assert_eq!(engine.find(b"abc"), Some((0, 3)));
        assert_eq!(engine.find(b"xxabcxx"), Some((2, 5)));
    }

    #[test]
    fn test_single_char() {
        let engine = make_engine("a").unwrap();
        assert!(engine.is_match(b"a"));
        assert!(engine.is_match(b"bab"));
        assert!(!engine.is_match(b"bbb"));
    }

    #[test]
    fn test_alternation() {
        let engine = make_engine("a|b").unwrap();
        assert!(engine.is_match(b"a"));
        assert!(engine.is_match(b"b"));
        assert!(engine.is_match(b"xax"));
        assert!(!engine.is_match(b"xyz"));
    }

    #[test]
    fn test_character_class() {
        let engine = make_engine("[a-z]").unwrap();
        assert!(engine.is_match(b"a"));
        assert!(engine.is_match(b"z"));
        assert!(engine.is_match(b"123m456"));
        assert!(!engine.is_match(b"123"));
        assert!(!engine.is_match(b"ABC"));
    }

    #[test]
    fn test_compatibility_check() {
        let ast = parse("abc").unwrap();
        let hir = translate(&ast).unwrap();
        assert!(is_shift_or_compatible(&hir));

        // Long pattern should not be compatible
        let long_pattern = "a".repeat(100);
        let ast = parse(&long_pattern).unwrap();
        let hir = translate(&ast).unwrap();
        assert!(!is_shift_or_compatible(&hir));
    }

    #[test]
    fn test_nullable_pattern() {
        let engine = make_engine("a?").unwrap();
        // Nullable patterns match empty string
        assert!(engine.is_match(b""));
        assert!(engine.is_match(b"a"));
        assert!(engine.is_match(b"b"));
    }

    #[test]
    fn test_no_false_positives() {
        let engine = make_engine("hello").unwrap();
        assert!(!engine.is_match(b"hell"));
        assert!(!engine.is_match(b"ello"));
        assert!(!engine.is_match(b"helo"));
        assert!(engine.is_match(b"hello"));
    }

    #[test]
    fn test_dot_star() {
        let engine = make_engine("a.*b").unwrap();
        assert!(engine.is_match(b"ab"), "ab should match a.*b");
        assert!(engine.is_match(b"axb"), "axb should match a.*b");
        assert!(engine.is_match(b"axxb"), "axxb should match a.*b");
        assert!(engine.is_match(b"a123b"), "a123b should match a.*b");
        assert!(!engine.is_match(b"a"), "a should not match a.*b");
        assert!(!engine.is_match(b"b"), "b should not match a.*b");
    }

    #[test]
    fn test_hello_dot_star_world() {
        let engine = make_engine("hello.*world").unwrap();
        assert!(engine.is_match(b"helloworld"), "helloworld should match");
        assert!(engine.is_match(b"hello world"), "hello world should match");
        assert!(engine.is_match(b"hello to the world"), "hello to the world should match");
        assert!(!engine.is_match(b"hello"), "hello should not match");
        assert!(!engine.is_match(b"world"), "world should not match");
    }

    #[test]
    fn test_word_boundary_not_supported() {
        // ShiftOr does not support word boundaries
        assert!(make_shift_or(r"\bword\b").is_none());
        assert!(make_shift_or(r"\Bword").is_none());
        assert!(make_shift_or(r"word\B").is_none());
    }

    #[test]
    fn test_anchors_not_supported() {
        // ShiftOr does not support anchors
        assert!(make_shift_or(r"^word").is_none());
        assert!(make_shift_or(r"word$").is_none());
    }

    #[test]
    fn test_non_greedy_not_supported() {
        // ShiftOr does not support non-greedy quantifiers
        assert!(make_shift_or(r"a.*?b").is_none());
        assert!(make_shift_or(r"a.+?b").is_none());
        assert!(make_shift_or(r"a??b").is_none());
        assert!(make_shift_or(r"a{1,3}?b").is_none());
    }

    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    mod jit_tests {
        use super::*;

        fn make_jit(pattern: &str) -> Option<JitShiftOr> {
            let so = make_shift_or(pattern)?;
            JitShiftOr::compile(&so)
        }

        #[test]
        fn test_jit_simple_literal() {
            let jit = make_jit("abc").unwrap();
            let result = jit.find(b"xyzabc123");
            assert!(result.is_some());
            let (_, end) = result.unwrap();
            assert!(end >= 6);
        }

        #[test]
        fn test_jit_digit_pattern() {
            let jit = make_jit(r"\d+").unwrap();
            let result = jit.find(b"abc123def");
            assert!(result.is_some());
        }

        #[test]
        fn test_jit_no_match() {
            let jit = make_jit("xyz").unwrap();
            let result = jit.find(b"abcdef");
            assert!(result.is_none());
        }
    }
}
