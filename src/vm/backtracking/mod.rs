//! Backtracking engine module.
//!
//! A PCRE-style backtracking engine that supports backreferences.
//! Uses bytecode compilation for efficient execution.
//!
//! ## Structure
//!
//! - `shared.rs` - Op bytecode enum and helpers (decode_utf8, is_word_byte)
//! - `interpreter/` - Pure Rust execution
//! - `jit/` - JIT-compiled execution (x86-64 only, feature-gated)
//! - `engine.rs` - Engine facade that selects the best backend
//!
//! ## Algorithm
//!
//! This is a traditional backtracking engine with explicit choice points.
//! It supports backreferences which cannot be handled by NFA-based engines.

mod engine;
pub mod interpreter;
pub(crate) mod shared;

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub mod jit;

// Re-exports
pub use engine::BacktrackingEngine;
pub use interpreter::BacktrackingVm;

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub use jit::{compile_backtracking, BacktrackingJit};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn make_engine(pattern: &str) -> BacktrackingEngine {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        BacktrackingEngine::new(&hir)
    }

    #[test]
    fn test_simple_literal() {
        let engine = make_engine("hello");
        assert_eq!(engine.find(b"hello world"), Some((0, 5)));
        assert_eq!(engine.find(b"say hello"), Some((4, 9)));
        assert_eq!(engine.find(b"goodbye"), None);
    }

    #[test]
    fn test_alternation() {
        let engine = make_engine("a|b");
        assert_eq!(engine.find(b"a"), Some((0, 1)));
        assert_eq!(engine.find(b"b"), Some((0, 1)));
        assert_eq!(engine.find(b"c"), None);
    }

    #[test]
    fn test_backref() {
        let engine = make_engine(r"(a)\1");
        assert!(engine.is_match(b"aa"));
        assert!(!engine.is_match(b"ab"));
    }

    #[test]
    fn test_captures() {
        let engine = make_engine(r"(a)(b)(c)");
        let caps = engine.captures(b"abc").unwrap();
        assert_eq!(caps[0], Some((0, 3)));
        assert_eq!(caps[1], Some((0, 1)));
        assert_eq!(caps[2], Some((1, 2)));
        assert_eq!(caps[3], Some((2, 3)));
    }

    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    mod jit_tests {
        use super::*;
        use crate::vm::backtracking::jit::compile_backtracking;

        fn make_jit(pattern: &str) -> Option<BacktrackingJit> {
            let ast = parse(pattern).ok()?;
            let hir = translate(&ast).ok()?;
            compile_backtracking(&hir).ok()
        }

        #[test]
        fn test_jit_simple_backref() {
            let jit = make_jit(r"(a)\1").unwrap();
            assert!(jit.is_match(b"aa"));
            assert!(!jit.is_match(b"ab"));
        }

        #[test]
        fn test_jit_quoted_string() {
            let jit = make_jit(r#"(['"])[^'"]*\1"#).unwrap();
            assert!(jit.is_match(br#""hello""#));
            assert!(jit.is_match(b"'world'"));
            assert!(!jit.is_match(br#""mixed'"#));
        }
    }
}
