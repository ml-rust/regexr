//! Eager DFA engine module.
//!
//! A DFA that pre-computes all states upfront for fast matching.
//!
//! ## Structure
//!
//! - `shared.rs` - StateMetadata, tagged state constants
//! - `interpreter/` - Pure Rust execution
//! - `engine.rs` - Engine facade
//!
//! ## Algorithm
//!
//! The eager DFA materializes all reachable states from a LazyDfa using BFS.
//! This trades compilation time for matching speed - O(1) transition lookups
//! with no state computation during matching.
//!
//! ## Performance
//!
//! - Flat transition table: `transitions[state * 256 + byte]`
//! - Tagged state IDs encode match/dead status in high bits
//! - No hash map lookups during matching
//! - Ideal for patterns that will be matched many times

mod engine;
pub mod interpreter;
pub(crate) mod shared;

// Re-exports
pub use engine::EagerDfaEngine;
pub use interpreter::EagerDfa;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dfa::lazy::LazyDfa;
    use crate::hir::translate;
    use crate::nfa::compile;
    use crate::parser::parse;

    fn make_eager_dfa(pattern: &str) -> EagerDfa {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = compile(&hir).unwrap();
        let mut lazy = LazyDfa::new(nfa);
        EagerDfa::from_lazy(&mut lazy)
    }

    fn make_engine(pattern: &str) -> EagerDfaEngine {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = compile(&hir).unwrap();
        EagerDfaEngine::new(nfa)
    }

    #[test]
    fn test_simple_literal() {
        let dfa = make_eager_dfa("abc");
        assert_eq!(dfa.find(b"xyzabc123"), Some((3, 6)));
        assert_eq!(dfa.find(b"abc"), Some((0, 3)));
        assert_eq!(dfa.find(b"xyz"), None);
    }

    #[test]
    fn test_alternation() {
        let dfa = make_eager_dfa("a|b");
        assert_eq!(dfa.find(b"xa"), Some((1, 2)));
        assert_eq!(dfa.find(b"xb"), Some((1, 2)));
        assert_eq!(dfa.find(b"c"), None);
    }

    #[test]
    fn test_repetition() {
        let dfa = make_eager_dfa("a+");
        assert_eq!(dfa.find(b"aaa"), Some((0, 3)));
        assert_eq!(dfa.find(b"baab"), Some((1, 3)));
        assert_eq!(dfa.find(b"bbb"), None);
    }

    #[test]
    fn test_anchored_pattern() {
        let dfa = make_eager_dfa("^abc$");
        assert_eq!(dfa.find(b"abc"), Some((0, 3)));
        assert_eq!(dfa.find(b"abcd"), None);
        assert_eq!(dfa.find(b"xabc"), None);
    }

    #[test]
    fn test_start_anchor() {
        let dfa = make_eager_dfa("^hello");
        assert!(dfa.has_start_anchor());
        assert_eq!(dfa.find(b"hello world"), Some((0, 5)));
        assert_eq!(dfa.find(b"say hello"), None);
    }

    #[test]
    fn test_end_anchor() {
        let dfa = make_eager_dfa("world$");
        assert!(dfa.has_end_anchor());
        assert_eq!(dfa.find(b"hello world"), Some((6, 11)));
        assert_eq!(dfa.find(b"world hello"), None);
    }

    #[test]
    fn test_word_boundary() {
        let dfa = make_eager_dfa(r"\bword\b");
        assert!(dfa.has_word_boundary());
        assert_eq!(dfa.find(b"a word here"), Some((2, 6)));
        assert_eq!(dfa.find(b"keyword"), None);
    }

    #[test]
    fn test_multiline_start() {
        let dfa = make_eager_dfa("(?m)^hello");
        assert!(dfa.has_multiline_anchors());
        assert_eq!(dfa.find(b"hello"), Some((0, 5)));
        assert_eq!(dfa.find(b"first\nhello"), Some((6, 11)));
    }

    #[test]
    fn test_multiline_end() {
        let dfa = make_eager_dfa("(?m)world$");
        assert!(dfa.has_multiline_anchors());
        assert_eq!(dfa.find(b"world\nnext"), Some((0, 5)));
        assert_eq!(dfa.find(b"hello world"), Some((6, 11)));
    }

    #[test]
    fn test_email_pattern() {
        let dfa = make_eager_dfa(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$");
        assert_eq!(dfa.find(b"user@example.com"), Some((0, 16)));
        assert_eq!(dfa.find(b"invalid-email"), None);
    }

    #[test]
    fn test_engine_facade() {
        let engine = make_engine("abc");
        assert_eq!(engine.find(b"xyzabc123"), Some((3, 6)));
        assert!(!engine.is_jit());
    }

    #[test]
    fn test_engine_state_count() {
        let engine = make_engine("[a-z]+");
        assert!(engine.state_count() > 0);
    }

    #[test]
    fn test_engine_from_lazy() {
        let ast = parse("test").unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = compile(&hir).unwrap();
        let mut lazy = LazyDfa::new(nfa);
        let engine = EagerDfaEngine::from_lazy(&mut lazy);
        assert_eq!(engine.find(b"test"), Some((0, 4)));
    }
}
