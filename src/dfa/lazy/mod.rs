//! Lazy DFA engine module.
//!
//! A DFA that builds states on-demand using subset construction.
//!
//! ## Structure
//!
//! - `shared.rs` - DfaState, CharClass, PositionContext, and helper types
//! - `interpreter/` - Pure Rust execution
//! - `engine.rs` - Engine facade
//!
//! ## Algorithm
//!
//! The lazy DFA uses subset construction to convert an NFA to a DFA on-demand.
//! Each DFA state represents a set of NFA states. Transitions are computed
//! lazily and cached for future use.
//!
//! ## Performance Optimizations
//!
//! 1. **Premultiplied State IDs**: State IDs are pre-multiplied by stride (256),
//!    so `transitions[state + byte]` is a simple addition.
//!
//! 2. **Tagged State IDs**: High bits encode status (match/dead/unknown),
//!    allowing status checks without memory dereference.
//!
//! 3. **Dense Transition Table**: A flat array for cache efficiency.
//!
//! 4. **Full Flush Cache Strategy**: When cache is full, flush all states
//!    rather than LRU (faster in practice).

mod engine;
pub mod interpreter;
pub(crate) mod shared;

// Re-exports
pub use engine::LazyDfaEngine;
pub use interpreter::LazyDfa;
pub use shared::{CharClass, DfaStateId, PositionContext};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::nfa::compile;
    use crate::parser::parse;

    fn make_dfa(pattern: &str) -> LazyDfa {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = compile(&hir).unwrap();
        LazyDfa::new(nfa)
    }

    fn make_engine(pattern: &str) -> LazyDfaEngine {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = compile(&hir).unwrap();
        LazyDfaEngine::new(nfa)
    }

    #[test]
    fn test_simple_match() {
        let mut dfa = make_dfa("abc");
        assert!(dfa.is_match_bytes(b"abc"));
        assert!(!dfa.is_match_bytes(b"ab"));
        assert!(!dfa.is_match_bytes(b"abcd"));
    }

    #[test]
    fn test_alternation() {
        let mut dfa = make_dfa("a|b");
        assert!(dfa.is_match_bytes(b"a"));
        assert!(dfa.is_match_bytes(b"b"));
        assert!(!dfa.is_match_bytes(b"c"));
    }

    #[test]
    fn test_repetition() {
        let mut dfa = make_dfa("a*");
        assert!(dfa.is_match_bytes(b""));
        assert!(dfa.is_match_bytes(b"a"));
        assert!(dfa.is_match_bytes(b"aaa"));
    }

    #[test]
    fn test_find() {
        let mut dfa = make_dfa("abc");
        assert_eq!(dfa.find(b"xyzabc123"), Some((3, 6)));
        assert_eq!(dfa.find(b"abc"), Some((0, 3)));
        assert_eq!(dfa.find(b"xyz"), None);
    }

    #[test]
    fn test_class() {
        let mut dfa = make_dfa("[a-z]+");
        assert!(dfa.is_match_bytes(b"hello"));
        assert!(!dfa.is_match_bytes(b"HELLO"));
        assert!(!dfa.is_match_bytes(b""));
    }

    #[test]
    fn test_cache_flush() {
        let mut dfa = make_dfa("a|b|c|d|e|f|g|h");
        dfa.set_cache_limit(3);

        let initial_count = dfa.state_count();
        assert!(initial_count >= 1);

        for _ in 0..10 {
            dfa.find(b"abcdefgh");
            dfa.find(b"xyzabcxyz");
        }

        let flush_count = dfa.flush_count();

        assert!(dfa.is_match_bytes(b"a"));
        assert!(dfa.is_match_bytes(b"h"));
        assert!(!dfa.is_match_bytes(b"z"));

        let _ = flush_count;
    }

    #[test]
    fn test_word_boundary_basic() {
        let mut dfa = make_dfa(r"\bthe\b");
        assert!(dfa.has_word_boundary(), "DFA should detect word boundary");

        assert_eq!(
            dfa.find(b"the cat"),
            Some((0, 3)),
            "Should match 'the' at start"
        );
        assert_eq!(
            dfa.find(b"see the cat"),
            Some((4, 7)),
            "Should match 'the' in middle"
        );

        assert_eq!(dfa.find(b"there"), None, "Should not match 'there'");
        assert_eq!(dfa.find(b"other"), None, "Should not match 'other'");
        assert_eq!(dfa.find(b"bathe"), None, "Should not match 'bathe'");
    }

    #[test]
    fn test_word_boundary_no_partial() {
        let mut dfa = make_dfa(r"\bword\b");

        assert_eq!(dfa.find(b"word"), Some((0, 4)));
        assert_eq!(dfa.find(b"a word here"), Some((2, 6)));

        assert_eq!(dfa.find(b"keyword"), None);
        assert_eq!(dfa.find(b"wording"), None);
        assert_eq!(dfa.find(b"swordfish"), None);
    }

    #[test]
    fn test_not_word_boundary() {
        let mut dfa = make_dfa(r"a\Bb");

        assert_eq!(dfa.find(b"ab"), Some((0, 2)));
        assert_eq!(dfa.find(b"cab"), Some((1, 3)));
        assert_eq!(dfa.find(b"cabin"), Some((1, 3)));

        let mut dfa2 = make_dfa(r"x\By");
        assert_eq!(dfa2.find(b"x y"), None);
        assert_eq!(dfa2.find(b"xy"), Some((0, 2)));
    }

    #[test]
    fn test_start_of_text_anchor() {
        let mut dfa = make_dfa("^hello");
        assert!(dfa.has_anchors(), "DFA should detect anchors");
        assert!(dfa.has_start_anchor(), "DFA should detect start anchor");

        assert_eq!(
            dfa.find(b"hello world"),
            Some((0, 5)),
            "Should match at start"
        );
        assert_eq!(dfa.find(b"hello"), Some((0, 5)), "Should match exact");

        assert_eq!(dfa.find(b"say hello"), None, "Should not match in middle");
        assert_eq!(dfa.find(b"  hello"), None, "Should not match after spaces");
    }

    #[test]
    fn test_end_of_text_anchor() {
        let mut dfa = make_dfa("world$");
        assert!(dfa.has_anchors(), "DFA should detect anchors");
        assert!(dfa.has_end_anchor(), "DFA should detect end anchor");

        assert_eq!(
            dfa.find(b"hello world"),
            Some((6, 11)),
            "Should match at end"
        );
        assert_eq!(dfa.find(b"world"), Some((0, 5)), "Should match exact");

        assert_eq!(dfa.find(b"world hello"), None, "Should not match at start");
        assert_eq!(dfa.find(b"world "), None, "Should not match before space");
    }

    #[test]
    fn test_both_anchors() {
        let mut dfa = make_dfa("^hello$");
        assert!(dfa.has_start_anchor() && dfa.has_end_anchor());

        assert_eq!(dfa.find(b"hello"), Some((0, 5)), "Should match exact");

        assert_eq!(
            dfa.find(b"hello world"),
            None,
            "Should not match with suffix"
        );
        assert_eq!(dfa.find(b"say hello"), None, "Should not match with prefix");
        assert_eq!(dfa.find(b" hello "), None, "Should not match with both");
    }

    #[test]
    fn test_anchor_with_pattern() {
        let mut dfa = make_dfa("^[a-z]+$");

        assert_eq!(dfa.find(b"hello"), Some((0, 5)));
        assert_eq!(dfa.find(b"world"), Some((0, 5)));
        assert_eq!(dfa.find(b"abc"), Some((0, 3)));

        assert_eq!(dfa.find(b"hello world"), None);
        assert_eq!(dfa.find(b"123abc"), None);
        assert_eq!(dfa.find(b"abc123"), None);
    }

    #[test]
    fn test_start_anchor_optimization() {
        let mut dfa = make_dfa("^test");

        assert_eq!(dfa.find(b"test here"), Some((0, 4)));
        assert_eq!(dfa.find(b"not test"), None);
    }

    #[test]
    fn test_multiline_start_anchor() {
        let mut dfa = make_dfa("(?m)^hello");
        assert!(
            dfa.has_multiline_anchors(),
            "DFA should detect multiline anchors"
        );

        assert_eq!(dfa.find(b"hello world"), Some((0, 5)));
        assert_eq!(dfa.find(b"first\nhello"), Some((6, 11)));
        assert_eq!(dfa.find(b"line1\nline2\nhello"), Some((12, 17)));

        assert_eq!(dfa.find(b"say hello"), None);
    }

    #[test]
    fn test_multiline_end_anchor() {
        let mut dfa = make_dfa("(?m)world$");
        assert!(dfa.has_multiline_anchors());

        assert_eq!(dfa.find(b"hello world"), Some((6, 11)));
        assert_eq!(dfa.find(b"world\nnext"), Some((0, 5)));

        assert_eq!(dfa.find(b"world hello"), None);
    }

    #[test]
    fn test_anchor_empty_input() {
        let mut dfa = make_dfa("^$");

        assert_eq!(dfa.find(b""), Some((0, 0)));
        assert_eq!(dfa.find(b"x"), None);
    }

    #[test]
    fn test_engine_facade() {
        let mut engine = make_engine("abc");
        assert_eq!(engine.find(b"xyzabc123"), Some((3, 6)));
        assert!(engine.is_match_bytes(b"abc"));
        assert!(!engine.is_jit());
    }

    #[test]
    fn test_engine_state_count() {
        let mut engine = make_engine("[a-z]+");
        engine.find(b"hello");
        assert!(engine.state_count() > 0);
    }
}
