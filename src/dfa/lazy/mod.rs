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
pub(crate) use shared::scan_budget;
pub use shared::{CacheCeilingExceeded, CharClass, DfaStateId, PositionContext};

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

    /// What the executable spec (`crate::reference`) says the match is.
    fn reference_find(pattern: &str, input: &[u8]) -> Option<(usize, usize)> {
        let hir = parse(pattern).and_then(|ast| translate(&ast)).unwrap();
        crate::reference::find(&hir.expr, hir.props.capture_count as usize, input)
    }

    #[test]
    fn test_simple_match() {
        let mut dfa = make_dfa("abc");
        assert_eq!(dfa.is_match_bytes(b"abc"), Ok(true));
        assert_eq!(dfa.is_match_bytes(b"ab"), Ok(false));
        assert_eq!(dfa.is_match_bytes(b"abcd"), Ok(false));
    }

    #[test]
    fn test_alternation() {
        let mut dfa = make_dfa("a|b");
        assert_eq!(dfa.is_match_bytes(b"a"), Ok(true));
        assert_eq!(dfa.is_match_bytes(b"b"), Ok(true));
        assert_eq!(dfa.is_match_bytes(b"c"), Ok(false));
    }

    #[test]
    fn test_repetition() {
        let mut dfa = make_dfa("a*");
        assert_eq!(dfa.is_match_bytes(b""), Ok(true));
        assert_eq!(dfa.is_match_bytes(b"a"), Ok(true));
        assert_eq!(dfa.is_match_bytes(b"aaa"), Ok(true));
    }

    #[test]
    fn test_find() {
        let mut dfa = make_dfa("abc");
        assert_eq!(dfa.find(b"xyzabc123"), Ok(Some((3, 6))));
        assert_eq!(dfa.find(b"abc"), Ok(Some((0, 3))));
        assert_eq!(dfa.find(b"xyz"), Ok(None));
    }

    #[test]
    fn test_class() {
        let mut dfa = make_dfa("[a-z]+");
        assert_eq!(dfa.is_match_bytes(b"hello"), Ok(true));
        assert_eq!(dfa.is_match_bytes(b"HELLO"), Ok(false));
        assert_eq!(dfa.is_match_bytes(b""), Ok(false));
    }

    #[test]
    fn test_cache_flush() {
        let mut dfa = make_dfa("a|b|c|d|e|f|g|h");
        dfa.set_cache_limit(3);

        let initial_count = dfa.state_count();
        assert!(initial_count >= 1);

        for _ in 0..10 {
            let _ = dfa.find(b"abcdefgh");
            let _ = dfa.find(b"xyzabcxyz");
        }

        let flush_count = dfa.flush_count();

        assert_eq!(dfa.is_match_bytes(b"a"), Ok(true));
        assert_eq!(dfa.is_match_bytes(b"h"), Ok(true));
        assert_eq!(dfa.is_match_bytes(b"z"), Ok(false));

        let _ = flush_count;
    }

    /// A cache flush renumbers every DFA state, so one firing mid-search used to
    /// leave the scanning loop holding a stale premultiplied ID: the next
    /// transition fell out of the table, read as dead, and the search returned
    /// the truncated `last_match` — `None` here — instead of the real match.
    ///
    /// `abcd` creates one state per literal byte, so a 3-state limit fires the
    /// flush while the loop is on the third one.
    #[test]
    fn flush_mid_search_does_not_truncate_the_match() {
        let mut dfa = make_dfa("abcd");
        dfa.set_cache_limit(3);

        assert_eq!(dfa.find_from(b"abcd", 0), Ok(Some((0, 4))));
        assert_eq!(reference_find("abcd", b"abcd"), Some((0, 4)));
    }

    /// Deferring flushes lets the cache grow, and that growth is capped. Past the
    /// cap the search cannot be finished, and the incomplete table would answer
    /// "no match" — a false negative. It has to say so instead.
    #[test]
    fn cache_ceiling_gives_up_rather_than_answering_wrongly() {
        let mut dfa = make_dfa("abcdef");
        dfa.set_cache_limit(1);

        assert_eq!(dfa.find(b"abcdef"), Err(CacheCeilingExceeded));
        // The give-up is not "no match": the pattern does match here.
        assert_eq!(reference_find("abcdef", b"abcdef"), Some((0, 6)));
    }

    /// The flush a search defers must happen when the search ends, or the growth
    /// accumulates across searches and the cache limit stops meaning anything.
    #[test]
    fn deferred_flush_runs_once_the_search_ends() {
        let mut dfa = make_dfa("abcd");
        dfa.set_cache_limit(3);

        for _ in 0..5 {
            assert_eq!(dfa.find(b"abcd"), Ok(Some((0, 4))));
            assert_eq!(
                dfa.state_count(),
                1,
                "the deferred flush should have reset the cache to the start state"
            );
        }

        assert!(
            dfa.flush_count() >= 5,
            "each search should have flushed once"
        );
    }

    #[test]
    fn test_word_boundary_basic() {
        let mut dfa = make_dfa(r"\bthe\b");
        assert!(dfa.has_word_boundary(), "DFA should detect word boundary");

        assert_eq!(
            dfa.find(b"the cat"),
            Ok(Some((0, 3))),
            "Should match 'the' at start"
        );
        assert_eq!(
            dfa.find(b"see the cat"),
            Ok(Some((4, 7))),
            "Should match 'the' in middle"
        );

        assert_eq!(dfa.find(b"there"), Ok(None), "Should not match 'there'");
        assert_eq!(dfa.find(b"other"), Ok(None), "Should not match 'other'");
        assert_eq!(dfa.find(b"bathe"), Ok(None), "Should not match 'bathe'");
    }

    #[test]
    fn test_word_boundary_no_partial() {
        let mut dfa = make_dfa(r"\bword\b");

        assert_eq!(dfa.find(b"word"), Ok(Some((0, 4))));
        assert_eq!(dfa.find(b"a word here"), Ok(Some((2, 6))));

        assert_eq!(dfa.find(b"keyword"), Ok(None));
        assert_eq!(dfa.find(b"wording"), Ok(None));
        assert_eq!(dfa.find(b"swordfish"), Ok(None));
    }

    #[test]
    fn test_not_word_boundary() {
        let mut dfa = make_dfa(r"a\Bb");

        assert_eq!(dfa.find(b"ab"), Ok(Some((0, 2))));
        assert_eq!(dfa.find(b"cab"), Ok(Some((1, 3))));
        assert_eq!(dfa.find(b"cabin"), Ok(Some((1, 3))));

        let mut dfa2 = make_dfa(r"x\By");
        assert_eq!(dfa2.find(b"x y"), Ok(None));
        assert_eq!(dfa2.find(b"xy"), Ok(Some((0, 2))));
    }

    #[test]
    fn test_start_of_text_anchor() {
        let mut dfa = make_dfa("^hello");
        assert!(dfa.has_anchors(), "DFA should detect anchors");
        assert!(dfa.has_start_anchor(), "DFA should detect start anchor");

        assert_eq!(
            dfa.find(b"hello world"),
            Ok(Some((0, 5))),
            "Should match at start"
        );
        assert_eq!(dfa.find(b"hello"), Ok(Some((0, 5))), "Should match exact");

        assert_eq!(
            dfa.find(b"say hello"),
            Ok(None),
            "Should not match in middle"
        );
        assert_eq!(
            dfa.find(b"  hello"),
            Ok(None),
            "Should not match after spaces"
        );
    }

    #[test]
    fn test_end_of_text_anchor() {
        let mut dfa = make_dfa("world$");
        assert!(dfa.has_anchors(), "DFA should detect anchors");
        assert!(dfa.has_end_anchor(), "DFA should detect end anchor");

        assert_eq!(
            dfa.find(b"hello world"),
            Ok(Some((6, 11))),
            "Should match at end"
        );
        assert_eq!(dfa.find(b"world"), Ok(Some((0, 5))), "Should match exact");

        assert_eq!(
            dfa.find(b"world hello"),
            Ok(None),
            "Should not match at start"
        );
        assert_eq!(
            dfa.find(b"world "),
            Ok(None),
            "Should not match before space"
        );
    }

    #[test]
    fn test_both_anchors() {
        let mut dfa = make_dfa("^hello$");
        assert!(dfa.has_start_anchor() && dfa.has_end_anchor());

        assert_eq!(dfa.find(b"hello"), Ok(Some((0, 5))), "Should match exact");

        assert_eq!(
            dfa.find(b"hello world"),
            Ok(None),
            "Should not match with suffix"
        );
        assert_eq!(
            dfa.find(b"say hello"),
            Ok(None),
            "Should not match with prefix"
        );
        assert_eq!(dfa.find(b" hello "), Ok(None), "Should not match with both");
    }

    #[test]
    fn test_anchor_with_pattern() {
        let mut dfa = make_dfa("^[a-z]+$");

        assert_eq!(dfa.find(b"hello"), Ok(Some((0, 5))));
        assert_eq!(dfa.find(b"world"), Ok(Some((0, 5))));
        assert_eq!(dfa.find(b"abc"), Ok(Some((0, 3))));

        assert_eq!(dfa.find(b"hello world"), Ok(None));
        assert_eq!(dfa.find(b"123abc"), Ok(None));
        assert_eq!(dfa.find(b"abc123"), Ok(None));
    }

    #[test]
    fn test_start_anchor_optimization() {
        let mut dfa = make_dfa("^test");

        assert_eq!(dfa.find(b"test here"), Ok(Some((0, 4))));
        assert_eq!(dfa.find(b"not test"), Ok(None));
    }

    #[test]
    fn test_multiline_start_anchor() {
        let mut dfa = make_dfa("(?m)^hello");
        assert!(
            dfa.has_multiline_anchors(),
            "DFA should detect multiline anchors"
        );

        assert_eq!(dfa.find(b"hello world"), Ok(Some((0, 5))));
        assert_eq!(dfa.find(b"first\nhello"), Ok(Some((6, 11))));
        assert_eq!(dfa.find(b"line1\nline2\nhello"), Ok(Some((12, 17))));

        assert_eq!(dfa.find(b"say hello"), Ok(None));
    }

    #[test]
    fn test_multiline_end_anchor() {
        let mut dfa = make_dfa("(?m)world$");
        assert!(dfa.has_multiline_anchors());

        assert_eq!(dfa.find(b"hello world"), Ok(Some((6, 11))));
        assert_eq!(dfa.find(b"world\nnext"), Ok(Some((0, 5))));

        assert_eq!(dfa.find(b"world hello"), Ok(None));
    }

    #[test]
    fn test_anchor_empty_input() {
        let mut dfa = make_dfa("^$");

        assert_eq!(dfa.find(b""), Ok(Some((0, 0))));
        assert_eq!(dfa.find(b"x"), Ok(None));
    }

    #[test]
    fn test_engine_facade() {
        let mut engine = make_engine("abc");
        assert_eq!(engine.find(b"xyzabc123"), Ok(Some((3, 6))));
        assert_eq!(engine.is_match_bytes(b"abc"), Ok(true));
        assert!(!engine.is_jit());
    }

    #[test]
    fn test_engine_state_count() {
        let mut engine = make_engine("[a-z]+");
        let _ = engine.find(b"hello");
        assert!(engine.state_count() > 0);
    }
}
