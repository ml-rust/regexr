//! PikeVM engine module.
//!
//! A thread-based NFA simulator that supports capture groups,
//! backreferences, and lookarounds.
//!
//! ## Structure
//!
//! - `shared.rs` - Thread, PikeVmContext, and helper types
//! - `interpreter/` - Pure Rust execution
//! - `engine.rs` - Engine facade
//!
//! ## Algorithm
//!
//! PikeVM (named after Rob Pike) simulates NFA execution using a set of
//! "threads" that process the input in lockstep. Each thread tracks its
//! own capture group values and position in the NFA.
//!
//! Non-greedy quantifiers are supported through thread priority:
//! - Threads have a priority that increases when taking "exit" paths
//! - For non-greedy quantifiers, the exit path has higher priority
//! - The first match from the highest-priority thread wins
//!
//! ## Optimizations
//!
//! - Sparse set deduplication: O(1) state deduplication using generation counters
//! - BinaryHeap scheduling: Efficient backref handling with min-heap
//! - Arc<Nfa> for lookarounds: Avoids expensive NFA cloning

pub(crate) mod shared;
mod engine;
pub mod interpreter;

// Re-exports
pub use engine::PikeVmEngine;
pub use interpreter::PikeVm;
pub use shared::PikeVmContext;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::nfa::compile;
    use crate::parser::parse;

    fn make_vm(pattern: &str) -> PikeVm {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = compile(&hir).unwrap();
        PikeVm::new(nfa)
    }

    fn make_engine(pattern: &str) -> PikeVmEngine {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = compile(&hir).unwrap();
        PikeVmEngine::new(nfa)
    }

    #[test]
    fn test_simple_match() {
        let vm = make_vm("abc");
        assert!(vm.is_match(b"abc"));
        assert!(vm.is_match(b"xyzabc123"));
        assert!(!vm.is_match(b"ab"));
    }

    #[test]
    fn test_find() {
        let vm = make_vm("abc");
        assert_eq!(vm.find(b"xyzabc123"), Some((3, 6)));
    }

    #[test]
    fn test_alternation() {
        let vm = make_vm("cat|dog");
        assert!(vm.is_match(b"cat"));
        assert!(vm.is_match(b"dog"));
        assert!(!vm.is_match(b"bird"));
    }

    #[test]
    fn test_repetition() {
        let vm = make_vm("a+");
        assert!(vm.is_match(b"a"));
        assert!(vm.is_match(b"aaa"));
        assert!(!vm.is_match(b""));
    }

    #[test]
    fn test_anchors() {
        let vm = make_vm("^abc$");
        assert!(vm.is_match(b"abc"));
        assert!(!vm.is_match(b"xabc"));
        assert!(!vm.is_match(b"abcx"));
    }

    #[test]
    fn test_word_boundary() {
        let vm = make_vm(r"\bword\b");
        assert!(vm.is_match(b"a word here"));
        assert!(!vm.is_match(b"awordhere"));
    }

    #[test]
    fn test_captures() {
        let vm = make_vm("(a+)(b+)");
        let caps = vm.captures(b"aaabbb").unwrap();
        assert_eq!(caps[0], Some((0, 6))); // Full match
        assert_eq!(caps[1], Some((0, 3))); // First group
        assert_eq!(caps[2], Some((3, 6))); // Second group
    }

    #[test]
    fn test_backref() {
        let vm = make_vm(r"(\w+) \1");
        assert!(vm.is_match(b"hello hello"));
        assert!(!vm.is_match(b"hello world"));
    }

    #[test]
    fn test_non_greedy() {
        let vm = make_vm("a+?");
        let m = vm.find(b"aaa");
        assert_eq!(m, Some((0, 1))); // Non-greedy should match shortest
    }

    #[test]
    fn test_context_reuse() {
        let vm = make_vm("(a+)");
        let mut ctx = vm.create_context();

        // First use
        let caps1 = vm.captures_from_start_with_context(b"aaa", &mut ctx);
        assert_eq!(caps1.as_ref().unwrap()[0], Some((0, 3)));

        // Reuse context
        let caps2 = vm.captures_from_start_with_context(b"aa", &mut ctx);
        assert_eq!(caps2.as_ref().unwrap()[0], Some((0, 2)));
    }

    #[test]
    fn test_engine_facade() {
        let engine = make_engine("abc");
        assert!(engine.is_match(b"abc"));
        assert_eq!(engine.find(b"xyzabc123"), Some((3, 6)));
        assert!(!engine.is_jit());
    }

    #[test]
    fn test_engine_captures() {
        let engine = make_engine("(a+)(b+)");
        let caps = engine.captures(b"aaabbb").unwrap();
        assert_eq!(caps[0], Some((0, 6)));
        assert_eq!(caps[1], Some((0, 3)));
        assert_eq!(caps[2], Some((3, 6)));
    }
}
