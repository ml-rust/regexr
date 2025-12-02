//! DFA to machine code compilation.
//!
//! This module compiles a LazyDFA to native x86-64 machine code using dynasm.
//! The compiled code is W^X compliant and optimized for performance.
//!
//! ## Word Boundary Support
//!
//! For patterns with word boundaries (`\b`, `\B`), the DFA uses character-class
//! augmented states. This means:
//! - States are keyed by (NFA state set, prev_char_class)
//! - We need two start states: one for NonWord prev_class, one for Word prev_class
//! - The find() method must select the correct start state based on the character
//!   before the start position

use crate::dfa::{CharClass, DfaStateId, LazyDfa};
use crate::error::Result;
use dynasmrt::{AssemblyOffset, ExecutableBuffer};

/// A JIT-compiled regex matcher.
///
/// This struct holds the executable machine code generated from a DFA.
/// The code is W^X compliant (never RWX) and uses 16-byte alignment for
/// optimal CPU instruction fetch performance.
pub struct CompiledRegex {
    /// The executable buffer containing the compiled machine code.
    /// This buffer is RX (read-execute) only - never RWX for security.
    code: ExecutableBuffer,
    /// Entry point offset into the executable buffer (for NonWord prev_class).
    entry_point: AssemblyOffset,
    /// Entry point for Word prev_class (only used when has_word_boundary is true).
    entry_point_word: Option<AssemblyOffset>,
    /// Whether this regex has word boundary assertions.
    pub(crate) has_word_boundary: bool,
    /// Whether any match state requires a word boundary (\b) at the end.
    match_needs_word_boundary: bool,
    /// Whether any match state requires NOT a word boundary (\B) at the end.
    match_needs_not_word_boundary: bool,
    /// Whether this regex has anchor assertions (^, $).
    pub(crate) has_anchors: bool,
    /// Whether this regex has a start anchor (^).
    pub(crate) has_start_anchor: bool,
    /// Whether this regex has an end anchor ($).
    /// Note: Currently used only in tests, but kept for API consistency.
    #[allow(dead_code)]
    pub(crate) has_end_anchor: bool,
    /// Whether this regex uses multiline mode for anchors.
    pub(crate) has_multiline_anchors: bool,
    /// Whether any match state requires EndOfText assertion.
    pub(crate) match_needs_end_of_text: bool,
    /// Whether any match state requires EndOfLine assertion.
    pub(crate) match_needs_end_of_line: bool,
}

impl CompiledRegex {
    /// Executes the compiled regex on the given input with a specific prev_class.
    ///
    /// Returns the end position of the match if found, or None.
    /// Executes the JIT-compiled regex and returns (start, end) positions.
    ///
    /// For unanchored patterns, the JIT code scans through the input internally,
    /// so a single call finds the first match anywhere in the input.
    ///
    /// # Arguments
    /// * `input` - The input bytes to match against
    /// * `prev_class` - The character class of the byte before input[0], or NonWord for start
    ///
    /// # Safety
    /// This method calls JIT-compiled machine code. The code is generated
    /// to be safe, but it's marked unsafe because it executes dynamically
    /// generated code.
    fn execute_with_class(&self, input: &[u8], prev_class: CharClass) -> Option<(usize, usize)> {
        // Function signature: fn(input_ptr: *const u8, len: usize) -> i64
        // Returns: packed (start << 32 | end) or -1 for no match
        type MatchFn = unsafe extern "C" fn(*const u8, usize) -> i64;

        // Select the correct entry point based on prev_class
        let entry = if self.has_word_boundary && prev_class == CharClass::Word {
            self.entry_point_word.unwrap_or(self.entry_point)
        } else {
            self.entry_point
        };

        let func: MatchFn = unsafe { std::mem::transmute(self.code.ptr(entry)) };

        let result = unsafe { func(input.as_ptr(), input.len()) };

        if result >= 0 {
            // Unpack the result: start in upper 32 bits, end in lower 32 bits
            let packed = result as u64;
            let start_pos = (packed >> 32) as usize;
            let end_pos = (packed & 0xFFFF_FFFF) as usize;

            // Validate end assertions (word boundaries and anchors)
            if !self.validate_end_assertions(input, start_pos, end_pos, prev_class) {
                return None;
            }

            Some((start_pos, end_pos))
        } else {
            None
        }
    }

    /// Validates that end assertions (word boundaries and anchors) are satisfied.
    fn validate_end_assertions(
        &self,
        input: &[u8],
        start_pos: usize,
        end_pos: usize,
        prev_class: CharClass,
    ) -> bool {
        // Validate word boundary assertions
        if self.has_word_boundary
            && (self.match_needs_word_boundary || self.match_needs_not_word_boundary)
        {
            // Compute whether we're at a word boundary at end_pos
            // For unanchored search, prev_class is relative to the original input,
            // but we need to consider the actual char before start_pos
            let actual_prev_class = if start_pos > 0 {
                CharClass::from_byte(input[start_pos - 1])
            } else {
                prev_class
            };

            let is_at_boundary = if end_pos == start_pos {
                // Empty match - check boundary at start
                if end_pos < input.len() {
                    actual_prev_class != CharClass::from_byte(input[end_pos])
                } else {
                    actual_prev_class != CharClass::NonWord
                }
            } else {
                // Check boundary between last matched char and next char
                let last_class = CharClass::from_byte(input[end_pos - 1]);
                let next_class = if end_pos < input.len() {
                    CharClass::from_byte(input[end_pos])
                } else {
                    CharClass::NonWord // End of input treated as non-word
                };
                last_class != next_class
            };

            // Validate against boundary requirements
            if self.match_needs_word_boundary && !is_at_boundary {
                return false;
            }
            if self.match_needs_not_word_boundary && is_at_boundary {
                return false;
            }
        }

        // Validate anchor assertions
        if self.has_anchors {
            // EndOfText: must be at end of input
            if self.match_needs_end_of_text && end_pos != input.len() {
                return false;
            }

            // EndOfLine: must be at end of input OR before newline
            if self.match_needs_end_of_line {
                let at_end_of_line = end_pos == input.len() || input.get(end_pos) == Some(&b'\n');
                if !at_end_of_line {
                    return false;
                }
            }
        }

        true
    }

    /// Executes the compiled regex on the given input (assumes NonWord prev_class).
    ///
    /// Returns (start, end) of the match if found, or None.
    pub fn execute(&self, input: &[u8]) -> Option<(usize, usize)> {
        self.execute_with_class(input, CharClass::NonWord)
    }

    /// Returns true if the regex matches anywhere in the input (unanchored).
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.find(input).is_some()
    }

    /// Returns true if the regex matches the entire input (anchored).
    pub fn is_full_match(&self, input: &[u8]) -> bool {
        match self.execute(input) {
            Some((start, end)) => start == 0 && end == input.len(),
            None => false,
        }
    }

    /// Finds the first match in the input.
    /// Returns (start, end) byte offsets.
    ///
    /// For simple unanchored patterns (including word boundaries), this is a single JIT call
    /// that searches the entire input. The JIT internally handles word boundary context tracking.
    /// For anchored patterns (^), we iterate over valid start positions.
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        // For patterns with start anchors, we need to try specific positions
        if self.has_start_anchor {
            if self.has_multiline_anchors {
                // Multiline mode: try position 0 and after each newline
                if let Some((start, end)) = self.find_at(input, 0) {
                    return Some((start, end));
                }
                for (i, &byte) in input.iter().enumerate() {
                    if byte == b'\n' && i + 1 <= input.len() {
                        if let Some((start, end)) = self.find_at(input, i + 1) {
                            return Some((start, end));
                        }
                    }
                }
                None
            } else {
                // Non-multiline: only try position 0
                self.find_at(input, 0)
            }
        } else {
            // Unanchored patterns (including word boundaries): JIT does the search internally.
            // The JIT tracks prev_char_class in r13 and uses dispatch to select the correct
            // start state based on character class context.
            self.execute(input)
        }
    }

    /// Finds a match starting at or after the given position.
    /// Returns (start, end) if found.
    ///
    /// This method correctly handles word boundaries and anchors by using the full input
    /// to determine the character class and position context before the start position.
    pub fn find_at(&self, input: &[u8], start_pos: usize) -> Option<(usize, usize)> {
        if start_pos > input.len() {
            return None;
        }

        // For anchored patterns, verify the start position is valid
        if self.has_start_anchor {
            let valid_start = if self.has_multiline_anchors {
                // Multiline: valid at position 0 or after newline
                start_pos == 0 || (start_pos > 0 && input[start_pos - 1] == b'\n')
            } else {
                // Non-multiline: only valid at position 0
                start_pos == 0
            };
            if !valid_start {
                return None;
            }
        }

        // Determine prev_class based on the character before start_pos
        let prev_class = if self.has_word_boundary && start_pos > 0 {
            CharClass::from_byte(input[start_pos - 1])
        } else {
            CharClass::NonWord
        };

        // Execute on the slice starting at start_pos
        // The JIT returns positions relative to the slice, which we then adjust
        self.execute_with_class(&input[start_pos..], prev_class)
            .map(|(rel_start, rel_end)| (start_pos + rel_start, start_pos + rel_end))
    }
}

/// JIT compiler for DFA states.
///
/// This struct handles the conversion of a DFA to native x86-64 machine code.
pub struct JitCompiler;

impl JitCompiler {
    /// Creates a new JIT compiler.
    pub fn new() -> Self {
        Self
    }

    /// Compiles a LazyDFA to native machine code.
    ///
    /// This method:
    /// 1. Forces full DFA materialization by exploring all reachable states
    /// 2. Allocates dynamic labels for all states
    /// 3. Emits optimized x86-64 assembly for each state
    /// 4. Returns an executable buffer (W^X compliant)
    ///
    /// For patterns with word boundaries, two entry points are generated:
    /// one for NonWord prev_class and one for Word prev_class.
    ///
    /// # Errors
    /// Returns an error if DFA materialization fails or assembly generation fails.
    pub fn compile_dfa(self, dfa: &mut LazyDfa) -> Result<CompiledRegex> {
        // Step 1: Materialize all reachable DFA states
        let materialized = self.materialize_dfa(dfa)?;

        // Step 2: Compile to machine code
        let (code, entry_point, entry_point_word) =
            crate::jit::x86_64::compile_states(&materialized)?;

        // Collect boundary and anchor requirements from all match states
        let mut match_needs_word_boundary = false;
        let mut match_needs_not_word_boundary = false;
        let mut match_needs_end_of_text = false;
        let mut match_needs_end_of_line = false;
        for state in &materialized.states {
            if state.is_match {
                match_needs_word_boundary |= state.needs_word_boundary;
                match_needs_not_word_boundary |= state.needs_not_word_boundary;
                match_needs_end_of_text |= state.needs_end_of_text;
                match_needs_end_of_line |= state.needs_end_of_line;
            }
        }

        Ok(CompiledRegex {
            code,
            entry_point,
            entry_point_word,
            has_word_boundary: materialized.has_word_boundary,
            match_needs_word_boundary,
            match_needs_not_word_boundary,
            has_anchors: materialized.has_anchors,
            has_start_anchor: materialized.has_start_anchor,
            has_end_anchor: materialized.has_end_anchor,
            has_multiline_anchors: materialized.has_multiline_anchors,
            match_needs_end_of_text,
            match_needs_end_of_line,
        })
    }

    /// Materializes all reachable states in the DFA.
    ///
    /// This performs a BFS from the start state(s), computing all transitions
    /// for all reachable states. Returns a snapshot of the fully-materialized DFA.
    ///
    /// For patterns with word boundaries, we materialize states reachable from
    /// both start states (NonWord and Word prev_class).
    fn materialize_dfa(&self, dfa: &mut LazyDfa) -> Result<MaterializedDfa> {
        let has_word_boundary = dfa.has_word_boundary();
        let has_anchors = dfa.has_anchors();
        let has_start_anchor = dfa.has_start_anchor();
        let has_end_anchor = dfa.has_end_anchor();
        let has_multiline_anchors = dfa.has_multiline_anchors();

        // Get both start states if pattern has word boundaries
        let start_nonword = dfa.get_start_state_for_class(CharClass::NonWord);
        let start_word = if has_word_boundary {
            Some(dfa.get_start_state_for_class(CharClass::Word))
        } else {
            None
        };

        let mut materialized = MaterializedDfa {
            states: Vec::new(),
            start: start_nonword,
            start_word,
            has_word_boundary,
            has_anchors,
            has_start_anchor,
            has_end_anchor,
            has_multiline_anchors,
        };

        let mut queue = vec![start_nonword];
        let mut visited = std::collections::HashSet::new();
        visited.insert(start_nonword);

        // Also add the Word start state to the queue if present
        if let Some(sw) = start_word {
            if visited.insert(sw) {
                queue.push(sw);
            }
        }

        while let Some(state_id) = queue.pop() {
            // Compute all 256 transitions at once using the optimized batch method
            let transitions = dfa.compute_all_transitions(state_id);

            // Add any new states to the queue
            for byte in 0..=255u8 {
                if let Some(next_state) = transitions[byte as usize] {
                    if visited.insert(next_state) {
                        queue.push(next_state);
                    }
                }
            }

            let is_match = dfa.is_match(state_id);
            let (needs_word_boundary, needs_not_word_boundary) =
                dfa.get_state_boundary_requirements(state_id);
            let (needs_end_of_text, needs_end_of_line) =
                dfa.get_state_anchor_requirements(state_id);

            materialized.states.push(MaterializedState {
                id: state_id,
                transitions,
                is_match,
                needs_word_boundary,
                needs_not_word_boundary,
                needs_end_of_text,
                needs_end_of_line,
            });
        }

        // Sort states by ID for deterministic code generation
        materialized.states.sort_by_key(|s| s.id);

        Ok(materialized)
    }
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// A fully-materialized DFA with all transitions computed.
///
/// Unlike LazyDfa, all transitions are pre-computed and stored in arrays.
pub struct MaterializedDfa {
    /// All DFA states, sorted by ID.
    pub states: Vec<MaterializedState>,
    /// The start state ID (for NonWord prev_class).
    pub start: DfaStateId,
    /// The start state ID for Word prev_class (only for word boundary patterns).
    pub start_word: Option<DfaStateId>,
    /// Whether this DFA has word boundary assertions.
    pub has_word_boundary: bool,
    /// Whether this DFA has anchor assertions (^, $).
    pub has_anchors: bool,
    /// Whether this DFA has a start anchor (^).
    pub has_start_anchor: bool,
    /// Whether this DFA has an end anchor ($).
    pub has_end_anchor: bool,
    /// Whether this DFA uses multiline mode for anchors.
    pub has_multiline_anchors: bool,
}

/// A materialized DFA state with all transitions computed.
#[derive(Debug, Clone)]
pub struct MaterializedState {
    /// The state ID.
    pub id: DfaStateId,
    /// All 256 transitions (None = dead state).
    pub transitions: [Option<DfaStateId>; 256],
    /// Whether this is a match state.
    pub is_match: bool,
    /// Whether this state requires a word boundary (\b) at the end.
    pub needs_word_boundary: bool,
    /// Whether this state requires NOT a word boundary (\B) at the end.
    pub needs_not_word_boundary: bool,
    /// Whether this state requires EndOfText ($) assertion.
    pub needs_end_of_text: bool,
    /// Whether this state requires EndOfLine ($) assertion (multiline).
    pub needs_end_of_line: bool,
}

impl MaterializedState {
    /// Analyzes transition density to choose optimal code generation strategy.
    ///
    /// Returns the number of unique non-None transitions.
    pub fn transition_density(&self) -> usize {
        self.transitions.iter().filter(|t| t.is_some()).count()
    }

    /// Returns true if this state should use a jump table.
    ///
    /// Jump tables are efficient for dense transitions (many valid bytes).
    /// Linear compare chains are better for sparse transitions.
    pub fn should_use_jump_table(&self) -> bool {
        // Use jump table if more than 10 unique transitions
        // This threshold balances code size vs execution speed
        self.transition_density() > 10
    }

    /// Groups consecutive transitions to the same target state.
    ///
    /// Returns a vector of (start_byte, end_byte_inclusive, target_state) tuples.
    /// This is used for optimizing sparse transition generation.
    pub fn transition_ranges(&self) -> Vec<(u8, u8, DfaStateId)> {
        let mut ranges = Vec::new();
        let mut current_target = None;
        let mut range_start = 0u8;

        for byte in 0..=255u8 {
            let target = self.transitions[byte as usize];

            match (current_target, target) {
                (None, Some(t)) => {
                    // Start a new range
                    current_target = Some(t);
                    range_start = byte;
                }
                (Some(curr), Some(t)) if curr == t => {
                    // Continue current range
                }
                (Some(curr), _) => {
                    // End current range (end is exclusive, so previous byte)
                    ranges.push((range_start, byte - 1, curr));
                    current_target = target;
                    range_start = byte;
                }
                (None, None) => {
                    // Stay in dead state
                }
            }

            // Handle the last byte specially
            if byte == 255 {
                if let Some(t) = current_target {
                    ranges.push((range_start, byte, t));
                }
            }
        }

        ranges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::nfa::compile;
    use crate::parser::parse;

    fn compile_pattern(pattern: &str) -> Result<CompiledRegex> {
        let ast = parse(pattern)?;
        let hir = translate(&ast)?;
        let nfa = compile(&hir)?;
        let mut dfa = LazyDfa::new(nfa);

        let compiler = JitCompiler::new();
        compiler.compile_dfa(&mut dfa)
    }

    #[test]
    fn test_compile_simple_literal() {
        let compiled = compile_pattern("abc").unwrap();
        assert!(compiled.is_full_match(b"abc"));
        assert!(!compiled.is_full_match(b"ab"));
        assert!(!compiled.is_full_match(b"abcd"));
        assert!(compiled.is_match(b"abcd")); // "abcd" contains "abc"
        assert!(compiled.is_match(b"xyzabc")); // "xyzabc" contains "abc"
        assert!(!compiled.is_match(b"xyz"));
    }

    #[test]
    fn test_compile_alternation() {
        let compiled = compile_pattern("a|b").unwrap();
        assert!(compiled.is_match(b"a"));
        assert!(compiled.is_match(b"b"));
        assert!(!compiled.is_match(b"c"));
    }

    #[test]
    fn test_compile_star() {
        let compiled = compile_pattern("a*").unwrap();
        assert!(compiled.is_match(b""));
        assert!(compiled.is_match(b"a"));
        assert!(compiled.is_match(b"aaaa"));
    }

    #[test]
    fn test_find() {
        let compiled = compile_pattern("abc").unwrap();
        assert_eq!(compiled.find(b"xyzabc123"), Some((3, 6)));
        assert_eq!(compiled.find(b"abc"), Some((0, 3)));
        assert_eq!(compiled.find(b"xyz"), None);
    }

    #[test]
    fn test_transition_ranges() {
        let mut state = MaterializedState {
            id: 0,
            transitions: [None; 256],
            is_match: false,
            needs_word_boundary: false,
            needs_not_word_boundary: false,
            needs_end_of_text: false,
            needs_end_of_line: false,
        };

        // Set up some transitions
        state.transitions[b'a' as usize] = Some(1);
        state.transitions[b'b' as usize] = Some(1);
        state.transitions[b'c' as usize] = Some(1);
        state.transitions[b'x' as usize] = Some(2);

        let ranges = state.transition_ranges();
        assert!(ranges.len() >= 2);

        // Should have grouped a,b,c together
        let abc_range = ranges.iter().find(|(_, _, target)| *target == 1).unwrap();
        assert_eq!(abc_range.0, b'a'); // start
        assert_eq!(abc_range.1, b'c'); // end (inclusive)
        assert_eq!(abc_range.2, 1); // target
    }

    #[test]
    fn test_word_boundary_jit() {
        // Test basic word boundary pattern
        let compiled = compile_pattern(r"\bword\b").unwrap();
        assert!(compiled.has_word_boundary);

        // Should match "word" as a whole word
        assert!(compiled.is_match(b"word"));
        assert!(compiled.is_match(b"word here"));
        assert!(compiled.is_match(b"a word here"));
        assert!(compiled.is_match(b"the word"));

        // Should NOT match "word" as part of another word
        assert!(!compiled.is_match(b"words"));
        assert!(!compiled.is_match(b"password"));
        assert!(!compiled.is_match(b"swordfish"));
    }

    #[test]
    fn test_word_boundary_find() {
        let compiled = compile_pattern(r"\bthe\b").unwrap();
        assert!(compiled.has_word_boundary);

        // Find "the" in various positions
        assert_eq!(compiled.find(b"the quick"), Some((0, 3)));
        assert_eq!(compiled.find(b"in the end"), Some((3, 6)));
        assert_eq!(compiled.find(b"at the"), Some((3, 6)));

        // Should not match "the" inside other words
        assert_eq!(compiled.find(b"then"), None);
        assert_eq!(compiled.find(b"other"), None);
        assert_eq!(compiled.find(b"bathe"), None);
    }

    #[test]
    fn test_not_word_boundary_jit() {
        // Test \B (not word boundary)
        let compiled = compile_pattern(r"\Bword\B").unwrap();
        assert!(compiled.has_word_boundary);

        // Should match "word" NOT at word boundaries (surrounded by word chars)
        assert!(compiled.is_match(b"swordfish"));
        assert!(compiled.is_match(b"passwords"));

        // Should NOT match "word" at word boundaries
        assert!(!compiled.is_match(b"word"));
        assert!(!compiled.is_match(b"word "));
        assert!(!compiled.is_match(b" word"));
    }

    #[test]
    fn test_mixed_boundary_jit() {
        // Start with word boundary, end with non-word boundary
        let compiled = compile_pattern(r"\bword\B").unwrap();

        assert!(compiled.is_match(b"words"));
        assert!(compiled.is_match(b"wording"));
        assert!(!compiled.is_match(b"word"));
        assert!(!compiled.is_match(b"sword"));
    }

    // =========================================================================
    // Anchor Tests (JIT)
    // =========================================================================

    #[test]
    fn test_start_anchor_jit() {
        let compiled = compile_pattern("^hello").unwrap();
        assert!(compiled.has_anchors);
        assert!(compiled.has_start_anchor);

        // Should match only at start
        assert!(compiled.is_match(b"hello world"));
        assert!(!compiled.is_match(b"say hello"));
        assert!(!compiled.is_match(b"  hello"));
    }

    #[test]
    fn test_end_anchor_jit() {
        let compiled = compile_pattern("world$").unwrap();
        assert!(compiled.has_anchors);
        assert!(compiled.has_end_anchor);
        assert!(compiled.match_needs_end_of_text);

        // Should match only at end
        assert!(compiled.is_match(b"hello world"));
        assert!(!compiled.is_match(b"world hello"));
        assert!(!compiled.is_match(b"world  "));
    }

    #[test]
    fn test_both_anchors_jit() {
        let compiled = compile_pattern("^hello$").unwrap();
        assert!(compiled.has_anchors);
        assert!(compiled.has_start_anchor);
        assert!(compiled.has_end_anchor);
        assert!(compiled.match_needs_end_of_text);

        // Should match exact string only
        assert!(compiled.is_match(b"hello"));
        assert!(!compiled.is_match(b"hello world"));
        assert!(!compiled.is_match(b"say hello"));
        assert!(!compiled.is_match(b" hello "));
    }

    #[test]
    fn test_anchor_with_pattern_jit() {
        let compiled = compile_pattern("^[a-z]+$").unwrap();

        // Should match lowercase-only strings
        assert!(compiled.is_match(b"hello"));
        assert!(compiled.is_match(b"world"));
        assert!(!compiled.is_match(b"Hello"));
        assert!(!compiled.is_match(b"hello world")); // has space
        assert!(!compiled.is_match(b"123"));
    }

    #[test]
    fn test_anchor_find_jit() {
        let compiled = compile_pattern("^hello").unwrap();

        assert_eq!(compiled.find(b"hello world"), Some((0, 5)));
        assert_eq!(compiled.find(b"say hello"), None);
    }

    #[test]
    fn test_multiline_start_anchor_jit() {
        let compiled = compile_pattern("(?m)^hello").unwrap();
        assert!(compiled.has_anchors);
        assert!(compiled.has_start_anchor);
        assert!(compiled.has_multiline_anchors);

        // Should match at start and after newlines
        assert!(compiled.is_match(b"hello world"));
        assert!(compiled.is_match(b"first\nhello"));
        assert!(compiled.is_match(b"line1\nline2\nhello"));
        assert!(!compiled.is_match(b"say hello"));
    }

    #[test]
    fn test_multiline_end_anchor_jit() {
        let compiled = compile_pattern("(?m)world$").unwrap();
        assert!(compiled.has_anchors);
        assert!(compiled.match_needs_end_of_line);

        // Should match at end and before newlines
        assert!(compiled.is_match(b"hello world"));
        assert!(compiled.is_match(b"world\nnext"));
        assert!(!compiled.is_match(b"world hello"));
    }
}
