//! DFA to machine code compilation for AArch64 (ARM64).
//!
//! This module compiles a LazyDFA to native ARM64 machine code using dynasm.
//! The compiled code is W^X compliant and optimized for performance.
//!
//! ## Features
//!
//! - Full DFA state machine compilation
//! - Word boundary and anchor support
//! - Self-loop optimization
//! - Dense and sparse transition encoding

use crate::dfa::{CharClass, DfaStateId, LazyDfa};
use crate::error::{Error, ErrorKind, Result};
use dynasmrt::{AssemblyOffset, ExecutableBuffer};

// ARM64 DFA JIT enabled
const ARM64_DFA_JIT_ENABLED: bool = true;

/// A JIT-compiled regex matcher for ARM64.
///
/// This struct holds the executable machine code generated from a DFA.
/// The code is W^X compliant (never RWX) and uses optimal alignment for
/// ARM64 instruction fetch performance.
pub struct CompiledRegex {
    /// The executable buffer containing the compiled machine code.
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
    fn execute_with_class(&self, input: &[u8], prev_class: CharClass) -> Option<(usize, usize)> {
        // ARM64 uses AAPCS64 calling convention (extern "C")
        type MatchFn = unsafe extern "C" fn(*const u8, usize) -> i64;

        let entry = if self.has_word_boundary && prev_class == CharClass::Word {
            self.entry_point_word.unwrap_or(self.entry_point)
        } else {
            self.entry_point
        };

        let func: MatchFn = unsafe { std::mem::transmute(self.code.ptr(entry)) };

        let result = unsafe { func(input.as_ptr(), input.len()) };

        if result >= 0 {
            let packed = result as u64;
            let start_pos = (packed >> 32) as usize;
            let end_pos = (packed & 0xFFFF_FFFF) as usize;

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
        if self.has_word_boundary
            && (self.match_needs_word_boundary || self.match_needs_not_word_boundary)
        {
            let actual_prev_class = if start_pos > 0 {
                CharClass::from_byte(input[start_pos - 1])
            } else {
                prev_class
            };

            let is_at_boundary = if end_pos == start_pos {
                if end_pos < input.len() {
                    actual_prev_class != CharClass::from_byte(input[end_pos])
                } else {
                    actual_prev_class != CharClass::NonWord
                }
            } else {
                let last_class = CharClass::from_byte(input[end_pos - 1]);
                let next_class = if end_pos < input.len() {
                    CharClass::from_byte(input[end_pos])
                } else {
                    CharClass::NonWord
                };
                last_class != next_class
            };

            if self.match_needs_word_boundary && !is_at_boundary {
                return false;
            }
            if self.match_needs_not_word_boundary && is_at_boundary {
                return false;
            }
        }

        if self.has_anchors {
            if self.match_needs_end_of_text && end_pos != input.len() {
                return false;
            }
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
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        if self.has_start_anchor {
            if self.has_multiline_anchors {
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
                self.find_at(input, 0)
            }
        } else {
            self.execute(input)
        }
    }

    /// Finds a match starting at or after the given position.
    pub fn find_at(&self, input: &[u8], start_pos: usize) -> Option<(usize, usize)> {
        if start_pos > input.len() {
            return None;
        }

        if self.has_start_anchor {
            let valid_start = if self.has_multiline_anchors {
                start_pos == 0 || (start_pos > 0 && input[start_pos - 1] == b'\n')
            } else {
                start_pos == 0
            };
            if !valid_start {
                return None;
            }
        }

        let prev_class = if self.has_word_boundary && start_pos > 0 {
            CharClass::from_byte(input[start_pos - 1])
        } else {
            CharClass::NonWord
        };

        self.execute_with_class(&input[start_pos..], prev_class)
            .map(|(rel_start, rel_end)| (start_pos + rel_start, start_pos + rel_end))
    }
}

/// JIT compiler for DFA states on ARM64.
///
/// This struct handles the conversion of a DFA to native ARM64 machine code.
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
    /// 3. Emits optimized ARM64 assembly for each state
    /// 4. Returns an executable buffer (W^X compliant)
    ///
    /// # Errors
    /// Returns an error if DFA materialization fails or assembly generation fails.
    pub fn compile_dfa(self, dfa: &mut LazyDfa) -> Result<CompiledRegex> {
        // ARM64 DFA JIT is disabled until assembly is fully debugged
        if !ARM64_DFA_JIT_ENABLED {
            return Err(Error::new(
                ErrorKind::Jit("ARM64 DFA JIT temporarily disabled".to_string()),
                "",
            ));
        }

        // Step 1: Materialize all reachable DFA states
        let materialized = self.materialize_dfa(dfa)?;

        // ARM64: Limit state count to avoid branch distance issues and code bloat
        // Large Unicode character classes can create many states
        const MAX_ARM64_DFA_STATES: usize = 64;
        if materialized.states.len() > MAX_ARM64_DFA_STATES {
            return Err(Error::new(
                ErrorKind::Jit(format!(
                    "DFA too large for ARM64 JIT ({} states, max {})",
                    materialized.states.len(),
                    MAX_ARM64_DFA_STATES
                )),
                "",
            ));
        }

        // Step 2: Compile to machine code
        let (code, entry_point, entry_point_word) =
            crate::jit::aarch64::compile_states(&materialized)?;

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
    fn materialize_dfa(&self, dfa: &mut LazyDfa) -> Result<MaterializedDfa> {
        let has_word_boundary = dfa.has_word_boundary();
        let has_anchors = dfa.has_anchors();
        let has_start_anchor = dfa.has_start_anchor();
        let has_end_anchor = dfa.has_end_anchor();
        let has_multiline_anchors = dfa.has_multiline_anchors();

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

        if let Some(sw) = start_word {
            if visited.insert(sw) {
                queue.push(sw);
            }
        }

        while let Some(state_id) = queue.pop() {
            let transitions = dfa.compute_all_transitions(state_id);

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
    pub fn transition_density(&self) -> usize {
        self.transitions.iter().filter(|t| t.is_some()).count()
    }

    /// Returns true if this state should use a jump table.
    pub fn should_use_jump_table(&self) -> bool {
        self.transition_density() > 10
    }

    /// Groups consecutive transitions to the same target state.
    pub fn transition_ranges(&self) -> Vec<(u8, u8, DfaStateId)> {
        let mut ranges = Vec::new();
        let mut current_target = None;
        let mut range_start = 0u8;

        for byte in 0..=255u8 {
            let target = self.transitions[byte as usize];

            match (current_target, target) {
                (None, Some(t)) => {
                    current_target = Some(t);
                    range_start = byte;
                }
                (Some(curr), Some(t)) if curr == t => {}
                (Some(curr), _) => {
                    ranges.push((range_start, byte - 1, curr));
                    current_target = target;
                    range_start = byte;
                }
                (None, None) => {}
            }

            if byte == 255 {
                if let Some(t) = current_target {
                    ranges.push((range_start, byte, t));
                }
            }
        }

        ranges
    }
}
