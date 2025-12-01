//! NFA state definitions.

use crate::hir::CodepointClass;
use std::collections::BTreeSet;

/// A state identifier.
pub type StateId = u32;

/// An NFA (Nondeterministic Finite Automaton).
#[derive(Debug, Clone)]
pub struct Nfa {
    /// All states in the NFA.
    pub states: Vec<NfaState>,
    /// The start state.
    pub start: StateId,
    /// Match states.
    pub matches: Vec<StateId>,
    /// Number of capture groups.
    pub capture_count: u32,
    /// Whether the pattern has backreferences.
    pub has_backrefs: bool,
    /// Whether the pattern has lookarounds.
    pub has_lookaround: bool,
    /// Precomputed epsilon closures for each state (optional optimization).
    /// When present, `epsilon_closure()` uses these instead of computing on-the-fly.
    pub epsilon_closures: Option<Vec<BTreeSet<StateId>>>,
}

impl Nfa {
    /// Creates a new empty NFA.
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            start: 0,
            matches: Vec::new(),
            capture_count: 0,
            has_backrefs: false,
            has_lookaround: false,
            epsilon_closures: None,
        }
    }

    /// Adds a new state and returns its ID.
    pub fn add_state(&mut self, state: NfaState) -> StateId {
        let id = self.states.len() as StateId;
        self.states.push(state);
        id
    }

    /// Returns the number of states.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Gets a state by ID.
    pub fn get(&self, id: StateId) -> Option<&NfaState> {
        self.states.get(id as usize)
    }

    /// Gets a mutable state by ID.
    pub fn get_mut(&mut self, id: StateId) -> Option<&mut NfaState> {
        self.states.get_mut(id as usize)
    }

    /// Computes the epsilon closure of a set of states.
    pub fn epsilon_closure(&self, states: &BTreeSet<StateId>) -> BTreeSet<StateId> {
        // Fast path: if we have precomputed closures, use them
        if let Some(ref precomputed) = self.epsilon_closures {
            let mut closure = BTreeSet::new();
            for &state_id in states {
                if let Some(state_closure) = precomputed.get(state_id as usize) {
                    closure.extend(state_closure.iter().copied());
                }
            }
            return closure;
        }

        // Slow path: compute epsilon closure on the fly
        let mut closure = states.clone();
        let mut stack: Vec<StateId> = states.iter().copied().collect();

        while let Some(state_id) = stack.pop() {
            if let Some(state) = self.get(state_id) {
                for &next in &state.epsilon {
                    if closure.insert(next) {
                        stack.push(next);
                    }
                }
            }
        }

        closure
    }

    /// Precomputes epsilon closures for all states.
    /// This significantly speeds up DFA construction for NFAs with many epsilon transitions.
    pub fn precompute_epsilon_closures(&mut self) {
        // Count epsilon transitions to decide if precomputation is worthwhile
        let epsilon_count: usize = self.states.iter().map(|s| s.epsilon.len()).sum();
        if epsilon_count < 100 {
            // Not enough epsilon transitions to justify precomputation
            return;
        }

        let mut closures = Vec::with_capacity(self.states.len());

        for state_id in 0..self.states.len() {
            let mut closure = BTreeSet::new();
            closure.insert(state_id as StateId);

            let mut stack = vec![state_id as StateId];
            while let Some(sid) = stack.pop() {
                if let Some(state) = self.get(sid) {
                    for &next in &state.epsilon {
                        if closure.insert(next) {
                            stack.push(next);
                        }
                    }
                }
            }

            closures.push(closure);
        }

        self.epsilon_closures = Some(closures);
    }
}

impl Default for Nfa {
    fn default() -> Self {
        Self::new()
    }
}

/// A single NFA state.
#[derive(Debug, Clone)]
pub struct NfaState {
    /// Byte transitions: (byte_range, target_state).
    pub transitions: Vec<(ByteRange, StateId)>,
    /// Epsilon (empty) transitions.
    pub epsilon: Vec<StateId>,
    /// Whether this is a match state.
    pub is_match: bool,
    /// Optional instruction for capture groups, lookarounds, etc.
    pub instruction: Option<NfaInstruction>,
}

impl NfaState {
    /// Creates a new empty state.
    pub fn new() -> Self {
        Self {
            transitions: Vec::new(),
            epsilon: Vec::new(),
            is_match: false,
            instruction: None,
        }
    }

    /// Creates a match state.
    pub fn match_state() -> Self {
        Self {
            transitions: Vec::new(),
            epsilon: Vec::new(),
            is_match: true,
            instruction: None,
        }
    }

    /// Adds a byte transition.
    pub fn add_transition(&mut self, range: ByteRange, target: StateId) {
        self.transitions.push((range, target));
    }

    /// Adds an epsilon transition.
    pub fn add_epsilon(&mut self, target: StateId) {
        self.epsilon.push(target);
    }
}

impl Default for NfaState {
    fn default() -> Self {
        Self::new()
    }
}

/// A byte range for transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    /// Start of range (inclusive).
    pub start: u8,
    /// End of range (inclusive).
    pub end: u8,
}

impl ByteRange {
    /// Creates a new byte range.
    pub fn new(start: u8, end: u8) -> Self {
        Self { start, end }
    }

    /// Creates a range for a single byte.
    pub fn single(byte: u8) -> Self {
        Self { start: byte, end: byte }
    }

    /// Creates a range matching any byte.
    pub fn any() -> Self {
        Self { start: 0, end: 255 }
    }

    /// Returns true if this range contains the byte.
    pub fn contains(&self, byte: u8) -> bool {
        byte >= self.start && byte <= self.end
    }

    /// Returns true if this range overlaps with another.
    pub fn overlaps(&self, other: &ByteRange) -> bool {
        self.start <= other.end && other.start <= self.end
    }
}

/// Special instructions for NFA states.
#[derive(Debug, Clone)]
pub enum NfaInstruction {
    /// Start of a capture group.
    CaptureStart(u32),
    /// End of a capture group.
    CaptureEnd(u32),
    /// Backreference to a capture group.
    Backref(u32),
    /// Word boundary assertion.
    WordBoundary,
    /// Not word boundary assertion.
    NotWordBoundary,
    /// Start of text assertion.
    StartOfText,
    /// End of text assertion.
    EndOfText,
    /// Start of line assertion.
    StartOfLine,
    /// End of line assertion.
    EndOfLine,
    /// Positive lookahead.
    PositiveLookahead(Box<Nfa>),
    /// Negative lookahead.
    NegativeLookahead(Box<Nfa>),
    /// Positive lookbehind.
    PositiveLookbehind(Box<Nfa>),
    /// Negative lookbehind.
    NegativeLookbehind(Box<Nfa>),
    /// Marker for non-greedy quantifier preference.
    /// When this state is reached and leads to a match, prefer this match
    /// over longer matches from continuing the quantifier.
    NonGreedyExit,
    /// Unicode codepoint class matching.
    /// Consumes a full UTF-8 codepoint and checks membership in the class.
    /// The StateId is the next state to transition to on match.
    CodepointClass(CodepointClass, StateId),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_range() {
        let range = ByteRange::new(b'a', b'z');
        assert!(range.contains(b'm'));
        assert!(!range.contains(b'A'));
    }

    #[test]
    fn test_epsilon_closure() {
        let mut nfa = Nfa::new();

        // State 0 -> epsilon -> State 1 -> epsilon -> State 2
        let mut s0 = NfaState::new();
        s0.add_epsilon(1);
        nfa.add_state(s0);

        let mut s1 = NfaState::new();
        s1.add_epsilon(2);
        nfa.add_state(s1);

        nfa.add_state(NfaState::new());

        let mut initial = BTreeSet::new();
        initial.insert(0);

        let closure = nfa.epsilon_closure(&initial);
        assert!(closure.contains(&0));
        assert!(closure.contains(&1));
        assert!(closure.contains(&2));
    }
}
