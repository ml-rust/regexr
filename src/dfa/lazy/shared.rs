//! Shared types for the Lazy DFA engine.
//!
//! Contains types used by both the interpreter and potentially a JIT backend.

use std::collections::{BTreeSet, HashMap};

use crate::hir::unicode::is_word_byte;
use crate::nfa::{Nfa, NfaInstruction, StateId as NfaStateId};

/// State ID in the DFA (premultiplied by STRIDE for direct indexing).
///
/// The ID is premultiplied: `real_state_index * STRIDE` so that
/// `transitions[state_id + byte]` works without multiplication.
pub type DfaStateId = u32;

/// Number of transitions per state (256 bytes).
pub const STRIDE: u32 = 256;

/// Tagged state ID encoding.
/// High bits encode status, low bits encode the premultiplied state ID.
///
/// Layout (32-bit):
/// - Bits 0-29:  Premultiplied state index (supports up to 4M states)
/// - Bit 30:     Match flag (1 = match state)
/// - Bit 31:     Dead flag (1 = no further transitions possible)
///
/// Special values:
/// - DEAD_STATE (0xFFFFFFFF): No valid transition, pattern failed
/// - UNKNOWN (0x80000000): Transition not yet computed
pub const TAG_MATCH: u32 = 1 << 30;
pub const TAG_DEAD: u32 = 1 << 31;
pub const TAG_MASK: u32 = TAG_MATCH | TAG_DEAD;
pub const STATE_MASK: u32 = !TAG_MASK;

/// Sentinel value for "dead" state (pattern cannot match).
pub const DEAD_STATE: u32 = TAG_DEAD | STATE_MASK;

/// Sentinel value for "unknown" transition (needs computation).
pub const UNKNOWN_STATE: u32 = TAG_DEAD;

/// Default cache limit (number of states).
pub const DEFAULT_CACHE_LIMIT: usize = 10_000;

/// Position context for anchor assertions.
/// Tracks what we know about the current position relative to input boundaries.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PositionContext {
    /// True if at start of input (position 0)
    pub at_start_of_input: bool,
    /// True if at start of line (position 0 or after \n)
    pub at_start_of_line: bool,
    /// True if at end of input
    pub at_end_of_input: bool,
    /// True if at end of line (at end of input or before \n)
    pub at_end_of_line: bool,
}

impl PositionContext {
    /// Context for start of input (position 0)
    pub fn start_of_input() -> Self {
        Self {
            at_start_of_input: true,
            at_start_of_line: true,
            at_end_of_input: false,
            at_end_of_line: false,
        }
    }

    /// Context for middle of input (not at any boundary)
    pub fn middle() -> Self {
        Self {
            at_start_of_input: false,
            at_start_of_line: false,
            at_end_of_input: false,
            at_end_of_line: false,
        }
    }

    /// Context after a newline character
    pub fn after_newline() -> Self {
        Self {
            at_start_of_input: false,
            at_start_of_line: true,
            at_end_of_input: false,
            at_end_of_line: false,
        }
    }
}

/// Character class for word boundary detection.
/// Tracks whether a character is a word character or not.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CharClass {
    /// Non-word character (anything except [a-zA-Z0-9_]) or start/end of input
    #[default]
    NonWord = 0,
    /// Word character [a-zA-Z0-9_]
    Word = 1,
}

impl CharClass {
    /// Classifies a byte as Word or NonWord.
    #[inline]
    pub fn from_byte(b: u8) -> Self {
        if is_word_byte(b) {
            CharClass::Word
        } else {
            CharClass::NonWord
        }
    }
}

/// Key for the state map.
/// For patterns without word boundaries: just the NFA state set.
/// For patterns with word boundaries: NFA state set + previous character class.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum StateKey {
    /// Simple key without character class (for patterns without word boundaries)
    Simple(BTreeSet<NfaStateId>),
    /// Key with character class (for patterns with word boundaries)
    WithClass(BTreeSet<NfaStateId>, CharClass),
}

/// A DFA state (metadata only, transitions are in dense table).
#[derive(Debug, Clone)]
pub struct DfaState {
    /// Whether this is a match state.
    pub is_match: bool,
    /// The set of NFA states this DFA state represents.
    pub nfa_states: BTreeSet<NfaStateId>,
    /// The character class this state was created with (for word boundary patterns).
    /// This is the class of the byte that transitioned INTO this state.
    pub prev_class: CharClass,
}

impl DfaState {
    /// Creates a new DFA state.
    pub fn new(nfa_states: BTreeSet<NfaStateId>, is_match: bool, prev_class: CharClass) -> Self {
        Self {
            is_match,
            nfa_states,
            prev_class,
        }
    }
}

/// Context for lazy DFA execution.
///
/// This struct contains the mutable state needed during DFA operation,
/// including the state cache and transition table.
#[derive(Debug, Clone)]
pub struct LazyDfaContext {
    /// The underlying NFA.
    pub(crate) nfa: Nfa,
    /// DFA states metadata (NFA state set, match status, etc.).
    pub(crate) states: Vec<DfaState>,
    /// Dense transition table: transitions[state_id + byte] = tagged next state.
    /// State IDs are premultiplied by STRIDE for direct indexing.
    /// Values are tagged: high bits indicate match/dead status.
    pub(crate) transitions: Vec<u32>,
    /// Map from state keys to DFA state IDs (premultiplied).
    pub(crate) state_map: HashMap<StateKey, DfaStateId>,
    /// The start state (premultiplied, for NonWord prev_class).
    pub(crate) start: DfaStateId,
    /// Cache size limit (number of states).
    /// When exceeded, the entire cache is flushed (not LRU - too slow).
    pub(crate) cache_limit: usize,
    /// Number of cache flushes (for debugging/profiling).
    pub(crate) flush_count: usize,
    /// Whether this pattern has word boundary assertions.
    /// When true, states are keyed by (nfa_states, prev_class).
    pub(crate) has_word_boundary: bool,
    /// Whether this pattern has anchor assertions (^, $).
    pub(crate) has_anchors: bool,
    /// Whether pattern has ^ (start of text/line) anchor.
    pub(crate) has_start_anchor: bool,
    /// Whether pattern has $ (end of text/line) anchor.
    pub(crate) has_end_anchor: bool,
    /// Whether pattern uses multiline mode (^ matches after \n, $ matches before \n).
    pub(crate) has_multiline_anchors: bool,
}

impl LazyDfaContext {
    /// Creates a new context for a given NFA.
    pub fn new(mut nfa: Nfa) -> Self {
        // Precompute epsilon closures for NFAs with many epsilon transitions
        nfa.precompute_epsilon_closures();

        // Check if the NFA has word boundary instructions
        let has_word_boundary = nfa_has_word_boundary(&nfa);

        // Check for anchor instructions
        let (has_anchors, has_start_anchor, has_end_anchor, has_multiline_anchors) =
            nfa_anchor_info(&nfa);

        let mut ctx = Self {
            nfa,
            states: Vec::new(),
            transitions: Vec::new(),
            state_map: HashMap::new(),
            start: 0,
            cache_limit: DEFAULT_CACHE_LIMIT,
            flush_count: 0,
            has_word_boundary,
            has_anchors,
            has_start_anchor,
            has_end_anchor,
            has_multiline_anchors,
        };

        // Create the start state
        let mut start_set = BTreeSet::new();
        start_set.insert(ctx.nfa.start);

        // Compute epsilon closure with assertion filtering
        let start_closure = if has_word_boundary || has_anchors {
            epsilon_closure_with_context(
                &ctx.nfa,
                &start_set,
                None,
                Some(PositionContext::start_of_input()),
            )
        } else {
            ctx.nfa.epsilon_closure(&start_set)
        };

        ctx.start = get_or_create_state_with_class(&mut ctx, start_closure, CharClass::NonWord);

        ctx
    }

    /// Returns whether this DFA has word boundary assertions.
    pub fn has_word_boundary(&self) -> bool {
        self.has_word_boundary
    }

    /// Returns whether this DFA has anchor assertions.
    pub fn has_anchors(&self) -> bool {
        self.has_anchors
    }

    /// Returns whether this DFA has a start anchor.
    pub fn has_start_anchor(&self) -> bool {
        self.has_start_anchor
    }

    /// Returns whether this DFA has an end anchor.
    pub fn has_end_anchor(&self) -> bool {
        self.has_end_anchor
    }

    /// Returns whether this DFA has multiline anchors.
    pub fn has_multiline_anchors(&self) -> bool {
        self.has_multiline_anchors
    }

    /// Returns the start state.
    pub fn start(&self) -> DfaStateId {
        self.start
    }

    /// Returns the number of cached states.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Returns the number of cache flushes.
    pub fn flush_count(&self) -> usize {
        self.flush_count
    }

    /// Sets the cache size limit.
    pub fn set_cache_limit(&mut self, limit: usize) {
        self.cache_limit = limit;
    }
}

/// Checks if an NFA contains word boundary instructions.
pub fn nfa_has_word_boundary(nfa: &Nfa) -> bool {
    nfa.states.iter().any(|state| {
        matches!(
            &state.instruction,
            Some(NfaInstruction::WordBoundary) | Some(NfaInstruction::NotWordBoundary)
        )
    })
}

/// Returns anchor information for an NFA.
/// Returns (has_anchors, has_start_anchor, has_end_anchor, has_multiline_anchors).
pub fn nfa_anchor_info(nfa: &Nfa) -> (bool, bool, bool, bool) {
    let mut has_start_anchor = false;
    let mut has_end_anchor = false;
    let mut has_multiline_anchors = false;

    for state in &nfa.states {
        match &state.instruction {
            Some(NfaInstruction::StartOfText) => has_start_anchor = true,
            Some(NfaInstruction::EndOfText) => has_end_anchor = true,
            Some(NfaInstruction::StartOfLine) => {
                has_start_anchor = true;
                has_multiline_anchors = true;
            }
            Some(NfaInstruction::EndOfLine) => {
                has_end_anchor = true;
                has_multiline_anchors = true;
            }
            _ => {}
        }
    }

    let has_anchors = has_start_anchor || has_end_anchor;
    (
        has_anchors,
        has_start_anchor,
        has_end_anchor,
        has_multiline_anchors,
    )
}

/// Computes epsilon closure with optional boundary filtering and position context.
pub fn epsilon_closure_with_context(
    nfa: &Nfa,
    seeds: &BTreeSet<NfaStateId>,
    is_at_boundary: Option<bool>,
    pos_ctx: Option<PositionContext>,
) -> BTreeSet<NfaStateId> {
    let mut closure = BTreeSet::new();
    let mut stack: Vec<NfaStateId> = seeds.iter().copied().collect();

    while let Some(state_id) = stack.pop() {
        if !closure.insert(state_id) {
            continue;
        }

        let state = match nfa.get(state_id) {
            Some(s) => s,
            None => continue,
        };

        match &state.instruction {
            Some(NfaInstruction::WordBoundary) => match is_at_boundary {
                Some(true) => {}
                Some(false) => continue,
                None => continue,
            },
            Some(NfaInstruction::NotWordBoundary) => match is_at_boundary {
                Some(false) => {}
                Some(true) => continue,
                None => continue,
            },
            Some(NfaInstruction::StartOfText) => match pos_ctx {
                Some(ctx) if ctx.at_start_of_input => {}
                Some(_) => continue,
                None => continue,
            },
            Some(NfaInstruction::EndOfText) => match pos_ctx {
                Some(ctx) if ctx.at_end_of_input => {}
                Some(_) => continue,
                None => continue,
            },
            Some(NfaInstruction::StartOfLine) => match pos_ctx {
                Some(ctx) if ctx.at_start_of_line => {}
                Some(_) => continue,
                None => continue,
            },
            Some(NfaInstruction::EndOfLine) => match pos_ctx {
                Some(ctx) if ctx.at_end_of_line => {}
                Some(_) => continue,
                None => continue,
            },
            _ => {}
        }

        for &eps_target in &state.epsilon {
            if !closure.contains(&eps_target) {
                stack.push(eps_target);
            }
        }
    }

    closure
}

/// Gets or creates a DFA state for a set of NFA states with a given character class.
pub fn get_or_create_state_with_class(
    ctx: &mut LazyDfaContext,
    nfa_states: BTreeSet<NfaStateId>,
    prev_class: CharClass,
) -> DfaStateId {
    let key = if ctx.has_word_boundary {
        StateKey::WithClass(nfa_states.clone(), prev_class)
    } else {
        StateKey::Simple(nfa_states.clone())
    };

    if let Some(&id) = ctx.state_map.get(&key) {
        return id;
    }

    // Check if cache is full - if so, flush it
    if ctx.states.len() >= ctx.cache_limit {
        flush_cache(ctx);
        if let Some(&id) = ctx.state_map.get(&key) {
            return id;
        }
    }

    // Check if this is a match state
    let is_match = nfa_states
        .iter()
        .any(|&s| ctx.nfa.get(s).map(|state| state.is_match).unwrap_or(false));

    let state_index = ctx.states.len();
    let premul_id = (state_index as u32) * STRIDE;

    ctx.states
        .push(DfaState::new(nfa_states, is_match, prev_class));
    ctx.transitions
        .resize(ctx.transitions.len() + STRIDE as usize, UNKNOWN_STATE);
    ctx.state_map.insert(key, premul_id);

    premul_id
}

/// Flushes the cache, keeping only the start state.
pub fn flush_cache(ctx: &mut LazyDfaContext) {
    ctx.flush_count += 1;

    let start_index = state_index(ctx.start);
    let start_nfa_states = ctx.states[start_index].nfa_states.clone();
    let start_is_match = ctx.states[start_index].is_match;
    let start_prev_class = ctx.states[start_index].prev_class;

    ctx.states.clear();
    ctx.transitions.clear();
    ctx.state_map.clear();

    let key = if ctx.has_word_boundary {
        StateKey::WithClass(start_nfa_states.clone(), start_prev_class)
    } else {
        StateKey::Simple(start_nfa_states.clone())
    };
    ctx.states.push(DfaState::new(
        start_nfa_states,
        start_is_match,
        start_prev_class,
    ));
    ctx.transitions.resize(STRIDE as usize, UNKNOWN_STATE);
    ctx.state_map.insert(key, 0);
    ctx.start = 0;
}

/// Converts a premultiplied state ID to a state index.
#[inline(always)]
pub fn state_index(premul_id: DfaStateId) -> usize {
    ((premul_id & STATE_MASK) / STRIDE) as usize
}

/// Creates a tagged state ID from a premultiplied ID and match status.
#[inline(always)]
pub fn tag_state(premul_id: DfaStateId, is_match: bool) -> u32 {
    if is_match {
        premul_id | TAG_MATCH
    } else {
        premul_id
    }
}

/// Checks if a tagged state ID indicates a dead state.
#[inline(always)]
pub fn is_dead_state(tagged: u32) -> bool {
    tagged == DEAD_STATE
}

/// Checks if a tagged state ID indicates an unknown transition.
#[inline(always)]
pub fn is_unknown_state(tagged: u32) -> bool {
    tagged == UNKNOWN_STATE
}

/// Checks if a tagged state ID indicates a match state.
#[inline(always)]
pub fn is_tagged_match(tagged: u32) -> bool {
    (tagged & TAG_MATCH) != 0
}

/// Extracts the premultiplied state ID from a tagged value.
#[inline(always)]
pub fn untag_state(tagged: u32) -> DfaStateId {
    tagged & STATE_MASK
}
