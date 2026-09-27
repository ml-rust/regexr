//! Shared types for the Lazy DFA engine.
//!
//! Contains types used by both the interpreter and potentially a JIT backend.

use std::sync::Arc;

use crate::hash::FxHashMap;
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

/// How far past [`LazyDfaContext::cache_limit`] the cache may grow while a
/// search is in flight.
///
/// A search holds premultiplied state IDs in registers, so flushing under it
/// would leave those IDs pointing at nothing; flushes are therefore deferred
/// until the search ends and the cache is allowed to grow instead. That growth
/// still has to stop somewhere: every cached state costs ~1KB for its 256×u32
/// transition row, plus a second row of the same size once the unanchored table
/// is built. At 4× the 10_000-state default that is a ~90MB worst case for a
/// single search — high enough that no ordinary pattern reaches it, bounded
/// enough that a pathological one cannot exhaust memory.
pub const CACHE_GROWTH_CEILING_FACTOR: usize = 4;

/// How much input the per-start attempts in an unanchored search may walk in
/// total before a single-pass fallback takes over. A few passes' worth: high
/// enough that attempts which fail near their start never reach it, low enough
/// that ones which scan to the end trip it within a handful of tries.
///
/// Shared by [`LazyDfa::find_from`](super::interpreter::LazyDfa::find_from)
/// and `EagerDfa::find_from`, through [`scan_budget`].
pub(crate) const SCAN_BUDGET_FACTOR: usize = 4;

/// The walked-byte budget of a start-by-start search of `input_len` bytes from
/// `from`, for a pattern whose matches span at most `max_match_len` bytes.
///
/// An attempt of a pattern with a bounded match length never walks past that
/// length, so trying every start already costs O(m·n), and does so at the
/// speed of the DFA's table walk. The budget then covers that bound, and the
/// fallback — whose own O(m·n) runs with a far larger constant when m is large
/// (`a{50000}`) — is kept for patterns that can walk without limit.
pub(crate) fn scan_budget(input_len: usize, from: usize, max_match_len: Option<usize>) -> usize {
    let per_byte = SCAN_BUDGET_FACTOR.max(max_match_len.unwrap_or(0));
    (input_len - from.min(input_len) + 1).saturating_mul(per_byte)
}

/// A lazy-DFA search stopped because the state cache could not grow further.
///
/// Distinct from "no match": the transition table is incomplete, so the answer
/// is unknown. Callers re-run the search on an engine without a state cache
/// (the PikeVM) rather than reporting the partial scan's result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheCeilingExceeded;

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

/// An NFA state subset, canonicalised: sorted ascending and duplicate-free.
///
/// Two subsets built independently from the same states therefore compare and
/// hash equal, which is what lets [`get_or_create_state_with_class`] intern them
/// to one DFA state. The `Arc` is not part of that identity — `Arc`'s `PartialEq`
/// and `Hash` both go through the pointee — it is only there so the same subset
/// can be the map key and the stored state's member list without being copied
/// twice.
///
/// Immutable once built. Under the sharing above, mutating one in place would
/// silently rewrite the other half of the pair, including a key already hashed
/// into the state map.
pub type NfaSubset = Arc<[NfaStateId]>;

/// Reusable scratch space for one epsilon-closure walk.
///
/// A walk needs a visited set over NFA state ids and a worklist. Both used to be
/// allocated fresh per call — a `BTreeSet` plus a `Vec` — which is most of what
/// subset construction spends its time on. This replaces them with a
/// generation-tagged sparse set: `stamp[id] == generation` means "seen in this
/// walk", so starting a walk costs a counter bump instead of an O(V) wipe, and
/// `touched` serves as both the worklist and the accumulator the canonical
/// subset is sorted out of.
///
/// One buffer per context is enough. No closure walk nests inside another, and
/// each one turns `touched` into an owned, buffer-independent value before the
/// next one starts.
#[derive(Debug, Clone)]
pub(crate) struct ClosureScratch {
    /// Generation in which each NFA state was last marked. `0` is "never".
    stamp: Vec<u32>,
    /// Generation of the walk in progress; never `0` once one has begun.
    generation: u32,
    /// States marked in this walk, in discovery order. Both the worklist (read
    /// through an advancing cursor) and the accumulated result.
    touched: Vec<NfaStateId>,
}

impl ClosureScratch {
    /// Creates scratch space for an NFA with `state_count` states.
    pub(crate) fn new(state_count: usize) -> Self {
        Self {
            stamp: vec![0; state_count],
            generation: 0,
            touched: Vec::new(),
        }
    }

    /// Starts a walk, retiring everything the previous one left behind.
    ///
    /// Both halves matter: the counter bump is what invalidates the previous
    /// walk's marks, and the clear is what stops its states from leaking into
    /// this walk's result.
    fn begin(&mut self) {
        self.touched.clear();
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Wrapped. Stamps still holding the old top-of-range generation
            // would now alias the new one, so retire them all — once every
            // ~4 billion walks.
            self.stamp.iter_mut().for_each(|slot| *slot = 0);
            self.generation = 1;
        }
    }

    /// Records `id` as seen in this walk, enqueuing it if it is new.
    fn mark(&mut self, id: NfaStateId) {
        let generation = self.generation;
        match self.stamp.get_mut(id as usize) {
            Some(slot) if *slot == generation => return,
            Some(slot) => *slot = generation,
            // No such NFA state, so it has no epsilon edges and cannot cycle.
            // Recorded anyway, because the `BTreeSet` walk this replaces
            // recorded it too; `finish` dedups, so a repeat cannot survive into
            // the subset.
            None => {}
        }
        self.touched.push(id);
    }

    /// The walk's result as an [`NfaSubset`], owned independently of this buffer.
    ///
    /// Canonicalisation is a sort of the touched states: O(k log k) in the
    /// closure's own size. Scanning `stamp` in id order instead would be O(V) in
    /// the whole NFA, which loses badly exactly where it matters — tokenizer
    /// shaped patterns have a large NFA and small closures, so every walk would
    /// pay for states it never touched.
    fn finish(&mut self) -> NfaSubset {
        self.touched.sort_unstable();
        self.touched.dedup();
        Arc::from(self.touched.as_slice())
    }
}

/// Key for the state map.
/// For patterns without word boundaries: just the NFA state set.
/// For patterns with word boundaries: NFA state set + previous character class.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum StateKey {
    /// Simple key without character class (for patterns without word boundaries)
    Simple(NfaSubset),
    /// Key with character class (for patterns with word boundaries)
    WithClass(NfaSubset, CharClass),
}

/// A DFA state (metadata only, transitions are in dense table).
#[derive(Debug, Clone)]
pub struct DfaState {
    /// Whether this is a match state.
    pub is_match: bool,
    /// The set of NFA states this DFA state represents.
    ///
    /// Shares its allocation with this state's key in
    /// [`LazyDfaContext::state_map`], so it must never be mutated in place.
    pub nfa_states: NfaSubset,
    /// The character class this state was created with (for word boundary patterns).
    /// This is the class of the byte that transitioned INTO this state.
    pub prev_class: CharClass,
    /// Whether a match is reachable here without passing an end assertion.
    ///
    /// An end anchor belongs to the branch that carries it, not to the pattern.
    /// After `a` in `a|b$`, a match is reachable through the anchor-free branch,
    /// so the match stands anywhere; after `a` in `a$|b` it is not, and `$` has
    /// to hold. The state set cannot tell those apart on its own — the closure
    /// follows the assertion's epsilon, so the shared match state appears in
    /// both — so this is settled while the closure is built and recorded here.
    pub match_without_end_assertion: bool,
}

impl DfaState {
    /// Creates a new DFA state.
    pub fn new(
        nfa_states: NfaSubset,
        is_match: bool,
        prev_class: CharClass,
        match_without_end_assertion: bool,
    ) -> Self {
        Self {
            is_match,
            nfa_states,
            prev_class,
            match_without_end_assertion,
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
    ///
    /// Shared rather than owned so a fallback engine — the PikeVM, when a search
    /// gives up on the cache ceiling — can be built from it without copying it.
    pub(crate) nfa: Arc<Nfa>,
    /// Scratch space shared by every epsilon-closure walk this context runs.
    ///
    /// Kept as its own field rather than passed around, so a call site can
    /// borrow it mutably alongside a shared borrow of [`LazyDfaContext::nfa`]
    /// and [`LazyDfaContext::states`]. That only works when the fields are
    /// named directly at the call — a `&mut self` method that hands out the
    /// scratch would borrow the whole context.
    pub(crate) scratch: ClosureScratch,
    /// DFA states metadata (NFA state set, match status, etc.).
    pub(crate) states: Vec<DfaState>,
    /// Dense transition table: transitions[state_id + byte] = tagged next state.
    /// State IDs are premultiplied by STRIDE for direct indexing.
    /// Values are tagged: high bits indicate match/dead status.
    pub(crate) transitions: Vec<u32>,
    /// Transitions for the *unanchored* automaton, indexed identically.
    ///
    /// Same subset construction, except the NFA start is folded into every
    /// target set — the automaton is always also considering a match that begins
    /// at the position just reached. One pass over the input therefore covers
    /// every start position at once, instead of one pass per start.
    ///
    /// Built lazily and only for the searches that ask for it, so a pattern that
    /// never leaves the anchored path never pays for it.
    pub(crate) transitions_unanchored: Vec<u32>,
    /// Input offset the last scan reached before dying, used to tell a start
    /// attempt that gave up immediately from one that scanned to the end.
    pub(crate) last_reach: usize,
    /// Map from state keys to DFA state IDs (premultiplied).
    pub(crate) state_map: FxHashMap<StateKey, DfaStateId>,
    /// The start state (premultiplied, for NonWord prev_class).
    pub(crate) start: DfaStateId,
    /// Cache size limit (number of states).
    /// When exceeded, the entire cache is flushed (not LRU - too slow).
    pub(crate) cache_limit: usize,
    /// Number of cache flushes (for debugging/profiling).
    pub(crate) flush_count: usize,
    /// How many searches are currently holding state IDs.
    ///
    /// A running search keeps premultiplied IDs outside the context, and a flush
    /// invalidates every one of them, so flushing is suppressed while this is
    /// non-zero and performed when it drops back to zero. It counts rather than
    /// flags because a search may nest inside another one.
    pub(crate) search_depth: u32,
    /// Set when a search needed a state the cache is no longer allowed to hold.
    ///
    /// Written only by [`get_or_create_state_with_class`] and consumed only by
    /// the outermost search entry point, which turns it into a
    /// [`CacheCeilingExceeded`] the caller cannot ignore. It is never a public
    /// signal: the typed return is.
    pub(crate) ceiling_exceeded: bool,
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
    /// Whether the *start* anchor specifically is line-mode — see
    /// [`nfa_anchor_info`].
    pub(crate) has_multiline_start_anchor: bool,
    /// Whether any accepting state is reachable without an end assertion.
    ///
    /// When none is — every match this pattern can make ends on `$` — the
    /// greedy end is the only candidate worth checking, because `$` holds at
    /// one position and a shorter end cannot reach it. That lets the anchored
    /// scan reject a start outright instead of re-checking it per state, which
    /// is what `([a-zA-Z]+)*$` needs to stay fast over a non-matching input.
    pub(crate) has_clean_accept: bool,
    /// How many transitions have been computed one byte at a time.
    ///
    /// `compute_all_transitions` computes a whole row in one batched pass, so
    /// this counter stays put across such a call. The unit test that proves the
    /// batched and per-byte paths agree asserts exactly that, which is what
    /// keeps the batching from being quietly replaced by a per-byte loop that
    /// would still pass the equality check.
    #[cfg(test)]
    pub(crate) per_byte_computations: usize,
    /// How many distinct runs `compute_all_transitions_with_context` computed
    /// a closure for, as opposed to skipping (cached/dead) or extending an
    /// existing run.
    ///
    /// Exists to pin the run-merge sweep's effectiveness: adjacent bytes that
    /// lead to the same target set are expected to collapse into one run
    /// rather than being recomputed one byte at a time, so this counter
    /// should stay near "one run per class of bytes" rather than climbing
    /// toward one run per byte. See the note in
    /// `compute_all_transitions_with_context` for why the target sets it
    /// compares are deliberately left uncanonicalised.
    #[cfg(test)]
    pub(crate) context_run_computations: usize,
}

impl LazyDfaContext {
    /// Creates a new context for a given NFA.
    pub fn new(mut nfa: Nfa) -> Self {
        // Precompute epsilon closures for NFAs with many epsilon transitions
        nfa.precompute_epsilon_closures();

        // Check if the NFA has word boundary instructions
        let has_word_boundary = nfa_has_word_boundary(&nfa);

        // Check for anchor instructions
        let (
            has_anchors,
            has_start_anchor,
            has_end_anchor,
            has_multiline_anchors,
            has_multiline_start_anchor,
        ) = nfa_anchor_info(&nfa);

        // Conservative: an accepting state that does not itself carry an end
        // assertion may be reachable without one. Saying "yes" only costs the
        // per-state re-check.
        let has_clean_accept = nfa.states.iter().any(|s| {
            s.is_match
                && !matches!(
                    s.instruction,
                    Some(NfaInstruction::EndOfText) | Some(NfaInstruction::EndOfLine)
                )
        });

        let scratch = ClosureScratch::new(nfa.states.len());

        let mut ctx = Self {
            nfa: Arc::new(nfa),
            scratch,
            states: Vec::new(),
            transitions: Vec::new(),
            transitions_unanchored: Vec::new(),
            last_reach: 0,
            state_map: FxHashMap::default(),
            start: 0,
            cache_limit: DEFAULT_CACHE_LIMIT,
            flush_count: 0,
            search_depth: 0,
            ceiling_exceeded: false,
            has_word_boundary,
            has_anchors,
            has_start_anchor,
            has_end_anchor,
            has_multiline_anchors,
            has_multiline_start_anchor,
            has_clean_accept,
            #[cfg(test)]
            per_byte_computations: 0,
            #[cfg(test)]
            context_run_computations: 0,
        };

        // Create the start state
        let start_seed = [ctx.nfa.start];

        // Whether position 0 is a word boundary depends on the byte that follows
        // it, which is not known while the start state is being built: `\b` holds
        // before "ab" and not before "  ". Leaving it unresolved keeps the
        // pre-assertion states as the seeds, and `compute_transition` re-expands
        // them once the byte — and so the boundary — is known. Claiming `true`
        // here instead baked the assertion in as satisfied, which is what let
        // `\b.` match at 0 in "  ".
        let is_at_boundary = None;

        // Field-split borrows: the scratch is taken mutably while the NFA is
        // taken shared. `ctx` is still a plain local here, so this is the same
        // disjoint-field borrow the rest of the engine uses.
        let start_closure = if has_word_boundary || has_anchors {
            epsilon_closure_with_context(
                &ctx.nfa,
                &mut ctx.scratch,
                &start_seed,
                is_at_boundary,
                Some(PositionContext::start_of_input()),
            )
        } else {
            epsilon_closure_subset(&ctx.nfa, &mut ctx.scratch, &start_seed)
        };

        let start_clean = match_reachable_without_end_assertion(
            &ctx.nfa,
            &mut ctx.scratch,
            &start_seed,
            is_at_boundary,
            Some(PositionContext::start_of_input()),
        );
        ctx.start = get_or_create_state_with_class(
            &mut ctx,
            start_closure,
            CharClass::NonWord,
            start_clean,
        );

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

    /// Returns whether the start anchor specifically is line-mode.
    pub fn has_multiline_start_anchor(&self) -> bool {
        self.has_multiline_start_anchor
    }

    /// Returns whether any accepting state is reachable without an end assertion.
    pub fn has_clean_accept(&self) -> bool {
        self.has_clean_accept
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

    /// The most states the cache may hold while a search defers its flush.
    ///
    /// See [`CACHE_GROWTH_CEILING_FACTOR`].
    pub fn cache_ceiling(&self) -> usize {
        self.cache_limit.saturating_mul(CACHE_GROWTH_CEILING_FACTOR)
    }

    /// The NFA this DFA is built from, shareable without copying it.
    pub(crate) fn nfa_arc(&self) -> Arc<Nfa> {
        Arc::clone(&self.nfa)
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
///
/// Returns `(has_anchors, has_start_anchor, has_end_anchor, has_multiline_anchors,
/// has_multiline_start_anchor)`.
///
/// The last two are not the same question. `has_multiline_anchors` says the
/// pattern contains *some* line anchor; `has_multiline_start_anchor` says the
/// *start* anchor is one. `^a(?m)$` has both a plain `^` and a line-mode `$`, so
/// its valid start positions are still "position 0 only" — treating every line
/// start as valid would make it match the second line of "a\nb".
pub fn nfa_anchor_info(nfa: &Nfa) -> (bool, bool, bool, bool, bool) {
    let mut has_start_anchor = false;
    let mut has_end_anchor = false;
    let mut has_multiline_anchors = false;
    let mut has_multiline_start_anchor = false;

    for state in &nfa.states {
        match &state.instruction {
            Some(NfaInstruction::StartOfText) => has_start_anchor = true,
            Some(NfaInstruction::EndOfText) => has_end_anchor = true,
            Some(NfaInstruction::StartOfLine) => {
                has_start_anchor = true;
                has_multiline_anchors = true;
                has_multiline_start_anchor = true;
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
        has_multiline_start_anchor,
    )
}

/// Computes epsilon closure with optional boundary filtering and position context.
///
/// States are always added to the closure, but epsilon targets are only followed
/// if the assertion check passes. This allows tracking assertion states in the
/// DFA state while preventing traversal of blocked paths.
///
/// - Word boundaries: Follow epsilons only when boundary condition matches
/// - START anchors: Follow epsilons only when at valid start position
/// - END anchors: Always follow epsilons (checked at match time)
///
/// Generic over the seed container so both an [`NfaSubset`] (an interned DFA
/// state's members) and a `&[NfaStateId]`/`&Vec<NfaStateId>` (an
/// accumulation-order, possibly-duplicated run of per-byte targets) can be
/// passed directly, with neither shape needing to be copied into the other
/// first. Duplicates in the seeds are harmless: `scratch.mark` below only
/// enqueues a state the first time it is seen.
///
/// The walk marks a state when it is *enqueued* rather than when it is dequeued;
/// the previous `BTreeSet` walk did the reverse. That does not change the result
/// — whether a state's epsilons are followed depends only on the state and the
/// fixed `is_at_boundary`/`pos_ctx`, never on the path taken to it, so the
/// visited set is the same reachable set either way — and it lets the marked
/// list double as the worklist.
pub fn epsilon_closure_with_context<'a, I>(
    nfa: &Nfa,
    scratch: &mut ClosureScratch,
    seeds: I,
    is_at_boundary: Option<bool>,
    pos_ctx: Option<PositionContext>,
) -> NfaSubset
where
    I: IntoIterator<Item = &'a NfaStateId>,
{
    scratch.begin();
    for &seed in seeds {
        scratch.mark(seed);
    }

    let mut cursor = 0usize;
    // Copied out so the read of the worklist ends before the body marks into
    // it. The cursor only advances, and `mark` only appends, so a state
    // enqueued later in this walk is still reached.
    while let Some(state_id) = scratch.touched.get(cursor).copied() {
        cursor += 1;

        // The state is already in the closure (marked on enqueue) even if its
        // assertion does not hold; that only decides whether its epsilons are
        // followed.
        let state = match nfa.get(state_id) {
            Some(s) => s,
            None => continue,
        };

        // Check if we should follow epsilon transitions from this state.
        // If assertion doesn't match, we include the state but don't follow its epsilons.
        let should_follow_epsilons = match &state.instruction {
            // Word boundaries: follow only when boundary condition matches
            Some(NfaInstruction::WordBoundary) => match is_at_boundary {
                Some(true) => true,   // At boundary, follow
                Some(false) => false, // Not at boundary, don't follow
                None => false,        // Unknown, don't follow
            },
            Some(NfaInstruction::NotWordBoundary) => match is_at_boundary {
                Some(false) => true, // Not at boundary, follow
                Some(true) => false, // At boundary, don't follow
                None => false,       // Unknown, don't follow
            },

            // START anchors: follow only when at valid start position
            Some(NfaInstruction::StartOfText) => match pos_ctx {
                Some(ctx) if ctx.at_start_of_input => true,
                Some(_) => false,
                None => false,
            },
            Some(NfaInstruction::StartOfLine) => match pos_ctx {
                Some(ctx) if ctx.at_start_of_line => true,
                Some(_) => false,
                None => false,
            },

            // END anchors: always follow epsilons (check at match time)
            Some(NfaInstruction::EndOfText) => true,
            Some(NfaInstruction::EndOfLine) => true,

            // No assertion: always follow
            _ => true,
        };

        if should_follow_epsilons {
            for &eps_target in &state.epsilon {
                scratch.mark(eps_target);
            }
        }
    }

    scratch.finish()
}

/// Epsilon closure for patterns with no word boundary and no anchor, as an
/// [`NfaSubset`].
///
/// The assertion-free counterpart of [`epsilon_closure_with_context`], and the
/// exact walk [`Nfa::epsilon_closure`] performs — including its use of the
/// precomputed per-state closures. Those are built by an unconditional DFS that
/// never consults an assertion, which is right for the patterns this is called
/// on and wrong the moment a pattern has one; that is why
/// [`epsilon_closure_with_context`] must never reach for them.
///
/// Exists so this path also produces a canonical subset through the shared
/// scratch, instead of building a `BTreeSet` and then copying it into one.
pub fn epsilon_closure_subset<'a, I>(nfa: &Nfa, scratch: &mut ClosureScratch, seeds: I) -> NfaSubset
where
    I: IntoIterator<Item = &'a NfaStateId>,
{
    scratch.begin();

    if let Some(precomputed) = nfa.epsilon_closures.as_ref() {
        for &seed in seeds {
            if let Some(state_closure) = precomputed.get(seed as usize) {
                for &member in state_closure {
                    scratch.mark(member);
                }
            }
        }
        return scratch.finish();
    }

    for &seed in seeds {
        scratch.mark(seed);
    }

    let mut cursor = 0usize;
    while let Some(state_id) = scratch.touched.get(cursor).copied() {
        cursor += 1;

        if let Some(state) = nfa.get(state_id) {
            for &next in &state.epsilon {
                scratch.mark(next);
            }
        }
    }

    scratch.finish()
}

/// Whether a match is reachable from `seeds` without passing an end assertion.
///
/// Same walk as [`epsilon_closure_with_context`], except it stops at `EndOfText`
/// and `EndOfLine` instead of stepping through them. If a match state is reached
/// anyway, some branch gets there without asking for `$`, and a match in this
/// state stands wherever it ends.
///
/// Generic over the seed container for the same reason as
/// [`epsilon_closure_with_context`]: both an interned [`NfaSubset`] and a
/// `&[NfaStateId]`/`&Vec<NfaStateId>` run of targets (accumulation order,
/// possibly with duplicates) can be passed directly, with no intermediate
/// allocation to bridge between the two.
///
/// Shares the caller's [`ClosureScratch`] as its visited set — it never runs
/// while a closure walk is in flight, and it returns a `bool` that outlives
/// nothing in the buffer. The answer does not depend on the traversal order:
/// it is "does the reachable set contain a match state that is not itself an
/// end assertion", which is a property of the set, not of how it is walked.
pub fn match_reachable_without_end_assertion<'a, I>(
    nfa: &Nfa,
    scratch: &mut ClosureScratch,
    seeds: I,
    is_at_boundary: Option<bool>,
    pos_ctx: Option<PositionContext>,
) -> bool
where
    I: IntoIterator<Item = &'a NfaStateId>,
{
    scratch.begin();
    for &seed in seeds {
        scratch.mark(seed);
    }

    let mut cursor = 0usize;
    while let Some(state_id) = scratch.touched.get(cursor).copied() {
        cursor += 1;

        let Some(state) = nfa.get(state_id) else {
            continue;
        };

        // Checked before `is_match`, not after: when `$` ends the whole pattern
        // the assertion sits *on* the accepting state (`world$`), while in
        // `a|b$` it sits on a predecessor. Either way the state is not a match
        // that skipped the assertion, and it is not stepped through.
        if matches!(
            state.instruction,
            Some(NfaInstruction::EndOfText) | Some(NfaInstruction::EndOfLine)
        ) {
            continue;
        }

        if state.is_match {
            return true;
        }

        let follow = match &state.instruction {
            Some(NfaInstruction::WordBoundary) => is_at_boundary == Some(true),
            Some(NfaInstruction::NotWordBoundary) => is_at_boundary == Some(false),
            Some(NfaInstruction::StartOfText) => pos_ctx.is_some_and(|ctx| ctx.at_start_of_input),
            Some(NfaInstruction::StartOfLine) => pos_ctx.is_some_and(|ctx| ctx.at_start_of_line),
            _ => true,
        };

        if follow {
            for &target in &state.epsilon {
                scratch.mark(target);
            }
        }
    }

    false
}

/// Gets or creates a DFA state for a set of NFA states with a given character class.
pub fn get_or_create_state_with_class(
    ctx: &mut LazyDfaContext,
    nfa_states: NfaSubset,
    prev_class: CharClass,
    match_without_end_assertion: bool,
) -> DfaStateId {
    // The probe key moves `nfa_states` in rather than cloning it: on the hit
    // path (the common case — most bytes land on an already-cached state)
    // the key is simply dropped afterwards. On the miss path the same subset
    // has to be both the map key and the stored `DfaState`'s member list; with
    // `Arc` that second copy is a refcount bump onto the one allocation the
    // closure walk already produced, not a second subset.
    let key = if ctx.has_word_boundary {
        StateKey::WithClass(nfa_states, prev_class)
    } else {
        StateKey::Simple(nfa_states)
    };

    if let Some(&id) = ctx.state_map.get(&key) {
        return id;
    }

    // Miss: recover the owned NFA state set from the probe key so it can be
    // reused below, first for the (rare) post-flush reprobe and then for the
    // stored state and the final insertion key.
    let nfa_states = match key {
        StateKey::WithClass(states, _) | StateKey::Simple(states) => states,
    };

    // Cache full. Flushing invalidates every premultiplied ID in flight, so it
    // is only safe between searches; under a running one the cache grows
    // instead, up to a hard ceiling.
    if ctx.states.len() >= ctx.cache_limit {
        if ctx.search_depth == 0 {
            flush_cache(ctx);
            // The flush wiped the map, so the only key that can still be there
            // is the reinstated start state's — this reprobe exists so a subset
            // that *is* the start state's does not get a second, duplicate DFA
            // state. Both key clones are refcount bumps on the subset already
            // in hand, so the cold path costs no copying.
            let reprobe_key = if ctx.has_word_boundary {
                StateKey::WithClass(Arc::clone(&nfa_states), prev_class)
            } else {
                StateKey::Simple(Arc::clone(&nfa_states))
            };
            if let Some(&id) = ctx.state_map.get(&reprobe_key) {
                return id;
            }
        } else if ctx.states.len() >= ctx.cache_ceiling() {
            // The search cannot be completed. The returned ID keeps the walk
            // well-formed until it unwinds; the flag makes the entry point
            // report the give-up instead of the partial scan's answer.
            ctx.ceiling_exceeded = true;
            return ctx.start;
        }
    }

    // Check if this is a match state
    let is_match = nfa_states
        .iter()
        .any(|&s| ctx.nfa.get(s).map(|state| state.is_match).unwrap_or(false));

    let state_index = ctx.states.len();
    let premul_id = (state_index as u32) * STRIDE;

    // Key and stored state share one allocation: `Arc::clone` is a refcount
    // bump, not a second subset. Which is also why neither side may ever be
    // mutated in place — the map key is already hashed.
    let key = if ctx.has_word_boundary {
        StateKey::WithClass(Arc::clone(&nfa_states), prev_class)
    } else {
        StateKey::Simple(Arc::clone(&nfa_states))
    };

    ctx.states.push(DfaState::new(
        nfa_states,
        is_match,
        prev_class,
        match_without_end_assertion,
    ));
    ctx.transitions
        .resize(ctx.transitions.len() + STRIDE as usize, UNKNOWN_STATE);
    if !ctx.transitions_unanchored.is_empty() {
        ctx.transitions_unanchored
            .resize(ctx.transitions.len(), UNKNOWN_STATE);
    }
    ctx.state_map.insert(key, premul_id);

    premul_id
}

/// Flushes the cache, keeping only the start state.
pub fn flush_cache(ctx: &mut LazyDfaContext) {
    // Lifted out whole rather than field by field: with the subset behind an
    // `Arc`, cloning the whole `DfaState` copies three scalars and bumps one
    // refcount, and the reinstated state is then bit-for-bit the one that was
    // there — no chance of reassembling it with a field left behind.
    let start_index = state_index(ctx.start);
    let Some(start_state) = ctx.states.get(start_index).cloned() else {
        // `ctx.start` always names a live state, so this cannot happen. If it
        // somehow did, clearing the cache would leave nothing to rebuild the
        // start from, so leave it exactly as it is — and do not report a flush,
        // since callers use `flush_count` to detect that state IDs were
        // renumbered.
        return;
    };

    ctx.flush_count += 1;

    ctx.states.clear();
    ctx.transitions.clear();
    ctx.transitions_unanchored.clear();
    ctx.state_map.clear();

    let key = if ctx.has_word_boundary {
        StateKey::WithClass(Arc::clone(&start_state.nfa_states), start_state.prev_class)
    } else {
        StateKey::Simple(Arc::clone(&start_state.nfa_states))
    };
    ctx.states.push(start_state);
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
