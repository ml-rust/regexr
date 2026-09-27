//! Shared types for Shift-Or engine.
//!
//! Contains the ShiftOr data structure used by both interpreter and JIT.

use crate::hir::{Hir, HirExpr};
use crate::nfa::{
    compile_glushkov, compile_glushkov_wide, BitSet256, GlushkovNfa, GlushkovWideNfa,
    MAX_POSITIONS, MAX_POSITIONS_WIDE,
};

/// A compiled Shift-Or pattern.
///
/// This is a data structure that holds the precomputed masks and follow sets
/// for the Shift-Or (Bitap) algorithm. The actual matching is performed by
/// either the interpreter or JIT.
///
/// **CRITICAL**: This implementation uses Glushkov NFA (epsilon-free), NOT Thompson NFA.
/// Thompson's epsilon-transitions break the 1-shift = 1-byte invariant.
///
/// Unlike classic Shift-Or which assumes linear position progression (i -> i+1),
/// this implementation uses explicit follow sets from Glushkov construction to
/// handle patterns like `a.*b` where nullable subexpressions create non-linear
/// transitions.
///
/// ## Limitations
///
/// ShiftOr does NOT support:
/// - Anchors (`^`, `$`)
/// - Word boundaries (`\b`, `\B`) - use LazyDFA instead
/// - Backreferences - use PikeVM or BacktrackingVM instead
/// - Lookaround - use PikeVM instead
/// - Non-greedy quantifiers (`.*?`, `.+?`) - Glushkov doesn't preserve match preference
/// - Patterns with more than 64 positions
#[derive(Debug, Clone)]
pub struct ShiftOr {
    /// Bit masks for each byte value.
    /// mask[b] has bit i cleared (0) if position i can transition on byte b.
    /// (Shift-Or uses inverted logic: 0 = "can be in this state")
    pub(crate) masks: [u64; 256],
    /// Accept state mask (inverted: 0 bits = accepting positions).
    pub(crate) accept: u64,
    /// First set: positions that can start a match.
    pub(crate) first: u64,
    /// Follow sets: follow[i] indicates which positions can follow position i.
    pub(crate) follow: Vec<u64>,
    /// Whether the pattern can match empty string.
    pub(crate) nullable: bool,
    /// Number of positions.
    pub(crate) position_count: usize,
    /// Whether the pattern has a leading word boundary (\b at start).
    pub(crate) has_leading_word_boundary: bool,
    /// Whether the pattern has a trailing word boundary (\b at end).
    pub(crate) has_trailing_word_boundary: bool,
    /// Whether the pattern has a start anchor (^).
    pub(crate) has_start_anchor: bool,
    /// Whether the pattern has an end anchor ($).
    pub(crate) has_end_anchor: bool,
    /// Set when the whole pattern is one repeated byte class, which a scan
    /// answers directly. See [`ClassRun`].
    pub(crate) class_run: Option<ClassRun>,
    /// Follow sets of the automaton read backwards: `reverse_follow[q]` holds
    /// every position `p` whose follow set contains `q`. See
    /// [`ShiftOr::leftmost_start`].
    pub(crate) reverse_follow: Vec<u64>,
}

/// A pattern that is nothing but a repeated byte class — `\w+`, `\d+`,
/// `[a-z]{2,}`.
///
/// Shift-Or answers such a pattern with two bit-parallel passes per match: one
/// to bound the search and one anchored at the start it settles on. Both are
/// per-byte loops over a state word, and neither learns anything a membership
/// test would not: the match is the maximal run of class members beginning at
/// the first member at or after the resume point.
///
/// `\w+` is the tokenizer pattern, and the one shape where this costs the most —
/// a word-dense haystack makes every byte part of some match, so the two passes
/// run over the whole input.
#[derive(Debug, Clone)]
pub(crate) struct ClassRun {
    /// `table[byte]` is non-zero for a member of the class.
    table: Box<[u8; 256]>,
    /// Shortest run that counts as a match; always at least one, because a
    /// nullable pattern matches everywhere and is not this shape.
    min: usize,
    /// Longest run to take, for `{n,m}`.
    max: Option<usize>,
}

/// Whether a pattern is one repeated byte class, which the interpreted
/// [`ShiftOr`] answers with a scan.
///
/// Engine selection needs this before it has built anything, so that a JIT build
/// does not compile the bit-parallel automaton the scan exists to replace.
pub fn is_class_run_shape(hir: &Hir) -> bool {
    ClassRun::from_hir(hir).is_some()
}

impl ClassRun {
    /// Recognises the shape in an HIR, or returns `None`.
    ///
    /// Deliberately narrow: one greedy repetition of one byte class, no capture
    /// to fill, no assertion to evaluate, and a minimum of at least one. A
    /// negated Unicode class does not qualify — it lowers to an alternation of
    /// an ASCII class and a UTF-8 trie, which is not one byte per iteration.
    fn from_hir(hir: &Hir) -> Option<Self> {
        if hir.props.capture_count > 0 {
            return None;
        }
        let HirExpr::Repeat(repeat) = &hir.expr else {
            return None;
        };
        let HirExpr::Class(class) = &repeat.expr else {
            return None;
        };
        if !repeat.greedy || repeat.min == 0 {
            return None;
        }
        Some(Self {
            table: Box::new(crate::literal::byte_class_set(&class.ranges, class.negated)),
            min: repeat.min as usize,
            max: repeat.max.map(|max| max as usize),
        })
    }

    #[inline]
    fn contains(&self, byte: u8) -> bool {
        self.table.get(byte as usize).is_some_and(|m| *m != 0)
    }

    /// The leftmost run of at least `min` members starting at or after `from`.
    ///
    /// A run shorter than the minimum is not a match, and no start inside it can
    /// begin a longer one, so the search resumes past it rather than at the next
    /// byte.
    fn find_from(&self, input: &[u8], from: usize) -> Option<(usize, usize)> {
        let mut pos = from;
        loop {
            while !self.contains(*input.get(pos)?) {
                pos += 1;
            }
            let start = pos;
            let limit = self.max.map_or(input.len(), |max| {
                input.len().min(start.saturating_add(max))
            });
            while pos < limit && input.get(pos).is_some_and(|&b| self.contains(b)) {
                pos += 1;
            }
            if pos - start >= self.min {
                return Some((start, pos));
            }
            // Resume past the whole run rather than one byte into it: a start
            // inside a run that was already too short can only produce a shorter
            // one. Rescanning from every position instead stays linear in the
            // input — a rejected run is shorter than `min`, and `min` is bounded
            // by Shift-Or's position limit — but it is still `min` times the
            // work, for an answer that cannot change.
            while input.get(pos).is_some_and(|&b| self.contains(b)) {
                pos += 1;
            }
        }
    }
}

/// The walked-byte budget of a start-by-start search from `from`, before the
/// linear-time search takes over. Shares [`crate::dfa::lazy::scan_budget`]
/// with the lazy DFA's rule: attempts that fail near their start never reach
/// it. Shift-Or has no bounded-match-length hint to sharpen it with.
fn scan_budget(input: &[u8], from: usize) -> usize {
    crate::dfa::lazy::scan_budget(input.len(), from, None)
}

/// Follow sets of a position automaton read backwards: entry `q` holds every
/// position `p` whose follow set contains `q`.
fn reverse_follow(follow: &[u64]) -> Vec<u64> {
    let mut reversed = vec![0u64; follow.len()];
    for (p, &targets) in follow.iter().enumerate() {
        let mut live = targets;
        while live != 0 {
            let q = live.trailing_zeros() as usize;
            if let Some(entry) = reversed.get_mut(q) {
                *entry |= 1u64 << p;
            }
            live &= live - 1;
        }
    }
    reversed
}

/// [`reverse_follow`] for the 256-position automaton.
fn reverse_follow_wide(follow: &[BitSet256]) -> Vec<BitSet256> {
    let mut reversed = vec![BitSet256::empty(); follow.len()];
    for (p, targets) in follow.iter().enumerate() {
        for (word_idx, &word) in targets.parts.iter().enumerate() {
            let mut live = word;
            while live != 0 {
                let q = word_idx * 64 + live.trailing_zeros() as usize;
                if let Some(entry) = reversed.get_mut(q) {
                    entry.set(p);
                }
                live &= live - 1;
            }
        }
    }
    reversed
}

impl ShiftOr {
    /// Tries to compile an HIR into a Shift-Or matcher.
    /// Returns None if the pattern is not suitable for Shift-Or.
    pub fn from_hir(hir: &Hir) -> Option<Self> {
        // Skip patterns with special features that can't be handled
        // Anchors (^, $), backrefs, lookarounds, and word boundaries require different engines.
        // Word boundaries (\b, \B) are complex to handle correctly in shift-or;
        // LazyDFA handles them properly with character-class augmented states.
        // Non-greedy quantifiers (.*?, .+?) require tracking match preference which
        // Glushkov NFA doesn't preserve - use TaggedNFA or PikeVM instead.
        if hir.props.has_backrefs
            || hir.props.has_lookaround
            || hir.props.has_anchors
            || hir.props.has_word_boundary
            || hir.props.has_non_greedy
        {
            return None;
        }

        // Build Glushkov NFA (epsilon-free)
        let glushkov = compile_glushkov(hir)?;

        let mut matcher = Self::from_glushkov_with_boundaries(&glushkov, false, false)?;
        matcher.class_run = ClassRun::from_hir(hir);
        Some(matcher)
    }

    /// Tries to compile an HIR with anchors into a Shift-Or matcher.
    /// Supports non-multiline ^ and $ anchors.
    /// Returns None if the pattern is not suitable for Shift-Or.
    pub fn from_hir_with_anchors(hir: &Hir) -> Option<Self> {
        // Skip patterns with unsupported features
        if hir.props.has_backrefs
            || hir.props.has_lookaround
            || hir.props.has_word_boundary
            || hir.props.has_non_greedy
        {
            return None;
        }

        // Build Glushkov NFA (epsilon-free) - anchors are treated as empty
        let glushkov = compile_glushkov(hir)?;

        // Detect anchor types from HIR properties
        let has_start_anchor = hir.props.has_start_anchor;
        let has_end_anchor = hir.props.has_end_anchor;

        Self::from_glushkov_with_options(&glushkov, false, false, has_start_anchor, has_end_anchor)
    }

    /// Creates a Shift-Or matcher from a Glushkov NFA.
    pub fn from_glushkov(nfa: &GlushkovNfa) -> Option<Self> {
        Self::from_glushkov_with_options(nfa, false, false, false, false)
    }

    /// Creates a Shift-Or matcher from a Glushkov NFA with word boundary info.
    fn from_glushkov_with_boundaries(
        nfa: &GlushkovNfa,
        has_leading_word_boundary: bool,
        has_trailing_word_boundary: bool,
    ) -> Option<Self> {
        Self::from_glushkov_with_options(
            nfa,
            has_leading_word_boundary,
            has_trailing_word_boundary,
            false,
            false,
        )
    }

    /// Creates a Shift-Or matcher from a Glushkov NFA with all options.
    fn from_glushkov_with_options(
        nfa: &GlushkovNfa,
        has_leading_word_boundary: bool,
        has_trailing_word_boundary: bool,
        has_start_anchor: bool,
        has_end_anchor: bool,
    ) -> Option<Self> {
        if nfa.position_count > MAX_POSITIONS || nfa.position_count == 0 {
            return None;
        }

        let masks = nfa.build_shift_or_masks();
        let accept = nfa.build_accept_mask();

        Some(Self {
            masks,
            accept,
            first: nfa.first,
            follow: nfa.follow.clone(),
            nullable: nfa.nullable,
            position_count: nfa.position_count,
            has_leading_word_boundary,
            has_trailing_word_boundary,
            has_start_anchor,
            has_end_anchor,
            class_run: None,
            reverse_follow: reverse_follow(&nfa.follow),
        })
    }

    /// Whether this pattern is answered by a class-run scan.
    ///
    /// The scan lives in the interpreter, so JIT-compiling the bit-parallel
    /// automaton instead would be a downgrade.
    #[inline]
    pub fn has_class_run(&self) -> bool {
        self.class_run.is_some()
    }

    /// Returns true if this pattern has word boundaries.
    /// Note: ShiftOr no longer accepts patterns with word boundaries,
    /// so this always returns false for valid ShiftOr instances.
    #[inline]
    pub fn has_word_boundary(&self) -> bool {
        self.has_leading_word_boundary || self.has_trailing_word_boundary
    }

    /// Returns the number of positions.
    pub fn state_count(&self) -> usize {
        self.position_count
    }

    /// Returns the masks table.
    pub fn masks(&self) -> &[u64; 256] {
        &self.masks
    }

    /// Returns the accept mask.
    pub fn accept(&self) -> u64 {
        self.accept
    }

    /// Returns the first set.
    pub fn first(&self) -> u64 {
        self.first
    }

    /// Returns the follow sets.
    pub fn follow(&self) -> &[u64] {
        &self.follow
    }

    /// Returns whether the pattern is nullable.
    pub fn is_nullable(&self) -> bool {
        self.nullable
    }

    /// Returns whether there's a leading word boundary.
    pub fn has_leading_word_boundary(&self) -> bool {
        self.has_leading_word_boundary
    }

    /// Returns whether there's a trailing word boundary.
    pub fn has_trailing_word_boundary(&self) -> bool {
        self.has_trailing_word_boundary
    }

    // ========================================================================
    // Convenience matching methods (delegate to interpreter)
    // ========================================================================

    /// Returns true if the pattern matches anywhere in the input.
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.find(input).is_some()
    }

    /// Finds the first match, returning (start, end).
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        // For start anchor: only try matching at position 0
        if self.has_start_anchor {
            if let Some(end) = self.match_at(input, 0) {
                // For end anchor: only accept if match ends at input end
                if self.has_end_anchor && !crate::nfa::at_end_or_before_final_newline(input, end) {
                    return None;
                }
                return Some((0, end));
            }
            // If pattern is nullable and has both anchors, empty match at 0 only if input is empty
            if self.nullable && (!self.has_end_anchor || input.is_empty()) {
                return Some((0, 0));
            }
            return None;
        }

        // Try matching at each position, preferring longest match (greedy).
        // `scan_limit` first rules out "no match anywhere" in a single pass and
        // otherwise caps how far this scan has to walk.
        let scan_end = self.scan_limit(input, 0)?;
        let found = if self.has_end_anchor {
            self.find_end_anchored(input, 0)
        } else {
            self.find_bounded(input, 0, scan_end)
        };
        if found.is_some() {
            return found;
        }

        // If pattern is nullable and no non-empty match found, return empty match at 0
        if self.nullable && !self.has_end_anchor {
            return Some((0, 0));
        }

        None
    }

    /// Finds a match starting at or after the given position.
    /// Returns (start, end) if found.
    pub fn find_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos > input.len() {
            return None;
        }

        if let Some(ref run) = self.class_run {
            return run.find_from(input, pos);
        }

        // For start anchor: can only match at position 0
        if self.has_start_anchor && pos > 0 {
            return None;
        }

        let search_start = if self.has_start_anchor { 0 } else { pos };

        // Try matching at each position from pos. `scan_limit` first rules out
        // "no match anywhere" in a single pass and otherwise caps how far this
        // scan has to walk.
        let scan_end = self.scan_limit(input, search_start)?;
        if self.has_end_anchor && !self.has_start_anchor {
            return self.find_end_anchored(input, search_start);
        }
        if !self.has_start_anchor {
            return self.find_bounded(input, search_start, scan_end);
        }
        for start in search_start..=scan_end {
            if let Some(end) = self.match_at(input, start) {
                // For end anchor: only accept if match ends at input end
                if self.has_end_anchor && !crate::nfa::at_end_or_before_final_newline(input, end) {
                    if self.has_start_anchor {
                        // Can't match anywhere else
                        return None;
                    }
                    continue;
                }
                return Some((start, end));
            }
            // For start anchor: only one position to try
            if self.has_start_anchor {
                break;
            }
        }
        None
    }

    /// One bit-parallel pass over `input[from..]` keeping a fresh start live at
    /// every position; returns the earliest position at which *some* non-empty
    /// match ends, or `None` if no match begins at or after `from`.
    ///
    /// This is the classic unanchored Shift-Or scan. Because every start is live
    /// at once it answers "is there a match at all?" in a single pass, where the
    /// anchored `match_at` needs one pass per start position.
    ///
    /// It deliberately reports only the earliest **end**. The state is a bitmask
    /// of reachable positions and records nothing about *which* start reached
    /// them, so it cannot express the leftmost-first preference between two
    /// starts — `abc|b` on "abc" ends a match at 2 via the `b` branch, while the
    /// leftmost match is `abc` spanning 0..3. What it does give is a bound: the
    /// match ending at `e` starts at or before `e`, and the leftmost start is no
    /// later than that start, so `s* <= e`. A caller can therefore stop its
    /// anchored scan at `e` instead of walking to the end of the input.
    pub(crate) fn earliest_match_end(&self, input: &[u8], from: usize) -> Option<usize> {
        // Inverted logic throughout, as in `match_at`: bit i == 0 means position
        // i is active. All 1s = nothing reached yet.
        let mut state = !0u64;

        for (i, &byte) in input[from..].iter().enumerate() {
            // Positions reachable from the active set, unioned with First — that
            // unconditional injection is what keeps a match starting *here* live
            // alongside every earlier one, and is the whole difference from the
            // anchored scan.
            let mut reachable = self.first;
            let mut active = !state;
            while active != 0 {
                let pos = active.trailing_zeros() as usize;
                reachable |= self.follow[pos];
                active &= active - 1;
            }

            state = (!reachable) | self.masks[byte as usize];

            if (state | self.accept) != !0u64 {
                return Some(from + i + 1);
            }
        }

        None
    }

    /// How far an anchored scan starting at `search_start` has to walk.
    ///
    /// Returns `None` when a single unanchored pass proves there is no match to
    /// find, letting the caller skip the scan entirely — that is what turns the
    /// no-match case from one pass per position into one pass overall.
    ///
    /// The pass is skipped where it cannot pay off: a start anchor means the scan
    /// tries a single position anyway, and a nullable pattern matches empty at
    /// the first position tried.
    pub(crate) fn scan_limit(&self, input: &[u8], search_start: usize) -> Option<usize> {
        if self.has_start_anchor || self.nullable {
            return Some(input.len());
        }
        match self.earliest_match_end(input, search_start) {
            None => None,
            // An end-anchored pattern may reject this match and keep looking, so
            // only the existence check carries over — the bound does not.
            Some(_) if self.has_end_anchor => Some(input.len()),
            Some(end) => Some(end),
        }
    }

    /// Tries starts from `from` through `scan_end` for an unanchored pattern.
    ///
    /// Trying one start at a time is right while the attempts give up near
    /// where they began. It collapses when they do not: `a[a-z]*!` over a long
    /// run of `a` walks to the end of the run from every start. So the attempts
    /// are metered, and once they have collectively walked several times the
    /// input, [`ShiftOr::find_linear`] takes over from the next start.
    fn find_bounded(&self, input: &[u8], from: usize, scan_end: usize) -> Option<(usize, usize)> {
        let budget = scan_budget(input, from);
        let mut walked = 0usize;

        for start in from..=scan_end {
            let (end, reach) = self.match_at_reach(input, start);
            if let Some(end) = end {
                return Some((start, end));
            }
            walked += reach.saturating_sub(start);
            if walked > budget {
                return self.find_linear(input, start + 1);
            }
        }
        None
    }

    /// Scans starts from `from` for a match that ends where `$` allows.
    ///
    /// Metered like [`ShiftOr::find_bounded`]: `(a+)+$` over `"aaaa…"` runs
    /// every attempt to the end of the input and never accepts.
    fn find_end_anchored(&self, input: &[u8], from: usize) -> Option<(usize, usize)> {
        let budget = scan_budget(input, from);
        let mut walked = 0usize;

        for start in from..=input.len() {
            let (end, reach) = self.match_at_reach(input, start);
            if let Some(end) = end {
                if crate::nfa::at_end_or_before_final_newline(input, end) {
                    return Some((start, end));
                }
            }
            walked += reach.saturating_sub(start);
            if walked > budget {
                return self.find_linear(input, start + 1);
            }
        }
        None
    }

    /// The leftmost match starting at or after `from`, in time linear in the
    /// input: one backward pass finds the start (see
    /// [`ShiftOr::leftmost_start`]) and one anchored pass from it the longest
    /// end.
    ///
    /// Leftmost-longest is the leftmost-first match for every pattern Shift-Or
    /// runs (see `automata_match_like_pikevm`). With an end anchor the start is
    /// that of a match ending where `$` holds, and the longest end from it is
    /// then at or past that point, so it satisfies `$` as well.
    pub(crate) fn find_linear(&self, input: &[u8], from: usize) -> Option<(usize, usize)> {
        if from > input.len() {
            return None;
        }
        if self.has_start_anchor {
            return if from == 0 {
                self.try_match_at(input, 0)
            } else {
                None
            };
        }
        let start = self.leftmost_start(input, from)?;
        let end = self.match_at(input, start)?;
        if self.has_end_anchor && !crate::nfa::at_end_or_before_final_newline(input, end) {
            return None;
        }
        Some((start, end))
    }

    /// The smallest position at or after `from` where a match begins, found in
    /// one right-to-left pass.
    ///
    /// The pass runs the position automaton backwards: a thread enters at a
    /// last position wherever a match may end (every position, or only where
    /// `$` holds), steps through [`ShiftOr::reverse_follow`], and has found a
    /// match start when it stands on a first position. Every end is live at
    /// once, so one pass answers for every start.
    pub(crate) fn leftmost_start(&self, input: &[u8], from: usize) -> Option<usize> {
        let may_end = |end: usize| {
            !self.has_end_anchor || crate::nfa::at_end_or_before_final_newline(input, end)
        };
        let last = !self.accept;
        let mut leftmost = None;
        if self.nullable && may_end(input.len()) {
            leftmost = Some(input.len());
        }
        // Positive logic here: bit `p` set means the byte just read was
        // matched by position `p`.
        let mut active = 0u64;
        for pos in (from..input.len()).rev() {
            let mut reachable = if may_end(pos + 1) { last } else { 0 };
            let mut live = active;
            while live != 0 {
                let p = live.trailing_zeros() as usize;
                reachable |= self.reverse_follow.get(p).copied().unwrap_or(0);
                live &= live - 1;
            }
            active = reachable & !self.masks[input[pos] as usize];
            if active & self.first != 0 || (self.nullable && may_end(pos)) {
                leftmost = Some(pos);
            }
        }
        leftmost
    }

    /// Tries to match at exactly the given position.
    /// Returns (start, end) if matched, None otherwise.
    /// Use this when you know the match should start at exactly `pos` (e.g., from a prefilter).
    pub fn try_match_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        self.try_match_at_reach(input, pos).0
    }

    /// [`ShiftOr::try_match_at`], also reporting how far the attempt walked,
    /// for callers that meter attempts at many positions.
    pub(crate) fn try_match_at_reach(
        &self,
        input: &[u8],
        pos: usize,
    ) -> (Option<(usize, usize)>, usize) {
        // For start anchor: only position 0 can match
        if self.has_start_anchor && pos != 0 {
            return (None, pos);
        }
        match self.match_at_reach(input, pos) {
            (Some(end), reach) => {
                // For end anchor: must match to end of input
                if self.has_end_anchor && !crate::nfa::at_end_or_before_final_newline(input, end) {
                    return (None, reach);
                }
                (Some((pos, end)), reach)
            }
            (None, reach) => (None, reach),
        }
    }

    /// Attempts to match at a specific position.
    fn match_at(&self, input: &[u8], start: usize) -> Option<usize> {
        self.match_at_reach(input, start).0
    }

    /// [`ShiftOr::match_at`], also reporting how far the scan walked before the
    /// state went empty. The start-by-start loops meter attempts with it.
    ///
    /// Inlined so the pair stays in registers instead of coming back through
    /// memory on every attempt.
    #[inline(always)]
    fn match_at_reach(&self, input: &[u8], start: usize) -> (Option<usize>, usize) {
        if start > input.len() {
            return (None, start);
        }

        // Track the last match position found
        let mut last_match = None;

        // Check if nullable (empty match)
        if self.nullable {
            last_match = Some(start);
        }

        // State tracking using inverted logic:
        // - bit i = 0 means we've reached position i (active)
        // - bit i = 1 means we haven't reached position i (inactive)
        //
        // Initial state: all 1s (no positions reached yet)
        let mut state = !0u64;

        for (i, &byte) in input[start..].iter().enumerate() {
            let byte_mask = self.masks[byte as usize];

            if i == 0 {
                // First byte: can only start at positions in First set
                // ~first gives us 0s at First positions, 1s elsewhere
                // Then apply byte mask to filter positions that don't accept this byte
                state = (!self.first) | byte_mask;
            } else {
                // Subsequent bytes: use follow sets for transitions
                let mut active = !state; // Flip: 1 = active, 0 = inactive

                // Compute union of follow sets for all active positions
                let mut reachable = 0u64;
                while active != 0 {
                    let pos = active.trailing_zeros() as usize;
                    reachable |= self.follow[pos];
                    active &= active - 1; // Clear lowest set bit
                }

                // Invert back to Shift-Or convention (0 = active)
                // Then apply byte mask (positions that don't accept byte become 1)
                state = (!reachable) | byte_mask;
            }

            // Check for match: if any accepting position is reached (bit is 0)
            if (state | self.accept) != !0u64 {
                last_match = Some(start + i + 1);
            }

            // If all bits are 1, no possible match from this starting point
            if state == !0u64 {
                return (last_match, start + i + 1);
            }
        }

        (last_match, input.len())
    }
}

/// Checks if an HIR is suitable for Shift-Or.
/// Whether `expr` can match the empty string. Such patterns have a leftmost
/// zero-width match that bit-parallel Shift-Or can't represent.
use crate::hir::matches_empty as hir_is_nullable;

/// Whether `hir` can be matched by the (narrow) Shift-Or engine.
pub fn is_shift_or_compatible(hir: &Hir) -> bool {
    // Backrefs, lookarounds, and word boundaries require different engines.
    // Word boundaries (\b, \B) are complex to handle correctly in shift-or;
    // LazyDFA handles them properly with character-class augmented states.
    // Non-greedy quantifiers (.*?, .+?) require tracking match preference which
    // Glushkov NFA doesn't preserve - use TaggedNFA or PikeVM instead.
    // Multiline anchors ((?m)^, (?m)$) need position-aware matching via LazyDFA.
    if hir.props.has_backrefs
        || hir.props.has_lookaround
        || hir.props.has_multiline_anchors
        || hir.props.has_word_boundary
        || hir.props.has_non_greedy
    {
        return false;
    }

    // Nullable patterns (those that can match the empty string, e.g. `a*`, ` *`,
    // `a*|b`) are not Shift-Or compatible: the Glushkov automaton has no empty-match
    // representation, so Shift-Or skips to the first literal occurrence and misses
    // the leftmost zero-width match. Route them to LazyDFA, which matches empty.
    if hir_is_nullable(&hir.expr) {
        return false;
    }

    // Try to build Glushkov NFA to check position count
    compile_glushkov(hir)
        .map(|nfa| nfa.position_count <= MAX_POSITIONS && nfa.position_count > 0)
        .unwrap_or(false)
}

// ============================================================================
// Wide Shift-Or (supports up to 256 positions)
// ============================================================================

/// A compiled Wide Shift-Or pattern supporting up to 256 positions.
///
/// Uses `[u64; 4]` (BitSet256) for state vectors instead of `u64`,
/// allowing patterns with 65-256 character positions to use the efficient
/// bit-parallel Shift-Or algorithm instead of falling back to PikeVM.
///
/// Performance notes:
/// - For patterns with ≤64 positions, use `ShiftOr` (faster due to single u64)
/// - For patterns with 65-256 positions, use `ShiftOrWide`
/// - For patterns with >256 positions, use LazyDFA or PikeVM
#[derive(Debug)]
pub struct ShiftOrWide {
    /// Bit masks for each byte value (256-bit wide).
    pub(crate) masks: Box<[BitSet256; 256]>,
    /// Accept state mask (inverted: 0 bits = accepting positions).
    pub(crate) accept: BitSet256,
    /// First set: positions that can start a match.
    pub(crate) first: BitSet256,
    /// Follow sets: follow[i] indicates which positions can follow position i.
    pub(crate) follow: Vec<BitSet256>,
    /// Whether the pattern can match empty string.
    pub(crate) nullable: bool,
    /// Number of positions.
    pub(crate) position_count: usize,
    /// Follow sets of the automaton read backwards; see
    /// [`ShiftOr::reverse_follow`].
    pub(crate) reverse_follow: Vec<BitSet256>,
}

impl ShiftOrWide {
    /// Tries to compile an HIR into a Wide Shift-Or matcher.
    /// Returns None if the pattern is not suitable.
    pub fn from_hir(hir: &Hir) -> Option<Self> {
        // Skip patterns with special features that can't be handled
        if hir.props.has_backrefs
            || hir.props.has_lookaround
            || hir.props.has_anchors
            || hir.props.has_word_boundary
            || hir.props.has_non_greedy
        {
            return None;
        }

        // Build Wide Glushkov NFA
        let glushkov = compile_glushkov_wide(hir)?;

        Self::from_glushkov(&glushkov)
    }

    /// Creates a Wide Shift-Or matcher from a Wide Glushkov NFA.
    pub fn from_glushkov(nfa: &GlushkovWideNfa) -> Option<Self> {
        if nfa.position_count > MAX_POSITIONS_WIDE || nfa.position_count == 0 {
            return None;
        }

        let masks = Box::new(nfa.build_shift_or_masks());
        let accept = nfa.build_accept_mask();

        Some(Self {
            masks,
            accept,
            first: nfa.first,
            follow: nfa.follow.clone(),
            nullable: nfa.nullable,
            position_count: nfa.position_count,
            reverse_follow: reverse_follow_wide(&nfa.follow),
        })
    }

    /// Returns the number of positions.
    pub fn state_count(&self) -> usize {
        self.position_count
    }

    /// Returns whether the pattern is nullable.
    pub fn is_nullable(&self) -> bool {
        self.nullable
    }

    // ========================================================================
    // Convenience matching methods (delegate to interpreter)
    // ========================================================================

    /// Returns true if the pattern matches anywhere in the input.
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.find(input).is_some()
    }

    /// Finds the first match, returning (start, end).
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        // Try matching at each position, preferring longest match (greedy).
        // `scan_limit` first rules out "no match anywhere" in a single pass and
        // otherwise caps how far this scan has to walk.
        // No match to find when this is `None`. A nullable pattern still
        // matches empty, and `scan_limit` never reports `None` for one.
        let scan_end = self.scan_limit(input, 0)?;
        if let Some(found) = self.find_bounded(input, 0, scan_end) {
            return Some(found);
        }

        // If pattern is nullable and no non-empty match found, return empty match at 0
        if self.nullable {
            return Some((0, 0));
        }

        None
    }

    /// Finds a match starting at or after the given position.
    pub fn find_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos > input.len() {
            return None;
        }

        let scan_end = self.scan_limit(input, pos)?;
        self.find_bounded(input, pos, scan_end)
    }

    /// Tries starts from `from` through `scan_end`, metered as in
    /// `ShiftOr::find_bounded`; past the budget, [`ShiftOrWide::find_linear`]
    /// takes over from the next start.
    fn find_bounded(&self, input: &[u8], from: usize, scan_end: usize) -> Option<(usize, usize)> {
        let budget = scan_budget(input, from);
        let mut walked = 0usize;

        for start in from..=scan_end {
            let (end, reach) = self.match_at_reach(input, start);
            if let Some(end) = end {
                return Some((start, end));
            }
            walked += reach.saturating_sub(start);
            if walked > budget {
                return self.find_linear(input, start + 1);
            }
        }
        None
    }

    /// The leftmost match starting at or after `from`, in time linear in the
    /// input; the 256-position counterpart of `ShiftOr::find_linear`.
    fn find_linear(&self, input: &[u8], from: usize) -> Option<(usize, usize)> {
        if from > input.len() {
            return None;
        }
        let start = self.leftmost_start(input, from)?;
        self.match_at(input, start).map(|end| (start, end))
    }

    /// The smallest position at or after `from` where a match begins, found in
    /// one right-to-left pass; see `ShiftOr::leftmost_start`.
    fn leftmost_start(&self, input: &[u8], from: usize) -> Option<usize> {
        let last = self.accept.complement();
        let mut leftmost = None;
        if self.nullable {
            leftmost = Some(input.len());
        }
        // Positive logic: bit `p` set means the byte just read was matched by
        // position `p`.
        let mut active = BitSet256::empty();
        for pos in (from..input.len()).rev() {
            let mut reachable = last;
            for (word_idx, &word) in active.parts.iter().enumerate() {
                let mut live = word;
                while live != 0 {
                    let p = word_idx * 64 + live.trailing_zeros() as usize;
                    if let Some(&targets) = self.reverse_follow.get(p) {
                        reachable.union_assign(targets);
                    }
                    live &= live - 1;
                }
            }
            active = reachable.intersection(self.masks[input[pos] as usize].complement());
            if self.nullable || !active.intersection(self.first).is_empty() {
                leftmost = Some(pos);
            }
        }
        leftmost
    }

    /// One bit-parallel pass over `input[from..]` keeping a fresh start live at
    /// every position; returns the earliest position at which *some* non-empty
    /// match ends, or `None` if no match begins at or after `from`.
    ///
    /// The 256-bit counterpart of `ShiftOr::earliest_match_end` — see there for
    /// why this reports the earliest *end* and how that bounds the anchored scan
    /// (`s* <= e`).
    fn earliest_match_end(&self, input: &[u8], from: usize) -> Option<usize> {
        let mut state = BitSet256::all_ones();

        for (i, &byte) in input[from..].iter().enumerate() {
            // Reachable from the active set, unioned with First — the
            // unconditional injection keeps a match starting *here* live
            // alongside every earlier one.
            let mut reachable = self.first;
            let active = state.complement();
            for word_idx in 0..4 {
                let mut word = active.parts[word_idx];
                while word != 0 {
                    let pos = word_idx * 64 + word.trailing_zeros() as usize;
                    if pos < self.follow.len() {
                        reachable.union_assign(self.follow[pos]);
                    }
                    word &= word - 1;
                }
            }

            state = reachable.complement().union(self.masks[byte as usize]);

            if !state.union(self.accept).is_all_ones() {
                return Some(from + i + 1);
            }
        }

        None
    }

    /// How far an anchored scan starting at `search_start` has to walk, or `None`
    /// when a single unanchored pass proves there is nothing to find. Skipped for
    /// a nullable pattern, which matches empty at the first position tried.
    fn scan_limit(&self, input: &[u8], search_start: usize) -> Option<usize> {
        if self.nullable {
            return Some(input.len());
        }
        self.earliest_match_end(input, search_start)
    }

    /// Tries to match at exactly the given position.
    pub fn try_match_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        self.match_at(input, pos).map(|end| (pos, end))
    }

    /// [`ShiftOrWide::try_match_at`], also reporting how far the attempt
    /// walked.
    pub(crate) fn try_match_at_reach(
        &self,
        input: &[u8],
        pos: usize,
    ) -> (Option<(usize, usize)>, usize) {
        let (end, reach) = self.match_at_reach(input, pos);
        (end.map(|end| (pos, end)), reach)
    }

    /// Core matching logic using 256-bit state vectors.
    fn match_at(&self, input: &[u8], start: usize) -> Option<usize> {
        self.match_at_reach(input, start).0
    }

    /// [`ShiftOrWide::match_at`], also reporting how far the scan walked
    /// before the state went empty.
    fn match_at_reach(&self, input: &[u8], start: usize) -> (Option<usize>, usize) {
        if start > input.len() {
            return (None, start);
        }

        let mut last_match = None;

        if self.nullable {
            last_match = Some(start);
        }

        // State tracking using inverted logic (same as u64 version):
        // - bit i = 0 means we've reached position i (active)
        // - bit i = 1 means we haven't reached position i (inactive)
        let mut state = BitSet256::all_ones();

        for (i, &byte) in input[start..].iter().enumerate() {
            let byte_mask = self.masks[byte as usize];

            if i == 0 {
                // First byte: can only start at positions in First set
                state = self.first.complement().union(byte_mask);
            } else {
                // Subsequent bytes: use follow sets for transitions
                // Flip state: 1 = active, 0 = inactive
                let active = state.complement();

                // Compute union of follow sets for all active positions
                let mut reachable = BitSet256::empty();

                // Iterate over all 4 words to find active positions
                for word_idx in 0..4 {
                    let mut word = active.parts[word_idx];
                    while word != 0 {
                        let bit_idx = word.trailing_zeros() as usize;
                        let pos = word_idx * 64 + bit_idx;
                        if pos < self.follow.len() {
                            reachable.union_assign(self.follow[pos]);
                        }
                        word &= word - 1; // Clear lowest set bit
                    }
                }

                // Invert back to Shift-Or convention (0 = active)
                // Then apply byte mask
                state = reachable.complement().union(byte_mask);
            }

            // Check for match: if any accepting position is reached (bit is 0)
            if !state.union(self.accept).is_all_ones() {
                last_match = Some(start + i + 1);
            }

            // If all bits are 1, no possible match from this starting point
            if state.is_all_ones() {
                return (last_match, start + i + 1);
            }
        }

        (last_match, input.len())
    }
}

/// Checks if an HIR is suitable for Wide Shift-Or (65-256 positions).
pub fn is_shift_or_wide_compatible(hir: &Hir) -> bool {
    if hir.props.has_backrefs
        || hir.props.has_lookaround
        || hir.props.has_anchors
        || hir.props.has_word_boundary
        || hir.props.has_non_greedy
    {
        return false;
    }

    // Nullable patterns can't be represented (see `is_shift_or_compatible`).
    if hir_is_nullable(&hir.expr) {
        return false;
    }

    // Try to build Wide Glushkov NFA to check position count
    compile_glushkov_wide(hir)
        .map(|nfa| {
            nfa.position_count > MAX_POSITIONS
                && nfa.position_count <= MAX_POSITIONS_WIDE
                && nfa.position_count > 0
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod scan_bound_tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn compile(pattern: &str) -> Option<ShiftOr> {
        let hir = parse(pattern).and_then(|ast| translate(&ast)).ok()?;
        if hir.props.has_anchors {
            ShiftOr::from_hir_with_anchors(&hir)
        } else {
            ShiftOr::from_hir(&hir)
        }
    }

    /// The unbounded scan the `scan_limit` bound replaced: try every start.
    ///
    /// `match_at` knows nothing about anchors — they are stripped during Glushkov
    /// construction and enforced by the callers — so this oracle applies them the
    /// same way, otherwise it would happily accept `^a` at position 1.
    fn brute_force(so: &ShiftOr, input: &[u8], from: usize) -> Option<(usize, usize)> {
        if so.has_start_anchor && from > 0 {
            return None;
        }
        let last = if so.has_start_anchor { 0 } else { input.len() };
        for start in from..=last {
            if let Some(end) = so.match_at(input, start) {
                if so.has_end_anchor && !crate::nfa::at_end_or_before_final_newline(input, end) {
                    continue;
                }
                return Some((start, end));
            }
        }
        None
    }

    const PATTERNS: &[&str] = &[
        "a", "ab", "abc", "[ab]", "a+", "a*", "a?", "a{2}", "a{1,3}", "\\w", "\\w+", "\\w*", "\\d",
        "\\w*\\d", "[a-z]*9", "a*b", "a.*b", "a.c", "(?:ab)+", "[^a]", "[^a]+", ".", ".*", ".+",
        "\\s*", "\\s+", "ab*c", "a[bc]d", "^a", "a$", "^ab$", "^a*", "a*$",
    ];

    const TEXTS: &[&str] = &[
        "",
        "a",
        "aa",
        "ab",
        "ba",
        "abc",
        "abab",
        "aaab",
        "aaa9",
        "9aaa",
        "aaa",
        "a b c",
        "  ",
        "xyz",
        "aaaaaaaaab",
        "baaaaaaaaa",
        "abcabcabc",
        "a\nb",
        "aaa\n",
        "9",
        "z",
    ];

    /// Bounding the anchored scan by the earliest match end must never change an
    /// answer: the leftmost start `s*` always satisfies `s* <= e`, because the
    /// match ending at `e` itself starts at or before `e`.
    #[test]
    fn bounded_scan_matches_brute_force() {
        let mut failures = Vec::new();
        for pattern in PATTERNS {
            let Some(so) = compile(pattern) else {
                continue;
            };
            for text in TEXTS {
                let bytes = text.as_bytes();
                for from in 0..=bytes.len() {
                    let expected = brute_force(&so, bytes, from);
                    let got = so.find_at(bytes, from);
                    if got != expected {
                        failures.push(format!(
                            "{pattern:?} on {text:?} from {from}: brute={expected:?} got={got:?}"
                        ));
                    }
                }
                // `find` is the same search anchored at zero.
                let got = so.find(bytes);
                let expected = brute_force(&so, bytes, 0);
                if got != expected && !(so.nullable && expected.is_none()) {
                    failures.push(format!(
                        "find {pattern:?} on {text:?}: brute={expected:?} got={got:?}"
                    ));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} divergences:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// The linear-time search the metered loops hand over to must report the
    /// match trying every start does, from every resume point.
    #[test]
    fn linear_search_matches_brute_force() {
        let mut failures = Vec::new();
        for pattern in PATTERNS {
            let Some(so) = compile(pattern) else {
                continue;
            };
            for text in TEXTS {
                let bytes = text.as_bytes();
                for from in 0..=bytes.len() {
                    let expected = brute_force(&so, bytes, from);
                    let got = so.find_linear(bytes, from);
                    if got != expected {
                        failures.push(format!(
                            "{pattern:?} on {text:?} from {from}: brute={expected:?} got={got:?}"
                        ));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} divergences:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// The same for the 256-position automaton, on patterns wide enough to
    /// need it.
    #[test]
    fn wide_linear_search_matches_brute_force() {
        let patterns = [
            format!("{}b", "a?".repeat(70)),
            format!("x[a-z]*{}", "y".repeat(70)),
            format!("(?:{})+", "ab".repeat(40)),
        ];
        let texts = [
            String::new(),
            "b".to_string(),
            format!("{}b", "a".repeat(80)),
            format!("x{}{}", "q".repeat(5), "y".repeat(70)),
            format!("xx{}", "y".repeat(71)),
            "ab".repeat(90),
        ];
        for pattern in &patterns {
            let hir = parse(pattern)
                .and_then(|ast| translate(&ast))
                .expect("pattern parses");
            let wide = ShiftOrWide::from_hir(&hir).expect("pattern fits the wide automaton");
            for text in &texts {
                let bytes = text.as_bytes();
                for from in 0..=bytes.len() {
                    let expected = (from..=bytes.len())
                        .find_map(|start| wide.match_at(bytes, start).map(|end| (start, end)));
                    assert_eq!(
                        wide.find_linear(bytes, from),
                        expected,
                        "{pattern:?} on {text:?} from {from}"
                    );
                }
            }
        }
    }

    /// The one-pass rejector must agree with the scan about whether anything
    /// matches at all — that equivalence is what makes the early return sound.
    #[test]
    fn earliest_match_end_agrees_with_scan() {
        for pattern in PATTERNS {
            let Some(so) = compile(pattern) else {
                continue;
            };
            if so.nullable || so.has_start_anchor {
                continue;
            }
            for text in TEXTS {
                let bytes = text.as_bytes();
                for from in 0..=bytes.len() {
                    let any_match = (from..=bytes.len())
                        .filter_map(|s| so.match_at(bytes, s).map(|e| (s, e)))
                        .find(|(s, e)| e > s);
                    let end = so.earliest_match_end(bytes, from);
                    assert_eq!(
                        end.is_some(),
                        any_match.is_some(),
                        "{pattern:?} on {text:?} from {from}: pass={end:?} scan={any_match:?}"
                    );
                    if let (Some(e), Some((s, _))) = (end, any_match) {
                        assert!(
                            s <= e,
                            "{pattern:?} on {text:?} from {from}: leftmost start {s} > bound {e}"
                        );
                    }
                }
            }
        }
    }
}
