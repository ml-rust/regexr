//! Tagged NFA interpreter for fast pattern matching.
//!
//! This interpreter executes pre-extracted pattern steps for fast matching.
//! It provides the same algorithm as the JIT but interpreted.

use crate::nfa::tagged::shared::PatternStep;
use crate::vm::PikeVm;

/// Fast step-based Tagged NFA matcher.
///
/// Executes pattern steps directly without full NFA simulation.
/// This is faster than Thompson NFA simulation for patterns that can
/// be expressed as a linear sequence of steps.
pub struct TaggedNfa;

/// How many units of work a metered search may spend per byte of the input
/// left to search, before its caller re-runs the search on the PikeVM.
///
/// Retrying every start and backtracking every greedy run is cheap when
/// attempts fail near where they began, and quadratic or worse when they do
/// not: `\p{L}+a\p{L}*z` over a long run of letters re-scans the run from
/// every start and every backtrack point.
///
/// Units are charged off the per-step path:
///
/// - A greedy run with steps after it pays one unit per byte it consumes. It
///   gives back at most those bytes, so its give-backs cost nothing more.
/// - A greedy run that ends its step list pays nothing. Its bytes lie inside
///   the match its walk returns. That match is either the search's answer or
///   an alternation branch whose following steps fail.
/// - An alternation branch whose following steps fail pays the bytes it
///   consumed plus one before the next branch runs.
/// - Entering a step costs nothing. A walk with no retry enters each step of
///   the program at most once.
///
/// So every walk after an attempt's first is paid for, each walk enters O(m)
/// steps, and a search tries at most n+1 starts: the work is O(m·n).
const STEP_BUDGET_FACTOR: usize = 16;

/// One step search: the input, and the work the search may still spend.
/// See [`STEP_BUDGET_FACTOR`].
///
/// The input and the allowance travel behind one pointer, so the recursive
/// matchers take no more arguments than they would unmetered.
///
/// Charging never branches: the allowance goes negative once spent. Only
/// retries, give-backs and failed attempts test it, so the work done past
/// the end of the allowance is one retry-free walk, O(m·n). The allowance
/// cannot wrap: it starts at most `isize::MAX`, and every unit charged is
/// work already done.
#[derive(Debug)]
struct Search<'a> {
    input: &'a [u8],
    remaining: isize,
}

impl<'a> Search<'a> {
    /// A search of `input` from `from`, with the allowance for it.
    fn bounded(input: &'a [u8], from: usize) -> Self {
        // A slice holds at most `isize::MAX` bytes, so the length left fits.
        let left = (input.len() - from.min(input.len())) as isize;
        Self {
            input,
            remaining: left
                .saturating_add(1)
                .saturating_mul(STEP_BUDGET_FACTOR as isize),
        }
    }

    /// A search of `input` whose allowance never runs out, for the unmetered
    /// public entry points.
    fn unlimited(input: &'a [u8]) -> Self {
        Self {
            input,
            remaining: isize::MAX,
        }
    }

    /// Spends `units` bytes consumed by a greedy run with steps after it.
    #[inline(always)]
    fn charge(&mut self, units: usize) {
        self.remaining = self.remaining.wrapping_sub_unsigned(units);
    }

    /// Spends the `consumed` bytes of an alternation branch whose following
    /// steps failed, plus one. Returns false once the allowance is gone; the
    /// caller then fails without trying the next branch, and the search's
    /// answer is not to be trusted.
    #[inline(always)]
    fn retry(&mut self, consumed: usize) -> bool {
        self.remaining = self
            .remaining
            .wrapping_sub_unsigned(consumed)
            .wrapping_sub(1);
        self.remaining >= 0
    }

    /// Whether the search ran out of allowance. A greedy run tests it before
    /// each give-back and fails once it holds.
    #[inline(always)]
    fn exhausted(&self) -> bool {
        self.remaining < 0
    }
}

impl TaggedNfa {
    /// Finds the first match in the input.
    pub fn find(steps: &[PatternStep], input: &[u8]) -> Option<(usize, usize)> {
        Self::find_at(steps, input, 0)
    }

    /// Finds a match starting at or after the given position.
    ///
    /// The full input is passed to every attempt, so steps that read left
    /// context (`^`, `\b`, lookbehind) see the real preceding bytes.
    ///
    /// Unmetered: attempts at every start, each backtracking its greedy runs,
    /// can cost far more than one pass over the input. The engines use
    /// `TaggedNfa::find_at_metered` and re-run a search that runs out of
    /// budget on the PikeVM.
    pub fn find_at(
        steps: &[PatternStep],
        input: &[u8],
        start_from: usize,
    ) -> Option<(usize, usize)> {
        Self::find_at_metered(steps, &mut Search::unlimited(input), start_from)
    }

    /// [`TaggedNfa::find_at`] under a work budget proportional to the input
    /// left to search; a search that runs out is re-run on `pike`, the PikeVM
    /// for the same pattern, whose single pass is O(m·n).
    #[inline]
    pub(crate) fn find_at_bounded(
        steps: &[PatternStep],
        pike: &PikeVm,
        input: &[u8],
        start_from: usize,
    ) -> Option<(usize, usize)> {
        let mut search = Search::bounded(input, start_from);
        let found = Self::find_at_metered(steps, &mut search, start_from);
        if search.exhausted() {
            return pike.find_from(input, start_from);
        }
        found
    }

    /// One attempt anchored at `start`, under the same budget as
    /// [`TaggedNfa::find_at_bounded`] and re-run anchored on `pike` when it
    /// runs out. Returns `(start, end)`.
    #[inline]
    pub(crate) fn match_at_bounded(
        steps: &[PatternStep],
        pike: &PikeVm,
        input: &[u8],
        start: usize,
    ) -> Option<(usize, usize)> {
        let mut search = Search::bounded(input, start);
        let found = Self::match_steps(steps, &mut search, start);
        if search.exhausted() {
            return pike.find_at(input, start);
        }
        found.map(|end| (start, end))
    }

    /// [`TaggedNfa::find_at`] over `search`. When its allowance runs out, the
    /// answer is `None` and [`Search::exhausted`] is set.
    fn find_at_metered(
        steps: &[PatternStep],
        search: &mut Search<'_>,
        start_from: usize,
    ) -> Option<(usize, usize)> {
        let input = search.input;
        for start in start_from..=input.len() {
            // Only start at UTF-8 codepoint boundaries (see `is_utf8_boundary`).
            if !crate::nfa::is_utf8_boundary(input, start) {
                continue;
            }
            if let Some(end) = Self::match_steps(steps, search, start) {
                return Some((start, end));
            }
            if search.exhausted() {
                return None;
            }
        }
        None
    }

    /// Matches a sequence of steps starting at the given position.
    fn match_steps(steps: &[PatternStep], search: &mut Search<'_>, start: usize) -> Option<usize> {
        let input = search.input;
        let mut pos = start;

        for (step_idx, step) in steps.iter().enumerate() {
            match step {
                PatternStep::Byte(b) => {
                    if pos >= input.len() || input[pos] != *b {
                        return None;
                    }
                    pos += 1;
                }
                PatternStep::ByteClass(byte_class) => {
                    if pos >= input.len() {
                        return None;
                    }
                    let byte = input[pos];
                    if !byte_class.contains(byte) {
                        return None;
                    }
                    pos += 1;
                }
                PatternStep::GreedyPlus(byte_class) => {
                    // Must match at least one
                    if pos >= input.len() {
                        return None;
                    }
                    let byte = input[pos];
                    if !byte_class.contains(byte) {
                        return None;
                    }
                    let min_pos = pos + 1;
                    pos += 1;
                    // Match as many as possible
                    while pos < input.len() {
                        let byte = input[pos];
                        if !byte_class.contains(byte) {
                            break;
                        }
                        pos += 1;
                    }
                    // Try to match remaining steps, backtracking if needed
                    let remaining_steps = &steps[step_idx + 1..];
                    if !remaining_steps.is_empty() {
                        search.charge(pos - min_pos);
                        loop {
                            if let Some(end) = Self::match_steps(remaining_steps, search, pos) {
                                return Some(end);
                            }
                            if pos <= min_pos || search.exhausted() {
                                return None; // Can't backtrack more
                            }
                            pos -= 1; // Backtrack one byte
                        }
                    }
                }
                PatternStep::GreedyStar(byte_class) => {
                    let min_pos = pos; // Can backtrack to zero matches
                                       // Match as many as possible (zero or more)
                    while pos < input.len() {
                        let byte = input[pos];
                        if !byte_class.contains(byte) {
                            break;
                        }
                        pos += 1;
                    }
                    // Try to match remaining steps, backtracking if needed
                    let remaining_steps = &steps[step_idx + 1..];
                    if !remaining_steps.is_empty() {
                        search.charge(pos - min_pos);
                        loop {
                            if let Some(end) = Self::match_steps(remaining_steps, search, pos) {
                                return Some(end);
                            }
                            if pos <= min_pos || search.exhausted() {
                                return None; // Can't backtrack more
                            }
                            pos -= 1; // Backtrack one byte
                        }
                    }
                }
                PatternStep::GreedyPlusLookahead(byte_class, lookahead_steps, is_positive) => {
                    // Must match at least one
                    if pos >= input.len() {
                        return None;
                    }
                    let byte = input[pos];
                    if !byte_class.contains(byte) {
                        return None;
                    }
                    let min_pos = pos + 1;
                    pos += 1;
                    // Greedily consume all matching
                    while pos < input.len() {
                        let byte = input[pos];
                        if !byte_class.contains(byte) {
                            break;
                        }
                        pos += 1;
                    }
                    search.charge(pos - min_pos);
                    // Backtrack until lookahead succeeds
                    pos = Self::backtrack_to_lookahead(
                        lookahead_steps,
                        *is_positive,
                        search,
                        pos,
                        min_pos,
                    )?;
                }
                PatternStep::GreedyStarLookahead(byte_class, lookahead_steps, is_positive) => {
                    let min_pos = pos;
                    // Greedily consume all matching
                    while pos < input.len() {
                        let byte = input[pos];
                        if !byte_class.contains(byte) {
                            break;
                        }
                        pos += 1;
                    }
                    search.charge(pos - min_pos);
                    // Backtrack until lookahead succeeds
                    pos = Self::backtrack_to_lookahead(
                        lookahead_steps,
                        *is_positive,
                        search,
                        pos,
                        min_pos,
                    )?;
                }
                PatternStep::PositiveLookahead(inner_steps) => {
                    if !Self::check_lookahead(inner_steps, search, pos) {
                        return None;
                    }
                    // Zero-width: don't advance pos
                }
                PatternStep::NegativeLookahead(inner_steps) => {
                    if Self::check_lookahead(inner_steps, search, pos) {
                        return None;
                    }
                    // Zero-width: don't advance pos
                }
                PatternStep::WordBoundary => {
                    if !crate::nfa::is_word_boundary(input, pos) {
                        return None;
                    }
                }
                PatternStep::NotWordBoundary => {
                    if crate::nfa::is_word_boundary(input, pos) {
                        return None;
                    }
                }
                PatternStep::StartOfText => {
                    if pos != 0 {
                        return None;
                    }
                }
                PatternStep::EndOfText => {
                    if !crate::nfa::at_end_or_before_final_newline(input, pos) {
                        return None;
                    }
                }
                PatternStep::PositiveLookbehind(inner_steps, widths) => {
                    if !Self::check_lookbehind(inner_steps, input, pos, widths) {
                        return None;
                    }
                    // Zero-width: don't advance pos
                }
                PatternStep::NegativeLookbehind(inner_steps, widths) => {
                    if Self::check_lookbehind(inner_steps, input, pos, widths) {
                        return None;
                    }
                    // Zero-width: don't advance pos
                }
                PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_) => {
                    // Capture markers don't consume input - skip them
                    // (we're only finding matches, not tracking captures)
                }
                PatternStep::CodepointClass(cpclass, _target) => {
                    // Decode one UTF-8 codepoint and check class membership
                    if let Some((cp, len)) = Self::decode_utf8(input, pos) {
                        if cpclass.contains(cp) {
                            pos += len;
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                PatternStep::GreedyCodepointPlus(cpclass) => {
                    // Must match at least one codepoint
                    if let Some((cp, len)) = Self::decode_utf8(input, pos) {
                        if !cpclass.contains(cp) {
                            return None;
                        }
                        pos += len;
                    } else {
                        return None;
                    }
                    // The first codepoint is mandatory, so it is also the
                    // shortest the run may backtrack to.
                    let min_pos = pos;
                    // Match as many as possible
                    while let Some((cp, len)) = Self::decode_utf8(input, pos) {
                        if !cpclass.contains(cp) {
                            break;
                        }
                        pos += len;
                    }
                    // Try to match remaining steps, backtracking if needed
                    let remaining_steps = &steps[step_idx + 1..];
                    if !remaining_steps.is_empty() {
                        search.charge(pos - min_pos);
                        // Backtrack from longest match to shortest, walking the
                        // run's codepoint boundaries backwards rather than
                        // recording them on the way in — see
                        // [`TaggedNfa::prev_boundary`].
                        let mut boundary = pos;
                        loop {
                            if let Some(end) = Self::match_steps(remaining_steps, search, boundary)
                            {
                                return Some(end);
                            }
                            if boundary <= min_pos || search.exhausted() {
                                return None;
                            }
                            boundary = Self::prev_boundary(input, boundary);
                        }
                    }
                }
                PatternStep::Alt(alternatives) => {
                    // Try each alternative
                    let remaining_steps = &steps[step_idx + 1..];
                    for alt_steps in alternatives {
                        // A branch that opens on a byte test fails at once
                        // when the byte at `pos` does not pass it; skip the
                        // call.
                        match alt_steps.first() {
                            Some(PatternStep::Byte(b)) if input.get(pos) != Some(b) => continue,
                            Some(PatternStep::ByteClass(class))
                                if !input.get(pos).is_some_and(|&byte| class.contains(byte)) =>
                            {
                                continue
                            }
                            _ => {}
                        }
                        // Match the alternative
                        if let Some(alt_end) = Self::match_steps(alt_steps, search, pos) {
                            // Then match remaining steps after the Alt
                            if remaining_steps.is_empty() {
                                return Some(alt_end);
                            }
                            if let Some(final_end) =
                                Self::match_steps(remaining_steps, search, alt_end)
                            {
                                return Some(final_end);
                            }
                            // This alternative matched but remaining steps failed, try next alternative
                            if !search.retry(alt_end - pos) {
                                return None;
                            }
                        }
                    }
                    return None;
                }
                _ => {
                    // Unsupported step - should have been filtered during extraction
                    return None;
                }
            }
        }

        Some(pos)
    }

    /// Walks a greedy run back from its end at `pos` down to `min_pos`, returning
    /// the longest end at which the lookahead assertion holds.
    ///
    /// The scan runs downward and takes the first success because a greedy
    /// quantifier prefers the longest run.
    ///
    /// When the assertion is positive and its pattern starts with a literal
    /// byte, only positions holding that byte can satisfy it — the recursive
    /// checker's `Byte` arm rejects every other position on its first step — so
    /// those positions are jumped over with a reverse byte search instead of
    /// being re-checked one at a time. A negative assertion is deliberately
    /// excluded: it is satisfied at *most* positions, which are exactly the ones
    /// such a skip would pass over.
    ///
    /// Kept out of line: inlined, its loop enlarges the frame of
    /// [`TaggedNfa::match_steps`], and every step pays for the extra spills.
    #[inline(never)]
    fn backtrack_to_lookahead(
        lookahead_steps: &[PatternStep],
        is_positive: bool,
        search: &mut Search<'_>,
        mut pos: usize,
        min_pos: usize,
    ) -> Option<usize> {
        let input = search.input;
        // One jump before the walk, never inside it. A positive assertion that
        // opens on a literal byte cannot hold anywhere that byte is absent, so
        // the stretch between the greedy end and the last occurrence is skipped
        // outright. The walk itself is left exactly as it was: putting the skip
        // in the loop measured slower than the decrement it replaced, because
        // the runs being walked are short enough that the extra test per step
        // costs more than the steps it saves.
        if is_positive {
            if let Some(PatternStep::Byte(b)) = lookahead_steps.first() {
                if !Self::byte_at(input, pos, *b) {
                    pos = Self::last_byte_in(input, *b, min_pos, pos)?;
                }
            }
        }
        loop {
            let lookahead_match = Self::check_lookahead(lookahead_steps, search, pos);
            if is_positive == lookahead_match {
                return Some(pos); // Lookahead succeeded
            }
            if pos <= min_pos || search.exhausted() {
                return None; // Can't backtrack more
            }
            pos -= 1;
        }
    }

    /// Whether `input[pos]` is `needle`, false past the end of the input.
    fn byte_at(input: &[u8], pos: usize, needle: u8) -> bool {
        input.get(pos).is_some_and(|&b| b == needle)
    }

    /// The greatest index in `[min_pos, end)` holding `needle`, if any.
    ///
    /// `end` is clamped to the input length, since a greedy run may reach the
    /// end of input, and an empty or inverted window yields `None` rather than
    /// slicing out of range.
    #[inline]
    fn last_byte_in(input: &[u8], needle: u8, min_pos: usize, end: usize) -> Option<usize> {
        let end = end.min(input.len());
        if end <= min_pos {
            return None;
        }
        let window = &input[min_pos..end];
        #[cfg(feature = "simd")]
        let offset = crate::simd::memrchr(needle, window);
        #[cfg(not(feature = "simd"))]
        let offset = window.iter().rposition(|&b| b == needle);
        offset.map(|i| min_pos + i)
    }

    /// Checks if the lookahead pattern matches at the given position.
    /// Uses backtracking for greedy quantifiers followed by other patterns.
    fn check_lookahead(steps: &[PatternStep], search: &mut Search<'_>, pos: usize) -> bool {
        let input = search.input;
        // Optimize common case: `.*X` where X is a character class or byte
        // For `(?=.*\d)`, we need to check if a digit exists within the range that `.*` can match
        if steps.len() == 2 {
            if let PatternStep::GreedyStar(star_class) = &steps[0] {
                // Find the extent of `.*` - it matches characters in star_class
                // For standard `.*`, star_class excludes newline (0x0a)
                let mut star_end = pos;
                while star_end < input.len() {
                    let byte = input[star_end];
                    if !star_class.contains(byte) {
                        break;
                    }
                    star_end += 1;
                }
                search.charge(star_end - pos);

                // Now check if the final step matches anywhere from pos to star_end
                match &steps[1] {
                    PatternStep::ByteClass(final_class) => {
                        for p in pos..=star_end {
                            if p >= input.len() {
                                break;
                            }
                            let byte = input[p];
                            if final_class.contains(byte) {
                                return true;
                            }
                        }
                        return false;
                    }
                    PatternStep::Byte(b) => {
                        for p in pos..=star_end {
                            if p >= input.len() {
                                break;
                            }
                            if input[p] == *b {
                                return true;
                            }
                        }
                        return false;
                    }
                    _ => {}
                }
            }
        }

        // General case: use recursive backtracking
        Self::check_lookahead_recursive(steps, search, pos)
    }

    /// Recursive backtracking lookahead checker.
    fn check_lookahead_recursive(
        steps: &[PatternStep],
        search: &mut Search<'_>,
        pos: usize,
    ) -> bool {
        if steps.is_empty() {
            return true;
        }
        let input = search.input;

        let step = &steps[0];
        let rest = &steps[1..];

        match step {
            PatternStep::Byte(b) => {
                if pos >= input.len() || input[pos] != *b {
                    return false;
                }
                Self::check_lookahead_recursive(rest, search, pos + 1)
            }
            PatternStep::ByteClass(byte_class) => {
                if pos >= input.len() {
                    return false;
                }
                let byte = input[pos];
                if !byte_class.contains(byte) {
                    return false;
                }
                Self::check_lookahead_recursive(rest, search, pos + 1)
            }
            PatternStep::GreedyPlus(byte_class) => {
                // Must match at least one
                if pos >= input.len() {
                    return false;
                }
                let byte = input[pos];
                if !byte_class.contains(byte) {
                    return false;
                }
                // Greedily match as many as possible, then backtrack
                let mut end = pos + 1;
                while end < input.len() {
                    let byte = input[end];
                    if !byte_class.contains(byte) {
                        break;
                    }
                    end += 1;
                }
                search.charge(end - pos);
                // Backtrack from longest match to shortest (at least 1)
                for p in (pos + 1..=end).rev() {
                    if Self::check_lookahead_recursive(rest, search, p) {
                        return true;
                    }
                    if search.exhausted() {
                        return false;
                    }
                }
                false
            }
            PatternStep::GreedyStar(byte_class) => {
                // Match as many as possible (zero or more), then backtrack
                let mut end = pos;
                while end < input.len() {
                    let byte = input[end];
                    if !byte_class.contains(byte) {
                        break;
                    }
                    end += 1;
                }
                search.charge(end - pos);
                // Backtrack from longest match to shortest (including 0)
                for p in (pos..=end).rev() {
                    if Self::check_lookahead_recursive(rest, search, p) {
                        return true;
                    }
                    if search.exhausted() {
                        return false;
                    }
                }
                false
            }
            PatternStep::WordBoundary => {
                if !crate::nfa::is_word_boundary(input, pos) {
                    return false;
                }
                Self::check_lookahead_recursive(rest, search, pos)
            }
            PatternStep::NotWordBoundary => {
                if crate::nfa::is_word_boundary(input, pos) {
                    return false;
                }
                Self::check_lookahead_recursive(rest, search, pos)
            }
            PatternStep::StartOfText => {
                if pos != 0 {
                    return false;
                }
                Self::check_lookahead_recursive(rest, search, pos)
            }
            PatternStep::EndOfText => {
                if !crate::nfa::at_end_or_before_final_newline(input, pos) {
                    return false;
                }
                Self::check_lookahead_recursive(rest, search, pos)
            }
            PatternStep::StartOfLine => {
                if !crate::nfa::at_line_start(input, pos) {
                    return false;
                }
                Self::check_lookahead_recursive(rest, search, pos)
            }
            PatternStep::EndOfLine => {
                if !crate::nfa::at_line_end(input, pos) {
                    return false;
                }
                Self::check_lookahead_recursive(rest, search, pos)
            }
            PatternStep::CodepointClass(cpclass, _target) => {
                if let Some((cp, len)) = Self::decode_utf8(input, pos) {
                    if cpclass.contains(cp) {
                        return Self::check_lookahead_recursive(rest, search, pos + len);
                    }
                }
                false
            }
            PatternStep::GreedyCodepointPlus(cpclass) => {
                // Must match at least one
                if let Some((cp, len)) = Self::decode_utf8(input, pos) {
                    if !cpclass.contains(cp) {
                        return false;
                    }
                    // Greedily match as many as possible
                    let mut end = pos + len;
                    while let Some((cp2, len2)) = Self::decode_utf8(input, end) {
                        if !cpclass.contains(cp2) {
                            break;
                        }
                        end += len2;
                    }
                    search.charge(end - pos);
                    // Backtrack from longest match to shortest (at least 1),
                    // walking the run's codepoint boundaries backwards rather
                    // than recording them on the way in — see
                    // [`TaggedNfa::prev_boundary`]. The first codepoint is
                    // mandatory, so `pos + len` is the shortest run allowed.
                    let min_pos = pos + len;
                    let mut boundary = end;
                    loop {
                        if Self::check_lookahead_recursive(rest, search, boundary) {
                            return true;
                        }
                        if boundary <= min_pos || search.exhausted() {
                            return false;
                        }
                        boundary = Self::prev_boundary(input, boundary);
                    }
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Checks if the lookbehind pattern matches at position `pos` looking
    /// backwards, trying each candidate total width in turn.
    ///
    /// A lookbehind is a zero-width boolean assertion that records no captures,
    /// so the candidates can be tried in any order and OR-ed: none of them can
    /// change where the surrounding match starts or ends, and leftmost/longest
    /// is unaffected. A candidate that would land mid-codepoint makes
    /// `pos - width` a continuation byte, which `check_lookbehind_at` rejects
    /// when it decodes there, so a wrong candidate cannot produce a false
    /// positive — it can only cost one wasted walk.
    fn check_lookbehind(steps: &[PatternStep], input: &[u8], pos: usize, widths: &[usize]) -> bool {
        widths
            .iter()
            .any(|&width| Self::check_lookbehind_at(steps, input, pos, width))
    }

    /// Checks if the lookbehind pattern matches at position `pos` looking
    /// backwards, assuming it consumes exactly `min_len` bytes.
    fn check_lookbehind_at(
        steps: &[PatternStep],
        input: &[u8],
        pos: usize,
        min_len: usize,
    ) -> bool {
        // Cannot match if not enough characters behind
        if pos < min_len {
            return false;
        }
        // Check pattern backwards from pos
        let start = pos - min_len;
        let mut p = start;
        for step in steps {
            match step {
                PatternStep::Byte(b) => {
                    if p >= pos || input[p] != *b {
                        return false;
                    }
                    p += 1;
                }
                PatternStep::ByteClass(byte_class) => {
                    if p >= pos {
                        return false;
                    }
                    let byte = input[p];
                    if !byte_class.contains(byte) {
                        return false;
                    }
                    p += 1;
                }
                PatternStep::WordBoundary => {
                    if !crate::nfa::is_word_boundary(input, p) {
                        return false;
                    }
                }
                PatternStep::NotWordBoundary => {
                    if crate::nfa::is_word_boundary(input, p) {
                        return false;
                    }
                }
                PatternStep::StartOfText => {
                    if p != 0 {
                        return false;
                    }
                }
                PatternStep::EndOfText => {
                    if !crate::nfa::at_end_or_before_final_newline(input, p) {
                        return false;
                    }
                }
                PatternStep::StartOfLine => {
                    if !crate::nfa::at_line_start(input, p) {
                        return false;
                    }
                }
                PatternStep::EndOfLine => {
                    if !crate::nfa::at_line_end(input, p) {
                        return false;
                    }
                }
                PatternStep::CodepointClass(cpclass, _target) => {
                    if let Some((cp, len)) = Self::decode_utf8(input, p) {
                        if p + len > pos {
                            return false; // Would go past lookbehind boundary
                        }
                        if !cpclass.contains(cp) {
                            return false;
                        }
                        p += len;
                    } else {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        // Match succeeds if we consumed exactly the required characters
        p == pos
    }

    /// The start of the UTF-8 codepoint that ends at `end`.
    ///
    /// Lets a greedy codepoint run backtrack without recording where it has
    /// been. UTF-8 is self-synchronizing — a continuation byte is `10xxxxxx`
    /// and a leading byte never is — so the previous boundary is found by
    /// stepping back over continuation bytes, in O(1) for the ≤4 bytes a
    /// codepoint can occupy.
    ///
    /// This replaced a `Vec` of boundaries collected on the way in, which grew
    /// by doubling with the length of the run: matching `\p{L}+` against a
    /// 400-character word cost nine reallocations, on every call, for a
    /// quantifier that in the common case never backtracks at all.
    ///
    /// `end` must be a codepoint boundary strictly inside `input` (every call
    /// site derives it from a decoded codepoint, and only steps back while it
    /// is above the run's mandatory first codepoint).
    #[inline]
    fn prev_boundary(input: &[u8], end: usize) -> usize {
        let mut i = end - 1;
        while i > 0 && (input[i] & 0xC0) == 0x80 {
            i -= 1;
        }
        i
    }

    /// Decodes one UTF-8 codepoint from input at the given position.
    /// Returns (codepoint, byte_length) on success, None if invalid UTF-8 or at end.
    #[inline]
    fn decode_utf8(input: &[u8], pos: usize) -> Option<(u32, usize)> {
        if pos >= input.len() {
            return None;
        }
        let b0 = input[pos];
        if b0 < 0x80 {
            // ASCII: single byte
            return Some((b0 as u32, 1));
        } else if b0 < 0xC0 {
            // Invalid: continuation byte at start
            return None;
        } else if b0 < 0xE0 {
            // 2-byte sequence
            if pos + 1 >= input.len() {
                return None;
            }
            let b1 = input[pos + 1];
            if (b1 & 0xC0) != 0x80 {
                return None;
            }
            let cp = ((b0 as u32 & 0x1F) << 6) | (b1 as u32 & 0x3F);
            return Some((cp, 2));
        } else if b0 < 0xF0 {
            // 3-byte sequence
            if pos + 2 >= input.len() {
                return None;
            }
            let b1 = input[pos + 1];
            let b2 = input[pos + 2];
            if (b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 {
                return None;
            }
            let cp = ((b0 as u32 & 0x0F) << 12) | ((b1 as u32 & 0x3F) << 6) | (b2 as u32 & 0x3F);
            return Some((cp, 3));
        } else if b0 < 0xF8 {
            // 4-byte sequence
            if pos + 3 >= input.len() {
                return None;
            }
            let b1 = input[pos + 1];
            let b2 = input[pos + 2];
            let b3 = input[pos + 3];
            if (b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80 {
                return None;
            }
            let cp = ((b0 as u32 & 0x07) << 18)
                | ((b1 as u32 & 0x3F) << 12)
                | ((b2 as u32 & 0x3F) << 6)
                | (b3 as u32 & 0x3F);
            return Some((cp, 4));
        }
        None
    }
}

#[cfg(test)]
mod greedy_lookahead_skip_tests {
    use super::*;
    use crate::nfa::tagged::steps::StepExtractor;

    /// Extracts the step program for `pattern`, asserting that it really uses a
    /// combined greedy+lookahead step.
    ///
    /// Without that assertion the tests would be blind: if extraction declined
    /// the pattern, or emitted a codepoint run instead, the engine would still
    /// answer correctly through another path and the backtracking loop under
    /// test would never execute.
    fn combined_steps_for(pattern: &str) -> Vec<PatternStep> {
        let ast = crate::parser::parse(pattern).unwrap();
        let hir = crate::hir::translate(&ast).unwrap();
        let nfa = crate::nfa::compile(&hir).unwrap();
        let steps = StepExtractor::new(&nfa)
            .extract()
            .unwrap_or_else(|| panic!("step extraction declined {pattern:?}"));
        assert!(
            steps.iter().any(|step| matches!(
                step,
                PatternStep::GreedyPlusLookahead(_, _, _)
                    | PatternStep::GreedyStarLookahead(_, _, _)
            )),
            "{pattern:?} extracted no combined greedy+lookahead step: {steps:?}"
        );
        steps
    }

    #[test]
    fn backtrack_lands_on_the_occurrence_that_satisfies_the_assertion() {
        // "ing" occurs at 1 and at 4, but only the one at 4 is followed by a
        // word boundary, so a skip that lands on the wrong occurrence shows up
        // here immediately.
        let steps = combined_steps_for(r"\w+(?=ing\b)");
        assert_eq!(TaggedNfa::find(&steps, b"singing"), Some((0, 4)));
    }

    #[test]
    fn skip_does_not_overshoot_repeated_first_bytes() {
        // The lookahead's first byte sits at 0, 2, 4 and 6 inside the run, and
        // only the last of them is followed by "y" at the end of the haystack.
        // A skip that searched the wrong window would settle on an earlier `x`
        // and report a shorter match.
        let steps = combined_steps_for(r"\w+(?=xy\b)");
        assert_eq!(TaggedNfa::find(&steps, b"xaxaxaxy"), Some((0, 6)));
    }

    #[test]
    fn greedy_prefers_the_longer_of_two_valid_ends() {
        // "ing" follows both position 1 and position 4. Greedy semantics demand
        // the longer run, so the scan must stay downward-from-the-end and take
        // its first success.
        let steps = combined_steps_for(r"\w+(?=ing)");
        assert_eq!(TaggedNfa::find(&steps, b"iinging"), Some((0, 4)));
    }

    #[test]
    fn negative_lookahead_with_a_literal_first_byte_still_backtracks() {
        // At the greedy end (2) the assertion's "s" matches, so the negative
        // lookahead fails and the run must shrink to 1, where "b" is not "s".
        // Skipping to positions holding "s" would jump over exactly the
        // positions that satisfy a negative assertion and report no match.
        let steps = combined_steps_for(r"[a-r]+(?!s)");
        assert_eq!(TaggedNfa::find(&steps, b"abs"), Some((0, 1)));
        let steps = combined_steps_for(r"[a-r]+(?!s\b)");
        assert_eq!(TaggedNfa::find(&steps, b"abs"), Some((0, 1)));
    }

    #[test]
    fn negative_lookahead_satisfied_at_the_greedy_end_keeps_the_whole_run() {
        let steps = combined_steps_for(r"\w+(?![s]\b)");
        assert_eq!(TaggedNfa::find(&steps, b"cats"), Some((0, 4)));
    }

    #[test]
    fn star_variant_skips_the_same_way() {
        // `StepExtractor` recognises the nullable run structurally and emits a
        // `GreedyStar`, which the combiner folds into the star-shaped combined
        // step — so this arm is exercised from a pattern string rather than
        // from a hand-built program. `steps::nullable_run_with_assertion_tests`
        // pins that it is the star form and not the plus form.
        let steps = combined_steps_for(r"\w*(?=ing)");
        // Empty run: the assertion holds at the start position itself.
        assert_eq!(TaggedNfa::find(&steps, b"ing"), Some((0, 0)));
        // Longest valid run wins here too.
        assert_eq!(TaggedNfa::find(&steps, b"singing"), Some((0, 4)));
    }

    #[test]
    fn run_reaching_end_of_input_and_exhausted_windows() {
        let steps = combined_steps_for(r"\w+(?=ing)");
        // The run reaches `input.len()`, where the assertion cannot hold; the
        // skip must step back to position 1, which is also the shortest run the
        // `+` allows.
        assert_eq!(TaggedNfa::find(&steps, b"sing"), Some((0, 1)));
        // The literal never occurs, so every start position runs out of window.
        let steps = combined_steps_for(r"\w+(?=zz)");
        assert_eq!(TaggedNfa::find(&steps, b"ab"), None);
        assert_eq!(TaggedNfa::find(&steps, b""), None);
    }
}

#[cfg(test)]
mod multi_width_lookbehind_tests {
    use super::*;
    use crate::nfa::tagged::steps::StepExtractor;

    /// Extracts the step program for `pattern`, failing the test if the
    /// extractor declines it.
    ///
    /// The assertion is half the point of these tests: a declined extraction is
    /// invisible from the public API — the engine silently falls back to the
    /// PikeVm and still answers correctly — so a test that only checked match
    /// results would keep passing with the step path dead. Asserting extraction
    /// here pins that `\s`-style multi-width lookbehinds stay on the step
    /// engine, and every `find` below then really runs `check_lookbehind`.
    fn steps_for(pattern: &str) -> Vec<PatternStep> {
        let ast = crate::parser::parse(pattern).unwrap();
        let hir = crate::hir::translate(&ast).unwrap();
        let nfa = crate::nfa::compile(&hir).unwrap();
        StepExtractor::new(&nfa)
            .extract()
            .unwrap_or_else(|| panic!("step extraction declined {pattern:?}"))
    }

    #[test]
    fn positive_lookbehind_accepts_every_whitespace_width() {
        // `\s` is the full Unicode White_Space set, whose members encode to 1, 2
        // or 3 UTF-8 bytes. Each width must be recognised behind the assertion.
        let steps = steps_for(r"(?<=\s)\w+");
        // ASCII space: one byte.
        assert_eq!(TaggedNfa::find(&steps, " ab".as_bytes()), Some((1, 3)));
        // U+00A0 NO-BREAK SPACE: two bytes.
        assert_eq!(TaggedNfa::find(&steps, "\u{A0}ab".as_bytes()), Some((2, 4)));
        // U+2003 EM SPACE and U+3000 IDEOGRAPHIC SPACE: three bytes.
        assert_eq!(
            TaggedNfa::find(&steps, "\u{2003}ab".as_bytes()),
            Some((3, 5))
        );
        assert_eq!(
            TaggedNfa::find(&steps, "\u{3000}ab".as_bytes()),
            Some((3, 5))
        );
        // A non-space of each width in front: no candidate may succeed, and the
        // wider candidates must not be fooled by landing on a continuation byte.
        assert_eq!(TaggedNfa::find(&steps, ",ab".as_bytes()), None);
        assert_eq!(TaggedNfa::find(&steps, "\u{E9}ab".as_bytes()), None);
        assert_eq!(TaggedNfa::find(&steps, "\u{4E2D}ab".as_bytes()), None);
    }

    #[test]
    fn a_candidate_width_wider_than_the_haystack_is_skipped() {
        // At position 1 only the one-byte candidate fits; the two- and
        // three-byte ones would run off the front of the haystack.
        let steps = steps_for(r"(?<=\s)x");
        assert_eq!(TaggedNfa::find(&steps, " x".as_bytes()), Some((1, 2)));
        // At position 0 no candidate fits at all.
        assert_eq!(TaggedNfa::find(&steps, "x".as_bytes()), None);
        assert_eq!(TaggedNfa::find(&steps, "".as_bytes()), None);
    }

    #[test]
    fn negative_lookbehind_negates_the_whole_candidate_set() {
        let steps = steps_for(r"(?<!\s)\w+");
        // Nothing behind the first position, so the assertion holds there.
        assert_eq!(TaggedNfa::find(&steps, "ab".as_bytes()), Some((0, 2)));
        // A space of any width behind must block that position — the two- and
        // three-byte cases only fail if the wider candidates are tried, since
        // the one-byte candidate lands on a continuation byte and rejects.
        assert_eq!(TaggedNfa::find(&steps, " ab".as_bytes()), Some((2, 3)));
        assert_eq!(TaggedNfa::find(&steps, "\u{A0}ab".as_bytes()), Some((3, 4)));
        assert_eq!(
            TaggedNfa::find(&steps, "\u{2003}ab".as_bytes()),
            Some((4, 5))
        );
    }

    #[test]
    fn widths_of_two_lookbehind_classes_combine() {
        // Two `\s` in the lookbehind: the totals are the sumset {2..=6}, and a
        // walk from the wrong total must not be mistaken for a match.
        let steps = steps_for(r"(?<=\s\s)x");
        assert_eq!(TaggedNfa::find(&steps, "  x".as_bytes()), Some((2, 3)));
        assert_eq!(TaggedNfa::find(&steps, " \u{A0}x".as_bytes()), Some((3, 4)));
        let two_em = "\u{2003}\u{2003}x";
        assert_eq!(TaggedNfa::find(&steps, two_em.as_bytes()), Some((6, 7)));
        // Only one space behind: no total may be satisfied.
        assert_eq!(TaggedNfa::find(&steps, " x".as_bytes()), None);
        assert_eq!(TaggedNfa::find(&steps, "\u{2003}x".as_bytes()), None);
    }
}

#[cfg(test)]
mod metering_tests {
    use super::*;
    use crate::nfa::{ByteClass, ByteRange};

    fn run_of(byte: u8) -> ByteClass {
        ByteClass::new(vec![ByteRange::single(byte)])
    }

    /// Runs a bounded search of `steps` over `input` and returns the answer
    /// and the allowance left.
    fn metered(steps: &[PatternStep], input: &[u8]) -> (Option<(usize, usize)>, isize) {
        let mut search = Search::bounded(input, 0);
        let found = TaggedNfa::find_at_metered(steps, &mut search, 0);
        (found, search.remaining)
    }

    #[test]
    fn a_run_that_ends_the_answer_costs_nothing() {
        let steps = [PatternStep::GreedyPlus(run_of(b'a'))];
        let input = vec![b'a'; 4096];
        let (found, left) = metered(&steps, &input);
        assert_eq!(found, Some((0, input.len())));
        assert_eq!(left, Search::bounded(&input, 0).remaining);
    }

    #[test]
    fn a_run_with_steps_after_it_runs_out_on_a_quadratic_input() {
        // `a+z` over a run of `a` with no `z`: every start re-scans the run.
        let steps = [
            PatternStep::GreedyPlus(run_of(b'a')),
            PatternStep::Byte(b'z'),
        ];
        let input = vec![b'a'; 4096];
        let (found, left) = metered(&steps, &input);
        assert_eq!(found, None);
        assert!(left < 0, "the search kept {left} units");
    }

    #[test]
    fn a_discarded_branch_pays_for_the_run_that_ends_it() {
        // `(?:a+)z` kept as an alternation with a step after it: the branch's
        // run ends its step list, so only the failed step after the branch
        // pays for the bytes the run consumed.
        let steps = [
            PatternStep::Alt(vec![vec![PatternStep::GreedyPlus(run_of(b'a'))]]),
            PatternStep::Byte(b'z'),
        ];
        let mut input = vec![b'a'; 4096];
        let (found, left) = metered(&steps, &input);
        assert_eq!(found, None);
        assert!(left < 0, "the search kept {left} units");

        input.push(b'z');
        assert_eq!(TaggedNfa::find(&steps, &input), Some((0, input.len())));
    }
}
