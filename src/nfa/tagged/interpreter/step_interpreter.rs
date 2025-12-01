//! Step-based interpreter for fast pattern matching.
//!
//! This interpreter executes pre-extracted pattern steps for fast matching.
//! It provides the same algorithm as the JIT but interpreted.

use crate::nfa::tagged::shared::PatternStep;

/// Fast step-based pattern matcher.
///
/// Executes pattern steps directly without full NFA simulation.
/// This is faster than Thompson NFA simulation for patterns that can
/// be expressed as a linear sequence of steps.
pub struct StepInterpreter;

impl StepInterpreter {
    /// Finds the first match in the input.
    pub fn find(steps: &[PatternStep], input: &[u8]) -> Option<(usize, usize)> {
        for start in 0..=input.len() {
            if let Some(end) = Self::match_at(steps, input, start) {
                return Some((start, end));
            }
        }
        None
    }

    /// Finds a match starting at or after the given position.
    pub fn find_at(steps: &[PatternStep], input: &[u8], start_from: usize) -> Option<(usize, usize)> {
        for start in start_from..=input.len() {
            if let Some(end) = Self::match_at(steps, input, start) {
                return Some((start, end));
            }
        }
        None
    }

    /// Attempts to match at a specific position, returning the end position on success.
    fn match_at(steps: &[PatternStep], input: &[u8], start: usize) -> Option<usize> {
        let mut pos = start;

        for step in steps {
            match step {
                PatternStep::Byte(b) => {
                    if pos >= input.len() || input[pos] != *b {
                        return None;
                    }
                    pos += 1;
                }
                PatternStep::Ranges(ranges) => {
                    if pos >= input.len() {
                        return None;
                    }
                    let byte = input[pos];
                    if !ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
                        return None;
                    }
                    pos += 1;
                }
                PatternStep::GreedyPlus(ranges) => {
                    // Must match at least one
                    if pos >= input.len() {
                        return None;
                    }
                    let byte = input[pos];
                    if !ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
                        return None;
                    }
                    pos += 1;
                    // Match as many as possible
                    while pos < input.len() {
                        let byte = input[pos];
                        if !ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
                            break;
                        }
                        pos += 1;
                    }
                }
                PatternStep::GreedyStar(ranges) => {
                    // Match as many as possible (zero or more)
                    while pos < input.len() {
                        let byte = input[pos];
                        if !ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
                            break;
                        }
                        pos += 1;
                    }
                }
                PatternStep::GreedyPlusLookahead(ranges, lookahead_steps, is_positive) => {
                    // Must match at least one
                    if pos >= input.len() {
                        return None;
                    }
                    let byte = input[pos];
                    if !ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
                        return None;
                    }
                    let min_pos = pos + 1;
                    pos += 1;
                    // Greedily consume all matching
                    while pos < input.len() {
                        let byte = input[pos];
                        if !ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
                            break;
                        }
                        pos += 1;
                    }
                    // Backtrack until lookahead succeeds
                    loop {
                        let lookahead_match = Self::check_lookahead(lookahead_steps, input, pos);
                        if *is_positive == lookahead_match {
                            break; // Lookahead succeeded
                        }
                        if pos <= min_pos {
                            return None; // Can't backtrack more
                        }
                        pos -= 1;
                    }
                }
                PatternStep::GreedyStarLookahead(ranges, lookahead_steps, is_positive) => {
                    let min_pos = pos;
                    // Greedily consume all matching
                    while pos < input.len() {
                        let byte = input[pos];
                        if !ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
                            break;
                        }
                        pos += 1;
                    }
                    // Backtrack until lookahead succeeds
                    loop {
                        let lookahead_match = Self::check_lookahead(lookahead_steps, input, pos);
                        if *is_positive == lookahead_match {
                            break;
                        }
                        if pos <= min_pos {
                            return None;
                        }
                        pos -= 1;
                    }
                }
                PatternStep::PositiveLookahead(inner_steps) => {
                    if !Self::check_lookahead(inner_steps, input, pos) {
                        return None;
                    }
                    // Zero-width: don't advance pos
                }
                PatternStep::NegativeLookahead(inner_steps) => {
                    if Self::check_lookahead(inner_steps, input, pos) {
                        return None;
                    }
                    // Zero-width: don't advance pos
                }
                PatternStep::WordBoundary => {
                    if !Self::is_word_boundary(input, pos) {
                        return None;
                    }
                }
                PatternStep::NotWordBoundary => {
                    if Self::is_word_boundary(input, pos) {
                        return None;
                    }
                }
                PatternStep::StartOfText => {
                    if pos != 0 {
                        return None;
                    }
                }
                PatternStep::EndOfText => {
                    if pos != input.len() {
                        return None;
                    }
                }
                PatternStep::PositiveLookbehind(inner_steps, min_len) => {
                    if !Self::check_lookbehind(inner_steps, input, pos, *min_len) {
                        return None;
                    }
                    // Zero-width: don't advance pos
                }
                PatternStep::NegativeLookbehind(inner_steps, min_len) => {
                    if Self::check_lookbehind(inner_steps, input, pos, *min_len) {
                        return None;
                    }
                    // Zero-width: don't advance pos
                }
                PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_) => {
                    // Capture markers don't consume input - skip them
                    // (we're only finding matches, not tracking captures)
                }
                _ => {
                    // Unsupported step - should have been filtered during extraction
                    return None;
                }
            }
        }

        Some(pos)
    }

    /// Checks if the lookahead pattern matches at the given position.
    /// Uses backtracking for greedy quantifiers followed by other patterns.
    fn check_lookahead(steps: &[PatternStep], input: &[u8], pos: usize) -> bool {
        // Optimize common case: `.*X` where X is a character class or byte
        // For `(?=.*\d)`, we need to check if a digit exists within the range that `.*` can match
        if steps.len() == 2 {
            if let PatternStep::GreedyStar(star_ranges) = &steps[0] {
                // Find the extent of `.*` - it matches characters in star_ranges
                // For standard `.*`, star_ranges excludes newline (0x0a)
                let mut star_end = pos;
                while star_end < input.len() {
                    let byte = input[star_end];
                    if !star_ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
                        break;
                    }
                    star_end += 1;
                }

                // Now check if the final step matches anywhere from pos to star_end
                match &steps[1] {
                    PatternStep::Ranges(final_ranges) => {
                        for p in pos..=star_end {
                            if p >= input.len() {
                                break;
                            }
                            let byte = input[p];
                            if final_ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
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
        Self::check_lookahead_recursive(steps, input, pos)
    }

    /// Recursive backtracking lookahead checker.
    fn check_lookahead_recursive(steps: &[PatternStep], input: &[u8], pos: usize) -> bool {
        if steps.is_empty() {
            return true;
        }

        let step = &steps[0];
        let rest = &steps[1..];

        match step {
            PatternStep::Byte(b) => {
                if pos >= input.len() || input[pos] != *b {
                    return false;
                }
                Self::check_lookahead_recursive(rest, input, pos + 1)
            }
            PatternStep::Ranges(ranges) => {
                if pos >= input.len() {
                    return false;
                }
                let byte = input[pos];
                if !ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
                    return false;
                }
                Self::check_lookahead_recursive(rest, input, pos + 1)
            }
            PatternStep::GreedyPlus(ranges) => {
                // Must match at least one
                if pos >= input.len() {
                    return false;
                }
                let byte = input[pos];
                if !ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
                    return false;
                }
                // Greedily match as many as possible, then backtrack
                let mut end = pos + 1;
                while end < input.len() {
                    let byte = input[end];
                    if !ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
                        break;
                    }
                    end += 1;
                }
                // Backtrack from longest match to shortest (at least 1)
                for p in (pos + 1..=end).rev() {
                    if Self::check_lookahead_recursive(rest, input, p) {
                        return true;
                    }
                }
                false
            }
            PatternStep::GreedyStar(ranges) => {
                // Match as many as possible (zero or more), then backtrack
                let mut end = pos;
                while end < input.len() {
                    let byte = input[end];
                    if !ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
                        break;
                    }
                    end += 1;
                }
                // Backtrack from longest match to shortest (including 0)
                for p in (pos..=end).rev() {
                    if Self::check_lookahead_recursive(rest, input, p) {
                        return true;
                    }
                }
                false
            }
            PatternStep::WordBoundary => {
                if !Self::is_word_boundary(input, pos) {
                    return false;
                }
                Self::check_lookahead_recursive(rest, input, pos)
            }
            PatternStep::StartOfText => {
                if pos != 0 {
                    return false;
                }
                Self::check_lookahead_recursive(rest, input, pos)
            }
            PatternStep::EndOfText => {
                if pos != input.len() {
                    return false;
                }
                Self::check_lookahead_recursive(rest, input, pos)
            }
            _ => false,
        }
    }

    /// Checks if the lookbehind pattern matches at position `pos` looking backwards.
    fn check_lookbehind(steps: &[PatternStep], input: &[u8], pos: usize, min_len: usize) -> bool {
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
                PatternStep::Ranges(ranges) => {
                    if p >= pos {
                        return false;
                    }
                    let byte = input[p];
                    if !ranges.iter().any(|r| byte >= r.start && byte <= r.end) {
                        return false;
                    }
                    p += 1;
                }
                PatternStep::WordBoundary => {
                    if !Self::is_word_boundary(input, p) {
                        return false;
                    }
                }
                PatternStep::StartOfText => {
                    if p != 0 {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        // Match succeeds if we consumed exactly the required characters
        p == pos
    }

    #[inline]
    fn is_word_boundary(input: &[u8], pos: usize) -> bool {
        let prev_word = pos > 0 && Self::is_word_char(input[pos - 1]);
        let curr_word = pos < input.len() && Self::is_word_char(input[pos]);
        prev_word != curr_word
    }

    #[inline]
    fn is_word_char(b: u8) -> bool {
        matches!(b, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
    }
}
