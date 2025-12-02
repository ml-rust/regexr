//! Shared data structures for Tagged NFA execution.
//!
//! These structures are used by both the interpreter and JIT implementations.

use crate::hir::CodepointClass;
use crate::nfa::{ByteClass, StateId};

/// Maximum threads per position (prevents unbounded memory growth).
pub const MAX_THREADS: usize = 256;

/// Structure-of-Arrays thread worklist for cache-efficient processing.
///
/// Hot data (states, flags) is stored separately from cold data (captures)
/// to improve cache utilization during the main matching loop.
///
/// # Dynamic Stride
///
/// The `captures` Vec uses a dynamic stride per pattern:
/// - Logical indexing: `captures[thread_idx * stride + slot_idx]`
/// - Stride = `(capture_count + 1) * 2` (each group has start + end)
/// - This allows unlimited capture groups without wasting memory
pub struct ThreadWorklist {
    /// Number of active threads.
    pub count: usize,
    /// Capture stride: slots per thread = (capture_count + 1) * 2.
    pub stride: usize,
    /// Thread states (HOT - accessed every step).
    pub states: Vec<u32>,
    /// Thread flags: priority (bits 0-15), non_greedy (bit 16).
    pub flags: Vec<u32>,
    /// Thread capture slots (COLD - accessed only on transitions).
    /// Layout: captures[thread_idx * stride + slot_idx]
    /// Slot layout per thread: [start0, end0, start1, end1, ...]
    pub captures: Vec<i64>,
    /// Visited bitmap for epsilon closure deduplication.
    pub visited: Vec<u64>,
}

impl ThreadWorklist {
    /// Creates a new worklist with the given capture stride.
    ///
    /// # Arguments
    /// * `capture_count` - Number of capture groups (stride = (capture_count + 1) * 2)
    /// * `state_count` - Number of NFA states (for visited bitmap sizing)
    pub fn new(capture_count: u32, state_count: usize) -> Self {
        let stride = (capture_count as usize + 1) * 2;
        let bitmap_words = state_count.max(1).div_ceil(64);

        Self {
            count: 0,
            stride,
            states: vec![0; MAX_THREADS],
            flags: vec![0; MAX_THREADS],
            captures: vec![-1i64; MAX_THREADS * stride],
            visited: vec![0; bitmap_words],
        }
    }

    /// Clears all threads and visited bitmap.
    #[inline]
    pub fn clear(&mut self) {
        self.count = 0;
        // Clear visited bitmap
        for word in &mut self.visited {
            *word = 0;
        }
        // Reset captures to -1
        for slot in &mut self.captures {
            *slot = -1;
        }
    }

    /// Checks if a state has been visited.
    #[inline]
    pub fn is_visited(&self, state: StateId) -> bool {
        let idx = state as usize;
        let word = idx / 64;
        let bit = idx % 64;
        if word >= self.visited.len() {
            return false; // State beyond bitmap: skip dedup (conservative)
        }
        (self.visited[word] & (1u64 << bit)) != 0
    }

    /// Marks a state as visited.
    #[inline]
    pub fn mark_visited(&mut self, state: StateId) {
        let idx = state as usize;
        let word = idx / 64;
        let bit = idx % 64;
        if word < self.visited.len() {
            self.visited[word] |= 1u64 << bit;
        }
    }

    /// Adds a thread if not at capacity.
    #[inline]
    pub fn add_thread(&mut self, state: StateId, flags: u32) -> Option<usize> {
        if self.count >= MAX_THREADS {
            return None;
        }
        let idx = self.count;
        self.states[idx] = state;
        self.flags[idx] = flags;
        // Captures already initialized to -1 in constructor
        self.count += 1;
        Some(idx)
    }

    /// Gets a capture slot for a thread.
    #[inline]
    pub fn get_capture(&self, thread_idx: usize, slot: usize) -> i64 {
        self.captures[thread_idx * self.stride + slot]
    }

    /// Sets a capture slot for a thread.
    #[inline]
    pub fn set_capture(&mut self, thread_idx: usize, slot: usize, value: i64) {
        self.captures[thread_idx * self.stride + slot] = value;
    }

    /// Copies captures from one thread to another.
    #[inline]
    #[allow(dead_code)]
    pub fn copy_captures(&mut self, from_idx: usize, to_idx: usize) {
        let from_start = from_idx * self.stride;
        let to_start = to_idx * self.stride;
        for i in 0..self.stride {
            self.captures[to_start + i] = self.captures[from_start + i];
        }
    }

    /// Returns the capture stride (slots per thread).
    #[inline]
    #[allow(dead_code)]
    pub fn stride(&self) -> usize {
        self.stride
    }
}

/// Lookaround memoization cache.
///
/// Caches lookaround results to avoid redundant evaluation when multiple
/// threads reach the same lookaround at the same position.
#[repr(C)]
pub struct LookaroundCache {
    /// Number of lookarounds.
    pub count: usize,
    /// Maximum input length for cache sizing.
    pub max_len: usize,
    /// Results bitmap: results[lookaround_id * words_per_pos + pos / 64] & (1 << pos % 64)
    pub results: Vec<u64>,
    /// Computed bitmap: which (lookaround, position) pairs have been evaluated.
    pub computed: Vec<u64>,
}

impl LookaroundCache {
    /// Creates a new cache for the given number of lookarounds and input length.
    pub fn new(lookaround_count: usize, max_input_len: usize) -> Self {
        let words_needed = max_input_len.div_ceil(64);
        let total_words = lookaround_count * words_needed;
        Self {
            count: lookaround_count,
            max_len: max_input_len,
            results: vec![0; total_words],
            computed: vec![0; total_words],
        }
    }

    /// Checks if a lookaround result is cached.
    #[inline]
    #[allow(dead_code)]
    pub fn is_computed(&self, lookaround_id: usize, pos: usize) -> bool {
        if lookaround_id >= self.count || pos >= self.max_len {
            return false;
        }
        let words_per_la = self.max_len.div_ceil(64);
        let word_idx = lookaround_id * words_per_la + pos / 64;
        let bit = pos % 64;
        (self.computed[word_idx] & (1u64 << bit)) != 0
    }

    /// Gets a cached lookaround result.
    #[inline]
    #[allow(dead_code)]
    pub fn get_result(&self, lookaround_id: usize, pos: usize) -> bool {
        if lookaround_id >= self.count || pos >= self.max_len {
            return false;
        }
        let words_per_la = self.max_len.div_ceil(64);
        let word_idx = lookaround_id * words_per_la + pos / 64;
        let bit = pos % 64;
        (self.results[word_idx] & (1u64 << bit)) != 0
    }

    /// Sets a lookaround result.
    #[inline]
    #[allow(dead_code)]
    pub fn set_result(&mut self, lookaround_id: usize, pos: usize, result: bool) {
        if lookaround_id >= self.count || pos >= self.max_len {
            return;
        }
        let words_per_la = self.max_len.div_ceil(64);
        let word_idx = lookaround_id * words_per_la + pos / 64;
        let bit = pos % 64;
        self.computed[word_idx] |= 1u64 << bit;
        if result {
            self.results[word_idx] |= 1u64 << bit;
        }
    }

    /// Clears the cache for reuse.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        for word in &mut self.results {
            *word = 0;
        }
        for word in &mut self.computed {
            *word = 0;
        }
    }
}

/// Runtime context for Tagged NFA execution.
#[allow(dead_code)]
pub struct TaggedNfaContext {
    /// Current position worklist.
    pub current: ThreadWorklist,
    /// Next position worklist.
    pub next: ThreadWorklist,
    /// Best match found (captures) - dynamically sized.
    pub best_captures: Vec<i64>,
    /// Best match end position (-1 if no match).
    pub best_end: i64,
    /// Best match priority (for non-greedy preference).
    pub best_priority: u32,
    /// Lookaround cache.
    pub lookaround_cache: LookaroundCache,
    /// Capture stride for this pattern.
    pub stride: usize,
}

#[allow(dead_code)]
impl TaggedNfaContext {
    /// Creates a new context.
    ///
    /// # Arguments
    /// * `capture_count` - Number of capture groups
    /// * `state_count` - Number of NFA states
    /// * `lookaround_count` - Number of lookarounds
    /// * `max_input_len` - Maximum input length for lookaround cache
    pub fn new(
        capture_count: u32,
        state_count: usize,
        lookaround_count: usize,
        max_input_len: usize,
    ) -> Self {
        let stride = (capture_count as usize + 1) * 2;
        Self {
            current: ThreadWorklist::new(capture_count, state_count),
            next: ThreadWorklist::new(capture_count, state_count),
            best_captures: vec![-1; stride],
            best_end: -1,
            best_priority: 0,
            lookaround_cache: LookaroundCache::new(lookaround_count, max_input_len),
            stride,
        }
    }

    /// Resets context for a new match.
    pub fn reset(&mut self) {
        self.current.clear();
        self.next.clear();
        for slot in &mut self.best_captures {
            *slot = -1;
        }
        self.best_end = -1;
        self.best_priority = 0;
        self.lookaround_cache.clear();
    }

    /// Swaps current and next worklists.
    #[inline]
    pub fn swap_worklists(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
        self.next.clear();
    }

    /// Gets a capture from the best match.
    #[inline]
    pub fn get_best_capture(&self, group: usize) -> Option<(usize, usize)> {
        let start_idx = group * 2;
        let end_idx = group * 2 + 1;
        if start_idx < self.best_captures.len() && end_idx < self.best_captures.len() {
            let start = self.best_captures[start_idx];
            let end = self.best_captures[end_idx];
            if start >= 0 && end >= 0 {
                return Some((start as usize, end as usize));
            }
        }
        None
    }
}

/// Represents a single matching step in a pattern.
/// Can be either a single byte, a character class, repetition, alternation, or capture marker.
#[derive(Debug, Clone)]
pub enum PatternStep {
    /// Match a single byte.
    Byte(u8),
    /// Match any byte in this class (character class with precomputed bitmap).
    ByteClass(ByteClass),
    /// Greedy one-or-more repetition of byte class.
    /// The ByteClass has precomputed bitmap for O(1) matching.
    GreedyPlus(ByteClass),
    /// Greedy zero-or-more repetition of byte class.
    #[allow(dead_code)]
    GreedyStar(ByteClass),
    /// Greedy one-or-more with lookahead: matches as many as possible, then backtracks
    /// until the lookahead succeeds. (byte_class, lookahead_steps, is_positive)
    GreedyPlusLookahead(ByteClass, Vec<PatternStep>, bool),
    /// Greedy zero-or-more with lookahead: matches as many as possible, then backtracks
    /// until the lookahead succeeds. (byte_class, lookahead_steps, is_positive)
    #[allow(dead_code)]
    GreedyStarLookahead(ByteClass, Vec<PatternStep>, bool),
    /// Non-greedy one-or-more repetition of byte class.
    /// Contains the byte class to repeat and the following step(s) that terminate the loop.
    /// The JIT generates code that tries to exit as soon as possible.
    NonGreedyPlus(ByteClass, Box<PatternStep>),
    /// Non-greedy zero-or-more repetition of byte class.
    /// Contains the byte class to repeat and the following step(s) that terminate the loop.
    /// The JIT generates code that tries to exit immediately (zero matches).
    NonGreedyStar(ByteClass, Box<PatternStep>),
    /// Alternation: try each alternative in order.
    /// Each alternative is a sequence of pattern steps.
    Alt(Vec<Vec<PatternStep>>),
    /// Start of a capture group - records current position.
    /// The u32 is the capture group index.
    CaptureStart(u32),
    /// End of a capture group - records current position.
    /// The u32 is the capture group index.
    CaptureEnd(u32),
    /// Unicode codepoint class (e.g., \p{Letter}, \p{Greek}).
    /// Contains codepoint ranges and whether it's negated.
    /// The StateId is the target state after successful match.
    #[allow(dead_code)]
    CodepointClass(CodepointClass, StateId),
    /// Greedy one-or-more repetition of a codepoint class (e.g., \p{Letter}+).
    GreedyCodepointPlus(CodepointClass),
    /// Word boundary assertion (\b).
    /// Matches at position where previous char is word and current is not (or vice versa).
    WordBoundary,
    /// Not word boundary assertion (\B).
    /// Matches at position where both adjacent chars are word or both are non-word.
    NotWordBoundary,
    /// Positive lookahead assertion (?=...).
    /// Contains the inner pattern as a sequence of steps.
    /// For simple patterns, JIT generates inline checking code.
    PositiveLookahead(Vec<PatternStep>),
    /// Negative lookahead assertion (?!...).
    /// Contains the inner pattern as a sequence of steps.
    NegativeLookahead(Vec<PatternStep>),
    /// Positive lookbehind assertion (?<=...).
    /// Contains the inner pattern as a sequence of steps and its minimum length.
    PositiveLookbehind(Vec<PatternStep>, usize),
    /// Negative lookbehind assertion (?<!...).
    /// Contains the inner pattern as a sequence of steps and its minimum length.
    NegativeLookbehind(Vec<PatternStep>, usize),
    /// Backreference to a capture group (\1, \2, etc.).
    /// The u32 is the capture group index (1-based in pattern, stored as-is).
    Backref(u32),
    /// Start of text anchor (\A or ^ without multiline).
    /// Only matches at position 0.
    StartOfText,
    /// End of text anchor (\z or $ without multiline).
    /// Only matches at position == input_len.
    EndOfText,
    /// Start of line anchor (^ with multiline).
    /// Matches at position 0 or after a newline.
    StartOfLine,
    /// End of line anchor ($ with multiline).
    /// Matches at position == input_len or before a newline.
    EndOfLine,
}

/// Helper function to check if a byte is a word character (ASCII: [a-zA-Z0-9_]).
#[inline]
pub fn is_word_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
