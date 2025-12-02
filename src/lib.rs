//! regexr - A high-performance regex engine built from scratch
//!
//! This crate provides a regex engine with multiple execution backends:
//! - PikeVM: Thread-based NFA simulation (supports backreferences, lookaround)
//! - Shift-Or: Bit-parallel NFA for patterns with ≤64 states
//! - Lazy DFA: On-demand determinization with caching
//! - JIT: Native x86-64 code generation (optional, requires `jit` feature)
//! - SIMD: AVX2-accelerated literal search (optional, requires `simd` feature)

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod dfa;
pub mod engine;
pub mod error;
pub mod hir;
pub mod literal;
pub mod nfa;
pub mod parser;
pub mod vm;

#[cfg(feature = "jit")]
pub mod jit;

#[cfg(feature = "simd")]
pub mod simd;

pub use error::{Error, Result};

use engine::CompiledRegex;
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration options for regex compilation.
#[derive(Debug, Clone, Default)]
pub struct RegexBuilder {
    pattern: String,
    /// Whether to enable JIT compilation.
    jit: bool,
    /// Whether to enable prefix optimization for large alternations.
    /// This is critical for tokenizer-style patterns with many literal alternatives.
    optimize_prefixes: bool,
}

impl RegexBuilder {
    /// Creates a new RegexBuilder with the given pattern.
    pub fn new(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
            jit: false,
            optimize_prefixes: false,
        }
    }

    /// Enables or disables JIT compilation.
    ///
    /// When enabled, the regex will be compiled to native machine code
    /// for maximum performance. This is ideal for patterns that will be
    /// matched many times (e.g., tokenization).
    ///
    /// JIT compilation has higher upfront cost but faster matching.
    /// Only available on x86-64 with the `jit` feature enabled.
    ///
    /// # Example
    ///
    /// ```
    /// use regexr::RegexBuilder;
    ///
    /// let re = RegexBuilder::new(r"\w+")
    ///     .jit(true)
    ///     .build()
    ///     .unwrap();
    /// assert!(re.is_match("hello"));
    /// ```
    pub fn jit(mut self, enabled: bool) -> Self {
        self.jit = enabled;
        self
    }

    /// Enables or disables prefix optimization for large alternations.
    ///
    /// When enabled, large alternations of literals (like `(token1|token2|...|tokenN)`)
    /// will be optimized by merging common prefixes into a trie structure.
    /// This reduces the number of active NFA threads from O(vocabulary_size) to O(token_length).
    ///
    /// This is critical for tokenizer-style patterns with many literal alternatives.
    ///
    /// # Example
    ///
    /// ```
    /// use regexr::RegexBuilder;
    ///
    /// // Pattern with many tokens sharing common prefixes
    /// let re = RegexBuilder::new(r"(the|that|them|they|this)")
    ///     .optimize_prefixes(true)
    ///     .build()
    ///     .unwrap();
    /// assert!(re.is_match("the"));
    /// ```
    pub fn optimize_prefixes(mut self, enabled: bool) -> Self {
        self.optimize_prefixes = enabled;
        self
    }

    /// Builds the regex with the configured options.
    pub fn build(self) -> Result<Regex> {
        let ast = parser::parse(&self.pattern)?;
        let mut hir_result = hir::translate(&ast)?;

        // Apply prefix optimization if enabled
        if self.optimize_prefixes {
            hir_result = hir::optimize_prefixes(hir_result);
        }

        let named_groups = Arc::new(hir_result.props.named_groups.clone());

        let inner = if self.jit {
            engine::compile_with_jit(&hir_result)?
        } else {
            // Use compile_from_hir for optimal engine selection (ShiftOr, LazyDfa, etc.)
            engine::compile_from_hir(&hir_result)?
        };

        Ok(Regex {
            inner,
            pattern: self.pattern,
            named_groups,
        })
    }
}

/// A compiled regular expression.
#[derive(Debug)]
pub struct Regex {
    inner: CompiledRegex,
    pattern: String,
    /// Named capture groups: maps name to index.
    named_groups: Arc<HashMap<String, u32>>,
}

impl Regex {
    /// Compiles a regular expression pattern.
    ///
    /// # Errors
    /// Returns an error if the pattern is invalid.
    pub fn new(pattern: &str) -> Result<Regex> {
        let ast = parser::parse(pattern)?;
        let hir = hir::translate(&ast)?;
        let named_groups = Arc::new(hir.props.named_groups.clone());
        // Use HIR-based compilation to enable Shift-Or and prefilters
        let inner = engine::compile_from_hir(&hir)?;

        Ok(Regex {
            inner,
            pattern: pattern.to_string(),
            named_groups,
        })
    }

    /// Returns the names of all named capture groups.
    pub fn capture_names(&self) -> impl Iterator<Item = &str> {
        self.named_groups.keys().map(|s| s.as_str())
    }

    /// Returns the original pattern string.
    pub fn as_str(&self) -> &str {
        &self.pattern
    }

    /// Returns true if the regex matches anywhere in the text.
    pub fn is_match(&self, text: &str) -> bool {
        self.inner.is_match(text.as_bytes())
    }

    /// Returns the first match in the text.
    pub fn find<'t>(&self, text: &'t str) -> Option<Match<'t>> {
        self.inner
            .find(text.as_bytes())
            .map(|(start, end)| Match { text, start, end })
    }

    /// Returns an iterator over all non-overlapping matches.
    pub fn find_iter<'a>(&'a self, text: &'a str) -> Matches<'a> {
        Matches::new(self, text)
    }

    /// Returns the capture groups for the first match.
    pub fn captures<'t>(&self, text: &'t str) -> Option<Captures<'t>> {
        self.inner.captures(text.as_bytes()).map(|slots| Captures {
            text,
            slots,
            named_groups: Arc::clone(&self.named_groups),
        })
    }

    /// Returns an iterator over all non-overlapping captures.
    pub fn captures_iter<'r, 't>(&'r self, text: &'t str) -> CapturesIter<'r, 't> {
        CapturesIter {
            regex: self,
            text,
            last_end: 0,
        }
    }

    /// Replaces the first match with the replacement string.
    pub fn replace<'t>(&self, text: &'t str, rep: &str) -> std::borrow::Cow<'t, str> {
        match self.find(text) {
            None => std::borrow::Cow::Borrowed(text),
            Some(m) => {
                let mut result = String::with_capacity(text.len() + rep.len());
                result.push_str(&text[..m.start()]);
                result.push_str(rep);
                result.push_str(&text[m.end()..]);
                std::borrow::Cow::Owned(result)
            }
        }
    }

    /// Returns the name of the engine being used (for debugging).
    pub fn engine_name(&self) -> &'static str {
        self.inner.engine_name()
    }

    /// Replaces all matches with the replacement string.
    pub fn replace_all<'t>(&self, text: &'t str, rep: &str) -> std::borrow::Cow<'t, str> {
        let mut last_end = 0;
        let mut result = String::new();
        let mut had_match = false;

        for m in self.find_iter(text) {
            had_match = true;
            result.push_str(&text[last_end..m.start()]);
            result.push_str(rep);
            last_end = m.end();
        }

        if !had_match {
            std::borrow::Cow::Borrowed(text)
        } else {
            result.push_str(&text[last_end..]);
            std::borrow::Cow::Owned(result)
        }
    }
}

/// A single match in the text.
#[derive(Debug, Clone, Copy)]
pub struct Match<'t> {
    text: &'t str,
    start: usize,
    end: usize,
}

impl<'t> Match<'t> {
    /// Returns the start byte offset of the match.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Returns the end byte offset of the match.
    pub fn end(&self) -> usize {
        self.end
    }

    /// Returns the matched text.
    pub fn as_str(&self) -> &'t str {
        &self.text[self.start..self.end]
    }

    /// Returns the byte range of the match.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start..self.end
    }

    /// Returns the length of the match in bytes.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Returns true if the match is empty.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// An iterator over all non-overlapping matches.
pub struct Matches<'a> {
    inner: MatchesInner<'a>,
    text: &'a str,
}

impl<'a> std::fmt::Debug for Matches<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Matches")
            .field("text_len", &self.text.len())
            .finish_non_exhaustive()
    }
}

/// Internal iterator state - either uses TeddyFull fast path or generic find().
enum MatchesInner<'a> {
    /// Fast path: use TeddyFull prefilter iterator directly.
    TeddyFull(literal::FullMatchIter<'a, 'a>),
    /// Generic path: call find() repeatedly.
    Generic { regex: &'a Regex, last_end: usize },
}

impl<'a> Matches<'a> {
    /// Creates a new matches iterator.
    fn new(regex: &'a Regex, text: &'a str) -> Self {
        let inner = if regex.inner.is_full_match_prefilter() {
            // Fast path: use Teddy iterator directly
            MatchesInner::TeddyFull(regex.inner.find_full_matches(text.as_bytes()))
        } else {
            // Generic path
            MatchesInner::Generic { regex, last_end: 0 }
        };
        Matches { inner, text }
    }
}

impl<'a> Iterator for Matches<'a> {
    type Item = Match<'a>;

    fn next(&mut self) -> Option<Match<'a>> {
        match &mut self.inner {
            MatchesInner::TeddyFull(iter) => {
                // Fast path: get match directly from Teddy iterator
                iter.next().map(|(start, end)| Match {
                    text: self.text,
                    start,
                    end,
                })
            }
            MatchesInner::Generic { regex, last_end } => {
                if *last_end > self.text.len() {
                    return None;
                }

                let search_text = &self.text[*last_end..];
                match regex.inner.find(search_text.as_bytes()) {
                    None => None,
                    Some((start, end)) => {
                        let abs_start = *last_end + start;
                        let abs_end = *last_end + end;

                        // Advance past the match, but ensure progress on empty matches
                        // For empty matches, advance to the next UTF-8 character boundary
                        *last_end = if abs_start == abs_end {
                            // Find the next char boundary after abs_end
                            let remaining = &self.text[abs_end..];
                            let next_char_len = remaining
                                .chars()
                                .next()
                                .map(|c| c.len_utf8())
                                .unwrap_or(1);
                            abs_end + next_char_len
                        } else {
                            abs_end
                        };

                        Some(Match {
                            text: self.text,
                            start: abs_start,
                            end: abs_end,
                        })
                    }
                }
            }
        }
    }
}

/// An iterator over all non-overlapping captures.
#[derive(Debug)]
pub struct CapturesIter<'r, 't> {
    regex: &'r Regex,
    text: &'t str,
    last_end: usize,
}

impl<'r, 't> Iterator for CapturesIter<'r, 't> {
    type Item = Captures<'t>;

    fn next(&mut self) -> Option<Captures<'t>> {
        if self.last_end > self.text.len() {
            return None;
        }

        let search_text = &self.text[self.last_end..];
        match self.regex.inner.captures(search_text.as_bytes()) {
            None => None,
            Some(slots) => {
                // Get the full match bounds (slot 0)
                let (start, end) = slots.first().and_then(|s| *s)?;
                let offset = self.last_end;

                // Advance past the match, but ensure progress on empty matches
                // For empty matches, advance to the next UTF-8 character boundary
                let abs_end = offset + end;
                self.last_end = if start == end {
                    let remaining = &self.text[abs_end..];
                    let next_char_len = remaining
                        .chars()
                        .next()
                        .map(|c| c.len_utf8())
                        .unwrap_or(1);
                    abs_end + next_char_len
                } else {
                    abs_end
                };

                // Adjust all slot positions to absolute positions
                let adjusted_slots: Vec<_> = slots
                    .into_iter()
                    .map(|slot| slot.map(|(s, e)| (offset + s, offset + e)))
                    .collect();

                Some(Captures {
                    text: self.text,
                    slots: adjusted_slots,
                    named_groups: Arc::clone(&self.regex.named_groups),
                })
            }
        }
    }
}

/// Captured groups from a regex match.
#[derive(Debug, Clone)]
pub struct Captures<'t> {
    text: &'t str,
    slots: Vec<Option<(usize, usize)>>,
    named_groups: Arc<HashMap<String, u32>>,
}

impl<'t> Captures<'t> {
    /// Returns the number of capture groups (including group 0 for the full match).
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Returns true if there are no captures.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Returns the capture group at the given index.
    pub fn get(&self, i: usize) -> Option<Match<'t>> {
        self.slots.get(i).and_then(|slot| {
            slot.map(|(start, end)| Match {
                text: self.text,
                start,
                end,
            })
        })
    }

    /// Returns the capture group with the given name.
    pub fn name(&self, name: &str) -> Option<Match<'t>> {
        self.named_groups
            .get(name)
            .and_then(|&idx| self.get(idx as usize))
    }
}

impl<'t> std::ops::Index<usize> for Captures<'t> {
    type Output = str;

    fn index(&self, i: usize) -> &str {
        self.get(i)
            .map(|m| m.as_str())
            .unwrap_or_else(|| panic!("no capture group at index {}", i))
    }
}

impl<'t> std::ops::Index<&str> for Captures<'t> {
    type Output = str;

    fn index(&self, name: &str) -> &str {
        self.name(name)
            .map(|m| m.as_str())
            .unwrap_or_else(|| panic!("no capture group named '{}'", name))
    }
}
