//! High-Level Intermediate Representation (HIR) module.
//!
//! Translates the AST into a byte-oriented representation suitable for NFA construction.

mod builder;
mod prefix_opt;
pub mod unicode;
pub mod unicode_data;

pub use builder::*;
pub use prefix_opt::optimize_prefixes;

use crate::error::Result;
use crate::parser::Ast;

/// High-level IR for a regex pattern.
#[derive(Debug, Clone)]
pub struct Hir {
    /// The root expression.
    pub expr: HirExpr,
    /// Properties of the pattern.
    pub props: HirProps,
}

/// Properties derived from analyzing the HIR.
#[derive(Debug, Clone, Default)]
pub struct HirProps {
    /// Whether the pattern contains backreferences.
    pub has_backrefs: bool,
    /// Whether the pattern contains lookarounds.
    pub has_lookaround: bool,
    /// Whether the pattern contains positional anchors (^, $).
    /// These require matching at specific input positions.
    pub has_anchors: bool,
    /// Whether the pattern contains word boundaries (\b, \B).
    /// These can be handled by DFA with position tracking.
    pub has_word_boundary: bool,
    /// Whether the pattern contains non-greedy quantifiers (*?, +?, ??, {n,m}?).
    pub has_non_greedy: bool,
    /// Whether the pattern contains large Unicode character classes.
    /// These cause DFA state explosion and should use PikeVM instead of JIT.
    pub has_large_unicode_class: bool,
    /// Number of capture groups.
    pub capture_count: u32,
    /// Minimum match length in bytes.
    pub min_len: usize,
    /// Maximum match length in bytes (None = unbounded).
    pub max_len: Option<usize>,
    /// Named capture groups: maps name to index.
    pub named_groups: std::collections::HashMap<String, u32>,
    /// If the pattern is a single codepoint class, store the ranges here
    /// for fast codepoint-level matching. This avoids byte-level expansion.
    pub codepoint_class: Option<CodepointClass>,
}

/// A codepoint-level character class (Unicode scalar values).
/// Used for fast matching of patterns like `[α-ω]` or `\p{Greek}`.
#[derive(Debug, Clone)]
pub struct CodepointClass {
    /// Codepoint ranges (sorted, non-overlapping). Each range is (start, end) inclusive.
    pub ranges: Vec<(u32, u32)>,
    /// Whether this class is negated.
    pub negated: bool,
}

impl CodepointClass {
    /// Creates a new codepoint class.
    pub fn new(ranges: Vec<(u32, u32)>, negated: bool) -> Self {
        Self { ranges, negated }
    }

    /// Checks if a codepoint is in this class.
    #[inline]
    pub fn contains(&self, cp: u32) -> bool {
        let in_ranges = self.ranges.binary_search_by(|&(start, end)| {
            if cp < start {
                std::cmp::Ordering::Greater
            } else if cp > end {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        }).is_ok();

        if self.negated { !in_ranges } else { in_ranges }
    }
}

/// An HIR expression node.
#[derive(Debug, Clone)]
pub enum HirExpr {
    /// Empty expression.
    Empty,
    /// A literal byte sequence.
    Literal(Vec<u8>),
    /// A byte class (set of byte ranges).
    Class(HirClass),
    /// A Unicode codepoint class (used for efficient matching of Unicode patterns).
    /// This matches a single UTF-8 encoded codepoint and checks membership.
    UnicodeCpClass(CodepointClass),
    /// Concatenation.
    Concat(Vec<HirExpr>),
    /// Alternation.
    Alt(Vec<HirExpr>),
    /// Repetition.
    Repeat(Box<HirRepeat>),
    /// Capture group.
    Capture(Box<HirCapture>),
    /// Anchor.
    Anchor(HirAnchor),
    /// Lookaround.
    Lookaround(Box<HirLookaround>),
    /// Backreference.
    Backref(u32),
}

/// A byte class - set of byte ranges.
#[derive(Debug, Clone)]
pub struct HirClass {
    /// Byte ranges (sorted, non-overlapping).
    pub ranges: Vec<(u8, u8)>,
    /// Whether this class is negated.
    pub negated: bool,
}

impl HirClass {
    /// Creates a new class.
    pub fn new(ranges: Vec<(u8, u8)>, negated: bool) -> Self {
        Self { ranges, negated }
    }

    /// Creates a class matching any byte.
    pub fn any() -> Self {
        Self {
            ranges: vec![(0, 255)],
            negated: false,
        }
    }

    /// Creates a class matching any byte except newline.
    pub fn dot() -> Self {
        Self {
            ranges: vec![(0, 9), (11, 255)],
            negated: false,
        }
    }
}

/// A repetition in HIR.
#[derive(Debug, Clone)]
pub struct HirRepeat {
    /// The expression being repeated.
    pub expr: HirExpr,
    /// Minimum repetitions.
    pub min: u32,
    /// Maximum repetitions.
    pub max: Option<u32>,
    /// Whether greedy.
    pub greedy: bool,
}

/// A capture group in HIR.
#[derive(Debug, Clone)]
pub struct HirCapture {
    /// Capture group index.
    pub index: u32,
    /// Optional name.
    pub name: Option<String>,
    /// The captured expression.
    pub expr: HirExpr,
}

/// An anchor in HIR.
#[derive(Debug, Clone, Copy)]
pub enum HirAnchor {
    /// Start of text.
    Start,
    /// End of text.
    End,
    /// Start of line.
    StartLine,
    /// End of line.
    EndLine,
    /// Word boundary.
    WordBoundary,
    /// Not word boundary.
    NotWordBoundary,
}

/// A lookaround in HIR.
#[derive(Debug, Clone)]
pub struct HirLookaround {
    /// The lookaround expression.
    pub expr: HirExpr,
    /// The kind of lookaround.
    pub kind: HirLookaroundKind,
}

/// The kind of lookaround.
#[derive(Debug, Clone, Copy)]
pub enum HirLookaroundKind {
    /// Positive lookahead.
    PositiveLookahead,
    /// Negative lookahead.
    NegativeLookahead,
    /// Positive lookbehind.
    PositiveLookbehind,
    /// Negative lookbehind.
    NegativeLookbehind,
}

/// Translates an AST to HIR.
pub fn translate(ast: &Ast) -> Result<Hir> {
    let mut translator = HirTranslator::new();
    translator.translate(ast)
}
