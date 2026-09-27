//! NFA (Nondeterministic Finite Automaton) module.
//!
//! Implements two NFA constructions:
//! - **Thompson's construction**: For PikeVM and DFA (has ε-transitions)
//! - **Glushkov construction**: For Shift-Or (ε-free, position-based)
//!
//! Also provides UTF-8 automata compilation for Unicode character classes.
//!
//! # Tagged NFA
//!
//! The `tagged` submodule provides Tagged NFA execution for patterns with:
//! - Lookaround assertions (lookahead, lookbehind)
//! - Non-greedy quantifiers
//! - Complex capture groups with liveness-optimized copying

mod assertions;
mod glushkov;
mod priority;
mod reverse;
mod state;
pub mod tagged;
mod thompson;
pub mod utf8_automata;

pub use assertions::assertions_at_edges;
pub use glushkov::{
    compile_glushkov, compile_glushkov_wide, BitSet256, BitSet256Iter, ByteSet, GlushkovNfa,
    GlushkovWideNfa, MAX_POSITIONS, MAX_POSITIONS_WIDE,
};
pub use priority::priority_is_irrelevant;
pub(crate) use reverse::reverse;
pub use state::*;
pub use thompson::*;

use crate::error::Result;
use crate::hir::Hir;

/// Whether `pos` satisfies the `$`/`\Z` end anchor (`NfaInstruction::EndOfText`):
/// it holds at the end of the input, or immediately before a single trailing
/// newline (PCRE/Python `$` semantics, non-multiline). `\z` (strict end) is
/// compiled separately and does not use this.
#[inline]
pub fn at_end_or_before_final_newline(input: &[u8], pos: usize) -> bool {
    pos == input.len() || (pos + 1 == input.len() && input[pos] == b'\n')
}

/// Whether `pos` satisfies multiline `^` (`NfaInstruction::StartOfLine`): the
/// start of the input, or just after a newline.
#[inline]
pub fn at_line_start(input: &[u8], pos: usize) -> bool {
    pos == 0 || input.get(pos - 1) == Some(&b'\n')
}

/// Whether `pos` satisfies multiline `$` (`NfaInstruction::EndOfLine`): the end
/// of the input, or just before a newline.
#[inline]
pub fn at_line_end(input: &[u8], pos: usize) -> bool {
    pos == input.len() || input.get(pos) == Some(&b'\n')
}

/// Whether `pos` satisfies `\b`: exactly one side of the position is a word
/// byte. `\B` is its negation.
///
/// Every engine that implements the assertion shares this definition, so a
/// change to what counts as a word byte cannot reach one engine and miss another.
#[inline]
pub fn is_word_boundary(input: &[u8], pos: usize) -> bool {
    let before = pos
        .checked_sub(1)
        .and_then(|i| input.get(i))
        .is_some_and(|&b| crate::hir::unicode::is_word_byte(b));
    let after = input
        .get(pos)
        .is_some_and(|&b| crate::hir::unicode::is_word_byte(b));
    before != after
}

/// Whether `pos` is a UTF-8 codepoint boundary in `input` (start of a codepoint,
/// or the end). Matches are only attempted at boundaries so a zero-width/optional
/// construct can't match in the middle of a multi-byte codepoint — the Unicode
/// semantics of PCRE/Python on valid UTF-8, which regexr targets for tokenization.
#[inline]
pub fn is_utf8_boundary(input: &[u8], pos: usize) -> bool {
    pos >= input.len() || (input[pos] & 0xC0) != 0x80
}

/// Compiles an HIR to a Thompson NFA.
pub fn compile(hir: &Hir) -> Result<Nfa> {
    let mut builder = NfaBuilder::new();
    builder.build(hir)
}
