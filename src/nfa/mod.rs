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

mod glushkov;
mod state;
mod thompson;
pub mod utf8_automata;
pub mod tagged;

pub use glushkov::{
    compile_glushkov, compile_glushkov_wide, BitSet256, BitSet256Iter, ByteSet, GlushkovNfa,
    GlushkovWideNfa, MAX_POSITIONS, MAX_POSITIONS_WIDE,
};
pub use state::*;
pub use thompson::*;

use crate::error::Result;
use crate::hir::Hir;

/// Compiles an HIR to a Thompson NFA.
pub fn compile(hir: &Hir) -> Result<Nfa> {
    let mut builder = NfaBuilder::new();
    builder.build(hir)
}
