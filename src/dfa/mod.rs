//! DFA (Deterministic Finite Automaton) module.
//!
//! Implements lazy and eager DFA construction and execution.
//!
//! ## Engines
//!
//! - `lazy/` - Lazy DFA (on-demand state construction)
//! - `eager/` - Eager DFA (pre-computed states)

pub mod eager;
pub mod lazy;

pub use eager::{EagerDfa, EagerDfaEngine};
pub use lazy::{CharClass, DfaStateId, LazyDfa, LazyDfaEngine, PositionContext};
