//! Backtracking interpreter.
//!
//! A PCRE-style backtracking interpreter that executes bytecode compiled from HIR.

mod vm;

pub use vm::BacktrackingVm;
