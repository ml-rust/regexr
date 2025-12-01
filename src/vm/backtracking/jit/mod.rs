//! Backtracking JIT compiler.
//!
//! Compiles HIR directly to native x86-64 machine code for patterns with backreferences.

mod jit;
mod x86_64;

pub use jit::{BacktrackingJit, compile_backtracking};
