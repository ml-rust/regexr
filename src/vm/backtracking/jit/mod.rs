//! Backtracking JIT compiler.
//!
//! Compiles HIR directly to native machine code for patterns with backreferences.
//!
//! # Architecture Support
//!
//! - **x86_64**: Uses dynasm for code generation
//! - **aarch64**: Uses dynasm for code generation

#[allow(clippy::module_inception)]
mod jit;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

pub use jit::{compile_backtracking, BacktrackingJit};
