//! JIT compilation for Shift-Or engine.
//!
//! Compiles the Shift-Or bit-parallel NFA to native code.
//! This eliminates interpreter overhead and keeps all state in registers.
//!
//! # Architecture Support
//!
//! - **x86_64**: Uses dynasm for code generation
//! - **aarch64**: Uses dynasm for code generation

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

mod jit;

pub use jit::JitShiftOr;
