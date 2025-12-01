//! JIT compilation for Shift-Or engine.
//!
//! Compiles the Shift-Or bit-parallel NFA to native x86-64 code.
//! This eliminates interpreter overhead and keeps all state in registers.

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "x86_64")]
mod jit;

#[cfg(target_arch = "x86_64")]
pub use jit::JitShiftOr;
