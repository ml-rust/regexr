//! JIT compilation for Tagged NFA execution.
//!
//! This module provides JIT-compiled execution for Tagged NFA patterns.
//! It is feature-gated behind `#[cfg(all(feature = "jit", target_arch = "x86_64"))]`.
//!
//! # Architecture
//!
//! The JIT compiler generates x86-64 code that mirrors the TaggedNfaInterpreter:
//! - Same algorithm, same semantics
//! - Uses Structure-of-Arrays (SoA) layout for cache efficiency
//! - Sparse capture copying based on liveness analysis
//!
//! # Module Organization
//!
//! - `jit.rs` - TaggedNfaJit struct and public API
//! - `x86_64.rs` - dynasm-based x86-64 code generation
//! - `helpers.rs` - JIT context and extern helper functions

mod helpers;
mod jit;
mod x86_64;

pub use helpers::JitContext;
pub use jit::{compile_tagged_nfa, compile_tagged_nfa_with_liveness, TaggedNfaJit};
