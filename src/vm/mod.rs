//! Virtual machine implementations for regex execution.
//!
//! ## Engines
//!
//! - `shift_or/` - Shift-Or (Bitap) bit-parallel matcher
//! - `backtracking/` - Backtracking VM for backreferences
//! - `pike/` - PikeVM parallel NFA simulation

pub mod backtracking;
mod codepoint_class;
pub mod pike;
pub mod shift_or;

pub use codepoint_class::*;

// Re-export key types from pike module
pub use pike::{PikeVm, PikeVmContext, PikeVmEngine};

// Re-export key types from shift_or module
pub use shift_or::{
    is_shift_or_compatible, is_shift_or_wide_compatible, ShiftOr, ShiftOrEngine,
    ShiftOrInterpreter, ShiftOrWide,
};

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub use shift_or::JitShiftOr;

// Re-export key types from backtracking module
pub use backtracking::{BacktrackingEngine, BacktrackingVm};

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub use backtracking::{compile_backtracking, BacktrackingJit};
