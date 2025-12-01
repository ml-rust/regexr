//! Interpreter for Shift-Or (Bitap) algorithm.
//!
//! A bit-parallel NFA simulation that runs entirely in CPU registers.
//! Only works for patterns with ≤64 character positions.

mod matcher;

pub use matcher::ShiftOrInterpreter;
