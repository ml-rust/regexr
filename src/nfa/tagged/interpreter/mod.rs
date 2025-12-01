//! Interpreter implementations for Tagged NFA execution.
//!
//! This module provides two interpreters:
//! - `StepInterpreter` - Fast step-based matching for simple patterns
//! - `TaggedNfaInterpreter` - Full Thompson NFA simulation with sparse capture copying

mod step_interpreter;
mod nfa_interpreter;

pub use step_interpreter::StepInterpreter;
pub use nfa_interpreter::TaggedNfaInterpreter;
