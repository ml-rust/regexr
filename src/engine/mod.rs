//! Engine module - coordinates regex execution.
//!
//! Selects the optimal execution strategy based on pattern properties.

mod executor;
mod selector;

pub use executor::*;
pub use selector::*;
