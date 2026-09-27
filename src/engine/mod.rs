//! Engine module - coordinates regex execution.
//!
//! Selects the optimal execution strategy based on pattern properties.

mod dfa_pool;
mod executor;
mod linear;
mod selector;

pub use executor::*;
pub use selector::*;
