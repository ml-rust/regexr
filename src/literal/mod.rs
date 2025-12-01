//! Literal extraction and prefilter module.
//!
//! Extracts literal prefixes/suffixes from patterns for fast prefiltering.
//!
//! # Architecture
//!
//! 1. **Literal Extraction** (`extractor.rs`): Analyzes HIR to find required
//!    literal prefixes/suffixes that must appear in any match.
//!
//! 2. **Prefilter** (`prefilter.rs`): Uses extracted literals to build a
//!    SIMD-accelerated candidate filter (memchr or Teddy).

mod extractor;
mod prefilter;

pub use extractor::*;
pub use prefilter::*;
