//! JIT compilation module for regex patterns.
//!
//! This module compiles DFA states to native x86-64 machine code using dynasm.
//! The JIT compiler provides significant performance improvements for repeated
//! pattern matching operations.
//!
//! # Features
//!
//! - **W^X Compliant**: Generated code is never RWX (read-write-execute)
//! - **Optimized**: 16-byte alignment for hot loops, efficient transition encoding
//! - **Safe**: Memory-safe API wrapping unsafe JIT execution
//!
//! # Architecture Support
//!
//! Currently only x86-64 is supported. The module is conditionally compiled
//! based on the `jit` feature and target architecture.
//!
//! # Example
//!
//! ```no_run
//! # use regexr::jit::*;
//! # use regexr::dfa::LazyDfa;
//! # use regexr::nfa::compile;
//! # use regexr::hir::translate;
//! # use regexr::parser::parse;
//! #
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Parse and compile a pattern
//! let ast = parse("abc")?;
//! let hir = translate(&ast)?;
//! let nfa = compile(&hir)?;
//! let mut dfa = LazyDfa::new(nfa);
//!
//! // Compile to native code
//! let jit = compile_dfa(&mut dfa)?;
//!
//! // Execute the compiled code
//! assert!(jit.is_match(b"abc"));
//! assert!(!jit.is_match(b"xyz"));
//! # Ok(())
//! # }
//! ```

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
mod codegen;

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
mod x86_64;

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub use codegen::{CompiledRegex, JitCompiler, MaterializedDfa, MaterializedState};

// Re-export liveness types from nfa::tagged (the canonical location)
#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub use crate::nfa::tagged::liveness::{
    analyze_liveness, CaptureBitSet, NfaLiveness, StateLiveness,
};

// Re-export TaggedNfaJit from nfa::tagged::jit (the canonical location)
#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub use crate::nfa::tagged::jit::{compile_tagged_nfa, TaggedNfaJit};

// Re-export BacktrackingJit from vm::backtracking::jit (the canonical location)
#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub use crate::vm::backtracking::jit::{compile_backtracking, BacktrackingJit};

// Re-export JitShiftOr from vm::shift_or::jit (the canonical location)
#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub use crate::vm::shift_or::jit::JitShiftOr;

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
use crate::dfa::LazyDfa;

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
use crate::error::Result;

/// Compiles a LazyDFA to native x86-64 machine code.
///
/// This is the main entry point for JIT compilation. It takes a LazyDFA
/// and returns a CompiledRegex that can be executed directly on the CPU.
///
/// # Platform Support
///
/// This function is only available on x86-64 platforms with the `jit` feature enabled.
///
/// # Errors
///
/// Returns an error if:
/// - DFA materialization fails
/// - Assembly generation fails
/// - Code finalization fails
///
/// # Example
///
/// ```no_run
/// # use regexr::jit::compile_dfa;
/// # use regexr::dfa::LazyDfa;
/// # use regexr::nfa::compile as compile_nfa;
/// # use regexr::hir::translate;
/// # use regexr::parser::parse;
/// #
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let ast = parse("a+")?;
/// let hir = translate(&ast)?;
/// let nfa = compile_nfa(&hir)?;
/// let mut dfa = LazyDfa::new(nfa);
///
/// let compiled = compile_dfa(&mut dfa)?;
/// assert!(compiled.is_match(b"aaa"));
/// # Ok(())
/// # }
/// ```
#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub fn compile_dfa(dfa: &mut LazyDfa) -> Result<CompiledRegex> {
    let compiler = JitCompiler::new();
    compiler.compile_dfa(dfa)
}

/// Returns true if JIT compilation is available on this platform.
///
/// JIT is only available on x86-64 with the `jit` feature enabled.
///
/// # Example
///
/// ```
/// # use regexr::jit::is_available;
/// if is_available() {
///     println!("JIT compilation is supported!");
/// } else {
///     println!("JIT compilation is not available on this platform.");
/// }
/// ```
pub const fn is_available() -> bool {
    cfg!(all(feature = "jit", target_arch = "x86_64"))
}

/// Returns the target architecture for JIT compilation.
///
/// Returns `Some("x86_64")` if JIT is available, `None` otherwise.
pub const fn target_arch() -> Option<&'static str> {
    if is_available() {
        Some("x86_64")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_availability() {
        // Test should pass on all platforms
        let available = is_available();

        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        assert!(available);

        #[cfg(not(all(feature = "jit", target_arch = "x86_64")))]
        assert!(!available);
    }

    #[test]
    fn test_target_arch() {
        let arch = target_arch();

        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        assert_eq!(arch, Some("x86_64"));

        #[cfg(not(all(feature = "jit", target_arch = "x86_64")))]
        assert_eq!(arch, None);
    }
}
