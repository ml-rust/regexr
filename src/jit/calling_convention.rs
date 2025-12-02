//! Platform-specific calling convention support for JIT code generation.
//!
//! This module provides macros and helpers to generate code that works with both:
//! - **System V AMD64 ABI** (Linux, macOS, BSD): Args in RDI, RSI, RDX, RCX, R8, R9
//! - **Microsoft x64 ABI** (Windows): Args in RCX, RDX, R8, R9
//!
//! # Key Differences
//!
//! | Aspect | System V (Unix) | Microsoft x64 (Windows) |
//! |--------|-----------------|-------------------------|
//! | Arg 1 | RDI | RCX |
//! | Arg 2 | RSI | RDX |
//! | Arg 3 | RDX | R8 |
//! | Arg 4 | RCX | R9 |
//! | Callee-saved | RBX, RBP, R12-R15 | RBX, RBP, RDI, RSI, R12-R15 |
//! | Shadow space | None | 32 bytes |
//!
//! # Usage
//!
//! All JIT modules use RDI and RSI internally for position and base pointer.
//! The prologue handles moving arguments from the platform's calling convention
//! to these internal registers. On Windows, RDI and RSI must also be saved/restored
//! since they are callee-saved.

use dynasm::dynasm;
use dynasmrt::x64::Assembler;

/// Emits the platform-specific function prologue.
///
/// After this prologue:
/// - `rdi` = first argument (input pointer)
/// - `rsi` = second argument (length)
///
/// On Windows, this also saves RDI and RSI (callee-saved) to the stack.
#[cfg(target_os = "windows")]
pub fn emit_abi_prologue(asm: &mut Assembler) {
    dynasm!(asm
        // Windows x64: args come in RCX, RDX
        // RDI and RSI are callee-saved on Windows, so we must preserve them
        ; push rdi
        ; push rsi
        // Move arguments to System V registers for internal use
        ; mov rdi, rcx  // arg1: input ptr
        ; mov rsi, rdx  // arg2: length
    );
}

/// Emits the platform-specific function prologue.
///
/// After this prologue:
/// - `rdi` = first argument (input pointer)
/// - `rsi` = second argument (length)
///
/// On Unix (System V ABI), arguments are already in the correct registers.
#[cfg(not(target_os = "windows"))]
pub fn emit_abi_prologue(asm: &mut Assembler) {
    // System V AMD64: args already in RDI, RSI - nothing to do
    let _ = asm;
}

/// Emits the platform-specific function epilogue before return.
///
/// On Windows, this restores RDI and RSI from the stack.
#[cfg(target_os = "windows")]
pub fn emit_abi_epilogue(asm: &mut Assembler) {
    dynasm!(asm
        // Restore callee-saved registers
        ; pop rsi
        ; pop rdi
    );
}

/// Emits the platform-specific function epilogue before return.
///
/// On Unix (System V ABI), no special cleanup is needed.
#[cfg(not(target_os = "windows"))]
pub fn emit_abi_epilogue(asm: &mut Assembler) {
    // System V AMD64: nothing to restore
    let _ = asm;
}

/// Emits prologue for functions that also save R13 (word boundary patterns).
///
/// On Windows: saves RDI, RSI, R13
/// On Unix: saves R13
#[cfg(target_os = "windows")]
pub fn emit_abi_prologue_with_r13(asm: &mut Assembler) {
    dynasm!(asm
        // Windows x64: RDI, RSI, R13 all need saving
        ; push rdi
        ; push rsi
        ; push r13
        // Move arguments to System V registers
        ; mov rdi, rcx
        ; mov rsi, rdx
    );
}

/// Emits prologue for functions that also save R13 (word boundary patterns).
#[cfg(not(target_os = "windows"))]
pub fn emit_abi_prologue_with_r13(asm: &mut Assembler) {
    dynasm!(asm
        // System V: only R13 needs saving (callee-saved)
        ; push r13
    );
}

/// Emits epilogue for functions that saved R13.
#[cfg(target_os = "windows")]
pub fn emit_abi_epilogue_with_r13(asm: &mut Assembler) {
    dynasm!(asm
        ; pop r13
        ; pop rsi
        ; pop rdi
    );
}

/// Emits epilogue for functions that saved R13.
#[cfg(not(target_os = "windows"))]
pub fn emit_abi_epilogue_with_r13(asm: &mut Assembler) {
    dynasm!(asm
        ; pop r13
    );
}

/// Returns whether the current platform is Windows.
#[inline]
pub const fn is_windows() -> bool {
    cfg!(target_os = "windows")
}

/// Returns the calling convention name for the current platform.
#[inline]
pub const fn calling_convention_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "Microsoft x64"
    } else {
        "System V AMD64"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calling_convention_name() {
        let name = calling_convention_name();
        #[cfg(target_os = "windows")]
        assert_eq!(name, "Microsoft x64");
        #[cfg(not(target_os = "windows"))]
        assert_eq!(name, "System V AMD64");
    }
}
