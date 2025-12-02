//! Platform-specific calling convention support for JIT code generation.
//!
//! This module provides macros and helpers to generate code that works with:
//!
//! ## x86_64 Platforms
//! - **System V AMD64 ABI** (Linux, macOS, BSD): Args in RDI, RSI, RDX, RCX, R8, R9
//! - **Microsoft x64 ABI** (Windows): Args in RCX, RDX, R8, R9
//!
//! ## ARM64 Platforms (AAPCS64)
//! - **All platforms** (Linux, macOS, Windows): Args in X0, X1, X2, X3, X4, X5, X6, X7
//!
//! # x86_64 Key Differences
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
//! # ARM64 (AAPCS64) - Same on all platforms
//!
//! | Aspect | Value |
//! |--------|-------|
//! | Args 1-8 | X0-X7 |
//! | Return | X0 (X0:X1 for 128-bit) |
//! | Callee-saved | X19-X28, X29 (FP), X30 (LR) |
//! | Stack alignment | 16 bytes |
//!
//! # Usage
//!
//! All JIT modules use consistent internal registers. The prologue handles
//! moving arguments from the platform's calling convention to internal registers.

// ============================================================================
// x86_64 Implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
use dynasm::dynasm;
#[cfg(target_arch = "x86_64")]
use dynasmrt::x64::Assembler;

/// Emits the platform-specific function prologue for x86_64.
///
/// After this prologue:
/// - `rdi` = first argument (input pointer)
/// - `rsi` = second argument (length)
///
/// On Windows, this also saves RDI and RSI (callee-saved) to the stack.
#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
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

/// Emits the platform-specific function prologue for x86_64.
///
/// After this prologue:
/// - `rdi` = first argument (input pointer)
/// - `rsi` = second argument (length)
///
/// On Unix (System V ABI), arguments are already in the correct registers.
#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
pub fn emit_abi_prologue(asm: &mut Assembler) {
    // System V AMD64: args already in RDI, RSI - nothing to do
    let _ = asm;
}

/// Emits the platform-specific function epilogue before return for x86_64.
///
/// On Windows, this restores RDI and RSI from the stack.
#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
pub fn emit_abi_epilogue(asm: &mut Assembler) {
    dynasm!(asm
        // Restore callee-saved registers
        ; pop rsi
        ; pop rdi
    );
}

/// Emits the platform-specific function epilogue before return for x86_64.
///
/// On Unix (System V ABI), no special cleanup is needed.
#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
pub fn emit_abi_epilogue(asm: &mut Assembler) {
    // System V AMD64: nothing to restore
    let _ = asm;
}

/// Emits prologue for functions that also save R13 (word boundary patterns).
///
/// On Windows: saves RDI, RSI, R13
/// On Unix: saves R13
#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
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
#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
pub fn emit_abi_prologue_with_r13(asm: &mut Assembler) {
    dynasm!(asm
        // System V: only R13 needs saving (callee-saved)
        ; push r13
    );
}

/// Emits epilogue for functions that saved R13.
#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
pub fn emit_abi_epilogue_with_r13(asm: &mut Assembler) {
    dynasm!(asm
        ; pop r13
        ; pop rsi
        ; pop rdi
    );
}

/// Emits epilogue for functions that saved R13.
#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
pub fn emit_abi_epilogue_with_r13(asm: &mut Assembler) {
    dynasm!(asm
        ; pop r13
    );
}

// ============================================================================
// ARM64 (AArch64) Implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
use dynasm::dynasm;
#[cfg(target_arch = "aarch64")]
use dynasmrt::aarch64::Assembler as Aarch64Assembler;

/// Emits the function prologue for ARM64 (AAPCS64).
///
/// AAPCS64 is used on all ARM64 platforms (Linux, macOS, Windows).
///
/// After this prologue:
/// - `x19` = first argument (input pointer, moved from x0)
/// - `x20` = second argument (length, moved from x1)
///
/// Saves X19, X20 (callee-saved) to the stack.
#[cfg(target_arch = "aarch64")]
pub fn emit_abi_prologue_aarch64(asm: &mut Aarch64Assembler) {
    dynasm!(asm
        ; .arch aarch64
        // Save callee-saved registers we'll use internally
        // X19-X20 for input ptr and length
        ; stp x19, x20, [sp, #-16]!
        // Move arguments to callee-saved registers for internal use
        ; mov x19, x0  // x19 = input ptr
        ; mov x20, x1  // x20 = length
    );
}

/// Emits the function epilogue for ARM64 (AAPCS64).
///
/// Restores X19, X20 from the stack.
#[cfg(target_arch = "aarch64")]
pub fn emit_abi_epilogue_aarch64(asm: &mut Aarch64Assembler) {
    dynasm!(asm
        ; .arch aarch64
        // Restore callee-saved registers
        ; ldp x19, x20, [sp], #16
    );
}

/// Emits prologue for ARM64 functions that need additional callee-saved registers.
///
/// Saves X19-X24 (6 registers) for complex patterns.
#[cfg(target_arch = "aarch64")]
pub fn emit_abi_prologue_full_aarch64(asm: &mut Aarch64Assembler) {
    dynasm!(asm
        ; .arch aarch64
        // Save frame pointer and link register
        ; stp x29, x30, [sp, #-16]!
        ; mov x29, sp
        // Save callee-saved registers we'll use
        ; stp x19, x20, [sp, #-16]!
        ; stp x21, x22, [sp, #-16]!
        ; stp x23, x24, [sp, #-16]!
        // Move arguments to callee-saved registers
        ; mov x19, x0  // x19 = input ptr
        ; mov x20, x1  // x20 = length
    );
}

/// Emits epilogue for ARM64 functions that saved full register set.
#[cfg(target_arch = "aarch64")]
pub fn emit_abi_epilogue_full_aarch64(asm: &mut Aarch64Assembler) {
    dynasm!(asm
        ; .arch aarch64
        // Restore callee-saved registers in reverse order
        ; ldp x23, x24, [sp], #16
        ; ldp x21, x22, [sp], #16
        ; ldp x19, x20, [sp], #16
        // Restore frame pointer and link register
        ; ldp x29, x30, [sp], #16
    );
}

// ============================================================================
// Common helpers
// ============================================================================

/// Returns whether the current platform is Windows.
#[inline]
pub const fn is_windows() -> bool {
    cfg!(target_os = "windows")
}

/// Returns whether the current architecture is ARM64.
#[inline]
pub const fn is_aarch64() -> bool {
    cfg!(target_arch = "aarch64")
}

/// Returns the calling convention name for the current platform.
#[inline]
pub const fn calling_convention_name() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "AAPCS64"
    } else if cfg!(target_os = "windows") {
        "Microsoft x64"
    } else {
        "System V AMD64"
    }
}

/// Returns the target architecture name.
#[inline]
pub const fn arch_name() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calling_convention_name() {
        let name = calling_convention_name();
        #[cfg(target_arch = "aarch64")]
        assert_eq!(name, "AAPCS64");
        #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
        assert_eq!(name, "Microsoft x64");
        #[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
        assert_eq!(name, "System V AMD64");
    }

    #[test]
    fn test_arch_name() {
        let arch = arch_name();
        #[cfg(target_arch = "aarch64")]
        assert_eq!(arch, "aarch64");
        #[cfg(target_arch = "x86_64")]
        assert_eq!(arch, "x86_64");
    }
}
