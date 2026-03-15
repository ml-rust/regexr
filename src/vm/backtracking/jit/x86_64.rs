//! x86-64 code generation for backtracking JIT.
//!
//! This module implements a PCRE-style backtracking JIT that generates native x86-64
//! code for patterns containing backreferences. Unlike the Thompson NFA-based PikeVM
//! or Tagged NFA JIT, this compiler generates single-threaded code with explicit
//! backtracking, which is much faster for backreference patterns.
//!
//! # Architecture
//!
//! The backtracking JIT directly compiles HIR (High-level IR) expressions to x86-64
//! assembly using dynasm. Key differences from the Tagged NFA JIT:
//!
//! - **Single thread**: No thread management overhead
//! - **Explicit backtrack stack**: Uses native stack for choice points
//! - **Direct codegen from HIR**: Simpler than NFA-based approaches
//!
//! # Register Allocation
//!
//! | Register | Purpose |
//! |----------|---------|
//! | rdi | Input base pointer (preserved) |
//! | rsi | Input length |
//! | rcx | Current position in input |
//! | rax | Scratch / return value |
//! | rbx | Backtrack stack pointer (callee-saved) |
//! | r12 | Captures base pointer (callee-saved) |
//! | r13 | Start position for current match attempt |
//! | r14 | Scratch for comparisons |
//! | r15 | Scratch for loop counters |

use crate::error::{Error, ErrorKind, Result};
use crate::hir::{Hir, HirAnchor, HirClass, HirExpr};

use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

use super::jit::BacktrackingJit;

/// The backtracking JIT compiler.
pub(super) struct BacktrackingCompiler {
    /// The assembler.
    asm: dynasmrt::x64::Assembler,
    /// The HIR to compile.
    hir: Hir,
    /// Label for the backtrack handler.
    backtrack_label: dynasmrt::DynamicLabel,
    /// Label for successful match.
    match_success_label: dynasmrt::DynamicLabel,
    /// Label for no match found.
    no_match_label: dynasmrt::DynamicLabel,
    /// Label for trying next start position.
    next_start_label: dynasmrt::DynamicLabel,
    /// Number of capture groups.
    capture_count: u32,
    /// Current capture index being filled (used to update capture end on backtrack).
    /// None if not inside a capture.
    current_capture: Option<u32>,
}

impl BacktrackingCompiler {
    pub(super) fn new(hir: &Hir) -> Result<Self> {
        let mut asm = dynasmrt::x64::Assembler::new().map_err(|e| {
            Error::new(
                ErrorKind::Jit(format!("Failed to create assembler: {:?}", e)),
                "",
            )
        })?;

        let backtrack_label = asm.new_dynamic_label();
        let match_success_label = asm.new_dynamic_label();
        let no_match_label = asm.new_dynamic_label();
        let next_start_label = asm.new_dynamic_label();

        Ok(Self {
            asm,
            hir: hir.clone(),
            backtrack_label,
            match_success_label,
            no_match_label,
            next_start_label,
            capture_count: hir.props.capture_count,
            current_capture: None,
        })
    }

    pub(super) fn compile(mut self) -> Result<BacktrackingJit> {
        let entry_offset = self.asm.offset();

        // Emit the prologue
        self.emit_prologue();

        // Emit the main matching loop (tries each start position)
        self.emit_main_loop()?;

        // Emit the pattern matching code
        self.emit_pattern(&self.hir.expr.clone())?;

        // After pattern matches, jump to success
        dynasm!(self.asm
            ; .arch x64
            ; jmp =>self.match_success_label
        );

        // Emit backtrack handler
        self.emit_backtrack_handler();

        // Emit success handler
        self.emit_success_handler();

        // Emit no-match handler
        self.emit_no_match_handler();

        // Emit epilogue (shared by success and no-match)
        self.emit_epilogue();

        // Finalize the code
        let code = self
            .asm
            .finalize()
            .map_err(|e| Error::new(ErrorKind::Jit(format!("Failed to finalize: {:?}", e)), ""))?;

        #[cfg(target_os = "windows")]
        let match_fn: unsafe extern "win64" fn(*const u8, usize, *mut i64) -> i64 =
            unsafe { std::mem::transmute(code.ptr(entry_offset)) };

        #[cfg(not(target_os = "windows"))]
        let match_fn: unsafe extern "sysv64" fn(*const u8, usize, *mut i64) -> i64 =
            unsafe { std::mem::transmute(code.ptr(entry_offset)) };

        Ok(BacktrackingJit {
            code,
            match_fn,
            capture_count: self.capture_count,
        })
    }

    /// Emits the function prologue.
    fn emit_prologue(&mut self) {
        // Function signature: fn(input_ptr: *const u8, input_len: usize, captures: *mut i64) -> i64
        // Unix: rdi = input_ptr, rsi = input_len, rdx = captures_ptr
        // Windows: rcx = input_ptr, rdx = input_len, r8 = captures_ptr

        #[cfg(target_os = "windows")]
        dynasm!(self.asm
            ; .arch x64
            ; push rdi              // Callee-saved on Windows
            ; push rsi              // Callee-saved on Windows
            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15
            ; push rbp
            ; mov rbp, rsp

            // Allocate space for backtrack stack
            ; sub rsp, 0x1008  // 4KB + 8 bytes for alignment

            // Move Windows args to internal registers
            ; mov rdi, rcx           // rdi = input_ptr
            ; mov rsi, rdx           // rsi = input_len
            ; mov r12, r8            // r12 = captures_ptr
            ; xor r13d, r13d         // r13 = start_pos = 0
            ; mov rbx, rsp           // rbx = backtrack stack pointer

            ; mov rax, -1i32 as i64 as i32
        );

        #[cfg(not(target_os = "windows"))]
        dynasm!(self.asm
            ; .arch x64
            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15
            ; push rbp
            ; mov rbp, rsp

            // Allocate space for backtrack stack (on native stack)
            // We use a simple approach: each backtrack point is 32 bytes
            // Stack grows UPWARD: rbx starts at bottom, add 32 to push, sub 32 to pop
            // Also ensure 16-byte stack alignment
            ; sub rsp, 0x1008  // 4KB + 8 bytes for alignment

            // Set up registers
            // rdi = input_ptr (keep as-is, it's our input base)
            // rsi = input_len (keep as length for offset comparisons)
            ; mov r12, rdx           // r12 = captures_ptr
            ; xor r13d, r13d         // r13 = start_pos = 0
            ; mov rbx, rsp           // rbx = backtrack stack pointer (grows UP from here)

            // Use rax to initialize captures to -1
            ; mov rax, -1i32 as i64 as i32
        );

        // Initialize all capture slots to -1
        let num_slots = (self.capture_count as usize + 1) * 2;
        for slot in 0..num_slots {
            let offset = (slot * 8) as i32;
            dynasm!(self.asm
                ; .arch x64
                ; mov QWORD [r12 + offset], rax
            );
        }
    }

    /// Emits the main loop that tries each start position.
    fn emit_main_loop(&mut self) -> Result<()> {
        dynasm!(self.asm
            ; .arch x64
            ; =>self.next_start_label
            // Reset captures for new attempt
            // Use rax = -1 for resetting
            ; mov rax, -1i32 as i64 as i32
        );

        // Reset capture slots to -1
        let num_slots = (self.capture_count as usize + 1) * 2;
        for slot in 0..num_slots {
            let offset = (slot * 8) as i32;
            dynasm!(self.asm
                ; .arch x64
                ; mov QWORD [r12 + offset], rax
            );
        }

        dynasm!(self.asm
            ; .arch x64
            // rcx = current position = start_pos
            ; mov rcx, r13

            // Set group 0 start = current position
            ; mov QWORD [r12], rcx

            // Reset backtrack stack to bottom (empty)
            ; lea rbx, [rbp - 0x1008]
        );

        Ok(())
    }

    /// Emits code to match the pattern.
    fn emit_pattern(&mut self, expr: &HirExpr) -> Result<()> {
        match expr {
            HirExpr::Empty => Ok(()),

            HirExpr::Literal(bytes) => self.emit_literal(bytes),

            HirExpr::Class(class) => self.emit_class(class),

            HirExpr::UnicodeCpClass(_) => {
                // Unicode codepoint classes require UTF-8 decoding - not supported yet
                Err(Error::new(
                    ErrorKind::Jit(
                        "Unicode codepoint classes not supported in backtracking JIT".to_string(),
                    ),
                    "",
                ))
            }

            HirExpr::Concat(parts) => {
                for part in parts {
                    self.emit_pattern(part)?;
                }
                Ok(())
            }

            HirExpr::Alt(alternatives) => self.emit_alternation(alternatives),

            HirExpr::Repeat(repeat) => {
                self.emit_repetition(&repeat.expr, repeat.min, repeat.max, repeat.greedy)
            }

            HirExpr::Capture(capture) => self.emit_capture(capture.index, &capture.expr),

            HirExpr::Backref(group) => self.emit_backref(*group),

            HirExpr::Anchor(anchor) => self.emit_anchor(*anchor),

            HirExpr::Lookaround(_) => {
                // Lookarounds not supported in backtracking JIT
                Err(Error::new(
                    ErrorKind::Jit("Lookarounds not supported in backtracking JIT".to_string()),
                    "",
                ))
            }
        }
    }

    /// Emits code to match a literal string.
    fn emit_literal(&mut self, bytes: &[u8]) -> Result<()> {
        for &byte in bytes {
            dynasm!(self.asm
                ; .arch x64
                // Check if we're at end of input
                ; cmp rcx, rsi
                ; jge =>self.backtrack_label

                // Load byte at current position
                ; movzx eax, BYTE [rdi + rcx]

                // Compare with expected byte
                ; cmp al, byte as i8
                ; jne =>self.backtrack_label

                // Advance position
                ; inc rcx
            );
        }
        Ok(())
    }

    /// Emits code to match a character class.
    fn emit_class(&mut self, class: &HirClass) -> Result<()> {
        let match_ok = self.asm.new_dynamic_label();
        let no_match = self.asm.new_dynamic_label();

        dynasm!(self.asm
            ; .arch x64
            // Check end of input
            ; cmp rcx, rsi
            ; jge =>self.backtrack_label

            // Load current byte
            ; movzx eax, BYTE [rdi + rcx]
        );

        // Generate range checks
        for &(start, end) in &class.ranges {
            if start == end {
                // Single byte
                dynasm!(self.asm
                    ; .arch x64
                    ; cmp al, start as i8
                    ; je =>match_ok
                );
            } else {
                // Range
                dynasm!(self.asm
                    ; .arch x64
                    ; cmp al, start as i8
                    ; jb >next_range
                    ; cmp al, end as i8
                    ; jbe =>match_ok
                    ; next_range:
                );
            }
        }

        // No range matched
        dynasm!(self.asm
            ; .arch x64
            ; jmp =>no_match
        );

        dynasm!(self.asm
            ; .arch x64
            ; =>match_ok
        );

        // Handle negation
        if class.negated {
            // If negated and we matched, backtrack
            dynasm!(self.asm
                ; .arch x64
                ; jmp =>self.backtrack_label
            );
            dynasm!(self.asm
                ; .arch x64
                ; =>no_match
                ; inc rcx
            );
        } else {
            // If not negated and we matched, advance
            dynasm!(self.asm
                ; .arch x64
                ; inc rcx
                ; jmp >done
            );
            dynasm!(self.asm
                ; .arch x64
                ; =>no_match
                ; jmp =>self.backtrack_label
                ; done:
            );
        }

        Ok(())
    }

    /// Emits code for alternation with backtracking.
    fn emit_alternation(&mut self, alternatives: &[HirExpr]) -> Result<()> {
        if alternatives.is_empty() {
            return Ok(());
        }

        let after_alt = self.asm.new_dynamic_label();

        for (i, alt) in alternatives.iter().enumerate() {
            let is_last = i == alternatives.len() - 1;

            if !is_last {
                // Push choice point before trying this alternative
                let try_next = self.asm.new_dynamic_label();

                // Save state for backtracking (32-byte entry, stack grows UP)
                dynasm!(self.asm
                    ; .arch x64
                    // Push backtrack point (add to grow up)
                    ; mov QWORD [rbx], rcx           // Save position
                    ; lea rax, [=>try_next]
                    ; mov QWORD [rbx + 8], rax       // Save resume address
                    ; mov QWORD [rbx + 16], r13      // Save start_pos
                    ; mov QWORD [rbx + 24], 0        // Unused slot (for consistency)
                    ; add rbx, 32
                );

                // Try this alternative
                self.emit_pattern(alt)?;

                // Success - jump past other alternatives
                dynasm!(self.asm
                    ; .arch x64
                    // Pop the choice point since we succeeded (sub to pop)
                    ; sub rbx, 32
                    ; jmp =>after_alt
                );

                // Label for trying next alternative (reached via backtrack)
                dynasm!(self.asm
                    ; .arch x64
                    ; =>try_next
                );
            } else {
                // Last alternative - no choice point needed
                self.emit_pattern(alt)?;
            }
        }

        dynasm!(self.asm
            ; .arch x64
            ; =>after_alt
        );

        Ok(())
    }

    /// Emits optimized code for exact repetitions {n,n}.
    ///
    /// This is PCRE2-JIT's OP_EXACT optimization: a tight loop with no backtracking.
    /// For patterns like `\d{4}`, we:
    /// 1. Check upfront that enough input remains (fast fail)
    /// 2. Run a simple countdown loop matching exactly N times
    /// 3. No choice points = no backtracking overhead
    fn emit_exact_repetition(&mut self, expr: &HirExpr, count: u32) -> Result<()> {
        // Special case: single byte character class (like \d, \w, \s)
        // We can generate even tighter code by inlining the check
        if self.try_emit_exact_class_repetition(expr, count)?.is_some() {
            // Already handled
            return Ok(());
        }

        // General case: emit a counted loop
        let loop_start = self.asm.new_dynamic_label();

        // Use r15 as countdown counter (avoids cmp instruction in loop)
        dynasm!(self.asm
            ; .arch x64
            ; mov r15d, count as i32    // r15 = count
        );

        dynasm!(self.asm
            ; .arch x64
            ; =>loop_start
        );

        // Match one instance of the subexpression
        self.emit_pattern(expr)?;

        dynasm!(self.asm
            ; .arch x64
            ; dec r15d
            ; jnz =>loop_start
        );

        Ok(())
    }

    /// Tries to emit optimized code for exact repetitions of simple character classes.
    /// Returns Ok(Some(bytes_consumed)) if handled, Ok(None) if not applicable.
    fn try_emit_exact_class_repetition(
        &mut self,
        expr: &HirExpr,
        count: u32,
    ) -> Result<Option<usize>> {
        // Check if this is a simple character class
        let class = match expr {
            HirExpr::Class(c) => c,
            _ => return Ok(None),
        };

        // Only optimize non-negated classes for now (simpler bounds checking)
        if class.negated {
            return Ok(None);
        }

        // Check if this is a contiguous byte range (like \d = 0x30-0x39)
        // This allows for very fast bounds checking
        let is_digit = class.ranges == [(b'0', b'9')];
        let is_word_simple = class.ranges.len() <= 3; // alphanumeric + underscore

        if !is_digit && !is_word_simple {
            return Ok(None);
        }

        // OPTIMIZATION 1: Bounds check - verify we have enough input upfront
        // This matches PCRE2-JIT's approach: fail fast before entering the loop
        dynasm!(self.asm
            ; .arch x64
            // Calculate remaining: remaining = input_len - current_pos
            ; mov rax, rsi              // rax = input_len
            ; sub rax, rcx              // rax = remaining = len - pos
            ; cmp rax, count as i32     // remaining >= count?
            ; jl =>self.backtrack_label // Not enough input, fail immediately
        );

        if is_digit {
            // OPTIMIZATION 2: Tight loop for \d{n}
            // Uses single range check: byte - '0' < 10
            let loop_start = self.asm.new_dynamic_label();

            dynasm!(self.asm
                ; .arch x64
                ; mov r15d, count as i32    // r15 = countdown
                ; =>loop_start

                // Load byte (we know we have enough input from bounds check)
                ; movzx eax, BYTE [rdi + rcx]

                // Fast digit check: (byte - '0') < 10
                ; sub eax, 0x30             // al = byte - '0'
                ; cmp eax, 10
                ; jae =>self.backtrack_label // Not a digit

                // Advance
                ; inc rcx
                ; dec r15d
                ; jnz =>loop_start
            );
        } else {
            // General class with multiple ranges - use emit_class for each iteration
            let loop_start = self.asm.new_dynamic_label();

            dynasm!(self.asm
                ; .arch x64
                ; mov r15d, count as i32    // r15 = countdown
                ; =>loop_start
            );

            self.emit_class(class)?;

            dynasm!(self.asm
                ; .arch x64
                ; dec r15d
                ; jnz =>loop_start
            );
        }

        Ok(Some(count as usize))
    }

    /// Emits code for repetition (*, +, ?, {n,m}).
    ///
    /// OPTIMIZED: Exact repetitions {n} use a tight loop without backtracking.
    /// This matches PCRE2-JIT's OP_EXACT optimization.
    fn emit_repetition(
        &mut self,
        expr: &HirExpr,
        min: u32,
        max: Option<u32>,
        greedy: bool,
    ) -> Result<()> {
        let loop_done = self.asm.new_dynamic_label();

        // OPTIMIZATION: Exact repetitions {n,n} don't need backtracking
        // This is PCRE2-JIT's OP_EXACT optimization - a tight loop with no choice points
        if let Some(max_val) = max {
            if min == max_val && min > 0 {
                return self.emit_exact_repetition(expr, min);
            }
        }

        // Use r15 as iteration counter
        dynasm!(self.asm
            ; .arch x64
            ; xor r15d, r15d    // r15 = count = 0
        );

        if greedy {
            // Greedy: match as many as possible, with proper backtracking.
            // For patterns like (a+)\1, we need to save choice points so we can
            // backtrack and try shorter matches.
            let loop_start = self.asm.new_dynamic_label();
            let try_backtrack = self.asm.new_dynamic_label();

            dynasm!(self.asm
                ; .arch x64
                ; =>loop_start
            );

            // Check max limit
            if let Some(max_val) = max {
                dynasm!(self.asm
                    ; .arch x64
                    ; cmp r15d, max_val as i32
                    ; jge =>loop_done
                );
            }

            // Save current position as a choice point for backtracking
            // When we fail later, we can come back here and try with fewer matches
            // Choice point format: [position, return_label, r13, r15] (32 bytes, stack grows UP)
            dynasm!(self.asm
                ; .arch x64
                ; mov QWORD [rbx], rcx              // Save position
                ; lea rax, [=>try_backtrack]
                ; mov QWORD [rbx + 8], rax          // Return address for backtrack
                ; mov QWORD [rbx + 16], r13         // Save start_pos
                ; mov QWORD [rbx + 24], r15         // Save iteration count
                ; add rbx, 32                       // Push (grow up)
            );

            // Try to match one more.
            // IMPORTANT: Don't override backtrack_label here! The inner pattern
            // (which might contain alternation) needs the global backtrack handler
            // to properly pop and try alternatives.
            //
            // We use a "success continuation" approach instead:
            // - If the pattern matches, continue to increment counter
            // - If the pattern fails, the backtrack handler will pop entries
            //   until it finds our try_backtrack entry

            // Create a label for "iteration matched successfully"
            let iteration_matched = self.asm.new_dynamic_label();

            // Create our own local backtrack handler for this iteration
            let iteration_backtrack = self.asm.new_dynamic_label();
            let old_backtrack = self.backtrack_label;
            self.backtrack_label = iteration_backtrack;

            self.emit_pattern(expr)?;

            self.backtrack_label = old_backtrack;

            // Pattern matched - jump to success path
            dynasm!(self.asm
                ; .arch x64
                ; jmp =>iteration_matched

                ; =>iteration_backtrack
                // Inner pattern failed. Check if there are backtrack entries
                // between our position and the bottom.
                ; lea rax, [rbp - 0x1008]   // Stack bottom
                ; cmp rbx, rax
                ; jle >empty_stack

                // Pop and check if it's our try_backtrack entry
                ; sub rbx, 32
                ; mov rax, QWORD [rbx + 8]  // Get resume address
                ; lea r14, [=>try_backtrack]
                ; cmp rax, r14
                ; jne >not_our_entry

                // It's our entry - restore state and pop to exit loop
                ; mov rcx, QWORD [rbx]
                ; mov r13, QWORD [rbx + 16]
                ; mov r15, QWORD [rbx + 24]
                ; jmp =>loop_done   // Exit loop, matched as many as we could

                ; not_our_entry:
                // It's someone else's entry (alternation, etc.) - jump to their resume
                ; mov rcx, QWORD [rbx]
                ; mov r13, QWORD [rbx + 16]
                ; mov r15, QWORD [rbx + 24]
                ; jmp rax

                ; empty_stack:
                // No entries - exit loop
                ; jmp =>loop_done
            );

            // Pattern succeeded - increment counter, continue
            dynasm!(self.asm
                ; .arch x64
                ; =>iteration_matched
                ; inc r15d
                ; jmp =>loop_start

                ; =>try_backtrack
                // Backtracked here from a later failure.
                // The backtrack handler has already popped our choice point and restored:
                // rcx = position, r13 = start_pos, r15 = count
                // But we need to re-read from the backtrack handler's frame.
                // Actually, the handler pops and jumps here, so the values are in rcx/r13/r15 already.
            );

            // If we're inside a capture, update the capture end to current position
            if let Some(cap_idx) = self.current_capture {
                let end_offset = (cap_idx as i32) * 16 + 8;
                dynasm!(self.asm
                    ; .arch x64
                    ; mov QWORD [r12 + end_offset], rcx
                );
            }

            // Check if we have enough matches to satisfy minimum
            dynasm!(self.asm
                ; .arch x64
                ; cmp r15d, min as i32
                ; jl =>self.backtrack_label    // Not enough matches, backtrack further
                // We have enough matches, try to continue with the rest of the pattern
                ; jmp =>loop_done
            );
        } else {
            // Non-greedy: match minimum first, then try to continue without matching more
            // First, match the minimum required
            for _ in 0..min {
                self.emit_pattern(expr)?;
                dynasm!(self.asm
                    ; .arch x64
                    ; inc r15d
                );
            }

            if max.is_none_or(|m| m > min) {
                // Can match more - set up choice points
                let loop_start = self.asm.new_dynamic_label();
                let try_more = self.asm.new_dynamic_label();

                dynasm!(self.asm
                    ; .arch x64
                    ; =>loop_start
                );

                // Check max limit
                if let Some(max_val) = max {
                    dynasm!(self.asm
                        ; .arch x64
                        ; cmp r15d, max_val as i32
                        ; jge =>loop_done
                    );
                }

                // Push choice point to try matching more later (stack grows UP)
                dynasm!(self.asm
                    ; .arch x64
                    ; mov QWORD [rbx], rcx
                    ; lea rax, [=>try_more]
                    ; mov QWORD [rbx + 8], rax
                    ; mov QWORD [rbx + 16], r13
                    ; mov QWORD [rbx + 24], r15
                    ; add rbx, 32                   // Push (grow up)

                    // Non-greedy: first try to continue without matching more
                    ; jmp =>loop_done

                    ; =>try_more
                    // Backtracked here - the handler has already popped and restored
                    // rcx, r13 from the entry. r15 was at [rbx+24] before pop.
                    // The handler reads r15 from the popped entry.
                );

                // For simplicity, just use saved position on native stack
                // Match one more
                self.emit_pattern(expr)?;
                dynasm!(self.asm
                    ; .arch x64
                    ; inc r15d
                    ; jmp =>loop_start
                );
            }
        }

        dynasm!(self.asm
            ; .arch x64
            ; =>loop_done
            // Check minimum count
            ; cmp r15d, min as i32
            ; jl =>self.backtrack_label
        );

        Ok(())
    }

    /// Emits code for a capture group.
    fn emit_capture(&mut self, index: u32, expr: &HirExpr) -> Result<()> {
        let start_offset = (index as i32) * 16; // Each group is 2 slots * 8 bytes
        let end_offset = start_offset + 8;

        // Record start position
        dynasm!(self.asm
            ; .arch x64
            ; mov QWORD [r12 + start_offset], rcx
        );

        // Track that we're inside this capture (for greedy backtracking to update capture end)
        let old_capture = self.current_capture;
        self.current_capture = Some(index);

        // Match inner expression
        self.emit_pattern(expr)?;

        // Restore previous capture context
        self.current_capture = old_capture;

        // Record end position
        dynasm!(self.asm
            ; .arch x64
            ; mov QWORD [r12 + end_offset], rcx
        );

        Ok(())
    }

    /// Emits code for a backreference.
    fn emit_backref(&mut self, group: u32) -> Result<()> {
        let start_offset = (group as i32) * 16;
        let end_offset = start_offset + 8;

        let backref_ok = self.asm.new_dynamic_label();

        dynasm!(self.asm
            ; .arch x64
            // Load captured text bounds
            ; mov r8, QWORD [r12 + start_offset]   // r8 = capture_start
            ; mov r9, QWORD [r12 + end_offset]     // r9 = capture_end

            // Check if capture is valid (both >= 0)
            ; test r8, r8
            ; js =>self.backtrack_label            // Not captured yet

            // Calculate capture length: r10 = capture_end - capture_start
            ; mov r10, r9
            ; sub r10, r8                          // r10 = capture_len

            // Empty capture always matches
            ; test r10, r10
            ; jz =>backref_ok

            // Check if enough input remains
            // rsi = input_len
            // rcx = current position offset
            // remaining = rsi - rcx = len - pos
            ; mov r11, rsi
            ; sub r11, rcx                         // r11 = remaining = len - pos
            ; cmp r10, r11
            ; jg =>self.backtrack_label            // Not enough input

            // Set up pointers for comparison:
            // r8 = input + capture_start (source pointer)
            // r9 = input + current_pos (dest pointer)
            ; add r8, rdi                          // r8 = rdi + capture_start
            ; lea r9, [rdi + rcx]                  // r9 = rdi + current_pos

            // Compare bytes using a simple loop
            ; xor r14d, r14d                       // r14 = comparison index
            ; cmp_loop:
            ; cmp r14, r10
            ; jge =>backref_ok                     // All bytes matched

            ; movzx eax, BYTE [r8 + r14]           // Byte from captured text
            ; movzx r11d, BYTE [r9 + r14]          // Byte from current position
            ; cmp eax, r11d
            ; jne =>self.backtrack_label           // Mismatch

            ; inc r14
            ; jmp <cmp_loop

            ; =>backref_ok
            // Advance position by capture length
            ; add rcx, r10
        );

        Ok(())
    }

    /// Emits code for anchors.
    fn emit_anchor(&mut self, anchor: HirAnchor) -> Result<()> {
        match anchor {
            HirAnchor::Start => {
                // Start of text: position must be 0
                dynasm!(self.asm
                    ; .arch x64
                    ; test rcx, rcx
                    ; jnz =>self.backtrack_label
                );
            }
            HirAnchor::End => {
                // End of text: position must equal length
                dynasm!(self.asm
                    ; .arch x64
                    ; cmp rcx, rsi
                    ; jne =>self.backtrack_label
                );
            }
            HirAnchor::StartLine => {
                // Start of line: position is 0 or preceded by newline
                let ok = self.asm.new_dynamic_label();
                dynasm!(self.asm
                    ; .arch x64
                    ; test rcx, rcx
                    ; jz =>ok
                    ; mov al, BYTE [rdi + rcx - 1]
                    ; cmp al, 0x0a  // newline
                    ; jne =>self.backtrack_label
                    ; =>ok
                );
            }
            HirAnchor::EndLine => {
                // End of line: at end or followed by newline
                let ok = self.asm.new_dynamic_label();
                dynasm!(self.asm
                    ; .arch x64
                    ; cmp rcx, rsi
                    ; je =>ok
                    ; mov al, BYTE [rdi + rcx]
                    ; cmp al, 0x0a  // newline
                    ; jne =>self.backtrack_label
                    ; =>ok
                );
            }
            HirAnchor::WordBoundary | HirAnchor::NotWordBoundary => {
                // Word boundaries require more complex logic
                // For now, just fail compilation
                return Err(Error::new(
                    ErrorKind::Jit(
                        "Word boundaries not yet supported in backtracking JIT".to_string(),
                    ),
                    "",
                ));
            }
        }
        Ok(())
    }

    /// Emits the backtrack handler.
    ///
    /// All backtrack entries are 32 bytes (stack grows UP):
    /// - `entry + 0`:  position (rcx)
    /// - `entry + 8`:  resume address
    /// - `entry + 16`: start_pos (r13)
    /// - `entry + 24`: extra data (count for repetition, unused for others)
    fn emit_backtrack_handler(&mut self) {
        dynasm!(self.asm
            ; .arch x64
            ; =>self.backtrack_label

            // Check if backtrack stack is empty (rbx == bottom means empty)
            ; lea rax, [rbp - 0x1008]        // rax = stack bottom
            ; cmp rbx, rax
            ; jle >try_next_pos              // Stack is empty if rbx <= bottom

            // Pop backtrack entry (32 bytes) - stack grows UP so subtract to pop
            ; sub rbx, 32                    // Pop entry
            ; mov rcx, QWORD [rbx]           // Restore position
            ; mov rax, QWORD [rbx + 8]       // Get resume address
            ; mov r13, QWORD [rbx + 16]      // Restore start_pos
            ; mov r15, QWORD [rbx + 24]      // Restore extra data (iteration count)

            // Jump to resume address
            ; jmp rax

            ; try_next_pos:
            // No more backtrack points - try next start position
            ; inc r13
            // rsi = input_len directly now
            ; cmp r13, rsi
            ; jg =>self.no_match_label
            ; jmp =>self.next_start_label
        );
    }

    /// Emits the success handler.
    fn emit_success_handler(&mut self) {
        let epilogue = self.asm.new_dynamic_label();
        dynasm!(self.asm
            ; .arch x64
            ; =>self.match_success_label
            // Set group 0 end = current position
            ; mov QWORD [r12 + 8], rcx

            // Return the end position (positive = success)
            ; mov rax, rcx
            ; jmp =>epilogue

            // No-match handler
            ; =>self.no_match_label
            ; mov rax, -1i32

            // Shared epilogue
            ; =>epilogue
            // Clean up stack
            ; mov rsp, rbp
            ; pop rbp
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
        );

        // Platform-specific epilogue
        #[cfg(target_os = "windows")]
        dynasm!(self.asm
            ; .arch x64
            ; pop rsi
            ; pop rdi
            ; ret
        );

        #[cfg(not(target_os = "windows"))]
        dynasm!(self.asm
            ; .arch x64
            ; ret
        );
    }

    /// Emits the no-match handler - merged into success_handler for control flow.
    fn emit_no_match_handler(&mut self) {
        // No-op - merged into emit_success_handler
    }

    /// Emits the function epilogue - merged into success_handler for control flow.
    fn emit_epilogue(&mut self) {
        // No-op - merged into emit_success_handler
    }
}
