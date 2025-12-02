//! AArch64 (ARM64) code generation using dynasm.
//!
//! This module emits native ARM64 assembly code for DFA state machines.
//! All code is W^X compliant and optimized for performance.
//!
//! # Platform Support
//!
//! Uses AAPCS64 calling convention on all ARM64 platforms:
//! - **Arguments**: X0-X7
//! - **Return value**: X0
//! - **Callee-saved**: X19-X28, SP
//! - **Link register**: X30
//! - **Frame pointer**: X29

use crate::dfa::DfaStateId;
use crate::error::{Error, ErrorKind, Result};
use crate::jit::codegen_aarch64::{MaterializedDfa, MaterializedState};
use dynasm::dynasm;
use dynasmrt::{
    aarch64::Assembler, AssemblyOffset, DynamicLabel, DynasmApi, DynasmLabelApi, ExecutableBuffer,
};

/// Compiles a materialized DFA to ARM64 machine code.
///
/// # Calling Convention (AAPCS64)
///
/// The function accepts two arguments:
/// - X0: input pointer (const uint8_t*)
/// - X1: length (size_t)
///
/// # Internal Register Usage
/// - `x19` = current position in input (mutable, incremented during execution)
/// - `x20` = input base pointer (immutable)
/// - `x21` = input end pointer (base + len, immutable)
/// - `x22` = last match position, initialized to -1 (for longest-match semantics)
/// - `x23` = search start position (for unanchored search)
/// - `x24` = prev_char_class (0 = NonWord, 1 = Word) for word boundary patterns
/// - `x9-x15` = scratch registers
///
/// # Function Signature
/// ```c
/// int64_t match_fn(const uint8_t* input, size_t len);
/// ```
///
/// Returns:
/// - >= 0: Match found, packed as (start << 32 | end)
/// - -1: No match
pub fn compile_states(
    dfa: &MaterializedDfa,
) -> Result<(ExecutableBuffer, AssemblyOffset, Option<AssemblyOffset>)> {
    let mut asm = Assembler::new().map_err(|e| {
        Error::new(
            ErrorKind::Jit(format!("Failed to create assembler: {:?}", e)),
            "",
        )
    })?;

    // Create state label lookup using Vec for O(1) lookup
    let max_state_id = dfa.states.iter().map(|s| s.id).max().unwrap_or(0) as usize;
    let mut state_labels: Vec<Option<DynamicLabel>> = vec![None; max_state_id + 1];
    for state in &dfa.states {
        state_labels[state.id as usize] = Some(asm.new_dynamic_label());
    }
    let dead_label = asm.new_dynamic_label();
    let no_match_label = asm.new_dynamic_label();

    // For unanchored patterns, create a restart label for the internal search loop
    let restart_label = if !dfa.has_start_anchor {
        Some(asm.new_dynamic_label())
    } else {
        None
    };

    // For word boundary patterns, create a dispatch label
    let dispatch_label = if dfa.has_word_boundary && dfa.start_word.is_some() {
        Some(asm.new_dynamic_label())
    } else {
        None
    };

    // Emit prologue for NonWord prev_class (primary entry point)
    let entry_point = asm.offset();
    emit_prologue(
        &mut asm,
        dfa.start,
        &state_labels,
        restart_label,
        dispatch_label,
        dfa.has_word_boundary,
        true,
    )?;

    // Emit prologue for Word prev_class (secondary entry point, if needed)
    let entry_point_word = if let Some(start_word) = dfa.start_word {
        let offset = asm.offset();
        emit_prologue(
            &mut asm,
            start_word,
            &state_labels,
            restart_label,
            dispatch_label,
            dfa.has_word_boundary,
            false,
        )?;
        Some(offset)
    } else {
        None
    };

    // Emit dispatch block for word boundary patterns
    if let (Some(dispatch), Some(start_word)) = (dispatch_label, dfa.start_word) {
        emit_dispatch(&mut asm, dispatch, dfa.start, start_word, &state_labels)?;
    }

    // Emit code for each DFA state
    for state in &dfa.states {
        emit_state(&mut asm, state, &state_labels, dead_label, no_match_label)?;
    }

    // Emit dead state
    emit_dead_state(
        &mut asm,
        dead_label,
        no_match_label,
        restart_label,
        dispatch_label,
        dfa.has_word_boundary,
    )?;

    // Emit no-match epilogue
    emit_no_match(&mut asm, no_match_label, dfa.has_word_boundary)?;

    // Finalize and get executable buffer
    let code = asm.finalize().map_err(|_| {
        Error::new(
            ErrorKind::Jit("Failed to finalize assembly".to_string()),
            "",
        )
    })?;

    Ok((code, entry_point, entry_point_word))
}

/// Emits the function prologue.
///
/// Register allocation:
/// - x19 = current position (starts at 0)
/// - x20 = input base pointer
/// - x21 = input end pointer
/// - x22 = last match position (-1)
/// - x23 = search start position (0)
/// - x24 = prev_char_class (for word boundaries)
fn emit_prologue(
    asm: &mut Assembler,
    start_state: DfaStateId,
    state_labels: &[Option<DynamicLabel>],
    restart_label: Option<DynamicLabel>,
    dispatch_label: Option<DynamicLabel>,
    has_word_boundary: bool,
    emit_restart_label: bool,
) -> Result<()> {
    let start_label = state_labels
        .get(start_state as usize)
        .and_then(|opt| opt.as_ref())
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Jit("Start state label not found".to_string()),
                "",
            )
        })?;

    // Save callee-saved registers and set up frame
    dynasm!(asm
        ; .arch aarch64
        // Save frame pointer and link register
        ; stp x29, x30, [sp, #-16]!
        ; mov x29, sp
        // Save callee-saved registers we'll use
        ; stp x19, x20, [sp, #-16]!
        ; stp x21, x22, [sp, #-16]!
        ; stp x23, x24, [sp, #-16]!
    );

    // AAPCS64: Arguments in X0, X1
    // X0 = input ptr, X1 = len
    dynasm!(asm
        ; mov x20, x0              // x20 = input base
        ; add x21, x0, x1          // x21 = input end (base + len)
        ; mov x19, #0              // x19 = position = 0
        ; movn x22, 0              // x22 = last match = -1
        ; mov x23, #0              // x23 = search start = 0
    );

    // For word boundary patterns, initialize x24 = prev_char_class
    if has_word_boundary {
        dynasm!(asm
            ; mov x24, #0             // x24 = 0 (NonWord at start)
        );
    }

    // Emit restart label if this is the primary entry point
    if let Some(restart) = restart_label {
        if emit_restart_label {
            dynasm!(asm
                ; =>restart
            );
        }
    }

    // For word boundary secondary entry, set x24 = 1 (Word)
    if dispatch_label.is_some() && !emit_restart_label {
        dynasm!(asm
            ; mov x24, #1             // x24 = 1 (Word)
        );
    }

    // Jump to start state
    dynasm!(asm
        ; b =>*start_label
    );

    Ok(())
}

/// Emits the dispatch block for word boundary patterns.
fn emit_dispatch(
    asm: &mut Assembler,
    dispatch_label: DynamicLabel,
    start: DfaStateId,
    start_word: DfaStateId,
    state_labels: &[Option<DynamicLabel>],
) -> Result<()> {
    let start_label = state_labels
        .get(start as usize)
        .and_then(|opt| opt.as_ref())
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Jit("Start state label not found".to_string()),
                "",
            )
        })?;
    let start_word_label = state_labels
        .get(start_word as usize)
        .and_then(|opt| opt.as_ref())
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Jit("Start word state label not found".to_string()),
                "",
            )
        })?;

    dynasm!(asm
        ; .align 4
        ; =>dispatch_label
        // Check x24: 0 = NonWord, 1 = Word
        ; cbnz x24, =>*start_word_label
        ; b =>*start_label
    );

    Ok(())
}

/// Analyzes if a state has a self-loop pattern suitable for fast-forward optimization.
fn analyze_self_loop(
    state: &MaterializedState,
) -> Option<(Vec<(u8, u8)>, Vec<(u8, u8, DfaStateId)>)> {
    let mut self_loop_bytes = Vec::new();
    let mut other_transitions = Vec::new();

    for byte in 0..=255u8 {
        if let Some(target) = state.transitions[byte as usize] {
            if target == state.id {
                self_loop_bytes.push(byte);
            } else {
                other_transitions.push((byte, target));
            }
        }
    }

    if self_loop_bytes.len() < 3 {
        return None;
    }

    // Convert to contiguous ranges
    let mut self_loop_ranges = Vec::new();
    if !self_loop_bytes.is_empty() {
        let mut start = self_loop_bytes[0];
        let mut end = self_loop_bytes[0];

        for &byte in &self_loop_bytes[1..] {
            if byte == end + 1 {
                end = byte;
            } else {
                self_loop_ranges.push((start, end));
                start = byte;
                end = byte;
            }
        }
        self_loop_ranges.push((start, end));
    }

    // Convert other transitions to ranges
    let mut other_ranges = Vec::new();
    if !other_transitions.is_empty() {
        let mut sorted = other_transitions.clone();
        sorted.sort_by_key(|(b, _)| *b);

        let mut start = sorted[0].0;
        let mut end = sorted[0].0;
        let mut target = sorted[0].1;

        for &(byte, t) in &sorted[1..] {
            if byte == end + 1 && t == target {
                end = byte;
            } else {
                other_ranges.push((start, end, target));
                start = byte;
                end = byte;
                target = t;
            }
        }
        other_ranges.push((start, end, target));
    }

    Some((self_loop_ranges, other_ranges))
}

/// Emits code for a single DFA state.
fn emit_state(
    asm: &mut Assembler,
    state: &MaterializedState,
    state_labels: &[Option<DynamicLabel>],
    dead_label: DynamicLabel,
    no_match_label: DynamicLabel,
) -> Result<()> {
    let state_label = state_labels
        .get(state.id as usize)
        .and_then(|opt| opt.as_ref())
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Jit(format!("Label for state {} not found", state.id)),
                "",
            )
        })?;

    let match_return = asm.new_dynamic_label();

    // Align for optimal instruction fetch
    dynasm!(asm
        ; .align 4
        ; =>*state_label
    );

    // Check if input is exhausted
    // x19 = position, x20 = base, x21 = end
    if state.is_match {
        // Save current position as last match
        dynasm!(asm
            ; mov x22, x19
        );
        // Check if exhausted
        dynasm!(asm
            ; add x9, x20, x19         // x9 = base + pos
            ; cmp x9, x21
            ; b.hs =>match_return
        );
    } else {
        dynasm!(asm
            ; add x9, x20, x19
            ; cmp x9, x21
            ; b.hs =>no_match_label
        );
    }

    // Check for self-loop optimization
    if let Some((self_loop_ranges, other_transitions)) = analyze_self_loop(state) {
        emit_fast_forward_loop(
            asm,
            state,
            &self_loop_ranges,
            &other_transitions,
            state_labels,
            dead_label,
            no_match_label,
        )?;
    } else {
        // Load next byte and increment position
        dynasm!(asm
            ; ldrb w9, [x20, x19]      // w9 = input[pos]
            ; add x19, x19, #1         // pos++
        );

        // Emit transitions
        if state.should_use_jump_table() {
            emit_dense_transitions(asm, state, state_labels, dead_label)?;
        } else {
            emit_sparse_transitions(asm, state, state_labels, dead_label)?;
        }
    }

    // Match return label
    if state.is_match {
        dynasm!(asm
            ; =>match_return
            ; b =>no_match_label
        );
    }

    Ok(())
}

/// Builds a 256-bit bitmap for the self-loop character class.
fn build_self_loop_bitmap(self_loop_ranges: &[(u8, u8)]) -> [u8; 32] {
    let mut bitmap = [0u8; 32];
    for &(start, end) in self_loop_ranges {
        for byte in start..=end {
            bitmap[byte as usize / 8] |= 1 << (byte % 8);
        }
    }
    bitmap
}

/// Emits a fast-forward loop for states with self-loops.
fn emit_fast_forward_loop(
    asm: &mut Assembler,
    state: &MaterializedState,
    self_loop_ranges: &[(u8, u8)],
    other_transitions: &[(u8, u8, DfaStateId)],
    state_labels: &[Option<DynamicLabel>],
    dead_label: DynamicLabel,
    no_match_label: DynamicLabel,
) -> Result<()> {
    let fast_forward_loop = asm.new_dynamic_label();
    let exhausted = asm.new_dynamic_label();
    let check_other = asm.new_dynamic_label();
    let consume_byte = asm.new_dynamic_label();

    // Use bitmap for 3+ ranges, otherwise use range checks
    let use_bitmap = self_loop_ranges.len() >= 3;

    if use_bitmap {
        let bitmap = build_self_loop_bitmap(self_loop_ranges);
        let bitmap_label = asm.new_dynamic_label();
        let start_label = asm.new_dynamic_label();

        // Embed bitmap data
        dynasm!(asm
            ; b =>start_label
            ; .align 8
            ; =>bitmap_label
            ; .bytes bitmap.as_slice()
            ; =>start_label
        );

        // Load bitmap address into x10
        dynasm!(asm
            ; adr x10, =>bitmap_label
        );

        // Fast-forward loop with bitmap
        dynasm!(asm
            ; =>fast_forward_loop
            // Check bounds
            ; add x9, x20, x19
            ; cmp x9, x21
            ; b.hs =>exhausted
            // Load byte
            ; ldrb w9, [x20, x19]
            // Bitmap lookup: bitmap[byte / 8] & (1 << (byte % 8))
            ; lsr w11, w9, #3          // w11 = byte / 8
            ; and w12, w9, #7          // w12 = byte % 8
            ; ldrb w13, [x10, x11]     // w13 = bitmap[byte / 8]
            ; mov w14, #1
            ; lsl w14, w14, w12        // w14 = 1 << (byte % 8)
            ; tst w13, w14
            ; b.eq =>check_other       // Not in class
            // Consume byte
            ; add x19, x19, #1
        );

        if state.is_match {
            dynasm!(asm
                ; mov x22, x19
            );
        }

        dynasm!(asm
            ; b =>fast_forward_loop
        );
    } else {
        // Range-based fast forward
        dynasm!(asm
            ; =>fast_forward_loop
            // Check bounds
            ; add x9, x20, x19
            ; cmp x9, x21
            ; b.hs =>exhausted
            // Load byte
            ; ldrb w9, [x20, x19]
        );

        // Check if byte is in self-loop ranges
        for (i, &(start, end)) in self_loop_ranges.iter().enumerate() {
            if start == end {
                dynasm!(asm
                    ; cmp w9, #start as u32
                    ; b.eq =>consume_byte
                );
            } else {
                let next_range = asm.new_dynamic_label();
                dynasm!(asm
                    ; cmp w9, #start as u32
                    ; b.lo =>next_range
                    ; cmp w9, #end as u32
                    ; b.ls =>consume_byte
                    ; =>next_range
                );
            }

            if i == self_loop_ranges.len() - 1 {
                dynasm!(asm
                    ; b =>check_other
                );
            }
        }

        // Consume byte
        dynasm!(asm
            ; =>consume_byte
            ; add x19, x19, #1
        );

        if state.is_match {
            dynasm!(asm
                ; mov x22, x19
            );
        }

        dynasm!(asm
            ; b =>fast_forward_loop
        );
    }

    // Exhausted - jump to no_match which will check x22
    dynasm!(asm
        ; =>exhausted
        ; b =>no_match_label
    );

    // Check other transitions
    dynasm!(asm
        ; =>check_other
    );

    if !other_transitions.is_empty() {
        // Reload byte and consume
        dynasm!(asm
            ; ldrb w9, [x20, x19]
            ; add x19, x19, #1
        );

        for &(start, end, target) in other_transitions {
            let target_label = state_labels
                .get(target as usize)
                .and_then(|opt| opt.as_ref())
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Jit(format!("Label for state {} not found", target)),
                        "",
                    )
                })?;

            if start == end {
                dynasm!(asm
                    ; cmp w9, #start as u32
                    ; b.eq =>*target_label
                );
            } else {
                let next = asm.new_dynamic_label();
                dynasm!(asm
                    ; cmp w9, #start as u32
                    ; b.lo =>next
                    ; cmp w9, #end as u32
                    ; b.ls =>*target_label
                    ; =>next
                );
            }
        }
    }

    // No transition matched
    dynasm!(asm
        ; b =>dead_label
    );

    Ok(())
}

/// Emits sparse transition code using linear compare chains.
fn emit_sparse_transitions(
    asm: &mut Assembler,
    state: &MaterializedState,
    state_labels: &[Option<DynamicLabel>],
    dead_label: DynamicLabel,
) -> Result<()> {
    let ranges = compute_byte_ranges(state);

    for (start, end, target) in ranges {
        let target_label = state_labels
            .get(target as usize)
            .and_then(|opt| opt.as_ref())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Jit(format!("Label for state {} not found", target)),
                    "",
                )
            })?;

        if start == end {
            dynasm!(asm
                ; cmp w9, #start as u32
                ; b.eq =>*target_label
            );
        } else {
            let next_check = asm.new_dynamic_label();
            dynasm!(asm
                ; cmp w9, #start as u32
                ; b.lo =>next_check
                ; cmp w9, #end as u32
                ; b.ls =>*target_label
                ; =>next_check
            );
        }
    }

    // No transition matched
    dynasm!(asm
        ; b =>dead_label
    );

    Ok(())
}

/// Emits dense transition code using range checks.
fn emit_dense_transitions(
    asm: &mut Assembler,
    state: &MaterializedState,
    state_labels: &[Option<DynamicLabel>],
    dead_label: DynamicLabel,
) -> Result<()> {
    let ranges = compute_byte_ranges(state);

    for (start, end, target) in ranges {
        let target_label = state_labels
            .get(target as usize)
            .and_then(|opt| opt.as_ref())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Jit(format!("Label for state {} not found", target)),
                    "",
                )
            })?;

        if start == end {
            dynasm!(asm
                ; cmp w9, #start as u32
                ; b.eq =>*target_label
            );
        } else {
            let next = asm.new_dynamic_label();
            dynasm!(asm
                ; cmp w9, #start as u32
                ; b.lo =>next
                ; cmp w9, #end as u32
                ; b.ls =>*target_label
                ; =>next
            );
        }
    }

    dynasm!(asm
        ; b =>dead_label
    );

    Ok(())
}

/// Computes byte ranges for efficient transitions.
fn compute_byte_ranges(state: &MaterializedState) -> Vec<(u8, u8, DfaStateId)> {
    let mut ranges = Vec::new();
    let mut current_target: Option<DfaStateId> = None;
    let mut range_start = 0u8;

    for byte in 0..=255u8 {
        let target = state.transitions[byte as usize];

        match (current_target, target) {
            (None, Some(t)) => {
                current_target = Some(t);
                range_start = byte;
            }
            (Some(curr), Some(t)) if curr == t => {}
            (Some(curr), _) => {
                ranges.push((range_start, byte - 1, curr));
                current_target = target;
                range_start = byte;
            }
            (None, None) => {}
        }

        if byte == 255 {
            if let Some(t) = current_target {
                ranges.push((range_start, byte, t));
            }
        }
    }

    ranges
}

/// Emits the dead state code.
fn emit_dead_state(
    asm: &mut Assembler,
    dead_label: DynamicLabel,
    no_match_label: DynamicLabel,
    restart_label: Option<DynamicLabel>,
    dispatch_label: Option<DynamicLabel>,
    has_word_boundary: bool,
) -> Result<()> {
    dynasm!(asm
        ; .align 4
        ; =>dead_label
    );

    if let Some(restart) = restart_label {
        // Check if we already have a match
        dynasm!(asm
            ; cmp x22, #0
            ; b.ge =>no_match_label
            // Advance search position
            ; add x23, x23, #1
            ; add x9, x20, x23
            ; cmp x9, x21
            ; b.hs =>no_match_label
            // Reset position to search start
            ; mov x19, x23
        );

        if has_word_boundary {
            if let Some(dispatch) = dispatch_label {
                let is_word = asm.new_dynamic_label();
                let not_word = asm.new_dynamic_label();

                // Classify byte at position (x23 - 1)
                dynasm!(asm
                    ; sub x9, x23, #1
                    ; ldrb w9, [x20, x9]
                    ; mov x24, #0              // Assume NonWord

                    // Check 0-9 (0x30-0x39)
                    ; cmp w9, #0x30
                    ; b.lo =>not_word
                    ; cmp w9, #0x39
                    ; b.ls =>is_word

                    // Check A-Z (0x41-0x5A)
                    ; cmp w9, #0x41
                    ; b.lo =>not_word
                    ; cmp w9, #0x5A
                    ; b.ls =>is_word

                    // Check _ (0x5F)
                    ; cmp w9, #0x5F
                    ; b.eq =>is_word

                    // Check a-z (0x61-0x7A)
                    ; cmp w9, #0x61
                    ; b.lo =>not_word
                    ; cmp w9, #0x7A
                    ; b.ls =>is_word
                    ; b =>not_word

                    ; =>is_word
                    ; mov x24, #1

                    ; =>not_word
                    ; b =>dispatch
                );
            } else {
                dynasm!(asm
                    ; b =>restart
                );
            }
        } else {
            dynasm!(asm
                ; b =>restart
            );
        }
    } else {
        // Anchored: no retry
        dynasm!(asm
            ; b =>no_match_label
        );
    }

    Ok(())
}

/// Emits the no-match epilogue.
fn emit_no_match(
    asm: &mut Assembler,
    no_match_label: DynamicLabel,
    _has_word_boundary: bool,
) -> Result<()> {
    let truly_no_match = asm.new_dynamic_label();
    let return_match = asm.new_dynamic_label();

    dynasm!(asm
        ; =>no_match_label
        // Check if we have a saved match
        ; cmp x22, #0
        ; b.lt =>truly_no_match
        // Pack result: (x23 << 32) | x22
        ; lsl x0, x23, #32
        ; orr x0, x0, x22
        ; b =>return_match

        ; =>truly_no_match
        ; movn x0, 0

        ; =>return_match
        // Restore callee-saved registers
        ; ldp x23, x24, [sp], #16
        ; ldp x21, x22, [sp], #16
        ; ldp x19, x20, [sp], #16
        ; ldp x29, x30, [sp], #16
        ; ret
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_arm64_jit_available() {
        // Basic test that the module compiles
        assert!(true);
    }
}
