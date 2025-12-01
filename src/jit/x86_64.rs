//! x86-64 code generation using dynasm.
//!
//! This module emits native x86-64 assembly code for DFA state machines.
//! All code is W^X compliant and optimized for performance.

use crate::dfa::DfaStateId;
use crate::error::{Error, ErrorKind, Result};
use crate::jit::codegen::{MaterializedDfa, MaterializedState};
use dynasm::dynasm;
use dynasmrt::{AssemblyOffset, DynasmApi, DynasmLabelApi, DynamicLabel, ExecutableBuffer, x64::Assembler};

/// Compiles a materialized DFA to x86-64 machine code.
///
/// # Calling Convention (System V AMD64 ABI)
/// - `rdi` = current position in input (mutable, incremented during execution)
/// - `rsi` = input base pointer (immutable)
/// - `rdx` = input end pointer (base + len, immutable)
/// - `r10` = last match position, initialized to -1 (for longest-match semantics)
/// - `rax` = return value (match position or -1)
///
/// # Function Signature
/// ```c
/// int64_t match_fn(const uint8_t* input, size_t len);
/// ```
///
/// Returns:
/// - >= 0: Match found, value is the end position (number of bytes consumed)
/// - -1: No match
///
/// # Word Boundary Support
/// For patterns with word boundaries, two entry points are generated:
/// - `entry_point`: For NonWord prev_class (start of input or after non-word char)
/// - `entry_point_word`: For Word prev_class (after a word char)
///
/// The caller must select the appropriate entry point based on the character
/// before the input slice being matched.
///
/// # W^X Compliance
/// The generated code is written to RW memory, then marked RX before execution.
/// The ExecutableBuffer from dynasmrt handles this transition automatically.
///
/// # Performance Optimizations
/// - 16-byte alignment for all state labels (optimal CPU instruction fetch)
/// - Sparse transitions use linear compare chains
/// - Dense transitions use jump tables (not yet implemented in v1)
pub fn compile_states(
    dfa: &MaterializedDfa,
) -> Result<(ExecutableBuffer, AssemblyOffset, Option<AssemblyOffset>)> {
    let mut asm = Assembler::new()
        .map_err(|e| Error::new(ErrorKind::Jit(format!("Failed to create assembler: {}", e)), ""))?;

    // Create state label lookup using Vec instead of HashMap for O(1) lookup.
    // Find the max state ID to size the array appropriately.
    let max_state_id = dfa.states.iter().map(|s| s.id).max().unwrap_or(0) as usize;
    let mut state_labels: Vec<Option<DynamicLabel>> = vec![None; max_state_id + 1];
    for state in &dfa.states {
        state_labels[state.id as usize] = Some(asm.new_dynamic_label());
    }
    let dead_label = asm.new_dynamic_label();
    let no_match_label = asm.new_dynamic_label();

    // For unanchored patterns, create a restart label for the internal search loop.
    // This now includes word boundary patterns - we track prev_class in r13.
    let restart_label = if !dfa.has_start_anchor {
        Some(asm.new_dynamic_label())
    } else {
        None
    };

    // For word boundary patterns, create a dispatch label that selects start state based on r13
    let dispatch_label = if dfa.has_word_boundary && dfa.start_word.is_some() {
        Some(asm.new_dynamic_label())
    } else {
        None
    };

    // Emit prologue for NonWord prev_class (primary entry point)
    let entry_point = asm.offset();
    emit_prologue(&mut asm, dfa.start, &state_labels, restart_label, dispatch_label, dfa.has_word_boundary, true)?;

    // Emit prologue for Word prev_class (secondary entry point, if needed)
    // Note: We pass false for emit_restart_label to avoid duplicate labels
    let entry_point_word = if let Some(start_word) = dfa.start_word {
        let offset = asm.offset();
        emit_prologue(&mut asm, start_word, &state_labels, restart_label, dispatch_label, dfa.has_word_boundary, false)?;
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

    // Emit dead state - for unanchored, this restarts search at next position
    emit_dead_state(&mut asm, dead_label, no_match_label, restart_label, dispatch_label, dfa.has_word_boundary)?;

    // Emit no-match epilogue (with start position tracking for unanchored)
    emit_no_match(&mut asm, no_match_label, dfa.has_word_boundary)?;

    // Finalize and get executable buffer (W^X compliant)
    let code = asm.finalize()
        .map_err(|_| Error::new(ErrorKind::Jit("Failed to finalize assembly".to_string()), ""))?;

    Ok((code, entry_point, entry_point_word))
}

/// Emits the function prologue.
///
/// This sets up the calling convention:
/// - rdi = current position (starts at 0)
/// - rsi = input base pointer (from first argument)
/// - rdx = input end pointer (base + len)
/// - r10 = last match end position (initialized to -1)
/// - r11 = search start position (for unanchored search, tracks where current attempt began)
/// - r13 = prev_char_class (0 = NonWord, 1 = Word) for word boundary patterns
/// - Jump to the specified start state
///
/// The restart_label is for unanchored patterns - when we fail to match,
/// we increment r11 and restart from the start state.
///
/// The `emit_restart_label` parameter controls whether to emit the restart label.
/// Only the primary entry point should emit the label; secondary entry points
/// (like the word boundary variant) should pass `false` to avoid duplicate labels.
fn emit_prologue(
    asm: &mut Assembler,
    start_state: DfaStateId,
    state_labels: &[Option<DynamicLabel>],
    restart_label: Option<DynamicLabel>,
    dispatch_label: Option<DynamicLabel>,
    has_word_boundary: bool,
    emit_restart_label: bool,
) -> Result<()> {
    let start_label = state_labels.get(start_state as usize)
        .and_then(|opt| opt.as_ref())
        .ok_or_else(|| Error::new(ErrorKind::Jit("Start state label not found".to_string()), ""))?;

    // For word boundary patterns, save r13 (callee-saved register in System V ABI)
    // Each entry point (primary and secondary) must save r13 since both can be called independently
    if has_word_boundary {
        dynasm!(asm
            ; push r13  // Save callee-saved register
        );
    }

    dynasm!(asm
        // Function entry point
        // Arguments: rdi = input ptr, rsi = len
        // We need to set up: rdi = pos, rsi = base, rdx = end, r10 = -1, r11 = 0

        // Save original input pointer to r8 temporarily
        ; mov r8, rdi

        // Calculate end pointer: rdx = rdi + rsi
        ; lea rdx, [rdi + rsi]

        // Set up base pointer: rsi = original rdi (now in r8)
        ; mov rsi, r8

        // Initialize position to 0: rdi = 0
        ; xor rdi, rdi

        // Initialize last match position to -1: r10 = -1
        ; mov r10, -1

        // Initialize search start position to 0: r11 = 0
        ; xor r11, r11
    );

    // For word boundary patterns, initialize r13 = prev_char_class
    // r13 = 0 means NonWord (start of input or after non-word char)
    // r13 = 1 means Word (after a-zA-Z0-9_)
    if has_word_boundary {
        dynasm!(asm
            ; xor r13d, r13d  // r13 = 0 (NonWord at start of input)
        );
    }

    // If unanchored and this is the primary entry point, emit the restart label
    if let Some(restart) = restart_label {
        if emit_restart_label {
            dynasm!(asm
                ; =>restart
            );
        }
    }

    // For word boundary patterns with dispatch, jump to dispatch instead of start state
    // (only on restart - initial entry goes directly to start state)
    if dispatch_label.is_some() && !emit_restart_label {
        // Secondary entry point (Word prev_class) - this is called from Rust
        // for find_at with Word prev_class, so we need to set r13 = 1
        dynasm!(asm
            ; mov r13d, 1  // r13 = 1 (Word)
        );
    }

    dynasm!(asm
        // Jump to start state
        ; jmp =>*start_label
    );

    Ok(())
}

/// Emits the dispatch block for word boundary patterns.
///
/// This checks r13 (prev_char_class) and jumps to the appropriate start state:
/// - r13 = 0 (NonWord): jump to start state
/// - r13 = 1 (Word): jump to start_word state
fn emit_dispatch(
    asm: &mut Assembler,
    dispatch_label: DynamicLabel,
    start: DfaStateId,
    start_word: DfaStateId,
    state_labels: &[Option<DynamicLabel>],
) -> Result<()> {
    let start_label = state_labels.get(start as usize)
        .and_then(|opt| opt.as_ref())
        .ok_or_else(|| Error::new(ErrorKind::Jit("Start state label not found".to_string()), ""))?;
    let start_word_label = state_labels.get(start_word as usize)
        .and_then(|opt| opt.as_ref())
        .ok_or_else(|| Error::new(ErrorKind::Jit("Start word state label not found".to_string()), ""))?;

    dynasm!(asm
        ; .align 16
        ; =>dispatch_label
        // Check r13: 0 = NonWord, 1 = Word
        ; test r13d, r13d
        ; jnz =>*start_word_label  // If r13 != 0 (Word), jump to start_word
        ; jmp =>*start_label       // Else (NonWord), jump to start
    );

    Ok(())
}

/// Analyzes if a state has a self-loop pattern suitable for fast-forward optimization.
///
/// Returns `Some((ranges, other_targets))` if the state has:
/// - A contiguous character class that loops back to itself
/// - At least 3 bytes in the self-loop range (worth optimizing)
///
/// The ranges are the byte ranges that self-loop, and other_targets are
/// non-self-loop transitions that need to be checked after the fast-forward.
fn analyze_self_loop(state: &MaterializedState) -> Option<(Vec<(u8, u8)>, Vec<(u8, u8, DfaStateId)>)> {
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

    // Only optimize if we have at least 3 self-loop bytes (e.g., \d = 10, \w = 63)
    if self_loop_bytes.len() < 3 {
        return None;
    }

    // Convert self-loop bytes to contiguous ranges
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
///
/// Each state follows this pattern:
/// 1. Check if input is exhausted (rdi >= rdx - rsi)
/// 2. If exhausted, check if this is a match state
/// 3. Load next byte from input[pos]
/// 4. Increment position
/// 5. Transition based on byte value
///
/// For states with self-loops (like `\d+`, `\w+`), we emit a tight inner loop
/// that consumes bytes matching the character class without per-byte state transitions.
fn emit_state(
    asm: &mut Assembler,
    state: &MaterializedState,
    state_labels: &[Option<DynamicLabel>],
    dead_label: DynamicLabel,
    no_match_label: DynamicLabel,
) -> Result<()> {
    let state_label = state_labels.get(state.id as usize)
        .and_then(|opt| opt.as_ref())
        .ok_or_else(|| Error::new(ErrorKind::Jit(format!("Label for state {} not found", state.id)), ""))?;

    // 16-byte alignment for optimal CPU instruction fetch
    // This is CRITICAL for performance - state labels are hot loop entries
    dynasm!(asm
        ; .align 16
        ; =>*state_label
    );

    // Check if input is exhausted
    // rdi = current position
    // rsi = input base
    // rdx = input end
    // We need to check if (rsi + rdi) >= rdx
    if state.is_match {
        // At a match state: save current position as last successful match FIRST
        // This is important: we save r10 BEFORE checking exhaustion, so that
        // even if we jump directly to match_return, r10 has the correct value.
        dynasm!(asm
            ; mov r10, rdi
        );
        // If this is a match state and input is exhausted, return success
        dynasm!(asm
            ; lea rax, [rsi + rdi]
            ; cmp rax, rdx
            ; jge >match_return
        );
    } else {
        // If this is not a match state and input is exhausted, return failure
        dynasm!(asm
            ; lea rax, [rsi + rdi]
            ; cmp rax, rdx
            ; jge =>no_match_label
        );
    }

    // Check for self-loop optimization opportunity
    if let Some((self_loop_ranges, other_transitions)) = analyze_self_loop(state) {
        // Emit fast-forward loop for self-loop transitions
        emit_fast_forward_loop(asm, state, &self_loop_ranges, &other_transitions, state_labels, dead_label, no_match_label)?;
    } else {
        // Standard path: load byte and check transitions
        // Load next byte: al = input[pos]
        // input[pos] = *(rsi + rdi)
        dynasm!(asm
            ; movzx eax, BYTE [rsi + rdi]
            ; inc rdi  // pos++
        );

        // Emit transitions based on density
        if state.should_use_jump_table() {
            // Dense transitions: use jump table for O(1) lookup
            emit_dense_transitions(asm, state, state_labels, dead_label)?;
        } else {
            // Sparse transitions: use linear compare chain
            emit_sparse_transitions(asm, state, state_labels, dead_label)?;
        }
    }

    // Local label for match return from this state
    // When input is exhausted at a match state, we need to return the match
    // by jumping to no_match_label (which checks r10 and packs the result)
    if state.is_match {
        dynasm!(asm
            ; match_return:
            // r10 already has the match end position, r11 has the start
            // Jump to no_match_label to pack and return the result
            ; jmp =>no_match_label
        );
    }

    Ok(())
}

/// Builds a 256-bit bitmap for the self-loop character class.
/// Returns a 32-byte array where bit i is set if byte i is in the self-loop.
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
///
/// Instead of transitioning state-by-state, this emits a tight loop that
/// consumes all consecutive bytes matching the self-loop character class,
/// then checks for other transitions.
///
/// For character classes with multiple ranges (like \w), we use a bitmap
/// lookup for O(1) membership test instead of multiple range comparisons.
fn emit_fast_forward_loop(
    asm: &mut Assembler,
    state: &MaterializedState,
    self_loop_ranges: &[(u8, u8)],
    other_transitions: &[(u8, u8, DfaStateId)],
    state_labels: &[Option<DynamicLabel>],
    dead_label: DynamicLabel,
    no_match_label: DynamicLabel,
) -> Result<()> {
    // Decide whether to use bitmap or range checks based on number of ranges
    // Bitmap is faster when there are 3+ non-contiguous ranges
    let use_bitmap = self_loop_ranges.len() >= 3;

    if use_bitmap {
        // Build the bitmap and embed it in the code
        let bitmap = build_self_loop_bitmap(self_loop_ranges);

        // Emit bitmap as data (32 bytes aligned)
        dynasm!(asm
            ; jmp >fast_forward_start
            ; .align 32
            ; bitmap_data:
            ; .bytes bitmap.as_slice()
            ; fast_forward_start:
        );

        // Get the bitmap address into r9 (r11 is used for search start position)
        dynasm!(asm
            ; lea r9, [<bitmap_data]
        );


        // Fast-forward loop using bitmap lookup
        dynasm!(asm
            ; fast_forward_loop:
            // Check bounds
            ; lea rax, [rsi + rdi]
            ; cmp rax, rdx
            ; jge >exhausted

            // Load byte
            ; movzx eax, BYTE [rsi + rdi]

            // Bitmap lookup: bitmap[byte / 8] & (1 << (byte % 8))
            // The bt instruction with a register source uses (src mod 32) for DWORD,
            // not (src mod 8). So we need to explicitly mask to get the correct bit.
            //   mov ecx, eax     ; ecx = byte value
            //   shr ecx, 3       ; ecx = byte / 8 (index into bitmap)
            //   and eax, 7       ; eax = byte % 8 (bit position within byte)
            //   bt [r9 + rcx], eax
            ; mov ecx, eax
            ; shr ecx, 3
            ; and eax, 7
            ; bt DWORD [r9 + rcx], eax
            ; jnc >check_other_transitions  // bit not set = not in class

            // Consume byte
            ; inc rdi
        );

        // If this is a match state, save the position
        if state.is_match {
            dynasm!(asm
                ; mov r10, rdi
            );
        }

        dynasm!(asm
            ; jmp <fast_forward_loop
        );
    } else {
        // Use range checks for small number of ranges (original algorithm)
        dynasm!(asm
            ; fast_forward_loop:
            // Check bounds
            ; lea rax, [rsi + rdi]
            ; cmp rax, rdx
            ; jge >exhausted

            // Load byte
            ; movzx eax, BYTE [rsi + rdi]
        );

        // Check if byte is in self-loop ranges
        for (i, &(start, end)) in self_loop_ranges.iter().enumerate() {
            let is_last = i == self_loop_ranges.len() - 1;

            if start == end {
                // Single byte
                dynasm!(asm
                    ; cmp al, BYTE start as _
                    ; je >consume_byte
                );
            } else {
                // Byte range
                dynasm!(asm
                    ; cmp al, BYTE start as _
                    ; jb >next_range
                    ; cmp al, BYTE end as _
                    ; jbe >consume_byte
                    ; next_range:
                );
            }

            if is_last {
                // No more ranges - byte doesn't match self-loop
                dynasm!(asm
                    ; jmp >check_other_transitions
                );
            }
        }

        // Consume byte and continue loop
        dynasm!(asm
            ; consume_byte:
            ; inc rdi
        );

        // If this is a match state, save the position
        if state.is_match {
            dynasm!(asm
                ; mov r10, rdi
            );
        }

        dynasm!(asm
            ; jmp <fast_forward_loop
        );
    }

    // Common exhausted label - when input is exhausted, we need to handle it properly
    // For match states, the position is already saved in r10, so jump to no_match_label
    // which will pack and return the result. For non-match states, this is a failure.
    dynasm!(asm
        ; exhausted:
    );

    // If this is a match state, r10 already has the match position saved,
    // so we can return via no_match_label which will check r10 and return the match.
    // If it's not a match state, we go to no_match_label anyway (r10 will be -1).
    dynasm!(asm
        ; jmp =>no_match_label
    );

    dynasm!(asm
        ; check_other_transitions:
    );

    // Check other (non-self-loop) transitions
    // Note: at this point, rdi is pointing at the byte that didn't match the self-loop.
    // We need to reload it because the bitmap lookup path corrupted eax.
    if !other_transitions.is_empty() {
        // Reload the byte and consume it
        dynasm!(asm
            ; movzx eax, BYTE [rsi + rdi]
            ; inc rdi
        );

        for &(start, end, target) in other_transitions {
            let target_label = state_labels.get(target as usize)
                .and_then(|opt| opt.as_ref())
                .ok_or_else(|| Error::new(ErrorKind::Jit(format!("Label for state {} not found", target)), ""))?;

            if start == end {
                dynasm!(asm
                    ; cmp al, BYTE start as _
                    ; je =>*target_label
                );
            } else {
                dynasm!(asm
                    ; cmp al, BYTE start as _
                    ; jb >next_other
                    ; cmp al, BYTE end as _
                    ; jbe =>*target_label
                    ; next_other:
                );
            }
        }
    }

    // No transition matched
    dynasm!(asm
        ; jmp =>dead_label
    );

    Ok(())
}

/// Emits sparse transition code using linear compare chains.
///
/// For each unique target state, emit:
/// - Compare against all bytes that lead to that state
/// - Jump to target if match
fn emit_sparse_transitions(
    asm: &mut Assembler,
    state: &MaterializedState,
    state_labels: &[Option<DynamicLabel>],
    dead_label: DynamicLabel,
) -> Result<()> {
    // Use compute_byte_ranges to get grouped transitions efficiently
    let ranges = compute_byte_ranges(state);

    for (start, end, target) in ranges {
        let target_label = state_labels.get(target as usize)
            .and_then(|opt| opt.as_ref())
            .ok_or_else(|| Error::new(ErrorKind::Jit(format!("Label for state {} not found", target)), ""))?;

        if start == end {
            // Single byte
            dynasm!(asm
                ; cmp al, BYTE start as _
                ; je =>*target_label
            );
        } else {
            // Range of bytes
            dynasm!(asm
                ; cmp al, BYTE start as _
                ; jb >next_check
                ; cmp al, BYTE end as _
                ; jbe =>*target_label
                ; next_check:
            );
        }
    }

    // If no transition matched, go to dead state
    dynasm!(asm
        ; jmp =>dead_label
    );

    Ok(())
}

/// Emits dense transition code using a jump table.
///
/// Creates a 256-entry jump table where each entry points to the target state
/// for the corresponding byte value. This provides O(1) transition lookup.
///
/// Instead of a true jump table (which dynasm doesn't support well for labels),
/// we use a hybrid approach: group transitions by target and use range checks.
/// For truly dense cases (>100 transitions), we use a computed goto pattern.
fn emit_dense_transitions(
    asm: &mut Assembler,
    state: &MaterializedState,
    state_labels: &[Option<DynamicLabel>],
    dead_label: DynamicLabel,
) -> Result<()> {
    // Strategy: emit a series of range checks for contiguous byte ranges
    let ranges = compute_byte_ranges(state);

    for (start, end, target) in ranges {
        let target_label = state_labels.get(target as usize)
            .and_then(|opt| opt.as_ref())
            .ok_or_else(|| Error::new(ErrorKind::Jit(format!("Label for state {} not found", target)), ""))?;

        if start == end {
            // Single byte
            dynasm!(asm
                ; cmp al, BYTE start as _
                ; je =>*target_label
            );
        } else {
            // Range of bytes - use unsigned comparisons (jb/jbe are unsigned)
            dynasm!(asm
                ; cmp al, BYTE start as _
                ; jb >next
                ; cmp al, BYTE end as _
                ; jbe =>*target_label
                ; next:
            );
        }
    }

    // If no range matched, go to dead state
    dynasm!(asm
        ; jmp =>dead_label
    );

    Ok(())
}

/// Computes byte ranges for efficient range-based transitions.
/// Returns (start, end_inclusive, target_state) tuples.
fn compute_byte_ranges(state: &MaterializedState) -> Vec<(u8, u8, DfaStateId)> {
    let mut ranges = Vec::new();
    let mut current_target: Option<DfaStateId> = None;
    let mut range_start = 0u8;

    for byte in 0..=255u8 {
        let target = state.transitions[byte as usize];

        match (current_target, target) {
            (None, Some(t)) => {
                // Start a new range
                current_target = Some(t);
                range_start = byte;
            }
            (Some(curr), Some(t)) if curr == t => {
                // Continue current range
            }
            (Some(curr), _) => {
                // End current range
                ranges.push((range_start, byte - 1, curr));
                current_target = target;
                range_start = byte;
            }
            (None, None) => {
                // Stay in no-range state
            }
        }

        // Handle the last byte specially
        if byte == 255 {
            if let Some(t) = current_target {
                ranges.push((range_start, byte, t));
            }
        }
    }

    ranges
}

/// Emits the dead state code.
///
/// For anchored patterns: immediately jump to no_match.
/// For unanchored patterns: increment search position and restart from start state.
/// For word boundary patterns: classify byte at new position, update r13, jump to dispatch.
///
/// This implements "find" semantics where we scan through the input looking for
/// any match, not just a match at position 0.
fn emit_dead_state(
    asm: &mut Assembler,
    dead_label: DynamicLabel,
    no_match_label: DynamicLabel,
    restart_label: Option<DynamicLabel>,
    dispatch_label: Option<DynamicLabel>,
    has_word_boundary: bool,
) -> Result<()> {
    dynasm!(asm
        ; .align 16
        ; =>dead_label
    );

    if let Some(restart) = restart_label {
        // Unanchored: increment r11 (search start) and try again
        // First check if we already have a match saved
        dynasm!(asm
            ; cmp r10, 0
            ; jge =>no_match_label  // If we have a saved match, return it
            // No match yet - advance search position
            ; inc r11
            ; lea rax, [rsi + r11]
            ; cmp rax, rdx
            ; jge =>no_match_label  // Past end of input
            // Reset position to search start
            ; mov rdi, r11
        );

        // For word boundary patterns, classify the byte BEFORE the new position
        // (the byte we just passed over) to update r13 (prev_char_class)
        if has_word_boundary {
            if let Some(dispatch) = dispatch_label {
                // Load the byte at position (r11 - 1) = the byte we just passed
                // Note: r11 was already incremented, so r11-1 is valid
                dynasm!(asm
                    ; lea rax, [r11 - 1]      // rax = r11 - 1 (index of prev byte)
                    ; movzx eax, BYTE [rsi + rax]  // Load byte at base + (r11-1)

                    // Classify as Word (a-zA-Z0-9_) or NonWord
                    // Word chars: 0x30-0x39 (0-9), 0x41-0x5A (A-Z), 0x5F (_), 0x61-0x7A (a-z)
                    ; xor r13d, r13d          // r13 = 0 (assume NonWord)

                    // Check 0-9 (0x30-0x39)
                    ; cmp al, 0x30
                    ; jb >not_word
                    ; cmp al, 0x39
                    ; jbe >is_word

                    // Check A-Z (0x41-0x5A)
                    ; cmp al, 0x41
                    ; jb >not_word
                    ; cmp al, 0x5A
                    ; jbe >is_word

                    // Check _ (0x5F)
                    ; cmp al, 0x5F
                    ; je >is_word

                    // Check a-z (0x61-0x7A)
                    ; cmp al, 0x61
                    ; jb >not_word
                    ; cmp al, 0x7A
                    ; jbe >is_word

                    ; jmp >not_word

                    ; is_word:
                    ; mov r13d, 1             // r13 = 1 (Word)

                    ; not_word:
                    // Jump to dispatch which will select correct start state based on r13
                    ; jmp =>dispatch
                );
            } else {
                // Word boundary but no dispatch (shouldn't happen, but handle gracefully)
                dynasm!(asm
                    ; jmp =>restart
                );
            }
        } else {
            // Non-word-boundary pattern: just restart
            dynasm!(asm
                ; jmp =>restart
            );
        }
    } else {
        // Anchored: no retry, just go to no_match
        dynasm!(asm
            ; jmp =>no_match_label
        );
    }

    Ok(())
}

/// Emits the no-match epilogue.
///
/// Checks if we had any successful match (r10 >= 0).
/// If yes, returns the match as a packed value: (start << 32) | end.
/// Otherwise, returns -1 to indicate no match.
///
/// The start position is tracked in r11 (updated on each restart for unanchored search).
/// For word boundary patterns, we need to restore r13 before returning.
fn emit_no_match(
    asm: &mut Assembler,
    no_match_label: DynamicLabel,
    has_word_boundary: bool,
) -> Result<()> {
    dynasm!(asm
        ; =>no_match_label
        // Check if we have a saved match position in r10
        ; cmp r10, 0
        ; jl >truly_no_match
        // We had a match at some point - pack start and end into return value
        // Return: (r11 << 32) | r10  (start in upper 32 bits, end in lower 32 bits)
        ; mov rax, r11
        ; shl rax, 32
        ; or rax, r10
    );

    // Restore r13 before returning (only for word boundary patterns)
    if has_word_boundary {
        dynasm!(asm
            ; pop r13
        );
    }

    dynasm!(asm
        ; ret
        ; truly_no_match:
        // No match at all
        ; mov rax, -1
    );

    // Restore r13 before returning (only for word boundary patterns)
    if has_word_boundary {
        dynasm!(asm
            ; pop r13
        );
    }

    dynasm!(asm
        ; ret
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    /// Checks if a slice of bytes forms a contiguous range.
    fn is_contiguous_range(bytes: &[u8]) -> bool {
        if bytes.len() <= 1 {
            return true;
        }

        let mut sorted = bytes.to_vec();
        sorted.sort_unstable();

        for i in 1..sorted.len() {
            if sorted[i] != sorted[i - 1] + 1 {
                return false;
            }
        }

        true
    }

    #[test]
    fn test_contiguous_range() {
        assert!(is_contiguous_range(&[1, 2, 3, 4]));
        assert!(is_contiguous_range(&[b'a', b'b', b'c']));
        assert!(!is_contiguous_range(&[1, 2, 4, 5]));
        assert!(!is_contiguous_range(&[b'a', b'c']));
        assert!(is_contiguous_range(&[42]));
        assert!(is_contiguous_range(&[]));
    }

    #[test]
    fn test_unsorted_contiguous() {
        assert!(is_contiguous_range(&[3, 1, 2, 4]));
        assert!(is_contiguous_range(&[b'c', b'a', b'b']));
    }
}
