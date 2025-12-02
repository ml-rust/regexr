//! x86-64 JIT code generation for Tagged NFA.
//!
//! This module contains the TaggedNfaJitCompiler which generates x86-64 assembly
//! code for Thompson NFA simulation with captures.

use crate::error::{Error, ErrorKind, Result};
use crate::hir::CodepointClass;
use crate::nfa::{ByteClass, ByteRange, Nfa, NfaInstruction, StateId};

use super::super::{NfaLiveness, PatternStep, TaggedNfaContext};
use super::jit::TaggedNfaJit;

use dynasmrt::{dynasm, DynasmApi};

/// Internal compiler for Tagged NFA JIT.
///
/// Generates x86-64 assembly code for Thompson NFA simulation with captures.
/// Uses interpreter fallback for complex patterns (lookarounds, backrefs).
///
/// Currently, all patterns use interpreter fallback while the JIT infrastructure
/// is being developed. The compiler structure and register allocation are in place
/// for future phases to enable real JIT compilation.
#[allow(dead_code)]
pub(super) struct TaggedNfaJitCompiler {
    asm: dynasmrt::x64::Assembler,
    nfa: Nfa,
    liveness: NfaLiveness,
    /// Dynamic labels for each NFA state.
    state_labels: Vec<dynasmrt::DynamicLabel>,
    /// Label for the main thread processing loop.
    thread_loop_label: dynasmrt::DynamicLabel,
    /// Label for advancing to next position.
    advance_pos_label: dynasmrt::DynamicLabel,
    /// Label for recording a match.
    match_found_label: dynasmrt::DynamicLabel,
    /// Label for the done/epilogue section.
    done_label: dynasmrt::DynamicLabel,
    /// Label for adding a thread to the next worklist.
    add_thread_label: dynasmrt::DynamicLabel,
    /// CodepointClasses collected during pattern extraction.
    /// Boxed to ensure stable addresses for JIT code references.
    #[allow(clippy::vec_box)]
    codepoint_classes: Vec<Box<CodepointClass>>,
    /// Lookaround NFAs collected during pattern extraction.
    /// Boxed to ensure stable addresses for JIT helper function references.
    #[allow(clippy::vec_box)]
    lookaround_nfas: Vec<Box<Nfa>>,
}

impl TaggedNfaJitCompiler {
    /// Creates a new compiler for the given NFA.
    #[allow(dead_code)]
    #[allow(unused_imports)]
    fn new(nfa: Nfa, liveness: NfaLiveness) -> Result<Self> {
        use dynasmrt::DynasmLabelApi;

        let mut asm = dynasmrt::x64::Assembler::new().map_err(|e| {
            Error::new(
                ErrorKind::Jit(format!("Failed to create assembler: {:?}", e)),
                "",
            )
        })?;

        let state_labels: Vec<_> = (0..nfa.states.len())
            .map(|_| asm.new_dynamic_label())
            .collect();

        let thread_loop_label = asm.new_dynamic_label();
        let advance_pos_label = asm.new_dynamic_label();
        let match_found_label = asm.new_dynamic_label();
        let done_label = asm.new_dynamic_label();
        let add_thread_label = asm.new_dynamic_label();

        Ok(Self {
            asm,
            nfa,
            liveness,
            state_labels,
            thread_loop_label,
            advance_pos_label,
            match_found_label,
            done_label,
            add_thread_label,
            codepoint_classes: Vec::new(),
            lookaround_nfas: Vec::new(),
        })
    }

    /// Checks if this NFA can be JIT compiled, or needs interpreter fallback.
    ///
    /// Currently JIT supports:
    /// - Simple literal patterns (linear chain of byte transitions)
    /// - Character classes
    /// - Simple greedy repetition (a+, [a-z]+)
    ///
    /// Uses interpreter for:
    /// - Patterns with captures
    /// - Patterns with alternation (foo|bar)
    /// - Complex repetition (nested, non-greedy)
    /// - Patterns with lookarounds or backrefs
    /// - Patterns with anchors
    fn needs_interpreter_fallback(&self) -> bool {
        // Check for complex instructions
        for state in &self.nfa.states {
            if let Some(ref instr) = state.instruction {
                match instr {
                    // Lookarounds are now supported - handled by pattern extraction
                    NfaInstruction::PositiveLookahead(_)
                    | NfaInstruction::NegativeLookahead(_)
                    | NfaInstruction::PositiveLookbehind(_)
                    | NfaInstruction::NegativeLookbehind(_) => {}
                    // Anchors are now supported
                    NfaInstruction::StartOfText
                    | NfaInstruction::EndOfText
                    | NfaInstruction::StartOfLine
                    | NfaInstruction::EndOfLine => {}
                    // Backref is now supported - handled in compile_full()
                    NfaInstruction::Backref(_) => {}
                    // Word boundary and non-greedy are handled by pattern extraction
                    NfaInstruction::WordBoundary
                    | NfaInstruction::NotWordBoundary
                    | NfaInstruction::NonGreedyExit => {}
                    // Capture instructions are supported
                    NfaInstruction::CaptureStart(_) | NfaInstruction::CaptureEnd(_) => {}
                    // Codepoint class is handled by pattern extraction
                    NfaInstruction::CodepointClass(_, _) => {}
                }
            }
        }

        // Large NFAs generate too much code
        if self.nfa.states.len() > 256 {
            // NFA too large for efficient JIT - fall back to interpreter
            return true;
        }

        // Let extract_pattern_steps() do the detailed analysis
        // If it returns an empty vec, we need fallback
        false
    }

    /// Compiles the NFA to a TaggedNfaJit.
    pub(super) fn compile(nfa: Nfa, liveness: NfaLiveness) -> Result<TaggedNfaJit> {
        let compiler = Self::new(nfa, liveness)?;

        // Check if we need interpreter fallback
        if compiler.needs_interpreter_fallback() {
            return compiler.compile_with_fallback(None);
        }

        // Generate full JIT code
        compiler.compile_full()
    }

    /// Generates stub code that triggers interpreter fallback.
    /// If `steps` is provided, they will be used for fast TaggedNfa fallback.
    fn compile_with_fallback(mut self, steps: Option<Vec<PatternStep>>) -> Result<TaggedNfaJit> {
        // Record entry point offsets
        let find_offset = self.asm.offset();

        // find_fn: Returns -2 to trigger interpreter fallback
        dynasm!(self.asm
            ; mov rax, -2i32
            ; ret
        );

        let captures_offset = self.asm.offset();

        // captures_fn: Returns -2 to trigger interpreter fallback
        dynasm!(self.asm
            ; mov rax, -2i32
            ; ret
        );

        // find_fn doesn't need context - it just returns -2 immediately
        // The fast path will detect this and call the interpreter directly
        // Pass steps for fast TaggedNfa fallback
        self.finalize(find_offset, captures_offset, false, steps)
    }

    // Generates JIT code for simple linear patterns (literals).
    //
    // For a pattern like "abc", generates code that:
    // 1. Tries each starting position
    // 2. At each position, walks the linear NFA chain
    // 3. Returns on first match
    //
    // Register allocation (System V AMD64 ABI):
    // - rdi = input_ptr (argument, then scratch)
    // - rsi = input_len (argument)
    // - rbx = input_ptr (callee-saved)
    // - r12 = input_len (callee-saved)
    // - r13 = start_pos for current attempt (callee-saved)
    // - r14 = current_pos (absolute position in input) (callee-saved)
    // - rax = scratch / return value

    /// Check if pattern contains backreferences (recursively in alternations).
    fn has_backref(steps: &[PatternStep]) -> bool {
        steps.iter().any(|s| match s {
            PatternStep::Backref(_) => true,
            PatternStep::Alt(alternatives) => alternatives.iter().any(|alt| Self::has_backref(alt)),
            _ => false,
        })
    }

    /// Check if alternation contains unsupported patterns (recursive).
    fn has_unsupported_in_alt(alternatives: &[Vec<PatternStep>]) -> bool {
        for alt_steps in alternatives.iter() {
            for step in alt_steps.iter() {
                match step {
                    // Nested alternation - supported via recursive emit
                    PatternStep::Alt(inner_alts) => {
                        // Recursively check nested alternation
                        if Self::has_unsupported_in_alt(inner_alts) {
                            return true;
                        }
                    }
                    // Non-greedy patterns not supported in alternation JIT
                    PatternStep::NonGreedyPlus(_, _) | PatternStep::NonGreedyStar(_, _) => {
                        return true;
                    }
                    // Greedy with lookahead - supported via emit_alt_step
                    PatternStep::GreedyPlusLookahead(_, _, _)
                    | PatternStep::GreedyStarLookahead(_, _, _) => {}
                    // Lookarounds in alternation - not yet supported
                    PatternStep::PositiveLookahead(_)
                    | PatternStep::NegativeLookahead(_)
                    | PatternStep::PositiveLookbehind(_, _)
                    | PatternStep::NegativeLookbehind(_, _) => {
                        return true;
                    }
                    _ => {}
                }
            }
        }
        false
    }

    /// Check if a step consumes input bytes.
    fn step_consumes_input(step: &PatternStep) -> bool {
        match step {
            PatternStep::Byte(_)
            | PatternStep::ByteClass(_)
            | PatternStep::GreedyPlus(_)
            | PatternStep::GreedyStar(_)
            | PatternStep::GreedyCodepointPlus(_)
            | PatternStep::CodepointClass(_, _)
            | PatternStep::NonGreedyPlus(_, _)
            | PatternStep::NonGreedyStar(_, _)
            | PatternStep::GreedyPlusLookahead(_, _, _)
            | PatternStep::GreedyStarLookahead(_, _, _)
            | PatternStep::Backref(_) => true,

            PatternStep::Alt(alternatives) => {
                // Consumes input if any alternative consumes input
                alternatives
                    .iter()
                    .any(|alt| alt.iter().any(Self::step_consumes_input))
            }

            // Zero-width assertions don't consume input
            PatternStep::PositiveLookahead(_)
            | PatternStep::NegativeLookahead(_)
            | PatternStep::PositiveLookbehind(_, _)
            | PatternStep::NegativeLookbehind(_, _)
            | PatternStep::WordBoundary
            | PatternStep::NotWordBoundary
            | PatternStep::StartOfText
            | PatternStep::EndOfText
            | PatternStep::StartOfLine
            | PatternStep::EndOfLine
            | PatternStep::CaptureStart(_)
            | PatternStep::CaptureEnd(_) => false,
        }
    }

    fn compile_full(mut self) -> Result<TaggedNfaJit> {
        use dynasmrt::DynasmLabelApi;

        // Extract the pattern as a sequence of steps
        let steps = self.extract_pattern_steps();
        // Combine greedy quantifiers followed by lookahead for proper backtracking
        let steps = Self::combine_greedy_with_lookahead(steps);

        if steps.is_empty() {
            // Empty pattern or couldn't extract - fall back to interpreter
            return self.compile_with_fallback(None);
        }

        // Check if top-level has unsupported patterns (like nested Alt)
        // This prevents orphan labels during code generation
        for step in &steps {
            if let PatternStep::Alt(alternatives) = step {
                if Self::has_unsupported_in_alt(alternatives) {
                    return self.compile_with_fallback(Some(steps));
                }
            }
        }

        // Standalone lookahead patterns are now fully JIT compiled

        // Check if pattern contains backrefs - if so, find_fn falls back but captures_fn is JIT'd
        let has_backrefs = Self::has_backref(&steps);

        // Calculate minimum pattern length (for initial bounds check)
        let min_len = Self::calc_min_len(&steps);

        // =====================================================================
        // find_fn entry point
        // Signature: fn(input_ptr: *const u8, input_len: usize, ctx: *mut TaggedNfaContext) -> i64
        // =====================================================================
        let find_offset = self.asm.offset();

        // If pattern has backrefs, find_fn falls back to interpreter (backrefs need captures)
        if has_backrefs {
            dynasm!(self.asm
                ; mov rax, -2i32   // JIT_USE_INTERPRETER
                ; ret
            );

            // Generate captures_fn with full JIT (backrefs work here since we track captures)
            let captures_offset = self.emit_captures_fn(&steps)?;
            // find_fn falls back to interpreter, needs context
            return self.finalize(find_offset, captures_offset, true, None);
        }

        // Prologue - save callee-saved registers
        #[cfg(target_os = "windows")]
        dynasm!(self.asm
            ; push rdi          // Callee-saved on Windows
            ; push rsi          // Callee-saved on Windows
            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15
            // Windows: args in RCX, RDX -> move to RDI, RSI for internal use
            ; mov rdi, rcx
            ; mov rsi, rdx
        );

        #[cfg(not(target_os = "windows"))]
        dynasm!(self.asm
            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15
        );

        // Set up registers
        // rdi = input_ptr, rsi = input_len (after platform-specific setup)
        dynasm!(self.asm
            ; mov rbx, rdi      // rbx = input_ptr
            ; mov r12, rsi      // r12 = input_len
            ; xor r13d, r13d    // r13 = start_pos = 0
        );

        // Main loop: try matching at each start position
        let start_loop = self.asm.new_dynamic_label();
        let match_found = self.asm.new_dynamic_label();
        let no_match = self.asm.new_dynamic_label();
        let byte_mismatch = self.asm.new_dynamic_label();

        dynasm!(self.asm
            ; =>start_loop
            // Check if enough bytes remain for minimum pattern length
            ; mov rax, r12
            ; sub rax, r13              // remaining = len - start_pos
            ; cmp rax, min_len as i32
            ; jl =>no_match             // Not enough bytes remaining
        );

        // r14 = current absolute position (starts at start_pos)
        dynasm!(self.asm
            ; mov r14, r13              // r14 = start_pos (current position)
        );

        // For each step in the pattern, generate matching code
        for (step_idx, step) in steps.iter().enumerate() {
            match step {
                PatternStep::Byte(byte) => {
                    // Check bounds
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>byte_mismatch
                        ; movzx eax, BYTE [rbx + r14]
                        ; cmp al, *byte as i8
                        ; jne =>byte_mismatch
                        ; inc r14                   // Advance position
                    );
                }
                PatternStep::ByteClass(byte_class) => {
                    // Check bounds
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>byte_mismatch
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(&byte_class.ranges, byte_mismatch)?;
                    dynasm!(self.asm
                        ; inc r14                   // Advance position
                    );
                }
                PatternStep::GreedyPlus(byte_class) => {
                    // Check if there are remaining steps that consume input
                    let remaining = &steps[step_idx + 1..];
                    let needs_backtrack = remaining.iter().any(Self::step_consumes_input);

                    if needs_backtrack {
                        // Backtracking version: greedily match, then try remaining, backtrack on failure
                        self.emit_greedy_plus_with_backtracking(
                            &byte_class.ranges,
                            remaining,
                            byte_mismatch,
                        )?;
                        // Remaining steps already handled in backtracking code
                        break;
                    } else {
                        // Simple version: no backtracking needed
                        let loop_start = self.asm.new_dynamic_label();
                        let loop_done = self.asm.new_dynamic_label();

                        // First iteration (must match)
                        dynasm!(self.asm
                            ; cmp r14, r12
                            ; jge =>byte_mismatch       // Must have at least one byte
                            ; movzx eax, BYTE [rbx + r14]
                        );
                        self.emit_range_check(&byte_class.ranges, byte_mismatch)?;
                        dynasm!(self.asm
                            ; inc r14                   // Consumed first byte

                            // Loop for additional matches
                            ; =>loop_start
                            ; cmp r14, r12
                            ; jge =>loop_done           // End of input - done looping
                            ; movzx eax, BYTE [rbx + r14]
                        );
                        self.emit_range_check(&byte_class.ranges, loop_done)?;
                        dynasm!(self.asm
                            ; inc r14                   // Consumed another byte
                            ; jmp =>loop_start
                            ; =>loop_done
                        );
                    }
                }
                PatternStep::GreedyStar(byte_class) => {
                    // Check if there are remaining steps that consume input
                    let remaining = &steps[step_idx + 1..];
                    let needs_backtrack = remaining.iter().any(Self::step_consumes_input);

                    if needs_backtrack {
                        // Backtracking version
                        self.emit_greedy_star_with_backtracking(
                            &byte_class.ranges,
                            remaining,
                            byte_mismatch,
                        )?;
                        break;
                    } else {
                        // Simple version: no backtracking needed
                        let loop_start = self.asm.new_dynamic_label();
                        let loop_done = self.asm.new_dynamic_label();

                        dynasm!(self.asm
                            ; =>loop_start
                            ; cmp r14, r12
                            ; jge =>loop_done           // End of input - done looping
                            ; movzx eax, BYTE [rbx + r14]
                        );
                        self.emit_range_check(&byte_class.ranges, loop_done)?;
                        dynasm!(self.asm
                            ; inc r14                   // Consumed a byte
                            ; jmp =>loop_start
                            ; =>loop_done
                        );
                    }
                }
                PatternStep::NonGreedyPlus(byte_class, suffix) => {
                    // Non-greedy one-or-more: match minimum (1), then try suffix
                    // If suffix fails, consume one more and retry
                    let try_suffix = self.asm.new_dynamic_label();
                    let consume_more = self.asm.new_dynamic_label();
                    let suffix_matched = self.asm.new_dynamic_label();

                    // Must match at least one
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>byte_mismatch       // Must have at least one byte
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(&byte_class.ranges, byte_mismatch)?;
                    dynasm!(self.asm
                        ; inc r14                   // Consumed first byte

                        ; =>try_suffix
                    );

                    // Try to match the suffix
                    self.emit_non_greedy_suffix_check(suffix, consume_more, suffix_matched)?;

                    // Suffix matched - continue
                    dynasm!(self.asm
                        ; jmp =>suffix_matched

                        ; =>consume_more
                        // Try to consume one more character
                        ; cmp r14, r12
                        ; jge =>byte_mismatch       // No more input - fail
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(&byte_class.ranges, byte_mismatch)?;
                    dynasm!(self.asm
                        ; inc r14                   // Consumed another byte
                        ; jmp =>try_suffix

                        ; =>suffix_matched
                    );
                }
                PatternStep::NonGreedyStar(byte_class, suffix) => {
                    // Non-greedy zero-or-more: try suffix first (zero matches)
                    // If suffix fails, consume one and retry
                    let try_suffix = self.asm.new_dynamic_label();
                    let consume_more = self.asm.new_dynamic_label();
                    let suffix_matched = self.asm.new_dynamic_label();

                    dynasm!(self.asm
                        ; =>try_suffix
                    );

                    // Try to match the suffix
                    self.emit_non_greedy_suffix_check(suffix, consume_more, suffix_matched)?;

                    // Suffix matched - continue
                    dynasm!(self.asm
                        ; jmp =>suffix_matched

                        ; =>consume_more
                        // Try to consume one more character
                        ; cmp r14, r12
                        ; jge =>byte_mismatch       // No more input - fail
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(&byte_class.ranges, byte_mismatch)?;
                    dynasm!(self.asm
                        ; inc r14                   // Consumed another byte
                        ; jmp =>try_suffix

                        ; =>suffix_matched
                    );
                }
                PatternStep::GreedyPlusLookahead(byte_class, lookahead_steps, is_positive) => {
                    // Greedy one-or-more with lookahead: greedily consume, then backtrack
                    // until the lookahead succeeds.
                    self.emit_greedy_plus_with_lookahead(
                        &byte_class.ranges,
                        lookahead_steps,
                        *is_positive,
                        byte_mismatch,
                    )?;
                }
                PatternStep::GreedyStarLookahead(byte_class, lookahead_steps, is_positive) => {
                    // Greedy zero-or-more with lookahead: greedily consume, then backtrack
                    // until the lookahead succeeds.
                    self.emit_greedy_star_with_lookahead(
                        &byte_class.ranges,
                        lookahead_steps,
                        *is_positive,
                        byte_mismatch,
                    )?;
                }
                PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_) => {
                    // Capture markers don't consume input - skip in find_fn
                    // (captures are handled separately in captures_fn)
                }
                PatternStep::CodepointClass(cpclass, _target) => {
                    // Unicode codepoint class - decode UTF-8 and check membership
                    self.emit_codepoint_class_check(cpclass, byte_mismatch)?;
                }
                PatternStep::GreedyCodepointPlus(cpclass) => {
                    // Check if there are remaining steps that consume input
                    let remaining = &steps[step_idx + 1..];
                    let needs_backtrack = remaining.iter().any(Self::step_consumes_input);

                    if needs_backtrack {
                        // Backtracking version: greedily match UTF-8, then try remaining, backtrack on failure
                        self.emit_greedy_codepoint_plus_with_backtracking(
                            cpclass,
                            remaining,
                            byte_mismatch,
                        )?;
                        break;
                    } else {
                        // Simple version: no backtracking needed
                        self.emit_greedy_codepoint_plus(cpclass, byte_mismatch)?;
                    }
                }
                PatternStep::WordBoundary => {
                    // Word boundary assertion - doesn't consume input
                    self.emit_word_boundary_check(byte_mismatch, true)?;
                }
                PatternStep::NotWordBoundary => {
                    // Not word boundary assertion - doesn't consume input
                    self.emit_word_boundary_check(byte_mismatch, false)?;
                }
                PatternStep::StartOfText => {
                    // Start of text: only matches at position 0
                    dynasm!(self.asm
                        ; test r14, r14
                        ; jnz =>byte_mismatch
                    );
                }
                PatternStep::EndOfText => {
                    // End of text: only matches at position == input_len
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jne =>byte_mismatch
                    );
                }
                PatternStep::StartOfLine => {
                    // Start of line: matches at position 0 OR after a newline
                    let at_start = self.asm.new_dynamic_label();
                    dynasm!(self.asm
                        ; test r14, r14
                        ; jz =>at_start
                        ; mov rax, r14
                        ; dec rax
                        ; movzx eax, BYTE [rbx + rax]
                        ; cmp al, 0x0A
                        ; jne =>byte_mismatch
                        ; =>at_start
                    );
                }
                PatternStep::EndOfLine => {
                    // End of line: matches at position == input_len OR before a newline
                    let at_end = self.asm.new_dynamic_label();
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; je =>at_end
                        ; movzx eax, BYTE [rbx + r14]
                        ; cmp al, 0x0A
                        ; jne =>byte_mismatch
                        ; =>at_end
                    );
                }
                PatternStep::PositiveLookahead(ref inner_steps) => {
                    // Positive lookahead: check if inner pattern matches at current position
                    // Zero-width: don't advance r14
                    self.emit_standalone_lookahead(inner_steps, byte_mismatch, true)?;
                }
                PatternStep::NegativeLookahead(ref inner_steps) => {
                    // Negative lookahead: check if inner pattern does NOT match at current position
                    // Zero-width: don't advance r14
                    self.emit_standalone_lookahead(inner_steps, byte_mismatch, false)?;
                }
                PatternStep::PositiveLookbehind(ref inner_steps, min_len) => {
                    // Positive lookbehind: check if inner pattern matches behind current position
                    self.emit_lookbehind_check(inner_steps, *min_len, byte_mismatch, true)?;
                }
                PatternStep::NegativeLookbehind(ref inner_steps, min_len) => {
                    // Negative lookbehind: check if inner pattern does NOT match behind
                    self.emit_lookbehind_check(inner_steps, *min_len, byte_mismatch, false)?;
                }
                PatternStep::Backref(_) => {
                    // Backrefs in find_fn are handled by early return above (has_backrefs check)
                    // This arm should never be reached
                    unreachable!("Backref in find_fn should have triggered early return");
                }
                PatternStep::Alt(alternatives) => {
                    // Check for nested alternations or unsupported steps before creating labels
                    // This prevents orphan labels when we fall back to interpreter
                    if Self::has_unsupported_in_alt(alternatives) {
                        return self.compile_with_fallback(Some(steps.clone()));
                    }

                    // Simple alternation: try each alternative in order
                    // Save current position, try each alternative, restore on failure
                    let alt_success = self.asm.new_dynamic_label();

                    // Save current position to r15
                    dynasm!(self.asm
                        ; mov r15, r14              // r15 = saved position
                    );

                    for (alt_idx, alt_steps) in alternatives.iter().enumerate() {
                        let is_last = alt_idx == alternatives.len() - 1;
                        let try_next_alt = if is_last {
                            byte_mismatch // Last alternative failing means overall failure
                        } else {
                            self.asm.new_dynamic_label()
                        };

                        // Generate code for this alternative
                        // Each step in the alternative jumps to try_next_alt on failure
                        for alt_step in alt_steps.iter() {
                            match alt_step {
                                PatternStep::Byte(byte) => {
                                    dynasm!(self.asm
                                        ; cmp r14, r12
                                        ; jge =>try_next_alt
                                        ; movzx eax, BYTE [rbx + r14]
                                        ; cmp al, *byte as i8
                                        ; jne =>try_next_alt
                                        ; inc r14
                                    );
                                }
                                PatternStep::ByteClass(byte_class) => {
                                    dynasm!(self.asm
                                        ; cmp r14, r12
                                        ; jge =>try_next_alt
                                        ; movzx eax, BYTE [rbx + r14]
                                    );
                                    self.emit_range_check(&byte_class.ranges, try_next_alt)?;
                                    dynasm!(self.asm
                                        ; inc r14
                                    );
                                }
                                PatternStep::GreedyPlus(byte_class) => {
                                    let loop_start = self.asm.new_dynamic_label();
                                    let loop_done = self.asm.new_dynamic_label();

                                    dynasm!(self.asm
                                        ; cmp r14, r12
                                        ; jge =>try_next_alt
                                        ; movzx eax, BYTE [rbx + r14]
                                    );
                                    self.emit_range_check(&byte_class.ranges, try_next_alt)?;
                                    dynasm!(self.asm
                                        ; inc r14
                                        ; =>loop_start
                                        ; cmp r14, r12
                                        ; jge =>loop_done
                                        ; movzx eax, BYTE [rbx + r14]
                                    );
                                    self.emit_range_check(&byte_class.ranges, loop_done)?;
                                    dynasm!(self.asm
                                        ; inc r14
                                        ; jmp =>loop_start
                                        ; =>loop_done
                                    );
                                }
                                PatternStep::GreedyStar(byte_class) => {
                                    let loop_start = self.asm.new_dynamic_label();
                                    let loop_done = self.asm.new_dynamic_label();

                                    dynasm!(self.asm
                                        ; =>loop_start
                                        ; cmp r14, r12
                                        ; jge =>loop_done
                                        ; movzx eax, BYTE [rbx + r14]
                                    );
                                    self.emit_range_check(&byte_class.ranges, loop_done)?;
                                    dynasm!(self.asm
                                        ; inc r14
                                        ; jmp =>loop_start
                                        ; =>loop_done
                                    );
                                }
                                PatternStep::Alt(inner_alternatives) => {
                                    // Nested alternation - emit recursively
                                    self.emit_nested_alt(inner_alternatives, try_next_alt)?;
                                }
                                PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_) => {
                                    // Capture markers in alternation - skip in find_fn
                                }
                                PatternStep::CodepointClass(cpclass, _target) => {
                                    // Unicode codepoint class in alternation
                                    self.emit_codepoint_class_check(cpclass, try_next_alt)?;
                                }
                                PatternStep::GreedyCodepointPlus(cpclass) => {
                                    // Greedy codepoint repetition in alternation
                                    self.emit_greedy_codepoint_plus(cpclass, try_next_alt)?;
                                }
                                PatternStep::WordBoundary => {
                                    // Word boundary in alternation - doesn't consume input
                                    self.emit_word_boundary_check(try_next_alt, true)?;
                                }
                                PatternStep::NotWordBoundary => {
                                    // Not word boundary in alternation - doesn't consume input
                                    self.emit_word_boundary_check(try_next_alt, false)?;
                                }
                                PatternStep::StartOfText => {
                                    // Start of text anchor in alternation
                                    dynasm!(self.asm
                                        ; test r14, r14         // r14 == 0?
                                        ; jnz =>try_next_alt    // Fail if not at start
                                    );
                                }
                                PatternStep::EndOfText => {
                                    // End of text anchor in alternation
                                    dynasm!(self.asm
                                        ; cmp r14, r12          // r14 == input_len?
                                        ; jne =>try_next_alt    // Fail if not at end
                                    );
                                }
                                PatternStep::StartOfLine => {
                                    // Start of line anchor in alternation
                                    let at_start = self.asm.new_dynamic_label();
                                    dynasm!(self.asm
                                        ; test r14, r14         // r14 == 0?
                                        ; jz =>at_start         // At start of text, it's start of line
                                        // Check if previous byte is newline
                                        ; mov rax, r14
                                        ; dec rax
                                        ; movzx eax, BYTE [rbx + rax]
                                        ; cmp al, 0x0A          // '\n'
                                        ; jne =>try_next_alt    // Not at start of line
                                        ; =>at_start
                                    );
                                }
                                PatternStep::EndOfLine => {
                                    // End of line anchor in alternation
                                    let at_end = self.asm.new_dynamic_label();
                                    dynasm!(self.asm
                                        ; cmp r14, r12          // r14 == input_len?
                                        ; je =>at_end           // At end of text, it's end of line
                                        // Check if current byte is newline
                                        ; movzx eax, BYTE [rbx + r14]
                                        ; cmp al, 0x0A          // '\n'
                                        ; jne =>try_next_alt    // Not at end of line
                                        ; =>at_end
                                    );
                                }
                                PatternStep::PositiveLookahead(_)
                                | PatternStep::NegativeLookahead(_)
                                | PatternStep::PositiveLookbehind(..)
                                | PatternStep::NegativeLookbehind(..) => {
                                    // Lookarounds in alternation - fall back to interpreter
                                    return self.compile_with_fallback(None);
                                }
                                PatternStep::Backref(_) => {
                                    // Backrefs in find_fn are handled by early return above
                                    unreachable!("Backref in find_fn alternation should have triggered early return");
                                }
                                PatternStep::NonGreedyPlus(_, _)
                                | PatternStep::NonGreedyStar(_, _) => {
                                    // Non-greedy in alternation - complex, fall back to interpreter
                                    return self.compile_with_fallback(None);
                                }
                                PatternStep::GreedyPlusLookahead(
                                    byte_class,
                                    lookahead_steps,
                                    is_positive,
                                ) => {
                                    // Greedy+ with lookahead in alternation
                                    self.emit_greedy_plus_with_lookahead_in_alt(
                                        &byte_class.ranges,
                                        lookahead_steps,
                                        *is_positive,
                                        try_next_alt,
                                    )?;
                                }
                                PatternStep::GreedyStarLookahead(
                                    byte_class,
                                    lookahead_steps,
                                    is_positive,
                                ) => {
                                    // Greedy* with lookahead in alternation
                                    self.emit_greedy_star_with_lookahead_in_alt(
                                        &byte_class.ranges,
                                        lookahead_steps,
                                        *is_positive,
                                        try_next_alt,
                                    )?;
                                }
                            }
                        }

                        // This alternative succeeded - jump past remaining alternatives
                        dynasm!(self.asm
                            ; jmp =>alt_success
                        );

                        // Label for trying next alternative (restore position first)
                        if !is_last {
                            dynasm!(self.asm
                                ; =>try_next_alt
                                ; mov r14, r15         // Restore position
                            );
                        }
                    }

                    // All alternatives tried and one succeeded
                    dynasm!(self.asm
                        ; =>alt_success
                    );
                }
            }
        }

        // All steps matched! r14 = end position
        dynasm!(self.asm
            ; jmp =>match_found
        );

        // Byte mismatch - try next position
        dynasm!(self.asm
            ; =>byte_mismatch
            ; inc r13                   // start_pos++
            ; jmp =>start_loop
        );

        // Match found - return (start << 32 | end)
        // r13 = start position, r14 = end position
        dynasm!(self.asm
            ; =>match_found
            ; mov rax, r13
            ; shl rax, 32               // rax = start << 32
            ; or rax, r14               // rax = (start << 32) | end
        );

        // Epilogue - restore callee-saved registers
        #[cfg(target_os = "windows")]
        dynasm!(self.asm
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; pop rsi
            ; pop rdi
            ; ret
        );
        #[cfg(not(target_os = "windows"))]
        dynasm!(self.asm
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; ret
        );

        // No match
        dynasm!(self.asm
            ; =>no_match
            ; mov rax, -1i32
        );

        // Epilogue - restore callee-saved registers
        #[cfg(target_os = "windows")]
        dynasm!(self.asm
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; pop rsi
            ; pop rdi
            ; ret
        );
        #[cfg(not(target_os = "windows"))]
        dynasm!(self.asm
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; ret
        );

        // =====================================================================
        // captures_fn - generates captures with proper capture tracking
        // =====================================================================
        // Check if pattern has captures
        let has_captures = steps
            .iter()
            .any(|s| matches!(s, PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_)));

        let captures_offset = if has_captures {
            // Generate full captures_fn with capture tracking
            self.emit_captures_fn(&steps)?
        } else {
            // No captures in pattern - fall back to interpreter for captures
            let offset = self.asm.offset();
            dynasm!(self.asm
                ; mov rax, -2i32
                ; ret
            );
            offset
        };

        // find_fn is fully JIT'd and doesn't use context - fast path enabled
        // Steps stored for fallback when JIT returns JIT_USE_INTERPRETER
        self.finalize(find_offset, captures_offset, false, Some(steps))
    }

    /// Emits range check code for character classes.
    /// Jumps to `fail_label` if the byte in `al` doesn't match any range.
    fn emit_range_check(
        &mut self,
        ranges: &[ByteRange],
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        if ranges.len() == 1 {
            let range = &ranges[0];
            let range_size = range.end.wrapping_sub(range.start);
            dynasm!(self.asm
                ; sub al, range.start as i8
                ; cmp al, range_size as i8
                ; ja =>fail_label
            );
        } else {
            let range_matched = self.asm.new_dynamic_label();

            for (ri, range) in ranges.iter().enumerate() {
                let is_last = ri == ranges.len() - 1;
                let range_size = range.end.wrapping_sub(range.start);

                if is_last {
                    dynasm!(self.asm
                        ; mov cl, al
                        ; sub cl, range.start as i8
                        ; cmp cl, range_size as i8
                        ; ja =>fail_label
                    );
                } else {
                    dynasm!(self.asm
                        ; mov cl, al
                        ; sub cl, range.start as i8
                        ; cmp cl, range_size as i8
                        ; jbe =>range_matched
                    );
                }
            }

            dynasm!(self.asm
                ; =>range_matched
            );
        }
        Ok(())
    }

    /// Emits code for a nested alternation inside an outer alternation.
    /// On success, continues to the next instruction.
    /// On failure of ALL alternatives, jumps to `fail_label`.
    ///
    /// This function generates code that saves position to the stack (with proper cleanup)
    /// for restoring between alternatives.
    fn emit_nested_alt(
        &mut self,
        alternatives: &[Vec<PatternStep>],
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        // For deeply nested alternations, we use the stack with explicit cleanup.
        // Each alternative that succeeds must pop the saved position.
        // The last alternative failing pops and jumps to fail_label.

        if alternatives.is_empty() {
            // Empty alternation - always fails
            dynasm!(self.asm
                ; jmp =>fail_label
            );
            return Ok(());
        }

        if alternatives.len() == 1 {
            // Single alternative - just emit it directly without saving
            for step in &alternatives[0] {
                self.emit_alt_step(step, fail_label)?;
            }
            return Ok(());
        }

        let alt_success = self.asm.new_dynamic_label();

        // Save current position on stack
        dynasm!(self.asm
            ; push r14                 // Save current position
        );

        for (alt_idx, alt_steps) in alternatives.iter().enumerate() {
            let is_last = alt_idx == alternatives.len() - 1;

            // Create label for trying next alternative (or cleanup for last)
            // Last alternative - if it fails, we need to clean up and fail
            let try_next_alt = self.asm.new_dynamic_label();

            // Emit code for each step in this alternative
            for alt_step in alt_steps.iter() {
                self.emit_alt_step(alt_step, try_next_alt)?;
            }

            // This alternative succeeded
            dynasm!(self.asm
                ; add rsp, 8           // Pop saved position
                ; jmp =>alt_success
            );

            // Define label for trying next alternative
            dynasm!(self.asm
                ; =>try_next_alt
            );

            if is_last {
                // Cleanup and jump to outer fail
                dynasm!(self.asm
                    ; add rsp, 8       // Pop saved position
                    ; jmp =>fail_label
                );
            } else {
                // Restore position and try next alternative
                dynasm!(self.asm
                    ; mov r14, [rsp]   // Restore position (keep on stack for next alt)
                );
            }
        }

        dynasm!(self.asm
            ; =>alt_success
        );

        Ok(())
    }

    /// Emits code for a single step inside an alternation.
    /// On failure, jumps to `fail_label`.
    fn emit_alt_step(
        &mut self,
        step: &PatternStep,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        match step {
            PatternStep::Byte(byte) => {
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                    ; cmp al, *byte as i8
                    ; jne =>fail_label
                    ; inc r14
                );
            }
            PatternStep::ByteClass(byte_class) => {
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14
                );
            }
            PatternStep::GreedyPlus(byte_class) => {
                let loop_start = self.asm.new_dynamic_label();
                let loop_done = self.asm.new_dynamic_label();

                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14
                    ; =>loop_start
                    ; cmp r14, r12
                    ; jge =>loop_done
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, loop_done)?;
                dynasm!(self.asm
                    ; inc r14
                    ; jmp =>loop_start
                    ; =>loop_done
                );
            }
            PatternStep::GreedyStar(byte_class) => {
                let loop_start = self.asm.new_dynamic_label();
                let loop_done = self.asm.new_dynamic_label();

                dynasm!(self.asm
                    ; =>loop_start
                    ; cmp r14, r12
                    ; jge =>loop_done
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, loop_done)?;
                dynasm!(self.asm
                    ; inc r14
                    ; jmp =>loop_start
                    ; =>loop_done
                );
            }
            PatternStep::Alt(inner_alternatives) => {
                // Nested alternation - recurse
                self.emit_nested_alt(inner_alternatives, fail_label)?;
            }
            PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_) => {
                // Capture markers - skip in find_fn
            }
            PatternStep::CodepointClass(cpclass, _target) => {
                self.emit_codepoint_class_check(cpclass, fail_label)?;
            }
            PatternStep::GreedyCodepointPlus(cpclass) => {
                self.emit_greedy_codepoint_plus(cpclass, fail_label)?;
            }
            PatternStep::WordBoundary => {
                self.emit_word_boundary_check(fail_label, true)?;
            }
            PatternStep::NotWordBoundary => {
                self.emit_word_boundary_check(fail_label, false)?;
            }
            PatternStep::StartOfText => {
                dynasm!(self.asm
                    ; test r14, r14
                    ; jnz =>fail_label
                );
            }
            PatternStep::EndOfText => {
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jne =>fail_label
                );
            }
            PatternStep::StartOfLine => {
                let at_start = self.asm.new_dynamic_label();
                dynasm!(self.asm
                    ; test r14, r14
                    ; jz =>at_start
                    ; mov rax, r14
                    ; dec rax
                    ; movzx eax, BYTE [rbx + rax]
                    ; cmp al, 0x0A
                    ; jne =>fail_label
                    ; =>at_start
                );
            }
            PatternStep::EndOfLine => {
                let at_end = self.asm.new_dynamic_label();
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; je =>at_end
                    ; movzx eax, BYTE [rbx + r14]
                    ; cmp al, 0x0A
                    ; jne =>fail_label
                    ; =>at_end
                );
            }
            PatternStep::PositiveLookahead(_)
            | PatternStep::NegativeLookahead(_)
            | PatternStep::PositiveLookbehind(_, _)
            | PatternStep::NegativeLookbehind(_, _) => {
                // Lookarounds in alternation - should have been filtered by has_unsupported_in_alt
                return Err(Error::new(
                    ErrorKind::Jit("Lookaround in alternation not yet supported".to_string()),
                    "",
                ));
            }
            PatternStep::Backref(_) => {
                // Backrefs in find_fn should have triggered early return
                return Err(Error::new(
                    ErrorKind::Jit(
                        "Backref in alternation not yet supported in find_fn".to_string(),
                    ),
                    "",
                ));
            }
            PatternStep::NonGreedyPlus(_, _) | PatternStep::NonGreedyStar(_, _) => {
                // Non-greedy in alternation - should have been filtered
                return Err(Error::new(
                    ErrorKind::Jit("Non-greedy in alternation not yet supported".to_string()),
                    "",
                ));
            }
            PatternStep::GreedyPlusLookahead(byte_class, lookahead_steps, is_positive) => {
                // Greedy one-or-more with lookahead in alternation
                self.emit_greedy_plus_with_lookahead_in_alt(
                    &byte_class.ranges,
                    lookahead_steps,
                    *is_positive,
                    fail_label,
                )?;
            }
            PatternStep::GreedyStarLookahead(byte_class, lookahead_steps, is_positive) => {
                // Greedy zero-or-more with lookahead in alternation
                self.emit_greedy_star_with_lookahead_in_alt(
                    &byte_class.ranges,
                    lookahead_steps,
                    *is_positive,
                    fail_label,
                )?;
            }
        }
        Ok(())
    }

    /// Emits code for greedy+ with lookahead inside an alternation.
    /// Similar to emit_greedy_plus_with_lookahead but adapted for alternation context.
    fn emit_greedy_plus_with_lookahead_in_alt(
        &mut self,
        ranges: &[ByteRange],
        lookahead_steps: &[PatternStep],
        is_positive: bool,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        let greedy_loop = self.asm.new_dynamic_label();
        let greedy_done = self.asm.new_dynamic_label();
        let try_lookahead = self.asm.new_dynamic_label();
        let lookahead_failed = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();

        // Must match at least one character (greedy plus)
        dynasm!(self.asm
            ; cmp r14, r12
            ; jge =>fail_label             // No input available
            ; movzx eax, BYTE [rbx + r14]
        );
        self.emit_range_check(ranges, fail_label)?;
        dynasm!(self.asm
            ; inc r14                      // Consumed first byte
            ; mov r9, r14                  // r9 = minimum position (need at least 1 match)

            // Greedy loop: consume as many as possible
            ; =>greedy_loop
            ; cmp r14, r12
            ; jge =>greedy_done            // End of input
            ; movzx eax, BYTE [rbx + r14]
        );
        self.emit_range_check(ranges, greedy_done)?;
        dynasm!(self.asm
            ; inc r14
            ; jmp =>greedy_loop

            ; =>greedy_done
            // r14 = max position after greedy consumption
            // Now backtrack until lookahead succeeds

            ; =>try_lookahead
        );

        // Emit lookahead check inline
        let lookahead_inner_match = self.asm.new_dynamic_label();
        let lookahead_inner_mismatch = self.asm.new_dynamic_label();

        dynasm!(self.asm
            ; mov r10, r14                 // Save position for restoration
        );

        // Try to match lookahead inner pattern
        for step in lookahead_steps {
            match step {
                PatternStep::Byte(byte) => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>lookahead_inner_mismatch
                        ; movzx eax, BYTE [rbx + r14]
                        ; cmp al, *byte as i8
                        ; jne =>lookahead_inner_mismatch
                        ; inc r14
                    );
                }
                PatternStep::ByteClass(inner_byte_class) => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>lookahead_inner_mismatch
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(&inner_byte_class.ranges, lookahead_inner_mismatch)?;
                    dynasm!(self.asm
                        ; inc r14
                    );
                }
                PatternStep::WordBoundary => {
                    self.emit_word_boundary_check(lookahead_inner_mismatch, true)?;
                }
                PatternStep::NotWordBoundary => {
                    self.emit_word_boundary_check(lookahead_inner_mismatch, false)?;
                }
                PatternStep::EndOfText => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jne =>lookahead_inner_mismatch
                    );
                }
                PatternStep::StartOfText => {
                    dynasm!(self.asm
                        ; test r14, r14
                        ; jnz =>lookahead_inner_mismatch
                    );
                }
                _ => {
                    // For other step types in lookahead, we need more complex handling
                    // For now, just continue (the lookahead check will likely fail)
                }
            }
        }

        // Lookahead inner pattern matched
        dynasm!(self.asm
            ; jmp =>lookahead_inner_match
            ; =>lookahead_inner_mismatch
        );

        if is_positive {
            // Positive lookahead failed - try backtracking
            dynasm!(self.asm
                ; mov r14, r10             // Restore position
                ; jmp =>lookahead_failed
                ; =>lookahead_inner_match
                ; mov r14, r10             // Restore position (zero-width)
                ; jmp =>success
            );
        } else {
            // Negative lookahead: inner mismatch means assertion succeeds
            // (we're at lookahead_inner_mismatch here, so inner pattern didn't match)
            dynasm!(self.asm
                ; mov r14, r10             // Restore position
                ; jmp =>success            // Inner didn't match = neg lookahead succeeds
                ; =>lookahead_inner_match
                ; mov r14, r10             // Restore position
                ; jmp =>lookahead_failed   // Inner matched = neg lookahead fails -> backtrack
            );
        }

        // Lookahead failed - backtrack one position
        dynasm!(self.asm
            ; =>lookahead_failed
            ; dec r14                      // Backtrack one position
            ; cmp r14, r9
            ; jl =>fail_label              // Below minimum - overall fail
            ; jmp =>try_lookahead          // Try lookahead at new position

            ; =>success
        );

        Ok(())
    }

    /// Emits code for greedy* with lookahead inside an alternation.
    fn emit_greedy_star_with_lookahead_in_alt(
        &mut self,
        ranges: &[ByteRange],
        lookahead_steps: &[PatternStep],
        is_positive: bool,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        let greedy_loop = self.asm.new_dynamic_label();
        let greedy_done = self.asm.new_dynamic_label();
        let try_lookahead = self.asm.new_dynamic_label();
        let lookahead_failed = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();

        // Star can match zero - save current position as minimum
        dynasm!(self.asm
            ; mov r9, r14                  // r9 = minimum position (can be 0 matches)

            // Greedy loop: consume as many as possible
            ; =>greedy_loop
            ; cmp r14, r12
            ; jge =>greedy_done            // End of input
            ; movzx eax, BYTE [rbx + r14]
        );
        self.emit_range_check(ranges, greedy_done)?;
        dynasm!(self.asm
            ; inc r14
            ; jmp =>greedy_loop

            ; =>greedy_done
            // r14 = max position after greedy consumption
            // Now backtrack until lookahead succeeds

            ; =>try_lookahead
        );

        // Emit lookahead check inline (same as plus version)
        let lookahead_inner_match = self.asm.new_dynamic_label();
        let lookahead_inner_mismatch = self.asm.new_dynamic_label();

        dynasm!(self.asm
            ; mov r10, r14                 // Save position for restoration
        );

        for step in lookahead_steps {
            match step {
                PatternStep::Byte(byte) => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>lookahead_inner_mismatch
                        ; movzx eax, BYTE [rbx + r14]
                        ; cmp al, *byte as i8
                        ; jne =>lookahead_inner_mismatch
                        ; inc r14
                    );
                }
                PatternStep::ByteClass(inner_byte_class) => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>lookahead_inner_mismatch
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(&inner_byte_class.ranges, lookahead_inner_mismatch)?;
                    dynasm!(self.asm
                        ; inc r14
                    );
                }
                PatternStep::WordBoundary => {
                    self.emit_word_boundary_check(lookahead_inner_mismatch, true)?;
                }
                PatternStep::NotWordBoundary => {
                    self.emit_word_boundary_check(lookahead_inner_mismatch, false)?;
                }
                PatternStep::EndOfText => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jne =>lookahead_inner_mismatch
                    );
                }
                PatternStep::StartOfText => {
                    dynasm!(self.asm
                        ; test r14, r14
                        ; jnz =>lookahead_inner_mismatch
                    );
                }
                _ => {}
            }
        }

        dynasm!(self.asm
            ; jmp =>lookahead_inner_match
            ; =>lookahead_inner_mismatch
        );

        if is_positive {
            // Positive lookahead: inner mismatch means lookahead failed
            dynasm!(self.asm
                ; mov r14, r10
                ; jmp =>lookahead_failed
                ; =>lookahead_inner_match
                ; mov r14, r10
                ; jmp =>success
            );
        } else {
            // Negative lookahead: inner match means lookahead assertion fails
            dynasm!(self.asm
                ; mov r14, r10
                ; jmp =>success             // Inner mismatch = negative lookahead succeeds
                ; =>lookahead_inner_match
                ; mov r14, r10
                ; jmp =>lookahead_failed    // Inner match = negative lookahead fails
            );
        }

        dynasm!(self.asm
            ; =>lookahead_failed
            ; cmp r14, r9
            ; jle =>fail_label             // At or below minimum - fail
            ; dec r14                      // Backtrack one position
            ; jmp =>try_lookahead

            ; =>success
        );

        Ok(())
    }

    /// Emits code to check if the suffix matches at current position.
    /// If suffix doesn't match, jumps to `fail_label`.
    /// If suffix matches, continues to `success_label`.
    /// The suffix is consumed (r14 is advanced).
    fn emit_non_greedy_suffix_check(
        &mut self,
        suffix: &PatternStep,
        fail_label: dynasmrt::DynamicLabel,
        _success_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        match suffix {
            PatternStep::Byte(byte) => {
                // Check bounds and byte value
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label           // Not enough input
                    ; movzx eax, BYTE [rbx + r14]
                    ; cmp al, *byte as i8
                    ; jne =>fail_label           // Byte doesn't match
                    ; inc r14                    // Consume the suffix byte
                );
            }
            PatternStep::ByteClass(byte_class) => {
                // Check bounds
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label           // Not enough input
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14                    // Consume the suffix byte
                );
            }
            _ => {
                // Other suffix types not supported - shouldn't reach here
                // since extract_single_step only returns Byte or ByteClass
                return Err(Error::new(
                    ErrorKind::Jit("Unsupported non-greedy suffix type".to_string()),
                    "",
                ));
            }
        }

        Ok(())
    }

    /// Emits code to check if a byte is a word character.
    /// Sets ZF (zero flag) if it IS a word char, clears ZF if not.
    /// Uses al as input, clobbers cl.
    fn emit_is_word_char(
        &mut self,
        word_char_label: dynasmrt::DynamicLabel,
        not_word_char_label: dynasmrt::DynamicLabel,
    ) {
        use dynasmrt::DynasmLabelApi;
        // Word characters: [a-zA-Z0-9_]
        // Check ranges: a-z (0x61-0x7a), A-Z (0x41-0x5a), 0-9 (0x30-0x39), _ (0x5f)
        dynasm!(self.asm
            // Check 'a'-'z'
            ; mov cl, al
            ; sub cl, 0x61u8 as i8   // 'a'
            ; cmp cl, 25u8 as i8     // 'z' - 'a' = 25
            ; jbe =>word_char_label

            // Check 'A'-'Z'
            ; mov cl, al
            ; sub cl, 0x41u8 as i8   // 'A'
            ; cmp cl, 25u8 as i8     // 'Z' - 'A' = 25
            ; jbe =>word_char_label

            // Check '0'-'9'
            ; mov cl, al
            ; sub cl, 0x30u8 as i8   // '0'
            ; cmp cl, 9u8 as i8      // '9' - '0' = 9
            ; jbe =>word_char_label

            // Check '_'
            ; cmp al, 0x5fu8 as i8   // '_'
            ; je =>word_char_label

            // Not a word char
            ; jmp =>not_word_char_label
        );
    }

    /// Emits code to check word boundary.
    /// rbx = input_ptr, r14 = current_pos, r12 = input_len
    /// Jumps to fail_label if the word boundary condition is not met.
    fn emit_word_boundary_check(
        &mut self,
        fail_label: dynasmrt::DynamicLabel,
        is_boundary: bool,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        // Word boundary is at position where:
        // - prev_is_word XOR curr_is_word is true
        //
        // At start (pos=0): prev is considered non-word
        // At end (pos=len): curr is considered non-word
        //
        // For \b (is_boundary=true): need XOR to be true (one word, one non-word)
        // For \B (is_boundary=false): need XOR to be false (both word or both non-word)

        let prev_word = self.asm.new_dynamic_label();
        let prev_not_word = self.asm.new_dynamic_label();
        let curr_word = self.asm.new_dynamic_label();
        let curr_not_word = self.asm.new_dynamic_label();
        let check_curr = self.asm.new_dynamic_label();
        let boundary_match = self.asm.new_dynamic_label();
        let _boundary_no_match = self.asm.new_dynamic_label();

        // Check previous character (at pos-1)
        dynasm!(self.asm
            ; cmp r14, 0
            ; je =>prev_not_word           // At start, prev is non-word
            ; movzx eax, BYTE [rbx + r14 - 1]
        );
        self.emit_is_word_char(prev_word, prev_not_word);

        // prev_is_word = true (stored in r8b: 1 = word, 0 = non-word)
        dynasm!(self.asm
            ; =>prev_word
            ; mov r8b, 1
            ; jmp =>check_curr
        );

        // prev_is_word = false
        dynasm!(self.asm
            ; =>prev_not_word
            ; mov r8b, 0
        );

        // Check current character (at pos)
        dynasm!(self.asm
            ; =>check_curr
            ; cmp r14, r12
            ; jge =>curr_not_word          // At end, curr is non-word
            ; movzx eax, BYTE [rbx + r14]
        );
        self.emit_is_word_char(curr_word, curr_not_word);

        // curr_is_word = true (stored in r9b)
        dynasm!(self.asm
            ; =>curr_word
            ; mov r9b, 1
            ; jmp =>boundary_match
        );

        // curr_is_word = false
        dynasm!(self.asm
            ; =>curr_not_word
            ; xor r9d, r9d                 // r9b = 0
        );

        // Check XOR of r8b and r9b
        dynasm!(self.asm
            ; =>boundary_match
            ; xor r8b, r9b                 // r8b = prev_word XOR curr_word
        );

        if is_boundary {
            // \b: need XOR to be 1 (boundary exists)
            dynasm!(self.asm
                ; test r8b, r8b
                ; jz =>fail_label          // XOR is 0 means no boundary
            );
        } else {
            // \B: need XOR to be 0 (no boundary)
            dynasm!(self.asm
                ; test r8b, r8b
                ; jnz =>fail_label         // XOR is 1 means there is a boundary
            );
        }

        Ok(())
    }

    /// Emits greedy one-or-more with lookahead: greedily consumes, then backtracks.
    ///
    /// Algorithm:
    /// 1. Save minimum position (start + 1, since + requires at least one match)
    /// 2. Greedily consume all matching bytes
    /// 3. Save maximum position
    /// 4. Try lookahead at current position
    /// 5. If lookahead fails, decrement position and retry
    /// 6. If position < minimum, overall match fails
    ///
    /// Register usage:
    /// - r14 = current_pos (modified during backtracking)
    /// - r9 = minimum valid position (start + 1)
    /// - r10 = saved for backtracking loop
    fn emit_greedy_plus_with_lookahead(
        &mut self,
        ranges: &[ByteRange],
        lookahead_steps: &[PatternStep],
        is_positive: bool,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        // Optimize for `.*X` patterns in lookahead - use O(n) scan instead of O(n^2) backtracking
        // For `\w+(?=.*\d)`, we first match `\w+`, then scan for ANY `\d` in the remaining text
        if lookahead_steps.len() == 2 {
            if let PatternStep::GreedyStar(star_byte_class) = &lookahead_steps[0] {
                match &lookahead_steps[1] {
                    PatternStep::ByteClass(final_byte_class) => {
                        return self.emit_greedy_plus_with_star_scan_lookahead(
                            ranges,
                            &star_byte_class.ranges,
                            &final_byte_class.ranges,
                            is_positive,
                            fail_label,
                        );
                    }
                    PatternStep::Byte(byte) => {
                        let final_ranges = vec![ByteRange {
                            start: *byte,
                            end: *byte,
                        }];
                        return self.emit_greedy_plus_with_star_scan_lookahead(
                            ranges,
                            &star_byte_class.ranges,
                            &final_ranges,
                            is_positive,
                            fail_label,
                        );
                    }
                    _ => {}
                }
            }
        }

        let greedy_loop = self.asm.new_dynamic_label();
        let greedy_done = self.asm.new_dynamic_label();
        let try_lookahead = self.asm.new_dynamic_label();
        let lookahead_failed = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();

        // Must match at least one character (greedy plus)
        dynasm!(self.asm
            ; cmp r14, r12
            ; jge =>fail_label             // No input available
            ; movzx eax, BYTE [rbx + r14]
        );
        self.emit_range_check(ranges, fail_label)?;
        dynasm!(self.asm
            ; inc r14                      // Consumed first byte
            ; mov r9, r14                  // r9 = minimum position (need at least 1 match)

            // Greedy loop: consume as many as possible
            ; =>greedy_loop
            ; cmp r14, r12
            ; jge =>greedy_done            // End of input
            ; movzx eax, BYTE [rbx + r14]
        );
        self.emit_range_check(ranges, greedy_done)?;
        dynasm!(self.asm
            ; inc r14
            ; jmp =>greedy_loop

            ; =>greedy_done
            // r14 = max position after greedy consumption
            // Now backtrack until lookahead succeeds

            ; =>try_lookahead
        );

        // Emit lookahead check inline
        // Save position for lookahead (it's zero-width)
        let lookahead_inner_match = self.asm.new_dynamic_label();
        let lookahead_inner_mismatch = self.asm.new_dynamic_label();

        dynasm!(self.asm
            ; mov r10, r14                 // Save position for restoration
        );

        // Try to match lookahead inner pattern
        for step in lookahead_steps {
            match step {
                PatternStep::Byte(byte) => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>lookahead_inner_mismatch
                        ; movzx eax, BYTE [rbx + r14]
                        ; cmp al, *byte as i8
                        ; jne =>lookahead_inner_mismatch
                        ; inc r14
                    );
                }
                PatternStep::ByteClass(inner_byte_class) => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>lookahead_inner_mismatch
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(&inner_byte_class.ranges, lookahead_inner_mismatch)?;
                    dynasm!(self.asm
                        ; inc r14
                    );
                }
                PatternStep::WordBoundary => {
                    // Word boundary in lookahead
                    self.emit_word_boundary_check(lookahead_inner_mismatch, true)?;
                }
                PatternStep::NotWordBoundary => {
                    self.emit_word_boundary_check(lookahead_inner_mismatch, false)?;
                }
                PatternStep::EndOfText => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jne =>lookahead_inner_mismatch
                    );
                }
                _ => {
                    // Unsupported step in lookahead - fall back to interpreter
                    return Err(Error::new(
                        ErrorKind::Jit("Unsupported pattern step in greedy+lookahead".to_string()),
                        "",
                    ));
                }
            }
        }

        // Lookahead inner pattern matched
        dynasm!(self.asm
            ; jmp =>lookahead_inner_match
            ; =>lookahead_inner_mismatch
        );

        if is_positive {
            // Positive lookahead failed - try backtracking
            dynasm!(self.asm
                ; mov r14, r10             // Restore position
                ; jmp =>lookahead_failed
                ; =>lookahead_inner_match
                ; mov r14, r10             // Restore position (zero-width)
                ; jmp =>success
            );
        } else {
            // Negative lookahead: inner mismatch means assertion succeeds
            // (we're at lookahead_inner_mismatch here, so inner pattern didn't match)
            dynasm!(self.asm
                ; mov r14, r10             // Restore position
                ; jmp =>success            // Inner didn't match = neg lookahead succeeds
                ; =>lookahead_inner_match
                ; mov r14, r10             // Restore position
                ; jmp =>lookahead_failed   // Inner matched = neg lookahead fails -> backtrack
            );
        }

        // Lookahead failed - backtrack one position
        dynasm!(self.asm
            ; =>lookahead_failed
            ; dec r14                      // Backtrack one position
            ; cmp r14, r9
            ; jl =>fail_label              // Below minimum - overall fail
            ; jmp =>try_lookahead          // Try lookahead at new position

            ; =>success
        );

        Ok(())
    }

    /// Emits optimized greedy+ with `.*X` lookahead pattern.
    ///
    /// For `\w+(?=.*\d)`:
    /// 1. Match at least one word char (greedy+)
    /// 2. Scan from current position for ANY digit (instead of backtracking)
    ///
    /// This is O(n) instead of O(n^2) because we don't need to backtrack the
    /// greedy+ - we just need to verify that a digit exists somewhere ahead.
    fn emit_greedy_plus_with_star_scan_lookahead(
        &mut self,
        ranges: &[ByteRange],
        star_ranges: &[ByteRange],
        final_ranges: &[ByteRange],
        is_positive: bool,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        let greedy_loop = self.asm.new_dynamic_label();
        let greedy_done = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();
        let star_loop = self.asm.new_dynamic_label();
        let star_done = self.asm.new_dynamic_label();
        let scan_loop = self.asm.new_dynamic_label();
        let scan_done = self.asm.new_dynamic_label();
        let found_match = self.asm.new_dynamic_label();

        // Must match at least one character (greedy plus)
        dynasm!(self.asm
            ; cmp r14, r12
            ; jge =>fail_label             // No input available
            ; movzx eax, BYTE [rbx + r14]
        );
        self.emit_range_check(ranges, fail_label)?;
        dynasm!(self.asm
            ; inc r14                      // Consumed first byte

            // Greedy loop: consume as many as possible
            ; =>greedy_loop
            ; cmp r14, r12
            ; jge =>greedy_done            // End of input
            ; movzx eax, BYTE [rbx + r14]
        );
        self.emit_range_check(ranges, greedy_done)?;
        dynasm!(self.asm
            ; inc r14
            ; jmp =>greedy_loop

            ; =>greedy_done
            // r14 = position after greedy consumption
            // Now check the lookahead using O(n) scan
        );

        // Step 1: Find star_end - the extent of where `.*` can match to
        // r9 starts at r14 (position after greedy+), then we scan forward while star_ranges match
        dynasm!(self.asm
            ; mov r9, r14                  // r9 = star_end, starts at current pos

            ; =>star_loop
            ; cmp r9, r12
            ; jge =>star_done              // End of input
            ; movzx eax, BYTE [rbx + r9]
        );

        // Check if byte matches star_ranges (e.g., for `.*`, this excludes newline)
        self.emit_range_check(star_ranges, star_done)?;

        dynasm!(self.asm
            ; inc r9                       // Matched, advance star_end
            ; jmp =>star_loop

            ; =>star_done
            // r9 = star_end (exclusive - position past all star matches)
        );

        // Step 2: Scan from r14 to r9 looking for ANY match of final_ranges
        // r10 = scan position
        dynasm!(self.asm
            ; mov r10, r14                 // r10 = scan position, starts at current pos

            ; =>scan_loop
            ; cmp r10, r9
            ; jg =>scan_done               // Scanned past star_end
            ; cmp r10, r12
            ; jge =>scan_done              // End of input

            ; movzx eax, BYTE [rbx + r10]
        );

        // Check if byte matches final_ranges (e.g., `\d`)
        let check_next = self.asm.new_dynamic_label();
        self.emit_range_check(final_ranges, check_next)?;

        // If we reach here, final_ranges matched!
        dynasm!(self.asm
            ; jmp =>found_match

            ; =>check_next
            ; inc r10                      // Not a match, try next position
            ; jmp =>scan_loop

            ; =>scan_done
            // Scanned entire range without finding a match
        );

        // Determine success/failure based on positive/negative lookahead
        if is_positive {
            // Positive lookahead: we need to find a match
            dynasm!(self.asm
                ; jmp =>fail_label         // No match found -> fail

                ; =>found_match
                // Match found, lookahead succeeds
                // r14 is already at the correct position after greedy+
                ; jmp =>success
            );
        } else {
            // Negative lookahead: we need to NOT find a match
            dynasm!(self.asm
                ; jmp =>success            // No match found -> success

                ; =>found_match
                ; jmp =>fail_label         // Match found -> fail
            );
        }

        dynasm!(self.asm
            ; =>success
            // r14 unchanged from greedy+ position (lookahead is zero-width)
        );

        Ok(())
    }

    /// Emits greedy zero-or-more with lookahead: greedily consumes, then backtracks.
    /// Similar to plus version but minimum position is start (0 matches allowed).
    fn emit_greedy_star_with_lookahead(
        &mut self,
        ranges: &[ByteRange],
        lookahead_steps: &[PatternStep],
        is_positive: bool,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        // Optimize for `.*X` patterns in lookahead - use O(n) scan instead of O(n^2) backtracking
        if lookahead_steps.len() == 2 {
            if let PatternStep::GreedyStar(star_byte_class) = &lookahead_steps[0] {
                match &lookahead_steps[1] {
                    PatternStep::ByteClass(final_byte_class) => {
                        return self.emit_greedy_star_with_star_scan_lookahead(
                            ranges,
                            &star_byte_class.ranges,
                            &final_byte_class.ranges,
                            is_positive,
                            fail_label,
                        );
                    }
                    PatternStep::Byte(byte) => {
                        let final_ranges = vec![ByteRange {
                            start: *byte,
                            end: *byte,
                        }];
                        return self.emit_greedy_star_with_star_scan_lookahead(
                            ranges,
                            &star_byte_class.ranges,
                            &final_ranges,
                            is_positive,
                            fail_label,
                        );
                    }
                    _ => {}
                }
            }
        }

        let greedy_loop = self.asm.new_dynamic_label();
        let greedy_done = self.asm.new_dynamic_label();
        let try_lookahead = self.asm.new_dynamic_label();
        let lookahead_failed = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();

        // Star can match zero - save current position as minimum
        dynasm!(self.asm
            ; mov r9, r14                  // r9 = minimum position (can be 0 matches)

            // Greedy loop: consume as many as possible
            ; =>greedy_loop
            ; cmp r14, r12
            ; jge =>greedy_done            // End of input
            ; movzx eax, BYTE [rbx + r14]
        );
        self.emit_range_check(ranges, greedy_done)?;
        dynasm!(self.asm
            ; inc r14
            ; jmp =>greedy_loop

            ; =>greedy_done
            // r14 = max position after greedy consumption
            // Now backtrack until lookahead succeeds

            ; =>try_lookahead
        );

        // Emit lookahead check inline (same as plus version)
        let lookahead_inner_match = self.asm.new_dynamic_label();
        let lookahead_inner_mismatch = self.asm.new_dynamic_label();

        dynasm!(self.asm
            ; mov r10, r14                 // Save position for restoration
        );

        for step in lookahead_steps {
            match step {
                PatternStep::Byte(byte) => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>lookahead_inner_mismatch
                        ; movzx eax, BYTE [rbx + r14]
                        ; cmp al, *byte as i8
                        ; jne =>lookahead_inner_mismatch
                        ; inc r14
                    );
                }
                PatternStep::ByteClass(inner_byte_class) => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>lookahead_inner_mismatch
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(&inner_byte_class.ranges, lookahead_inner_mismatch)?;
                    dynasm!(self.asm
                        ; inc r14
                    );
                }
                PatternStep::WordBoundary => {
                    self.emit_word_boundary_check(lookahead_inner_mismatch, true)?;
                }
                PatternStep::NotWordBoundary => {
                    self.emit_word_boundary_check(lookahead_inner_mismatch, false)?;
                }
                PatternStep::EndOfText => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jne =>lookahead_inner_mismatch
                    );
                }
                _ => {
                    return Err(Error::new(
                        ErrorKind::Jit("Unsupported pattern step in greedy*lookahead".to_string()),
                        "",
                    ));
                }
            }
        }

        dynasm!(self.asm
            ; jmp =>lookahead_inner_match
            ; =>lookahead_inner_mismatch
        );

        if is_positive {
            // Positive lookahead: inner mismatch means assertion fails
            dynasm!(self.asm
                ; mov r14, r10
                ; jmp =>lookahead_failed
                ; =>lookahead_inner_match
                ; mov r14, r10
                ; jmp =>success
            );
        } else {
            // Negative lookahead: inner mismatch means assertion succeeds
            dynasm!(self.asm
                ; mov r14, r10
                ; jmp =>success            // Inner didn't match = neg lookahead succeeds
                ; =>lookahead_inner_match
                ; mov r14, r10
                ; jmp =>lookahead_failed   // Inner matched = neg lookahead fails -> backtrack
            );
        }

        dynasm!(self.asm
            ; =>lookahead_failed
            ; cmp r14, r9
            ; jle =>fail_label             // At or below minimum - fail
            ; dec r14                      // Backtrack one position
            ; jmp =>try_lookahead

            ; =>success
        );

        Ok(())
    }

    /// Emits optimized greedy* with `.*X` lookahead pattern.
    ///
    /// For `\w*(?=.*\d)`:
    /// 1. Match zero or more word chars (greedy*)
    /// 2. Scan from current position for ANY digit (instead of backtracking)
    fn emit_greedy_star_with_star_scan_lookahead(
        &mut self,
        ranges: &[ByteRange],
        star_ranges: &[ByteRange],
        final_ranges: &[ByteRange],
        is_positive: bool,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        let greedy_loop = self.asm.new_dynamic_label();
        let greedy_done = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();
        let star_loop = self.asm.new_dynamic_label();
        let star_done = self.asm.new_dynamic_label();
        let scan_loop = self.asm.new_dynamic_label();
        let scan_done = self.asm.new_dynamic_label();
        let found_match = self.asm.new_dynamic_label();

        // Greedy star: match zero or more characters
        dynasm!(self.asm
            ; =>greedy_loop
            ; cmp r14, r12
            ; jge =>greedy_done            // End of input
            ; movzx eax, BYTE [rbx + r14]
        );
        self.emit_range_check(ranges, greedy_done)?;
        dynasm!(self.asm
            ; inc r14
            ; jmp =>greedy_loop

            ; =>greedy_done
            // r14 = position after greedy consumption
            // Now check the lookahead using O(n) scan
        );

        // Step 1: Find star_end - the extent of where `.*` can match to
        dynasm!(self.asm
            ; mov r9, r14                  // r9 = star_end, starts at current pos

            ; =>star_loop
            ; cmp r9, r12
            ; jge =>star_done              // End of input
            ; movzx eax, BYTE [rbx + r9]
        );

        self.emit_range_check(star_ranges, star_done)?;

        dynasm!(self.asm
            ; inc r9
            ; jmp =>star_loop

            ; =>star_done
            // r9 = star_end
        );

        // Step 2: Scan from r14 to r9 looking for ANY match of final_ranges
        dynasm!(self.asm
            ; mov r10, r14                 // r10 = scan position

            ; =>scan_loop
            ; cmp r10, r9
            ; jg =>scan_done
            ; cmp r10, r12
            ; jge =>scan_done

            ; movzx eax, BYTE [rbx + r10]
        );

        let check_next = self.asm.new_dynamic_label();
        self.emit_range_check(final_ranges, check_next)?;

        dynasm!(self.asm
            ; jmp =>found_match

            ; =>check_next
            ; inc r10
            ; jmp =>scan_loop

            ; =>scan_done
        );

        if is_positive {
            dynasm!(self.asm
                ; jmp =>fail_label

                ; =>found_match
                ; jmp =>success
            );
        } else {
            dynasm!(self.asm
                ; jmp =>success

                ; =>found_match
                ; jmp =>fail_label
            );
        }

        dynasm!(self.asm
            ; =>success
        );

        Ok(())
    }

    /// Emits greedy+ with backtracking for remaining steps.
    ///
    /// Algorithm:
    /// 1. Match at least one character (required for +)
    /// 2. Greedily match as many as possible
    /// 3. Try to match remaining steps
    /// 4. If remaining steps fail, backtrack (give up one character) and retry
    /// 5. If position drops below minimum (start+1), overall match fails
    fn emit_greedy_plus_with_backtracking(
        &mut self,
        ranges: &[ByteRange],
        remaining_steps: &[PatternStep],
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        let greedy_loop = self.asm.new_dynamic_label();
        let greedy_done = self.asm.new_dynamic_label();
        let try_remaining = self.asm.new_dynamic_label();
        let backtrack = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();

        // Must match at least one character
        dynasm!(self.asm
            ; cmp r14, r12
            ; jge =>fail_label             // No input available
            ; movzx eax, BYTE [rbx + r14]
        );
        self.emit_range_check(ranges, fail_label)?;
        dynasm!(self.asm
            ; inc r14                      // Consumed first byte
            ; mov r9, r14                  // r9 = minimum position (need at least 1 match)

            // Greedy loop: consume as many as possible
            ; =>greedy_loop
            ; cmp r14, r12
            ; jge =>greedy_done            // End of input
            ; movzx eax, BYTE [rbx + r14]
        );
        self.emit_range_check(ranges, greedy_done)?;
        dynasm!(self.asm
            ; inc r14
            ; jmp =>greedy_loop

            ; =>greedy_done
            // r14 = max position after greedy consumption
            // Now try remaining steps, backtracking on failure

            ; =>try_remaining
        );

        // Emit code for remaining steps
        // Generate code that jumps to backtrack on failure
        for step in remaining_steps {
            self.emit_step_inline(step, backtrack)?;
        }

        // All remaining steps matched - success
        dynasm!(self.asm
            ; jmp =>success

            ; =>backtrack
            // Remaining steps failed - backtrack one position
            ; dec r14
            ; cmp r14, r9
            ; jl =>fail_label              // Below minimum - overall fail
            ; jmp =>try_remaining          // Try again with one less character

            ; =>success
        );

        Ok(())
    }

    /// Emits greedy* with backtracking for remaining steps.
    ///
    /// Similar to plus version but minimum position is start (0 matches allowed).
    fn emit_greedy_star_with_backtracking(
        &mut self,
        ranges: &[ByteRange],
        remaining_steps: &[PatternStep],
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        let greedy_loop = self.asm.new_dynamic_label();
        let greedy_done = self.asm.new_dynamic_label();
        let try_remaining = self.asm.new_dynamic_label();
        let backtrack = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();

        dynasm!(self.asm
            ; mov r9, r14                  // r9 = minimum position (can match 0)

            // Greedy loop: consume as many as possible
            ; =>greedy_loop
            ; cmp r14, r12
            ; jge =>greedy_done            // End of input
            ; movzx eax, BYTE [rbx + r14]
        );
        self.emit_range_check(ranges, greedy_done)?;
        dynasm!(self.asm
            ; inc r14
            ; jmp =>greedy_loop

            ; =>greedy_done
            // Try remaining steps, backtracking on failure

            ; =>try_remaining
        );

        // Emit code for remaining steps
        for step in remaining_steps {
            self.emit_step_inline(step, backtrack)?;
        }

        // All remaining steps matched - success
        dynasm!(self.asm
            ; jmp =>success

            ; =>backtrack
            // Remaining steps failed - backtrack one position
            ; cmp r14, r9
            ; jle =>fail_label             // At or below minimum - overall fail
            ; dec r14
            ; jmp =>try_remaining

            ; =>success
        );

        Ok(())
    }

    /// Emits greedy codepoint+ with backtracking for remaining steps.
    ///
    /// For UTF-8 patterns, we need to track character boundaries since we can't
    /// simply decrement position (multi-byte characters).
    ///
    /// Algorithm:
    /// 1. Match at least one codepoint (required for +)
    /// 2. Greedily match as many codepoints as possible, saving boundaries
    /// 3. Try to match remaining steps
    /// 4. If fail, restore to previous boundary and retry
    fn emit_greedy_codepoint_plus_with_backtracking(
        &mut self,
        cpclass: &CodepointClass,
        remaining_steps: &[PatternStep],
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        // For codepoint backtracking, we need to save character boundaries
        // We'll use the stack to save positions
        let loop_start = self.asm.new_dynamic_label();
        let loop_done = self.asm.new_dynamic_label();
        let try_remaining = self.asm.new_dynamic_label();
        let backtrack = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();
        let first_fail_with_stack = self.asm.new_dynamic_label();
        let loop_fail_no_stack = self.asm.new_dynamic_label();
        let loop_fail_with_stack = self.asm.new_dynamic_label();
        let no_more_boundaries = self.asm.new_dynamic_label();

        // r10 will track the number of saved boundaries on stack
        dynasm!(self.asm
            ; xor r10d, r10d               // r10 = boundary count = 0
        );

        // First iteration: must match at least one codepoint
        self.emit_utf8_decode(fail_label)?;
        dynasm!(self.asm
            ; push rcx                     // Save byte length
        );
        self.emit_codepoint_class_membership_check(cpclass, first_fail_with_stack)?;
        dynasm!(self.asm
            ; pop rcx
            ; add r14, rcx                 // Advance position
            ; push r14                     // Save boundary position
            ; inc r10                      // boundary count++

            // Greedy loop: match more codepoints
            ; =>loop_start
        );

        self.emit_utf8_decode(loop_fail_no_stack)?;
        dynasm!(self.asm
            ; push rcx                     // Save byte length
        );
        self.emit_codepoint_class_membership_check(cpclass, loop_fail_with_stack)?;
        dynasm!(self.asm
            ; pop rcx
            ; add r14, rcx                 // Advance position
            ; push r14                     // Save boundary position
            ; inc r10                      // boundary count++
            ; jmp =>loop_start

            ; =>first_fail_with_stack
            ; add rsp, 8                   // Pop saved rcx
            ; jmp =>fail_label             // First match failed - overall fail

            ; =>loop_fail_no_stack
            ; jmp =>loop_done

            ; =>loop_fail_with_stack
            ; add rsp, 8                   // Pop saved rcx
            ; jmp =>loop_done

            ; =>loop_done
            // Greedy matching done
            // Stack has boundary positions, r10 = count
            // Try remaining steps with backtracking

            ; =>try_remaining
        );

        // Emit code for remaining steps
        for step in remaining_steps {
            self.emit_step_inline(step, backtrack)?;
        }

        // All remaining steps matched - success!
        // Clean up stack (pop all saved boundaries)
        dynasm!(self.asm
            ; =>success
            ; lea rsp, [rsp + r10 * 8]     // Pop all boundary positions
            ; jmp >done

            ; =>backtrack
            // Remaining steps failed - backtrack to previous boundary
            ; cmp r10, 1
            ; jle =>no_more_boundaries     // Need at least 1 match (plus semantics)

            ; pop r14                      // Restore previous boundary position (discard current)
            ; dec r10
            ; pop r14                      // Get actual position to retry from
            ; dec r10
            ; push r14                     // Put it back for next backtrack
            ; inc r10
            ; jmp =>try_remaining

            ; =>no_more_boundaries
            // Can't backtrack more - clean up and fail
            ; lea rsp, [rsp + r10 * 8]     // Pop all remaining boundaries
            ; jmp =>fail_label

            ; done:
        );

        Ok(())
    }

    /// Emits code for a single pattern step inline.
    /// Used when generating code for remaining steps in backtracking.
    fn emit_step_inline(
        &mut self,
        step: &PatternStep,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        match step {
            PatternStep::Byte(byte) => {
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                    ; cmp al, *byte as i8
                    ; jne =>fail_label
                    ; inc r14
                );
            }
            PatternStep::ByteClass(byte_class) => {
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14
                );
            }
            PatternStep::GreedyPlus(byte_class) => {
                // Nested greedy+ - emit simple version (no recursive backtracking for simplicity)
                let loop_start = self.asm.new_dynamic_label();
                let loop_done = self.asm.new_dynamic_label();

                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14
                    ; =>loop_start
                    ; cmp r14, r12
                    ; jge =>loop_done
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, loop_done)?;
                dynasm!(self.asm
                    ; inc r14
                    ; jmp =>loop_start
                    ; =>loop_done
                );
            }
            PatternStep::GreedyStar(byte_class) => {
                let loop_start = self.asm.new_dynamic_label();
                let loop_done = self.asm.new_dynamic_label();

                dynasm!(self.asm
                    ; =>loop_start
                    ; cmp r14, r12
                    ; jge =>loop_done
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, loop_done)?;
                dynasm!(self.asm
                    ; inc r14
                    ; jmp =>loop_start
                    ; =>loop_done
                );
            }
            PatternStep::CodepointClass(cpclass, _target) => {
                self.emit_codepoint_class_check(cpclass, fail_label)?;
            }
            PatternStep::GreedyCodepointPlus(cpclass) => {
                // Simple greedy codepoint+ without backtracking
                self.emit_greedy_codepoint_plus(cpclass, fail_label)?;
            }
            PatternStep::WordBoundary => {
                self.emit_word_boundary_check(fail_label, true)?;
            }
            PatternStep::NotWordBoundary => {
                self.emit_word_boundary_check(fail_label, false)?;
            }
            PatternStep::StartOfText => {
                dynasm!(self.asm
                    ; test r14, r14
                    ; jnz =>fail_label
                );
            }
            PatternStep::EndOfText => {
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jne =>fail_label
                );
            }
            PatternStep::StartOfLine => {
                let at_start = self.asm.new_dynamic_label();
                dynasm!(self.asm
                    ; test r14, r14
                    ; jz =>at_start
                    ; mov rax, r14
                    ; dec rax
                    ; movzx eax, BYTE [rbx + rax]
                    ; cmp al, 0x0A
                    ; jne =>fail_label
                    ; =>at_start
                );
            }
            PatternStep::EndOfLine => {
                let at_end = self.asm.new_dynamic_label();
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; je =>at_end
                    ; movzx eax, BYTE [rbx + r14]
                    ; cmp al, 0x0A
                    ; jne =>fail_label
                    ; =>at_end
                );
            }
            PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_) => {
                // Capture markers don't consume input - skip
            }
            PatternStep::PositiveLookahead(inner_steps) => {
                self.emit_standalone_lookahead(inner_steps, fail_label, true)?;
            }
            PatternStep::NegativeLookahead(inner_steps) => {
                self.emit_standalone_lookahead(inner_steps, fail_label, false)?;
            }
            PatternStep::PositiveLookbehind(inner_steps, min_len) => {
                self.emit_lookbehind_check(inner_steps, *min_len, fail_label, true)?;
            }
            PatternStep::NegativeLookbehind(inner_steps, min_len) => {
                self.emit_lookbehind_check(inner_steps, *min_len, fail_label, false)?;
            }
            PatternStep::Alt(alternatives) => {
                // Simple alternation - try each and jump to success if one matches
                let alt_success = self.asm.new_dynamic_label();
                for (i, alt_steps) in alternatives.iter().enumerate() {
                    let is_last = i == alternatives.len() - 1;
                    // Each alternative needs its own failure label to clean up stack
                    let try_next = self.asm.new_dynamic_label();

                    // Save position for this alternative
                    dynasm!(self.asm
                        ; push r14
                    );

                    for alt_step in alt_steps {
                        self.emit_step_inline(alt_step, try_next)?;
                    }

                    // This alternative succeeded
                    dynasm!(self.asm
                        ; add rsp, 8               // Pop saved position (don't restore)
                        ; jmp =>alt_success
                    );

                    // Restore position and try next alternative (or fail)
                    dynasm!(self.asm
                        ; =>try_next
                        ; pop r14
                    );

                    if is_last {
                        // Last alternative failed - jump to outer fail_label
                        dynasm!(self.asm
                            ; jmp =>fail_label
                        );
                    }
                }
                dynasm!(self.asm
                    ; =>alt_success
                );
            }
            _ => {
                // Unsupported step - return error
                return Err(crate::error::Error::new(
                    crate::error::ErrorKind::Jit(format!(
                        "Unsupported step in backtracking: {:?}",
                        step
                    )),
                    "",
                ));
            }
        }
        Ok(())
    }

    /// Emits standalone lookahead assertion check.
    ///
    /// For positive lookahead (?=...): continues if inner pattern matches, else jumps to fail_label.
    /// For negative lookahead (?!...): continues if inner pattern does NOT match, else jumps to fail_label.
    ///
    /// Lookahead is zero-width: position (r14) is NOT advanced.
    ///
    /// Register usage:
    /// - r14 = current_pos (preserved)
    /// - r9 = temporary position for lookahead check
    fn emit_standalone_lookahead(
        &mut self,
        inner_steps: &[PatternStep],
        fail_label: dynasmrt::DynamicLabel,
        positive: bool,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        // Optimize common case: `.*X` where X is a single step (ByteClass or Byte)
        // For `(?=.*\d)`, we need to check if a match of X exists within the range that `.*` can match
        // This is much faster than backtracking: O(n) scan vs O(n^2) backtracking
        if inner_steps.len() == 2 {
            if let PatternStep::GreedyStar(star_byte_class) = &inner_steps[0] {
                match &inner_steps[1] {
                    PatternStep::ByteClass(final_byte_class) => {
                        return self.emit_lookahead_star_scan(
                            &star_byte_class.ranges,
                            &final_byte_class.ranges,
                            fail_label,
                            positive,
                        );
                    }
                    PatternStep::Byte(byte) => {
                        // Convert single byte to a range for uniform handling
                        let final_ranges = vec![ByteRange {
                            start: *byte,
                            end: *byte,
                        }];
                        return self.emit_lookahead_star_scan(
                            &star_byte_class.ranges,
                            &final_ranges,
                            fail_label,
                            positive,
                        );
                    }
                    _ => {}
                }
            }
        }

        // Save current position (lookahead is zero-width)
        dynasm!(self.asm
            ; mov r9, r14                  // r9 = saved position
        );

        // Label for when inner pattern matches
        let inner_match = self.asm.new_dynamic_label();

        // Try to match inner pattern at current position
        for step in inner_steps {
            match step {
                PatternStep::Byte(byte) => {
                    // Check bounds and byte
                    if positive {
                        // Positive: mismatch means lookahead fails
                        dynasm!(self.asm
                            ; cmp r9, r12
                            ; jge =>fail_label         // No more input - fail
                            ; movzx eax, BYTE [rbx + r9]
                            ; cmp al, *byte as i8
                            ; jne =>fail_label         // Byte mismatch - fail
                            ; inc r9
                        );
                    } else {
                        // Negative: mismatch means lookahead succeeds (inner didn't match)
                        dynasm!(self.asm
                            ; cmp r9, r12
                            ; jge =>inner_match        // No more input - inner didn't match, success
                            ; movzx eax, BYTE [rbx + r9]
                            ; cmp al, *byte as i8
                            ; jne =>inner_match        // Byte mismatch - inner didn't match, success
                            ; inc r9
                        );
                    }
                }
                PatternStep::ByteClass(byte_class) => {
                    // Check bounds
                    if positive {
                        dynasm!(self.asm
                            ; cmp r9, r12
                            ; jge =>fail_label
                            ; movzx eax, BYTE [rbx + r9]
                        );
                        // Use a temp label for range check failure
                        let range_fail = self.asm.new_dynamic_label();
                        self.emit_range_check_with_label(
                            &byte_class.ranges,
                            range_fail,
                            fail_label,
                        )?;
                        dynasm!(self.asm
                            ; inc r9
                        );
                    } else {
                        dynasm!(self.asm
                            ; cmp r9, r12
                            ; jge =>inner_match
                            ; movzx eax, BYTE [rbx + r9]
                        );
                        let range_fail = self.asm.new_dynamic_label();
                        self.emit_range_check_with_label(
                            &byte_class.ranges,
                            range_fail,
                            inner_match,
                        )?;
                        dynasm!(self.asm
                            ; inc r9
                        );
                    }
                }
                PatternStep::WordBoundary => {
                    // Word boundary check at position r9
                    if positive {
                        self.emit_word_boundary_at_r9(fail_label, true)?;
                    } else {
                        self.emit_word_boundary_at_r9(inner_match, true)?;
                    }
                }
                PatternStep::EndOfText => {
                    if positive {
                        dynasm!(self.asm
                            ; cmp r9, r12
                            ; jne =>fail_label
                        );
                    } else {
                        dynasm!(self.asm
                            ; cmp r9, r12
                            ; jne =>inner_match
                        );
                    }
                }
                PatternStep::StartOfText => {
                    if positive {
                        dynasm!(self.asm
                            ; test r9, r9
                            ; jnz =>fail_label
                        );
                    } else {
                        dynasm!(self.asm
                            ; test r9, r9
                            ; jnz =>inner_match
                        );
                    }
                }
                PatternStep::GreedyStar(byte_class) => {
                    // Greedy star in lookahead: match as many as possible
                    // r9 is advanced through matching characters
                    let loop_start = self.asm.new_dynamic_label();
                    let loop_done = self.asm.new_dynamic_label();

                    dynasm!(self.asm
                        ; =>loop_start
                        ; cmp r9, r12
                        ; jge =>loop_done              // End of input - done looping
                        ; movzx eax, BYTE [rbx + r9]
                    );

                    // Check if byte matches any range
                    self.emit_range_check(&byte_class.ranges, loop_done)?;

                    dynasm!(self.asm
                        ; inc r9                       // Consumed another byte
                        ; jmp =>loop_start
                        ; =>loop_done
                    );
                    // r9 now points past all matched characters
                }
                PatternStep::GreedyPlus(byte_class) => {
                    // Greedy plus in lookahead: match at least one, then as many as possible
                    // First check we have at least one match
                    if positive {
                        dynasm!(self.asm
                            ; cmp r9, r12
                            ; jge =>fail_label         // No input - fail
                            ; movzx eax, BYTE [rbx + r9]
                        );
                        self.emit_range_check(&byte_class.ranges, fail_label)?;
                    } else {
                        dynasm!(self.asm
                            ; cmp r9, r12
                            ; jge =>inner_match        // No input - inner didn't match
                            ; movzx eax, BYTE [rbx + r9]
                        );
                        self.emit_range_check(&byte_class.ranges, inner_match)?;
                    }

                    dynasm!(self.asm
                        ; inc r9                       // Consumed first byte
                    );

                    // Now match as many more as possible (like GreedyStar)
                    let loop_start = self.asm.new_dynamic_label();
                    let loop_done = self.asm.new_dynamic_label();

                    dynasm!(self.asm
                        ; =>loop_start
                        ; cmp r9, r12
                        ; jge =>loop_done
                        ; movzx eax, BYTE [rbx + r9]
                    );

                    self.emit_range_check(&byte_class.ranges, loop_done)?;

                    dynasm!(self.asm
                        ; inc r9
                        ; jmp =>loop_start
                        ; =>loop_done
                    );
                }
                _ => {
                    // For complex steps in lookahead, fall back to interpreter
                    return Err(Error::new(
                        ErrorKind::Jit("Complex step in lookahead - falling back".to_string()),
                        "",
                    ));
                }
            }
        }

        // If we get here, all inner steps matched
        if positive {
            // Positive lookahead: inner matched, continue (success)
            // Position r14 unchanged (zero-width)
        } else {
            // Negative lookahead: inner matched, so assertion fails
            dynasm!(self.asm
                ; jmp =>fail_label
            );
        }

        // Label for negative lookahead success (inner didn't match)
        dynasm!(self.asm
            ; =>inner_match
        );
        // Position r14 unchanged (zero-width)

        Ok(())
    }

    /// Emits optimized lookahead code for `.*X` patterns.
    ///
    /// For a pattern like `(?=.*\d)`, instead of greedily matching `.*` and then
    /// checking if `\d` matches at the end (which always fails), we:
    /// 1. Find the extent of `.*` (star_end = where the star can match to)
    /// 2. Scan from current position to star_end looking for ANY match of the final pattern
    ///
    /// This is O(n) instead of O(n^2) and matches the interpreter's `check_lookahead()` optimization.
    ///
    /// Register usage:
    /// - r14 = current_pos (preserved, lookahead is zero-width)
    /// - r9 = star_end (where `.*` can extend to)
    /// - r10 = scan position (iterates from r14 to r9)
    fn emit_lookahead_star_scan(
        &mut self,
        star_ranges: &[ByteRange],
        final_ranges: &[ByteRange],
        fail_label: dynasmrt::DynamicLabel,
        positive: bool,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        let found_match = self.asm.new_dynamic_label();
        let scan_loop = self.asm.new_dynamic_label();
        let scan_done = self.asm.new_dynamic_label();
        let star_loop = self.asm.new_dynamic_label();
        let star_done = self.asm.new_dynamic_label();

        // Step 1: Find star_end - the extent of where `.*` can match to
        // r9 starts at r14 (current position), then we scan forward while star_ranges match
        dynasm!(self.asm
            ; mov r9, r14                  // r9 = star_end, starts at current pos

            ; =>star_loop
            ; cmp r9, r12
            ; jge =>star_done              // End of input
            ; movzx eax, BYTE [rbx + r9]
        );

        // Check if byte matches star_ranges (e.g., for `.*`, this excludes newline)
        self.emit_range_check(star_ranges, star_done)?;

        dynasm!(self.asm
            ; inc r9                       // Matched, advance star_end
            ; jmp =>star_loop

            ; =>star_done
            // r9 = star_end (exclusive - position past all star matches)
        );

        // Step 2: Scan from r14 to r9 looking for ANY match of final_ranges
        // r10 = scan position
        dynasm!(self.asm
            ; mov r10, r14                 // r10 = scan position, starts at current pos

            ; =>scan_loop
            ; cmp r10, r9
            ; jg =>scan_done               // Scanned past star_end (inclusive means <=)
            ; cmp r10, r12
            ; jge =>scan_done              // End of input

            ; movzx eax, BYTE [rbx + r10]
        );

        // Check if byte matches final_ranges (e.g., `\d`)
        // If it matches, we found it - jump to found_match
        // If it doesn't match, continue scanning
        let check_next = self.asm.new_dynamic_label();
        self.emit_range_check(final_ranges, check_next)?;

        // If we reach here, final_ranges matched!
        dynasm!(self.asm
            ; jmp =>found_match

            ; =>check_next
            ; inc r10                      // Not a match, try next position
            ; jmp =>scan_loop

            ; =>scan_done
            // Scanned entire range without finding a match
        );

        // Determine success/failure based on positive/negative lookahead
        if positive {
            // Positive lookahead: we need to find a match
            // scan_done means we didn't find one -> fail
            // found_match means we found one -> success (continue)
            dynasm!(self.asm
                ; jmp =>fail_label         // No match found -> fail

                ; =>found_match
                // Match found, positive lookahead succeeds
                // r14 unchanged (zero-width)
            );
        } else {
            // Negative lookahead: we need to NOT find a match
            // scan_done means we didn't find one -> success (continue)
            // found_match means we found one -> fail
            dynasm!(self.asm
                ; jmp >neg_success         // No match found -> success

                ; =>found_match
                ; jmp =>fail_label         // Match found -> fail

                ; neg_success:
                // No match found, negative lookahead succeeds
                // r14 unchanged (zero-width)
            );
        }

        Ok(())
    }

    /// Helper: emit range check that jumps to success_label on mismatch (for negative lookahead)
    fn emit_range_check_with_label(
        &mut self,
        ranges: &[ByteRange],
        _range_fail: dynasmrt::DynamicLabel,
        target_on_fail: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        if ranges.len() == 1 {
            let range = &ranges[0];
            let range_size = range.end.wrapping_sub(range.start);
            dynasm!(self.asm
                ; sub al, range.start as i8
                ; cmp al, range_size as i8
                ; ja =>target_on_fail
            );
        } else {
            let range_matched = self.asm.new_dynamic_label();
            for (ri, range) in ranges.iter().enumerate() {
                let is_last = ri == ranges.len() - 1;
                let range_size = range.end.wrapping_sub(range.start);

                if is_last {
                    dynasm!(self.asm
                        ; mov cl, al
                        ; sub cl, range.start as i8
                        ; cmp cl, range_size as i8
                        ; ja =>target_on_fail
                    );
                } else {
                    dynasm!(self.asm
                        ; mov cl, al
                        ; sub cl, range.start as i8
                        ; cmp cl, range_size as i8
                        ; jbe =>range_matched
                    );
                }
            }
            dynasm!(self.asm
                ; =>range_matched
            );
        }
        Ok(())
    }

    /// Helper: emit word boundary check using r9 as position
    fn emit_word_boundary_at_r9(
        &mut self,
        fail_label: dynasmrt::DynamicLabel,
        is_boundary: bool,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        // Get prev_is_word (r8b = 0 or 1)
        dynasm!(self.asm
            ; xor r8d, r8d                 // prev_is_word = false
            ; test r9, r9
            ; jz >check_curr               // At start, no prev char
            ; mov rax, r9
            ; dec rax
            ; movzx eax, BYTE [rbx + rax]
            // Check if prev is word char: [A-Za-z0-9_]
            ; cmp al, b'_' as i8
            ; je >prev_is_word
            ; mov cl, al
            ; sub cl, b'0' as i8
            ; cmp cl, 9
            ; jbe >prev_is_word
            ; mov cl, al
            ; sub cl, b'A' as i8
            ; cmp cl, 25
            ; jbe >prev_is_word
            ; mov cl, al
            ; sub cl, b'a' as i8
            ; cmp cl, 25
            ; jbe >prev_is_word
            ; jmp >check_curr
            ; prev_is_word:
            ; mov r8b, 1
            ; check_curr:
        );

        // Get curr_is_word (compare with r8b)
        dynasm!(self.asm
            ; cmp r9, r12
            ; jge >at_end                  // At end, curr_is_word = false
            ; movzx eax, BYTE [rbx + r9]
            // Check if curr is word char
            ; cmp al, b'_' as i8
            ; je >curr_is_word
            ; mov cl, al
            ; sub cl, b'0' as i8
            ; cmp cl, 9
            ; jbe >curr_is_word
            ; mov cl, al
            ; sub cl, b'A' as i8
            ; cmp cl, 25
            ; jbe >curr_is_word
            ; mov cl, al
            ; sub cl, b'a' as i8
            ; cmp cl, 25
            ; jbe >curr_is_word
            ; jmp >curr_not_word
            ; curr_is_word:
            ; xor r8b, 1                   // XOR with curr_is_word=1
            ; jmp >check_result
            ; curr_not_word:
            ; jmp >check_result
            ; at_end:
            ; check_result:
        );

        // r8b now contains prev_is_word XOR curr_is_word
        // For \b: need XOR to be 1 (boundary exists)
        // For \B: need XOR to be 0 (no boundary)
        if is_boundary {
            dynasm!(self.asm
                ; test r8b, r8b
                ; jz =>fail_label          // XOR is 0 means no boundary
            );
        } else {
            dynasm!(self.asm
                ; test r8b, r8b
                ; jnz =>fail_label         // XOR is 1 means boundary exists
            );
        }

        Ok(())
    }

    /// Emits lookbehind assertion check.
    ///
    /// For positive lookbehind (?<=...): continues if inner pattern matches behind, else jumps to fail_label.
    /// For negative lookbehind (?<!...): continues if inner pattern does NOT match behind, else jumps to fail_label.
    ///
    /// Lookbehind is zero-width: position (r14) is NOT advanced.
    ///
    /// Register usage:
    /// - r14 = current_pos (preserved)
    /// - r9 = saved position / position for lookbehind check
    fn emit_lookbehind_check(
        &mut self,
        inner_steps: &[PatternStep],
        min_len: usize,
        fail_label: dynasmrt::DynamicLabel,
        positive: bool,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        // Labels for lookbehind control flow
        let inner_match = self.asm.new_dynamic_label();
        let inner_mismatch = self.asm.new_dynamic_label();
        let done = self.asm.new_dynamic_label();

        // Save current position
        dynasm!(self.asm
            ; mov r9, r14                  // r9 = saved position
        );

        // Check if there's enough space behind (current_pos >= min_len)
        if min_len > 0 {
            dynasm!(self.asm
                ; cmp r14, min_len as i32
                ; jl =>inner_mismatch          // Not enough characters behind
            );
        }

        // Set position to start of lookbehind check
        dynasm!(self.asm
            ; sub r14, min_len as i32      // r14 = current_pos - min_len
        );

        // Try to match inner pattern (starting from r14)
        for step in inner_steps {
            match step {
                PatternStep::Byte(byte) => {
                    // We know there's exactly min_len bytes available
                    dynasm!(self.asm
                        ; movzx eax, BYTE [rbx + r14]
                        ; cmp al, *byte as i8
                        ; jne =>inner_mismatch
                        ; inc r14
                    );
                }
                PatternStep::ByteClass(byte_class) => {
                    dynasm!(self.asm
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(&byte_class.ranges, inner_mismatch)?;
                    dynasm!(self.asm
                        ; inc r14
                    );
                }
                _ => {
                    // Unsupported step in lookbehind - fall back
                    return Err(Error::new(
                        ErrorKind::Jit("Unsupported pattern step in lookbehind".to_string()),
                        "",
                    ));
                }
            }
        }

        // Inner pattern matched
        dynasm!(self.asm
            ; jmp =>inner_match
        );

        // Inner pattern didn't match
        dynasm!(self.asm
            ; =>inner_mismatch
        );

        if positive {
            // Positive lookbehind: inner mismatch means assertion fails
            dynasm!(self.asm
                ; mov r14, r9              // Restore position
                ; jmp =>fail_label
            );

            dynasm!(self.asm
                ; =>inner_match
                ; mov r14, r9              // Restore position (lookbehind is zero-width)
                ; jmp =>done
            );
        } else {
            // Negative lookbehind: inner mismatch means assertion succeeds
            dynasm!(self.asm
                ; mov r14, r9              // Restore position
                ; jmp =>done
            );

            dynasm!(self.asm
                ; =>inner_match
                ; mov r14, r9              // Restore position
                ; jmp =>fail_label
            );
        }

        dynasm!(self.asm
            ; =>done
        );

        Ok(())
    }

    /// Emits greedy+ with lookahead for the captures path.
    /// Reuses the alternation version since register conventions are the same.
    fn emit_greedy_plus_with_lookahead_in_captures(
        &mut self,
        ranges: &[ByteRange],
        lookahead_steps: &[PatternStep],
        is_positive: bool,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        // The captures path and alternation path use the same register conventions:
        // r14 = current position, rbx = input base, r12 = input length
        // So we can reuse the alternation implementation directly.
        self.emit_greedy_plus_with_lookahead_in_alt(
            ranges,
            lookahead_steps,
            is_positive,
            fail_label,
        )
    }

    /// Emits greedy* with lookahead for the captures path.
    /// Reuses the alternation version since register conventions are the same.
    fn emit_greedy_star_with_lookahead_in_captures(
        &mut self,
        ranges: &[ByteRange],
        lookahead_steps: &[PatternStep],
        is_positive: bool,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        // The captures path and alternation path use the same register conventions:
        // r14 = current position, rbx = input base, r12 = input length
        // So we can reuse the alternation implementation directly.
        self.emit_greedy_star_with_lookahead_in_alt(
            ranges,
            lookahead_steps,
            is_positive,
            fail_label,
        )
    }

    /// Emits code to decode one UTF-8 codepoint from input.
    ///
    /// On entry:
    /// - rbx = input_ptr
    /// - r14 = current position
    /// - r12 = input length
    ///
    /// On success:
    /// - eax = decoded codepoint (u32)
    /// - ecx = byte length (1-4)
    /// - Does NOT modify r14 (caller advances position)
    ///
    /// On failure (end of input or invalid UTF-8):
    /// - Jumps to fail_label
    ///
    /// Clobbers: rax, rcx, r8, r9
    fn emit_utf8_decode(&mut self, fail_label: dynasmrt::DynamicLabel) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        // The interpreter's decode_utf8 algorithm:
        // - 0x00-0x7F: 1 byte ASCII (cp = b0)
        // - 0x80-0xBF: Invalid (continuation at start)
        // - 0xC0-0xDF: 2 bytes (cp = (b0 & 0x1F) << 6 | (b1 & 0x3F))
        // - 0xE0-0xEF: 3 bytes (cp = (b0 & 0x0F) << 12 | (b1 & 0x3F) << 6 | (b2 & 0x3F))
        // - 0xF0-0xF7: 4 bytes (cp = (b0 & 0x07) << 18 | (b1 & 0x3F) << 12 | (b2 & 0x3F) << 6 | (b3 & 0x3F))

        let ascii = self.asm.new_dynamic_label();
        let two_byte = self.asm.new_dynamic_label();
        let three_byte = self.asm.new_dynamic_label();
        let four_byte = self.asm.new_dynamic_label();
        let done = self.asm.new_dynamic_label();

        // Check if at end of input
        dynasm!(self.asm
            ; cmp r14, r12
            ; jge =>fail_label

            // Load first byte
            ; movzx eax, BYTE [rbx + r14]

            // Check lead byte type
            ; cmp al, 0x80u8 as i8
            ; jb =>ascii          // < 0x80: ASCII

            ; cmp al, 0xC0u8 as i8
            ; jb =>fail_label     // 0x80-0xBF: invalid (continuation at start)

            ; cmp al, 0xE0u8 as i8
            ; jb =>two_byte       // 0xC0-0xDF: 2-byte sequence

            ; cmp al, 0xF0u8 as i8
            ; jb =>three_byte     // 0xE0-0xEF: 3-byte sequence

            ; cmp al, 0xF8u8 as i8
            ; jb =>four_byte      // 0xF0-0xF7: 4-byte sequence

            ; jmp =>fail_label    // >= 0xF8: invalid
        );

        // ASCII (1 byte): codepoint = b0, len = 1
        dynasm!(self.asm
            ; =>ascii
            ; mov ecx, 1
            // eax already contains the codepoint
            ; jmp =>done
        );

        // 2-byte sequence: need 1 more byte
        dynasm!(self.asm
            ; =>two_byte
            ; mov r8, r14
            ; inc r8
            ; cmp r8, r12
            ; jge =>fail_label    // Not enough bytes

            // Load second byte, check it's a continuation (0x80-0xBF)
            ; movzx r9d, BYTE [rbx + r8]
            ; mov ecx, r9d
            ; and ecx, 0xC0
            ; cmp ecx, 0x80
            ; jne =>fail_label    // Not a continuation byte

            // Decode: cp = (b0 & 0x1F) << 6 | (b1 & 0x3F)
            ; and eax, 0x1F       // eax = b0 & 0x1F
            ; shl eax, 6          // eax = (b0 & 0x1F) << 6
            ; and r9d, 0x3F       // r9d = b1 & 0x3F
            ; or eax, r9d         // eax = codepoint
            ; mov ecx, 2
            ; jmp =>done
        );

        // 3-byte sequence: need 2 more bytes
        dynasm!(self.asm
            ; =>three_byte
            ; mov r8, r14
            ; add r8, 2
            ; cmp r8, r12
            ; jge =>fail_label    // Not enough bytes

            // Check continuation bytes
            ; movzx r9d, BYTE [rbx + r14 + 1]
            ; mov ecx, r9d
            ; and ecx, 0xC0
            ; cmp ecx, 0x80
            ; jne =>fail_label

            ; movzx r8d, BYTE [rbx + r14 + 2]
            ; mov ecx, r8d
            ; and ecx, 0xC0
            ; cmp ecx, 0x80
            ; jne =>fail_label

            // Decode: cp = (b0 & 0x0F) << 12 | (b1 & 0x3F) << 6 | (b2 & 0x3F)
            ; and eax, 0x0F       // eax = b0 & 0x0F
            ; shl eax, 12         // eax = (b0 & 0x0F) << 12
            ; and r9d, 0x3F       // r9d = b1 & 0x3F
            ; shl r9d, 6          // r9d = (b1 & 0x3F) << 6
            ; or eax, r9d         // eax |= (b1 & 0x3F) << 6
            ; and r8d, 0x3F       // r8d = b2 & 0x3F
            ; or eax, r8d         // eax = codepoint
            ; mov ecx, 3
            ; jmp =>done
        );

        // 4-byte sequence: need 3 more bytes
        dynasm!(self.asm
            ; =>four_byte
            ; mov r8, r14
            ; add r8, 3
            ; cmp r8, r12
            ; jge =>fail_label    // Not enough bytes

            // Check continuation bytes
            ; movzx r9d, BYTE [rbx + r14 + 1]
            ; mov ecx, r9d
            ; and ecx, 0xC0
            ; cmp ecx, 0x80
            ; jne =>fail_label

            ; movzx r8d, BYTE [rbx + r14 + 2]
            ; mov ecx, r8d
            ; and ecx, 0xC0
            ; cmp ecx, 0x80
            ; jne =>fail_label

            ; push r10            // Save r10 since we need another register
            ; movzx r10d, BYTE [rbx + r14 + 3]
            ; mov ecx, r10d
            ; and ecx, 0xC0
            ; cmp ecx, 0x80
            ; jne >four_byte_fail

            // Decode: cp = (b0 & 0x07) << 18 | (b1 & 0x3F) << 12 | (b2 & 0x3F) << 6 | (b3 & 0x3F)
            ; and eax, 0x07       // eax = b0 & 0x07
            ; shl eax, 18         // eax = (b0 & 0x07) << 18
            ; and r9d, 0x3F       // r9d = b1 & 0x3F
            ; shl r9d, 12         // r9d = (b1 & 0x3F) << 12
            ; or eax, r9d
            ; and r8d, 0x3F       // r8d = b2 & 0x3F
            ; shl r8d, 6          // r8d = (b2 & 0x3F) << 6
            ; or eax, r8d
            ; and r10d, 0x3F      // r10d = b3 & 0x3F
            ; or eax, r10d        // eax = codepoint
            ; pop r10
            ; mov ecx, 4
            ; jmp =>done

            ; four_byte_fail:
            ; pop r10
            ; jmp =>fail_label
        );

        dynasm!(self.asm
            ; =>done
            // eax = codepoint, ecx = byte length
        );

        Ok(())
    }

    /// Emits code to check if a codepoint (in eax) is in the given class.
    ///
    /// Uses ASCII fast path (bitmap lookup) for codepoints < 128,
    /// falls back to Rust function call for non-ASCII.
    ///
    /// On entry:
    /// - eax = codepoint to check
    ///
    /// On success: falls through
    /// On failure: jumps to fail_label
    /// Clobbers: rax, rcx, rdi, rsi, caller-saved registers
    fn emit_codepoint_class_membership_check(
        &mut self,
        cpclass: &CodepointClass,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        let ascii_fast_path = self.asm.new_dynamic_label();
        let check_done = self.asm.new_dynamic_label();

        // Store bitmap values (these are compile-time constants)
        let bitmap_lo = cpclass.ascii_bitmap[0];
        let bitmap_hi = cpclass.ascii_bitmap[1];
        let is_negated = cpclass.negated;

        // ASCII fast path: if codepoint < 128, use bitmap lookup
        dynasm!(self.asm
            ; cmp eax, 128
            ; jb =>ascii_fast_path
        );

        // Slow path: call Rust function for non-ASCII codepoints
        // Store the CodepointClass in a Box to ensure stable address
        let cpclass_box = Box::new(cpclass.clone());
        let cpclass_ptr = cpclass_box.as_ref() as *const CodepointClass;

        // Keep the box alive by storing it
        self.codepoint_classes.push(cpclass_box);

        // Helper function that checks membership
        // Use platform-specific calling convention
        #[cfg(target_os = "windows")]
        extern "win64" fn check_membership(codepoint: u32, cpclass: *const CodepointClass) -> bool {
            let cpclass = unsafe { &*cpclass };
            cpclass.contains(codepoint)
        }

        #[cfg(not(target_os = "windows"))]
        extern "sysv64" fn check_membership(
            codepoint: u32,
            cpclass: *const CodepointClass,
        ) -> bool {
            let cpclass = unsafe { &*cpclass };
            cpclass.contains(codepoint)
        }

        #[cfg(target_os = "windows")]
        let check_fn_ptr = {
            let check_fn: extern "win64" fn(u32, *const CodepointClass) -> bool = check_membership;
            check_fn as usize as i64
        };

        #[cfg(not(target_os = "windows"))]
        let check_fn_ptr = {
            let check_fn: extern "sysv64" fn(u32, *const CodepointClass) -> bool = check_membership;
            check_fn as usize as i64
        };

        // Call the helper function with platform-specific calling convention
        #[cfg(target_os = "windows")]
        dynasm!(self.asm
            // eax already contains codepoint
            // Windows x64: args in RCX, RDX
            ; mov ecx, eax                    // rcx = codepoint (zero-extended)
            ; mov rdx, QWORD cpclass_ptr as i64  // rdx = cpclass pointer
            ; sub rsp, 32                     // Shadow space
            ; mov rax, QWORD check_fn_ptr     // Load function pointer
            ; call rax                        // Call check_membership
            ; add rsp, 32                     // Restore stack

            // rax (al) = result: true (1) if in class, false (0) if not
            ; test al, al
            ; jz =>fail_label                 // If false, jump to fail
            ; jmp =>check_done
        );

        #[cfg(not(target_os = "windows"))]
        dynasm!(self.asm
            // eax already contains codepoint
            // System V ABI: args in RDI, RSI
            ; mov edi, eax                    // rdi = codepoint (zero-extended)
            ; mov rsi, QWORD cpclass_ptr as i64  // rsi = cpclass pointer
            ; mov rax, QWORD check_fn_ptr     // Load function pointer
            ; call rax                        // Call check_membership

            // rax (al) = result: true (1) if in class, false (0) if not
            ; test al, al
            ; jz =>fail_label                 // If false, jump to fail
            ; jmp =>check_done
        );

        // ASCII fast path: inline bitmap test
        // eax = codepoint (0-127)
        // bitmap[0] covers bits 0-63, bitmap[1] covers bits 64-127
        dynasm!(self.asm
            ; =>ascii_fast_path
            // Check if codepoint < 64 (use bitmap[0]) or >= 64 (use bitmap[1])
            ; cmp eax, 64
            ; jae >use_hi_bitmap

            // codepoint < 64: test bit in bitmap[0]
            ; mov rcx, QWORD bitmap_lo as i64
            ; mov rdi, rax          // rdi = codepoint (for bt instruction)
            ; bt rcx, rdi           // CF = bit at position rdi in rcx
            ; jmp >check_bitmap_result

            ; use_hi_bitmap:
            // codepoint >= 64: test bit in bitmap[1]
            ; mov rcx, QWORD bitmap_hi as i64
            ; mov rdi, rax
            ; sub rdi, 64           // Adjust for bitmap[1] offset
            ; bt rcx, rdi           // CF = bit at position rdi in rcx

            ; check_bitmap_result:
        );

        // CF is set if bit is 1 (codepoint in ranges)
        // Handle negation: if negated, we want to SUCCEED when bit is NOT set
        if is_negated {
            // Negated class: succeed if NOT in bitmap (CF=0)
            dynasm!(self.asm
                ; jc =>fail_label    // CF=1 means in bitmap, but negated => fail
            );
        } else {
            // Normal class: succeed if in bitmap (CF=1)
            dynasm!(self.asm
                ; jnc =>fail_label   // CF=0 means not in bitmap => fail
            );
        }

        dynasm!(self.asm
            ; =>check_done
        );

        Ok(())
    }

    /// Emits code for a single CodepointClass check.
    ///
    /// Mirrors interpreter's CodepointClass handling:
    /// 1. Decode one UTF-8 codepoint
    /// 2. Check if codepoint is in the class (fast path for ASCII, slow path for others)
    /// 3. Advance position by the byte length
    ///
    /// Register usage:
    /// - rbx = input_ptr
    /// - r14 = current position (advanced on success)
    /// - r12 = input length
    fn emit_codepoint_class_check(
        &mut self,
        cpclass: &CodepointClass,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        let fail_with_stack = self.asm.new_dynamic_label();

        // Decode UTF-8 codepoint
        // After: eax = codepoint, ecx = byte_length
        self.emit_utf8_decode(fail_label)?;

        // Save byte length on stack (we need it after the membership check)
        dynasm!(self.asm
            ; push rcx            // Save byte length
        );

        // Check codepoint membership (includes ASCII fast path)
        // If fail, need to pop stack first
        self.emit_codepoint_class_membership_check(cpclass, fail_with_stack)?;

        // Restore byte length and advance position
        dynasm!(self.asm
            ; pop rcx             // Restore byte length
            ; add r14, rcx        // Advance position by byte_length
            ; jmp >done

            ; =>fail_with_stack
            ; add rsp, 8          // Pop the saved rcx
            ; jmp =>fail_label

            ; done:
        );

        Ok(())
    }

    /// Emits code for GreedyCodepointPlus - greedy one-or-more codepoint matching.
    ///
    /// Mirrors interpreter's GreedyCodepointPlus handling:
    /// 1. Must match at least one codepoint
    /// 2. Then match as many codepoints as possible (greedy)
    ///
    /// Register usage:
    /// - rbx = input_ptr
    /// - r14 = current position (advanced on matches)
    /// - r12 = input length
    fn emit_greedy_codepoint_plus(
        &mut self,
        cpclass: &CodepointClass,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        let loop_start = self.asm.new_dynamic_label();
        let loop_done = self.asm.new_dynamic_label();
        let first_fail_with_stack = self.asm.new_dynamic_label();
        let loop_fail_no_stack = self.asm.new_dynamic_label();
        let loop_fail_with_stack = self.asm.new_dynamic_label();

        // First iteration: must match at least one codepoint
        // Decode UTF-8 and check membership
        self.emit_utf8_decode(fail_label)?;

        dynasm!(self.asm
            ; push rcx            // Save byte length
        );

        // Check membership - if fail, need to pop stack first
        self.emit_codepoint_class_membership_check(cpclass, first_fail_with_stack)?;

        dynasm!(self.asm
            ; pop rcx             // Restore byte length
            ; add r14, rcx        // Advance position
        );

        // Greedy loop: match as many more codepoints as possible
        dynasm!(self.asm
            ; =>loop_start
        );

        // Try to decode another codepoint (failure means end of greedy match, not overall fail)
        // No stack push has happened yet at this point
        self.emit_utf8_decode(loop_fail_no_stack)?;

        dynasm!(self.asm
            ; push rcx            // Save byte length
        );

        // Check membership - on failure, need to pop stack first then exit loop
        self.emit_codepoint_class_membership_check(cpclass, loop_fail_with_stack)?;

        dynasm!(self.asm
            ; pop rcx             // Restore byte length
            ; add r14, rcx        // Advance position
            ; jmp =>loop_start

            // First match failed with stack (first codepoint not in class)
            ; =>first_fail_with_stack
            ; add rsp, 8          // Pop the saved rcx
            ; jmp =>fail_label

            // Loop exit: UTF-8 decode failed (end of input or invalid UTF-8)
            // No stack cleanup needed
            ; =>loop_fail_no_stack
            ; jmp =>loop_done

            // Loop exit: codepoint not in class (after push rcx)
            // Need to pop stack
            ; =>loop_fail_with_stack
            ; add rsp, 8          // Pop the saved rcx
            ; jmp =>loop_done

            ; =>loop_done
            // Successfully matched at least one codepoint, greedy match complete
        );

        Ok(())
    }

    /// Emits the captures_fn with full capture tracking.
    ///
    /// Returns the offset of the generated code.
    ///
    /// Register allocation:
    /// - rbx = input_ptr (callee-saved)
    /// - r12 = input_len (callee-saved)
    /// - r13 = start_pos (callee-saved)
    /// - r14 = current_pos (callee-saved)
    /// - r15 = captures_out pointer (callee-saved)
    /// - rax = scratch / return value
    ///
    /// Arguments (System V AMD64 ABI):
    /// - rdi = input_ptr
    /// - rsi = input_len
    /// - rdx = ctx (unused in this impl)
    /// - rcx = captures_out (pointer to i64 array)
    ///
    /// Captures are written directly to the captures_out buffer.
    fn emit_captures_fn(&mut self, steps: &[PatternStep]) -> Result<dynasmrt::AssemblyOffset> {
        use dynasmrt::DynasmLabelApi;

        let offset = self.asm.offset();
        let min_len = Self::calc_min_len(steps);

        // Count max capture group index
        // Note: capture indices are 1-based (group 0 is the full match)
        let max_capture_idx = steps
            .iter()
            .filter_map(|s| match s {
                PatternStep::CaptureStart(idx) | PatternStep::CaptureEnd(idx) => Some(*idx),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        // Number of slots: (max_capture_idx + 1) groups * 2 slots each
        // This includes group 0 (full match) since max_capture_idx >= 1 for patterns with captures
        // E.g., for capture group 1: max_capture_idx=1, num_slots = (1+1)*2 = 4
        let num_slots = (max_capture_idx as usize + 1) * 2;

        // Prologue - save callee-saved registers
        // On function entry: RSP is 8-mod-16 (return address pushed)
        #[cfg(target_os = "windows")]
        dynasm!(self.asm
            ; push rdi          // Callee-saved on Windows
            ; push rsi          // Callee-saved on Windows
            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15
            // Windows x64: args in RCX, RDX, R8, R9
            // RCX=input_ptr, RDX=input_len, R8=ctx (unused), R9=captures_out
            ; mov rbx, rcx      // rbx = input_ptr
            ; mov r12, rdx      // r12 = input_len
            ; mov r15, r9       // r15 = captures_out pointer
            ; xor r13d, r13d    // r13 = start_pos = 0
        );

        #[cfg(not(target_os = "windows"))]
        dynasm!(self.asm
            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15
            // System V AMD64: args in RDI, RSI, RDX, RCX
            // rdi = input_ptr, rsi = input_len, rdx = ctx (unused), rcx = captures_out
            ; mov rbx, rdi      // rbx = input_ptr
            ; mov r12, rsi      // r12 = input_len
            ; mov r15, rcx      // r15 = captures_out pointer
            ; xor r13d, r13d    // r13 = start_pos = 0
        );

        // Initialize all capture slots to -1
        for slot in 0..num_slots {
            let slot_offset = (slot * 8) as i32;
            dynasm!(self.asm
                ; mov QWORD [r15 + slot_offset], -1i32
            );
        }

        // Main loop
        let start_loop = self.asm.new_dynamic_label();
        let match_found = self.asm.new_dynamic_label();
        let no_match = self.asm.new_dynamic_label();
        let byte_mismatch = self.asm.new_dynamic_label();

        dynasm!(self.asm
            ; =>start_loop
            ; mov rax, r12
            ; sub rax, r13
            ; cmp rax, min_len as i32
            ; jl =>no_match
        );

        // r14 = current position
        dynasm!(self.asm
            ; mov r14, r13
        );

        // Set group 0 start (slot 0)
        dynasm!(self.asm
            ; mov QWORD [r15], r13
        );

        // Generate matching code with capture handling
        for step in steps.iter() {
            self.emit_capture_step(step, byte_mismatch, 0)?; // stack_align unused now
        }

        // Match found
        dynasm!(self.asm
            ; jmp =>match_found
        );

        // Byte mismatch - reset captures and try next position
        dynasm!(self.asm
            ; =>byte_mismatch
        );
        // Reset capture slots to -1
        for slot in 0..num_slots {
            let slot_offset = (slot * 8) as i32;
            dynasm!(self.asm
                ; mov QWORD [r15 + slot_offset], -1i32
            );
        }
        dynasm!(self.asm
            ; inc r13
            ; jmp =>start_loop
        );

        // Match found - set group 0 end (slot 1)
        dynasm!(self.asm
            ; =>match_found
            ; mov QWORD [r15 + 8], r14
        );

        // Return (start << 32 | end) to indicate success and positions
        dynasm!(self.asm
            ; mov rax, r13
            ; shl rax, 32
            ; or rax, r14
        );

        // Epilogue - restore callee-saved registers
        #[cfg(target_os = "windows")]
        dynasm!(self.asm
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; pop rsi
            ; pop rdi
            ; ret
        );
        #[cfg(not(target_os = "windows"))]
        dynasm!(self.asm
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; ret
        );

        // No match
        dynasm!(self.asm
            ; =>no_match
            ; mov rax, -1i32
        );

        // Epilogue - restore callee-saved registers
        #[cfg(target_os = "windows")]
        dynasm!(self.asm
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; pop rsi
            ; pop rdi
            ; ret
        );
        #[cfg(not(target_os = "windows"))]
        dynasm!(self.asm
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; ret
        );

        Ok(offset)
    }

    /// Emits code for a single pattern step with capture handling.
    fn emit_capture_step(
        &mut self,
        step: &PatternStep,
        fail_label: dynasmrt::DynamicLabel,
        _stack_align: i32,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        match step {
            PatternStep::Byte(byte) => {
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                    ; cmp al, *byte as i8
                    ; jne =>fail_label
                    ; inc r14
                );
            }
            PatternStep::ByteClass(byte_class) => {
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14
                );
            }
            PatternStep::GreedyPlus(byte_class) => {
                let loop_start = self.asm.new_dynamic_label();
                let loop_done = self.asm.new_dynamic_label();

                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14
                    ; =>loop_start
                    ; cmp r14, r12
                    ; jge =>loop_done
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, loop_done)?;
                dynasm!(self.asm
                    ; inc r14
                    ; jmp =>loop_start
                    ; =>loop_done
                );
            }
            PatternStep::GreedyStar(byte_class) => {
                let loop_start = self.asm.new_dynamic_label();
                let loop_done = self.asm.new_dynamic_label();

                dynasm!(self.asm
                    ; =>loop_start
                    ; cmp r14, r12
                    ; jge =>loop_done
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, loop_done)?;
                dynasm!(self.asm
                    ; inc r14
                    ; jmp =>loop_start
                    ; =>loop_done
                );
            }
            PatternStep::CaptureStart(idx) => {
                // Write current position to capture start slot in r15 (captures buffer)
                // Slot layout: [grp0_start, grp0_end, grp1_start, grp1_end, ...]
                // Slot index = idx * 2 (but idx starts at 1 for capture groups, 0 is full match)
                let slot_offset = ((*idx as usize) * 2 * 8) as i32;
                dynasm!(self.asm
                    ; mov QWORD [r15 + slot_offset], r14
                );
            }
            PatternStep::CaptureEnd(idx) => {
                // Write current position to capture end slot in r15 (captures buffer)
                // Slot index = idx * 2 + 1
                let slot_offset = ((*idx as usize) * 2 * 8 + 8) as i32;
                dynasm!(self.asm
                    ; mov QWORD [r15 + slot_offset], r14
                );
            }
            PatternStep::Alt(alternatives) => {
                // Alternation with captures
                // NOTE: We use the stack to save position since r15 is used for captures buffer
                let alt_success = self.asm.new_dynamic_label();
                let alt_fail = self.asm.new_dynamic_label();

                dynasm!(self.asm
                    ; push r14    // Save position on stack
                );

                for (alt_idx, alt_steps) in alternatives.iter().enumerate() {
                    let is_last = alt_idx == alternatives.len() - 1;
                    let try_next_alt = if is_last {
                        alt_fail // Jump to our local fail handler that cleans up stack
                    } else {
                        self.asm.new_dynamic_label()
                    };

                    for alt_step in alt_steps.iter() {
                        self.emit_capture_step(alt_step, try_next_alt, _stack_align)?;
                    }

                    dynasm!(self.asm
                        ; add rsp, 8    // Clean up saved position
                        ; jmp =>alt_success
                    );

                    if !is_last {
                        dynasm!(self.asm
                            ; =>try_next_alt
                            ; mov r14, [rsp]    // Restore position from stack
                        );
                    }
                }

                // All alternatives failed - clean up and jump to outer fail label
                dynasm!(self.asm
                    ; =>alt_fail
                    ; add rsp, 8    // Clean up saved position
                    ; jmp =>fail_label
                );

                dynasm!(self.asm
                    ; =>alt_success
                );
            }
            PatternStep::CodepointClass(cpclass, _target) => {
                // Unicode codepoint class - decode UTF-8 and check membership
                self.emit_codepoint_class_check(cpclass, fail_label)?;
            }
            PatternStep::GreedyCodepointPlus(cpclass) => {
                // Greedy codepoint repetition - decode UTF-8 and match greedily
                self.emit_greedy_codepoint_plus(cpclass, fail_label)?;
            }
            PatternStep::WordBoundary => {
                // Word boundary assertion in captures - doesn't consume input
                self.emit_word_boundary_check(fail_label, true)?;
            }
            PatternStep::NotWordBoundary => {
                // Not word boundary assertion in captures - doesn't consume input
                self.emit_word_boundary_check(fail_label, false)?;
            }
            PatternStep::PositiveLookahead(inner_steps) => {
                // Zero-width assertion - doesn't consume input, doesn't affect captures
                self.emit_standalone_lookahead(inner_steps, fail_label, true)?;
            }
            PatternStep::NegativeLookahead(inner_steps) => {
                // Zero-width assertion - doesn't consume input, doesn't affect captures
                self.emit_standalone_lookahead(inner_steps, fail_label, false)?;
            }
            PatternStep::PositiveLookbehind(inner_steps, min_len) => {
                // Zero-width assertion - doesn't consume input, doesn't affect captures
                self.emit_lookbehind_check(inner_steps, *min_len, fail_label, true)?;
            }
            PatternStep::NegativeLookbehind(inner_steps, min_len) => {
                // Zero-width assertion - doesn't consume input, doesn't affect captures
                self.emit_lookbehind_check(inner_steps, *min_len, fail_label, false)?;
            }
            PatternStep::GreedyPlusLookahead(byte_class, lookahead_steps, is_positive) => {
                // Greedy+ with lookahead in captures path
                self.emit_greedy_plus_with_lookahead_in_captures(
                    &byte_class.ranges,
                    lookahead_steps,
                    *is_positive,
                    fail_label,
                )?;
            }
            PatternStep::GreedyStarLookahead(byte_class, lookahead_steps, is_positive) => {
                // Greedy* with lookahead in captures path
                self.emit_greedy_star_with_lookahead_in_captures(
                    &byte_class.ranges,
                    lookahead_steps,
                    *is_positive,
                    fail_label,
                )?;
            }
            PatternStep::Backref(idx) => {
                // Backreference: compare captured text with current position
                //
                // r15 = captures_out buffer (layout: [start0, end0, start1, end1, ...])
                // r14 = current position in input
                // r12 = input length
                // rbx = input base pointer
                //
                // Slot for capture group `idx`:
                //   start_slot = idx * 2, offset = idx * 2 * 8
                //   end_slot = idx * 2 + 1, offset = idx * 2 * 8 + 8

                let idx = *idx as usize;
                let start_offset = (idx * 2 * 8) as i32;
                let end_offset = (idx * 2 * 8 + 8) as i32;

                let backref_match = self.asm.new_dynamic_label();

                dynasm!(self.asm
                    // Load capture start and end
                    ; mov r8, QWORD [r15 + start_offset]   // r8 = cap_start
                    ; mov r9, QWORD [r15 + end_offset]     // r9 = cap_end

                    // Check if capture is valid (both >= 0)
                    ; test r8, r8
                    ; js =>fail_label                      // cap_start < 0 means not captured
                    ; test r9, r9
                    ; js =>fail_label                      // cap_end < 0 means not captured

                    // Calculate capture length: r10 = cap_end - cap_start
                    ; mov r10, r9
                    ; sub r10, r8                          // r10 = cap_len

                    // Check if empty capture (length 0) - always matches
                    ; test r10, r10
                    ; jz =>backref_match

                    // Check if enough input remains: r14 + cap_len <= r12
                    ; mov rax, r14
                    ; add rax, r10
                    ; cmp rax, r12
                    ; jg =>fail_label                      // Not enough input

                    // Compare bytes using repe cmpsb
                    // rsi = source (captured text): input + cap_start
                    // rdi = dest (current pos): input + r14
                    // rcx = count: cap_len
                    // Note: We need to save/restore rsi, rdi since they're used for arguments

                    ; push rsi
                    ; push rdi

                    ; lea rsi, [rbx + r8]                  // rsi = input + cap_start
                    ; lea rdi, [rbx + r14]                 // rdi = input + current_pos
                    ; mov rcx, r10                         // rcx = cap_len

                    ; repe cmpsb                           // Compare strings

                    ; pop rdi
                    ; pop rsi

                    ; jne =>fail_label                     // Strings don't match

                    // Backref matched - advance position by cap_len
                    ; add r14, r10

                    ; =>backref_match
                );
            }
            PatternStep::StartOfText => {
                // Start of text: only matches at position 0
                // r14 = current position, r13 = start_pos
                // For anchored patterns, we only try matching at position 0
                dynasm!(self.asm
                    ; test r14, r14                 // r14 == 0?
                    ; jnz =>fail_label              // Fail if not at start
                );
            }
            PatternStep::EndOfText => {
                // End of text: only matches at position == input_len
                // r14 = current position, r12 = input_len
                dynasm!(self.asm
                    ; cmp r14, r12                  // r14 == r12?
                    ; jne =>fail_label              // Fail if not at end
                );
            }
            PatternStep::StartOfLine => {
                // Start of line: matches at position 0 OR after a newline
                // r14 = current position, rbx = input_ptr
                let at_start = self.asm.new_dynamic_label();
                dynasm!(self.asm
                    ; test r14, r14                 // r14 == 0?
                    ; jz =>at_start                 // At start of input - OK
                    // Check if previous byte is newline
                    ; mov rax, r14
                    ; dec rax                       // rax = r14 - 1
                    ; movzx eax, BYTE [rbx + rax]   // Load previous byte
                    ; cmp al, 0x0A                  // Is it '\n'?
                    ; jne =>fail_label              // Not after newline - fail
                    ; =>at_start
                );
            }
            PatternStep::EndOfLine => {
                // End of line: matches at position == input_len OR before a newline
                // r14 = current position, r12 = input_len, rbx = input_ptr
                let at_end = self.asm.new_dynamic_label();
                dynasm!(self.asm
                    ; cmp r14, r12                  // r14 == r12?
                    ; je =>at_end                   // At end of input - OK
                    // Check if current byte is newline
                    ; movzx eax, BYTE [rbx + r14]   // Load current byte
                    ; cmp al, 0x0A                  // Is it '\n'?
                    ; jne =>fail_label              // Not before newline - fail
                    ; =>at_end
                );
            }
            PatternStep::NonGreedyPlus(byte_class, suffix) => {
                // Non-greedy one-or-more in captures_fn
                let try_suffix = self.asm.new_dynamic_label();
                let consume_more = self.asm.new_dynamic_label();
                let suffix_matched = self.asm.new_dynamic_label();

                // Must match at least one
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14

                    ; =>try_suffix
                );

                // Try to match the suffix
                self.emit_non_greedy_suffix_check(suffix, consume_more, suffix_matched)?;

                dynasm!(self.asm
                    ; jmp =>suffix_matched

                    ; =>consume_more
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14
                    ; jmp =>try_suffix

                    ; =>suffix_matched
                );
            }
            PatternStep::NonGreedyStar(byte_class, suffix) => {
                // Non-greedy zero-or-more in captures_fn
                let try_suffix = self.asm.new_dynamic_label();
                let consume_more = self.asm.new_dynamic_label();
                let suffix_matched = self.asm.new_dynamic_label();

                dynasm!(self.asm
                    ; =>try_suffix
                );

                // Try to match the suffix
                self.emit_non_greedy_suffix_check(suffix, consume_more, suffix_matched)?;

                dynasm!(self.asm
                    ; jmp =>suffix_matched

                    ; =>consume_more
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(&byte_class.ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14
                    ; jmp =>try_suffix

                    ; =>suffix_matched
                );
            }
        }
        Ok(())
    }

    /// Returns a short name for a step type (for debugging).
    #[allow(dead_code)]
    fn step_name(step: &PatternStep) -> &'static str {
        match step {
            PatternStep::Byte(_) => "Byte",
            PatternStep::ByteClass(_) => "ByteClass",
            PatternStep::GreedyPlus(_) => "GreedyPlus",
            PatternStep::GreedyStar(_) => "GreedyStar",
            PatternStep::GreedyPlusLookahead(_, _, _) => "GreedyPlusLookahead",
            PatternStep::GreedyStarLookahead(_, _, _) => "GreedyStarLookahead",
            PatternStep::NonGreedyPlus(_, _) => "NonGreedyPlus",
            PatternStep::NonGreedyStar(_, _) => "NonGreedyStar",
            PatternStep::Alt(_) => "Alt",
            PatternStep::CaptureStart(_) => "CaptureStart",
            PatternStep::CaptureEnd(_) => "CaptureEnd",
            PatternStep::CodepointClass(_, _) => "CodepointClass",
            PatternStep::GreedyCodepointPlus(_) => "GreedyCodepointPlus",
            PatternStep::WordBoundary => "WordBoundary",
            PatternStep::NotWordBoundary => "NotWordBoundary",
            PatternStep::PositiveLookahead(_) => "PositiveLookahead",
            PatternStep::NegativeLookahead(_) => "NegativeLookahead",
            PatternStep::PositiveLookbehind(_, _) => "PositiveLookbehind",
            PatternStep::NegativeLookbehind(_, _) => "NegativeLookbehind",
            PatternStep::Backref(_) => "Backref",
            PatternStep::StartOfText => "StartOfText",
            PatternStep::EndOfText => "EndOfText",
            PatternStep::StartOfLine => "StartOfLine",
            PatternStep::EndOfLine => "EndOfLine",
        }
    }

    /// Calculates the minimum length of input needed to match a pattern.
    fn calc_min_len(steps: &[PatternStep]) -> usize {
        steps
            .iter()
            .map(|s| match s {
                PatternStep::Byte(_) | PatternStep::ByteClass(_) => 1,
                PatternStep::GreedyPlus(_) => 1,
                PatternStep::GreedyStar(_) => 0,
                // Greedy with lookahead: lookahead is zero-width, only repetition counts
                PatternStep::GreedyPlusLookahead(_, _, _) => 1,
                PatternStep::GreedyStarLookahead(_, _, _) => 0,
                // Non-greedy plus needs at least 1 char for the repetition + the suffix
                PatternStep::NonGreedyPlus(_, suffix) => {
                    1 + Self::calc_min_len(&[(**suffix).clone()])
                }
                // Non-greedy star needs 0 for the repetition + the suffix
                PatternStep::NonGreedyStar(_, suffix) => Self::calc_min_len(&[(**suffix).clone()]),
                PatternStep::Alt(alternatives) => {
                    // Minimum length is the minimum of all alternatives
                    alternatives
                        .iter()
                        .map(|alt| Self::calc_min_len(alt))
                        .min()
                        .unwrap_or(0)
                }
                // Capture markers don't consume input
                PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_) => 0,
                // Unicode codepoint classes consume at least 1 byte
                PatternStep::CodepointClass(_, _) => 1,
                // Greedy codepoint repetition consumes at least 1 byte (UTF-8 codepoint is 1-4 bytes)
                PatternStep::GreedyCodepointPlus(_) => 1,
                // Word boundaries don't consume input - they're zero-width assertions
                PatternStep::WordBoundary | PatternStep::NotWordBoundary => 0,
                // Lookarounds don't consume input - they're zero-width assertions
                PatternStep::PositiveLookahead(_)
                | PatternStep::NegativeLookahead(_)
                | PatternStep::PositiveLookbehind(_, _)
                | PatternStep::NegativeLookbehind(_, _) => 0,
                // Backrefs consume variable length (unknown at compile time, could be 0)
                PatternStep::Backref(_) => 0,
                // Anchors don't consume input - they're zero-width assertions
                PatternStep::StartOfText
                | PatternStep::EndOfText
                | PatternStep::StartOfLine
                | PatternStep::EndOfLine => 0,
            })
            .sum()
    }

    /// Combines greedy quantifiers (GreedyPlus/GreedyStar) followed by lookahead
    /// into special combined variants that support backtracking.
    ///
    /// When a greedy quantifier is followed by a lookahead, the greedy quantifier
    /// may consume characters that are needed by the lookahead. This function
    /// combines them into GreedyPlusLookahead/GreedyStarLookahead which emit
    /// code that backtracks when the lookahead fails.
    fn combine_greedy_with_lookahead(steps: Vec<PatternStep>) -> Vec<PatternStep> {
        let mut result = Vec::with_capacity(steps.len());
        let mut i = 0;

        while i < steps.len() {
            match &steps[i] {
                PatternStep::GreedyPlus(ranges) if i + 1 < steps.len() => {
                    // Check if followed by lookahead
                    match &steps[i + 1] {
                        PatternStep::PositiveLookahead(inner) => {
                            result.push(PatternStep::GreedyPlusLookahead(
                                ranges.clone(),
                                inner.clone(),
                                true, // positive
                            ));
                            i += 2; // Skip both steps
                            continue;
                        }
                        PatternStep::NegativeLookahead(inner) => {
                            result.push(PatternStep::GreedyPlusLookahead(
                                ranges.clone(),
                                inner.clone(),
                                false, // negative
                            ));
                            i += 2;
                            continue;
                        }
                        _ => {}
                    }
                }
                PatternStep::GreedyStar(ranges) if i + 1 < steps.len() => {
                    // Check if followed by lookahead
                    match &steps[i + 1] {
                        PatternStep::PositiveLookahead(inner) => {
                            result.push(PatternStep::GreedyStarLookahead(
                                ranges.clone(),
                                inner.clone(),
                                true,
                            ));
                            i += 2;
                            continue;
                        }
                        PatternStep::NegativeLookahead(inner) => {
                            result.push(PatternStep::GreedyStarLookahead(
                                ranges.clone(),
                                inner.clone(),
                                false,
                            ));
                            i += 2;
                            continue;
                        }
                        _ => {}
                    }
                }
                // Recursively process alternation branches
                PatternStep::Alt(alternatives) => {
                    let combined_alts: Vec<Vec<PatternStep>> = alternatives
                        .iter()
                        .map(|alt| Self::combine_greedy_with_lookahead(alt.clone()))
                        .collect();
                    result.push(PatternStep::Alt(combined_alts));
                    i += 1;
                    continue;
                }
                _ => {}
            }
            // No combination - keep original step
            result.push(steps[i].clone());
            i += 1;
        }

        result
    }

    /// Extracts the pattern as a sequence of match steps.
    ///
    /// Returns empty vec if the NFA cannot be JIT compiled.
    /// Supports:
    /// - Linear patterns (literals, character classes)
    /// - Simple greedy repetition (a+, [a-z]+)
    /// - Alternation (foo|bar)
    fn extract_pattern_steps(&self) -> Vec<PatternStep> {
        let mut visited = vec![false; self.nfa.states.len()];
        self.extract_from_state(self.nfa.start, &mut visited, None)
    }

    /// Extracts pattern steps starting from a given state.
    /// `end_state` is the optional target state where extraction should stop (for alternation branches).
    fn extract_from_state(
        &self,
        start: StateId,
        visited: &mut [bool],
        end_state: Option<StateId>,
    ) -> Vec<PatternStep> {
        let mut steps = Vec::new();
        let mut current = start;

        loop {
            // Check if we've reached the target end state
            if let Some(end) = end_state {
                if current == end {
                    break;
                }
            }

            let state = &self.nfa.states[current as usize];

            // Handle capture instructions first (they don't consume input)
            if let Some(ref instr) = state.instruction {
                match instr {
                    NfaInstruction::CaptureStart(idx) => {
                        steps.push(PatternStep::CaptureStart(*idx));
                    }
                    NfaInstruction::CaptureEnd(idx) => {
                        steps.push(PatternStep::CaptureEnd(*idx));
                    }
                    // Unicode codepoint class - can JIT with helper function
                    NfaInstruction::CodepointClass(cpclass, target) => {
                        // Check if target state forms a greedy loop (like \p{L}+)
                        let target_state = &self.nfa.states[*target as usize];
                        if target_state.epsilon.len() == 2 && target_state.transitions.is_empty() {
                            let eps0 = target_state.epsilon[0];
                            let eps1 = target_state.epsilon[1];

                            // Greedy loop: first epsilon goes back to current (loop), second goes forward
                            if eps0 == current {
                                steps.push(PatternStep::GreedyCodepointPlus(cpclass.clone()));
                                visited[current as usize] = true;
                                visited[*target as usize] = true;
                                current = eps1;
                                continue;
                            } else if eps1 == current {
                                // Alternative loop structure
                                steps.push(PatternStep::GreedyCodepointPlus(cpclass.clone()));
                                visited[current as usize] = true;
                                visited[*target as usize] = true;
                                current = eps0;
                                continue;
                            }
                        }
                        // Not a greedy loop - single codepoint class match
                        steps.push(PatternStep::CodepointClass(cpclass.clone(), *target));
                        // CodepointClass instruction consumes input and moves to target
                        current = *target;
                        continue;
                    }
                    // Backreferences - can be JIT'd
                    NfaInstruction::Backref(idx) => {
                        steps.push(PatternStep::Backref(*idx));
                        // Backref consumes variable-length input
                        // Follow epsilon transitions to next state
                        if state.epsilon.len() == 1 {
                            visited[current as usize] = true;
                            current = state.epsilon[0];
                            continue;
                        } else if state.epsilon.is_empty() && state.is_match {
                            // Backref at end of pattern
                            break;
                        } else {
                            // Complex epsilon structure after backref - fall back
                            return Vec::new();
                        }
                    }
                    // Lookarounds - extract inner pattern
                    NfaInstruction::PositiveLookahead(inner_nfa) => {
                        // Extract pattern steps from inner NFA
                        let inner_steps = self.extract_lookaround_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return Vec::new(); // Complex inner pattern
                        }
                        steps.push(PatternStep::PositiveLookahead(inner_steps));
                    }
                    NfaInstruction::NegativeLookahead(inner_nfa) => {
                        let inner_steps = self.extract_lookaround_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        steps.push(PatternStep::NegativeLookahead(inner_steps));
                    }
                    NfaInstruction::PositiveLookbehind(inner_nfa) => {
                        let inner_steps = self.extract_lookaround_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        let min_len = Self::calc_min_len(&inner_steps);
                        steps.push(PatternStep::PositiveLookbehind(inner_steps, min_len));
                    }
                    NfaInstruction::NegativeLookbehind(inner_nfa) => {
                        let inner_steps = self.extract_lookaround_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        let min_len = Self::calc_min_len(&inner_steps);
                        steps.push(PatternStep::NegativeLookbehind(inner_steps, min_len));
                    }
                    // Word boundaries - can be JIT'd
                    NfaInstruction::WordBoundary => {
                        steps.push(PatternStep::WordBoundary);
                    }
                    NfaInstruction::NotWordBoundary => {
                        steps.push(PatternStep::NotWordBoundary);
                    }
                    // Anchors - now supported
                    NfaInstruction::StartOfText => {
                        steps.push(PatternStep::StartOfText);
                    }
                    NfaInstruction::EndOfText => {
                        steps.push(PatternStep::EndOfText);
                    }
                    NfaInstruction::StartOfLine => {
                        steps.push(PatternStep::StartOfLine);
                    }
                    NfaInstruction::EndOfLine => {
                        steps.push(PatternStep::EndOfLine);
                    }
                    // Non-greedy exit marker - can skip
                    NfaInstruction::NonGreedyExit => {}
                }
            }

            // Match state - we're done
            if state.is_match {
                break;
            }

            // Handle byte transitions
            if !state.transitions.is_empty() {
                // Check if all transitions go to the same target
                let target = state.transitions[0].1;
                let all_same_target = state.transitions.iter().all(|(_, t)| *t == target);
                if !all_same_target {
                    return Vec::new(); // Different targets - can't JIT
                }

                // Extract the ranges for this step
                let ranges: Vec<ByteRange> = state.transitions.iter().map(|(r, _)| *r).collect();

                // Check if target state forms a loop (greedy or non-greedy)
                let target_state = &self.nfa.states[target as usize];
                if target_state.epsilon.len() == 2 && target_state.transitions.is_empty() {
                    let eps0 = target_state.epsilon[0];
                    let eps1 = target_state.epsilon[1];

                    // Greedy loop: first epsilon goes back to current (loop), second goes forward
                    if eps0 == current {
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        current = eps1;
                        visited[target as usize] = true;
                        continue;
                    }

                    // Non-greedy loop: eps0 goes to marker (NonGreedyExit), eps1 loops back
                    let marker_state = &self.nfa.states[eps0 as usize];
                    if eps1 == current
                        && marker_state.transitions.is_empty()
                        && marker_state.epsilon.len() == 1
                        && matches!(
                            marker_state.instruction,
                            Some(NfaInstruction::NonGreedyExit)
                        )
                    {
                        // Non-greedy plus pattern detected: a+?
                        // Extract the suffix (what comes after the quantifier)
                        let exit_state = marker_state.epsilon[0];
                        if let Some(suffix) = self.extract_single_step(exit_state) {
                            steps.push(PatternStep::NonGreedyPlus(
                                ByteClass::new(ranges),
                                Box::new(suffix),
                            ));
                            // Skip past the exit state and its suffix
                            visited[target as usize] = true;
                            visited[eps0 as usize] = true;
                            visited[exit_state as usize] = true;
                            current = self.advance_past_step(exit_state);
                            continue;
                        }
                        // Can't extract suffix - fall back to interpreter
                        return Vec::new();
                    }
                }

                // Not a loop - regular character class or literal
                if visited[current as usize] {
                    return Vec::new(); // Cycle without proper loop structure
                }
                visited[current as usize] = true;

                if ranges.len() == 1 && ranges[0].start == ranges[0].end {
                    steps.push(PatternStep::Byte(ranges[0].start));
                } else {
                    steps.push(PatternStep::ByteClass(ByteClass::new(ranges)));
                }
                current = target;
                continue;
            }

            // Single epsilon transition - just follow it
            if state.epsilon.len() == 1 && state.transitions.is_empty() {
                if visited[current as usize] {
                    return Vec::new(); // Cycle
                }
                visited[current as usize] = true;
                current = state.epsilon[0];
                continue;
            }

            // Multiple epsilon transitions = alternation or non-greedy star
            if state.epsilon.len() > 1 && state.transitions.is_empty() {
                // Check for non-greedy star pattern: a*?
                // Structure: eps[0] -> marker(NonGreedyExit) -> end, eps[1] -> pattern that loops back
                if state.epsilon.len() == 2 {
                    let eps0_state = &self.nfa.states[state.epsilon[0] as usize];

                    // Check if eps[0] is a NonGreedyExit marker
                    if eps0_state.transitions.is_empty()
                        && eps0_state.epsilon.len() == 1
                        && matches!(eps0_state.instruction, Some(NfaInstruction::NonGreedyExit))
                    {
                        // Non-greedy star detected: a*?
                        // eps[1] leads to the pattern that loops back
                        let pattern_start = state.epsilon[1];
                        let pattern_state = &self.nfa.states[pattern_start as usize];

                        // Extract the character ranges from the pattern
                        if !pattern_state.transitions.is_empty() {
                            let target = pattern_state.transitions[0].1;
                            let all_same_target =
                                pattern_state.transitions.iter().all(|(_, t)| *t == target);

                            if all_same_target {
                                let ranges: Vec<ByteRange> =
                                    pattern_state.transitions.iter().map(|(r, _)| *r).collect();

                                // Find the exit state (after the NonGreedyExit marker)
                                let exit_state = eps0_state.epsilon[0];

                                // Extract the suffix
                                if let Some(suffix) = self.extract_single_step(exit_state) {
                                    steps.push(PatternStep::NonGreedyStar(
                                        ByteClass::new(ranges),
                                        Box::new(suffix),
                                    ));
                                    visited[current as usize] = true;
                                    visited[state.epsilon[0] as usize] = true;
                                    visited[pattern_start as usize] = true;
                                    visited[exit_state as usize] = true;
                                    current = self.advance_past_step(exit_state);
                                    continue;
                                }
                            }
                        }
                        // Can't extract - fall back to interpreter
                        return Vec::new();
                    }
                }

                // Find the common end state for all alternatives
                // Each alternative should eventually reach a common merge point
                let common_end = self.find_alternation_end(current);
                if common_end.is_none() {
                    return Vec::new(); // Can't find common end - complex pattern
                }
                let common_end = common_end.unwrap();

                // Extract each alternative
                let mut alternatives = Vec::new();
                for &alt_start in &state.epsilon {
                    let mut alt_visited = visited.to_vec();
                    let alt_steps =
                        self.extract_from_state(alt_start, &mut alt_visited, Some(common_end));
                    if alt_steps.is_empty() && !self.is_trivial_path(alt_start, common_end) {
                        return Vec::new(); // Alternative too complex
                    }
                    alternatives.push(alt_steps);
                }

                steps.push(PatternStep::Alt(alternatives));
                visited[current as usize] = true;
                current = common_end;
                continue;
            }

            // No valid transition
            if state.transitions.is_empty() && state.epsilon.is_empty() {
                break;
            }

            // Complex pattern
            return Vec::new();
        }

        steps
    }

    /// Extracts pattern steps from a lookaround's inner NFA.
    /// Returns empty vec if the inner pattern is too complex for JIT.
    fn extract_lookaround_steps(&self, inner_nfa: &Nfa) -> Vec<PatternStep> {
        // Create a temporary compiler for the inner NFA to extract steps
        let mut visited = vec![false; inner_nfa.states.len()];
        let mut steps = Vec::new();
        let mut current = inner_nfa.start;

        loop {
            if current as usize >= inner_nfa.states.len() {
                return Vec::new();
            }

            let state = &inner_nfa.states[current as usize];

            // Check for match state
            if state.is_match {
                break;
            }

            // Handle instructions in lookaround
            if let Some(ref instr) = state.instruction {
                match instr {
                    NfaInstruction::WordBoundary => {
                        steps.push(PatternStep::WordBoundary);
                    }
                    NfaInstruction::EndOfText => {
                        steps.push(PatternStep::EndOfText);
                    }
                    NfaInstruction::StartOfText => {
                        steps.push(PatternStep::StartOfText);
                    }
                    _ => return Vec::new(),
                }
            }

            // Handle byte transitions
            if !state.transitions.is_empty() {
                let target = state.transitions[0].1;
                if !state.transitions.iter().all(|(_, t)| *t == target) {
                    return Vec::new(); // Different targets
                }

                let ranges: Vec<ByteRange> = state.transitions.iter().map(|(r, _)| *r).collect();

                // Check for greedy plus pattern: current -[byte]-> target -[eps]-> current (loop back)
                //                                              |-> next (exit)
                let target_state = &inner_nfa.states[target as usize];
                if target_state.transitions.is_empty() && target_state.epsilon.len() == 2 {
                    let eps0 = target_state.epsilon[0];
                    let eps1 = target_state.epsilon[1];

                    // Check if one epsilon leads back to current (greedy loop)
                    if eps0 == current {
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        if visited[target as usize] {
                            return Vec::new();
                        }
                        visited[target as usize] = true;
                        current = eps1; // Continue from exit path
                        continue;
                    } else if eps1 == current {
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        if visited[target as usize] {
                            return Vec::new();
                        }
                        visited[target as usize] = true;
                        current = eps0; // Continue from exit path
                        continue;
                    }
                }

                if visited[current as usize] {
                    return Vec::new(); // Cycle
                }
                visited[current as usize] = true;

                if ranges.len() == 1 && ranges[0].start == ranges[0].end {
                    steps.push(PatternStep::Byte(ranges[0].start));
                } else {
                    steps.push(PatternStep::ByteClass(ByteClass::new(ranges)));
                }
                current = target;
                continue;
            }

            // Handle single epsilon transition
            if state.epsilon.len() == 1 && state.transitions.is_empty() {
                if visited[current as usize] {
                    return Vec::new(); // Cycle
                }
                visited[current as usize] = true;
                current = state.epsilon[0];
                continue;
            }

            // Handle greedy star: two epsilons where one leads to byte transitions
            // that loop back, and one leads to exit
            if state.epsilon.len() == 2 && state.transitions.is_empty() {
                let eps0 = state.epsilon[0];
                let eps1 = state.epsilon[1];

                // Try to detect: eps0 has transitions that loop, eps1 exits
                // or vice versa
                if let Some((ranges, exit_state)) =
                    self.detect_greedy_star_lookaround(inner_nfa, current, eps0, eps1, &visited)
                {
                    steps.push(PatternStep::GreedyStar(ByteClass::new(ranges)));
                    visited[current as usize] = true;
                    current = exit_state;
                    continue;
                }
                if let Some((ranges, exit_state)) =
                    self.detect_greedy_star_lookaround(inner_nfa, current, eps1, eps0, &visited)
                {
                    steps.push(PatternStep::GreedyStar(ByteClass::new(ranges)));
                    visited[current as usize] = true;
                    current = exit_state;
                    continue;
                }

                // Not a recognized greedy star pattern
                return Vec::new();
            }

            // Complex structure - not supported
            if !state.epsilon.is_empty() || !state.transitions.is_empty() {
                return Vec::new();
            }

            break;
        }

        steps
    }

    /// Detects a greedy star pattern in a lookaround's inner NFA.
    fn detect_greedy_star_lookaround(
        &self,
        inner_nfa: &Nfa,
        branch_state: StateId,
        loop_start: StateId,
        exit_state: StateId,
        visited: &[bool],
    ) -> Option<(Vec<ByteRange>, StateId)> {
        if loop_start as usize >= inner_nfa.states.len() {
            return None;
        }

        let loop_state = &inner_nfa.states[loop_start as usize];

        // The loop state should have byte transitions
        if loop_state.transitions.is_empty() {
            return None;
        }

        // All transitions should go to the same target
        let target = loop_state.transitions[0].1;
        if !loop_state.transitions.iter().all(|(_, t)| *t == target) {
            return None;
        }

        let ranges: Vec<ByteRange> = loop_state.transitions.iter().map(|(r, _)| *r).collect();

        // The target should have epsilon back to branch_state (completing the loop)
        let target_state = &inner_nfa.states[target as usize];

        // Simple case: target has single epsilon back to branch_state or loop_start
        if target_state.epsilon.len() == 1 {
            let back_to = target_state.epsilon[0];
            if (back_to == branch_state || back_to == loop_start) && !visited[loop_start as usize] {
                return Some((ranges, exit_state));
            }
        }

        // Alternative: target has two epsilons, one back to branch_state or loop_start
        if target_state.epsilon.len() == 2 {
            let eps0 = target_state.epsilon[0];
            let eps1 = target_state.epsilon[1];

            // Check if one epsilon goes back (to branch_state or loop_start) and one goes forward (to exit)
            let (back, fwd) = if eps0 == branch_state || eps0 == loop_start {
                (eps0, eps1)
            } else if eps1 == branch_state || eps1 == loop_start {
                (eps1, eps0)
            } else {
                return None;
            };
            let _ = back; // suppress warning

            // The forward epsilon should lead to the same as exit_state or be exit_state
            if fwd == exit_state && !visited[loop_start as usize] {
                return Some((ranges, exit_state));
            }
        }

        None
    }

    /// Finds the common end state for an alternation starting at `start`.
    /// Returns None if no common end is found.
    fn find_alternation_end(&self, start: StateId) -> Option<StateId> {
        self.find_alternation_end_with_depth(start, 0)
    }

    fn find_alternation_end_with_depth(&self, start: StateId, depth: usize) -> Option<StateId> {
        // Limit recursion depth to prevent stack overflow on deeply nested patterns
        if depth > 20 {
            return None;
        }

        let state = &self.nfa.states[start as usize];
        if state.epsilon.len() < 2 {
            return None;
        }

        // For each alternative, find where it ends up
        // All alternatives should converge to the same state
        let mut end_states: Vec<StateId> = Vec::new();

        for &alt_start in &state.epsilon {
            if let Some(end) = self.trace_to_merge_point_with_depth(alt_start, start, depth) {
                end_states.push(end);
            } else {
                return None;
            }
        }

        // All end states should be the same
        if end_states.is_empty() {
            return None;
        }
        let first = end_states[0];
        if end_states.iter().all(|&e| e == first) {
            Some(first)
        } else {
            None
        }
    }

    /// Traces from a state to find where it merges (state with incoming epsilon from multiple sources).
    /// Returns the merge point state ID or None.
    fn trace_to_merge_point_with_depth(
        &self,
        start: StateId,
        alt_start: StateId,
        depth: usize,
    ) -> Option<StateId> {
        // Limit recursion depth to prevent stack overflow
        if depth > 20 {
            return None;
        }

        let mut current = start;
        let mut visited = vec![false; self.nfa.states.len()];
        visited[alt_start as usize] = true; // Don't go back to alternation start

        for _ in 0..200 {
            // Limit iterations (increased for nested patterns)
            if visited[current as usize] {
                return None; // Cycle
            }
            visited[current as usize] = true;

            let state = &self.nfa.states[current as usize];

            // Match state
            if state.is_match {
                return Some(current);
            }

            // CodepointClass instruction has its target embedded in the instruction
            if let Some(NfaInstruction::CodepointClass(_, target)) = &state.instruction {
                current = *target;
                continue;
            }

            // State with no outgoing transitions (and no CodepointClass) - end point
            if state.transitions.is_empty() && state.epsilon.is_empty() {
                return Some(current);
            }

            // State with single epsilon going forward
            if state.epsilon.len() == 1 && state.transitions.is_empty() {
                current = state.epsilon[0];
                continue;
            }

            // State with byte transitions only - follow them
            if !state.transitions.is_empty() && state.epsilon.is_empty() {
                let target = state.transitions[0].1;
                current = target;
                continue;
            }

            // State with byte transitions AND single epsilon - follow transitions
            if !state.transitions.is_empty() && state.epsilon.len() == 1 {
                let target = state.transitions[0].1;
                current = target;
                continue;
            }

            // State with multiple epsilons - could be alternation or greedy loop
            if state.epsilon.len() >= 2 && state.transitions.is_empty() {
                // Check if this is a greedy loop (one epsilon goes back to visited state)
                let mut forward_eps: Vec<StateId> = Vec::new();
                for &eps in &state.epsilon {
                    if visited[eps as usize] {
                        // This epsilon loops back - it's a greedy quantifier, skip it
                        continue;
                    }
                    forward_eps.push(eps);
                }

                // If only one forward epsilon, follow it (greedy loop exit)
                if forward_eps.len() == 1 {
                    current = forward_eps[0];
                    continue;
                }

                // Multiple forward epsilons - this is a nested alternation
                if let Some(nested_end) = self.find_alternation_end_with_depth(current, depth + 1) {
                    current = nested_end;
                    continue;
                }
                return None;
            }

            // Complex - can't determine
            return None;
        }

        None
    }

    /// Checks if there's a trivial (empty) path from start to end.
    fn is_trivial_path(&self, start: StateId, end: StateId) -> bool {
        self.is_trivial_path_with_depth(start, end, 0)
    }

    fn is_trivial_path_with_depth(&self, start: StateId, end: StateId, depth: usize) -> bool {
        // Limit recursion depth to prevent stack overflow
        if depth > 100 {
            return false;
        }
        if start == end {
            return true;
        }
        let state = &self.nfa.states[start as usize];
        if state.epsilon.len() == 1 && state.transitions.is_empty() {
            return state.epsilon[0] == end
                || self.is_trivial_path_with_depth(state.epsilon[0], end, depth + 1);
        }
        false
    }

    /// Extracts a single pattern step from a state (for non-greedy suffix extraction).
    /// Returns None if the state doesn't represent a simple step (byte or ranges).
    fn extract_single_step(&self, state_id: StateId) -> Option<PatternStep> {
        let mut current = state_id;

        // Follow epsilon transitions to find actual content
        loop {
            let state = &self.nfa.states[current as usize];

            // Check for byte transitions
            if !state.transitions.is_empty() {
                // All transitions must go to the same target
                let target = state.transitions[0].1;
                if !state.transitions.iter().all(|(_, t)| *t == target) {
                    return None;
                }

                let ranges: Vec<ByteRange> = state.transitions.iter().map(|(r, _)| *r).collect();

                return if ranges.len() == 1 && ranges[0].start == ranges[0].end {
                    Some(PatternStep::Byte(ranges[0].start))
                } else {
                    Some(PatternStep::ByteClass(ByteClass::new(ranges)))
                };
            }

            // Follow single epsilon transition
            if state.epsilon.len() == 1 && state.transitions.is_empty() {
                current = state.epsilon[0];
                continue;
            }

            // Match state or complex structure - can't extract simple step
            if state.is_match || state.epsilon.len() > 1 {
                return None;
            }

            return None;
        }
    }

    /// Advances past a single step and returns the next state to continue from.
    /// Used after extracting a suffix for non-greedy quantifiers.
    fn advance_past_step(&self, state_id: StateId) -> StateId {
        let mut current = state_id;

        // Follow epsilon transitions to find actual content
        loop {
            let state = &self.nfa.states[current as usize];

            // If we have byte transitions, advance to their target
            if !state.transitions.is_empty() {
                return state.transitions[0].1;
            }

            // Follow single epsilon transition
            if state.epsilon.len() == 1 {
                current = state.epsilon[0];
                continue;
            }

            // Match state or complex structure
            return current;
        }
    }

    /// Finalizes compilation and returns the TaggedNfaJit.
    fn finalize(
        self,
        find_offset: dynasmrt::AssemblyOffset,
        captures_offset: dynasmrt::AssemblyOffset,
        find_needs_ctx: bool,
        fallback_steps: Option<Vec<PatternStep>>,
    ) -> Result<TaggedNfaJit> {
        let code = self.asm.finalize().map_err(|e| {
            Error::new(
                ErrorKind::Jit(format!("Failed to finalize JIT code: {:?}", e)),
                "",
            )
        })?;

        // Get function pointers with platform-specific calling convention
        #[cfg(target_os = "windows")]
        let find_fn: unsafe extern "win64" fn(
            *const u8,
            usize,
            *mut TaggedNfaContext,
        ) -> i64 = unsafe { std::mem::transmute(code.ptr(find_offset)) };

        #[cfg(not(target_os = "windows"))]
        let find_fn: unsafe extern "sysv64" fn(
            *const u8,
            usize,
            *mut TaggedNfaContext,
        ) -> i64 = unsafe { std::mem::transmute(code.ptr(find_offset)) };

        #[cfg(target_os = "windows")]
        let captures_fn: unsafe extern "win64" fn(
            *const u8,
            usize,
            *mut TaggedNfaContext,
            *mut i64,
        ) -> i64 = unsafe { std::mem::transmute(code.ptr(captures_offset)) };

        #[cfg(not(target_os = "windows"))]
        let captures_fn: unsafe extern "sysv64" fn(
            *const u8,
            usize,
            *mut TaggedNfaContext,
            *mut i64,
        ) -> i64 = unsafe { std::mem::transmute(code.ptr(captures_offset)) };

        let capture_count = self.nfa.capture_count;
        let state_count = self.nfa.states.len();
        let lookaround_count = self.liveness.lookaround_count;
        let stride = (capture_count as usize + 1) * 2;

        Ok(TaggedNfaJit::new(
            code,
            find_fn,
            captures_fn,
            self.liveness,
            self.nfa,
            capture_count,
            state_count,
            lookaround_count,
            stride,
            self.codepoint_classes,
            self.lookaround_nfas,
            find_needs_ctx,
            fallback_steps,
        ))
    }
}
