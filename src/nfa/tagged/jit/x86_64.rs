//! x86-64 JIT code generation for Tagged NFA.
//!
//! This module contains the TaggedNfaJitCompiler which generates x86-64 assembly
//! code for Thompson NFA simulation with captures.

use crate::error::{Error, ErrorKind, Result};
use crate::hir::CodepointClass;
use crate::nfa::{ByteRange, Nfa, NfaInstruction, StateId};

use super::super::{NfaLiveness, TaggedNfaContext, PatternStep};
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
    codepoint_classes: Vec<Box<CodepointClass>>,
    /// Lookaround NFAs collected during pattern extraction.
    /// Boxed to ensure stable addresses for JIT helper function references.
    lookaround_nfas: Vec<Box<Nfa>>,
}

impl TaggedNfaJitCompiler {
    /// Creates a new compiler for the given NFA.
    #[allow(dead_code)]
    #[allow(unused_imports)]
    fn new(nfa: Nfa, liveness: NfaLiveness) -> Result<Self> {
        use dynasmrt::DynasmLabelApi;

        let mut asm = dynasmrt::x64::Assembler::new().map_err(|e| {
            Error::new(ErrorKind::Jit(format!("Failed to create assembler: {:?}", e)), "")
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
                    | NfaInstruction::NegativeLookbehind(_) => {},
                    // Anchors are now supported
                    NfaInstruction::StartOfText
                    | NfaInstruction::EndOfText
                    | NfaInstruction::StartOfLine
                    | NfaInstruction::EndOfLine => {},
                    // Backref is now supported - handled in compile_full()
                    NfaInstruction::Backref(_) => {},
                    // Word boundary and non-greedy are handled by pattern extraction
                    NfaInstruction::WordBoundary
                    | NfaInstruction::NotWordBoundary
                    | NfaInstruction::NonGreedyExit => {},
                    // Capture instructions are supported
                    NfaInstruction::CaptureStart(_)
                    | NfaInstruction::CaptureEnd(_) => {},
                    // Codepoint class is handled by pattern extraction
                    NfaInstruction::CodepointClass(_, _) => {},
                }
            }
        }

        // Large NFAs generate too much code
        if self.nfa.states.len() > 256 {
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
    /// If `steps` is provided, they will be used for fast StepInterpreter fallback.
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
        // Pass steps for fast StepInterpreter fallback
        self.finalize(find_offset, captures_offset, false, steps)
    }

    /// Generates JIT code for simple linear patterns (literals).
    ///
    /// For a pattern like "abc", generates code that:
    /// 1. Tries each starting position
    /// 2. At each position, walks the linear NFA chain
    /// 3. Returns on first match
    ///
    /// Register allocation (System V AMD64 ABI):
    /// - rdi = input_ptr (argument, then scratch)
    /// - rsi = input_len (argument)
    /// - rbx = input_ptr (callee-saved)
    /// - r12 = input_len (callee-saved)
    /// - r13 = start_pos for current attempt (callee-saved)
    /// - r14 = current_pos (absolute position in input) (callee-saved)
    /// - rax = scratch / return value
    /// Check if pattern contains backreferences (recursively in alternations).
    fn has_backref(steps: &[PatternStep]) -> bool {
        steps.iter().any(|s| match s {
            PatternStep::Backref(_) => true,
            PatternStep::Alt(alternatives) => alternatives.iter().any(|alt| Self::has_backref(alt)),
            _ => false,
        })
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

        // Prologue
        dynasm!(self.asm
            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15
        );

        // Set up registers
        // rdi = input_ptr, rsi = input_len, rdx = ctx (unused)
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
        for step in steps.iter() {
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
                PatternStep::Ranges(ranges) => {
                    // Check bounds
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>byte_mismatch
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(ranges, byte_mismatch)?;
                    dynasm!(self.asm
                        ; inc r14                   // Advance position
                    );
                }
                PatternStep::GreedyPlus(ranges) => {
                    // Greedy one-or-more: must match at least once, then keep matching
                    let loop_start = self.asm.new_dynamic_label();
                    let loop_done = self.asm.new_dynamic_label();

                    // First iteration (must match)
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>byte_mismatch       // Must have at least one byte
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(ranges, byte_mismatch)?;
                    dynasm!(self.asm
                        ; inc r14                   // Consumed first byte

                        // Loop for additional matches
                        ; =>loop_start
                        ; cmp r14, r12
                        ; jge =>loop_done           // End of input - done looping
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(ranges, loop_done)?;
                    dynasm!(self.asm
                        ; inc r14                   // Consumed another byte
                        ; jmp =>loop_start
                        ; =>loop_done
                    );
                }
                PatternStep::GreedyStar(ranges) => {
                    // Greedy zero-or-more: keep matching as long as possible
                    let loop_start = self.asm.new_dynamic_label();
                    let loop_done = self.asm.new_dynamic_label();

                    dynasm!(self.asm
                        ; =>loop_start
                        ; cmp r14, r12
                        ; jge =>loop_done           // End of input - done looping
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(ranges, loop_done)?;
                    dynasm!(self.asm
                        ; inc r14                   // Consumed a byte
                        ; jmp =>loop_start
                        ; =>loop_done
                    );
                }
                PatternStep::NonGreedyPlus(ranges, suffix) => {
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
                    self.emit_range_check(ranges, byte_mismatch)?;
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
                    self.emit_range_check(ranges, byte_mismatch)?;
                    dynasm!(self.asm
                        ; inc r14                   // Consumed another byte
                        ; jmp =>try_suffix

                        ; =>suffix_matched
                    );
                }
                PatternStep::NonGreedyStar(ranges, suffix) => {
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
                    self.emit_range_check(ranges, byte_mismatch)?;
                    dynasm!(self.asm
                        ; inc r14                   // Consumed another byte
                        ; jmp =>try_suffix

                        ; =>suffix_matched
                    );
                }
                PatternStep::GreedyPlusLookahead(ranges, lookahead_steps, is_positive) => {
                    // Greedy one-or-more with lookahead: greedily consume, then backtrack
                    // until the lookahead succeeds.
                    self.emit_greedy_plus_with_lookahead(ranges, lookahead_steps, *is_positive, byte_mismatch)?;
                }
                PatternStep::GreedyStarLookahead(ranges, lookahead_steps, is_positive) => {
                    // Greedy zero-or-more with lookahead: greedily consume, then backtrack
                    // until the lookahead succeeds.
                    self.emit_greedy_star_with_lookahead(ranges, lookahead_steps, *is_positive, byte_mismatch)?;
                }
                PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_) => {
                    // Capture markers don't consume input - skip in find_fn
                    // (captures are handled separately in captures_fn)
                }
                PatternStep::CodepointClass(_, _) => {
                    // Unicode codepoint classes require helper function - fall back to interpreter for now
                    return self.compile_with_fallback(None);
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
                    // Alternation: try each alternative in order
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
                                PatternStep::Ranges(ranges) => {
                                    dynasm!(self.asm
                                        ; cmp r14, r12
                                        ; jge =>try_next_alt
                                        ; movzx eax, BYTE [rbx + r14]
                                    );
                                    self.emit_range_check(ranges, try_next_alt)?;
                                    dynasm!(self.asm
                                        ; inc r14
                                    );
                                }
                                PatternStep::GreedyPlus(ranges) => {
                                    let loop_start = self.asm.new_dynamic_label();
                                    let loop_done = self.asm.new_dynamic_label();

                                    dynasm!(self.asm
                                        ; cmp r14, r12
                                        ; jge =>try_next_alt
                                        ; movzx eax, BYTE [rbx + r14]
                                    );
                                    self.emit_range_check(ranges, try_next_alt)?;
                                    dynasm!(self.asm
                                        ; inc r14
                                        ; =>loop_start
                                        ; cmp r14, r12
                                        ; jge =>loop_done
                                        ; movzx eax, BYTE [rbx + r14]
                                    );
                                    self.emit_range_check(ranges, loop_done)?;
                                    dynasm!(self.asm
                                        ; inc r14
                                        ; jmp =>loop_start
                                        ; =>loop_done
                                    );
                                }
                                PatternStep::GreedyStar(ranges) => {
                                    let loop_start = self.asm.new_dynamic_label();
                                    let loop_done = self.asm.new_dynamic_label();

                                    dynasm!(self.asm
                                        ; =>loop_start
                                        ; cmp r14, r12
                                        ; jge =>loop_done
                                        ; movzx eax, BYTE [rbx + r14]
                                    );
                                    self.emit_range_check(ranges, loop_done)?;
                                    dynasm!(self.asm
                                        ; inc r14
                                        ; jmp =>loop_start
                                        ; =>loop_done
                                    );
                                }
                                PatternStep::Alt(_) => {
                                    // Nested alternation - fall back to interpreter
                                    return self.compile_with_fallback(None);
                                }
                                PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_) => {
                                    // Capture markers in alternation - skip in find_fn
                                }
                                PatternStep::CodepointClass(_, _) => {
                                    // Unicode codepoint classes in alternation - fall back to interpreter
                                    return self.compile_with_fallback(None);
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
                                PatternStep::PositiveLookahead(_) |
                                PatternStep::NegativeLookahead(_) |
                                PatternStep::PositiveLookbehind(..) |
                                PatternStep::NegativeLookbehind(..) => {
                                    // Lookarounds in alternation - fall back to interpreter
                                    return self.compile_with_fallback(None);
                                }
                                PatternStep::Backref(_) => {
                                    // Backrefs in find_fn are handled by early return above
                                    unreachable!("Backref in find_fn alternation should have triggered early return");
                                }
                                PatternStep::NonGreedyPlus(_, _) | PatternStep::NonGreedyStar(_, _) => {
                                    // Non-greedy in alternation - complex, fall back to interpreter
                                    return self.compile_with_fallback(None);
                                }
                                PatternStep::GreedyPlusLookahead(_, _, _) | PatternStep::GreedyStarLookahead(_, _, _) => {
                                    // Greedy with lookahead in alternation - complex, fall back
                                    return self.compile_with_fallback(None);
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

            // Epilogue
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

            // Epilogue
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
        let has_captures = steps.iter().any(|s| matches!(s, PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_)));

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
        // Store steps for find_at to use StepInterpreter (JIT doesn't support start offset yet)
        self.finalize(find_offset, captures_offset, false, Some(steps))
    }

    /// Emits range check code for character classes.
    /// Jumps to `fail_label` if the byte in `al` doesn't match any range.
    fn emit_range_check(&mut self, ranges: &[ByteRange], fail_label: dynasmrt::DynamicLabel) -> Result<()> {
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
            PatternStep::Ranges(ranges) => {
                // Check bounds
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label           // Not enough input
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14                    // Consume the suffix byte
                );
            }
            _ => {
                // Other suffix types not supported - shouldn't reach here
                // since extract_single_step only returns Byte or Ranges
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
    fn emit_is_word_char(&mut self, word_char_label: dynasmrt::DynamicLabel, not_word_char_label: dynasmrt::DynamicLabel) {
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
    fn emit_word_boundary_check(&mut self, fail_label: dynasmrt::DynamicLabel, is_boundary: bool) -> Result<()> {
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
                PatternStep::Ranges(inner_ranges) => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>lookahead_inner_mismatch
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(inner_ranges, lookahead_inner_mismatch)?;
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
            // Negative lookahead: inner match means assertion fails
            dynasm!(self.asm
                ; mov r14, r10             // Restore position
                ; jmp =>lookahead_failed   // Inner matched = neg lookahead fails
                ; =>lookahead_inner_match
                ; mov r14, r10             // Restore position
                ; jmp =>success            // Inner didn't match = neg lookahead succeeds
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
                PatternStep::Ranges(inner_ranges) => {
                    dynasm!(self.asm
                        ; cmp r14, r12
                        ; jge =>lookahead_inner_mismatch
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(inner_ranges, lookahead_inner_mismatch)?;
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
            dynasm!(self.asm
                ; mov r14, r10
                ; jmp =>lookahead_failed
                ; =>lookahead_inner_match
                ; mov r14, r10
                ; jmp =>success
            );
        } else {
            dynasm!(self.asm
                ; mov r14, r10
                ; jmp =>lookahead_failed
                ; =>lookahead_inner_match
                ; mov r14, r10
                ; jmp =>success
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
                PatternStep::Ranges(ranges) => {
                    // Check bounds
                    if positive {
                        dynasm!(self.asm
                            ; cmp r9, r12
                            ; jge =>fail_label
                            ; movzx eax, BYTE [rbx + r9]
                        );
                        // Use a temp label for range check failure
                        let range_fail = self.asm.new_dynamic_label();
                        self.emit_range_check_with_label(ranges, range_fail, fail_label)?;
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
                        self.emit_range_check_with_label(ranges, range_fail, inner_match)?;
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
                PatternStep::Ranges(ranges) => {
                    dynasm!(self.asm
                        ; movzx eax, BYTE [rbx + r14]
                    );
                    self.emit_range_check(ranges, inner_mismatch)?;
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
        let max_capture_idx = steps.iter().filter_map(|s| match s {
            PatternStep::CaptureStart(idx) | PatternStep::CaptureEnd(idx) => Some(*idx),
            _ => None,
        }).max().unwrap_or(0);

        // Number of slots: (max_capture_idx + 1) groups * 2 slots each
        // This includes group 0 (full match) since max_capture_idx >= 1 for patterns with captures
        // E.g., for capture group 1: max_capture_idx=1, num_slots = (1+1)*2 = 4
        let num_slots = (max_capture_idx as usize + 1) * 2;

        // Prologue - save callee-saved registers
        // On function entry: RSP is 8-mod-16 (return address pushed)
        // 5 pushes = 40 bytes -> RSP is (8+40) = 48 = 0 mod 16 -> aligned!
        // No sub rsp needed for alignment
        dynasm!(self.asm
            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15
        );

        // Set up registers
        // rdi = input_ptr, rsi = input_len, rdx = ctx (unused), rcx = captures_out
        dynasm!(self.asm
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
    fn emit_capture_step(&mut self, step: &PatternStep, fail_label: dynasmrt::DynamicLabel, _stack_align: i32) -> Result<()> {
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
            PatternStep::Ranges(ranges) => {
                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14
                );
            }
            PatternStep::GreedyPlus(ranges) => {
                let loop_start = self.asm.new_dynamic_label();
                let loop_done = self.asm.new_dynamic_label();

                dynasm!(self.asm
                    ; cmp r14, r12
                    ; jge =>fail_label
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14
                    ; =>loop_start
                    ; cmp r14, r12
                    ; jge =>loop_done
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(ranges, loop_done)?;
                dynasm!(self.asm
                    ; inc r14
                    ; jmp =>loop_start
                    ; =>loop_done
                );
            }
            PatternStep::GreedyStar(ranges) => {
                let loop_start = self.asm.new_dynamic_label();
                let loop_done = self.asm.new_dynamic_label();

                dynasm!(self.asm
                    ; =>loop_start
                    ; cmp r14, r12
                    ; jge =>loop_done
                    ; movzx eax, BYTE [rbx + r14]
                );
                self.emit_range_check(ranges, loop_done)?;
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
                        alt_fail  // Jump to our local fail handler that cleans up stack
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
            PatternStep::CodepointClass(_, _) => {
                // Unicode codepoint classes in captures - fall back to interpreter
                // This shouldn't normally be reached since extract_pattern_steps handles these
                // by continuing to the target state, but we need to handle the case
                // Return an error that will trigger interpreter fallback
                return Err(Error::new(
                    ErrorKind::Jit("CodepointClass in captures not supported yet".to_string()),
                    "",
                ));
            }
            PatternStep::WordBoundary => {
                // Word boundary assertion in captures - doesn't consume input
                self.emit_word_boundary_check(fail_label, true)?;
            }
            PatternStep::NotWordBoundary => {
                // Not word boundary assertion in captures - doesn't consume input
                self.emit_word_boundary_check(fail_label, false)?;
            }
            PatternStep::PositiveLookahead(_) |
            PatternStep::NegativeLookahead(_) |
            PatternStep::PositiveLookbehind(..) |
            PatternStep::NegativeLookbehind(..) => {
                // Lookarounds in captures - fall back to interpreter
                return Err(Error::new(
                    ErrorKind::Jit("Lookarounds in captures not supported yet".to_string()),
                    "",
                ));
            }
            PatternStep::GreedyPlusLookahead(_, _, _) |
            PatternStep::GreedyStarLookahead(_, _, _) => {
                // Greedy with lookahead in captures - fall back to interpreter
                return Err(Error::new(
                    ErrorKind::Jit("Greedy with lookahead in captures not supported yet".to_string()),
                    "",
                ));
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
            PatternStep::NonGreedyPlus(ranges, suffix) => {
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
                self.emit_range_check(ranges, fail_label)?;
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
                self.emit_range_check(ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14
                    ; jmp =>try_suffix

                    ; =>suffix_matched
                );
            }
            PatternStep::NonGreedyStar(ranges, suffix) => {
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
                self.emit_range_check(ranges, fail_label)?;
                dynasm!(self.asm
                    ; inc r14
                    ; jmp =>try_suffix

                    ; =>suffix_matched
                );
            }
        }
        Ok(())
    }

    /// Calculates the minimum length of input needed to match a pattern.
    fn calc_min_len(steps: &[PatternStep]) -> usize {
        steps.iter().map(|s| match s {
            PatternStep::Byte(_) | PatternStep::Ranges(_) => 1,
            PatternStep::GreedyPlus(_) => 1,
            PatternStep::GreedyStar(_) => 0,
            // Greedy with lookahead: lookahead is zero-width, only repetition counts
            PatternStep::GreedyPlusLookahead(_, _, _) => 1,
            PatternStep::GreedyStarLookahead(_, _, _) => 0,
            // Non-greedy plus needs at least 1 char for the repetition + the suffix
            PatternStep::NonGreedyPlus(_, suffix) => 1 + Self::calc_min_len(&[(**suffix).clone()]),
            // Non-greedy star needs 0 for the repetition + the suffix
            PatternStep::NonGreedyStar(_, suffix) => Self::calc_min_len(&[(**suffix).clone()]),
            PatternStep::Alt(alternatives) => {
                // Minimum length is the minimum of all alternatives
                alternatives.iter()
                    .map(|alt| Self::calc_min_len(alt))
                    .min()
                    .unwrap_or(0)
            }
            // Capture markers don't consume input
            PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_) => 0,
            // Unicode codepoint classes consume at least 1 byte
            PatternStep::CodepointClass(_, _) => 1,
            // Word boundaries don't consume input - they're zero-width assertions
            PatternStep::WordBoundary | PatternStep::NotWordBoundary => 0,
            // Lookarounds don't consume input - they're zero-width assertions
            PatternStep::PositiveLookahead(_) |
            PatternStep::NegativeLookahead(_) |
            PatternStep::PositiveLookbehind(_, _) |
            PatternStep::NegativeLookbehind(_, _) => 0,
            // Backrefs consume variable length (unknown at compile time, could be 0)
            PatternStep::Backref(_) => 0,
            // Anchors don't consume input - they're zero-width assertions
            PatternStep::StartOfText | PatternStep::EndOfText |
            PatternStep::StartOfLine | PatternStep::EndOfLine => 0,
        }).sum()
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
        use crate::nfa::NfaInstruction;

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
                let ranges: Vec<ByteRange> = state.transitions.iter()
                    .map(|(r, _)| r.clone())
                    .collect();

                // Check if target state forms a loop (greedy or non-greedy)
                let target_state = &self.nfa.states[target as usize];
                if target_state.epsilon.len() == 2 && target_state.transitions.is_empty() {
                    let eps0 = target_state.epsilon[0];
                    let eps1 = target_state.epsilon[1];

                    // Greedy loop: first epsilon goes back to current (loop), second goes forward
                    if eps0 == current {
                        steps.push(PatternStep::GreedyPlus(ranges));
                        current = eps1;
                        visited[target as usize] = true;
                        continue;
                    }

                    // Non-greedy loop: eps0 goes to marker (NonGreedyExit), eps1 loops back
                    let marker_state = &self.nfa.states[eps0 as usize];
                    if eps1 == current
                        && marker_state.transitions.is_empty()
                        && marker_state.epsilon.len() == 1
                        && matches!(marker_state.instruction, Some(NfaInstruction::NonGreedyExit))
                    {
                        // Non-greedy plus pattern detected: a+?
                        // Extract the suffix (what comes after the quantifier)
                        let exit_state = marker_state.epsilon[0];
                        if let Some(suffix) = self.extract_single_step(exit_state) {
                            steps.push(PatternStep::NonGreedyPlus(ranges, Box::new(suffix)));
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
                    steps.push(PatternStep::Ranges(ranges));
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
                            let all_same_target = pattern_state.transitions.iter().all(|(_, t)| *t == target);

                            if all_same_target {
                                let ranges: Vec<ByteRange> = pattern_state.transitions.iter()
                                    .map(|(r, _)| r.clone())
                                    .collect();

                                // Find the exit state (after the NonGreedyExit marker)
                                let exit_state = eps0_state.epsilon[0];

                                // Extract the suffix
                                if let Some(suffix) = self.extract_single_step(exit_state) {
                                    steps.push(PatternStep::NonGreedyStar(ranges, Box::new(suffix)));
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
                    let alt_steps = self.extract_from_state(alt_start, &mut alt_visited, Some(common_end));
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
        // Note: We only support simple linear patterns in lookarounds for now
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

            // Handle byte transitions
            if !state.transitions.is_empty() {
                let target = state.transitions[0].1;
                if !state.transitions.iter().all(|(_, t)| *t == target) {
                    return Vec::new(); // Different targets
                }

                let ranges: Vec<ByteRange> = state.transitions.iter()
                    .map(|(r, _)| r.clone())
                    .collect();

                if visited[current as usize] {
                    return Vec::new(); // Cycle
                }
                visited[current as usize] = true;

                if ranges.len() == 1 && ranges[0].start == ranges[0].end {
                    steps.push(PatternStep::Byte(ranges[0].start));
                } else {
                    steps.push(PatternStep::Ranges(ranges));
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

            // Complex structure - not supported
            if !state.epsilon.is_empty() || !state.transitions.is_empty() {
                return Vec::new();
            }

            break;
        }

        steps
    }

    /// Finds the common end state for an alternation starting at `start`.
    /// Returns None if no common end is found.
    fn find_alternation_end(&self, start: StateId) -> Option<StateId> {
        let state = &self.nfa.states[start as usize];
        if state.epsilon.len() < 2 {
            return None;
        }

        // For each alternative, find where it ends up
        // All alternatives should converge to the same state
        let mut end_states: Vec<StateId> = Vec::new();

        for &alt_start in &state.epsilon {
            if let Some(end) = self.trace_to_merge_point(alt_start, start) {
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
    fn trace_to_merge_point(&self, start: StateId, alt_start: StateId) -> Option<StateId> {
        let mut current = start;
        let mut visited = vec![false; self.nfa.states.len()];
        visited[alt_start as usize] = true; // Don't go back to alternation start

        for _ in 0..100 {
            // Limit iterations
            if visited[current as usize] {
                return None; // Cycle
            }
            visited[current as usize] = true;

            let state = &self.nfa.states[current as usize];

            // Match state or state with no outgoing transitions
            if state.is_match || (state.transitions.is_empty() && state.epsilon.is_empty()) {
                return Some(current);
            }

            // State with single epsilon going forward - this might be merge point
            // Check if it has other incoming epsilons (from other alternatives)
            if state.epsilon.len() == 1 && state.transitions.is_empty() {
                // Could be part of the path or the merge point
                // For now, follow it
                current = state.epsilon[0];
                continue;
            }

            // State with byte transitions - follow them
            if !state.transitions.is_empty() && state.epsilon.is_empty() {
                // Follow the byte transition
                let target = state.transitions[0].1;
                current = target;
                continue;
            }

            // State with both or multiple epsilons - likely the end
            if state.epsilon.len() == 1 {
                current = state.epsilon[0];
                continue;
            }

            // Complex - can't determine
            return None;
        }

        None
    }

    /// Checks if there's a trivial (empty) path from start to end.
    fn is_trivial_path(&self, start: StateId, end: StateId) -> bool {
        if start == end {
            return true;
        }
        let state = &self.nfa.states[start as usize];
        if state.epsilon.len() == 1 && state.transitions.is_empty() {
            return state.epsilon[0] == end || self.is_trivial_path(state.epsilon[0], end);
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

                let ranges: Vec<ByteRange> = state.transitions.iter()
                    .map(|(r, _)| r.clone())
                    .collect();

                return if ranges.len() == 1 && ranges[0].start == ranges[0].end {
                    Some(PatternStep::Byte(ranges[0].start))
                } else {
                    Some(PatternStep::Ranges(ranges))
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
            Error::new(ErrorKind::Jit(format!("Failed to finalize JIT code: {:?}", e)), "")
        })?;

        // Get function pointers
        let find_fn: unsafe extern "sysv64" fn(*const u8, usize, *mut TaggedNfaContext) -> i64 =
            unsafe { std::mem::transmute(code.ptr(find_offset)) };

        let captures_fn: unsafe extern "sysv64" fn(*const u8, usize, *mut TaggedNfaContext, *mut i64) -> i64 =
            unsafe { std::mem::transmute(code.ptr(captures_offset)) };

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
