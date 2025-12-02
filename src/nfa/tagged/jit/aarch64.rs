//! AArch64 JIT code generation for Tagged NFA.
//!
//! This module contains the TaggedNfaJitCompiler which generates AArch64 assembly
//! code for Thompson NFA simulation with captures.

use crate::error::{Error, ErrorKind, Result};
use crate::hir::CodepointClass;
use crate::nfa::{ByteClass, ByteRange, Nfa, NfaInstruction, StateId};

use super::super::{NfaLiveness, PatternStep, TaggedNfaContext};
use super::jit::TaggedNfaJit;

use dynasmrt::{dynasm, DynasmApi};

/// Internal compiler for Tagged NFA JIT on AArch64.
///
/// Register allocation (AAPCS64):
/// - x0-x7 = argument/result registers
/// - x9-x15 = scratch registers (caller-saved)
/// - x19-x28 = callee-saved registers
/// - x30 = link register (LR)
///
/// Pattern matching registers:
/// - x19 = input_ptr (callee-saved)
/// - x20 = input_len (callee-saved)
/// - x21 = start_pos (callee-saved)
/// - x22 = current_pos (callee-saved)
/// - x23 = captures_out pointer (in captures_fn) / saved_pos (in find_fn)
/// - w0/x0 = scratch / return value
#[allow(dead_code)]
pub(super) struct TaggedNfaJitCompiler {
    asm: dynasmrt::aarch64::Assembler,
    nfa: Nfa,
    liveness: NfaLiveness,
    state_labels: Vec<dynasmrt::DynamicLabel>,
    thread_loop_label: dynasmrt::DynamicLabel,
    advance_pos_label: dynasmrt::DynamicLabel,
    match_found_label: dynasmrt::DynamicLabel,
    done_label: dynasmrt::DynamicLabel,
    add_thread_label: dynasmrt::DynamicLabel,
    codepoint_classes: Vec<Box<CodepointClass>>,
    lookaround_nfas: Vec<Box<Nfa>>,
}

impl TaggedNfaJitCompiler {
    #[allow(dead_code)]
    fn new(nfa: Nfa, liveness: NfaLiveness) -> Result<Self> {
        let mut asm = dynasmrt::aarch64::Assembler::new().map_err(|e| {
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

    fn needs_interpreter_fallback(&self) -> bool {
        if self.nfa.states.len() > 256 {
            return true;
        }
        false
    }

    pub(super) fn compile(nfa: Nfa, liveness: NfaLiveness) -> Result<TaggedNfaJit> {
        let compiler = Self::new(nfa, liveness)?;
        if compiler.needs_interpreter_fallback() {
            return compiler.compile_with_fallback(None);
        }
        compiler.compile_full()
    }

    fn compile_with_fallback(mut self, steps: Option<Vec<PatternStep>>) -> Result<TaggedNfaJit> {
        let find_offset = self.asm.offset();
        dynasm!(self.asm
            ; .arch aarch64
            ; movn x0, 1            // x0 = -2
            ; ret
        );

        let captures_offset = self.asm.offset();
        dynasm!(self.asm
            ; .arch aarch64
            ; movn x0, 1
            ; ret
        );

        self.finalize(find_offset, captures_offset, false, steps)
    }

    fn has_backref(steps: &[PatternStep]) -> bool {
        steps.iter().any(|s| match s {
            PatternStep::Backref(_) => true,
            PatternStep::Alt(alts) => alts.iter().any(|alt| Self::has_backref(alt)),
            _ => false,
        })
    }

    fn has_unsupported_in_alt(alternatives: &[Vec<PatternStep>]) -> bool {
        for alt_steps in alternatives {
            for step in alt_steps {
                match step {
                    PatternStep::Alt(inner) => {
                        if Self::has_unsupported_in_alt(inner) {
                            return true;
                        }
                    }
                    PatternStep::NonGreedyPlus(_, _) | PatternStep::NonGreedyStar(_, _) => {
                        return true;
                    }
                    PatternStep::PositiveLookahead(_)
                    | PatternStep::NegativeLookahead(_)
                    | PatternStep::PositiveLookbehind(_, _)
                    | PatternStep::NegativeLookbehind(_, _) => return true,
                    _ => {}
                }
            }
        }
        false
    }

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
            PatternStep::Alt(alts) => alts
                .iter()
                .any(|a| a.iter().any(|s| Self::step_consumes_input(s))),
            _ => false,
        }
    }

    fn calc_min_len(steps: &[PatternStep]) -> usize {
        steps
            .iter()
            .map(|s| match s {
                PatternStep::Byte(_) | PatternStep::ByteClass(_) => 1,
                PatternStep::GreedyPlus(_) | PatternStep::GreedyPlusLookahead(_, _, _) => 1,
                PatternStep::GreedyStar(_) | PatternStep::GreedyStarLookahead(_, _, _) => 0,
                PatternStep::NonGreedyPlus(_, suf) => 1 + Self::calc_min_len(&[(**suf).clone()]),
                PatternStep::NonGreedyStar(_, suf) => Self::calc_min_len(&[(**suf).clone()]),
                PatternStep::Alt(alts) => alts
                    .iter()
                    .map(|a| Self::calc_min_len(a))
                    .min()
                    .unwrap_or(0),
                PatternStep::CodepointClass(_, _) | PatternStep::GreedyCodepointPlus(_) => 1,
                _ => 0,
            })
            .sum()
    }

    fn combine_greedy_with_lookahead(steps: Vec<PatternStep>) -> Vec<PatternStep> {
        let mut result = Vec::with_capacity(steps.len());
        let mut i = 0;
        while i < steps.len() {
            match &steps[i] {
                PatternStep::GreedyPlus(r) if i + 1 < steps.len() => match &steps[i + 1] {
                    PatternStep::PositiveLookahead(inner) => {
                        result.push(PatternStep::GreedyPlusLookahead(
                            r.clone(),
                            inner.clone(),
                            true,
                        ));
                        i += 2;
                        continue;
                    }
                    PatternStep::NegativeLookahead(inner) => {
                        result.push(PatternStep::GreedyPlusLookahead(
                            r.clone(),
                            inner.clone(),
                            false,
                        ));
                        i += 2;
                        continue;
                    }
                    _ => {}
                },
                PatternStep::GreedyStar(r) if i + 1 < steps.len() => match &steps[i + 1] {
                    PatternStep::PositiveLookahead(inner) => {
                        result.push(PatternStep::GreedyStarLookahead(
                            r.clone(),
                            inner.clone(),
                            true,
                        ));
                        i += 2;
                        continue;
                    }
                    PatternStep::NegativeLookahead(inner) => {
                        result.push(PatternStep::GreedyStarLookahead(
                            r.clone(),
                            inner.clone(),
                            false,
                        ));
                        i += 2;
                        continue;
                    }
                    _ => {}
                },
                PatternStep::Alt(alts) => {
                    let combined: Vec<Vec<PatternStep>> = alts
                        .iter()
                        .map(|a| Self::combine_greedy_with_lookahead(a.clone()))
                        .collect();
                    result.push(PatternStep::Alt(combined));
                    i += 1;
                    continue;
                }
                _ => {}
            }
            result.push(steps[i].clone());
            i += 1;
        }
        result
    }

    fn emit_range_check(
        &mut self,
        ranges: &[ByteRange],
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        if ranges.len() == 1 {
            let r = &ranges[0];
            let sz = r.end.wrapping_sub(r.start);
            dynasm!(self.asm
                ; .arch aarch64
                ; sub w1, w0, r.start as u32
                ; cmp w1, sz as u32
                ; b.hi =>fail_label
            );
        } else {
            let matched = self.asm.new_dynamic_label();
            for (ri, r) in ranges.iter().enumerate() {
                let sz = r.end.wrapping_sub(r.start);
                if ri == ranges.len() - 1 {
                    dynasm!(self.asm
                        ; .arch aarch64
                        ; sub w1, w0, r.start as u32
                        ; cmp w1, sz as u32
                        ; b.hi =>fail_label
                    );
                } else {
                    dynasm!(self.asm
                        ; .arch aarch64
                        ; sub w1, w0, r.start as u32
                        ; cmp w1, sz as u32
                        ; b.ls =>matched
                    );
                }
            }
            dynasm!(self.asm ; .arch aarch64 ; =>matched);
        }
        Ok(())
    }

    fn emit_is_word_char(
        &mut self,
        word_label: dynasmrt::DynamicLabel,
        not_word_label: dynasmrt::DynamicLabel,
    ) {
        use dynasmrt::DynasmLabelApi;
        dynasm!(self.asm
            ; .arch aarch64
            ; sub w1, w0, 0x61
            ; cmp w1, 25
            ; b.ls =>word_label
            ; sub w1, w0, 0x41
            ; cmp w1, 25
            ; b.ls =>word_label
            ; sub w1, w0, 0x30
            ; cmp w1, 9
            ; b.ls =>word_label
            ; cmp w0, 0x5f
            ; b.eq =>word_label
            ; b =>not_word_label
        );
    }

    fn emit_word_boundary_check(
        &mut self,
        fail_label: dynasmrt::DynamicLabel,
        is_boundary: bool,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        let prev_word = self.asm.new_dynamic_label();
        let prev_not_word = self.asm.new_dynamic_label();
        let curr_word = self.asm.new_dynamic_label();
        let curr_not_word = self.asm.new_dynamic_label();
        let check_curr = self.asm.new_dynamic_label();
        let boundary_match = self.asm.new_dynamic_label();

        dynasm!(self.asm
            ; .arch aarch64
            ; cbz x22, =>prev_not_word
            ; sub x1, x22, 1
            ; ldrb w0, [x19, x1]
        );
        self.emit_is_word_char(prev_word, prev_not_word);
        dynasm!(self.asm ; .arch aarch64 ; =>prev_word ; mov w9, 1 ; b =>check_curr);
        dynasm!(self.asm ; .arch aarch64 ; =>prev_not_word ; mov w9, 0);
        dynasm!(self.asm
            ; .arch aarch64
            ; =>check_curr
            ; cmp x22, x20
            ; b.ge =>curr_not_word
            ; ldrb w0, [x19, x22]
        );
        self.emit_is_word_char(curr_word, curr_not_word);
        dynasm!(self.asm ; .arch aarch64 ; =>curr_word ; mov w10, 1 ; b =>boundary_match);
        dynasm!(self.asm ; .arch aarch64 ; =>curr_not_word ; mov w10, 0);
        dynasm!(self.asm ; .arch aarch64 ; =>boundary_match ; eor w9, w9, w10);
        if is_boundary {
            dynasm!(self.asm ; .arch aarch64 ; cbz w9, =>fail_label);
        } else {
            dynasm!(self.asm ; .arch aarch64 ; cbnz w9, =>fail_label);
        }
        Ok(())
    }

    fn emit_utf8_decode(&mut self, fail_label: dynasmrt::DynamicLabel) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        let ascii = self.asm.new_dynamic_label();
        let two_byte = self.asm.new_dynamic_label();
        let three_byte = self.asm.new_dynamic_label();
        let four_byte = self.asm.new_dynamic_label();
        let done = self.asm.new_dynamic_label();

        dynasm!(self.asm
            ; .arch aarch64
            ; cmp x22, x20
            ; b.ge =>fail_label
            ; ldrb w0, [x19, x22]
            ; cmp w0, 0x80
            ; b.lo =>ascii
            ; cmp w0, 0xC0
            ; b.lo =>fail_label
            ; cmp w0, 0xE0
            ; b.lo =>two_byte
            ; cmp w0, 0xF0
            ; b.lo =>three_byte
            ; cmp w0, 0xF8
            ; b.lo =>four_byte
            ; b =>fail_label
        );
        dynasm!(self.asm ; .arch aarch64 ; =>ascii ; mov w1, 1 ; b =>done);
        dynasm!(self.asm
            ; .arch aarch64
            ; =>two_byte
            ; add x2, x22, 1
            ; cmp x2, x20
            ; b.ge =>fail_label
            ; ldrb w3, [x19, x2]
            ; and w4, w3, 0xC0
            ; cmp w4, 0x80
            ; b.ne =>fail_label
            ; and w0, w0, 0x1F
            ; lsl w0, w0, 6
            ; and w3, w3, 0x3F
            ; orr w0, w0, w3
            ; mov w1, 2
            ; b =>done
        );
        dynasm!(self.asm
            ; .arch aarch64
            ; =>three_byte
            ; add x2, x22, 2
            ; cmp x2, x20
            ; b.ge =>fail_label
            ; add x4, x22, 1
            ; ldrb w3, [x19, x4]
            ; and w5, w3, 0xC0
            ; cmp w5, 0x80
            ; b.ne =>fail_label
            ; ldrb w4, [x19, x2]
            ; and w5, w4, 0xC0
            ; cmp w5, 0x80
            ; b.ne =>fail_label
            ; and w0, w0, 0x0F
            ; lsl w0, w0, 12
            ; and w3, w3, 0x3F
            ; lsl w3, w3, 6
            ; orr w0, w0, w3
            ; and w4, w4, 0x3F
            ; orr w0, w0, w4
            ; mov w1, 3
            ; b =>done
        );
        dynasm!(self.asm
            ; .arch aarch64
            ; =>four_byte
            ; add x2, x22, 3
            ; cmp x2, x20
            ; b.ge =>fail_label
            ; add x4, x22, 1
            ; ldrb w3, [x19, x4]
            ; and w5, w3, 0xC0
            ; cmp w5, 0x80
            ; b.ne =>fail_label
            ; add x4, x22, 2
            ; ldrb w4, [x19, x4]
            ; and w5, w4, 0xC0
            ; cmp w5, 0x80
            ; b.ne =>fail_label
            ; ldrb w5, [x19, x2]
            ; and w6, w5, 0xC0
            ; cmp w6, 0x80
            ; b.ne =>fail_label
            ; and w0, w0, 0x07
            ; lsl w0, w0, 18
            ; and w3, w3, 0x3F
            ; lsl w3, w3, 12
            ; orr w0, w0, w3
            ; and w4, w4, 0x3F
            ; lsl w4, w4, 6
            ; orr w0, w0, w4
            ; and w5, w5, 0x3F
            ; orr w0, w0, w5
            ; mov w1, 4
        );
        dynasm!(self.asm ; .arch aarch64 ; =>done);
        Ok(())
    }

    fn emit_codepoint_class_membership_check(
        &mut self,
        cpclass: &CodepointClass,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        let ascii_fast = self.asm.new_dynamic_label();
        let check_done = self.asm.new_dynamic_label();
        let bitmap_lo = cpclass.ascii_bitmap[0];
        let bitmap_hi = cpclass.ascii_bitmap[1];
        let is_negated = cpclass.negated;

        dynasm!(self.asm ; .arch aarch64 ; cmp w0, 128 ; b.lo =>ascii_fast);

        let cpclass_box = Box::new(cpclass.clone());
        let cpclass_ptr = cpclass_box.as_ref() as *const CodepointClass;
        self.codepoint_classes.push(cpclass_box);

        extern "C" fn check_membership(cp: u32, cls: *const CodepointClass) -> bool {
            unsafe { &*cls }.contains(cp)
        }
        let fn_ptr = check_membership as usize as u64;
        let cpclass_ptr_u64 = cpclass_ptr as u64;

        // Split 64-bit pointers into 16-bit chunks for movz/movk (ARM64 requirement)
        let cls_lo = (cpclass_ptr_u64 & 0xFFFF) as u32;
        let cls_16 = ((cpclass_ptr_u64 >> 16) & 0xFFFF) as u32;
        let cls_32 = ((cpclass_ptr_u64 >> 32) & 0xFFFF) as u32;
        let cls_48 = ((cpclass_ptr_u64 >> 48) & 0xFFFF) as u32;

        let fn_lo = (fn_ptr & 0xFFFF) as u32;
        let fn_16 = ((fn_ptr >> 16) & 0xFFFF) as u32;
        let fn_32 = ((fn_ptr >> 32) & 0xFFFF) as u32;
        let fn_48 = ((fn_ptr >> 48) & 0xFFFF) as u32;

        // Split bitmap values into 16-bit chunks
        let bm_lo_0 = (bitmap_lo & 0xFFFF) as u32;
        let bm_lo_16 = ((bitmap_lo >> 16) & 0xFFFF) as u32;
        let bm_lo_32 = ((bitmap_lo >> 32) & 0xFFFF) as u32;
        let bm_lo_48 = ((bitmap_lo >> 48) & 0xFFFF) as u32;

        let bm_hi_0 = (bitmap_hi & 0xFFFF) as u32;
        let bm_hi_16 = ((bitmap_hi >> 16) & 0xFFFF) as u32;
        let bm_hi_32 = ((bitmap_hi >> 32) & 0xFFFF) as u32;
        let bm_hi_48 = ((bitmap_hi >> 48) & 0xFFFF) as u32;

        // Non-ASCII path: call helper function
        // Note: contains() already handles negation internally, so we just check if result is false
        dynasm!(self.asm
            ; .arch aarch64
            // Load cpclass pointer into x1
            ; movz x1, #cls_lo
            ; movk x1, #cls_16, lsl #16
            ; movk x1, #cls_32, lsl #32
            ; movk x1, #cls_48, lsl #48
            // Load function pointer into x9
            ; movz x9, #fn_lo
            ; movk x9, #fn_16, lsl #16
            ; movk x9, #fn_32, lsl #32
            ; movk x9, #fn_48, lsl #48
            ; blr x9
            ; cbz w0, =>fail_label
            ; b =>check_done
        );

        dynasm!(self.asm
            ; .arch aarch64
            ; =>ascii_fast
            ; cmp w0, 64
            ; b.hs >use_hi
            // Load bitmap_lo into x2
            ; movz x2, #bm_lo_0
            ; movk x2, #bm_lo_16, lsl #16
            ; movk x2, #bm_lo_32, lsl #32
            ; movk x2, #bm_lo_48, lsl #48
            ; mov x3, 1
            ; lsl x3, x3, x0
            ; tst x2, x3
            ; b >check_result
            ; use_hi:
            // Load bitmap_hi into x2
            ; movz x2, #bm_hi_0
            ; movk x2, #bm_hi_16, lsl #16
            ; movk x2, #bm_hi_32, lsl #32
            ; movk x2, #bm_hi_48, lsl #48
            ; sub w4, w0, 64
            ; mov x3, 1
            ; lsl x3, x3, x4
            ; tst x2, x3
            ; check_result:
        );

        if is_negated {
            dynasm!(self.asm ; .arch aarch64 ; b.ne =>fail_label);
        } else {
            dynasm!(self.asm ; .arch aarch64 ; b.eq =>fail_label);
        }
        dynasm!(self.asm ; .arch aarch64 ; =>check_done);
        Ok(())
    }

    fn emit_codepoint_class_check(
        &mut self,
        cpclass: &CodepointClass,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        let fail_stack = self.asm.new_dynamic_label();
        self.emit_utf8_decode(fail_label)?;
        dynasm!(self.asm ; .arch aarch64 ; str x1, [sp, -16]!);
        self.emit_codepoint_class_membership_check(cpclass, fail_stack)?;
        dynasm!(self.asm
            ; .arch aarch64
            ; ldr x1, [sp], 16
            ; add x22, x22, x1
            ; b >done
            ; =>fail_stack
            ; add sp, sp, 16
            ; b =>fail_label
            ; done:
        );
        Ok(())
    }

    fn emit_greedy_codepoint_plus(
        &mut self,
        cpclass: &CodepointClass,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        let loop_start = self.asm.new_dynamic_label();
        let loop_done = self.asm.new_dynamic_label();
        let first_fail_stack = self.asm.new_dynamic_label();
        let loop_fail_no_stack = self.asm.new_dynamic_label();
        let loop_fail_stack = self.asm.new_dynamic_label();

        self.emit_utf8_decode(fail_label)?;
        dynasm!(self.asm ; .arch aarch64 ; str x1, [sp, -16]!);
        self.emit_codepoint_class_membership_check(cpclass, first_fail_stack)?;
        dynasm!(self.asm ; .arch aarch64 ; ldr x1, [sp], 16 ; add x22, x22, x1);
        dynasm!(self.asm ; .arch aarch64 ; =>loop_start);
        self.emit_utf8_decode(loop_fail_no_stack)?;
        dynasm!(self.asm ; .arch aarch64 ; str x1, [sp, -16]!);
        self.emit_codepoint_class_membership_check(cpclass, loop_fail_stack)?;
        dynasm!(self.asm
            ; .arch aarch64
            ; ldr x1, [sp], 16
            ; add x22, x22, x1
            ; b =>loop_start
            ; =>first_fail_stack
            ; add sp, sp, 16
            ; b =>fail_label
            ; =>loop_fail_no_stack
            ; b =>loop_done
            ; =>loop_fail_stack
            ; add sp, sp, 16
            ; =>loop_done
        );
        Ok(())
    }

    fn emit_non_greedy_suffix_check(
        &mut self,
        suffix: &PatternStep,
        fail_label: dynasmrt::DynamicLabel,
        _success: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        match suffix {
            PatternStep::Byte(b) => {
                dynasm!(self.asm
                    ; .arch aarch64
                    ; cmp x22, x20
                    ; b.ge =>fail_label
                    ; ldrb w0, [x19, x22]
                    ; cmp w0, *b as u32
                    ; b.ne =>fail_label
                    ; add x22, x22, 1
                );
            }
            PatternStep::ByteClass(bc) => {
                dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>fail_label ; ldrb w0, [x19, x22]);
                self.emit_range_check(&bc.ranges, fail_label)?;
                dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1);
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::Jit("Unsupported suffix".to_string()),
                    "",
                ))
            }
        }
        Ok(())
    }

    fn emit_step_inline(
        &mut self,
        step: &PatternStep,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        match step {
            PatternStep::Byte(b) => {
                dynasm!(self.asm
                    ; .arch aarch64
                    ; cmp x22, x20 ; b.ge =>fail_label
                    ; ldrb w0, [x19, x22]
                    ; cmp w0, *b as u32 ; b.ne =>fail_label
                    ; add x22, x22, 1
                );
            }
            PatternStep::ByteClass(bc) => {
                dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>fail_label ; ldrb w0, [x19, x22]);
                self.emit_range_check(&bc.ranges, fail_label)?;
                dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1);
            }
            PatternStep::GreedyPlus(bc) => {
                let ls = self.asm.new_dynamic_label();
                let ld = self.asm.new_dynamic_label();
                dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>fail_label ; ldrb w0, [x19, x22]);
                self.emit_range_check(&bc.ranges, fail_label)?;
                dynasm!(self.asm
                    ; .arch aarch64
                    ; add x22, x22, 1
                    ; =>ls ; cmp x22, x20 ; b.ge =>ld ; ldrb w0, [x19, x22]
                );
                self.emit_range_check(&bc.ranges, ld)?;
                dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>ls ; =>ld);
            }
            PatternStep::GreedyStar(bc) => {
                let ls = self.asm.new_dynamic_label();
                let ld = self.asm.new_dynamic_label();
                dynasm!(self.asm ; .arch aarch64 ; =>ls ; cmp x22, x20 ; b.ge =>ld ; ldrb w0, [x19, x22]);
                self.emit_range_check(&bc.ranges, ld)?;
                dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>ls ; =>ld);
            }
            PatternStep::CodepointClass(cp, _) => {
                self.emit_codepoint_class_check(cp, fail_label)?
            }
            PatternStep::GreedyCodepointPlus(cp) => {
                self.emit_greedy_codepoint_plus(cp, fail_label)?
            }
            PatternStep::WordBoundary => self.emit_word_boundary_check(fail_label, true)?,
            PatternStep::NotWordBoundary => self.emit_word_boundary_check(fail_label, false)?,
            PatternStep::StartOfText => {
                dynasm!(self.asm ; .arch aarch64 ; cbnz x22, =>fail_label);
            }
            PatternStep::EndOfText => {
                dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ne =>fail_label);
            }
            PatternStep::StartOfLine => {
                let at_start = self.asm.new_dynamic_label();
                dynasm!(self.asm
                    ; .arch aarch64
                    ; cbz x22, =>at_start
                    ; sub x1, x22, 1 ; ldrb w0, [x19, x1] ; cmp w0, 0x0A ; b.ne =>fail_label
                    ; =>at_start
                );
            }
            PatternStep::EndOfLine => {
                let at_end = self.asm.new_dynamic_label();
                dynasm!(self.asm
                    ; .arch aarch64
                    ; cmp x22, x20 ; b.eq =>at_end
                    ; ldrb w0, [x19, x22] ; cmp w0, 0x0A ; b.ne =>fail_label
                    ; =>at_end
                );
            }
            PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_) => {}
            PatternStep::Alt(alts) => {
                let success = self.asm.new_dynamic_label();
                for (i, alt_steps) in alts.iter().enumerate() {
                    let is_last = i == alts.len() - 1;
                    let try_next = self.asm.new_dynamic_label();
                    dynasm!(self.asm ; .arch aarch64 ; str x22, [sp, -16]!);
                    for s in alt_steps {
                        self.emit_step_inline(s, try_next)?;
                    }
                    dynasm!(self.asm ; .arch aarch64 ; add sp, sp, 16 ; b =>success);
                    dynasm!(self.asm ; .arch aarch64 ; =>try_next ; ldr x22, [sp], 16);
                    if is_last {
                        dynasm!(self.asm ; .arch aarch64 ; b =>fail_label);
                    }
                }
                dynasm!(self.asm ; .arch aarch64 ; =>success);
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::Jit(format!("Unsupported step: {:?}", step)),
                    "",
                ))
            }
        }
        Ok(())
    }

    fn emit_standalone_lookahead(
        &mut self,
        inner: &[PatternStep],
        fail_label: dynasmrt::DynamicLabel,
        positive: bool,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        let inner_match = self.asm.new_dynamic_label();
        dynasm!(self.asm ; .arch aarch64 ; mov x9, x22); // Save position

        for step in inner {
            match step {
                PatternStep::Byte(b) => {
                    if positive {
                        dynasm!(self.asm ; .arch aarch64 ; cmp x9, x20 ; b.ge =>fail_label ; ldrb w0, [x19, x9] ; cmp w0, *b as u32 ; b.ne =>fail_label ; add x9, x9, 1);
                    } else {
                        dynasm!(self.asm ; .arch aarch64 ; cmp x9, x20 ; b.ge =>inner_match ; ldrb w0, [x19, x9] ; cmp w0, *b as u32 ; b.ne =>inner_match ; add x9, x9, 1);
                    }
                }
                PatternStep::ByteClass(bc) => {
                    if positive {
                        dynasm!(self.asm ; .arch aarch64 ; cmp x9, x20 ; b.ge =>fail_label ; ldrb w0, [x19, x9]);
                        self.emit_range_check(&bc.ranges, fail_label)?;
                        dynasm!(self.asm ; .arch aarch64 ; add x9, x9, 1);
                    } else {
                        dynasm!(self.asm ; .arch aarch64 ; cmp x9, x20 ; b.ge =>inner_match ; ldrb w0, [x19, x9]);
                        self.emit_range_check(&bc.ranges, inner_match)?;
                        dynasm!(self.asm ; .arch aarch64 ; add x9, x9, 1);
                    }
                }
                PatternStep::EndOfText => {
                    if positive {
                        dynasm!(self.asm ; .arch aarch64 ; cmp x9, x20 ; b.ne =>fail_label);
                    } else {
                        dynasm!(self.asm ; .arch aarch64 ; cmp x9, x20 ; b.ne =>inner_match);
                    }
                }
                _ => {
                    return Err(Error::new(
                        ErrorKind::Jit("Complex lookahead".to_string()),
                        "",
                    ))
                }
            }
        }

        if !positive {
            dynasm!(self.asm ; .arch aarch64 ; b =>fail_label);
        }
        dynasm!(self.asm ; .arch aarch64 ; =>inner_match);
        Ok(())
    }

    fn emit_lookbehind_check(
        &mut self,
        inner: &[PatternStep],
        min_len: usize,
        fail_label: dynasmrt::DynamicLabel,
        positive: bool,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        let inner_match = self.asm.new_dynamic_label();
        let inner_mismatch = self.asm.new_dynamic_label();
        let done = self.asm.new_dynamic_label();

        dynasm!(self.asm ; .arch aarch64 ; mov x9, x22);
        if min_len > 0 {
            dynasm!(self.asm ; .arch aarch64 ; cmp x22, min_len as u32 ; b.lo =>inner_mismatch);
        }
        dynasm!(self.asm ; .arch aarch64 ; sub x22, x22, min_len as u32);

        for step in inner {
            match step {
                PatternStep::Byte(b) => {
                    dynasm!(self.asm ; .arch aarch64 ; ldrb w0, [x19, x22] ; cmp w0, *b as u32 ; b.ne =>inner_mismatch ; add x22, x22, 1);
                }
                PatternStep::ByteClass(bc) => {
                    dynasm!(self.asm ; .arch aarch64 ; ldrb w0, [x19, x22]);
                    self.emit_range_check(&bc.ranges, inner_mismatch)?;
                    dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1);
                }
                _ => {
                    return Err(Error::new(
                        ErrorKind::Jit("Unsupported lookbehind step".to_string()),
                        "",
                    ))
                }
            }
        }
        dynasm!(self.asm ; .arch aarch64 ; b =>inner_match);
        dynasm!(self.asm ; .arch aarch64 ; =>inner_mismatch);
        if positive {
            dynasm!(self.asm ; .arch aarch64 ; mov x22, x9 ; b =>fail_label);
            dynasm!(self.asm ; .arch aarch64 ; =>inner_match ; mov x22, x9 ; b =>done);
        } else {
            dynasm!(self.asm ; .arch aarch64 ; mov x22, x9 ; b =>done);
            dynasm!(self.asm ; .arch aarch64 ; =>inner_match ; mov x22, x9 ; b =>fail_label);
        }
        dynasm!(self.asm ; .arch aarch64 ; =>done);
        Ok(())
    }

    fn compile_full(mut self) -> Result<TaggedNfaJit> {
        use dynasmrt::DynasmLabelApi;
        let steps = self.extract_pattern_steps();
        let steps = Self::combine_greedy_with_lookahead(steps);
        if steps.is_empty() {
            return self.compile_with_fallback(None);
        }
        for step in &steps {
            if let PatternStep::Alt(alts) = step {
                if Self::has_unsupported_in_alt(alts) {
                    return self.compile_with_fallback(Some(steps));
                }
            }
        }
        let has_backrefs = Self::has_backref(&steps);
        let min_len = Self::calc_min_len(&steps);

        let find_offset = self.asm.offset();
        if has_backrefs {
            dynasm!(self.asm ; .arch aarch64 ; movn x0, 1 ; ret);
            let caps_off = self.emit_captures_fn(&steps)?;
            return self.finalize(find_offset, caps_off, true, None);
        }

        // Prologue
        dynasm!(self.asm
            ; .arch aarch64
            ; stp x29, x30, [sp, -16]!
            ; mov x29, sp
            ; stp x19, x20, [sp, -16]!
            ; stp x21, x22, [sp, -16]!
            ; stp x23, x24, [sp, -16]!
            ; mov x19, x0   // input_ptr
            ; mov x20, x1   // input_len
            ; mov x21, xzr  // start_pos = 0
        );

        let start_loop = self.asm.new_dynamic_label();
        let match_found = self.asm.new_dynamic_label();
        let no_match = self.asm.new_dynamic_label();
        let byte_mismatch = self.asm.new_dynamic_label();

        dynasm!(self.asm
            ; .arch aarch64
            ; =>start_loop
            ; sub x0, x20, x21
            ; cmp x0, min_len as u32
            ; b.lo =>no_match
            ; mov x22, x21
        );

        for (si, step) in steps.iter().enumerate() {
            match step {
                PatternStep::Byte(b) => {
                    dynasm!(self.asm
                        ; .arch aarch64
                        ; cmp x22, x20 ; b.ge =>byte_mismatch
                        ; ldrb w0, [x19, x22]
                        ; cmp w0, *b as u32 ; b.ne =>byte_mismatch
                        ; add x22, x22, 1
                    );
                }
                PatternStep::ByteClass(bc) => {
                    dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>byte_mismatch ; ldrb w0, [x19, x22]);
                    self.emit_range_check(&bc.ranges, byte_mismatch)?;
                    dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1);
                }
                PatternStep::GreedyPlus(bc) => {
                    let remaining = &steps[si + 1..];
                    let needs_bt = remaining.iter().any(|s| Self::step_consumes_input(s));
                    if needs_bt {
                        self.emit_greedy_plus_with_backtracking(
                            &bc.ranges,
                            remaining,
                            byte_mismatch,
                        )?;
                        break;
                    } else {
                        let ls = self.asm.new_dynamic_label();
                        let ld = self.asm.new_dynamic_label();
                        dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>byte_mismatch ; ldrb w0, [x19, x22]);
                        self.emit_range_check(&bc.ranges, byte_mismatch)?;
                        dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; =>ls ; cmp x22, x20 ; b.ge =>ld ; ldrb w0, [x19, x22]);
                        self.emit_range_check(&bc.ranges, ld)?;
                        dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>ls ; =>ld);
                    }
                }
                PatternStep::GreedyStar(bc) => {
                    let remaining = &steps[si + 1..];
                    let needs_bt = remaining.iter().any(|s| Self::step_consumes_input(s));
                    if needs_bt {
                        self.emit_greedy_star_with_backtracking(
                            &bc.ranges,
                            remaining,
                            byte_mismatch,
                        )?;
                        break;
                    } else {
                        let ls = self.asm.new_dynamic_label();
                        let ld = self.asm.new_dynamic_label();
                        dynasm!(self.asm ; .arch aarch64 ; =>ls ; cmp x22, x20 ; b.ge =>ld ; ldrb w0, [x19, x22]);
                        self.emit_range_check(&bc.ranges, ld)?;
                        dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>ls ; =>ld);
                    }
                }
                PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_) => {}
                PatternStep::CodepointClass(cp, _) => {
                    self.emit_codepoint_class_check(cp, byte_mismatch)?
                }
                PatternStep::GreedyCodepointPlus(cp) => {
                    let remaining = &steps[si + 1..];
                    let needs_bt = remaining.iter().any(|s| Self::step_consumes_input(s));
                    if needs_bt {
                        self.emit_greedy_codepoint_plus_with_backtracking(
                            cp,
                            remaining,
                            byte_mismatch,
                        )?;
                        break;
                    } else {
                        self.emit_greedy_codepoint_plus(cp, byte_mismatch)?;
                    }
                }
                PatternStep::WordBoundary => self.emit_word_boundary_check(byte_mismatch, true)?,
                PatternStep::NotWordBoundary => {
                    self.emit_word_boundary_check(byte_mismatch, false)?
                }
                PatternStep::StartOfText => {
                    dynasm!(self.asm ; .arch aarch64 ; cbnz x22, =>byte_mismatch);
                }
                PatternStep::EndOfText => {
                    dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ne =>byte_mismatch);
                }
                PatternStep::StartOfLine => {
                    let at_start = self.asm.new_dynamic_label();
                    dynasm!(self.asm ; .arch aarch64 ; cbz x22, =>at_start ; sub x1, x22, 1 ; ldrb w0, [x19, x1] ; cmp w0, 0x0A ; b.ne =>byte_mismatch ; =>at_start);
                }
                PatternStep::EndOfLine => {
                    let at_end = self.asm.new_dynamic_label();
                    dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.eq =>at_end ; ldrb w0, [x19, x22] ; cmp w0, 0x0A ; b.ne =>byte_mismatch ; =>at_end);
                }
                PatternStep::PositiveLookahead(inner) => {
                    self.emit_standalone_lookahead(inner, byte_mismatch, true)?
                }
                PatternStep::NegativeLookahead(inner) => {
                    self.emit_standalone_lookahead(inner, byte_mismatch, false)?
                }
                PatternStep::PositiveLookbehind(inner, ml) => {
                    self.emit_lookbehind_check(inner, *ml, byte_mismatch, true)?
                }
                PatternStep::NegativeLookbehind(inner, ml) => {
                    self.emit_lookbehind_check(inner, *ml, byte_mismatch, false)?
                }
                PatternStep::Alt(alts) => {
                    if Self::has_unsupported_in_alt(alts) {
                        return self.compile_with_fallback(Some(steps.clone()));
                    }
                    let alt_success = self.asm.new_dynamic_label();
                    dynasm!(self.asm ; .arch aarch64 ; mov x23, x22);
                    for (ai, alt_steps) in alts.iter().enumerate() {
                        let is_last = ai == alts.len() - 1;
                        let try_next = if is_last {
                            byte_mismatch
                        } else {
                            self.asm.new_dynamic_label()
                        };
                        for s in alt_steps {
                            self.emit_alt_step(s, try_next)?;
                        }
                        dynasm!(self.asm ; .arch aarch64 ; b =>alt_success);
                        if !is_last {
                            dynasm!(self.asm ; .arch aarch64 ; =>try_next ; mov x22, x23);
                        }
                    }
                    dynasm!(self.asm ; .arch aarch64 ; =>alt_success);
                }
                PatternStep::NonGreedyPlus(bc, suf) => {
                    let try_suf = self.asm.new_dynamic_label();
                    let consume = self.asm.new_dynamic_label();
                    let matched = self.asm.new_dynamic_label();
                    dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>byte_mismatch ; ldrb w0, [x19, x22]);
                    self.emit_range_check(&bc.ranges, byte_mismatch)?;
                    dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; =>try_suf);
                    self.emit_non_greedy_suffix_check(suf, consume, matched)?;
                    dynasm!(self.asm
                        ; .arch aarch64
                        ; b =>matched
                        ; =>consume
                        ; cmp x22, x20 ; b.ge =>byte_mismatch ; ldrb w0, [x19, x22]
                    );
                    self.emit_range_check(&bc.ranges, byte_mismatch)?;
                    dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>try_suf ; =>matched);
                }
                PatternStep::NonGreedyStar(bc, suf) => {
                    let try_suf = self.asm.new_dynamic_label();
                    let consume = self.asm.new_dynamic_label();
                    let matched = self.asm.new_dynamic_label();
                    dynasm!(self.asm ; .arch aarch64 ; =>try_suf);
                    self.emit_non_greedy_suffix_check(suf, consume, matched)?;
                    dynasm!(self.asm
                        ; .arch aarch64
                        ; b =>matched
                        ; =>consume
                        ; cmp x22, x20 ; b.ge =>byte_mismatch ; ldrb w0, [x19, x22]
                    );
                    self.emit_range_check(&bc.ranges, byte_mismatch)?;
                    dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>try_suf ; =>matched);
                }
                PatternStep::GreedyPlusLookahead(bc, la, pos) => {
                    self.emit_greedy_plus_with_lookahead(&bc.ranges, la, *pos, byte_mismatch)?
                }
                PatternStep::GreedyStarLookahead(bc, la, pos) => {
                    self.emit_greedy_star_with_lookahead(&bc.ranges, la, *pos, byte_mismatch)?
                }
                PatternStep::Backref(_) => unreachable!("Backref handled above"),
            }
        }

        dynasm!(self.asm ; .arch aarch64 ; b =>match_found);
        dynasm!(self.asm ; .arch aarch64 ; =>byte_mismatch ; add x21, x21, 1 ; b =>start_loop);
        dynasm!(self.asm
            ; .arch aarch64
            ; =>match_found
            ; lsl x0, x21, 32
            ; orr x0, x0, x22
            ; ldp x23, x24, [sp], 16
            ; ldp x21, x22, [sp], 16
            ; ldp x19, x20, [sp], 16
            ; ldp x29, x30, [sp], 16
            ; ret
        );
        dynasm!(self.asm
            ; .arch aarch64
            ; =>no_match
            ; movn x0, 0
            ; ldp x23, x24, [sp], 16
            ; ldp x21, x22, [sp], 16
            ; ldp x19, x20, [sp], 16
            ; ldp x29, x30, [sp], 16
            ; ret
        );

        let has_captures = steps
            .iter()
            .any(|s| matches!(s, PatternStep::CaptureStart(_) | PatternStep::CaptureEnd(_)));
        let caps_off = if has_captures {
            self.emit_captures_fn(&steps)?
        } else {
            let off = self.asm.offset();
            dynasm!(self.asm ; .arch aarch64 ; movn x0, 1 ; ret);
            off
        };

        self.finalize(find_offset, caps_off, false, Some(steps))
    }

    fn emit_alt_step(
        &mut self,
        step: &PatternStep,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        self.emit_step_inline(step, fail_label)
    }

    fn emit_greedy_plus_with_backtracking(
        &mut self,
        ranges: &[ByteRange],
        remaining: &[PatternStep],
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        let greedy_loop = self.asm.new_dynamic_label();
        let greedy_done = self.asm.new_dynamic_label();
        let try_remaining = self.asm.new_dynamic_label();
        let backtrack = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();

        dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>fail_label ; ldrb w0, [x19, x22]);
        self.emit_range_check(ranges, fail_label)?;
        dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; mov x9, x22);
        dynasm!(self.asm ; .arch aarch64 ; =>greedy_loop ; cmp x22, x20 ; b.ge =>greedy_done ; ldrb w0, [x19, x22]);
        self.emit_range_check(ranges, greedy_done)?;
        dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>greedy_loop ; =>greedy_done ; =>try_remaining);
        for s in remaining {
            self.emit_step_inline(s, backtrack)?;
        }
        dynasm!(self.asm
            ; .arch aarch64
            ; b =>success
            ; =>backtrack ; sub x22, x22, 1 ; cmp x22, x9 ; b.lo =>fail_label ; b =>try_remaining
            ; =>success
        );
        Ok(())
    }

    fn emit_greedy_star_with_backtracking(
        &mut self,
        ranges: &[ByteRange],
        remaining: &[PatternStep],
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        let greedy_loop = self.asm.new_dynamic_label();
        let greedy_done = self.asm.new_dynamic_label();
        let try_remaining = self.asm.new_dynamic_label();
        let backtrack = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();

        dynasm!(self.asm ; .arch aarch64 ; mov x9, x22);
        dynasm!(self.asm ; .arch aarch64 ; =>greedy_loop ; cmp x22, x20 ; b.ge =>greedy_done ; ldrb w0, [x19, x22]);
        self.emit_range_check(ranges, greedy_done)?;
        dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>greedy_loop ; =>greedy_done ; =>try_remaining);
        for s in remaining {
            self.emit_step_inline(s, backtrack)?;
        }
        dynasm!(self.asm
            ; .arch aarch64
            ; b =>success
            ; =>backtrack ; cmp x22, x9 ; b.ls =>fail_label ; sub x22, x22, 1 ; b =>try_remaining
            ; =>success
        );
        Ok(())
    }

    fn emit_greedy_codepoint_plus_with_backtracking(
        &mut self,
        cpclass: &CodepointClass,
        remaining: &[PatternStep],
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;

        // For codepoint backtracking, we save character boundaries on the stack
        let loop_start = self.asm.new_dynamic_label();
        let loop_done = self.asm.new_dynamic_label();
        let try_remaining = self.asm.new_dynamic_label();
        let backtrack = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();
        let first_fail_stack = self.asm.new_dynamic_label();
        let loop_fail_no_stack = self.asm.new_dynamic_label();
        let loop_fail_stack = self.asm.new_dynamic_label();
        let no_more_boundaries = self.asm.new_dynamic_label();

        // x24 will track the number of saved boundaries on stack
        dynasm!(self.asm
            ; .arch aarch64
            ; mov x24, xzr                    // x24 = boundary count = 0
        );

        // First iteration: must match at least one codepoint
        self.emit_utf8_decode(fail_label)?;
        dynasm!(self.asm
            ; .arch aarch64
            ; str x1, [sp, -16]!              // Save byte length
        );
        self.emit_codepoint_class_membership_check(cpclass, first_fail_stack)?;
        dynasm!(self.asm
            ; .arch aarch64
            ; ldr x1, [sp], 16                // Restore byte length
            ; add x22, x22, x1                // Advance position
            ; str x22, [sp, -16]!             // Save boundary position
            ; add x24, x24, 1                 // boundary count++

            // Greedy loop: match more codepoints
            ; =>loop_start
        );

        self.emit_utf8_decode(loop_fail_no_stack)?;
        dynasm!(self.asm
            ; .arch aarch64
            ; str x1, [sp, -16]!              // Save byte length
        );
        self.emit_codepoint_class_membership_check(cpclass, loop_fail_stack)?;
        dynasm!(self.asm
            ; .arch aarch64
            ; ldr x1, [sp], 16                // Restore byte length
            ; add x22, x22, x1                // Advance position
            ; str x22, [sp, -16]!             // Save boundary position
            ; add x24, x24, 1                 // boundary count++
            ; b =>loop_start

            ; =>first_fail_stack
            ; add sp, sp, 16                  // Pop saved byte length
            ; b =>fail_label                  // First match failed - overall fail

            ; =>loop_fail_no_stack
            ; b =>loop_done

            ; =>loop_fail_stack
            ; add sp, sp, 16                  // Pop saved byte length
            ; b =>loop_done

            ; =>loop_done
            // Greedy matching done
            // Stack has boundary positions, x24 = count
            // Try remaining steps with backtracking

            ; =>try_remaining
        );

        // Emit code for remaining steps
        for step in remaining {
            self.emit_step_inline(step, backtrack)?;
        }

        // All remaining steps matched - success!
        // Clean up stack (pop all saved boundaries)
        dynasm!(self.asm
            ; .arch aarch64
            ; =>success
            ; lsl x0, x24, 4                  // x0 = boundary_count * 16 (stack slot size)
            ; add sp, sp, x0                  // Pop all boundary positions
            ; b >done

            ; =>backtrack
            // Remaining steps failed - backtrack to previous boundary
            ; cmp x24, 1
            ; b.le =>no_more_boundaries       // Need at least 1 match (plus semantics)

            ; ldr x22, [sp], 16               // Pop and discard current boundary
            ; sub x24, x24, 1
            ; ldr x22, [sp]                   // Peek previous boundary (don't pop yet)
            ; b =>try_remaining

            ; =>no_more_boundaries
            // Can't backtrack more - clean up and fail
            ; lsl x0, x24, 4                  // x0 = boundary_count * 16
            ; add sp, sp, x0                  // Pop all remaining boundaries
            ; b =>fail_label

            ; done:
        );

        Ok(())
    }

    fn emit_greedy_plus_with_lookahead(
        &mut self,
        ranges: &[ByteRange],
        la_steps: &[PatternStep],
        positive: bool,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        let greedy_loop = self.asm.new_dynamic_label();
        let greedy_done = self.asm.new_dynamic_label();
        let try_la = self.asm.new_dynamic_label();
        let la_failed = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();

        dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>fail_label ; ldrb w0, [x19, x22]);
        self.emit_range_check(ranges, fail_label)?;
        dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; mov x9, x22);
        dynasm!(self.asm ; .arch aarch64 ; =>greedy_loop ; cmp x22, x20 ; b.ge =>greedy_done ; ldrb w0, [x19, x22]);
        self.emit_range_check(ranges, greedy_done)?;
        dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>greedy_loop ; =>greedy_done ; =>try_la ; mov x10, x22);

        let la_match = self.asm.new_dynamic_label();
        let la_mismatch = self.asm.new_dynamic_label();
        for step in la_steps {
            match step {
                PatternStep::Byte(b) => {
                    dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>la_mismatch ; ldrb w0, [x19, x22] ; cmp w0, *b as u32 ; b.ne =>la_mismatch ; add x22, x22, 1);
                }
                PatternStep::ByteClass(bc) => {
                    dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>la_mismatch ; ldrb w0, [x19, x22]);
                    self.emit_range_check(&bc.ranges, la_mismatch)?;
                    dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1);
                }
                PatternStep::EndOfText => {
                    dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ne =>la_mismatch);
                }
                _ => {}
            }
        }
        dynasm!(self.asm ; .arch aarch64 ; b =>la_match ; =>la_mismatch);
        if positive {
            dynasm!(self.asm ; .arch aarch64 ; mov x22, x10 ; b =>la_failed ; =>la_match ; mov x22, x10 ; b =>success);
        } else {
            dynasm!(self.asm ; .arch aarch64 ; mov x22, x10 ; b =>success ; =>la_match ; mov x22, x10 ; b =>la_failed);
        }
        dynasm!(self.asm ; .arch aarch64 ; =>la_failed ; sub x22, x22, 1 ; cmp x22, x9 ; b.lo =>fail_label ; b =>try_la ; =>success);
        Ok(())
    }

    fn emit_greedy_star_with_lookahead(
        &mut self,
        ranges: &[ByteRange],
        la_steps: &[PatternStep],
        positive: bool,
        fail_label: dynasmrt::DynamicLabel,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        let greedy_loop = self.asm.new_dynamic_label();
        let greedy_done = self.asm.new_dynamic_label();
        let try_la = self.asm.new_dynamic_label();
        let la_failed = self.asm.new_dynamic_label();
        let success = self.asm.new_dynamic_label();

        dynasm!(self.asm ; .arch aarch64 ; mov x9, x22);
        dynasm!(self.asm ; .arch aarch64 ; =>greedy_loop ; cmp x22, x20 ; b.ge =>greedy_done ; ldrb w0, [x19, x22]);
        self.emit_range_check(ranges, greedy_done)?;
        dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>greedy_loop ; =>greedy_done ; =>try_la ; mov x10, x22);

        let la_match = self.asm.new_dynamic_label();
        let la_mismatch = self.asm.new_dynamic_label();
        for step in la_steps {
            match step {
                PatternStep::Byte(b) => {
                    dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>la_mismatch ; ldrb w0, [x19, x22] ; cmp w0, *b as u32 ; b.ne =>la_mismatch ; add x22, x22, 1);
                }
                PatternStep::ByteClass(bc) => {
                    dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>la_mismatch ; ldrb w0, [x19, x22]);
                    self.emit_range_check(&bc.ranges, la_mismatch)?;
                    dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1);
                }
                PatternStep::EndOfText => {
                    dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ne =>la_mismatch);
                }
                _ => {}
            }
        }
        dynasm!(self.asm ; .arch aarch64 ; b =>la_match ; =>la_mismatch);
        if positive {
            dynasm!(self.asm ; .arch aarch64 ; mov x22, x10 ; b =>la_failed ; =>la_match ; mov x22, x10 ; b =>success);
        } else {
            dynasm!(self.asm ; .arch aarch64 ; mov x22, x10 ; b =>success ; =>la_match ; mov x22, x10 ; b =>la_failed);
        }
        dynasm!(self.asm ; .arch aarch64 ; =>la_failed ; cmp x22, x9 ; b.ls =>fail_label ; sub x22, x22, 1 ; b =>try_la ; =>success);
        Ok(())
    }

    fn emit_captures_fn(&mut self, steps: &[PatternStep]) -> Result<dynasmrt::AssemblyOffset> {
        use dynasmrt::DynasmLabelApi;
        let offset = self.asm.offset();
        let min_len = Self::calc_min_len(steps);
        let max_cap_idx = steps
            .iter()
            .filter_map(|s| match s {
                PatternStep::CaptureStart(i) | PatternStep::CaptureEnd(i) => Some(*i),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let num_slots = (max_cap_idx as usize + 1) * 2;

        // Prologue: x0=input, x1=len, x2=ctx, x3=captures
        dynasm!(self.asm
            ; .arch aarch64
            ; stp x29, x30, [sp, -16]!
            ; mov x29, sp
            ; stp x19, x20, [sp, -16]!
            ; stp x21, x22, [sp, -16]!
            ; stp x23, x24, [sp, -16]!
            ; mov x19, x0
            ; mov x20, x1
            ; mov x23, x3  // captures ptr
            ; mov x21, xzr
        );

        // Init captures to -1
        for slot in 0..num_slots {
            let off = (slot * 8) as u32;
            dynasm!(self.asm ; .arch aarch64 ; movn x0, 0 ; str x0, [x23, off]);
        }

        let start_loop = self.asm.new_dynamic_label();
        let match_found = self.asm.new_dynamic_label();
        let no_match = self.asm.new_dynamic_label();
        let byte_mismatch = self.asm.new_dynamic_label();

        dynasm!(self.asm
            ; .arch aarch64
            ; =>start_loop
            ; sub x0, x20, x21
            ; cmp x0, min_len as u32
            ; b.lo =>no_match
            ; mov x22, x21
            ; str x21, [x23]  // group 0 start
        );

        for step in steps {
            self.emit_capture_step(step, byte_mismatch, num_slots)?;
        }

        dynasm!(self.asm ; .arch aarch64 ; b =>match_found);
        dynasm!(self.asm ; .arch aarch64 ; =>byte_mismatch);
        for slot in 0..num_slots {
            let off = (slot * 8) as u32;
            dynasm!(self.asm ; .arch aarch64 ; movn x0, 0 ; str x0, [x23, off]);
        }
        dynasm!(self.asm ; .arch aarch64 ; add x21, x21, 1 ; b =>start_loop);

        dynasm!(self.asm
            ; .arch aarch64
            ; =>match_found
            ; str x22, [x23, 8]  // group 0 end
            ; lsl x0, x21, 32
            ; orr x0, x0, x22
            ; ldp x23, x24, [sp], 16
            ; ldp x21, x22, [sp], 16
            ; ldp x19, x20, [sp], 16
            ; ldp x29, x30, [sp], 16
            ; ret
        );
        dynasm!(self.asm
            ; .arch aarch64
            ; =>no_match
            ; movn x0, 0
            ; ldp x23, x24, [sp], 16
            ; ldp x21, x22, [sp], 16
            ; ldp x19, x20, [sp], 16
            ; ldp x29, x30, [sp], 16
            ; ret
        );
        Ok(offset)
    }

    fn emit_capture_step(
        &mut self,
        step: &PatternStep,
        fail_label: dynasmrt::DynamicLabel,
        _num_slots: usize,
    ) -> Result<()> {
        use dynasmrt::DynasmLabelApi;
        match step {
            PatternStep::Byte(b) => {
                dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>fail_label ; ldrb w0, [x19, x22] ; cmp w0, *b as u32 ; b.ne =>fail_label ; add x22, x22, 1);
            }
            PatternStep::ByteClass(bc) => {
                dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>fail_label ; ldrb w0, [x19, x22]);
                self.emit_range_check(&bc.ranges, fail_label)?;
                dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1);
            }
            PatternStep::GreedyPlus(bc) => {
                let ls = self.asm.new_dynamic_label();
                let ld = self.asm.new_dynamic_label();
                dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>fail_label ; ldrb w0, [x19, x22]);
                self.emit_range_check(&bc.ranges, fail_label)?;
                dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; =>ls ; cmp x22, x20 ; b.ge =>ld ; ldrb w0, [x19, x22]);
                self.emit_range_check(&bc.ranges, ld)?;
                dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>ls ; =>ld);
            }
            PatternStep::GreedyStar(bc) => {
                let ls = self.asm.new_dynamic_label();
                let ld = self.asm.new_dynamic_label();
                dynasm!(self.asm ; .arch aarch64 ; =>ls ; cmp x22, x20 ; b.ge =>ld ; ldrb w0, [x19, x22]);
                self.emit_range_check(&bc.ranges, ld)?;
                dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>ls ; =>ld);
            }
            PatternStep::CaptureStart(idx) => {
                let off = ((*idx as usize) * 2 * 8) as u32;
                dynasm!(self.asm ; .arch aarch64 ; str x22, [x23, off]);
            }
            PatternStep::CaptureEnd(idx) => {
                let off = ((*idx as usize) * 2 * 8 + 8) as u32;
                dynasm!(self.asm ; .arch aarch64 ; str x22, [x23, off]);
            }
            PatternStep::Alt(alts) => {
                let success = self.asm.new_dynamic_label();
                let alt_fail = self.asm.new_dynamic_label();
                dynasm!(self.asm ; .arch aarch64 ; str x22, [sp, -16]!);
                for (ai, alt_steps) in alts.iter().enumerate() {
                    let is_last = ai == alts.len() - 1;
                    let try_next = if is_last {
                        alt_fail
                    } else {
                        self.asm.new_dynamic_label()
                    };
                    for s in alt_steps {
                        self.emit_capture_step(s, try_next, _num_slots)?;
                    }
                    dynasm!(self.asm ; .arch aarch64 ; add sp, sp, 16 ; b =>success);
                    if !is_last {
                        dynasm!(self.asm ; .arch aarch64 ; =>try_next ; ldr x22, [sp]);
                    }
                }
                dynasm!(self.asm ; .arch aarch64 ; =>alt_fail ; add sp, sp, 16 ; b =>fail_label);
                dynasm!(self.asm ; .arch aarch64 ; =>success);
            }
            PatternStep::CodepointClass(cp, _) => {
                self.emit_codepoint_class_check(cp, fail_label)?
            }
            PatternStep::GreedyCodepointPlus(cp) => {
                self.emit_greedy_codepoint_plus(cp, fail_label)?
            }
            PatternStep::WordBoundary => self.emit_word_boundary_check(fail_label, true)?,
            PatternStep::NotWordBoundary => self.emit_word_boundary_check(fail_label, false)?,
            PatternStep::PositiveLookahead(inner) => {
                self.emit_standalone_lookahead(inner, fail_label, true)?
            }
            PatternStep::NegativeLookahead(inner) => {
                self.emit_standalone_lookahead(inner, fail_label, false)?
            }
            PatternStep::PositiveLookbehind(inner, ml) => {
                self.emit_lookbehind_check(inner, *ml, fail_label, true)?
            }
            PatternStep::NegativeLookbehind(inner, ml) => {
                self.emit_lookbehind_check(inner, *ml, fail_label, false)?
            }
            PatternStep::StartOfText => {
                dynasm!(self.asm ; .arch aarch64 ; cbnz x22, =>fail_label);
            }
            PatternStep::EndOfText => {
                dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ne =>fail_label);
            }
            PatternStep::StartOfLine => {
                let at_start = self.asm.new_dynamic_label();
                dynasm!(self.asm ; .arch aarch64 ; cbz x22, =>at_start ; sub x1, x22, 1 ; ldrb w0, [x19, x1] ; cmp w0, 0x0A ; b.ne =>fail_label ; =>at_start);
            }
            PatternStep::EndOfLine => {
                let at_end = self.asm.new_dynamic_label();
                dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.eq =>at_end ; ldrb w0, [x19, x22] ; cmp w0, 0x0A ; b.ne =>fail_label ; =>at_end);
            }
            PatternStep::Backref(idx) => {
                let idx = *idx as usize;
                let start_off = (idx * 2 * 8) as u32;
                let end_off = (idx * 2 * 8 + 8) as u32;
                let backref_match = self.asm.new_dynamic_label();
                dynasm!(self.asm
                    ; .arch aarch64
                    ; ldr x8, [x23, start_off]
                    ; ldr x9, [x23, end_off]
                    ; tst x8, x8
                    ; b.mi =>fail_label
                    ; tst x9, x9
                    ; b.mi =>fail_label
                    ; sub x10, x9, x8  // cap_len
                    ; cbz x10, =>backref_match
                    ; add x0, x22, x10
                    ; cmp x0, x20
                    ; b.hi =>fail_label
                );
                // Compare bytes
                let cmp_loop = self.asm.new_dynamic_label();
                let cmp_done = self.asm.new_dynamic_label();
                dynasm!(self.asm
                    ; .arch aarch64
                    ; mov x11, x8   // cap_start
                    ; mov x12, x22  // current
                    ; =>cmp_loop
                    ; cmp x11, x9
                    ; b.ge =>cmp_done
                    ; ldrb w0, [x19, x11]
                    ; ldrb w1, [x19, x12]
                    ; cmp w0, w1
                    ; b.ne =>fail_label
                    ; add x11, x11, 1
                    ; add x12, x12, 1
                    ; b =>cmp_loop
                    ; =>cmp_done
                    ; add x22, x22, x10
                    ; =>backref_match
                );
            }
            PatternStep::NonGreedyPlus(bc, suf) => {
                let try_suf = self.asm.new_dynamic_label();
                let consume = self.asm.new_dynamic_label();
                let matched = self.asm.new_dynamic_label();
                dynasm!(self.asm ; .arch aarch64 ; cmp x22, x20 ; b.ge =>fail_label ; ldrb w0, [x19, x22]);
                self.emit_range_check(&bc.ranges, fail_label)?;
                dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; =>try_suf);
                self.emit_non_greedy_suffix_check(suf, consume, matched)?;
                dynasm!(self.asm ; .arch aarch64 ; b =>matched ; =>consume ; cmp x22, x20 ; b.ge =>fail_label ; ldrb w0, [x19, x22]);
                self.emit_range_check(&bc.ranges, fail_label)?;
                dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>try_suf ; =>matched);
            }
            PatternStep::NonGreedyStar(bc, suf) => {
                let try_suf = self.asm.new_dynamic_label();
                let consume = self.asm.new_dynamic_label();
                let matched = self.asm.new_dynamic_label();
                dynasm!(self.asm ; .arch aarch64 ; =>try_suf);
                self.emit_non_greedy_suffix_check(suf, consume, matched)?;
                dynasm!(self.asm ; .arch aarch64 ; b =>matched ; =>consume ; cmp x22, x20 ; b.ge =>fail_label ; ldrb w0, [x19, x22]);
                self.emit_range_check(&bc.ranges, fail_label)?;
                dynasm!(self.asm ; .arch aarch64 ; add x22, x22, 1 ; b =>try_suf ; =>matched);
            }
            PatternStep::GreedyPlusLookahead(bc, la, pos) => {
                self.emit_greedy_plus_with_lookahead(&bc.ranges, la, *pos, fail_label)?
            }
            PatternStep::GreedyStarLookahead(bc, la, pos) => {
                self.emit_greedy_star_with_lookahead(&bc.ranges, la, *pos, fail_label)?
            }
        }
        Ok(())
    }

    fn extract_pattern_steps(&self) -> Vec<PatternStep> {
        let mut visited = vec![false; self.nfa.states.len()];
        self.extract_from_state(self.nfa.start, &mut visited, None)
    }

    fn extract_from_state(
        &self,
        start: StateId,
        visited: &mut [bool],
        end_state: Option<StateId>,
    ) -> Vec<PatternStep> {
        let mut steps = Vec::new();
        let mut current = start;
        loop {
            if let Some(end) = end_state {
                if current == end {
                    break;
                }
            }
            let state = &self.nfa.states[current as usize];
            if let Some(ref instr) = state.instruction {
                match instr {
                    NfaInstruction::CaptureStart(i) => steps.push(PatternStep::CaptureStart(*i)),
                    NfaInstruction::CaptureEnd(i) => steps.push(PatternStep::CaptureEnd(*i)),
                    NfaInstruction::CodepointClass(cp, t) => {
                        let ts = &self.nfa.states[*t as usize];
                        if ts.epsilon.len() == 2 && ts.transitions.is_empty() {
                            let (e0, e1) = (ts.epsilon[0], ts.epsilon[1]);
                            if e0 == current {
                                steps.push(PatternStep::GreedyCodepointPlus(cp.clone()));
                                visited[current as usize] = true;
                                visited[*t as usize] = true;
                                current = e1;
                                continue;
                            } else if e1 == current {
                                steps.push(PatternStep::GreedyCodepointPlus(cp.clone()));
                                visited[current as usize] = true;
                                visited[*t as usize] = true;
                                current = e0;
                                continue;
                            }
                        }
                        steps.push(PatternStep::CodepointClass(cp.clone(), *t));
                        current = *t;
                        continue;
                    }
                    NfaInstruction::Backref(i) => {
                        steps.push(PatternStep::Backref(*i));
                        if state.epsilon.len() == 1 {
                            visited[current as usize] = true;
                            current = state.epsilon[0];
                            continue;
                        } else if state.epsilon.is_empty() && state.is_match {
                            break;
                        } else {
                            return Vec::new();
                        }
                    }
                    NfaInstruction::PositiveLookahead(inner) => {
                        let inner_steps = self.extract_lookaround_steps(inner);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        steps.push(PatternStep::PositiveLookahead(inner_steps));
                    }
                    NfaInstruction::NegativeLookahead(inner) => {
                        let inner_steps = self.extract_lookaround_steps(inner);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        steps.push(PatternStep::NegativeLookahead(inner_steps));
                    }
                    NfaInstruction::PositiveLookbehind(inner) => {
                        let inner_steps = self.extract_lookaround_steps(inner);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        let ml = Self::calc_min_len(&inner_steps);
                        steps.push(PatternStep::PositiveLookbehind(inner_steps, ml));
                    }
                    NfaInstruction::NegativeLookbehind(inner) => {
                        let inner_steps = self.extract_lookaround_steps(inner);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        let ml = Self::calc_min_len(&inner_steps);
                        steps.push(PatternStep::NegativeLookbehind(inner_steps, ml));
                    }
                    NfaInstruction::WordBoundary => steps.push(PatternStep::WordBoundary),
                    NfaInstruction::NotWordBoundary => steps.push(PatternStep::NotWordBoundary),
                    NfaInstruction::StartOfText => steps.push(PatternStep::StartOfText),
                    NfaInstruction::EndOfText => steps.push(PatternStep::EndOfText),
                    NfaInstruction::StartOfLine => steps.push(PatternStep::StartOfLine),
                    NfaInstruction::EndOfLine => steps.push(PatternStep::EndOfLine),
                    NfaInstruction::NonGreedyExit => {}
                }
            }
            if state.is_match {
                break;
            }
            if !state.transitions.is_empty() {
                let target = state.transitions[0].1;
                if !state.transitions.iter().all(|(_, t)| *t == target) {
                    return Vec::new();
                }
                let ranges: Vec<ByteRange> =
                    state.transitions.iter().map(|(r, _)| r.clone()).collect();
                let ts = &self.nfa.states[target as usize];
                if ts.epsilon.len() == 2 && ts.transitions.is_empty() {
                    let (e0, e1) = (ts.epsilon[0], ts.epsilon[1]);
                    if e0 == current {
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        current = e1;
                        visited[target as usize] = true;
                        continue;
                    }
                    let ms = &self.nfa.states[e0 as usize];
                    if e1 == current
                        && ms.transitions.is_empty()
                        && ms.epsilon.len() == 1
                        && matches!(ms.instruction, Some(NfaInstruction::NonGreedyExit))
                    {
                        let exit = ms.epsilon[0];
                        if let Some(suf) = self.extract_single_step(exit) {
                            steps.push(PatternStep::NonGreedyPlus(
                                ByteClass::new(ranges),
                                Box::new(suf),
                            ));
                            visited[target as usize] = true;
                            visited[e0 as usize] = true;
                            visited[exit as usize] = true;
                            current = self.advance_past_step(exit);
                            continue;
                        }
                        return Vec::new();
                    }
                }
                if visited[current as usize] {
                    return Vec::new();
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
            if state.epsilon.len() == 1 && state.transitions.is_empty() {
                if visited[current as usize] {
                    return Vec::new();
                }
                visited[current as usize] = true;
                current = state.epsilon[0];
                continue;
            }
            if state.epsilon.len() > 1 && state.transitions.is_empty() {
                if state.epsilon.len() == 2 {
                    let e0s = &self.nfa.states[state.epsilon[0] as usize];
                    if e0s.transitions.is_empty()
                        && e0s.epsilon.len() == 1
                        && matches!(e0s.instruction, Some(NfaInstruction::NonGreedyExit))
                    {
                        let ps = state.epsilon[1];
                        let pst = &self.nfa.states[ps as usize];
                        if !pst.transitions.is_empty() {
                            let t = pst.transitions[0].1;
                            if pst.transitions.iter().all(|(_, tt)| *tt == t) {
                                let ranges: Vec<ByteRange> =
                                    pst.transitions.iter().map(|(r, _)| r.clone()).collect();
                                let exit = e0s.epsilon[0];
                                if let Some(suf) = self.extract_single_step(exit) {
                                    steps.push(PatternStep::NonGreedyStar(
                                        ByteClass::new(ranges),
                                        Box::new(suf),
                                    ));
                                    visited[current as usize] = true;
                                    visited[state.epsilon[0] as usize] = true;
                                    visited[ps as usize] = true;
                                    visited[exit as usize] = true;
                                    current = self.advance_past_step(exit);
                                    continue;
                                }
                            }
                        }
                        return Vec::new();
                    }
                }
                let common_end = self.find_alternation_end(current);
                if common_end.is_none() {
                    return Vec::new();
                }
                let ce = common_end.unwrap();
                let mut alts = Vec::new();
                for &alt_start in &state.epsilon {
                    let mut av = visited.to_vec();
                    let alt_steps = self.extract_from_state(alt_start, &mut av, Some(ce));
                    if alt_steps.is_empty() && !self.is_trivial_path(alt_start, ce) {
                        return Vec::new();
                    }
                    alts.push(alt_steps);
                }
                steps.push(PatternStep::Alt(alts));
                visited[current as usize] = true;
                current = ce;
                continue;
            }
            if state.transitions.is_empty() && state.epsilon.is_empty() {
                break;
            }
            return Vec::new();
        }
        steps
    }

    fn extract_lookaround_steps(&self, inner: &Nfa) -> Vec<PatternStep> {
        let mut visited = vec![false; inner.states.len()];
        let mut steps = Vec::new();
        let mut current = inner.start;
        loop {
            if current as usize >= inner.states.len() {
                return Vec::new();
            }
            let state = &inner.states[current as usize];
            if state.is_match {
                break;
            }
            if let Some(ref instr) = state.instruction {
                match instr {
                    NfaInstruction::WordBoundary => steps.push(PatternStep::WordBoundary),
                    NfaInstruction::EndOfText => steps.push(PatternStep::EndOfText),
                    NfaInstruction::StartOfText => steps.push(PatternStep::StartOfText),
                    _ => return Vec::new(),
                }
            }
            if !state.transitions.is_empty() {
                let t = state.transitions[0].1;
                if !state.transitions.iter().all(|(_, tt)| *tt == t) {
                    return Vec::new();
                }
                let ranges: Vec<ByteRange> =
                    state.transitions.iter().map(|(r, _)| r.clone()).collect();
                let ts = &inner.states[t as usize];
                if ts.transitions.is_empty() && ts.epsilon.len() == 2 {
                    let (e0, e1) = (ts.epsilon[0], ts.epsilon[1]);
                    if e0 == current {
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        if visited[t as usize] {
                            return Vec::new();
                        }
                        visited[t as usize] = true;
                        current = e1;
                        continue;
                    } else if e1 == current {
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        if visited[t as usize] {
                            return Vec::new();
                        }
                        visited[t as usize] = true;
                        current = e0;
                        continue;
                    }
                }
                if visited[current as usize] {
                    return Vec::new();
                }
                visited[current as usize] = true;
                if ranges.len() == 1 && ranges[0].start == ranges[0].end {
                    steps.push(PatternStep::Byte(ranges[0].start));
                } else {
                    steps.push(PatternStep::ByteClass(ByteClass::new(ranges)));
                }
                current = t;
                continue;
            }
            if state.epsilon.len() == 1 && state.transitions.is_empty() {
                if visited[current as usize] {
                    return Vec::new();
                }
                visited[current as usize] = true;
                current = state.epsilon[0];
                continue;
            }
            if state.epsilon.len() == 2 && state.transitions.is_empty() {
                let (e0, e1) = (state.epsilon[0], state.epsilon[1]);
                if let Some((r, exit)) =
                    self.detect_greedy_star_lookaround(inner, current, e0, e1, &visited)
                {
                    steps.push(PatternStep::GreedyStar(ByteClass::new(r)));
                    visited[current as usize] = true;
                    current = exit;
                    continue;
                }
                if let Some((r, exit)) =
                    self.detect_greedy_star_lookaround(inner, current, e1, e0, &visited)
                {
                    steps.push(PatternStep::GreedyStar(ByteClass::new(r)));
                    visited[current as usize] = true;
                    current = exit;
                    continue;
                }
                return Vec::new();
            }
            if !state.epsilon.is_empty() || !state.transitions.is_empty() {
                return Vec::new();
            }
            break;
        }
        steps
    }

    fn detect_greedy_star_lookaround(
        &self,
        inner: &Nfa,
        branch: StateId,
        loop_start: StateId,
        exit: StateId,
        visited: &[bool],
    ) -> Option<(Vec<ByteRange>, StateId)> {
        if loop_start as usize >= inner.states.len() {
            return None;
        }
        let ls = &inner.states[loop_start as usize];
        if ls.transitions.is_empty() {
            return None;
        }
        let t = ls.transitions[0].1;
        if !ls.transitions.iter().all(|(_, tt)| *tt == t) {
            return None;
        }
        let ranges: Vec<ByteRange> = ls.transitions.iter().map(|(r, _)| r.clone()).collect();
        let ts = &inner.states[t as usize];
        if ts.epsilon.len() == 1 {
            let back = ts.epsilon[0];
            if (back == branch || back == loop_start) && !visited[loop_start as usize] {
                return Some((ranges, exit));
            }
        }
        if ts.epsilon.len() == 2 {
            let (e0, e1) = (ts.epsilon[0], ts.epsilon[1]);
            let (back, fwd) = if e0 == branch || e0 == loop_start {
                (e0, e1)
            } else if e1 == branch || e1 == loop_start {
                (e1, e0)
            } else {
                return None;
            };
            let _ = back;
            if fwd == exit && !visited[loop_start as usize] {
                return Some((ranges, exit));
            }
        }
        None
    }

    fn find_alternation_end(&self, start: StateId) -> Option<StateId> {
        self.find_alternation_end_depth(start, 0)
    }

    fn find_alternation_end_depth(&self, start: StateId, depth: usize) -> Option<StateId> {
        if depth > 20 {
            return None;
        }
        let state = &self.nfa.states[start as usize];
        if state.epsilon.len() < 2 {
            return None;
        }
        let mut ends = Vec::new();
        for &alt_start in &state.epsilon {
            if let Some(e) = self.trace_to_merge_depth(alt_start, start, depth) {
                ends.push(e);
            } else {
                return None;
            }
        }
        if ends.is_empty() {
            return None;
        }
        let first = ends[0];
        if ends.iter().all(|&e| e == first) {
            Some(first)
        } else {
            None
        }
    }

    fn trace_to_merge_depth(
        &self,
        start: StateId,
        alt_start: StateId,
        depth: usize,
    ) -> Option<StateId> {
        if depth > 20 {
            return None;
        }
        let mut current = start;
        let mut visited = vec![false; self.nfa.states.len()];
        visited[alt_start as usize] = true;
        for _ in 0..200 {
            if visited[current as usize] {
                return None;
            }
            visited[current as usize] = true;
            let state = &self.nfa.states[current as usize];
            if state.is_match {
                return Some(current);
            }
            if let Some(NfaInstruction::CodepointClass(_, t)) = &state.instruction {
                current = *t;
                continue;
            }
            if state.transitions.is_empty() && state.epsilon.is_empty() {
                return Some(current);
            }
            if state.epsilon.len() == 1 && state.transitions.is_empty() {
                current = state.epsilon[0];
                continue;
            }
            if !state.transitions.is_empty() && state.epsilon.is_empty() {
                current = state.transitions[0].1;
                continue;
            }
            if !state.transitions.is_empty() && state.epsilon.len() == 1 {
                current = state.transitions[0].1;
                continue;
            }
            if state.epsilon.len() >= 2 && state.transitions.is_empty() {
                let mut fwd = Vec::new();
                for &e in &state.epsilon {
                    if !visited[e as usize] {
                        fwd.push(e);
                    }
                }
                if fwd.len() == 1 {
                    current = fwd[0];
                    continue;
                }
                if let Some(ne) = self.find_alternation_end_depth(current, depth + 1) {
                    current = ne;
                    continue;
                }
                return None;
            }
            return None;
        }
        None
    }

    fn is_trivial_path(&self, start: StateId, end: StateId) -> bool {
        self.is_trivial_path_depth(start, end, 0)
    }

    fn is_trivial_path_depth(&self, start: StateId, end: StateId, depth: usize) -> bool {
        if depth > 100 {
            return false;
        }
        if start == end {
            return true;
        }
        let state = &self.nfa.states[start as usize];
        if state.epsilon.len() == 1 && state.transitions.is_empty() {
            return state.epsilon[0] == end
                || self.is_trivial_path_depth(state.epsilon[0], end, depth + 1);
        }
        false
    }

    fn extract_single_step(&self, state_id: StateId) -> Option<PatternStep> {
        let mut current = state_id;
        loop {
            let state = &self.nfa.states[current as usize];
            if !state.transitions.is_empty() {
                let t = state.transitions[0].1;
                if !state.transitions.iter().all(|(_, tt)| *tt == t) {
                    return None;
                }
                let ranges: Vec<ByteRange> =
                    state.transitions.iter().map(|(r, _)| r.clone()).collect();
                return if ranges.len() == 1 && ranges[0].start == ranges[0].end {
                    Some(PatternStep::Byte(ranges[0].start))
                } else {
                    Some(PatternStep::ByteClass(ByteClass::new(ranges)))
                };
            }
            if state.epsilon.len() == 1 && state.transitions.is_empty() {
                current = state.epsilon[0];
                continue;
            }
            if state.is_match || state.epsilon.len() > 1 {
                return None;
            }
            return None;
        }
    }

    fn advance_past_step(&self, state_id: StateId) -> StateId {
        let mut current = state_id;
        loop {
            let state = &self.nfa.states[current as usize];
            if !state.transitions.is_empty() {
                return state.transitions[0].1;
            }
            if state.epsilon.len() == 1 {
                current = state.epsilon[0];
                continue;
            }
            return current;
        }
    }

    fn finalize(
        self,
        find_offset: dynasmrt::AssemblyOffset,
        captures_offset: dynasmrt::AssemblyOffset,
        find_needs_ctx: bool,
        fallback_steps: Option<Vec<PatternStep>>,
    ) -> Result<TaggedNfaJit> {
        let code = self
            .asm
            .finalize()
            .map_err(|e| Error::new(ErrorKind::Jit(format!("Failed to finalize: {:?}", e)), ""))?;
        let find_fn: extern "C" fn(*const u8, usize, *mut TaggedNfaContext) -> i64 =
            unsafe { std::mem::transmute(code.ptr(find_offset)) };
        let captures_fn: extern "C" fn(*const u8, usize, *mut TaggedNfaContext, *mut i64) -> i64 =
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
