//! AArch64 (ARM64) code generation for Shift-Or JIT.

use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

use super::super::ShiftOr;
use super::jit::JitShiftOr;

// ARM64 Shift-Or JIT enabled
const ARM64_SHIFT_OR_JIT_ENABLED: bool = true;

/// Compiler for Shift-Or JIT on AArch64.
pub(super) struct ShiftOrJitCompiler;

impl ShiftOrJitCompiler {
    /// Compiles a ShiftOr matcher to native code.
    pub(super) fn compile(shift_or: &ShiftOr) -> Option<JitShiftOr> {
        // ARM64 Shift-Or JIT is disabled until assembly is fully debugged
        if !ARM64_SHIFT_OR_JIT_ENABLED {
            return None;
        }

        // Copy masks to heap FIRST (needs stable address for embedded pointer)
        let mut masks = Box::new([0u64; 256]);
        for (i, m) in shift_or.masks.iter().enumerate() {
            masks[i] = *m;
        }

        // Copy follow sets to heap FIRST (needs stable address for embedded pointer)
        let mut follow = Box::new([0u64; 64]);
        for (i, f) in shift_or.follow.iter().enumerate() {
            if i < 64 {
                follow[i] = *f;
            }
        }

        // Get stable pointers to embed in JIT code
        let masks_ptr = masks.as_ptr() as u64;
        let follow_ptr = follow.as_ptr() as u64;

        let mut ops = dynasmrt::aarch64::Assembler::new().ok()?;
        let find_offset = Self::emit_find(&mut ops, shift_or, masks_ptr, follow_ptr);

        // `finalize` panics on a failed final commit; commit first so an
        // out-of-range branch declines the JIT instead.
        ops.commit().ok()?;
        let code = ops.finalize().ok()?;

        Some(JitShiftOr::new(
            code,
            find_offset,
            masks,
            follow,
            shift_or.accept,
            shift_or.first,
            shift_or.position_count,
            shift_or.nullable,
            shift_or.has_leading_word_boundary,
            shift_or.has_trailing_word_boundary,
            shift_or.has_start_anchor,
            shift_or.has_end_anchor,
            shift_or.clone(),
        ))
    }

    fn emit_find(
        ops: &mut dynasmrt::aarch64::Assembler,
        shift_or: &ShiftOr,
        masks_ptr: u64,
        follow_ptr: u64,
    ) -> dynasmrt::AssemblyOffset {
        // Function signature: fn(input: *const u8, len: usize, accept: u64, first: u64) -> i64
        // Returns: packed (start << 32 | end) on match, -1 if no match, or
        // -(resume + 2) when the attempts ran out of budget: no match starts
        // before `resume`, and starts from `resume` on were not tried.
        //
        // AAPCS64 calling convention:
        //   x0 = input, x1 = len, x2 = accept, x3 = first
        //
        // Register allocation (all callee-saved for internal use):
        //   x19 = input pointer
        //   x20 = len
        //   x21 = accept mask
        //   x22 = current start position
        //   x23 = state (inverted: 0 = active position)
        //   x24 = follow pointer (embedded)
        //   x25 = masks pointer (embedded)
        //   x26 = first mask
        //   x27 = last_match_start
        //   x28 = last_match_end
        //
        // Temporary registers (caller-saved):
        //   x7 = bytes the failed attempts may still walk (4 * (len + 1))
        //   x9-x15 = scratch

        let offset = ops.offset();
        let _ = shift_or.position_count;

        // Split masks_ptr and follow_ptr into 16-bit chunks for movz/movk
        let masks_lo = (masks_ptr & 0xFFFF) as u32;
        let masks_16 = ((masks_ptr >> 16) & 0xFFFF) as u32;
        let masks_32 = ((masks_ptr >> 32) & 0xFFFF) as u32;
        let masks_48 = ((masks_ptr >> 48) & 0xFFFF) as u32;

        let follow_lo = (follow_ptr & 0xFFFF) as u32;
        let follow_16 = ((follow_ptr >> 16) & 0xFFFF) as u32;
        let follow_32 = ((follow_ptr >> 32) & 0xFFFF) as u32;
        let follow_48 = ((follow_ptr >> 48) & 0xFFFF) as u32;

        dynasm!(ops
            ; .arch aarch64

            // Prologue - save callee-saved registers
            ; stp x29, x30, [sp, #-16]!
            ; mov x29, sp
            ; stp x19, x20, [sp, #-16]!
            ; stp x21, x22, [sp, #-16]!
            ; stp x23, x24, [sp, #-16]!
            ; stp x25, x26, [sp, #-16]!
            ; stp x27, x28, [sp, #-16]!

            // Move arguments to callee-saved registers
            ; mov x19, x0              // x19 = input
            ; mov x20, x1              // x20 = len
            ; mov x21, x2              // x21 = accept
            ; mov x26, x3              // x26 = first

            // Load embedded pointers (64-bit immediates via movz/movk)
            ; movz x25, #masks_lo
            ; movk x25, #masks_16, lsl #16
            ; movk x25, #masks_32, lsl #32
            ; movk x25, #masks_48, lsl #48

            ; movz x24, #follow_lo
            ; movk x24, #follow_16, lsl #16
            ; movk x24, #follow_32, lsl #32
            ; movk x24, #follow_48, lsl #48

            // Initialize
            ; mov x22, #0              // x22 = start position = 0
            ; lsl x7, x20, #2          // x7 = 4 * len
            ; add x7, x7, #4           // x7 = walk budget = 4 * (len + 1)
            ; movn x28, 0              // x28 = last_match_end = -1
            ; mov x27, #0              // x27 = last_match_start = 0

            // Outer loop: try each start position
            ; ->start_loop:
            ; cmp x22, x20
            ; b.hs ->done

            // Initialize state for this start position
            // state = !first | mask[input[start]]
            ; mvn x23, x26             // x23 = !first
            ; ldrb w9, [x19, x22]      // w9 = input[start]
            ; lsl x9, x9, #3           // x9 = byte * 8 (offset into masks array)
            ; ldr x10, [x25, x9]       // x10 = mask[byte]
            ; orr x23, x23, x10        // state |= mask[byte]

            // Check immediate match at first byte
            ; orr x9, x23, x21         // x9 = state | accept
            ; cmn x9, #1               // compare with -1 (all 1s)
            ; b.ne ->found_at_start

            // The attempt's second byte, peeled out of the loop below: an
            // attempt that dies on it restarts without being charged to the
            // walk budget, which keeps the common short attempt unmetered.
            ; add x10, x22, #1         // x10 = pos = start + 1
            ; cmp x10, x20
            ; b.hs ->next_start
            ; mvn x9, x23              // x9 = active positions
            ; mov x11, #0              // x11 = reachable = 0
            ; ->follow_loop_first:
            ; cbz x9, ->follow_done_first
            ; rbit x12, x9
            ; clz x12, x12
            ; lsl x13, x12, #3
            ; ldr x14, [x24, x13]
            ; orr x11, x11, x14
            ; sub x15, x9, #1
            ; and x9, x9, x15
            ; b ->follow_loop_first
            ; ->follow_done_first:
            ; mvn x23, x11
            ; ldrb w9, [x19, x10]
            ; lsl x9, x9, #3
            ; ldr x12, [x25, x9]
            ; orr x23, x23, x12
            ; orr x9, x23, x21
            ; cmn x9, #1
            ; b.ne ->found_in_loop
            ; cmn x23, #1
            ; b.eq ->next_start
            ; add x10, x10, #1

            // Inner loop: process remaining bytes
            ; ->inner_loop:
            ; cmp x10, x20
            ; b.hs ->inner_done

            // Glushkov follow set computation:
            // reachable = union of follow[i] for all active positions i
            // state = !reachable | mask[byte]
            //
            // Active positions have bit=0 in state (inverted logic)
            // So active = !state gives us 1 for active positions

            ; mvn x9, x23              // x9 = active positions (1 = active)
            ; mov x11, #0              // x11 = reachable = 0

            // Iterate through set bits in x9 (active positions)
            ; ->follow_loop:
            ; cbz x9, ->follow_done

            // Get lowest set bit position using CLZ on reversed bits
            // ARM64 has RBIT (reverse bits) + CLZ to find trailing zeros
            ; rbit x12, x9             // x12 = bit-reversed x9
            ; clz x12, x12             // x12 = position of lowest set bit in x9
            ; lsl x13, x12, #3         // x13 = position * 8
            ; ldr x14, [x24, x13]      // x14 = follow[position]
            ; orr x11, x11, x14        // reachable |= follow[position]

            // Clear lowest set bit: x9 &= (x9 - 1)
            ; sub x15, x9, #1
            ; and x9, x9, x15
            ; b ->follow_loop

            ; ->follow_done:
            // Now x11 = reachable (positions that can be reached)
            // state = !reachable | mask[byte]
            ; mvn x23, x11             // x23 = !reachable
            ; ldrb w9, [x19, x10]      // w9 = input[pos]
            ; lsl x9, x9, #3           // x9 = byte * 8
            ; ldr x12, [x25, x9]       // x12 = mask[byte]
            ; orr x23, x23, x12        // state |= mask[byte]

            // Check for match - accept is in x21
            ; orr x9, x23, x21         // x9 = state | accept
            ; cmn x9, #1               // compare with -1
            ; b.ne ->found_in_loop

            // Check if dead state (all 1s = no active positions)
            ; cmn x23, #1
            ; b.eq ->inner_done

            // Next byte
            ; add x10, x10, #1
            ; b ->inner_loop

            ; ->found_at_start:
            // Match found at start position (after first byte)
            ; mov x27, x22             // last_match_start = start
            ; add x28, x22, #1         // last_match_end = start + 1
            // Continue to find longer match
            ; add x10, x22, #1         // pos = start + 1
            ; b ->inner_loop

            ; ->found_in_loop:
            // Match found at position x10
            ; mov x27, x22             // last_match_start = start
            ; add x28, x10, #1         // last_match_end = pos + 1
            // Continue to find longest match
            ; add x10, x10, #1
            ; b ->inner_loop

            ; ->inner_done:
            // If we found a match, we're done (first match wins for unanchored)
            ; cmn x28, #1
            ; b.ne ->done

            // Charge the failed attempt's walk to the budget. Past it, the
            // caller takes over with a linear-time search.
            ; sub x9, x10, x22         // x9 = bytes this attempt walked
            ; subs x7, x7, x9
            ; b.lo ->gave_up

            // Try next start position
            ; ->next_start:
            ; add x22, x22, #1
            ; b ->start_loop

            ; ->gave_up:
            // Return -(resume + 2) with resume = start + 1.
            ; add x0, x22, #3
            ; neg x0, x0
            ; b ->epilogue

            ; ->done:
            // Check if we have a match
            ; cmn x28, #1
            ; b.eq ->no_match

            // Pack result: (start << 32) | end
            ; lsl x0, x27, #32         // x0 = start << 32
            ; orr x0, x0, x28          // x0 = (start << 32) | end
            ; b ->epilogue

            ; ->no_match:
            ; movn x0, 0

            ; ->epilogue:
            // Restore callee-saved registers
            ; ldp x27, x28, [sp], #16
            ; ldp x25, x26, [sp], #16
            ; ldp x23, x24, [sp], #16
            ; ldp x21, x22, [sp], #16
            ; ldp x19, x20, [sp], #16
            ; ldp x29, x30, [sp], #16
            ; ret
        );

        offset
    }
}
