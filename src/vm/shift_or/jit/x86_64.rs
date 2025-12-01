//! x86-64 code generation for Shift-Or JIT.

use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

use super::super::ShiftOr;
use super::jit::JitShiftOr;

/// Compiler for Shift-Or JIT on x86-64.
pub(super) struct ShiftOrJitCompiler;

impl ShiftOrJitCompiler {
    /// Compiles a ShiftOr matcher to native code.
    pub(super) fn compile(shift_or: &ShiftOr) -> Option<JitShiftOr> {
        let mut ops = dynasmrt::x64::Assembler::new().ok()?;

        // Copy masks to heap (needs stable address for JIT code)
        let mut masks = Box::new([0u64; 256]);
        for (i, m) in shift_or.masks.iter().enumerate() {
            masks[i] = *m;
        }

        // Copy follow sets to heap (needs stable address for JIT code)
        let mut follow = Box::new([0u64; 64]);
        for (i, f) in shift_or.follow.iter().enumerate() {
            if i < 64 {
                follow[i] = *f;
            }
        }

        let find_offset = Self::emit_find(&mut ops, shift_or);

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
        ))
    }

    fn emit_find(ops: &mut dynasmrt::x64::Assembler, shift_or: &ShiftOr) -> dynasmrt::AssemblyOffset {
        // Function signature: fn(input: *const u8, len: usize, masks: *const u64, follow: *const u64, accept: u64, first: u64) -> i64
        // Returns: packed (start << 32 | end) on match, or -1 if no match
        //
        // Register allocation (System V AMD64 ABI):
        //   rdi = input pointer
        //   rsi = input length
        //   rdx = masks pointer
        //   rcx = follow pointer
        //   r8  = accept mask
        //   r9  = first mask
        //
        // OPTIMIZED Working registers (keep hot values in registers, not stack):
        //   r10 = current start position being tried
        //   r11 = state (inverted: 0 = active position)
        //   r12 = follow pointer (kept in register for fast inner loop)
        //   r13 = accept mask (kept in register - checked every iteration)
        //   r14 = saved input pointer
        //   r15 = saved len
        //   rbx = saved masks pointer
        //   [rsp+0] = first mask (only used at start of each position)
        //   [rsp+8] = last_match_start
        //   [rsp+16] = last_match_end

        let offset = ops.offset();
        let _ = shift_or.position_count;

        dynasm!(ops
            ; .arch x64
            // Prologue - save callee-saved registers
            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15
            ; sub rsp, 24           // Allocate stack space for saved values

            // Save arguments we'll need later - OPTIMIZED register allocation
            ; mov r14, rdi           // r14 = input
            ; mov r15, rsi           // r15 = len
            ; mov rbx, rdx           // rbx = masks
            ; mov r12, rcx           // r12 = follow (HOT - keep in register!)
            ; mov r13, r8            // r13 = accept (HOT - keep in register!)
            ; mov [rsp], r9          // [rsp] = first (only used at start)

            // Initialize - match state on stack (less frequently accessed)
            ; xor r10d, r10d         // r10 = start position = 0
            ; mov QWORD [rsp+16], -1 // last_match_end = -1
            ; mov QWORD [rsp+8], 0   // last_match_start = 0

            // Outer loop: try each start position
            ; ->start_loop:
            ; cmp r10, r15
            ; jae ->done

            // Initialize state for this start position
            // state = !first | mask[input[start]]
            ; mov r11, [rsp]         // r11 = first
            ; not r11                // r11 = !first
            ; movzx eax, BYTE [r14 + r10]  // byte = input[start]
            ; or r11, [rbx + rax*8]  // state |= mask[byte]

            // Check immediate match at first byte
            ; mov rax, r11
            ; or rax, r13            // rax = state | accept (accept now in register!)
            ; cmp rax, -1
            ; jne ->found_at_start

            // Inner loop: process remaining bytes
            ; lea rcx, [r10 + 1]     // rcx = pos = start + 1

            ; ->inner_loop:
            ; cmp rcx, r15
            ; jae ->inner_done

            // ============================================================
            // Glushkov follow set computation:
            // reachable = union of follow[i] for all active positions i
            // state = !reachable | mask[byte]
            //
            // Active positions have bit=0 in state (inverted logic)
            // So active = !state gives us 1 for active positions
            // ============================================================

            ; mov rax, r11           // rax = state
            ; not rax                // rax = active positions (1 = active)
            ; xor rdi, rdi           // rdi = reachable = 0

            // Iterate through set bits in rax (active positions)
            // r12 contains follow pointer - no stack access needed!
            ; ->follow_loop:
            ; test rax, rax
            ; jz ->follow_done

            // Get lowest set bit position using BSF
            ; bsf rsi, rax           // rsi = position of lowest set bit
            ; or rdi, [r12 + rsi*8]  // reachable |= follow[position] (r12 = follow!)
            ; blsr rax, rax          // Clear lowest set bit (rax &= rax - 1)
            ; jmp ->follow_loop

            ; ->follow_done:
            // Now rdi = reachable (positions that can be reached)
            // state = !reachable | mask[byte]
            ; mov r11, rdi           // r11 = reachable
            ; not r11                // r11 = !reachable
            ; movzx eax, BYTE [r14 + rcx]  // byte = input[pos]
            ; or r11, [rbx + rax*8]  // state |= mask[byte]

            // Check for match - accept is in r13, no stack access!
            ; mov rax, r11
            ; or rax, r13            // rax = state | accept
            ; cmp rax, -1
            ; jne ->found_in_loop

            // Check if dead state (all 1s = no active positions)
            ; cmp r11, -1
            ; je ->inner_done

            // Next byte
            ; inc rcx
            ; jmp ->inner_loop

            ; ->found_at_start:
            // Match found at start position (after first byte)
            ; mov [rsp+8], r10       // last_match_start = start
            ; lea rax, [r10 + 1]
            ; mov [rsp+16], rax      // last_match_end = start + 1
            // Continue to find longer match
            ; lea rcx, [r10 + 1]     // pos = start + 1
            ; jmp ->inner_loop

            ; ->found_in_loop:
            // Match found at position rcx
            ; mov [rsp+8], r10       // last_match_start = start
            ; lea rax, [rcx + 1]
            ; mov [rsp+16], rax      // last_match_end = pos + 1
            // Continue to find longest match
            ; inc rcx
            ; jmp ->inner_loop

            ; ->inner_done:
            // If we found a match, we're done (first match wins for unanchored)
            ; cmp QWORD [rsp+16], -1
            ; jne ->done

            // Try next start position
            ; inc r10
            ; jmp ->start_loop

            ; ->done:
            // Check if we have a match
            ; mov rdx, [rsp+16]      // rdx = last_match_end
            ; cmp rdx, -1
            ; je ->no_match

            // Pack result: (start << 32) | end
            ; mov rax, [rsp+8]       // rax = last_match_start
            ; shl rax, 32            // rax = start << 32
            ; or rax, rdx            // rax = (start << 32) | end
            ; jmp ->epilogue

            ; ->no_match:
            ; mov rax, -1

            ; ->epilogue:
            ; add rsp, 24            // Deallocate stack space
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; ret
        );

        offset
    }
}
