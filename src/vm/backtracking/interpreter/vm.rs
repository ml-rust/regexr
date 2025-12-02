//! Backtracking regex engine for fast capture extraction.
//!
//! This is a PCRE-style backtracking interpreter that uses bytecode compilation
//! for efficient execution. Unlike tree-walking interpreters, this compiles HIR
//! to a flat bytecode representation first, then executes with minimal overhead.

use crate::hir::{CodepointClass, Hir, HirAnchor, HirExpr};
use crate::nfa::{ByteClass, ByteRange};

use super::super::shared::{decode_utf8, is_word_byte, Op};

/// A compiled backtracking regex.
pub struct BacktrackingVm {
    /// Bytecode program.
    code: Vec<Op>,
    /// Number of capture groups (not slots).
    capture_count: u32,
    /// Large byte classes (for classes with >4 ranges).
    /// Uses ByteClass for fast O(1) bitmap lookup.
    byte_classes: Vec<ByteClass>,
    /// Large codepoint classes (for Unicode classes with multiple ranges).
    /// Uses CodepointClass for fast ASCII bitmap lookup.
    cp_classes: Vec<CodepointClass>,
}

impl BacktrackingVm {
    /// Creates a new backtracking VM from HIR.
    pub fn new(hir: &Hir) -> Self {
        let mut compiler = Compiler::new();
        compiler.compile(&hir.expr);
        compiler.emit(Op::Match);

        Self {
            code: compiler.code,
            capture_count: hir.props.capture_count,
            byte_classes: compiler.byte_classes,
            cp_classes: compiler.cp_classes,
        }
    }

    /// Returns the number of capture groups.
    pub fn capture_count(&self) -> u32 {
        self.capture_count
    }

    /// Finds the first match in the input.
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        let num_slots = (self.capture_count as usize + 1) * 2;
        let mut slots = vec![-1i32; num_slots];

        for start in 0..=input.len() {
            slots.fill(-1);
            if self.exec(input, start, &mut slots) {
                let s = slots[0];
                let e = slots[1];
                if s >= 0 && e >= 0 {
                    return Some((s as usize, e as usize));
                }
            }
        }
        None
    }

    /// Finds a match starting at the given position.
    pub fn find_at(&self, input: &[u8], start: usize) -> Option<(usize, usize)> {
        let num_slots = (self.capture_count as usize + 1) * 2;
        let mut slots = vec![-1i32; num_slots];

        for pos in start..=input.len() {
            slots.fill(-1);
            if self.exec(input, pos, &mut slots) {
                let s = slots[0];
                let e = slots[1];
                if s >= 0 && e >= 0 {
                    return Some((s as usize, e as usize));
                }
            }
        }
        None
    }

    /// Returns captures for the first match.
    pub fn captures(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        let num_slots = (self.capture_count as usize + 1) * 2;
        let mut slots = vec![-1i32; num_slots];

        for start in 0..=input.len() {
            slots.fill(-1);
            if self.exec(input, start, &mut slots) {
                return Some(self.extract_captures(&slots));
            }
        }
        None
    }

    /// Extract captures from slots.
    fn extract_captures(&self, slots: &[i32]) -> Vec<Option<(usize, usize)>> {
        let mut result = Vec::with_capacity(self.capture_count as usize + 1);
        for i in 0..=self.capture_count as usize {
            let s = slots[i * 2];
            let e = slots[i * 2 + 1];
            if s >= 0 && e >= 0 {
                result.push(Some((s as usize, e as usize)));
            } else {
                result.push(None);
            }
        }
        result
    }

    /// Execute the bytecode.
    #[inline(never)]
    fn exec(&self, input: &[u8], start: usize, slots: &mut [i32]) -> bool {
        // Backtrack stack: (pc, pos, saved_slots)
        // We use a more efficient representation: save only the slots that change
        let mut stack: Vec<(u32, usize)> = Vec::with_capacity(32);
        let mut slot_stack: Vec<(u16, i32)> = Vec::with_capacity(64);
        let mut slot_stack_frames: Vec<usize> = Vec::with_capacity(32);

        let mut pc = 0u32;
        let mut pos = start;

        // Set group 0 start
        slots[0] = start as i32;

        loop {
            if pc as usize >= self.code.len() {
                return false;
            }

            match self.code[pc as usize] {
                Op::Byte(b) => {
                    if pos < input.len() && input[pos] == b {
                        pos += 1;
                        pc += 1;
                    } else {
                        // Backtrack
                        if !self.backtrack(
                            &mut pc,
                            &mut pos,
                            &mut stack,
                            slots,
                            &mut slot_stack,
                            &mut slot_stack_frames,
                        ) {
                            return false;
                        }
                    }
                }

                Op::ByteRange(lo, hi) => {
                    if pos < input.len() && input[pos] >= lo && input[pos] <= hi {
                        pos += 1;
                        pc += 1;
                    } else if !self.backtrack(
                        &mut pc,
                        &mut pos,
                        &mut stack,
                        slots,
                        &mut slot_stack,
                        &mut slot_stack_frames,
                    ) {
                        return false;
                    }
                }

                Op::ByteRanges { count, ranges } => {
                    if pos < input.len() {
                        let b = input[pos];
                        let mut matched = false;
                        for i in 0..count as usize {
                            let (lo, hi) = ranges[i];
                            if b >= lo && b <= hi {
                                matched = true;
                                break;
                            }
                        }
                        if matched {
                            pos += 1;
                            pc += 1;
                            continue;
                        }
                    }
                    if !self.backtrack(
                        &mut pc,
                        &mut pos,
                        &mut stack,
                        slots,
                        &mut slot_stack,
                        &mut slot_stack_frames,
                    ) {
                        return false;
                    }
                }

                Op::NotByteRanges { count, ranges } => {
                    if pos < input.len() {
                        let b = input[pos];
                        let mut in_range = false;
                        for i in 0..count as usize {
                            let (lo, hi) = ranges[i];
                            if b >= lo && b <= hi {
                                in_range = true;
                                break;
                            }
                        }
                        if !in_range {
                            pos += 1;
                            pc += 1;
                            continue;
                        }
                    }
                    if !self.backtrack(
                        &mut pc,
                        &mut pos,
                        &mut stack,
                        slots,
                        &mut slot_stack,
                        &mut slot_stack_frames,
                    ) {
                        return false;
                    }
                }

                Op::CpRange(lo, hi) => {
                    if let Some((cp, len)) = decode_utf8(&input[pos..]) {
                        if cp >= lo && cp <= hi {
                            pos += len;
                            pc += 1;
                            continue;
                        }
                    }
                    if !self.backtrack(
                        &mut pc,
                        &mut pos,
                        &mut stack,
                        slots,
                        &mut slot_stack,
                        &mut slot_stack_frames,
                    ) {
                        return false;
                    }
                }

                Op::NotCpRange(lo, hi) => {
                    if let Some((cp, len)) = decode_utf8(&input[pos..]) {
                        if cp < lo || cp > hi {
                            pos += len;
                            pc += 1;
                            continue;
                        }
                    }
                    if !self.backtrack(
                        &mut pc,
                        &mut pos,
                        &mut stack,
                        slots,
                        &mut slot_stack,
                        &mut slot_stack_frames,
                    ) {
                        return false;
                    }
                }

                Op::ByteClassRef { index, negated } => {
                    if pos < input.len() {
                        let b = input[pos];
                        let byte_class = &self.byte_classes[index as usize];
                        // Use ByteClass::contains() for O(1) bitmap lookup
                        let in_class = byte_class.contains(b);
                        let matched = if negated { !in_class } else { in_class };
                        if matched {
                            pos += 1;
                            pc += 1;
                            continue;
                        }
                    }
                    if !self.backtrack(
                        &mut pc,
                        &mut pos,
                        &mut stack,
                        slots,
                        &mut slot_stack,
                        &mut slot_stack_frames,
                    ) {
                        return false;
                    }
                }

                Op::CpClassRef { index, negated } => {
                    if let Some((cp, len)) = decode_utf8(&input[pos..]) {
                        let cpclass = &self.cp_classes[index as usize];
                        // Use CodepointClass::contains() which has ASCII bitmap fast path
                        let in_class = cpclass.contains_raw(cp);
                        let matched = if negated { !in_class } else { in_class };
                        if matched {
                            pos += len;
                            pc += 1;
                            continue;
                        }
                    }
                    if !self.backtrack(
                        &mut pc,
                        &mut pos,
                        &mut stack,
                        slots,
                        &mut slot_stack,
                        &mut slot_stack_frames,
                    ) {
                        return false;
                    }
                }

                Op::Any => {
                    if pos < input.len() {
                        pos += 1;
                        pc += 1;
                    } else if !self.backtrack(
                        &mut pc,
                        &mut pos,
                        &mut stack,
                        slots,
                        &mut slot_stack,
                        &mut slot_stack_frames,
                    ) {
                        return false;
                    }
                }

                Op::Split(target) => {
                    // Save backtrack point for target
                    slot_stack_frames.push(slot_stack.len());
                    stack.push((target, pos));
                    pc += 1;
                }

                Op::Jump(target) => {
                    pc = target;
                }

                Op::Save(slot) => {
                    let slot = slot as usize;
                    if slot < slots.len() {
                        // Save old value for potential restore
                        if let Some(&frame_start) = slot_stack_frames.last() {
                            // Check if we already saved this slot in current frame
                            let mut already_saved = false;
                            for i in frame_start..slot_stack.len() {
                                if slot_stack[i].0 == slot as u16 {
                                    already_saved = true;
                                    break;
                                }
                            }
                            if !already_saved {
                                slot_stack.push((slot as u16, slots[slot]));
                            }
                        }
                        slots[slot] = pos as i32;
                    }
                    pc += 1;
                }

                Op::Match => {
                    slots[1] = pos as i32;
                    return true;
                }

                Op::StartAnchor => {
                    if pos == 0 {
                        pc += 1;
                    } else if !self.backtrack(
                        &mut pc,
                        &mut pos,
                        &mut stack,
                        slots,
                        &mut slot_stack,
                        &mut slot_stack_frames,
                    ) {
                        return false;
                    }
                }

                Op::EndAnchor => {
                    if pos == input.len() {
                        pc += 1;
                    } else if !self.backtrack(
                        &mut pc,
                        &mut pos,
                        &mut stack,
                        slots,
                        &mut slot_stack,
                        &mut slot_stack_frames,
                    ) {
                        return false;
                    }
                }

                Op::WordBoundary => {
                    let before = pos > 0 && is_word_byte(input[pos - 1]);
                    let after = pos < input.len() && is_word_byte(input[pos]);
                    if before != after {
                        pc += 1;
                    } else if !self.backtrack(
                        &mut pc,
                        &mut pos,
                        &mut stack,
                        slots,
                        &mut slot_stack,
                        &mut slot_stack_frames,
                    ) {
                        return false;
                    }
                }

                Op::NotWordBoundary => {
                    let before = pos > 0 && is_word_byte(input[pos - 1]);
                    let after = pos < input.len() && is_word_byte(input[pos]);
                    if before == after {
                        pc += 1;
                    } else if !self.backtrack(
                        &mut pc,
                        &mut pos,
                        &mut stack,
                        slots,
                        &mut slot_stack,
                        &mut slot_stack_frames,
                    ) {
                        return false;
                    }
                }

                Op::Backref(group) => {
                    let idx = group as usize;
                    let s = slots[idx * 2];
                    let e = slots[idx * 2 + 1];
                    if s >= 0 && e >= 0 {
                        let captured = &input[s as usize..e as usize];
                        let len = captured.len();
                        if pos + len <= input.len() && &input[pos..pos + len] == captured {
                            pos += len;
                            pc += 1;
                            continue;
                        }
                    }
                    if !self.backtrack(
                        &mut pc,
                        &mut pos,
                        &mut stack,
                        slots,
                        &mut slot_stack,
                        &mut slot_stack_frames,
                    ) {
                        return false;
                    }
                }
            }
        }
    }

    /// Backtrack to the previous choice point.
    #[inline]
    fn backtrack(
        &self,
        pc: &mut u32,
        pos: &mut usize,
        stack: &mut Vec<(u32, usize)>,
        slots: &mut [i32],
        slot_stack: &mut Vec<(u16, i32)>,
        slot_stack_frames: &mut Vec<usize>,
    ) -> bool {
        if let Some((saved_pc, saved_pos)) = stack.pop() {
            // Restore slots
            if let Some(frame_start) = slot_stack_frames.pop() {
                while slot_stack.len() > frame_start {
                    let (slot, val) = slot_stack.pop().unwrap();
                    slots[slot as usize] = val;
                }
            }
            *pc = saved_pc;
            *pos = saved_pos;
            true
        } else {
            false
        }
    }
}

/// Compiler from HIR to bytecode.
struct Compiler {
    code: Vec<Op>,
    byte_classes: Vec<ByteClass>,
    cp_classes: Vec<CodepointClass>,
}

impl Compiler {
    fn new() -> Self {
        Self {
            code: Vec::with_capacity(64),
            byte_classes: Vec::new(),
            cp_classes: Vec::new(),
        }
    }

    fn emit(&mut self, op: Op) {
        self.code.push(op);
    }

    fn pc(&self) -> u32 {
        self.code.len() as u32
    }

    fn compile(&mut self, expr: &HirExpr) {
        match expr {
            HirExpr::Empty => {}

            HirExpr::Literal(bytes) => {
                for &b in bytes {
                    self.emit(Op::Byte(b));
                }
            }

            HirExpr::Class(class) => {
                if class.ranges.len() <= 4 {
                    let mut ranges = [(0u8, 0u8); 4];
                    for (i, &(lo, hi)) in class.ranges.iter().enumerate() {
                        ranges[i] = (lo, hi);
                    }
                    if class.negated {
                        self.emit(Op::NotByteRanges {
                            count: class.ranges.len() as u8,
                            ranges,
                        });
                    } else {
                        self.emit(Op::ByteRanges {
                            count: class.ranges.len() as u8,
                            ranges,
                        });
                    }
                } else {
                    // Too many ranges - store in byte_classes table and use ByteClassRef
                    let index = self.byte_classes.len() as u16;
                    // Convert (u8, u8) tuples to ByteRange and create ByteClass with bitmap
                    let byte_ranges: Vec<ByteRange> = class
                        .ranges
                        .iter()
                        .map(|&(lo, hi)| ByteRange::new(lo, hi))
                        .collect();
                    self.byte_classes.push(ByteClass::new(byte_ranges));
                    self.emit(Op::ByteClassRef {
                        index,
                        negated: class.negated,
                    });
                }
            }

            HirExpr::UnicodeCpClass(class) => {
                if class.ranges.len() == 1 {
                    // Single range - use inline op
                    let (lo, hi) = class.ranges[0];
                    if class.negated {
                        self.emit(Op::NotCpRange(lo, hi));
                    } else {
                        self.emit(Op::CpRange(lo, hi));
                    }
                } else if !class.ranges.is_empty() {
                    // Multiple ranges - store full CodepointClass for ASCII bitmap optimization
                    let index = self.cp_classes.len() as u16;
                    self.cp_classes.push(class.clone());
                    self.emit(Op::CpClassRef {
                        index,
                        negated: class.negated,
                    });
                }
            }

            HirExpr::Concat(parts) => {
                for part in parts {
                    self.compile(part);
                }
            }

            HirExpr::Alt(alts) => {
                if alts.is_empty() {
                    return;
                }
                if alts.len() == 1 {
                    self.compile(&alts[0]);
                    return;
                }

                // For each alternative except the last, emit Split
                let mut jump_patches = Vec::new();

                for (i, alt) in alts.iter().enumerate() {
                    if i + 1 < alts.len() {
                        let split_pc = self.pc();
                        self.emit(Op::Split(0)); // Placeholder, will patch

                        self.compile(alt);

                        let jump_pc = self.pc();
                        self.emit(Op::Jump(0)); // Placeholder
                        jump_patches.push(jump_pc);

                        // Patch split target to here
                        let target = self.pc();
                        self.code[split_pc as usize] = Op::Split(target);
                    } else {
                        self.compile(alt);
                    }
                }

                // Patch all jumps to after the alternation
                let after = self.pc();
                for jp in jump_patches {
                    self.code[jp as usize] = Op::Jump(after);
                }
            }

            HirExpr::Repeat(rep) => {
                let min = rep.min;
                let max = rep.max;
                let greedy = rep.greedy;

                // Emit min copies
                for _ in 0..min {
                    self.compile(&rep.expr);
                }

                match max {
                    Some(max_val) if max_val == min => {
                        // Exact count, nothing more to do
                    }
                    Some(max_val) => {
                        // {min, max}: emit (max - min) optional copies
                        for _ in min..max_val {
                            if greedy {
                                let split_pc = self.pc();
                                self.emit(Op::Split(0)); // Try match, on fail skip
                                self.compile(&rep.expr);
                                let target = self.pc();
                                self.code[split_pc as usize] = Op::Split(target);
                            } else {
                                // Non-greedy: try skip first
                                let split_pc = self.pc();
                                self.emit(Op::Split(0));
                                let jump_pc = self.pc();
                                self.emit(Op::Jump(0));
                                let target = self.pc();
                                self.code[split_pc as usize] = Op::Split(target);
                                self.compile(&rep.expr);
                                let after = self.pc();
                                self.code[jump_pc as usize] = Op::Jump(after);
                            }
                        }
                    }
                    None => {
                        // Unbounded: *
                        let loop_start = self.pc();
                        if greedy {
                            let split_pc = self.pc();
                            self.emit(Op::Split(0)); // Try match, on fail exit
                            self.compile(&rep.expr);
                            self.emit(Op::Jump(loop_start));
                            let exit = self.pc();
                            self.code[split_pc as usize] = Op::Split(exit);
                        } else {
                            // Non-greedy: try exit first
                            let split_pc = self.pc();
                            self.emit(Op::Split(0));
                            let jump_pc = self.pc();
                            self.emit(Op::Jump(0));
                            let loop_body = self.pc();
                            self.code[split_pc as usize] = Op::Split(loop_body);
                            self.compile(&rep.expr);
                            self.emit(Op::Jump(loop_start));
                            let exit = self.pc();
                            self.code[jump_pc as usize] = Op::Jump(exit);
                        }
                    }
                }
            }

            HirExpr::Capture(cap) => {
                let start_slot = (cap.index as u16) * 2;
                let end_slot = start_slot + 1;

                self.emit(Op::Save(start_slot));
                self.compile(&cap.expr);
                self.emit(Op::Save(end_slot));
            }

            HirExpr::Backref(group) => {
                self.emit(Op::Backref(*group as u16));
            }

            HirExpr::Anchor(anchor) => match anchor {
                HirAnchor::Start | HirAnchor::StartLine => {
                    self.emit(Op::StartAnchor);
                }
                HirAnchor::End | HirAnchor::EndLine => {
                    self.emit(Op::EndAnchor);
                }
                HirAnchor::WordBoundary => {
                    self.emit(Op::WordBoundary);
                }
                HirAnchor::NotWordBoundary => {
                    self.emit(Op::NotWordBoundary);
                }
            },

            HirExpr::Lookaround(_) => {
                // Not supported in this VM
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn make_vm(pattern: &str) -> BacktrackingVm {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        BacktrackingVm::new(&hir)
    }

    #[test]
    fn test_simple_literal() {
        let vm = make_vm("hello");
        assert_eq!(vm.find(b"hello world"), Some((0, 5)));
        assert_eq!(vm.find(b"say hello"), Some((4, 9)));
        assert_eq!(vm.find(b"goodbye"), None);
    }

    #[test]
    fn test_alternation() {
        let vm = make_vm("a|b");
        assert_eq!(vm.find(b"a"), Some((0, 1)));
        assert_eq!(vm.find(b"b"), Some((0, 1)));
        assert_eq!(vm.find(b"c"), None);
    }

    #[test]
    fn test_star() {
        let vm = make_vm("a*");
        assert_eq!(vm.find(b"aaa"), Some((0, 3)));
        assert_eq!(vm.find(b"b"), Some((0, 0)));
    }

    #[test]
    fn test_capture_in_star() {
        let vm = make_vm("x(a|b)*y");
        assert_eq!(vm.find(b"xy"), Some((0, 2)));
        assert_eq!(vm.find(b"xay"), Some((0, 3)));
        assert_eq!(vm.find(b"xby"), Some((0, 3)));
        assert_eq!(vm.find(b"xaby"), Some((0, 4)));
        assert_eq!(vm.find(b"xaaby"), Some((0, 5)));
    }

    #[test]
    fn test_json_string() {
        let vm = make_vm(r#""([^"\\]|\\.)*""#);
        assert_eq!(vm.find(br#""""#), Some((0, 2)));
        assert_eq!(vm.find(br#""hello""#), Some((0, 7)));
        assert_eq!(vm.find(br#""hello\"world""#), Some((0, 14)));
        assert_eq!(vm.find(br#""\\""#), Some((0, 4)));
    }

    #[test]
    fn test_captures() {
        let vm = make_vm(r#"(a)(b)(c)"#);
        let caps = vm.captures(b"abc").unwrap();
        assert_eq!(caps[0], Some((0, 3)));
        assert_eq!(caps[1], Some((0, 1)));
        assert_eq!(caps[2], Some((1, 2)));
        assert_eq!(caps[3], Some((2, 3)));
    }
}
