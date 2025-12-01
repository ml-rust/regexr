//! Thompson's construction for building NFAs.

use crate::error::Result;
use crate::hir::{CodepointClass, Hir, HirAnchor, HirCapture, HirClass, HirExpr, HirLookaround, HirLookaroundKind, HirRepeat};

use super::{ByteRange, Nfa, NfaInstruction, NfaState, StateId};

/// An NFA fragment with start and end states.
struct Fragment {
    start: StateId,
    end: StateId,
}

/// NFA builder using Thompson's construction.
pub struct NfaBuilder {
    nfa: Nfa,
}

impl NfaBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            nfa: Nfa::new(),
        }
    }

    /// Builds an NFA from HIR.
    pub fn build(&mut self, hir: &Hir) -> Result<Nfa> {
        self.nfa.capture_count = hir.props.capture_count;
        self.nfa.has_backrefs = hir.props.has_backrefs;
        self.nfa.has_lookaround = hir.props.has_lookaround;

        let fragment = self.build_expr(&hir.expr)?;

        // Set start state
        self.nfa.start = fragment.start;

        // Mark end state as match
        if let Some(end_state) = self.nfa.get_mut(fragment.end) {
            end_state.is_match = true;
        }
        self.nfa.matches.push(fragment.end);

        Ok(std::mem::take(&mut self.nfa))
    }

    /// Adds a new empty state.
    fn add_state(&mut self) -> StateId {
        self.nfa.add_state(NfaState::new())
    }

    /// Builds a fragment from an HIR expression.
    fn build_expr(&mut self, expr: &HirExpr) -> Result<Fragment> {
        match expr {
            HirExpr::Empty => self.build_empty(),
            HirExpr::Literal(bytes) => self.build_literal(bytes),
            HirExpr::Class(class) => self.build_class(class),
            HirExpr::UnicodeCpClass(cpclass) => self.build_codepoint_class(cpclass),
            HirExpr::Concat(exprs) => self.build_concat(exprs),
            HirExpr::Alt(exprs) => self.build_alt(exprs),
            HirExpr::Repeat(rep) => self.build_repeat(rep),
            HirExpr::Capture(cap) => self.build_capture(cap),
            HirExpr::Anchor(anchor) => self.build_anchor(*anchor),
            HirExpr::Lookaround(la) => self.build_lookaround(la),
            HirExpr::Backref(n) => self.build_backref(*n),
        }
    }

    /// Builds an empty fragment (epsilon).
    fn build_empty(&mut self) -> Result<Fragment> {
        let state = self.add_state();
        Ok(Fragment { start: state, end: state })
    }

    /// Builds a literal byte sequence.
    fn build_literal(&mut self, bytes: &[u8]) -> Result<Fragment> {
        if bytes.is_empty() {
            return self.build_empty();
        }

        let start = self.add_state();
        let mut current = start;

        for (i, &byte) in bytes.iter().enumerate() {
            let next = if i == bytes.len() - 1 {
                self.add_state()
            } else {
                self.add_state()
            };

            if let Some(state) = self.nfa.get_mut(current) {
                state.add_transition(ByteRange::single(byte), next);
            }

            current = next;
        }

        Ok(Fragment { start, end: current })
    }

    /// Builds a character class.
    fn build_class(&mut self, class: &HirClass) -> Result<Fragment> {
        let start = self.add_state();
        let end = self.add_state();

        if class.negated {
            // For negated class, add all bytes NOT in the ranges
            let mut covered = vec![false; 256];
            for &(lo, hi) in &class.ranges {
                for b in lo..=hi {
                    covered[b as usize] = true;
                }
            }

            // Build ranges from uncovered bytes
            let mut ranges = Vec::new();
            let mut i = 0;
            while i < 256 {
                if !covered[i] {
                    let range_start = i as u8;
                    while i < 256 && !covered[i] {
                        i += 1;
                    }
                    let range_end = (i - 1) as u8;
                    ranges.push(ByteRange::new(range_start, range_end));
                } else {
                    i += 1;
                }
            }

            if let Some(state) = self.nfa.get_mut(start) {
                for range in ranges {
                    state.add_transition(range, end);
                }
            }
        } else {
            if let Some(state) = self.nfa.get_mut(start) {
                for &(lo, hi) in &class.ranges {
                    state.add_transition(ByteRange::new(lo, hi), end);
                }
            }
        }

        Ok(Fragment { start, end })
    }

    /// Builds a Unicode codepoint class.
    /// Uses a special instruction that consumes a full UTF-8 codepoint and checks membership.
    fn build_codepoint_class(&mut self, cpclass: &CodepointClass) -> Result<Fragment> {
        let start = self.add_state();
        let end = self.add_state();

        // Add the CodepointClass instruction to the start state
        if let Some(state) = self.nfa.get_mut(start) {
            state.instruction = Some(NfaInstruction::CodepointClass(cpclass.clone(), end));
        }

        Ok(Fragment { start, end })
    }

    /// Builds concatenation.
    fn build_concat(&mut self, exprs: &[HirExpr]) -> Result<Fragment> {
        if exprs.is_empty() {
            return self.build_empty();
        }

        let mut fragments = Vec::with_capacity(exprs.len());
        for expr in exprs {
            fragments.push(self.build_expr(expr)?);
        }

        // Chain fragments together with epsilon transitions
        for i in 0..fragments.len() - 1 {
            let from_end = fragments[i].end;
            let to_start = fragments[i + 1].start;
            if let Some(state) = self.nfa.get_mut(from_end) {
                state.add_epsilon(to_start);
            }
        }

        Ok(Fragment {
            start: fragments[0].start,
            end: fragments.last().unwrap().end,
        })
    }

    /// Builds alternation.
    fn build_alt(&mut self, exprs: &[HirExpr]) -> Result<Fragment> {
        if exprs.is_empty() {
            return self.build_empty();
        }

        if exprs.len() == 1 {
            return self.build_expr(&exprs[0]);
        }

        let start = self.add_state();
        let end = self.add_state();

        for expr in exprs {
            let fragment = self.build_expr(expr)?;

            // Connect start to fragment start
            if let Some(state) = self.nfa.get_mut(start) {
                state.add_epsilon(fragment.start);
            }

            // Connect fragment end to end
            if let Some(state) = self.nfa.get_mut(fragment.end) {
                state.add_epsilon(end);
            }
        }

        Ok(Fragment { start, end })
    }

    /// Builds repetition.
    fn build_repeat(&mut self, rep: &HirRepeat) -> Result<Fragment> {
        match (rep.min, rep.max) {
            // a? -> optional
            (0, Some(1)) => self.build_optional(&rep.expr, rep.greedy),
            // a* -> zero or more
            (0, None) => self.build_star(&rep.expr, rep.greedy),
            // a+ -> one or more
            (1, None) => self.build_plus(&rep.expr, rep.greedy),
            // a{n} -> exactly n
            (n, Some(m)) if n == m => self.build_exactly(&rep.expr, n),
            // a{n,} -> at least n
            (n, None) => self.build_at_least(&rep.expr, n, rep.greedy),
            // a{n,m} -> between n and m
            (n, Some(m)) => self.build_bounded(&rep.expr, n, m, rep.greedy),
        }
    }

    /// Builds a? (optional).
    fn build_optional(&mut self, expr: &HirExpr, greedy: bool) -> Result<Fragment> {
        let fragment = self.build_expr(expr)?;
        let start = self.add_state();
        let end = self.add_state();

        if greedy {
            // Greedy: try match first
            if let Some(state) = self.nfa.get_mut(start) {
                state.add_epsilon(fragment.start);
                state.add_epsilon(end);
            }
        } else {
            // Non-greedy: try skip first, mark exit with NonGreedyExit
            // Insert a marker state before the exit
            let marker = self.add_state();
            if let Some(state) = self.nfa.get_mut(marker) {
                state.instruction = Some(NfaInstruction::NonGreedyExit);
                state.add_epsilon(end);
            }
            if let Some(state) = self.nfa.get_mut(start) {
                state.add_epsilon(marker);
                state.add_epsilon(fragment.start);
            }
        }

        if let Some(state) = self.nfa.get_mut(fragment.end) {
            state.add_epsilon(end);
        }

        Ok(Fragment { start, end })
    }

    /// Builds a* (zero or more).
    fn build_star(&mut self, expr: &HirExpr, greedy: bool) -> Result<Fragment> {
        let fragment = self.build_expr(expr)?;
        let start = self.add_state();
        let end = self.add_state();

        if greedy {
            if let Some(state) = self.nfa.get_mut(start) {
                state.add_epsilon(fragment.start);
                state.add_epsilon(end);
            }
        } else {
            // Non-greedy: add marker for exit preference
            let marker = self.add_state();
            if let Some(state) = self.nfa.get_mut(marker) {
                state.instruction = Some(NfaInstruction::NonGreedyExit);
                state.add_epsilon(end);
            }
            if let Some(state) = self.nfa.get_mut(start) {
                state.add_epsilon(marker);
                state.add_epsilon(fragment.start);
            }
        }

        // Loop back
        if greedy {
            if let Some(state) = self.nfa.get_mut(fragment.end) {
                state.add_epsilon(fragment.start);
                state.add_epsilon(end);
            }
        } else {
            // Non-greedy loop: prefer exit
            let loop_marker = self.add_state();
            if let Some(state) = self.nfa.get_mut(loop_marker) {
                state.instruction = Some(NfaInstruction::NonGreedyExit);
                state.add_epsilon(end);
            }
            if let Some(state) = self.nfa.get_mut(fragment.end) {
                state.add_epsilon(loop_marker);
                state.add_epsilon(fragment.start);
            }
        }

        Ok(Fragment { start, end })
    }

    /// Builds a+ (one or more).
    fn build_plus(&mut self, expr: &HirExpr, greedy: bool) -> Result<Fragment> {
        let fragment = self.build_expr(expr)?;
        let end = self.add_state();

        // Loop back
        if greedy {
            if let Some(state) = self.nfa.get_mut(fragment.end) {
                state.add_epsilon(fragment.start);
                state.add_epsilon(end);
            }
        } else {
            // Non-greedy: prefer exit
            let marker = self.add_state();
            if let Some(state) = self.nfa.get_mut(marker) {
                state.instruction = Some(NfaInstruction::NonGreedyExit);
                state.add_epsilon(end);
            }
            if let Some(state) = self.nfa.get_mut(fragment.end) {
                state.add_epsilon(marker);
                state.add_epsilon(fragment.start);
            }
        }

        Ok(Fragment { start: fragment.start, end })
    }

    /// Builds a{n} (exactly n).
    fn build_exactly(&mut self, expr: &HirExpr, n: u32) -> Result<Fragment> {
        if n == 0 {
            return self.build_empty();
        }

        let mut fragments = Vec::with_capacity(n as usize);
        for _ in 0..n {
            fragments.push(self.build_expr(expr)?);
        }

        // Chain them together
        for i in 0..fragments.len() - 1 {
            let from_end = fragments[i].end;
            let to_start = fragments[i + 1].start;
            if let Some(state) = self.nfa.get_mut(from_end) {
                state.add_epsilon(to_start);
            }
        }

        Ok(Fragment {
            start: fragments[0].start,
            end: fragments.last().unwrap().end,
        })
    }

    /// Builds a{n,} (at least n).
    fn build_at_least(&mut self, expr: &HirExpr, n: u32, greedy: bool) -> Result<Fragment> {
        // Build n required copies, then a*
        let required = self.build_exactly(expr, n)?;
        let star = self.build_star(expr, greedy)?;

        if let Some(state) = self.nfa.get_mut(required.end) {
            state.add_epsilon(star.start);
        }

        Ok(Fragment {
            start: required.start,
            end: star.end,
        })
    }

    /// Builds a{n,m} (between n and m).
    fn build_bounded(&mut self, expr: &HirExpr, n: u32, m: u32, greedy: bool) -> Result<Fragment> {
        if n > m {
            return self.build_empty();
        }

        // Build n required copies
        let required = if n > 0 {
            Some(self.build_exactly(expr, n)?)
        } else {
            None
        };

        // Build m-n optional copies
        let optional_count = m - n;
        let mut optional_fragments = Vec::with_capacity(optional_count as usize);
        for _ in 0..optional_count {
            optional_fragments.push(self.build_optional(expr, greedy)?);
        }

        // Chain optional fragments
        for i in 0..optional_fragments.len().saturating_sub(1) {
            let from_end = optional_fragments[i].end;
            let to_start = optional_fragments[i + 1].start;
            if let Some(state) = self.nfa.get_mut(from_end) {
                state.add_epsilon(to_start);
            }
        }

        let (start, end) = match (required, optional_fragments.first(), optional_fragments.last()) {
            (Some(req), Some(opt_first), Some(opt_last)) => {
                if let Some(state) = self.nfa.get_mut(req.end) {
                    state.add_epsilon(opt_first.start);
                }
                (req.start, opt_last.end)
            }
            (Some(req), None, None) => (req.start, req.end),
            (None, Some(opt_first), Some(opt_last)) => (opt_first.start, opt_last.end),
            (None, None, None) => {
                let state = self.add_state();
                (state, state)
            }
            _ => unreachable!(),
        };

        Ok(Fragment { start, end })
    }

    /// Builds a capture group.
    fn build_capture(&mut self, cap: &HirCapture) -> Result<Fragment> {
        let start = self.add_state();
        let end = self.add_state();

        // Set capture start instruction
        if let Some(state) = self.nfa.get_mut(start) {
            state.instruction = Some(NfaInstruction::CaptureStart(cap.index));
        }

        let inner = self.build_expr(&cap.expr)?;

        // Connect start to inner
        if let Some(state) = self.nfa.get_mut(start) {
            state.add_epsilon(inner.start);
        }

        // Create capture end state
        let cap_end = self.add_state();
        if let Some(state) = self.nfa.get_mut(cap_end) {
            state.instruction = Some(NfaInstruction::CaptureEnd(cap.index));
            state.add_epsilon(end);
        }

        // Connect inner end to capture end
        if let Some(state) = self.nfa.get_mut(inner.end) {
            state.add_epsilon(cap_end);
        }

        Ok(Fragment { start, end })
    }

    /// Builds an anchor.
    fn build_anchor(&mut self, anchor: HirAnchor) -> Result<Fragment> {
        let state = self.add_state();

        let instruction = match anchor {
            HirAnchor::Start => NfaInstruction::StartOfText,
            HirAnchor::End => NfaInstruction::EndOfText,
            HirAnchor::StartLine => NfaInstruction::StartOfLine,
            HirAnchor::EndLine => NfaInstruction::EndOfLine,
            HirAnchor::WordBoundary => NfaInstruction::WordBoundary,
            HirAnchor::NotWordBoundary => NfaInstruction::NotWordBoundary,
        };

        if let Some(s) = self.nfa.get_mut(state) {
            s.instruction = Some(instruction);
        }

        Ok(Fragment { start: state, end: state })
    }

    /// Builds a lookaround.
    fn build_lookaround(&mut self, la: &HirLookaround) -> Result<Fragment> {
        // Build the inner NFA
        let mut inner_builder = NfaBuilder::new();
        let inner_nfa = inner_builder.build(&crate::hir::Hir {
            expr: la.expr.clone(),
            props: Default::default(),
        })?;

        let state = self.add_state();

        let instruction = match la.kind {
            HirLookaroundKind::PositiveLookahead => {
                NfaInstruction::PositiveLookahead(Box::new(inner_nfa))
            }
            HirLookaroundKind::NegativeLookahead => {
                NfaInstruction::NegativeLookahead(Box::new(inner_nfa))
            }
            HirLookaroundKind::PositiveLookbehind => {
                NfaInstruction::PositiveLookbehind(Box::new(inner_nfa))
            }
            HirLookaroundKind::NegativeLookbehind => {
                NfaInstruction::NegativeLookbehind(Box::new(inner_nfa))
            }
        };

        if let Some(s) = self.nfa.get_mut(state) {
            s.instruction = Some(instruction);
        }

        Ok(Fragment { start: state, end: state })
    }

    /// Builds a backreference.
    fn build_backref(&mut self, n: u32) -> Result<Fragment> {
        let state = self.add_state();

        if let Some(s) = self.nfa.get_mut(state) {
            s.instruction = Some(NfaInstruction::Backref(n));
        }

        Ok(Fragment { start: state, end: state })
    }
}

impl Default for NfaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn build_nfa(pattern: &str) -> Nfa {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        compile(&hir).unwrap()
    }

    #[test]
    fn test_literal() {
        let nfa = build_nfa("abc");
        // Should have states for: start, a, b, c, end
        assert!(nfa.state_count() >= 4);
    }

    #[test]
    fn test_alternation() {
        let nfa = build_nfa("a|b");
        // Should have start, end, and states for each alternative
        assert!(nfa.state_count() >= 4);
    }

    #[test]
    fn test_repetition() {
        let nfa = build_nfa("a*");
        assert!(nfa.state_count() >= 2);
    }

    #[test]
    fn test_class() {
        let nfa = build_nfa("[a-z]");
        assert!(nfa.state_count() >= 2);
    }

    use super::super::compile;
}
