//! Where a pattern's zero-width assertions sit relative to what it consumes.
//!
//! The DFA family and Shift-Or resolve an assertion from the start context of
//! an attempt or check it against the end of a match. An assertion between
//! two consuming parts, as in `a$a` or `a\ba?`, needs the position-by-position
//! evaluation only the PikeVM does. [`assertions_at_edges`] tells the two
//! shapes apart.

use super::{Nfa, NfaInstruction, StateId};

/// Assertion flavours, by which side of the match they constrain.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// `^` and multiline `^`: only meaningful before anything is consumed.
    Start,
    /// `$` and multiline `$`: only meaningful after everything is consumed.
    End,
    /// `\b` and `\B`: meaningful at either edge.
    Boundary,
}

fn kind(state: &super::NfaState) -> Option<Kind> {
    match state.instruction {
        Some(NfaInstruction::StartOfText | NfaInstruction::StartOfLine) => Some(Kind::Start),
        Some(NfaInstruction::EndOfText | NfaInstruction::EndOfLine) => Some(Kind::End),
        Some(NfaInstruction::WordBoundary | NfaInstruction::NotWordBoundary) => {
            Some(Kind::Boundary)
        }
        _ => None,
    }
}

/// Whether a state consumes input: a byte edge or a codepoint class.
fn consumes(state: &super::NfaState) -> bool {
    !state.transitions.is_empty()
        || matches!(
            state.instruction,
            Some(NfaInstruction::CodepointClass(_, _))
        )
}

/// Whether every assertion sits at an edge of the match, with no other
/// assertion beside it.
///
/// An assertion is *leading* when no path from the start consumes input
/// before reaching it, and *trailing* when no path from it consumes input
/// afterwards. `^` must be leading and `$` trailing, each on every path to a
/// match. `\b`/`\B` must be leading or the accepting state itself. A leading
/// assertion must not be reached from another assertion, and a trailing one
/// must not reach another: `a\b$` stacks two end checks, which these engines
/// do not combine.
pub fn assertions_at_edges(nfa: &Nfa) -> bool {
    let n = nfa.states.len();
    let assertions: Vec<StateId> = (0..n as StateId)
        .filter(|&id| kind(&nfa.states[id as usize]).is_some())
        .collect();
    if assertions.is_empty() {
        return true;
    }

    // States reachable from the start with nothing consumed, and states
    // reachable after at least one consuming step.
    let mut before_input = vec![false; n];
    let mut after_input = vec![false; n];
    let mut stack: Vec<(StateId, bool)> = vec![(nfa.start, false)];
    while let Some((id, consumed)) = stack.pop() {
        let Some(state) = nfa.get(id) else {
            continue;
        };
        let seen = if consumed {
            &mut after_input
        } else {
            &mut before_input
        };
        if std::mem::replace(&mut seen[id as usize], true) {
            continue;
        }
        for &e in &state.epsilon {
            stack.push((e, consumed));
        }
        for &(_, t) in &state.transitions {
            stack.push((t, true));
        }
        if let Some(NfaInstruction::CodepointClass(_, t)) = &state.instruction {
            stack.push((*t, true));
        }
    }

    // States from which a consuming step is still reachable.
    let mut preds: Vec<Vec<StateId>> = vec![Vec::new(); n];
    for (id, state) in nfa.states.iter().enumerate() {
        let targets = state
            .epsilon
            .iter()
            .copied()
            .chain(state.transitions.iter().map(|&(_, t)| t));
        for t in targets {
            if let Some(p) = preds.get_mut(t as usize) {
                p.push(id as StateId);
            }
        }
        if let Some(NfaInstruction::CodepointClass(_, t)) = &state.instruction {
            if let Some(p) = preds.get_mut(*t as usize) {
                p.push(id as StateId);
            }
        }
    }
    let mut input_follows = vec![false; n];
    let mut stack: Vec<StateId> = Vec::new();
    for (id, state) in nfa.states.iter().enumerate() {
        if consumes(state) {
            input_follows[id] = true;
            stack.push(id as StateId);
        }
    }
    while let Some(id) = stack.pop() {
        for &p in &preds[id as usize] {
            if !std::mem::replace(&mut input_follows[p as usize], true) {
                stack.push(p);
            }
        }
    }

    // An anchor makes these engines treat the whole pattern as anchored: a
    // start anchor limits the positions they try, and Shift-Or checks an end
    // anchor on every match. So every match has to pass one. `(?:^a)?` also
    // matches empty anywhere, and `a(?:a$)?` matches "a" mid-text.
    for anchor in [Kind::Start, Kind::End] {
        let present = assertions
            .iter()
            .any(|&a| nfa.get(a).and_then(kind) == Some(anchor));
        if present && match_bypasses(nfa, anchor) {
            return false;
        }
    }

    let is_assertion = |id: StateId| nfa.get(id).and_then(kind).is_some();
    for &a in &assertions {
        let Some(k) = nfa.get(a).and_then(kind) else {
            continue;
        };
        let leading = before_input[a as usize] && !after_input[a as usize];
        let trailing = !input_follows[a as usize];
        // The automata step through `$` and check it when the match ends, but
        // they stop at an unresolved `\b` and only check one that is the
        // accepting state itself: `c\b` works, `c\b()` does not.
        let placed = match k {
            Kind::Start => leading,
            Kind::End => trailing,
            Kind::Boundary => leading || (trailing && nfa.states[a as usize].is_match),
        };
        if !placed {
            return false;
        }
        if trailing
            && reaches_by_epsilon(nfa, a)
                .into_iter()
                .any(|s| s != a && is_assertion(s))
        {
            return false;
        }
        if leading && !trailing {
            let stacked = assertions
                .iter()
                .any(|&other| other != a && reaches_by_epsilon(nfa, other).contains(&a));
            if stacked {
                return false;
            }
        }
    }
    true
}

/// Whether a match state is reachable from the start without passing an
/// assertion of kind `anchor`.
fn match_bypasses(nfa: &Nfa, anchor: Kind) -> bool {
    let mut seen = vec![false; nfa.states.len()];
    let mut stack = vec![nfa.start];
    while let Some(id) = stack.pop() {
        let Some(state) = nfa.get(id) else {
            continue;
        };
        if std::mem::replace(&mut seen[id as usize], true) || kind(state) == Some(anchor) {
            continue;
        }
        if state.is_match {
            return true;
        }
        stack.extend(state.epsilon.iter().copied());
        stack.extend(state.transitions.iter().map(|&(_, t)| t));
        if let Some(NfaInstruction::CodepointClass(_, t)) = &state.instruction {
            stack.push(*t);
        }
    }
    false
}

/// The states reachable from `from` over epsilon edges, `from` included.
fn reaches_by_epsilon(nfa: &Nfa, from: StateId) -> Vec<StateId> {
    let mut seen = vec![false; nfa.states.len()];
    let mut out = Vec::new();
    let mut stack = vec![from];
    while let Some(id) = stack.pop() {
        let Some(state) = nfa.get(id) else {
            continue;
        };
        if std::mem::replace(&mut seen[id as usize], true) {
            continue;
        }
        out.push(id);
        stack.extend(state.epsilon.iter().copied());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn at_edges(pattern: &str) -> bool {
        let hir = translate(&parse(pattern).unwrap()).unwrap();
        assertions_at_edges(&super::super::compile(&hir).unwrap())
    }

    #[test]
    fn edge_assertions_are_accepted() {
        for pattern in [
            "abc",
            "^abc",
            "abc$",
            r"^\s*$",
            r"\bword\b",
            r"\b\w+\b",
            r"(?m)^\w+:",
            r"\d+\b",
        ] {
            assert!(at_edges(pattern), "{pattern}");
        }
    }

    #[test]
    fn interior_or_stacked_assertions_are_refused() {
        for pattern in [
            "a$a",
            "a^",
            r"a\ba?",
            r"[ab]?\bb",
            r"a\b$",
            r"$a\b",
            r"^\bfoo",
            r"c(\b)",
            "(?:^a)?",
            "a(?:a$)?",
        ] {
            assert!(!at_edges(pattern), "{pattern}");
        }
    }
}
