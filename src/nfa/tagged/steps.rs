//! Pattern step extraction from NFA.
//!
//! Extracts pattern steps from NFA for fast step-based matching.

use super::shared::PatternStep;
use crate::nfa::{ByteClass, ByteRange, Nfa, NfaInstruction, NfaState, StateId};

/// Combines greedy quantifiers followed by lookahead into combined variants.
/// This is needed for both JIT and interpreter to handle backtracking correctly.
///
/// Only a lookahead that ENDS the program is folded in. The combined step
/// resolves the run and the assertion together and yields one position — the
/// largest the assertion accepts — with no way to reconsider it. That is
/// complete when nothing follows, and wrong when something does: `[ab]+(?=b)bb`
/// on `"abbabb"` settles the run at 5, fails `bb` off the end of the input, and
/// reports no match where `(0, 6)` is correct. Left uncombined, the plain
/// `GreedyPlus` step retries the whole remainder at each backtrack position and
/// finds it. The lookahead-terminated shape this exists to accelerate —
/// `\w+(?=ing\b)` — is unaffected, since there the assertion is the last step.
pub fn combine_greedy_with_lookahead(steps: Vec<PatternStep>) -> Vec<PatternStep> {
    let mut result = Vec::with_capacity(steps.len());
    let mut i = 0;

    while i < steps.len() {
        match &steps[i] {
            PatternStep::GreedyPlus(ranges) if i + 2 == steps.len() => match &steps[i + 1] {
                PatternStep::PositiveLookahead(inner) => {
                    result.push(PatternStep::GreedyPlusLookahead(
                        ranges.clone(),
                        inner.clone(),
                        true,
                    ));
                    i += 2;
                    continue;
                }
                PatternStep::NegativeLookahead(inner) => {
                    result.push(PatternStep::GreedyPlusLookahead(
                        ranges.clone(),
                        inner.clone(),
                        false,
                    ));
                    i += 2;
                    continue;
                }
                _ => {}
            },
            PatternStep::GreedyStar(ranges) if i + 2 == steps.len() => match &steps[i + 1] {
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
            },
            _ => {}
        }
        result.push(steps[i].clone());
        i += 1;
    }

    result
}

/// Per-kind tally of zero-width assertions. Comparing tallies kind-by-kind (not
/// just a single total) is essential: a quantifier `Alt` can simultaneously
/// DUPLICATE one assertion and DROP another (`a*(?=\S)$` keeps the lookahead in
/// both branches but loses the `$`), and the two errors would cancel in a single
/// total (2 == 1+1) yet differ per kind (lookahead 2≠1, anchor 0≠1).
///
/// The tally covers the *whole* assertion tree, descending into the interior of
/// a lookaround as well as into `Alt` branches. Counting a lookaround as one
/// opaque unit would make the comparison blind exactly where it is needed: an
/// assertion dropped inside a lookaround's inner pattern (`(?<=a\b)` extracting
/// as `(?<=a)`) cancels out on both sides and the guard passes a step program
/// that no longer means what the pattern means.
#[derive(Default, PartialEq, Eq)]
pub(crate) struct AssertionTally {
    lookahead: usize,
    lookbehind: usize,
    anchor: usize,
    word_boundary: usize,
    backref: usize,
}

impl AssertionTally {
    fn add(&mut self, other: Self) {
        self.lookahead += other.lookahead;
        self.lookbehind += other.lookbehind;
        self.anchor += other.anchor;
        self.word_boundary += other.word_boundary;
        self.backref += other.backref;
    }
}

/// How many distinct total widths a lookbehind may carry before extraction
/// gives up.
///
/// Every candidate costs one extra backwards walk per position tried, and a
/// program whose totals fan out this far is no longer the fixed-width shape the
/// lookbehind checker is built for. Sixteen covers the real cases by a wide
/// margin — the widest common one, a `\s`-style class spanning UTF-8 widths 1, 2
/// and 3, contributes three candidates, and even three such classes in sequence
/// stay under the cap. Above it [`byte_len_set`] answers `None`, which routes
/// the pattern to the PikeVm, exactly as any other shape it declines; it never
/// truncates the set, because a truncated set would silently stop matching
/// inputs whose width it dropped.
pub(crate) const MAX_LOOKBEHIND_WIDTHS: usize = 16;

/// Every total byte count a step program can consume, ascending and without
/// duplicates, or `None` when the program admits a width this model cannot
/// enumerate (see the refused arms below) or fans out past
/// [`MAX_LOOKBEHIND_WIDTHS`].
///
/// A lookbehind is checked by walking forwards from `pos - total`, so a *total*
/// has to be exact, not a lower bound: start too early and the walk ends past
/// `pos`, start too late and it ends short. What does not have to be unique is
/// which total — the checker tries each candidate and the walk itself rejects
/// the wrong ones, since it derives every codepoint's real length from the
/// bytes and demands it land exactly on `pos`. So a class like `\s`, whose
/// members encode to 1, 2 or 3 bytes, is extractable as the candidate set
/// `[1, 2, 3]` rather than being refused for having no single width.
///
/// The set of totals for a sequence of steps is the sumset of the per-step
/// width sets. Since UTF-8 widths come from `{1, 2, 3, 4}` the set stays small,
/// and the cap keeps it that way for pathological input.
///
/// The match is deliberately exhaustive. A new [`PatternStep`] variant then
/// fails to compile here rather than falling into a default arm that guesses a
/// width, which is the only way this stays honest as the step model grows.
pub(crate) fn byte_len_set(steps: &[PatternStep]) -> Option<Vec<usize>> {
    let mut totals = vec![0usize];
    for step in steps {
        let step_widths: Vec<usize> = match step {
            PatternStep::Byte(_) | PatternStep::ByteClass(_) => vec![1],
            // A codepoint class contributes every encoded width its members span.
            PatternStep::CodepointClass(cpclass, _) => utf8_width_set(cpclass)?,
            // Zero-width: assertions and capture markers.
            PatternStep::WordBoundary
            | PatternStep::NotWordBoundary
            | PatternStep::StartOfText
            | PatternStep::EndOfText
            | PatternStep::StartOfLine
            | PatternStep::EndOfLine
            | PatternStep::CaptureStart(_)
            | PatternStep::CaptureEnd(_)
            | PatternStep::PositiveLookahead(_)
            | PatternStep::NegativeLookahead(_)
            | PatternStep::PositiveLookbehind(_, _)
            | PatternStep::NegativeLookbehind(_, _) => vec![0],
            // Repetition, alternation and backreferences admit unboundedly many
            // widths (or, for a backreference, a width not known until match
            // time), so no finite candidate set describes them.
            PatternStep::GreedyPlus(_)
            | PatternStep::GreedyStar(_)
            | PatternStep::GreedyPlusLookahead(_, _, _)
            | PatternStep::GreedyStarLookahead(_, _, _)
            | PatternStep::NonGreedyPlus(_, _)
            | PatternStep::NonGreedyStar(_, _)
            | PatternStep::GreedyCodepointPlus(_)
            | PatternStep::Alt(_)
            | PatternStep::Backref(_) => return None,
        };
        // Sumset: every running total extended by every width this step admits.
        let mut next = Vec::with_capacity(totals.len() * step_widths.len());
        for &total in &totals {
            for &width in &step_widths {
                next.push(total + width);
            }
        }
        next.sort_unstable();
        next.dedup();
        if next.len() > MAX_LOOKBEHIND_WIDTHS {
            return None;
        }
        totals = next;
    }
    Some(totals)
}

/// The exact number of bytes a step program consumes, or `None` when that is
/// not the same for every path through it.
///
/// The single-width answer [`byte_len_set`] gives when it finds exactly one
/// candidate. Kept as its own entry point because the JIT needs precisely this
/// question: its lookbehind codegen bakes one offset into the emitted
/// instructions, so it can only compile a program with one possible total.
// The JIT is its only caller, so it is genuinely unused in a build without one —
// scoped rather than a blanket `allow`, so it still reports as dead if the JIT
// stops calling it too.
#[cfg_attr(
    not(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64"))),
    allow(dead_code)
)]
pub(crate) fn fixed_byte_len(steps: &[PatternStep]) -> Option<usize> {
    match byte_len_set(steps)?.as_slice() {
        [width] => Some(*width),
        _ => None,
    }
}

/// The fewest bytes a step program can consume.
///
/// Unlike [`fixed_byte_len`] this always answers, because a lower bound exists
/// even for variable-width steps. It is used as a "can there be enough input
/// left?" precheck, where under-counting only costs a wasted attempt — but
/// over-counting would skip real matches, so every arm errs downwards.
///
/// Exhaustive for the same reason [`fixed_byte_len`] is: a new [`PatternStep`]
/// must be classified here deliberately rather than defaulting to zero.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) fn min_byte_len(steps: &[PatternStep]) -> usize {
    steps
        .iter()
        .map(|s| match s {
            PatternStep::Byte(_) | PatternStep::ByteClass(_) => 1,
            // A codepoint is 1..=4 bytes, so at least one.
            PatternStep::CodepointClass(_, _) | PatternStep::GreedyCodepointPlus(_) => 1,
            // A `+` runs at least once, a `*` may run not at all.
            PatternStep::GreedyPlus(_) | PatternStep::GreedyPlusLookahead(_, _, _) => 1,
            PatternStep::GreedyStar(_) | PatternStep::GreedyStarLookahead(_, _, _) => 0,
            // Non-greedy quantifiers carry their continuation inside them.
            PatternStep::NonGreedyPlus(_, suffix) => 1 + min_byte_len(std::slice::from_ref(suffix)),
            PatternStep::NonGreedyStar(_, suffix) => min_byte_len(std::slice::from_ref(suffix)),
            // The shortest branch bounds the alternation.
            PatternStep::Alt(branches) => {
                branches.iter().map(|b| min_byte_len(b)).min().unwrap_or(0)
            }
            // Zero-width: assertions, capture markers, and a backreference,
            // whose length is unknown here and may be empty.
            PatternStep::CaptureStart(_)
            | PatternStep::CaptureEnd(_)
            | PatternStep::WordBoundary
            | PatternStep::NotWordBoundary
            | PatternStep::StartOfText
            | PatternStep::EndOfText
            | PatternStep::StartOfLine
            | PatternStep::EndOfLine
            | PatternStep::PositiveLookahead(_)
            | PatternStep::NegativeLookahead(_)
            | PatternStep::PositiveLookbehind(_, _)
            | PatternStep::NegativeLookbehind(_, _)
            | PatternStep::Backref(_) => 0,
        })
        .sum()
}

/// The set of UTF-8 encoded widths the codepoints in `cpclass` span, ascending,
/// or `None` when the class has no enumerable width set.
///
/// A range that straddles an encoding boundary contributes every width it
/// touches rather than being refused: the lookbehind checker tries each
/// candidate total and rejects the ones that do not land on a codepoint
/// boundary, so several widths are as usable as one.
fn utf8_width_set(cpclass: &crate::hir::CodepointClass) -> Option<Vec<usize>> {
    // A negated class admits codepoints from across the whole range, so it spans
    // every encoded width. Left refused deliberately: the complement of a class
    // is not modelled here, and answering `[1, 2, 3, 4]` would claim more than
    // this function knows.
    if cpclass.negated || cpclass.ranges.is_empty() {
        return None;
    }
    // `UTF8_WIDTH_BOUNDARIES[n]` is the lowest codepoint needing `n + 1` bytes,
    // so band `n` is `[BOUNDARIES[n], BOUNDARIES[n + 1])` and a class range
    // touches it when the two overlap.
    let bounds = crate::nfa::utf8_automata::UTF8_WIDTH_BOUNDARIES;
    let mut widths = Vec::new();
    for &(start, end) in &cpclass.ranges {
        for band in 0..4 {
            if start < bounds[band + 1] && end >= bounds[band] && !widths.contains(&(band + 1)) {
                widths.push(band + 1);
            }
        }
    }
    // A class whose ranges all fall outside the encodable code space admits no
    // width at all; refuse rather than report an empty candidate set, which
    // would read as "matches at width nothing".
    if widths.is_empty() {
        return None;
    }
    widths.sort_unstable();
    Some(widths)
}

/// What an inner lookaround NFA's match state carries, for the linear walks
/// that stop there.
pub(crate) enum TerminalAssertion {
    /// Nothing that affects whether the inner pattern matches.
    Nothing,
    /// A zero-width assertion that is still part of the inner pattern and must
    /// be appended to its step list.
    Step(PatternStep),
    /// Something the linear step model cannot represent; extraction must refuse
    /// so the caller falls back to the PikeVm.
    Unsupported,
}

/// Reads the assertion an inner NFA's match state carries.
///
/// The inner extractors walk to the match state and stop, so without this the
/// state's instruction is never read and a trailing assertion silently
/// disappears — `(?<=a\b)` extracting as `(?<=a)`, `(?=a$)` as `(?=a)`. The two
/// are different assertions, and dropping one turns an unsupported pattern into
/// a wrong match rather than a fallback.
pub(crate) fn terminal_assertion(state: &NfaState) -> TerminalAssertion {
    match state.instruction {
        None => TerminalAssertion::Nothing,
        // Captures are not tracked while evaluating a lookaround.
        Some(NfaInstruction::CaptureStart(_) | NfaInstruction::CaptureEnd(_)) => {
            TerminalAssertion::Nothing
        }
        Some(NfaInstruction::WordBoundary) => TerminalAssertion::Step(PatternStep::WordBoundary),
        Some(NfaInstruction::NotWordBoundary) => {
            TerminalAssertion::Step(PatternStep::NotWordBoundary)
        }
        Some(NfaInstruction::StartOfText) => TerminalAssertion::Step(PatternStep::StartOfText),
        Some(NfaInstruction::EndOfText) => TerminalAssertion::Step(PatternStep::EndOfText),
        Some(NfaInstruction::StartOfLine) => TerminalAssertion::Step(PatternStep::StartOfLine),
        Some(NfaInstruction::EndOfLine) => TerminalAssertion::Step(PatternStep::EndOfLine),
        Some(_) => TerminalAssertion::Unsupported,
    }
}

/// Whether an NFA contains any lookaround assertion.
///
/// Used to keep the linear step extractor away from nested assertions, which it
/// cannot model.
fn nfa_has_lookaround(nfa: &Nfa) -> bool {
    nfa.states.iter().any(|s| {
        matches!(
            s.instruction,
            Some(
                NfaInstruction::PositiveLookahead(_)
                    | NfaInstruction::NegativeLookahead(_)
                    | NfaInstruction::PositiveLookbehind(_)
                    | NfaInstruction::NegativeLookbehind(_)
            )
        )
    })
}

/// Tallies zero-width assertions reachable from an NFA, including those inside
/// lookaround inner NFAs (held behind `Arc`, so not part of `nfa.states`).
pub(crate) fn count_assertions_in_nfa(nfa: &Nfa) -> AssertionTally {
    let mut t = AssertionTally::default();
    for s in &nfa.states {
        match s.instruction {
            Some(
                NfaInstruction::PositiveLookahead(ref inner)
                | NfaInstruction::NegativeLookahead(ref inner),
            ) => {
                t.lookahead += 1;
                t.add(count_assertions_in_nfa(inner));
            }
            Some(
                NfaInstruction::PositiveLookbehind(ref inner)
                | NfaInstruction::NegativeLookbehind(ref inner),
            ) => {
                t.lookbehind += 1;
                t.add(count_assertions_in_nfa(inner));
            }
            Some(
                NfaInstruction::StartOfText
                | NfaInstruction::EndOfText
                | NfaInstruction::StartOfLine
                | NfaInstruction::EndOfLine,
            ) => t.anchor += 1,
            Some(NfaInstruction::WordBoundary | NfaInstruction::NotWordBoundary) => {
                t.word_boundary += 1
            }
            Some(NfaInstruction::Backref(_)) => t.backref += 1,
            _ => {}
        }
    }
    t
}

/// Tallies zero-width assertions in a step program, recursing into `Alt`
/// branches and into lookaround interiors.
///
/// A genuine alternation carries each assertion in exactly one branch, so
/// summing branches reproduces the NFA tally; a quantifier `Alt` duplicates
/// (and/or drops) assertions, producing a tally that differs from the NFA's.
///
/// Lookaround interiors are summed for the same reason one level down. The
/// inner extractors walk to the inner NFA's match state and stop there, so an
/// assertion compiled onto that state — the `\b` in `(?<=a\b)`, the `$` in
/// `(?=a$)` — never reaches the step list. Descending here is what turns that
/// loss into a tally mismatch, and so into a fallback to the PikeVm.
pub(crate) fn count_assertions_in_steps(steps: &[PatternStep]) -> AssertionTally {
    let mut t = AssertionTally::default();
    for step in steps {
        match step {
            PatternStep::PositiveLookahead(inner) | PatternStep::NegativeLookahead(inner) => {
                t.lookahead += 1;
                t.add(count_assertions_in_steps(inner));
            }
            PatternStep::GreedyPlusLookahead(_, inner, _)
            | PatternStep::GreedyStarLookahead(_, inner, _) => {
                t.lookahead += 1;
                t.add(count_assertions_in_steps(inner));
            }
            PatternStep::PositiveLookbehind(inner, _)
            | PatternStep::NegativeLookbehind(inner, _) => {
                t.lookbehind += 1;
                t.add(count_assertions_in_steps(inner));
            }
            PatternStep::StartOfText
            | PatternStep::EndOfText
            | PatternStep::StartOfLine
            | PatternStep::EndOfLine => t.anchor += 1,
            PatternStep::WordBoundary | PatternStep::NotWordBoundary => t.word_boundary += 1,
            PatternStep::Backref(_) => t.backref += 1,
            PatternStep::Alt(branches) => {
                for b in branches {
                    t.add(count_assertions_in_steps(b));
                }
            }
            _ => {}
        }
    }
    t
}

/// Whether a step program must be executed by the TaggedNfa **interpreter**
/// rather than the JIT, because the JIT's emitted backtracking is unreliable for
/// the shape. Two shapes are deferred:
///
/// - a greedy quantifier together with a *separate* lookaround step (the
///   quantifier must give characters back until the assertion holds), and
/// - two or more quantifiers (adjacent greedy needs nested backtracking, e.g.
///   `\S+\S+\S`).
///
/// A `Greedy*Lookahead` step is neither: the quantifier and its assertion are
/// emitted as one unit that backtracks the one against the other, so `\w+(?=x)`
/// is JIT-compiled rather than deferred. `tests/tagged_jit_agreement.rs` holds
/// the differential coverage for that.
///
/// The interpreter (`TaggedNfa::find`) handles every shape correctly via its
/// boundary-backtracking + recursion, so deferring costs correctness nothing.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) fn jit_must_defer(steps: &[PatternStep]) -> bool {
    let counts = count_step_kinds(steps);
    let quantifiers = counts.greedy + counts.combined;
    // Nested backtracking across two quantifiers, of any kind.
    if quantifiers >= 2 {
        return true;
    }
    // A quantifier and a lookaround the generic step sequencing has to satisfy
    // by giving characters back — the shape the emitted backtracking gets wrong.
    // A `Greedy*Lookahead` step on its own is not this case: its emitter owns
    // both halves and backtracks the quantifier against its own assertion, so
    // `\w+(?=x)` is compiled rather than deferred.
    quantifiers >= 1 && counts.lookaround >= 1
}

/// Whether a step program repeats something without bound, anywhere —
/// including inside an alternation branch or a lookaround body.
///
/// The generated code retries every start and backtracks its runs, with no
/// budget on either: one failed attempt at each start of a run is one scan of
/// the run each, so a program with an unbounded repetition can cost time
/// quadratic in the input. Without one, every attempt reads a bounded number
/// of bytes, and the code is linear. Engine selection keeps programs that
/// repeat without bound on the metered interpreter instead.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) fn has_unbounded_repetition(steps: &[PatternStep]) -> bool {
    steps.iter().any(|step| match step {
        PatternStep::GreedyPlus(_)
        | PatternStep::GreedyStar(_)
        | PatternStep::GreedyCodepointPlus(_)
        | PatternStep::NonGreedyPlus(_, _)
        | PatternStep::NonGreedyStar(_, _)
        | PatternStep::GreedyPlusLookahead(_, _, _)
        | PatternStep::GreedyStarLookahead(_, _, _) => true,
        PatternStep::Alt(branches) => branches
            .iter()
            .any(|branch| has_unbounded_repetition(branch)),
        PatternStep::PositiveLookahead(inner)
        | PatternStep::NegativeLookahead(inner)
        | PatternStep::PositiveLookbehind(inner, _)
        | PatternStep::NegativeLookbehind(inner, _) => has_unbounded_repetition(inner),
        _ => false,
    })
}

/// Step kinds that decide whether the JIT can emit a program (see
/// [`jit_must_defer`]).
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[derive(Default, Clone, Copy)]
struct StepKinds {
    /// Greedy quantifiers with no attached assertion.
    greedy: usize,
    /// `Greedy*Lookahead`: quantifier and assertion emitted as one unit.
    combined: usize,
    /// Lookarounds that stand alone as their own step.
    lookaround: usize,
}

/// Returns `(greedy_quantifier_count, lookaround_count)` over a step program,
/// recursing into `Alt` branches (taking the per-branch max, since only one
/// branch executes). A combined `Greedy*Lookahead` counts as both.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
fn count_step_kinds(steps: &[PatternStep]) -> StepKinds {
    let mut counts = StepKinds::default();
    for step in steps {
        match step {
            PatternStep::GreedyPlus(_)
            | PatternStep::GreedyStar(_)
            | PatternStep::GreedyCodepointPlus(_)
            | PatternStep::NonGreedyPlus(_, _)
            | PatternStep::NonGreedyStar(_, _) => counts.greedy += 1,
            PatternStep::GreedyPlusLookahead(_, _, _)
            | PatternStep::GreedyStarLookahead(_, _, _) => counts.combined += 1,
            PatternStep::PositiveLookahead(_)
            | PatternStep::NegativeLookahead(_)
            | PatternStep::PositiveLookbehind(_, _)
            | PatternStep::NegativeLookbehind(_, _) => counts.lookaround += 1,
            PatternStep::Alt(branches) => {
                let mut worst = StepKinds::default();
                for branch in branches {
                    let branch_counts = count_step_kinds(branch);
                    worst.greedy = worst.greedy.max(branch_counts.greedy);
                    worst.combined = worst.combined.max(branch_counts.combined);
                    worst.lookaround = worst.lookaround.max(branch_counts.lookaround);
                }
                counts.greedy += worst.greedy;
                counts.combined += worst.combined;
                counts.lookaround += worst.lookaround;
            }
            _ => {}
        }
    }
    counts
}

/// Recognises the NFA shape a *greedy* `X*` over a byte class compiles to,
/// at a split state whose epsilons are exactly `[enter, exit]`.
///
/// On success returns the repeated class's byte ranges and the loop state,
/// so the caller can emit one `GreedyStar` and continue the walk at `exit`.
///
/// Shared deliberately: `StepExtractor::extract_from_state` and the JIT
/// backends' own step extractors (`jit::x86_64` / `jit::aarch64`) must model
/// `X*` identically, or the JIT's soundness gate — which asks the interpreter's
/// `StepExtractor` for permission and then generates code from its own walk —
/// no longer means what it says. One definition, three callers.
///
/// `ThompsonBuilder::build_star` emits, for the greedy form, a split state
/// preferring the body (`epsilon = [body.start, end]`) and a body-end state
/// preferring another iteration (`epsilon = [body.start, end]` again). The
/// non-greedy form reverses both orders *and* routes the exit through a
/// `NonGreedyExit` marker state, so demanding the greedy order exactly is
/// what keeps `X*?` out of this path — a marker carries an instruction and
/// no byte transitions, and is refused below.
///
/// The demand that the loop's own exit is *the same state* as the split's
/// exit is what separates a quantifier split from a genuine alternation: in
/// `(?:a+|b)` the `a+` loop leaves to the plus fragment's end, not to the
/// state the `b` branch starts at, so this answers `None` and the caller
/// keeps its `Alt` treatment. Without that check the pattern would be
/// silently miscompiled as `a*b`.
///
/// The body must be a bare byte class and nothing else: an instruction, an
/// epsilon or a match flag on it would mean something a `GreedyStar` does
/// not carry, so those shapes are declined too (`\p{L}*`, `(?:ab)*`, `\s*`
/// where the class expands to an alternation of UTF-8 shapes).
pub(crate) fn greedy_star_body(
    nfa: &Nfa,
    enter: StateId,
    exit: StateId,
) -> Option<(Vec<ByteRange>, StateId)> {
    let body = nfa.get(enter)?;
    if !body.epsilon.is_empty() || body.instruction.is_some() || body.is_match {
        return None;
    }
    let &(_, loop_id) = body.transitions.first()?;
    if !body.transitions.iter().all(|(_, t)| *t == loop_id) {
        return None;
    }

    let loop_state = nfa.get(loop_id)?;
    if !loop_state.transitions.is_empty() || loop_state.instruction.is_some() || loop_state.is_match
    {
        return None;
    }
    // Greedy order: another iteration first, the exit second — and that
    // exit is the split's own exit.
    let [loop_back, loop_exit] = loop_state.epsilon.as_slice() else {
        return None;
    };
    if *loop_back != enter || *loop_exit != exit {
        return None;
    }

    let ranges = body.transitions.iter().map(|(r, _)| *r).collect();
    Some((ranges, loop_id))
}

/// Upper bound on the number of NFA states the extractor may visit (and thus
/// steps it may emit) while building a single step program.
///
/// The emitted program is not merely explored exponentially, it IS exponential:
/// at an `Alt`, everything after the alternation is copied into every branch,
/// so `k` sequential alternation groups (`(?:ab|cd)` repeated `k` times, as in
/// tokenizer-style patterns gone pathological) produce a step program of size
/// ~2^k. Left unbounded, a ~220-character pattern with 24 such groups costs
/// ~18s and ~16.7M emitted `PatternStep`s just to build a program that is then
/// thrown away. Above this cap, extraction gives up and returns `None`, which
/// routes the pattern to the PikeVm — correct, just slower per search, and
/// already the fallback this extractor uses for every other shape it declines.
const MAX_EXTRACTED_STEPS: usize = 200_000;

/// Extracts pattern steps from an NFA for fast matching.
pub struct StepExtractor<'a> {
    nfa: &'a Nfa,
    /// Remaining budget of states/steps this extraction may still spend. See
    /// [`MAX_EXTRACTED_STEPS`]. `Cell` because the extraction methods take
    /// `&self` (they thread a mutable `visited` slice instead of `&mut self`),
    /// but every recursive call must observe and deplete the same budget.
    budget: std::cell::Cell<usize>,
}

impl<'a> StepExtractor<'a> {
    /// Creates a new step extractor for the given NFA.
    pub fn new(nfa: &'a Nfa) -> Self {
        Self {
            nfa,
            budget: std::cell::Cell::new(MAX_EXTRACTED_STEPS),
        }
    }

    /// Charges one state visit against the extraction budget, returning
    /// `false` once it is exhausted. Called at the top of every loop
    /// iteration in `extract_from_state`, `extract_branch`, and
    /// `extract_lookaround_steps` — including inside recursive `Alt` branch
    /// extraction — so a pathologically-branching pattern is caught and
    /// unwound while the program is still being built, not after.
    fn charge(&self) -> bool {
        let remaining = self.budget.get();
        if remaining == 0 {
            false
        } else {
            self.budget.set(remaining - 1);
            true
        }
    }

    /// Extracts pattern steps, returning None if pattern is too complex.
    pub fn extract(&self) -> Option<Vec<PatternStep>> {
        let mut visited = vec![false; self.nfa.states.len()];
        let steps = self.extract_from_state(self.nfa.start, &mut visited);
        if steps.is_empty() {
            return None;
        }
        // Soundness guard: the linear step extractor mis-models a quantifier split
        // (`X?`, `X*`, `X{n,m}`) as an `Alt`, which corrupts any zero-width
        // assertion attached to the quantifier in one of two ways:
        //   * it DROPS the trailing assertion (`a{1,3}(?=x)` loses the lookahead,
        //     `a{1,3}$` loses the `$`) — the step count is then LESS than the NFA's;
        //   * it DUPLICATES the assertion into each branch (`a*(?=\S)$` puts the
        //     `(?=\S)$` in both the "took some" and "took none" branches) — the
        //     step count is then GREATER than the NFA's.
        // Either way the step program no longer faithfully represents the pattern
        // and the interpreter's non-backtracking `Alt` handler diverges. A genuine
        // alternation instead carries each assertion in exactly one branch, so its
        // count matches the NFA. Bail on any mismatch to the PikeVM, which is
        // correct; tiktoken's `\s+(?!\S)` and the full cl100k/o200k patterns match
        // exactly and stay on the fast path.
        if count_assertions_in_steps(&steps) != count_assertions_in_nfa(self.nfa) {
            return None;
        }
        // Combine greedy quantifiers with following lookahead
        Some(combine_greedy_with_lookahead(steps))
    }

    /// Whether an alternation at `state` can re-enter itself.
    ///
    /// The extractor emits `Alt` as the *last* step and stops, which is only
    /// faithful when the branches run to the end of the pattern. A quantified
    /// alternation (`(?:a|b)+`, and now `\s+`, whose class expands to an
    /// alternation of UTF-8 shapes) loops back instead, so everything after the
    /// alternation — including the repetition itself — would be silently
    /// dropped, leaving a program that matches one iteration and stops. Refusing
    /// hands the pattern to the PikeVM, which models the loop.
    fn alternation_loops_back(&self, state: StateId) -> bool {
        let mut seen = vec![false; self.nfa.states.len()];
        let mut stack: Vec<StateId> = Vec::new();

        let Some(root) = self.nfa.get(state) else {
            return true;
        };
        stack.extend(root.epsilon.iter().copied());
        stack.extend(root.transitions.iter().map(|&(_, target)| target));

        while let Some(id) = stack.pop() {
            if id == state {
                return true;
            }
            match seen.get_mut(id as usize) {
                Some(flag) if !*flag => *flag = true,
                Some(_) => continue,
                // Out of range: treat as unknown, which must read as unsafe.
                None => return true,
            }
            let Some(next) = self.nfa.get(id) else {
                return true;
            };
            stack.extend(next.epsilon.iter().copied());
            stack.extend(next.transitions.iter().map(|&(_, target)| target));
        }
        false
    }

    fn extract_from_state(&self, start: StateId, visited: &mut [bool]) -> Vec<PatternStep> {
        let mut steps = Vec::new();
        let mut current = start;
        let mut iteration = 0;

        loop {
            {
                iteration += 1;
                if iteration > 1000 {
                    return Vec::new();
                }
            }

            // Extraction-budget check: see `MAX_EXTRACTED_STEPS`. Must fire
            // before any work is done for this state, so a program that would
            // blow the cap never finishes being built.
            if !self.charge() {
                return Vec::new();
            }

            if current as usize >= self.nfa.states.len() {
                return Vec::new();
            }

            let state = &self.nfa.states[current as usize];

            // Handle instructions BEFORE checking match state
            // (lookahead instructions can be on the match state itself)
            if let Some(ref instr) = state.instruction {
                match instr {
                    NfaInstruction::CaptureStart(_) | NfaInstruction::CaptureEnd(_) => {
                        // Skip capture markers for find (they don't affect matching)
                    }
                    NfaInstruction::WordBoundary => {
                        steps.push(PatternStep::WordBoundary);
                    }
                    NfaInstruction::NotWordBoundary => {
                        steps.push(PatternStep::NotWordBoundary);
                    }
                    NfaInstruction::StartOfText => {
                        steps.push(PatternStep::StartOfText);
                    }
                    NfaInstruction::EndOfText => {
                        steps.push(PatternStep::EndOfText);
                    }
                    NfaInstruction::PositiveLookahead(inner_nfa) => {
                        let inner_steps = self.extract_lookaround_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return Vec::new();
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
                        // The lookbehind walk needs exact totals, not a minimum,
                        // but not a *unique* total — see `byte_len_set`.
                        let inner_steps = self.extract_lookbehind_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        let Some(widths) = byte_len_set(&inner_steps) else {
                            return Vec::new();
                        };
                        steps.push(PatternStep::PositiveLookbehind(inner_steps, widths));
                    }
                    NfaInstruction::NegativeLookbehind(inner_nfa) => {
                        let inner_steps = self.extract_lookbehind_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return Vec::new();
                        }
                        let Some(widths) = byte_len_set(&inner_steps) else {
                            return Vec::new();
                        };
                        steps.push(PatternStep::NegativeLookbehind(inner_steps, widths));
                    }
                    NfaInstruction::CodepointClass(cpclass, target) => {
                        // Unicode codepoint class - check for greedy loop pattern
                        if (*target as usize) < self.nfa.states.len() {
                            let target_state = &self.nfa.states[*target as usize];
                            if target_state.epsilon.len() == 2
                                && target_state.transitions.is_empty()
                            {
                                let eps0 = target_state.epsilon[0];
                                let eps1 = target_state.epsilon[1];

                                // Check if this is a greedy plus (X+) pattern
                                if eps0 == current {
                                    // Found greedy plus: emit GreedyCodepointPlus and continue to eps1
                                    steps.push(PatternStep::GreedyCodepointPlus(cpclass.clone()));
                                    visited[current as usize] = true;
                                    visited[*target as usize] = true;
                                    current = eps1;
                                    continue;
                                } else if eps1 == current {
                                    // Found greedy plus: emit GreedyCodepointPlus and continue to eps0
                                    steps.push(PatternStep::GreedyCodepointPlus(cpclass.clone()));
                                    visited[current as usize] = true;
                                    visited[*target as usize] = true;
                                    current = eps0;
                                    continue;
                                }
                            }
                        }
                        // Not a greedy loop - emit single CodepointClass and continue
                        steps.push(PatternStep::CodepointClass(cpclass.clone(), *target));
                        if visited[current as usize] {
                            return Vec::new();
                        }
                        visited[current as usize] = true;
                        current = *target;
                        continue;
                    }
                    _ => {
                        // Unsupported instruction
                        return Vec::new();
                    }
                }
            }

            // Handle byte transitions
            if !state.transitions.is_empty() {
                let target = state.transitions[0].1;
                if !state.transitions.iter().all(|(_, t)| *t == target) {
                    return Vec::new();
                }

                let ranges: Vec<ByteRange> = state.transitions.iter().map(|(r, _)| *r).collect();

                // Check for greedy loop
                let target_state = &self.nfa.states[target as usize];
                if target_state.epsilon.len() == 2 && target_state.transitions.is_empty() {
                    let eps0 = target_state.epsilon[0];
                    let eps1 = target_state.epsilon[1];

                    if eps0 == current {
                        // Greedy plus: loop back
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        if visited[target as usize] {
                            return Vec::new();
                        }
                        visited[target as usize] = true;
                        current = eps1;
                        continue;
                    }
                }

                // Regular transition
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

            // Handle epsilon transitions
            if state.epsilon.len() == 1 {
                if visited[current as usize] {
                    return Vec::new();
                }
                visited[current as usize] = true;
                current = state.epsilon[0];
                continue;
            }

            // A nullable greedy run (`X*`) is a split state, and the generic
            // alternation path below models it as a two-branch `Alt` with every
            // trailing step duplicated into both branches. That duplication is
            // what the assertion tally rejects (`\w*$`, `\w*\b`), and for a
            // trailing lookaround it is worse: that assertion is compiled onto
            // the match state, where `extract_branch` reads it as unsupported
            // (`terminal_assertion`) and abandons the whole walk, so
            // `\w*(?=ing)` extracts nothing at all.
            // Recognising the shape here emits a single `GreedyStar` and keeps
            // the walk linear, so the trailing steps are extracted exactly once
            // and stay on this engine instead of falling back to the PikeVm.
            //
            // Only the exact greedy `X*` shape takes this path — see
            // `greedy_star_body`. Anything else, a non-greedy `X*?` and a
            // genuine alternation included, falls through to the `Alt` handling
            // below unchanged.
            if let [enter, exit] = state.epsilon.as_slice() {
                let (enter, exit) = (*enter, *exit);
                if let Some((ranges, loop_state)) = greedy_star_body(self.nfa, enter, exit) {
                    // Same cycle rule as the `GreedyPlus` case above: every
                    // state this step consumes must be fresh, or a cyclic NFA
                    // could walk it forever.
                    let fresh = [current, enter, loop_state]
                        .iter()
                        .all(|&id| visited.get(id as usize).is_some_and(|seen| !*seen));
                    if fresh {
                        steps.push(PatternStep::GreedyStar(ByteClass::new(ranges)));
                        for id in [current, enter, loop_state] {
                            if let Some(seen) = visited.get_mut(id as usize) {
                                *seen = true;
                            }
                        }
                        current = exit;
                        continue;
                    }
                }
            }

            // Multiple epsilon = alternation
            if state.epsilon.len() >= 2 {
                // Extract each alternative branch
                let mut alternatives: Vec<Vec<PatternStep>> = Vec::new();
                for &target in state.epsilon.iter() {
                    let mut branch_visited = visited.to_vec();
                    branch_visited[current as usize] = true;
                    let Some(branch_steps) = self.extract_branch(target, &mut branch_visited)
                    else {
                        // A branch this walk cannot represent: fall back.
                        return Vec::new();
                    };
                    alternatives.push(branch_steps);
                }
                steps.push(PatternStep::Alt(alternatives));
                break; // Alternation consumes the rest of the pattern
            }

            break;
        }

        steps
    }

    /// Extracts steps from a single branch of an alternation.
    fn extract_branch(&self, start: StateId, visited: &mut [bool]) -> Option<Vec<PatternStep>> {
        let mut steps = Vec::new();
        let mut current = start;
        let mut iteration = 0;

        loop {
            {
                iteration += 1;
                if iteration > 10000 {
                    return None;
                }
            }

            // Extraction-budget check: see `MAX_EXTRACTED_STEPS`. This is the
            // function alternation branches recurse through, so it is what
            // catches the exponential case — the check fires before the
            // recursive `Alt` branch calls that would otherwise double the
            // work at every nested alternation.
            if !self.charge() {
                return None;
            }

            if current as usize >= self.nfa.states.len() {
                return None;
            }

            if visited[current as usize] {
                return None;
            }

            let state = &self.nfa.states[current as usize];

            // Match state = end of this branch. An assertion compiled onto it is
            // still part of the branch — `a*(?=b)(?=c)` puts the `(?=c)` here —
            // so read it before stopping. See `terminal_assertion`.
            if state.is_match {
                match terminal_assertion(state) {
                    TerminalAssertion::Nothing => {}
                    TerminalAssertion::Step(step) => steps.push(step),
                    TerminalAssertion::Unsupported => return None,
                }
                break;
            }

            // Handle instructions
            if let Some(ref instr) = state.instruction {
                match instr {
                    NfaInstruction::CaptureStart(_) | NfaInstruction::CaptureEnd(_) => {
                        // Skip capture markers
                    }
                    NfaInstruction::WordBoundary => {
                        steps.push(PatternStep::WordBoundary);
                    }
                    NfaInstruction::NotWordBoundary => {
                        steps.push(PatternStep::NotWordBoundary);
                    }
                    NfaInstruction::StartOfText => {
                        steps.push(PatternStep::StartOfText);
                    }
                    NfaInstruction::EndOfText => {
                        steps.push(PatternStep::EndOfText);
                    }
                    NfaInstruction::PositiveLookahead(inner_nfa) => {
                        let inner_steps = self.extract_lookaround_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return None;
                        }
                        steps.push(PatternStep::PositiveLookahead(inner_steps));
                    }
                    NfaInstruction::NegativeLookahead(inner_nfa) => {
                        let inner_steps = self.extract_lookaround_steps(inner_nfa);
                        if inner_steps.is_empty() {
                            return None;
                        }
                        steps.push(PatternStep::NegativeLookahead(inner_steps));
                    }
                    NfaInstruction::CodepointClass(cpclass, target) => {
                        // Check for greedy loop pattern: CodepointClass -> epsilon state -> back to current
                        if (*target as usize) >= self.nfa.states.len() {
                            return None;
                        }

                        let target_state = &self.nfa.states[*target as usize];
                        if target_state.epsilon.len() == 2 && target_state.transitions.is_empty() {
                            let eps0 = target_state.epsilon[0];
                            let eps1 = target_state.epsilon[1];

                            // Check if this is a greedy plus (X+) pattern
                            if eps0 == current {
                                // Found greedy plus: CodepointClass+ - emit GreedyCodepointPlus and continue to eps1
                                steps.push(PatternStep::GreedyCodepointPlus(cpclass.clone()));
                                visited[current as usize] = true;
                                visited[*target as usize] = true;
                                current = eps1;
                                continue;
                            } else if eps1 == current {
                                // Found greedy plus: CodepointClass+ - emit GreedyCodepointPlus and continue to eps0
                                steps.push(PatternStep::GreedyCodepointPlus(cpclass.clone()));
                                visited[current as usize] = true;
                                visited[*target as usize] = true;
                                current = eps0;
                                continue;
                            }
                        }

                        // Not a greedy loop - just emit and continue
                        steps.push(PatternStep::CodepointClass(cpclass.clone(), *target));
                        visited[current as usize] = true;
                        current = *target;
                        continue;
                    }
                    _ => {
                        return None;
                    }
                }
            }

            // Handle byte transitions
            if !state.transitions.is_empty() {
                let target = state.transitions[0].1;
                if !state.transitions.iter().all(|(_, t)| *t == target) {
                    return None;
                }

                let ranges: Vec<ByteRange> = state.transitions.iter().map(|(r, _)| *r).collect();

                // Check for greedy loop
                let target_state = &self.nfa.states[target as usize];
                if target_state.epsilon.len() == 2 && target_state.transitions.is_empty() {
                    let eps0 = target_state.epsilon[0];
                    let eps1 = target_state.epsilon[1];

                    if eps0 == current {
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        visited[current as usize] = true;
                        visited[target as usize] = true;
                        current = eps1;
                        continue;
                    } else if eps1 == current {
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        visited[current as usize] = true;
                        visited[target as usize] = true;
                        current = eps0;
                        continue;
                    }
                }

                // Simple transition
                steps.push(PatternStep::ByteClass(ByteClass::new(ranges)));
                visited[current as usize] = true;
                current = target;
                continue;
            }

            // Handle epsilon transitions
            if state.epsilon.len() == 1 {
                visited[current as usize] = true;
                current = state.epsilon[0];
                continue;
            }

            // Two epsilons - could be greedy star or alternation
            if state.epsilon.len() == 2 {
                let eps0 = state.epsilon[0];
                let eps1 = state.epsilon[1];

                // Check for greedy loop: if one epsilon leads to an already-visited state,
                // and we have a CodepointClass step, this is a greedy plus/star pattern
                let eps0_visited = visited[eps0 as usize];
                let eps1_visited = visited[eps1 as usize];

                if eps0_visited && !eps1_visited {
                    // eps0 is the back-edge of a greedy loop, eps1 is the exit
                    // Continue with the exit branch
                    visited[current as usize] = true;
                    current = eps1;
                    continue;
                }
                if eps1_visited && !eps0_visited {
                    // eps1 is the back-edge of a greedy loop, eps0 is the exit
                    visited[current as usize] = true;
                    current = eps0;
                    continue;
                }

                // Not a greedy star, treat as alternation
                if self.alternation_loops_back(current) {
                    return None;
                }
                let mut alternatives: Vec<Vec<PatternStep>> = Vec::new();
                let mut any_valid = false;
                for &target in state.epsilon.iter() {
                    let mut branch_visited = visited.to_vec();
                    branch_visited[current as usize] = true;
                    // Check if this branch can reach the match state
                    let target_state = &self.nfa.states[target as usize];
                    if target_state.is_match {
                        // Empty branch directly to match (like in X? patterns)
                        alternatives.push(Vec::new());
                        any_valid = true;
                        continue;
                    }
                    // A branch that cannot be represented is not an empty branch.
                    // Treating the two alike lets the rest of the pattern vanish
                    // from that arm while the step program still looks complete.
                    let branch_steps = self.extract_branch(target, &mut branch_visited)?;
                    alternatives.push(branch_steps);
                    any_valid = true;
                }
                if !any_valid {
                    return None;
                }
                steps.push(PatternStep::Alt(alternatives));
                break;
            }

            // More than 2 epsilons - must be alternation
            if state.epsilon.len() > 2 {
                if self.alternation_loops_back(current) {
                    return None;
                }
                let mut alternatives: Vec<Vec<PatternStep>> = Vec::new();
                let mut any_valid = false;
                for &target in state.epsilon.iter() {
                    let mut branch_visited = visited.to_vec();
                    branch_visited[current as usize] = true;
                    // Check if this branch directly reaches match state
                    let target_state = &self.nfa.states[target as usize];
                    if target_state.is_match {
                        alternatives.push(Vec::new());
                        any_valid = true;
                        continue;
                    }
                    let branch_steps = self.extract_branch(target, &mut branch_visited)?;
                    alternatives.push(branch_steps);
                    any_valid = true;
                }
                if !any_valid {
                    return None;
                }
                steps.push(PatternStep::Alt(alternatives));
                break;
            }

            // Dead end - no transitions, no epsilon, not a match state
            return None;
        }

        Some(steps)
    }

    fn extract_lookaround_steps(&self, inner_nfa: &Nfa) -> Vec<PatternStep> {
        // A nested assertion cannot be modelled by this linear walk: the loop
        // breaks on the match state before reading that state's instruction, so
        // a trailing inner assertion would be dropped and the resulting empty
        // step list read as "assertion satisfied" — turning an unsupported
        // pattern into a wrong match. Refuse instead, so the caller falls back
        // to the PikeVm, which evaluates nested lookaround correctly.
        if nfa_has_lookaround(inner_nfa) {
            return Vec::new();
        }

        let mut visited = vec![false; inner_nfa.states.len()];
        let mut steps = Vec::new();
        let mut current = inner_nfa.start;
        let mut iteration = 0;

        loop {
            {
                iteration += 1;
                if iteration > 10000 {
                    return Vec::new();
                }
            }

            // Extraction-budget check: see `MAX_EXTRACTED_STEPS`. Shares the
            // same budget as the outer walk, since a lookaround interior is
            // still part of the same extraction attempt.
            if !self.charge() {
                return Vec::new();
            }

            if current as usize >= inner_nfa.states.len() {
                return Vec::new();
            }

            let state = &inner_nfa.states[current as usize];

            if state.is_match {
                match terminal_assertion(state) {
                    TerminalAssertion::Nothing => {}
                    TerminalAssertion::Step(step) => steps.push(step),
                    TerminalAssertion::Unsupported => return Vec::new(),
                }
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
                    NfaInstruction::CaptureStart(_) | NfaInstruction::CaptureEnd(_) => {
                        // Skip capture markers in lookaround
                    }
                    NfaInstruction::CodepointClass(cpclass, target) => {
                        // Unicode codepoint class in lookaround
                        steps.push(PatternStep::CodepointClass(cpclass.clone(), *target));
                        if visited[current as usize] {
                            return Vec::new();
                        }
                        visited[current as usize] = true;
                        current = *target;
                        continue;
                    }
                    _ => {
                        return Vec::new();
                    }
                }
            }

            if !state.transitions.is_empty() {
                let target = state.transitions[0].1;
                if !state.transitions.iter().all(|(_, t)| *t == target) {
                    return Vec::new();
                }

                let ranges: Vec<ByteRange> = state.transitions.iter().map(|(r, _)| *r).collect();

                // Check for greedy star/plus pattern: state has transitions to target,
                // and target has epsilon transitions where one leads back to current state
                let target_state = &inner_nfa.states[target as usize];

                // Pattern for greedy plus: current -[byte]-> target -[eps]-> current (loop back)
                //                                              |-> next (exit)
                if target_state.transitions.is_empty() && target_state.epsilon.len() == 2 {
                    let eps0 = target_state.epsilon[0];
                    let eps1 = target_state.epsilon[1];

                    // Check if one epsilon leads back to current (greedy loop)
                    if eps0 == current {
                        // Greedy plus: must match at least one
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        if visited[target as usize] {
                            return Vec::new();
                        }
                        visited[target as usize] = true;
                        current = eps1; // Continue from exit path
                        continue;
                    } else if eps1 == current {
                        // Greedy plus: loop back is second epsilon
                        steps.push(PatternStep::GreedyPlus(ByteClass::new(ranges)));
                        if visited[target as usize] {
                            return Vec::new();
                        }
                        visited[target as usize] = true;
                        current = eps0; // Continue from exit path
                        continue;
                    }
                }

                // Check for greedy star pattern: current has epsilon to both:
                // - a state with byte transitions (the loop body)
                // - a state that continues (the exit)
                // This is handled below in epsilon section

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

            // Handle greedy star: two epsilons where one leads to byte transitions
            // that loop back, and one leads to exit
            if state.epsilon.len() == 2 && state.transitions.is_empty() {
                let eps0 = state.epsilon[0];
                let eps1 = state.epsilon[1];

                // Try to detect: eps0 has transitions that loop, eps1 exits
                // or vice versa
                if let Some((ranges, exit_state)) =
                    self.detect_greedy_star_in_lookaround(inner_nfa, current, eps0, eps1, &visited)
                {
                    steps.push(PatternStep::GreedyStar(ByteClass::new(ranges)));
                    visited[current as usize] = true;
                    current = exit_state;
                    continue;
                }
                if let Some((ranges, exit_state)) =
                    self.detect_greedy_star_in_lookaround(inner_nfa, current, eps1, eps0, &visited)
                {
                    steps.push(PatternStep::GreedyStar(ByteClass::new(ranges)));
                    visited[current as usize] = true;
                    current = exit_state;
                    continue;
                }

                // Not a recognized greedy star pattern
                return Vec::new();
            }

            if !state.epsilon.is_empty() {
                return Vec::new();
            }

            break;
        }

        steps
    }

    /// Extracts pattern steps from a lookbehind inner NFA.
    /// This is simpler than lookahead extraction - it doesn't recognize GreedyStar/Plus
    /// because check_lookbehind walks from a fixed offset behind the position.
    /// Patterns with repetitions in lookbehind will return empty, causing fallback to PikeVM.
    ///
    /// Not charged against [`MAX_EXTRACTED_STEPS`]: unlike `extract_from_state` /
    /// `extract_branch`, this walk never emits `PatternStep::Alt` or recurses into
    /// a branch extractor — any epsilon fan-out it meets is rejected outright (see
    /// the loop-back check below) rather than expanded — so it cannot itself
    /// contribute to the exponential blowup the budget guards against.
    fn extract_lookbehind_steps(&self, inner_nfa: &Nfa) -> Vec<PatternStep> {
        // Same reason as `extract_lookaround_steps`: refuse a nested assertion
        // rather than silently dropping it.
        if nfa_has_lookaround(inner_nfa) {
            return Vec::new();
        }

        let mut visited = vec![false; inner_nfa.states.len()];
        let mut steps = Vec::new();
        let mut current = inner_nfa.start;

        loop {
            if current as usize >= inner_nfa.states.len() {
                return Vec::new();
            }

            let state = &inner_nfa.states[current as usize];

            if state.is_match {
                match terminal_assertion(state) {
                    TerminalAssertion::Nothing => {}
                    TerminalAssertion::Step(step) => steps.push(step),
                    TerminalAssertion::Unsupported => return Vec::new(),
                }
                break;
            }

            // A codepoint class consumes input, so it advances the walk rather
            // than being appended as a zero-width step. `byte_len_set` enumerates
            // every total the class can contribute — and refuses the whole
            // lookbehind if it cannot — so the interpreter has an exact offset to
            // start each backwards walk from.
            if let Some(NfaInstruction::CodepointClass(cpclass, target)) = &state.instruction {
                steps.push(PatternStep::CodepointClass(cpclass.clone(), *target));
                if visited[current as usize] {
                    return Vec::new();
                }
                visited[current as usize] = true;
                current = *target;
                continue;
            }

            // Interior assertions translate the same way as a terminal one — the
            // position in the walk does not change what the assertion means.
            match terminal_assertion(state) {
                TerminalAssertion::Nothing => {}
                TerminalAssertion::Step(step) => steps.push(step),
                TerminalAssertion::Unsupported => return Vec::new(),
            }

            if !state.transitions.is_empty() {
                let target = state.transitions[0].1;
                if !state.transitions.iter().all(|(_, t)| *t == target) {
                    return Vec::new();
                }

                let ranges: Vec<ByteRange> = state.transitions.iter().map(|(r, _)| *r).collect();

                // Check for repetition patterns - we can't handle these in lookbehind
                let target_state = &inner_nfa.states[target as usize];
                if target_state.transitions.is_empty() && target_state.epsilon.len() == 2 {
                    let eps0 = target_state.epsilon[0];
                    let eps1 = target_state.epsilon[1];
                    // If any epsilon leads back to current, it's a loop - reject
                    if eps0 == current || eps1 == current {
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

            // Any other epsilon patterns (including repetitions) - reject
            if !state.epsilon.is_empty() {
                return Vec::new();
            }

            break;
        }

        steps
    }

    /// Detects a greedy star pattern where `loop_start` has transitions that loop back
    /// to `loop_start` itself (or to a state that leads back), and `exit_state` is the continuation.
    /// Returns (ranges, exit_state) if detected, None otherwise.
    fn detect_greedy_star_in_lookaround(
        &self,
        inner_nfa: &Nfa,
        _branch_state: StateId,
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

        // The target should have epsilon back to loop_start (completing the loop)
        let target_state = &inner_nfa.states[target as usize];

        // Check if target has two epsilons where one leads back to loop_start
        if target_state.epsilon.len() == 2 {
            let eps0 = target_state.epsilon[0];
            let eps1 = target_state.epsilon[1];

            // Check if one epsilon leads back to loop_start
            if eps0 == loop_start {
                // eps0 loops back, eps1 exits
                if !visited[loop_start as usize] {
                    // The exit should eventually lead to exit_state or be it
                    return Some((ranges, exit_state));
                }
            } else if eps1 == loop_start {
                // eps1 loops back, eps0 exits
                if !visited[loop_start as usize] {
                    return Some((ranges, exit_state));
                }
            }
        }

        // Simple case: target has single epsilon back to loop_start
        if target_state.epsilon.len() == 1
            && target_state.epsilon[0] == loop_start
            && !visited[loop_start as usize]
        {
            return Some((ranges, exit_state));
        }

        None
    }
}

#[cfg(test)]
mod fixed_byte_len_tests {
    use super::*;
    use crate::hir::CodepointClass;

    fn class(ranges: &[(u32, u32)], negated: bool) -> PatternStep {
        PatternStep::CodepointClass(CodepointClass::new(ranges.to_vec(), negated), 0)
    }

    #[test]
    fn consuming_steps_add_their_width() {
        assert_eq!(fixed_byte_len(&[]), Some(0));
        assert_eq!(
            fixed_byte_len(&[PatternStep::Byte(b'a'), PatternStep::Byte(b'b')]),
            Some(2)
        );
    }

    #[test]
    fn assertions_and_capture_markers_are_zero_width() {
        assert_eq!(
            fixed_byte_len(&[
                PatternStep::StartOfText,
                PatternStep::Byte(b'a'),
                PatternStep::WordBoundary,
                PatternStep::NotWordBoundary,
                PatternStep::EndOfText,
                PatternStep::StartOfLine,
                PatternStep::EndOfLine,
                PatternStep::CaptureStart(1),
                PatternStep::CaptureEnd(1),
            ]),
            Some(1)
        );
    }

    #[test]
    fn codepoint_class_counts_its_utf8_width() {
        // Greek: every codepoint encodes to two bytes.
        assert_eq!(fixed_byte_len(&[class(&[(0x3B1, 0x3C9)], false)]), Some(2));
        // CJK: three bytes.
        assert_eq!(
            fixed_byte_len(&[class(&[(0x4E00, 0x9FFF)], false)]),
            Some(3)
        );
        // ASCII: one byte.
        assert_eq!(
            fixed_byte_len(&[class(&[(b'a' as u32, b'z' as u32)], false)]),
            Some(1)
        );
        // Astral: four bytes.
        assert_eq!(
            fixed_byte_len(&[class(&[(0x1F600, 0x1F64F)], false)]),
            Some(4)
        );
    }

    #[test]
    fn variable_width_codepoint_classes_are_refused() {
        // A range straddling the 1/2-byte boundary.
        assert_eq!(fixed_byte_len(&[class(&[(0x7F, 0x80)], false)]), None);
        // Two ranges of differing width.
        assert_eq!(
            fixed_byte_len(&[class(&[(b'a' as u32, b'z' as u32), (0x3B1, 0x3C9)], false)]),
            None
        );
        // A negated class admits codepoints of every width.
        assert_eq!(fixed_byte_len(&[class(&[(0x3B1, 0x3C9)], true)]), None);
        // An empty class matches nothing, so it has no width to report.
        assert_eq!(fixed_byte_len(&[class(&[], false)]), None);
    }

    #[test]
    fn variable_width_steps_are_refused() {
        let bytes = ByteClass::new(vec![ByteRange::new(b'a', b'z')]);
        assert_eq!(
            fixed_byte_len(&[PatternStep::GreedyPlus(bytes.clone())]),
            None
        );
        assert_eq!(
            fixed_byte_len(&[PatternStep::GreedyStar(bytes.clone())]),
            None
        );
        assert_eq!(fixed_byte_len(&[PatternStep::Backref(1)]), None);
        assert_eq!(
            fixed_byte_len(&[PatternStep::Alt(vec![
                vec![PatternStep::Byte(b'a')],
                vec![PatternStep::Byte(b'a'), PatternStep::Byte(b'b')],
            ])]),
            None
        );
    }

    #[test]
    fn one_refusal_refuses_the_whole_program() {
        assert_eq!(
            fixed_byte_len(&[PatternStep::Byte(b'a'), PatternStep::Backref(1)]),
            None
        );
    }

    #[test]
    fn width_set_of_a_fixed_program_is_a_single_total() {
        assert_eq!(byte_len_set(&[]), Some(vec![0]));
        assert_eq!(
            byte_len_set(&[PatternStep::Byte(b'a'), PatternStep::Byte(b'b')]),
            Some(vec![2])
        );
        assert_eq!(
            byte_len_set(&[PatternStep::StartOfText, class(&[(0x3B1, 0x3C9)], false)]),
            Some(vec![2])
        );
    }

    #[test]
    fn a_range_straddling_an_encoding_boundary_spans_both_widths() {
        assert_eq!(
            byte_len_set(&[class(&[(0x7F, 0x80)], false)]),
            Some(vec![1, 2])
        );
        // The whole encodable space touches every width.
        assert_eq!(
            byte_len_set(&[class(&[(0, 0x10FFFF)], false)]),
            Some(vec![1, 2, 3, 4])
        );
    }

    #[test]
    fn separate_ranges_contribute_their_own_widths() {
        // `\s`-shaped: ASCII space (1), NBSP (2), em space (3). No 4-byte member.
        assert_eq!(
            byte_len_set(&[class(
                &[(0x20, 0x20), (0xA0, 0xA0), (0x2003, 0x2003)],
                false
            )]),
            Some(vec![1, 2, 3])
        );
    }

    #[test]
    fn widths_of_successive_steps_form_a_sumset() {
        let two_spaces = [
            class(&[(0x20, 0x20), (0xA0, 0xA0)], false),
            class(&[(0x20, 0x20), (0xA0, 0xA0)], false),
        ];
        // {1,2} + {1,2} = {2,3,4}, deduplicated: 1+2 and 2+1 both give 3.
        assert_eq!(byte_len_set(&two_spaces), Some(vec![2, 3, 4]));
        // A fixed step shifts every candidate by its own width.
        let space_then_x = [
            class(&[(0x20, 0x20), (0xA0, 0xA0)], false),
            PatternStep::Byte(b'x'),
        ];
        assert_eq!(byte_len_set(&space_then_x), Some(vec![2, 3]));
    }

    #[test]
    fn width_sets_that_fan_out_past_the_cap_are_declined() {
        // Each class admits 1..=4 bytes, so `n` of them total `n..=4n`: five
        // give exactly 16 candidates (the cap) and six give 19 (over it).
        let all_widths = class(&[(0, 0x10FFFF)], false);
        let five = vec![all_widths.clone(); 5];
        let six = vec![all_widths; 6];
        assert_eq!(MAX_LOOKBEHIND_WIDTHS, 16);
        assert_eq!(byte_len_set(&five).map(|w| w.len()), Some(16));
        assert_eq!(byte_len_set(&six), None);
    }

    #[test]
    fn variable_width_steps_have_no_width_set() {
        let bytes = ByteClass::new(vec![ByteRange::new(b'a', b'z')]);
        assert_eq!(byte_len_set(&[PatternStep::GreedyPlus(bytes)]), None);
        assert_eq!(byte_len_set(&[PatternStep::Backref(1)]), None);
        // A negated class stays out of scope, as in `fixed_byte_len`.
        assert_eq!(byte_len_set(&[class(&[(0x3B1, 0x3C9)], true)]), None);
        assert_eq!(byte_len_set(&[class(&[], false)]), None);
    }

    #[test]
    fn fixed_byte_len_is_the_single_candidate_case() {
        // Multi-width programs are extractable as a set but have no fixed width.
        let mixed = [class(&[(0x20, 0x20), (0xA0, 0xA0)], false)];
        assert_eq!(byte_len_set(&mixed), Some(vec![1, 2]));
        assert_eq!(fixed_byte_len(&mixed), None);
    }
}

#[cfg(test)]
mod extraction_budget_tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn build_nfa(pattern: &str) -> Nfa {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        crate::nfa::compile(&hir).unwrap()
    }

    // The real tiktoken split patterns (cl100k/o200k-style). Guarding these
    // against `MAX_EXTRACTED_STEPS` matters: regexr backs a BPE tokenizer whose
    // throughput depends on these specific patterns staying on the step-based
    // fast path rather than falling back to the PikeVm. Reused verbatim from
    // `tests/reference_conformance.rs`.
    const CL100K: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s";
    const O200K: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    #[test]
    fn tokenizer_patterns_stay_under_the_budget() {
        for pattern in [CL100K, O200K] {
            let nfa = build_nfa(pattern);
            assert!(
                StepExtractor::new(&nfa).extract().is_some(),
                "extraction budget pushed a real tokenizer pattern to the PikeVm fallback: {pattern}"
            );
        }
    }

    #[test]
    fn pathological_sequential_alternations_are_declined_not_built() {
        // (?=a) + k sequential alternation groups. Each group duplicates every
        // step after it into both branches, so unbounded extraction is ~2^k
        // steps. Comfortably above `MAX_EXTRACTED_STEPS`, this must decline
        // (`None`) rather than spend seconds building a multi-million-step
        // program that is then discarded.
        let groups = ["(?:ab|cd)", "(?:ef|gh)", "(?:ij|kl)", "(?:mn|op)"];
        let mut pattern = String::from("(?=a)");
        for i in 0..24 {
            pattern.push_str(groups[i % groups.len()]);
        }
        let nfa = build_nfa(&pattern);
        assert!(StepExtractor::new(&nfa).extract().is_none());
    }
}

/// Pins which nullable runs beside an assertion extract, and which still do not.
///
/// A greedy `X*` over a byte class is recognised structurally by
/// `greedy_star_body` and emitted as one [`PatternStep::GreedyStar`], so the
/// steps after it are extracted once and the pattern stays on this engine —
/// that is what makes [`PatternStep::GreedyStarLookahead`] reachable from a
/// pattern string at all.
///
/// Every other nullable shape is still modelled as an `Alt` of "took
/// some"/"took none", which duplicates the trailing steps into both branches.
/// That `Alt` must not be "fixed" by emitting the trailing steps once: an
/// attempt to do so produced 23 reference divergences. So those shapes stay
/// declined — a duplicated assertion trips the tally guard, and a *trailing*
/// lookaround is compiled onto the match state, which `extract_branch` reads as
/// unsupported and gives up on — and fall back to the PikeVm, the slowest
/// engine available.
///
/// Both directions are pinned on purpose: the first test fails if the
/// recognition stops firing, the rest if it starts firing on a shape it must
/// leave alone.
#[cfg(test)]
mod nullable_run_with_assertion_tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn extract(pattern: &str) -> Option<Vec<PatternStep>> {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = crate::nfa::compile(&hir).unwrap();
        StepExtractor::new(&nfa).extract()
    }

    fn has_star_lookahead(steps: &[PatternStep]) -> bool {
        steps
            .iter()
            .any(|s| matches!(s, PatternStep::GreedyStarLookahead(_, _, _)))
    }

    #[test]
    fn a_greedy_byte_class_run_beside_an_assertion_now_extracts() {
        // Bare, and with a consuming prefix so the run is not at the start.
        for pattern in [
            r"\w*(?=ing)",
            r"a\w*(?=ing)",
            r"[a-z]*(?=xy)",
            r"\d[a-z]*(?=xy)",
            r"[a-z]*(?=x)",
            r"\w*(?!ing)",
        ] {
            let steps =
                extract(pattern).unwrap_or_else(|| panic!("{pattern:?} no longer extracts at all"));
            assert!(
                has_star_lookahead(&steps),
                "{pattern:?} extracted no combined greedy-star+lookahead step: {steps:?}"
            );
        }
    }

    #[test]
    fn a_nullable_run_that_is_not_a_greedy_byte_class_is_still_declined() {
        for (pattern, why) in [
            // Non-greedy: the split prefers the exit and routes it through a
            // `NonGreedyExit` marker, which is not the shape recognised.
            (r"\w*?(?=ing)", "non-greedy star"),
            // Multi-state body: the loop state carries byte transitions of its
            // own, so it is not a single repeated class.
            (r"(?:ab)*(?=x)", "multi-state body"),
            // Codepoint-class body: the body state carries an instruction.
            (r"\p{L}*(?=x)", "codepoint class body"),
            // `X?` is a split whose body never loops back.
            (r"\w?(?=ing)", "optional, not a repeat"),
            // Nested in an alternation branch: the recognition lives in the
            // top-level walk, and the branch walk refuses a split that can
            // re-enter itself, so the inner run is never reached.
            (r"(?:x|\w*(?=ing))", "nullable run inside an Alt branch"),
            (r"(?:\w*(?=ing)|q)", "nullable run inside an Alt branch"),
        ] {
            assert!(
                extract(pattern).is_none(),
                "{pattern:?} ({why}) now extracts; the structural star check may \
                 be too loose"
            );
        }
    }

    #[test]
    fn a_genuine_alternation_is_not_recognised_as_a_star() {
        // `(?:a+|b)` reaches the same split-with-two-epsilons shape, but the
        // loop's exit is the plus fragment's end, not the state the `b` branch
        // starts at. Recognising it as a star would silently compile `a*b`.
        for pattern in [r"(?:a+|b)", r"(?:a+|b)c"] {
            let steps =
                extract(pattern).unwrap_or_else(|| panic!("{pattern:?} no longer extracts"));
            assert!(
                !steps
                    .iter()
                    .any(|s| matches!(s, PatternStep::GreedyStar(_)))
                    && !has_star_lookahead(&steps),
                "{pattern:?} was miscompiled as `a*b`: {steps:?}"
            );
        }
        // The same shape with the trailing lookahead that makes the branch walk
        // give up: it must stay declined rather than be rescued by a star
        // recognition that ignores where the loop exits.
        assert!(
            extract(r"(?:\w+|x)(?=ing)").is_none(),
            "an alternation was recognised as a star: the exit-convergence check \
             is too loose"
        );
    }

    #[test]
    fn the_non_nullable_sibling_still_takes_the_fast_path() {
        // `+` instead of `*`: the loop is entered unconditionally, so there is
        // no split to recognise and the plus-shaped combined step is built.
        let steps = extract(r"\w+(?=ing)").expect("plus form extracts");
        assert!(
            steps
                .iter()
                .any(|s| matches!(s, PatternStep::GreedyPlusLookahead(_, _, _))),
            "expected a combined greedy+lookahead step, got {steps:?}"
        );
    }
}
