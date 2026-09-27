//! The tagged-NFA JIT must agree with its interpreter, step for step.
//!
//! `jit_must_defer` decides which step programs the JIT may emit. Loosening it
//! trades a guaranteed-correct interpreter for generated code, so every shape it
//! now admits — greedy quantifiers carrying their own lookahead — is compared
//! against the interpreter here, both through engine selection and compiled
//! directly (selection keeps programs that repeat without bound on the
//! interpreter). A divergence means the guard was loosened too far, not that a
//! test is too strict.
//!
//! The engine-name assertions need the `jit` feature; the agreement sweep does
//! not and runs in every configuration.

use regexr::{Regex, RegexBuilder};

/// Shapes that reach `PatternStep::GreedyPlusLookahead` / `GreedyStarLookahead`,
/// plus the adjacent-greedy shapes that must stay deferred, plus lookbehind over
/// classes with more than one candidate width — the JIT emits one attempt per
/// candidate there, and picking the wrong one returns wrong spans rather than
/// failing loudly.
const PATTERNS: &[&str] = &[
    r"[\r\n]+(?=\S)",
    r"\S+\S+\S",
    r"\w+(?=ing\b)",
    r"\s+(?!\S)",
    r"\S+(?=\s)",
    r"a+(?=b)",
    r"a*(?=b)",
    r"a+(?!b)",
    r"a*(?!b)",
    r"[a-z]+(?=[0-9])",
    r"[a-z]*(?=[0-9])",
    r"\w+\w+",
    r"\s+\s+",
    r"\S+\s+\S+",
    r"a+b+",
    r"a+b+c",
    r"[\r\n]+(?=\S)x?",
    r"\w+(?=\d)\d",
    r"a+(?=ab)ab",
    r"\s*(?=\S)\S+",
    r"[ab]+(?=c)",
    r"[ab]*(?!c)",
    r"\d+(?=px\b)",
    r"\w+(?=\.)",
    r"[^,]+(?=,)",
    // Lookaheads whose body is a Unicode class, so the assertion needs a
    // codepoint check rather than a byte one. On a non-ASCII follower that check
    // is a call, and the greedy loop keeps its backtracking floor in a
    // caller-saved register across it.
    r"a+(?!\S)",
    r"a+(?=\S)",
    r"\w+(?!\p{L})",
    // Lookbehind, both polarities, over classes whose members encode to more
    // than one UTF-8 width. The JIT emits one attempt per candidate total, so
    // these are the shapes where picking the wrong width — or ordering the
    // attempts wrongly, or short-circuiting the negative case on the first
    // failing width — returns wrong spans rather than crashing.
    r"(?<=\s)\w+",
    r"(?<!\s)\w+",
    r"(?<=\p{L})x",
    r"(?<!\p{L})x",
    r"(?<=\s)x",
    r"(?<=\p{L})\w",
    // A class spanning all four UTF-8 widths: 'a', Cyrillic 'а', CJK '中',
    // emoji '😀' — four candidate widths, so four attempts.
    "(?<=[a\u{430}\u{4e2d}\u{1F600}])x",
    "(?<![a\u{430}\u{4e2d}\u{1F600}])x",
    // Two candidate widths (1 for 'a', 2 for 'é') that can succeed against
    // *different* bytes: on "aXb" the 2-byte attempt starts on the 'a', matches
    // it as a 1-byte codepoint and stops one byte short of the assertion
    // position. Only requiring the walk to land exactly on that position
    // rejects it; an implementation that just runs out of steps reports a match
    // the interpreter does not.
    "(?<=[a\u{e9}])b",
    "(?<![a\u{e9}])b",
    // A multi-width class followed by a fixed byte, so the candidate totals
    // (2 and 3) both sit above zero and the wrong one lands off by one.
    "(?<=[a\u{e9}]x)y",
    "(?<![a\u{e9}]x)y",
    // Nullable greedy runs (`X*`). Both step extractors — the interpreter's and
    // the JIT's own — recognise this shape structurally and emit one
    // `GreedyStar`; before that it became an `Alt` that duplicated or dropped
    // whatever followed. The trailing step is what the two models disagree
    // about, so vary it: a lookahead, a literal, a class, an anchor.
    r"\w*(?=ing)",
    r"\w*a",
    r"[a-z]*x",
    r"[a-z]*",
    r"\w*\b",
    r"\w*$",
    r"a*b",
    // The shapes the star recognition must NOT swallow: a genuine alternation
    // whose first branch is a `+` loop. Its loop exits to the plus fragment's
    // end, not to the state the second branch starts at, so the recognition
    // declines it. See `greedy_star_recognition_does_not_swallow_alternation`
    // for the hard-coded expectations that catch a loosened check.
    r"(?:a+|b)",
    r"(?:a+|b)c",
    r"(?:[a-z]+|0)x",
    // Non-greedy stars stay non-greedy: their split prefers the exit and routes
    // it through a `NonGreedyExit` marker, which the recognition refuses.
    r"\w*?a",
    r"\w*?(?=ing)",
    r"a*?b",
    // The tokenizer pattern this engine was built for.
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
];

const INPUTS: &[&str] = &[
    "",
    "a",
    "aa",
    "ab",
    "aab",
    "abb",
    "aabb",
    "abc",
    "aaa",
    "b",
    "c",
    "abcabc",
    "\r\n\r\nx",
    "\r\n ",
    "\n\nabc",
    "  \t x",
    "hello world",
    "running sing",
    "singing in the rain",
    "a1b2",
    "x  ",
    "  ",
    "\n",
    "foo bar baz",
    "'s don't  won't",
    "abc123 def",
    "  \n\n  x",
    "aXbXc",
    "test(ing)",
    "12px 34px",
    "a,b,,c",
    "file.name.ext",
    "\u{4e2d}\u{6587} \u{3042}",
    // A greedy run ending against a multi-byte character, which is what sends
    // the lookahead's codepoint check down its calling path.
    "aa\u{4e2d}",
    "e.aae00 a\u{4e2d}\u{4e2d}\n",
    "a\u{3000}",
    "MiXeD CaSe 42",
    // A lookbehind is checked by walking forwards from `pos - width`, so a wrong
    // width shows up as reading from the middle of the character *immediately
    // before* the assertion position. Every input above puts its multi-byte
    // characters at the end or next to each other, which never exercises that;
    // these put one directly before the ASCII the assertion is anchored on, at
    // each of the four UTF-8 widths.
    "\u{430}x",
    "\u{4e2d}x",
    "\u{1F600}x",
    "a\u{4e2d}x",
    "\u{4e2d}\u{4e2d}x",
    "x\u{4e2d}x\u{4e2d}",
    // Multi-byte whitespace (ideographic space, NBSP) directly before word
    // characters, which is the `\s` lookbehind case: widths 1, 2 and 3 for one
    // class.
    "\u{3000}word",
    "a\u{a0}word b",
    "\u{3000}x \u{a0}x x",
    // 'é' (2 bytes) and 'a' (1 byte) before the assertion point, plus the
    // one-byte-short case where a wrong candidate width can match unrelated
    // bytes and stop early.
    "\u{e9}b",
    "ab",
    "aXb",
    "a\u{e9}b",
    "\u{e9}\u{e9}b",
    "\u{e9}xy",
    "axy",
    "aXxy",
    "1x x \u{4e2d}x",
    // The wrong-width trap for a class that really is compiled as a codepoint
    // class (`\p{L}`, `\s`): the character before the assertion point does NOT
    // satisfy it, but the one before *that* does. A wide candidate starts on the
    // satisfying character, matches it at its own (narrower) length and stops a
    // byte short of the assertion position, so only requiring the walk to land
    // exactly on that position keeps these non-matches non-matching.
    "a.x",
    "ab.x",
    "a x",
    "1.x",
    " .x",
    "\u{4e2d}.x",
    "\u{4e2d} x",
    "\u{1F600}.x",
    // Nullable-star haystacks: nothing to match, everything matching, and the
    // one-branch-or-the-other inputs an alternation misread as a star gets
    // wrong ("aa" has no `b`, so `a*b` finds nothing where `(?:a+|b)` finds the
    // whole run).
    "aaaa",
    "zzzz",
    "aab c",
    "bc",
    "aac",
    "cb",
    "bbb",
    "zzzx",
    "abcx",
    "x",
    "0x 1x",
    "sing",
    "singing ing",
    "ingesting",
    "\u{4e2d}ing",
    "a\u{4e2d}b",
    "\u{4e2d}\u{4e2d}",
];

#[test]
fn tagged_jit_agrees_with_its_interpreter() {
    let mut divergences = Vec::new();
    let mut compared = 0usize;

    for pattern in PATTERNS {
        let (Ok(interpreted), Ok(jitted)) = (
            Regex::new(pattern),
            RegexBuilder::new(pattern).jit(true).build(),
        ) else {
            continue;
        };

        for input in INPUTS {
            let a: Vec<_> = interpreted
                .find_iter(input)
                .map(|m| (m.start(), m.end()))
                .collect();
            let b: Vec<_> = jitted
                .find_iter(input)
                .map(|m| (m.start(), m.end()))
                .collect();
            compared += 1;
            if a != b {
                divergences.push(format!(
                    "  {pattern:?} on {input:?}: interpreter={a:?} jit={b:?} (jit engine {})",
                    jitted.engine_name()
                ));
                continue;
            }

            // Captures too: a span can be right while a group inside it is not.
            let ca = interpreted.captures(input).map(|c| {
                (0..c.len())
                    .map(|i| c.get(i).map(|m| (m.start(), m.end())))
                    .collect::<Vec<_>>()
            });
            let cb = jitted.captures(input).map(|c| {
                (0..c.len())
                    .map(|i| c.get(i).map(|m| (m.start(), m.end())))
                    .collect::<Vec<_>>()
            });
            if ca != cb {
                divergences.push(format!(
                    "  {pattern:?} on {input:?} captures: interpreter={ca:?} jit={cb:?}"
                ));
            }
        }
    }

    assert!(compared > 0, "no pattern/input pair was compared");
    assert!(
        divergences.is_empty(),
        "{} of {compared} comparisons disagree between the tagged-NFA JIT and \
         its interpreter:\n{}",
        divergences.len(),
        divergences.join("\n")
    );
}

/// The generated code itself, compiled directly, agrees with its interpreter.
///
/// Engine selection keeps every program that repeats without bound off the
/// tagged JIT (see `greedy_with_attached_lookahead_stays_on_the_interpreter`),
/// so the sweep above compares such patterns on the interpreter twice. The
/// code generator still accepts them through `jit::compile_tagged_nfa`, and
/// this holds that code to the same answers.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[test]
fn directly_compiled_tagged_jit_agrees_with_its_interpreter() {
    use regexr::nfa::tagged::TaggedNfaEngine;

    let mut divergences = Vec::new();
    let mut compared = 0usize;

    for pattern in PATTERNS {
        let Ok(hir) = regexr::parser::parse(pattern).and_then(|ast| regexr::hir::translate(&ast))
        else {
            continue;
        };
        let Ok(nfa) = regexr::nfa::compile(&hir) else {
            continue;
        };
        let Ok(jitted) = regexr::jit::compile_tagged_nfa(&nfa) else {
            continue;
        };
        let interpreted = TaggedNfaEngine::new(nfa);

        for input in INPUTS {
            let bytes = input.as_bytes();
            compared += 1;
            let a = interpreted.find(bytes);
            let b = jitted.find(bytes);
            if a != b {
                divergences.push(format!(
                    "  {pattern:?} on {input:?}: interpreter={a:?} jit={b:?}"
                ));
                continue;
            }
            let ca = interpreted.captures(bytes);
            let cb = jitted.captures(bytes);
            if ca != cb {
                divergences.push(format!(
                    "  {pattern:?} on {input:?} captures: interpreter={ca:?} jit={cb:?}"
                ));
            }
        }
    }

    assert!(compared > 0, "no pattern/input pair was compared");
    assert!(
        divergences.is_empty(),
        "{} of {compared} comparisons disagree between the directly compiled \
         tagged-NFA JIT and its interpreter:\n{}",
        divergences.len(),
        divergences.join("\n")
    );
}

/// A pattern that can match empty must still end its search.
///
/// The unanchored loop bounds itself on `len - start_pos`, which wraps once
/// `start_pos` passes the end. Only a zero minimum length gets that far, so a
/// nullable pattern with no match anywhere is the one shape that can leave the
/// generated code spinning — and it does so as a hang, which the agreement
/// sweep above reports only as a timeout.
#[test]
fn a_nullable_pattern_that_never_matches_terminates() {
    for pattern in [r"a*(?=b)", r"a*(?!a)b", r"[0-9]*(?=z)", r"\w*(?=ing\b)"] {
        for input in ["", "x", "xxxxxxxx"] {
            let jitted = RegexBuilder::new(pattern).jit(true).build().unwrap();
            assert_eq!(
                jitted.find(input).map(|m| (m.start(), m.end())),
                Regex::new(pattern)
                    .unwrap()
                    .find(input)
                    .map(|m| (m.start(), m.end())),
                "{pattern:?} on {input:?}"
            );
        }
    }
}

/// The shapes that stay deferred must actually stay deferred.
///
/// Without this the guard could be loosened to nothing and the test above would
/// still pass by comparing the interpreter against itself.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[test]
fn generic_backtracking_shapes_stay_on_the_interpreter() {
    // Two quantifiers, or a quantifier plus a lookaround the generic step
    // sequencing has to satisfy by giving characters back.
    for pattern in [
        r"a+(?=b)c+",
        r"a+(?=b)b+(?=c)",
        r"[ab]+(?=c)[de]+",
        r"(?=abc)a+x",
        r"a+b(?=c)",
        r"x(?=y)a+",
        r"a+(?=b)(?=bc)",
    ] {
        let jitted = RegexBuilder::new(pattern).jit(true).build().unwrap();
        assert_eq!(
            jitted.engine_name(),
            "TaggedNfa",
            "{pattern} needs generic backtracking and must not be JIT-compiled"
        );
    }
}

/// A greedy quantifier stays on the interpreter, lookahead or not.
///
/// The generated code retries every start and backtracks every run with no
/// budget on its work, so `a+(?=b)` over a long run of `a` costs one scan of
/// the run per start. The interpreter meters that work and hands a runaway
/// search to the PikeVM, which keeps the search linear.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[test]
fn greedy_with_attached_lookahead_stays_on_the_interpreter() {
    for pattern in [r"\w+(?=ing\b)", r"a+(?=b)", r"[ab]+(?=c)", r"[0-9]+(?=px)"] {
        let jitted = RegexBuilder::new(pattern).jit(true).build().unwrap();
        assert_eq!(
            jitted.engine_name(),
            "TaggedNfa",
            "{pattern} repeats without bound and must not be JIT-compiled"
        );
    }
}

/// An alternation whose first branch is a `+` loop must stay an alternation.
///
/// `X*` is recognised structurally at a split state whose epsilons are exactly
/// `[enter, exit]`; what separates that split from a genuine alternation is the
/// demand that the body loop's own exit is *the same state* as the split's exit.
/// Drop that demand and `(?:a+|b)` compiles as `a*b` — the star swallows the
/// second branch and turns it into a mandatory suffix.
///
/// The expectations here are hard-coded rather than compared against the
/// interpreter on purpose: the recognition is shared code, so loosening it
/// breaks both engines identically and a differential check would still pass.
/// Both engines are asserted, because both walk through the same recogniser.
///
/// It does NOT by itself catch a loosened check, and should not be read as the
/// guard: these patterns carry no assertion, so engine selection sends every one
/// of them to the eager DFA and the tagged extractor is never consulted. Checked
/// by loosening the demand — this test still passed. What bites is
/// `steps::nullable_run_with_assertion_tests::a_genuine_alternation_is_not_recognised_as_a_star`,
/// which inspects the extracted program directly. Kept here as a plain
/// end-to-end assertion that these shapes match correctly whichever engine runs
/// them.
#[test]
fn greedy_star_recognition_does_not_swallow_alternation() {
    /// Pattern, haystack, and the exact spans `find_iter` must yield.
    type Case = (&'static str, &'static str, &'static [(usize, usize)]);

    let cases: &[Case] = &[
        // `a*b` finds nothing in a run of `a`s; `(?:a+|b)` takes the run.
        (r"(?:a+|b)", "aa", &[(0, 2)]),
        (r"(?:a+|b)", "aab", &[(0, 2), (2, 3)]),
        (r"(?:a+|b)", "b", &[(0, 1)]),
        (r"(?:a+|b)", "cb", &[(1, 2)]),
        (r"(?:a+|b)", "bbb", &[(0, 1), (1, 2), (2, 3)]),
        // With a suffix, the misreading `a*bc` loses the `a+` branch entirely.
        (r"(?:a+|b)c", "aac", &[(0, 3)]),
        (r"(?:a+|b)c", "ac", &[(0, 2)]),
        (r"(?:a+|b)c", "bc", &[(0, 2)]),
        (r"(?:a+|b)c", "c", &[]),
        (r"(?:a+|b)c", "aa", &[]),
        // A class branch rather than a literal one, so the misreading is
        // `[a-z]*0x` and the letters-only input stops matching.
        (r"(?:[a-z]+|0)x", "abx", &[(0, 3)]),
        (r"(?:[a-z]+|0)x", "0x", &[(0, 2)]),
        (r"(?:[a-z]+|0)x", "x", &[]),
    ];

    for &(pattern, input, expected) in cases {
        for jit in [false, true] {
            let re = RegexBuilder::new(pattern).jit(jit).build().unwrap();
            let got: Vec<_> = re.find_iter(input).map(|m| (m.start(), m.end())).collect();
            assert_eq!(
                got,
                expected,
                "{pattern:?} on {input:?} with jit={jit} (engine {})",
                re.engine_name()
            );
        }
    }
}

/// A nullable greedy run carrying a lookahead stays on the metered
/// interpreter, for the reason given in
/// `greedy_with_attached_lookahead_stays_on_the_interpreter`; without one it
/// still reaches a JIT engine.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[test]
fn nullable_greedy_run_engine_selection() {
    let jitted = RegexBuilder::new(r"\w*(?=ing)").jit(true).build().unwrap();
    assert_eq!(
        jitted.engine_name(),
        "TaggedNfa",
        r"\w*(?=ing) repeats without bound and must not reach the tagged JIT"
    );

    // Without an assertion to satisfy, a recognised star is free to take a
    // cheaper engine still — `\w*a` and `[a-z]*x` select `JitShiftOr`. Assert
    // only that they reach *some* JIT engine, so this does not fail the day
    // selection legitimately improves again.
    for pattern in [r"\w*a", r"[a-z]*x"] {
        let name = RegexBuilder::new(pattern)
            .jit(true)
            .build()
            .unwrap()
            .engine_name()
            .to_string();
        assert!(
            name.contains("Jit"),
            "{pattern} is a recognised greedy star and should be JIT-compiled, got {name}"
        );
    }
}

/// A lookaround whose inner pattern ends in a zero-width assertion must reach
/// the JIT.
///
/// The inner extractors walk to the inner NFA's match state, and an assertion
/// compiled onto that state has to be read before they stop. If one is missed,
/// the assertion tally no longer matches the NFA and the JIT correctly refuses —
/// so these patterns silently drop to a slower engine rather than returning
/// wrong answers. Assert the engine to catch that.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[test]
fn lookaround_with_trailing_assertion_is_jit_compiled() {
    // Word boundaries and the start anchor are emitted by both backends. The
    // end/line anchors inside a *lookbehind* are x86-64 only: the AArch64
    // lookbehind refuses them, because emitting them there produced code that
    // faulted at run time (`(?<=a\Z)x?`). Refusing keeps those patterns on the
    // interpreter, which is correct — so this asserts the coverage each backend
    // actually has rather than the coverage we would like it to have.
    let mut patterns = vec![
        r"(?<=a\b)x",
        r"(?<=a\B)x",
        r"(?<!a\b)x",
        r"(?<=^a)x",
        r"x(?=a\b)",
        r"x(?=a\B)",
        r"x(?!a\b)",
        r"x(?=a$)",
        r"\w(?=\w\b)",
    ];
    if cfg!(target_arch = "x86_64") {
        patterns.extend([r"(?<=a$)b", r"(?<!a$)x"]);
    }
    for pattern in patterns {
        let jitted = RegexBuilder::new(pattern).jit(true).build().unwrap();
        assert_eq!(
            jitted.engine_name(),
            "TaggedNfaJit",
            "{pattern} is expressible as steps and should be JIT-compiled"
        );
    }
}
