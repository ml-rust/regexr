//! Match results checked against the `regex` crate.
//!
//! Alternation branches that share a first class, differ in length, or are
//! prefixes of one another resolve leftmost-first. So does greedy repetition,
//! which prefers one more iteration even when that ends the match early.
//! Assertions hold at the exact position they sit. `find`, `is_match`,
//! `captures` and `find_iter` agree with each other and with `regex`.
//!
//! Haystacks hold ASCII only and no `\n`, where regexr's `$` and `\b` mean the
//! same as the `regex` crate's.

use regexr::Regex;
#[cfg(feature = "jit")]
use regexr::RegexBuilder;

type Span = Option<(usize, usize)>;

/// Everything one build reports for one pattern on one haystack.
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    find: Span,
    is_match: bool,
    groups: Option<Vec<Span>>,
    iter: Vec<(usize, usize)>,
}

fn expected(re: &regex::Regex, hay: &str) -> Outcome {
    Outcome {
        find: re.find(hay).map(|m| (m.start(), m.end())),
        is_match: re.is_match(hay),
        groups: re.captures(hay).map(|caps| {
            caps.iter()
                .map(|g| g.map(|m| (m.start(), m.end())))
                .collect()
        }),
        iter: re.find_iter(hay).map(|m| (m.start(), m.end())).collect(),
    }
}

fn actual(re: &Regex, hay: &str) -> Outcome {
    Outcome {
        find: re.find(hay).map(|m| (m.start(), m.end())),
        is_match: re.is_match(hay),
        groups: re.captures(hay).map(|caps| {
            (0..caps.len())
                .map(|i| caps.get(i).map(|m| (m.start(), m.end())))
                .collect()
        }),
        iter: re.find_iter(hay).map(|m| (m.start(), m.end())).collect(),
    }
}

/// Every regexr build under test: the default one, plus the JIT one when the
/// `jit` feature is on.
fn builds(pattern: &str) -> Vec<(&'static str, Regex)> {
    let default =
        Regex::new(pattern).unwrap_or_else(|e| panic!("{pattern:?} failed to compile: {e}"));
    #[cfg(feature = "jit")]
    let jit = RegexBuilder::new(pattern)
        .jit(true)
        .build()
        .unwrap_or_else(|e| panic!("{pattern:?} failed to compile with jit: {e}"));
    vec![
        ("default", default),
        #[cfg(feature = "jit")]
        ("jit", jit),
    ]
}

/// Lists every disagreement between regexr and `regex` for `pattern`.
fn mismatches(pattern: &str, haystacks: &[&str]) -> Vec<String> {
    let reference = regex::Regex::new(pattern)
        .unwrap_or_else(|e| panic!("{pattern:?} rejected by the regex crate: {e}"));
    let mut out = Vec::new();
    for (build, re) in builds(pattern) {
        for &hay in haystacks {
            let want = expected(&reference, hay);
            let got = actual(&re, hay);
            if got != want {
                out.push(format!(
                    "{build} engine={} pattern={pattern:?} haystack={hay:?}\n  got:  {got:?}\n  want: {want:?}",
                    re.engine_name()
                ));
            }
        }
    }
    out
}

fn assert_agrees(pattern: &str, haystacks: &[&str]) {
    let diffs = mismatches(pattern, haystacks);
    assert!(diffs.is_empty(), "{}", diffs.join("\n"));
}

fn span(re: &Regex, hay: &str) -> Span {
    re.find(hay).map(|m| (m.start(), m.end()))
}

#[test]
fn class_or_longer_class_sequence_takes_first_branch() {
    for pattern in [
        "[:sS]|[sS][eE][cC]|[sS][aA][aA][tT]",
        "[:sS]|[sS][eE][cC]",
        "[sS]|[sS][eE][cC]",
        "s|sec",
        "[:s]|sec",
        "[:sS]|saat",
    ] {
        for (build, re) in builds(pattern) {
            assert_eq!(span(&re, "10 sec"), Some((3, 4)), "{build} {pattern:?}");
            assert!(re.is_match("10 sec"), "{build} {pattern:?}");
        }
        assert_agrees(pattern, &["10 sec", "SEC", "saat", "x", ""]);
    }
}

#[test]
fn alternation_of_classes_is_not_a_single_class() {
    for pattern in [
        "[a][b]|[c]",
        "[ab]|[c][d]",
        "[a]|[b]|[c][d]",
        "[a]|x",
        "(?i)s|sec",
        "(?i:[s])|[e][c]",
    ] {
        assert_agrees(pattern, &["cd", "abcd", "xa", "c", "SEC", "ec", ""]);
    }
}

#[test]
fn single_class_forms_still_match() {
    for pattern in ["[sc]", "(?:[sc])", "(?i:[S])", "([sc])", "[^s]", "[é-ü]"] {
        assert_agrees(pattern, &["10 sec", "SSS", "", "é", "aü"]);
    }
}

#[test]
fn greedy_repeat_priority_can_end_the_match_early() {
    for (pattern, hay, want) in [
        ("a?(?:ab)?", "ab", (0, 1)),
        ("a*(?:ab)*", "ab", (0, 1)),
        ("aa?(?:ab)?", "aab", (0, 2)),
        ("a?(?:abcd|b)", "abcd", (0, 2)),
        ("(?:.{1,2}[^s])*", "Ea11ECS", (0, 6)),
    ] {
        for (build, re) in builds(pattern) {
            assert_eq!(span(&re, hay), Some(want), "{build} {pattern:?} on {hay:?}");
        }
        assert_agrees(pattern, &[hay, "", "abab", "aabab"]);
    }
}

#[test]
fn a_match_after_an_attempt_that_reaches_the_end() {
    for (pattern, hay, want) in [
        ("a(?:ab)?b", "aab", (1, 3)),
        ("a(?:ab)*b", "aab", (1, 3)),
        ("aa(?:ab)?b", "aaab", (1, 4)),
    ] {
        for (build, re) in builds(pattern) {
            assert_eq!(span(&re, hay), Some(want), "{build} {pattern:?} on {hay:?}");
            assert!(re.is_match(hay), "{build} {pattern:?} on {hay:?}");
        }
        assert_agrees(pattern, &[hay, "aaaab", "aabaab"]);
    }
}

#[test]
fn attempts_that_reach_the_end_do_not_rescan_per_start() {
    // Every start runs to the end of the text without a match. Resuming one
    // start at a time would scan the text once per start.
    let pattern = r"xy[\x00-y]*z";
    let hay = "xy".repeat(100_000);
    for (build, re) in builds(pattern) {
        assert_eq!(span(&re, &hay), None, "{build}");
    }
    let hay = format!("{hay}z");
    for (build, re) in builds(pattern) {
        assert_eq!(span(&re, &hay), Some((0, hay.len())), "{build}");
    }
}

#[test]
fn assertions_hold_where_they_sit() {
    for (pattern, hay, want) in [
        ("a$a", "aa", None),
        ("a$a?", "aa", Some((1, 2))),
        ("a^", "aa", None),
        (r"a\ba?", "a", Some((0, 1))),
        (r"[ab]?\bb", "ab b", Some((3, 4))),
        (r"a\b$", "ba a", Some((3, 4))),
        (r"a*\B(?:a|bb)", "aa", Some((0, 2))),
    ] {
        for (build, re) in builds(pattern) {
            assert_eq!(span(&re, hay), want, "{build} {pattern:?} on {hay:?}");
        }
        assert_agrees(pattern, &[hay, "", "ab", "a b a", "aab bba"]);
    }
}

/// Deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }
}

const ATOMS: &[&str] = &[
    "s",
    "S",
    "e",
    "c",
    ":",
    "a",
    "[sS]",
    "[:sS]",
    "[eE]",
    "[cC]",
    "[sec]",
    "[^s]",
    "[a:]",
    ".",
    "(?:se)",
    "(?:sec|s)",
];

const ASSERTIONS: &[&str] = &["^", "$", r"\b", r"\B"];

const SUFFIXES: &[&str] = &["", "", "", "?", "*", "+", "{1,2}", "{0,2}", "??", "+?"];

fn branch(rng: &mut Rng, with_assertions: bool) -> String {
    let len = 1 + rng.below(4);
    let mut out = String::new();
    for _ in 0..len {
        if with_assertions && rng.below(4) == 0 {
            out.push_str(rng.pick(ASSERTIONS));
            continue;
        }
        out.push_str(rng.pick(ATOMS));
        if rng.below(3) == 0 {
            out.push_str(rng.pick(SUFFIXES));
        }
    }
    out
}

fn alternation(rng: &mut Rng, with_assertions: bool) -> String {
    let count = 1 + rng.below(4);
    (0..count)
        .map(|_| branch(rng, with_assertions))
        .collect::<Vec<_>>()
        .join("|")
}

fn pattern(rng: &mut Rng, with_assertions: bool) -> String {
    let alt = alternation(rng, with_assertions);
    match rng.below(6) {
        0 => format!("({alt})"),
        1 => format!("(?:{alt}){}", rng.pick(ATOMS)),
        2 => format!("{}({alt})", rng.pick(ATOMS)),
        3 => format!("(?:{alt}){}", rng.pick(&["?", "*", "+", "{2}"])),
        _ => alt,
    }
}

fn haystack(rng: &mut Rng) -> String {
    const CHARS: &[char] = &['s', 'S', 'e', 'E', 'c', 'C', ':', 'a', ' ', '1'];
    let len = rng.below(9);
    (0..len).map(|_| CHARS[rng.below(CHARS.len())]).collect()
}

fn sweep(seed: u64, count: usize, with_assertions: bool) {
    let mut rng = Rng(seed);
    let mut diffs = Vec::new();
    for _ in 0..count {
        let pattern = pattern(&mut rng, with_assertions);
        let mut owned: Vec<String> = (0..8).map(|_| haystack(&mut rng)).collect();
        owned.push("10 sec".to_string());
        owned.push(String::new());
        let haystacks: Vec<&str> = owned.iter().map(String::as_str).collect();
        diffs.extend(mismatches(&pattern, &haystacks));
    }
    let shown = diffs.len().min(40);
    assert!(
        diffs.is_empty(),
        "{} disagreements with the regex crate, first {shown}:\n{}",
        diffs.len(),
        diffs[..shown].join("\n")
    );
}

#[test]
fn generated_alternations_match_regex_crate() {
    sweep(0x9E37_79B9_7F4A_7C15, 4000, false);
}

#[test]
fn generated_assertions_match_regex_crate() {
    sweep(0xD1B5_4A32_D192_ED03, 3000, true);
}
