//! Capture extraction must honour the same execution bound as `find`.
//!
//! `find` runs on the automaton engines and is linear in the input. Capture
//! extraction owes the caller the same guarantee: every search terminates, in
//! time and memory bounded by the input, for every pattern the parser accepts.
//!
//! Three bounds are checked here:
//!
//! 1. A repetition whose body matches the empty string runs it once and then
//!    leaves the loop. Without that, `(a*)*` re-enters its body at a position it
//!    can never advance past and the search never returns.
//! 2. Alternative-heavy repetition does not cost exponential time. `(a+)+c` over
//!    a run of `a` has exponentially many ways to split the run, and an engine
//!    that explores them one at a time will not finish.
//! 3. The choice-point stack stays inside the memory it owns. A search that
//!    needs more choice points than the stack was sized for must still answer
//!    correctly, not write past the end of its frame.
//!
//! Expected spans are the leftmost-first, greedy semantics of the executable
//! spec in `regexr::reference`, cross-checked against the `regex` crate.
//!
//! Every case runs in a child process. The failure modes under test are an
//! unbounded allocation (abort) and a choice-point stack that runs past its
//! frame (SIGSEGV); both take down the whole test binary, so isolating each
//! case keeps one broken bound from hiding the state of the others.

use std::env;
use std::process::Command;
use std::time::{Duration, Instant};

use regexr::{Regex, RegexBuilder};

/// Set by the parent on the child it spawns, naming the single case to run.
const CASE_VAR: &str = "REGEXR_BOUNDED_EXECUTION_CASE";

/// How long a bounded search is allowed to take. Correct engines answer every
/// case below in microseconds; this only has to be short enough that an
/// unbounded one is reported as a failure rather than left running.
const DEADLINE: Duration = Duration::from_secs(20);

/// Address-space ceiling for the child, so an unbounded search aborts on a
/// failed allocation instead of pushing the machine into swap.
const CHILD_ADDRESS_SPACE_KIB: u64 = 4_000_000;

/// Runs `body` in a child process dedicated to `case`, and fails if that child
/// crashes, is killed, or has not finished within [`DEADLINE`].
///
/// `case` must be the name of the calling test, because that is the filter the
/// child is invoked with.
fn bounded(case: &str, body: impl FnOnce()) {
    bounded_within(case, DEADLINE, body)
}

/// [`bounded`] with an explicit deadline, for cases whose correct behaviour is
/// still measured in seconds rather than microseconds.
fn bounded_within(case: &str, deadline: Duration, body: impl FnOnce()) {
    if env::var(CASE_VAR).as_deref() == Ok(case) {
        body();
        return;
    }

    let exe = env::current_exe().expect("test binary path");
    let mut command = if cfg!(unix) {
        let mut c = Command::new("sh");
        c.arg("-c")
            .arg(format!(
                "ulimit -v {CHILD_ADDRESS_SPACE_KIB} 2>/dev/null; exec \"$@\""
            ))
            .arg("sh")
            .arg(&exe);
        c
    } else {
        Command::new(&exe)
    };
    let mut child = command
        .arg(case)
        .arg("--exact")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CASE_VAR, case)
        .spawn()
        .expect("spawn isolated case");

    let started = Instant::now();
    loop {
        match child.try_wait().expect("poll isolated case") {
            Some(status) if status.success() => return,
            Some(status) => panic!("{case}: isolated run failed ({status})"),
            None if started.elapsed() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("{case}: search did not terminate within {deadline:?}");
            }
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

/// The two builds under test, labelled for failure messages.
fn builds(pattern: &str) -> Vec<(&'static str, Regex)> {
    vec![
        (
            "jit",
            RegexBuilder::new(pattern)
                .jit(true)
                .build()
                .expect("pattern should compile"),
        ),
        (
            "interp",
            RegexBuilder::new(pattern)
                .jit(false)
                .build()
                .expect("pattern should compile"),
        ),
    ]
}

type Spans = Vec<Option<(usize, usize)>>;

/// Spans of every group of the first match, group 0 first.
fn capture_spans(re: &Regex, text: &str) -> Option<Spans> {
    re.captures(text).map(|caps| {
        (0..caps.len())
            .map(|i| caps.get(i).map(|m| (m.start(), m.end())))
            .collect()
    })
}

/// Spans of every group of every match produced by iteration.
fn iterated_spans(re: &Regex, text: &str) -> Vec<Spans> {
    re.captures_iter(text)
        .map(|caps| {
            (0..caps.len())
                .map(|i| caps.get(i).map(|m| (m.start(), m.end())))
                .collect()
        })
        .collect()
}

fn span(start: usize, end: usize) -> Option<(usize, usize)> {
    Some((start, end))
}

// =============================================================================
// A repetition whose body matches empty runs it once, then leaves the loop
// =============================================================================
// The loop takes a zero-width iteration while it still owes iterations — the
// first one, plus any `min` demands — and refuses it after that. So `(a*)*` on
// "b" reports group 1 as the empty span at 0, while on "a" it reports 0..1 and
// not the empty span that would follow. Every expectation below was cross-checked
// against `regexr::reference` and the `regex` crate.

#[test]
fn nested_star_over_nullable_body_terminates() {
    bounded("nested_star_over_nullable_body_terminates", || {
        for (label, re) in builds(r"(a*)*") {
            assert_eq!(
                capture_spans(&re, "a"),
                Some(vec![span(0, 1), span(0, 1)]),
                "{label}"
            );
            assert_eq!(
                capture_spans(&re, "aaa"),
                Some(vec![span(0, 3), span(0, 3)]),
                "{label}"
            );
        }
    });
}

#[test]
fn nested_star_over_nullable_body_matches_empty_at_start() {
    bounded(
        "nested_star_over_nullable_body_matches_empty_at_start",
        || {
            for (label, re) in builds(r"(a*)*") {
                assert_eq!(
                    capture_spans(&re, "b"),
                    Some(vec![span(0, 0), span(0, 0)]),
                    "{label}"
                );
            }
        },
    );
}

#[test]
fn star_over_optional_body_terminates() {
    bounded("star_over_optional_body_terminates", || {
        for (label, re) in builds(r"(a?)*") {
            assert_eq!(
                capture_spans(&re, "a"),
                Some(vec![span(0, 1), span(0, 1)]),
                "{label}"
            );
        }
    });
}

#[test]
fn plus_over_empty_group_terminates() {
    bounded("plus_over_empty_group_terminates", || {
        for (label, re) in builds(r"()+") {
            assert_eq!(
                capture_spans(&re, "a"),
                Some(vec![span(0, 0), span(0, 0)]),
                "{label}"
            );
        }
    });
}

#[test]
fn plus_over_nullable_body_terminates() {
    bounded("plus_over_nullable_body_terminates", || {
        for (label, re) in builds(r"(a*)+") {
            assert_eq!(
                capture_spans(&re, "a"),
                Some(vec![span(0, 1), span(0, 1)]),
                "{label}"
            );
        }
    });
}

#[test]
fn star_over_bounded_nullable_body_terminates() {
    bounded("star_over_bounded_nullable_body_terminates", || {
        for (label, re) in builds(r"(a{0,2})*") {
            assert_eq!(
                capture_spans(&re, "aaa"),
                Some(vec![span(0, 3), span(2, 3)]),
                "{label}"
            );
        }
    });
}

#[test]
fn nested_capture_inside_nullable_loop_terminates() {
    bounded("nested_capture_inside_nullable_loop_terminates", || {
        for (label, re) in builds(r"((a)*)*") {
            assert_eq!(
                capture_spans(&re, "a"),
                Some(vec![span(0, 1), span(0, 1), span(0, 1)]),
                "{label}"
            );
        }
    });
}

#[test]
fn nullable_loop_followed_by_literal_terminates() {
    bounded("nullable_loop_followed_by_literal_terminates", || {
        for (label, re) in builds(r"(a*)*b") {
            assert_eq!(
                capture_spans(&re, "ab"),
                Some(vec![span(0, 2), span(0, 1)]),
                "{label}"
            );
            assert_eq!(capture_spans(&re, "a"), None, "{label}");
        }
    });
}

#[test]
fn iteration_over_nullable_loop_terminates() {
    bounded("iteration_over_nullable_loop_terminates", || {
        for (label, re) in builds(r"(a*)*") {
            // The empty match at 1 is where the non-empty match ended, so
            // iteration drops it; the one at 2 follows an empty match and stays.
            assert_eq!(
                iterated_spans(&re, "ab"),
                vec![vec![span(0, 1), span(0, 1)], vec![span(2, 2), span(2, 2)],],
                "{label}"
            );
        }
    });
}

#[test]
fn iteration_over_empty_group_loop_terminates() {
    bounded("iteration_over_empty_group_loop_terminates", || {
        for (label, re) in builds(r"()+") {
            assert_eq!(
                iterated_spans(&re, "ab"),
                vec![
                    vec![span(0, 0), span(0, 0)],
                    vec![span(1, 1), span(1, 1)],
                    vec![span(2, 2), span(2, 2)],
                ],
                "{label}"
            );
        }
    });
}

/// The capture path must agree with the automaton path, which already reports
/// these spans correctly — the two must not disagree about the same pattern.
#[test]
fn nullable_loop_captures_agree_with_find() {
    bounded("nullable_loop_captures_agree_with_find", || {
        for pattern in [r"(a*)*", r"(a?)*", r"()+", r"(a*)+"] {
            for (label, re) in builds(pattern) {
                for text in ["", "a", "aaa", "ab", "b"] {
                    let found = re.find(text).map(|m| (m.start(), m.end()));
                    let captured = capture_spans(&re, text).and_then(|s| s[0]);
                    assert_eq!(found, captured, "{label} {pattern:?} {text:?}");
                }
            }
        }
    });
}

// =============================================================================
// Repetition with many ways to split the input costs bounded time
// =============================================================================

#[test]
fn nested_plus_does_not_backtrack_exponentially() {
    bounded("nested_plus_does_not_backtrack_exponentially", || {
        let text = "a".repeat(30);
        for (label, re) in builds(r"(a+)+c") {
            assert_eq!(capture_spans(&re, &text), None, "{label}");
        }
    });
}

#[test]
fn nested_star_does_not_backtrack_exponentially() {
    bounded("nested_star_does_not_backtrack_exponentially", || {
        let text = "a".repeat(30);
        for (label, re) in builds(r"(a*)*c") {
            assert_eq!(capture_spans(&re, &text), None, "{label}");
        }
    });
}

#[test]
fn nested_plus_reports_the_match_it_finds() {
    bounded("nested_plus_reports_the_match_it_finds", || {
        let text = format!("{}c", "a".repeat(30));
        for (label, re) in builds(r"(a+)+c") {
            assert_eq!(
                capture_spans(&re, &text),
                Some(vec![span(0, 31), span(0, 30)]),
                "{label}"
            );
        }
    });
}

// =============================================================================
// The choice-point stack stays inside the memory it owns
// =============================================================================
// A greedy repetition records one choice point per iteration. The JIT engines
// hold those in a fixed frame, so a run longer than that frame is the case that
// distinguishes "grew the stack" from "wrote past the end of it".

#[test]
fn long_greedy_run_before_a_literal_stays_in_bounds() {
    bounded("long_greedy_run_before_a_literal_stays_in_bounds", || {
        let matching = format!("{}@", "a".repeat(300));
        let non_matching = "a".repeat(300);
        for (label, re) in builds(r"([a-z]+)@") {
            assert_eq!(
                capture_spans(&re, &matching),
                Some(vec![span(0, 301), span(0, 300)]),
                "{label}"
            );
            assert_eq!(capture_spans(&re, &non_matching), None, "{label}");
        }
    });
}

#[test]
fn very_long_greedy_run_before_a_literal_stays_in_bounds() {
    bounded(
        "very_long_greedy_run_before_a_literal_stays_in_bounds",
        || {
            let text = format!("{}!", "a".repeat(4000));
            for (label, re) in builds(r"(\w+)!") {
                assert_eq!(
                    capture_spans(&re, &text),
                    Some(vec![span(0, 4001), span(0, 4000)]),
                    "{label}"
                );
            }
        },
    );
}

#[test]
fn long_greedy_run_captures_agree_with_find() {
    bounded("long_greedy_run_captures_agree_with_find", || {
        let text = format!("{}@", "a".repeat(1000));
        for (label, re) in builds(r"([a-z]+)@") {
            let found = re.find(&text).map(|m| (m.start(), m.end()));
            let captured = capture_spans(&re, &text).and_then(|s| s[0]);
            assert_eq!(found, captured, "{label}");
        }
    });
}

// =============================================================================
// A search over a long non-matching input stays linear in its length
// =============================================================================
// Every engine here searches unanchored input by trying start positions. That is
// right while a failed attempt gives up near where it began, and quadratic when
// it does not: `(a+)+$` over a run of `a` consumes the whole run from every
// start and rejects it at `$`, so the cost is one full scan per byte. The
// engines answer that with a single pass that covers every start at once.
//
// The input below is long enough that the difference is not a matter of
// constants: linear finishes in milliseconds, and one scan per byte would need
// far longer than [`DEADLINE`] allows.

/// Input length for the scaling cases.
const LONG_INPUT: usize = 20_000;

/// Deadline for the scaling cases.
///
/// Unlike the termination cases, these do real work even when correct — tens of
/// thousands of bytes through every engine, in a debug build. The deadline is
/// not a performance target and is deliberately far above what linear costs on
/// any machine, emulated ones included; it only has to sit below what one scan
/// per byte would take at [`LONG_INPUT`], which is minutes.
const SCALING_DEADLINE: Duration = Duration::from_secs(120);

/// Patterns whose failed attempts each consume the whole run before rejecting
/// it, spread across the engines that search by start position: Shift-Or, the
/// lazy DFA and its JIT, and the PikeVM.
const LONG_RUN_NON_MATCHING: &[&str] = &[
    r"(a|a)+$",
    r"(?:a|a)+$",
    r"(a|aa)+$",
    r"(a+)+$",
    r"(a|b)+$",
    r"(x+x+)+y",
    r"([a-zA-Z]+)*b$",
];

#[test]
fn long_run_rejection_stays_linear() {
    bounded_within("long_run_rejection_stays_linear", SCALING_DEADLINE, || {
        let text = format!("{}!", "a".repeat(LONG_INPUT));
        for pattern in LONG_RUN_NON_MATCHING {
            for (label, re) in builds(pattern) {
                assert!(!re.is_match(&text), "{label} {pattern}");
                assert_eq!(
                    re.find(&text).map(|m| (m.start(), m.end())),
                    None,
                    "{label} {pattern}"
                );
                assert_eq!(capture_spans(&re, &text), None, "{label} {pattern}");
            }
        }
    });
}

#[test]
fn long_run_iteration_stays_linear() {
    bounded_within("long_run_iteration_stays_linear", SCALING_DEADLINE, || {
        let text = format!("{}!", "a".repeat(LONG_INPUT));
        for pattern in LONG_RUN_NON_MATCHING {
            for (label, re) in builds(pattern) {
                assert_eq!(re.find_iter(&text).count(), 0, "{label} {pattern}");
                assert_eq!(iterated_spans(&re, &text).len(), 0, "{label} {pattern}");
            }
        }
    });
}

// =============================================================================
// EagerDfa's unanchored start-position loop does not go quadratic on
// word-boundary patterns
// =============================================================================
// `EagerDfa::find_from` (the non-JIT engine LazyDfa promotes simple patterns
// to) tries every start position with no single pass covering all of them at
// once, unlike LazyDfa. A word-boundary pattern whose failed attempts each
// scan to the end of the input turns that into one full scan per start —
// quadratic. `\b[a-y ]+\d` over a run of "bb " is exactly that shape: no
// digit ever appears, so every attempt consumes the rest of the run before
// dying.
//
// This pattern has no anchors and no large Unicode class, so engine selection
// (`EngineType::LazyDfa` branch in `compile_hir`) sends the interpreted build
// to `EagerDfa`, not `LazyDfa` — only the "interp" build is exercised here,
// unlike `LONG_RUN_NON_MATCHING` above (the JIT build takes a different
// engine for this shape).
//
// The length is picked so the pre-fix quadratic cost — measured at roughly
// 60/248/944 ms for 10/20/40 KB inputs of this same shape — lands far past
// [`SCALING_DEADLINE`], while the fixed, metered cost stays linear and well
// within it.
const WORD_BOUNDARY_LONG_RUN_REPEATS: usize = 200_000;

#[test]
fn word_boundary_long_run_rejection_stays_linear() {
    bounded_within(
        "word_boundary_long_run_rejection_stays_linear",
        SCALING_DEADLINE,
        || {
            let text = "bb ".repeat(WORD_BOUNDARY_LONG_RUN_REPEATS);
            let re = RegexBuilder::new(r"\b[a-y ]+\d")
                .jit(false)
                .build()
                .expect("pattern should compile");
            assert!(!re.is_match(&text));
            assert_eq!(re.find(&text).map(|m| (m.start(), m.end())), None);
        },
    );
}

// =============================================================================
// Every engine's search grows linearly with its input
// =============================================================================
// The engines that try one start position at a time go quadratic when every
// attempt walks far before failing: `(?s)a.*b` over a run of `a` scans to the
// end from every start. Each engine meters those attempts and hands an
// expensive search to a linear-time one. The cases below time each search at
// `n` and `4n` bytes: linear work grows about 4x, one scan per start about 16x.
//
// The engines expose no step counter, so the measurement is the fastest of
// several runs, which drops a run the scheduler preempted. A pair of searches
// that both finish in under `SCALING_FLOOR` is too fast for a ratio to mean
// anything and passes; a quadratic search at `4n` is far above it.

/// Input length `n` for the growth checks; each search also runs at `4n`.
const GROWTH_INPUT: usize = 20_000;

/// Largest accepted ratio of the `4n` time to the `n` time. Linear work gives
/// 4 and one scan per start 16; this sits between them with room for noise.
const MAX_GROWTH: f64 = 8.0;

/// Below this, both timings are too small for their ratio to be measured.
const SCALING_FLOOR: Duration = Duration::from_millis(2);

/// The fastest of several runs of `search`.
fn fastest(search: impl Fn()) -> Duration {
    (0..5)
        .map(|_| {
            let started = Instant::now();
            search();
            started.elapsed()
        })
        .min()
        .unwrap_or_default()
}

/// A search whose haystack is `head`, then `unit` repeated `n` times, then
/// `tail`.
struct GrowthCase {
    pattern: &'static str,
    head: &'static str,
    unit: &'static str,
    tail: &'static str,
    /// The engine each build must select, so the case keeps exercising the
    /// engine it was written for. `None` skips the check for that build.
    interp_engine: Option<&'static str>,
    jit_engine: Option<&'static str>,
}

impl GrowthCase {
    const fn new(pattern: &'static str, unit: &'static str, tail: &'static str) -> Self {
        Self {
            pattern,
            head: "",
            unit,
            tail,
            interp_engine: None,
            jit_engine: None,
        }
    }

    const fn head(mut self, head: &'static str) -> Self {
        self.head = head;
        self
    }

    const fn engines(mut self, interp: &'static str, jit: &'static str) -> Self {
        self.interp_engine = Some(interp);
        self.jit_engine = Some(jit);
        self
    }
}

/// One of the searches a growth case times, named for failure messages.
type TimedSearch<'a> = (&'static str, &'a dyn Fn(&str));

/// Asserts that `find`, `find_iter`, `is_match` and `captures` on every build
/// of each case grow linearly between `n` and `4n`.
fn assert_linear_growth(cases: &[GrowthCase]) {
    let mut failures = Vec::new();
    for case in cases {
        let small = format!(
            "{}{}{}",
            case.head,
            case.unit.repeat(GROWTH_INPUT),
            case.tail
        );
        let large = format!(
            "{}{}{}",
            case.head,
            case.unit.repeat(4 * GROWTH_INPUT),
            case.tail
        );
        for (label, re) in builds(case.pattern) {
            let expected = if label == "jit" && cfg!(feature = "jit") {
                case.jit_engine
            } else {
                case.interp_engine
            };
            if let Some(engine) = expected {
                assert_eq!(re.engine_name(), engine, "{label} {:?}", case.pattern);
            }
            let searches: [TimedSearch; 4] = [
                ("find", &|text| {
                    std::hint::black_box(re.find(text));
                }),
                ("find_iter", &|text| {
                    std::hint::black_box(re.find_iter(text).count());
                }),
                ("is_match", &|text| {
                    std::hint::black_box(re.is_match(text));
                }),
                ("captures", &|text| {
                    std::hint::black_box(re.captures(text).is_some());
                }),
            ];
            for (op, search) in searches {
                let at_n = fastest(|| search(&small));
                let at_4n = fastest(|| search(&large));
                let growth = at_4n.as_secs_f64() / at_n.as_secs_f64().max(1e-9);
                if at_4n >= SCALING_FLOOR && growth > MAX_GROWTH {
                    failures.push(format!(
                        "{label} {} {:?} {op}: {at_n:?} at n, {at_4n:?} at 4n ({growth:.1}x)",
                        re.engine_name(),
                        case.pattern
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "superlinear growth:\n{}",
        failures.join("\n")
    );
}

/// The eager DFA: its start-by-start loop over patterns whose attempts run to
/// the end of a run of `a`.
#[test]
fn eager_dfa_search_grows_linearly() {
    bounded_within("eager_dfa_search_grows_linearly", SCALING_DEADLINE, || {
        assert_linear_growth(&[
            GrowthCase::new(r"(?s)a.*b", "a", "").engines("EagerDfa", "EagerDfa"),
            GrowthCase::new(r"a.*b", "a", "").engines("EagerDfa", "EagerDfa"),
            GrowthCase::new(r"[^b]*b", "a", "").engines("EagerDfa", "EagerDfa"),
            GrowthCase::new(r"(?is)a.*b", "a", "").engines("EagerDfa", "EagerDfa"),
            GrowthCase::new(r"(?i)a.*b", "a", "").engines("EagerDfa", "EagerDfa"),
            GrowthCase::new(r"(?i)[^b]*b", "a", "").engines("EagerDfa", "EagerDfa"),
            // A `b` ahead of the run defeats the required-literal rejection,
            // so the search itself has to prove there is no match.
            GrowthCase::new(r"a.*b", "a", "")
                .head("b")
                .engines("EagerDfa", "EagerDfa"),
            GrowthCase::new(r"(?s)a.*b", "a", "")
                .head("b")
                .engines("EagerDfa", "EagerDfa"),
            GrowthCase::new(r"\w+\s", "a", "").engines("EagerDfa", "EagerDfa"),
            GrowthCase::new(r"\ba[a-z]*\d", "a", "").engines("EagerDfa", "EagerDfa"),
        ]);
    });
}

/// The PikeVM: one pass by construction, checked on the alternation shapes
/// engine selection sends it.
#[test]
fn pikevm_search_grows_linearly() {
    bounded_within("pikevm_search_grows_linearly", SCALING_DEADLINE, || {
        assert_linear_growth(&[
            GrowthCase::new(r"(?:a|aa)*c", "a", "").engines("PikeVm", "PikeVm"),
            GrowthCase::new(r"(?i)(?:a|aa)*c", "a", "").engines("PikeVm", "PikeVm"),
        ]);
    });
}

/// Shift-Or, interpreted and JIT-compiled: its start-by-start loop, including
/// a match past the long run so the earliest-end bound does not end the scan.
#[test]
fn shift_or_search_grows_linearly() {
    bounded_within("shift_or_search_grows_linearly", SCALING_DEADLINE, || {
        assert_linear_growth(&[
            GrowthCase::new(r"\w+z", "a", "").engines("ShiftOr", "JitShiftOr"),
            GrowthCase::new(r"(?i)\w+z", "a", "").engines("ShiftOr", "JitShiftOr"),
            GrowthCase::new(r"[a-z]+\d", "a", "").engines("ShiftOr", "JitShiftOr"),
            GrowthCase::new(r"a[a-z]*!", "a", "9a!"),
            GrowthCase::new(r"(a+)+$", "a", "!"),
            GrowthCase::new(r"x[a-z]*y{70}", "x", "!").engines("ShiftOrWide", "Jit"),
        ]);
    });
}

/// The lazy DFA and the DFA JIT: anchored patterns the lazy DFA keeps, and a
/// prefix-literal pattern the JIT build compiles to native code.
#[test]
fn lazy_dfa_and_dfa_jit_search_grows_linearly() {
    bounded_within(
        "lazy_dfa_and_dfa_jit_search_grows_linearly",
        SCALING_DEADLINE,
        || {
            assert_linear_growth(&[
                GrowthCase::new(r"a.*b$", "a", "").engines("LazyDfa", "LazyDfa"),
                GrowthCase::new(r"(?m)^a.*b$", "a", "").engines("LazyDfa", "LazyDfa"),
                GrowthCase::new(r"x[a-z]*y", "x", "!").engines("ShiftOr", "Jit"),
            ]);
        },
    );
}

/// The tagged-NFA interpreter: greedy runs it retries at every start and
/// backtracks into, the second shape cubic before its work was metered.
#[test]
fn tagged_nfa_search_grows_linearly() {
    bounded_within("tagged_nfa_search_grows_linearly", SCALING_DEADLINE, || {
        assert_linear_growth(&[
            GrowthCase::new(r"\p{L}+z", "a", "9z").engines("TaggedNfa", "TaggedNfa"),
            GrowthCase::new(r"\p{L}+a\p{L}*z", "a", "9z").engines("TaggedNfa", "TaggedNfa"),
            GrowthCase::new(r"a\p{L}*z", "a", "9z").engines("TaggedNfa", "TaggedNfa"),
            GrowthCase::new(r"\w+(?=z)", "x", "!z").engines("TaggedNfa", "TaggedNfa"),
        ]);
    });
}

// =============================================================================
// Parsing does not overflow the stack on deeply nested patterns
// =============================================================================
// The parser is mutually recursive with no explicit stack, so a pattern that
// nests deeply enough overflows the real call stack — which in Rust is an
// uncatchable SIGSEGV/abort, not a `Result::Err` a caller can handle. That
// makes `assert!(Regex::new(deep).is_err())` unsafe as a standalone test: if
// the depth guard ever regresses, the process crashes instead of the assertion
// failing, taking down the rest of the `cargo test` binary with it. Each case
// below is isolated in a child process for exactly that reason (see the module
// doc comment above).
//
// Patterns are held well past `parser::DEFAULT_NEST_LIMIT` (250) so the cases
// stay meaningful even if the limit is later raised somewhat.

/// A run of unmatched `(` is malformed — parsing would eventually reach an
/// unmatched-paren error — but the depth guard must reject it long before
/// that, on the way down through the nesting, not on the way back up.
#[test]
fn many_unmatched_open_parens_is_rejected_not_crashed() {
    bounded("many_unmatched_open_parens_is_rejected_not_crashed", || {
        let pattern = "(".repeat(50_000);
        let err = Regex::new(&pattern)
            .expect_err("unbounded nesting must not compile")
            .to_string();
        assert!(
            err.contains("nest"),
            "expected a nesting error, got {err:?}"
        );
    });
}

/// A well-formed pattern nested via `(?:...)` recurses just as deeply while
/// parsing the opening half, so the guard must catch it there rather than
/// relying on the (never reached) closing half to bound anything.
#[test]
fn many_well_formed_non_capturing_groups_is_rejected_not_crashed() {
    bounded(
        "many_well_formed_non_capturing_groups_is_rejected_not_crashed",
        || {
            let pattern = format!("{}{}", "(?:".repeat(50_000), ")".repeat(50_000));
            let err = Regex::new(&pattern)
                .expect_err("unbounded nesting must not compile")
                .to_string();
            assert!(
                err.contains("nest"),
                "expected a nesting error, got {err:?}"
            );
        },
    );
}

/// Character classes recurse on their own path (`parse_class` <->
/// `parse_class_term`), separate from the group/alternation cycle, and must be
/// bounded independently: `[a[a[a...` opens a fresh nested class after every
/// literal `a`.
#[test]
fn many_nested_classes_is_rejected_not_crashed() {
    bounded("many_nested_classes_is_rejected_not_crashed", || {
        let pattern = "[a".repeat(50_000);
        let err = Regex::new(&pattern)
            .expect_err("unbounded class nesting must not compile")
            .to_string();
        assert!(
            err.contains("nest"),
            "expected a nesting error, got {err:?}"
        );
    });
}

/// A bare `(?flags)` opens a new flag scope over the rest of its branch by
/// recursing into `parse_concat` again — a third recursion path, distinct from
/// both the group cycle and the class path, that a long run of flag changes
/// can walk arbitrarily deep.
#[test]
fn many_inline_flag_changes_is_rejected_not_crashed() {
    bounded("many_inline_flag_changes_is_rejected_not_crashed", || {
        let pattern = "(?i)(?-i)".repeat(50_000);
        let err = Regex::new(&pattern)
            .expect_err("unbounded flag-scope nesting must not compile")
            .to_string();
        assert!(
            err.contains("nest"),
            "expected a nesting error, got {err:?}"
        );
    });
}

/// The other side of the same guard: nesting one level under the default
/// limit must still compile cleanly. This is what catches an off-by-one that
/// rejects legitimate patterns, which the crash-only cases above cannot show.
#[test]
fn nesting_just_under_the_limit_still_compiles() {
    bounded("nesting_just_under_the_limit_still_compiles", || {
        let depth = (regexr::parser::DEFAULT_NEST_LIMIT - 1) as usize;
        let pattern = format!("{}a{}", "(?:".repeat(depth), ")".repeat(depth));
        Regex::new(&pattern).expect("nesting one under the limit must compile");
    });
}

// =============================================================================
// Literal extraction does not cost exponential time on deeply nested patterns
// =============================================================================
// `LiteralExtractor::extract` walks the HIR once per `Regex::new`. Two sites
// used to walk the same subtree twice: the `Concat` arm's extend-loop and its
// trailing-element check can land on the same node, and the `Alt` arm's
// bail-out into `extract_common_prefix` used to re-walk every branch from
// scratch, including the ones its own loop had already extracted. Either one
// gives `T(depth) = 2*T(depth-1)`, which crosses into "does not return"
// around depth 40-60 - well under the parser's 250-level nesting cap, so a
// pattern the parser happily accepts could still hang `Regex::new` itself,
// before any matching happens.
//
// Each case nests to depth 60: comfortably past the exponential crossover,
// comfortably under `DEFAULT_NEST_LIMIT`. Exponential behaviour hangs and
// trips the harness deadline; linear behaviour returns instantly.

/// Nesting depth for the extractor-doubling cases: past the exponential
/// crossover, under the parser's nesting cap.
const EXTRACTOR_DOUBLING_DEPTH: usize = 60;

/// `a(?:a(?:a(?:...(?:ab)...)))` - the `Concat` shape that doubles: every
/// level is a two-element `Concat[Literal, Tail]`, exactly the shape where
/// the extend-loop's break node and the trailing-element check's
/// `actual_last` are the same node.
fn nested_concat_doubling_pattern(depth: usize) -> String {
    format!("{}ab{}", "a(?:".repeat(depth), ")".repeat(depth))
}

/// Nested `(?:a...|\d)` alternations - the `Alt` shape that doubles: at
/// every level the `\d` branch has no literal prefix and forces a bail into
/// `extract_common_prefix`, which used to re-walk every branch - including
/// the nested alternation in the other branch - from scratch.
fn nested_alt_doubling_pattern(depth: usize) -> String {
    let mut pattern = String::from(r"\d");
    for _ in 0..depth {
        pattern = format!(r"(?:a{pattern}|\d)");
    }
    pattern
}

#[test]
fn nested_concat_literal_extraction_terminates() {
    bounded("nested_concat_literal_extraction_terminates", || {
        let pattern = nested_concat_doubling_pattern(EXTRACTOR_DOUBLING_DEPTH);
        Regex::new(&pattern).expect("well under the nesting cap, must compile");
    });
}

#[test]
fn nested_alt_literal_extraction_terminates() {
    bounded("nested_alt_literal_extraction_terminates", || {
        let pattern = nested_alt_doubling_pattern(EXTRACTOR_DOUBLING_DEPTH);
        Regex::new(&pattern).expect("well under the nesting cap, must compile");
    });
}

/// `required_literal`'s `Lookaround` arm (`src/literal/extractor.rs:601-606`)
/// reaches `LiteralExtractor::extract` on the lookahead's inner expression
/// directly, independent of the top-level prefix extraction - the outer
/// `x(?=...)` concat never doubles on its own (a lookaround is zero-width, so
/// the extend-loop `continue`s past it rather than breaking on it), so this
/// isolates that call path from the one the two cases above already cover.
#[test]
fn nested_concat_in_lookahead_literal_extraction_terminates() {
    bounded(
        "nested_concat_in_lookahead_literal_extraction_terminates",
        || {
            let inner = nested_concat_doubling_pattern(EXTRACTOR_DOUBLING_DEPTH);
            let pattern = format!("x(?={inner})");
            Regex::new(&pattern).expect("well under the nesting cap, must compile");
        },
    );
}

// =============================================================================
// Tagged-NFA step extraction does not cost exponential time on sequential
// alternation groups
// =============================================================================
// `StepExtractor` (src/nfa/tagged/steps.rs) emits an `Alt` step whose branches
// each carry a full copy of everything after the alternation. That is not just
// explored exponentially, the *emitted program itself* is exponentially sized:
// `k` sequential alternation groups produce ~2^k steps. Without a cap on the
// extraction budget, `Regex::new` on a pattern with a lookaround (which forces
// the tagged-NFA path) and enough sequential groups does not return in any
// reasonable time - 24 groups measured at ~18s and ~16.7M emitted steps before
// the fix. `MAX_EXTRACTED_STEPS` bounds that work by bailing out of extraction
// early, which sends the pattern to the PikeVm instead: `Regex::new` returns
// immediately, matching just costs more per search.

#[test]
fn pathological_sequential_alternations_do_not_blow_up_compile_time() {
    bounded(
        "pathological_sequential_alternations_do_not_blow_up_compile_time",
        || {
            let groups = ["(?:ab|cd)", "(?:ef|gh)", "(?:ij|kl)", "(?:mn|op)"];
            let mut pattern = String::from("(?=a)");
            for i in 0..32 {
                pattern.push_str(groups[i % groups.len()]);
            }
            Regex::new(&pattern).expect("pattern is well-formed and must compile");
        },
    );
}

// =============================================================================
// `EagerDfa::from_lazy`'s materialization BFS does not blow up compile time
// =============================================================================
// `(?:a?){n}` selects `EngineType::LazyDfa`, and without anchors or a large
// Unicode class that used to always materialize eagerly via
// `EagerDfa::from_lazy` — a BFS whose per-state cost scales with the size of
// that state's NFA subset. For this pattern shape the subset at DFA state
// `S_k` is Θ(n−k), so materializing costs Θ((n−k)²) per state and Θ(n³)
// overall: measured at n=2000, `Regex::new` did not return inside 500s.
// `MATERIALIZATION_WORK_BUDGET` (src/dfa/eager/shared.rs) now meters that BFS
// by cumulative NFA-subset size and declines partway through, falling back to
// `LazyDfa` instead of finishing the materialization.
#[test]
fn nullable_repetition_does_not_blow_up_compile_time() {
    bounded("nullable_repetition_does_not_blow_up_compile_time", || {
        Regex::new("(?:a?){2000}").expect("pattern is well-formed and must compile");
    });
}

/// `\X{4}` is the shape the engine selector cites as its reason for routing
/// every codepoint-class pattern to the PikeVM: `\X` is a nested alternation,
/// and the tagged step extractor copies each branch's continuation into every
/// branch, so repeating it compounds multiplicatively.
///
/// That reasoning predates `MAX_EXTRACTED_STEPS`, which now bounds the emitted
/// program and declines — falling back to the PikeVM — rather than letting it
/// grow without limit. This pins the outcome the budget is supposed to
/// guarantee: whichever engine ends up running it, compiling and searching
/// `\X{4}` terminates. It is checked on the tagged path explicitly, because
/// that is the path the selector is avoiding.
#[test]
fn repeated_grapheme_cluster_terminates_on_the_tagged_path() {
    bounded(
        "repeated_grapheme_cluster_terminates_on_the_tagged_path",
        || {
            let haystack = "a\u{0301}e\u{0302}i\u{0303}o\u{0304}".repeat(64);
            for pattern in [r"\X{4}", r"\X{2,4}", r"\X{4}z"] {
                for jit in [false, true] {
                    let re = RegexBuilder::new(pattern)
                        .jit(jit)
                        .build()
                        .expect("pattern is well-formed and must compile");
                    let count = re.find_iter(&haystack).count();
                    std::hint::black_box(count);
                }
            }
        },
    );
}
