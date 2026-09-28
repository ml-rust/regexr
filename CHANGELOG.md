# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each release must have a non-empty section here before it can be tagged — `.github/workflows/release-validate.yml` refuses a tag whose version has no entry, and the GitHub Release body is this file's section for that version.

## [0.6.0] - 2026-09-28

### Fixed

- An alternation of classes or class sequences, like `[sS]|[sS][eE][cC]`, ran as its last class alone and found the wrong match or none.
- The DFA engines and Shift-Or returned the longest match where a greedy repeat sets leftmost-first priority: `a?(?:ab)?` on `ab` matched `ab`, not `a`.
- The JIT-compiled DFA gave up at the first start whose attempt reached the end of the input: `a(?:ab)?b` found no match in `aab`.
- `^`, `$`, `\b` and `\B` inside a match, stacked, or on an optional path were misjudged by the DFA engines and Shift-Or: `a$a` matched `aa`.
- A search whose attempts each scanned far before failing took time quadratic in the input on the DFA, Shift-Or and tagged-NFA engines: `(?s)a.*b` over 50 KB of `a` took seconds.

### Added

- `engine::automata_match_like_pikevm`, which reports whether the DFA engines and Shift-Or give a pattern the PikeVM's match.
- `nfa::priority_is_irrelevant` and `nfa::assertions_at_edges`, the two NFA properties that decision rests on.

### Changed

- Patterns whose match priority or assertion placement the DFA engines cannot honour run on the PikeVM.
- `EagerDfa::find_from_simple` returns `Result`, reporting a search that gives up on its scan budget as `find_from` does.
- With the JIT enabled, a tagged-NFA pattern that repeats without bound runs on the interpreter, whose search is metered.

## [0.5.0] - 2026-08-14

### Fixed

- Deeply nested patterns overflowed the stack; the parser now refuses them past a nesting limit.
- Compilation or search blew up on nested alternations, repeated zero-width groups, and shapes like `(?:a?){n}`.
- A class could match code points it never named, and a negated class exclude them.
- `\w`, `\W`, `\d` and `\D` in a bracketed class now honour Unicode mode, so `[\w]` and `\w` agree.
- A lazy-DFA cache flush mid-search could truncate or drop a match, in both the interpreter and JIT-compiled code.
- Concurrent searches on one `Regex` no longer serialize on a shared lazy DFA.
- A greedy run before a lookahead settled too early: `[ab]+(?=b)bb` found no match in `abbabb`.

### Added

- `RegexBuilder::nest_limit`, the parser's nesting cap (default 250).
- Anchored `match_at` on the tagged-NFA interpreter and its JIT.
- `dfa::CacheCeilingExceeded`, returned when a lazy DFA outgrows its cache mid-search.

### Changed

- `EagerDfa::from_lazy` and `EagerDfaEngine::new`/`from_lazy` return `Result`, declining patterns too costly to materialize.
- Lookbehind over a class of mixed UTF-8 width, like `(?<=\s)x`, is supported by the tagged-NFA interpreter.
- `\w+(?=ing\b)` searches its lookahead literal end-first; `(?<=@)\w+` prefilters on the lookbehind literal.
- Codepoint-class patterns like `\p{L}+` run on the tagged NFA rather than the PikeVM.

## [0.4.0] - 2026-08-13

### Fixed

- A lookbehind holds when _any_ path through its inner pattern ends at the current position, not only the path that pattern prefers. `(?<=..?)b` failed on `ab` and `(?<!..?)b` matched it; `(?<=\w\w?)x` and `(?<=a.*)x` failed likewise, and `(?<=ab|a)b` failed on `ab` because only the longer branch was tried.
- A zero-width assertion at the end of a lookaround is part of it. `(?<=a\b)`, `(?<=a$)`, `(?=a$)` and `(?!a\b)` evaluated as `(?<=a)`, `(?=a)` and `(?!a)`, so `(?<=a$)b` matched `ab` and `x(?=a$)` matched `xab`. `\b`, `$`, `\Z` and multiline `$` were all affected, in positive and negative lookahead and lookbehind alike.
- `find` and `captures` no longer disagree on those patterns, in either direction: on `ax`, `(?<=a\b)x` reported a match from `find` and none from `captures`; on `xab`, `x(?!a\b)` reported none from `find` and a match from `captures`.
- An assertion following a quantifier is no longer lost to it: `a*(?=b)(?=c)` matched `ab`, ignoring the second lookahead.
- A greedy byte class no longer loses match priority to a Unicode class that crosses several bytes in one step. `[^\s]*\p{N}` matched only the leading `½` of `½½` rather than the whole input; any greedy byte class followed by a class able to consume a multi-byte codepoint it could have taken itself was affected.

### Added

- `Nfa::compute_max_match_len`, `Nfa::needs_left_context` and `Nfa::compute_splits_codepoints`, with the `Nfa::max_match_len` and `Nfa::splits_codepoints` fields they fill in.
- `PikeVm::matches_ending_at`, which reports whether any match ends at a given position.
- `nfa::at_line_start` and `nfa::at_line_end`, now the single definition of multiline `^`/`$` for every engine.
- `nfa::utf8_automata::utf8_width` and `UTF8_WIDTH_BOUNDARIES`.

### Removed

- `StepExtractor::calc_min_len`.

### Changed

- Evaluating a bounded lookbehind is linear in the input where it was quadratic: the search for a start position is confined to the longest match its inner pattern admits.
- `Nfa` is `#[non_exhaustive]` and carries two further public fields, so it can no longer be built with a struct literal from outside the crate; use `Nfa::new`. Later fields will not be breaking.

## [0.3.2] - 2026-08-09

### Changed

- Literal prefiltering is NEON-accelerated on AArch64, which previously ran the scalar path.

## [0.3.1] - 2026-08-08

### Fixed

- A small Unicode class stays a codepoint node when a lookaround or non-greedy quantifier already pins the engine. 0.3.0 lowered it to a byte trie regardless, running a tokenizer's split pattern an order of magnitude slower than 0.2.2.
- The tagged-NFA JIT reported a corrupt match for a greedy quantifier before a lookaround over a Unicode class, such as `a+(?!\S)` on a non-ASCII follower.
- The AArch64 tagged-NFA JIT silently dropped lookaround steps it does not implement; it now defers to the interpreter.

### Changed

- The PikeVM tries an anchored match at the search start before its unanchored sweep.

## [0.3.0] - 2026-08-08

### Added

- POSIX bracket expressions, `\x{...}`, `\cX`, `[\b]`, `(?#…)`, `(?'name'…)`, named backreferences, `\h`/`\H`, `\R`, `\N`, and the class set operators `&&`, `--`, `~~`.
- A one-pass capture engine, JIT-compiled on x86-64 and AArch64 under the `jit` feature.
- `RegexBuilder::size_limit`, which sets how large a pattern may expand to.

### Fixed

- `.` and the negated Perl classes match one whole codepoint, not one byte.
- A class encodes codepoints above U+007F as UTF-8 rather than raw bytes.
- Nested lookaround matched wrongly in both directions; lookbehind now sees the full input.
- `(a|b)+` reported only its first iteration.
- An empty match at the previous match's end is no longer reported twice. Affects every nullable pattern (`a*`, `\d*`, `\b\w*`).
- `RegexBuilder::jit(true)` no longer selects a slower engine than `Regex::new`.
- A prefilter no longer makes an engine without an anchored match quadratic in the input.

### Changed

- Possessive quantifiers (`a*+`) and atomic groups (`(?>…)`) are rejected.
- A pattern is refused past 65535 for a single `{n,m}` bound, or 10000 expanded elements; raise it with `RegexBuilder::size_limit`.

## [0.2.2] - 2026-08-07

### Fixed

- A malformed escape at the start of a pattern is rejected instead of compiling to the empty pattern, which matched every input. `\e`, `\a`, `\q`, `\x`, `\u{`, `\p{` and a lone `\` all reported success and matched everywhere; only the first token was affected, so the same escape one character later already errored.
- Inline flag negation works: `(?-i)`, `(?im-sx)` and `(?-x:…)` were rejected as invalid groups because `-` never reached the flag parser.
- A mid-pattern `(?flags)` applies to what follows it rather than to the whole pattern. The change previously reached only the AST's single global flag set, so `(?i)ab(?-i)CD` matched case-sensitively throughout.
- An escape that is invalid inside a character class names itself: `[\b]` reported `invalid escape sequence '\?'`.

### Added

- Extended mode (`x`) is implemented. Unescaped ASCII whitespace and `#` comments outside a character class are no longer part of the pattern; previously the flag was accepted and ignored.
- `\X` matches one extended grapheme cluster, following UAX #29 in full — Hangul syllables, emoji ZWJ sequences, regional-indicator flag pairs and Indic conjuncts included. Checked against the complete Unicode grapheme-break test suite.
- `\Q…\E` quotes everything between it literally, including metacharacters, extended-mode whitespace and character-class syntax. An unterminated `\Q` runs to the end of the pattern and a stray `\E` is ignored, as in PCRE.
- `\a` (alert, U+0007) and `\e` (escape, U+001B).
- `\` followed by any non-alphanumeric ASCII character is that character, matching PCRE, Perl, Java and the `regex` crate. `\ `, `\"`, `\@` and `\#` previously failed to compile.

### Changed

- `escape` also escapes `#` and ASCII whitespace, so its output is safe to splice into an extended-mode pattern.
- `Parser::new` returns `Result<Self>`, and `Lexer::read_ident` is replaced by the infallible `Lexer::read_ident_rest`. Both are internal parsing types exposed through `regexr::parser`; `Regex` and `RegexBuilder` are unaffected.

## [0.2.1] - 2026-08-06

### Fixed

- aarch64 agrees with x86-64 on capture groups after a greedy Unicode quantifier. The ARM64 tagged-NFA JIT let `\p{L}+` consume its whole run when extracting captures, so `(\p{L}+)(\p{L})` reported no match on input it matches.

### Performance

- A greedy Unicode quantifier (`\p{L}+`, `\p{N}+`, …) no longer allocates while matching. The tagged-NFA interpreter recorded every codepoint boundary of the run in case one had to be given back; it now steps backwards through UTF-8 only when that happens.

## [0.2.0] - 2026-08-06

### Fixed

- Capture extraction runs on the same linear-time engines as `find`. Backtracking is now reserved for backreferences, bounded by a step budget.
- Repetition over a body that can match the empty string terminates (`(a*)*`, `(a?)*`, `()+`).
- Choice-point stack growth in the backtracking JIT is bounded; deep searches continue on the interpreter.
- Freeing a match's capture history no longer recurses once per recorded group, so a long match over a capturing loop cannot exhaust the stack.
- `$` matches before a trailing newline in every engine, and `(?m)` line anchors are distinguished from text anchors — including when `(?m)` appears mid-pattern.
- Matches never start inside a multi-byte codepoint.
- Negated classes match whole characters rather than single bytes, so `[^a]b` matches `"中b"`.
- `\b`/`\B` at the start of a search is decided against the following byte rather than assumed.
- aarch64 agrees with x86-64 on `\b`/`\B`-terminated patterns, which the ARM64 JIT compiled where a shorter end could satisfy the assertion (`a+\B` on `"aa"`).

### Performance

- Unanchored search is linear in the input across every engine. Patterns that consume a long run before failing — `(a+)+$`, `(a|a)+$`, `([a-zA-Z]+)*$` — used to cost one full scan per start position.
- A prefilter that keeps most positions no longer drives one verification per candidate; the engine's own search takes over.
- A `$`-anchored pattern whose every match ends on `$` rejects a non-matching start outright instead of re-checking it state by state.

### Added

- `RegexBuilder::backtrack_limit` and `Regex::try_find` / `try_captures` / `try_is_match`, reporting `ErrorKind::MatchLimitExceeded` instead of "no match" when a backreference search exhausts its budget.
- `reference::captures` / `captures_from`, so the executable spec can answer for group spans and not just the overall match.

### Changed

- `engine::hir_matches_empty` moved to `hir::matches_empty`. The predicate is unchanged; it now lives with the IR it inspects, and the duplicate copies in the selector and Shift-Or are gone.
- `jit::MaterializedState` carries the character class of the byte that reached the state, which the JIT needs to decide whether a word-boundary match is safe to compile.
- The DFA JIT is one driver for both architectures; only code emission is target-specific. `jit::CompiledRegex` and `jit::MaterializedState` are single types rather than a pair per target.
- `vm::ShiftOrInterpreter` delegates to `ShiftOr` instead of reimplementing it.
- `vm::PikeVmContext::new` takes only a state count, and its `capture_slots` field is gone; threads carry their own capture history.

## [0.1.5] - 2026-08-05

### Added

- `escape()` — quote a string so it matches literally, for building patterns from untrusted or user-supplied text.

### Fixed

- Matching no longer panics when a match boundary falls inside a multi-byte codepoint. The public API now reports byte offsets that are always valid UTF-8 boundaries.
- Left context is preserved across a resumed search, so look-behind assertions and `^`/`\b` decide against the real preceding text rather than the start of the resumed slice. This affected both `find_at`-style resumption and iteration.

### Performance

- Unanchored capture extraction in the PikeVM runs as a single pass instead of restarting the VM at every start offset.
- Unanchored Shift-Or scans are bounded by a single-pass existence check, so a non-matching haystack is rejected once rather than re-scanned per candidate start.

### Changed

- Minimum supported Rust version is declared as 1.83 in `Cargo.toml`. It was already the version CI checked; it is now a resolver constraint and a build-time error rather than a CI-only convention.
- Dependencies moved to their current releases: `dynasm`/`dynasmrt` 5.1, `memchr` 2.8, `aho-corasick` 1.1, and — for benchmarks only — `regex` 1.13 and `fancy-regex` 0.19.
- The published crate no longer carries repository scaffolding — README screenshots, the raw UCD input files, CI config, `docs/` — that nothing in the build reads. The packaged archive drops from 1.2 MiB to 410 KiB compressed. The README now references its images and its `docs/` pages by absolute GitHub URL, so they render on crates.io and docs.rs from the repository rather than from a copy inside every downloaded crate.

## [0.1.4] - 2026-06-06

### Fixed

- Leftmost-first branch priority is enforced across engines, so an alternation picks the first branch that can match rather than whichever engine-internal order happened to win.
- `$` follows PCRE/Python semantics consistently, and end-anchor handling agrees between the interpreted engines and the JIT.
- aarch64 JIT: end-of-text validation, unanchored retry, and greedy backtracking corrected.
- `TaggedNfa` gained a per-kind assertion-tally soundness guard, closing a case where an assertion count could be trusted after it went stale.
- Parser: scoped inline flags, nested character classes, and Unicode `\s` are handled.

### Added

- A reference matcher and a differential conformance suite: every engine is now checked against one independent implementation on the same inputs, so an engine-specific divergence fails the suite instead of hiding behind engine selection.

## [0.1.3] - 2026-06-04

### Fixed

- aarch64 JIT clippy warnings across the JIT backends.

### Changed

- Dependency bumps: `dynasm`/`dynasmrt` 5.0, `criterion` 0.8, `fancy-regex` 0.18.

## [0.1.2] - 2026-03-15

### Fixed

- Missing `.arch x64` directives in the dynasm blocks that implement the calling convention, which made x86-64 codegen depend on the host's default architecture rather than the target's.

## [0.1.1] - 2026-03-15

### Fixed

- Missing `.arch aarch64` directives in every dynasm block, so ARM64 JIT compilation no longer depends on the assembler's default architecture.

## [0.1.0] - 2026-03-12

The first release.

### Added

- **Multiple execution backends, selected per pattern** — Shift-Or (bit-parallel, small patterns), Wide Shift-Or (65–256 positions), eager DFA (anchors and word boundaries), lazy DFA (general), backtracking VM (backreferences), and PikeVM (lookaround, non-greedy).
- **JIT compilation** for x86-64 and aarch64 behind the `jit` feature: a backtracking JIT for backreferences, a tagged-NFA JIT at parity with the interpreter for lookaround and non-greedy quantifiers, a Shift-Or JIT for alternation-led patterns, and a DFA JIT with SIMD prefiltering. Windows x64 ABI supported.
- **SIMD literal search** (`simd`, on by default): AVX2 `memchr` family and Teddy multi-literal matching.
- **Unicode support** generated from the UCD — general categories, scripts, and properties, with a `CodepointClass` instruction so large classes do not explode the DFA.
- **Lookaround and backreferences**, which the `regex` crate does not offer.
- Dual MIT / Apache-2.0 licensing.
