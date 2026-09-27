//! HIR builder - translates AST to HIR.

use crate::error::{Error, ErrorKind, Result};
use crate::nfa::utf8_automata::{
    compile_utf8_complement, compile_utf8_range, optimize_sequences, Utf8Sequence,
};
use crate::parser::{
    Anchor, Ast, Class, ClassRange, Expr, Flags, Group, GroupKind, Lookaround, LookaroundKind,
    PerlClassKind, Repeat,
};

/// Code points matched by `\h` (horizontal whitespace): Tab, Space, and every
/// Unicode horizontal space separator. A fixed 18-code-point list per PCRE —
/// not derived from a UCD property table. This is the single source of truth
/// for the set; `parser::class::horizontal_whitespace_ranges` (used for
/// `[\h]`/`[\H]`) builds its ranges from this same constant via the `hir`
/// crate-root re-export.
pub const HORIZONTAL_WHITESPACE: &[(u32, u32)] = &[
    (0x0009, 0x0009),
    (0x0020, 0x0020),
    (0x00A0, 0x00A0),
    (0x1680, 0x1680),
    (0x2000, 0x200A),
    (0x202F, 0x202F),
    (0x205F, 0x205F),
    (0x3000, 0x3000),
];

/// Code points `\R` matches singly, once the two-character `\r\n` sequence
/// has already been tried first (see `HirTranslator::build_line_break_expr`):
/// LF, VT, FF, CR, NEL, LINE SEPARATOR, PARAGRAPH SEPARATOR.
const LINE_BREAK_SINGLE: &[(u32, u32)] = &[(0x000A, 0x000D), (0x0085, 0x0085), (0x2028, 0x2029)];

use super::unicode_data;

use super::{
    CodepointClass, Hir, HirAnchor, HirCapture, HirClass, HirExpr, HirLookaround,
    HirLookaroundKind, HirProps, HirRepeat,
};

/// Builds the HIR for `\z` (strict end of text): the negative lookahead
/// `(?![\s\S])` — "no character ahead". Lets `HirAnchor::End` carry the common
/// `$`/`\Z` before-newline semantics while `\z` stays exactly strict.
fn strict_end_of_text_lookahead() -> HirExpr {
    let any_byte = HirExpr::Class(HirClass::new(vec![(0, 255)], false));
    HirExpr::Lookaround(Box::new(HirLookaround {
        expr: any_byte,
        kind: HirLookaroundKind::NegativeLookahead,
    }))
}

/// Translator from AST to HIR.
pub struct HirTranslator {
    props: HirProps,
    flags: Flags,
    /// Maximum backreference index used in the pattern.
    max_backref: u32,
    /// Tracks codepoint ranges during class translation for potential fast matching.
    /// Set during translate_class, consumed by translate if pattern is a simple class.
    current_class_codepoints: Option<(Vec<(u32, u32)>, bool)>,
    /// Whether something *other than* class lowering already pins this pattern
    /// to a codepoint-capable engine — see [`pins_codepoint_engine`].
    ///
    /// Read from the AST once, before translation, rather than from
    /// [`HirProps`]: those flags are filled in as translation proceeds, and a
    /// class is usually lowered before the construct that pins the engine has
    /// been reached (`\s+(?!\S)` lowers `\s` before it sees the lookahead).
    engine_already_pinned: bool,
}

impl HirTranslator {
    /// Creates a new translator.
    pub fn new() -> Self {
        Self {
            props: HirProps::default(),
            flags: Flags::default(),
            max_backref: 0,
            current_class_codepoints: None,
            engine_already_pinned: false,
        }
    }

    /// Translates an AST to HIR.
    pub fn translate(&mut self, ast: &Ast) -> Result<Hir> {
        self.translate_with_limit(ast, DEFAULT_EXPANDED_SIZE)
    }

    /// [`Self::translate`] under a caller-chosen expansion ceiling.
    pub fn translate_with_limit(&mut self, ast: &Ast, limit: u32) -> Result<Hir> {
        self.flags = ast.flags;
        self.engine_already_pinned = pins_codepoint_engine(&ast.expr);
        let expr = self.translate_expr(&ast.expr)?;

        // Validate backreferences: all referenced groups must exist
        if self.max_backref > self.props.capture_count {
            return Err(Error::new(
                ErrorKind::BackrefNotFound(self.max_backref as usize),
                format!(
                    "backreference \\{} references non-existent capture group (only {} groups defined)",
                    self.max_backref, self.props.capture_count
                ),
            ));
        }

        // CodepointClassMatcher runs only when the whole pattern is one bracket
        // class. That is decided on the AST: the HIR of `[sS]|[sS][eE][cC]` has
        // the same shape as the UTF-8 lowering of a single class, and
        // `current_class_codepoints` holds only the last class translated.
        if let Some((ranges, negated)) = self.current_class_codepoints.take() {
            if is_single_class(&ast.expr) && Self::is_simple_unicode_class(&expr) {
                self.props.codepoint_class = Some(CodepointClass::new(ranges, negated));
            }
        }

        let size = expanded_size(&expr);
        if size > limit {
            return Err(Error::new(
                ErrorKind::ExpansionTooLarge { size, limit },
                format!(
                    "the pattern expands to {size} elements, past the limit of {limit}; \
                     raise it with RegexBuilder::size_limit if the cost is acceptable"
                ),
            ));
        }

        Ok(Hir {
            expr,
            props: self.props.clone(),
        })
    }

    /// Translates an expression.
    fn translate_expr(&mut self, expr: &Expr) -> Result<HirExpr> {
        match expr {
            Expr::Empty => Ok(HirExpr::Empty),

            Expr::Literal(c) => self.translate_literal(*c),

            Expr::GraphemeCluster => Ok(super::grapheme::grapheme_cluster()),

            Expr::Dot => {
                let dot_all = self.flags.dot_all;
                Ok(self.build_dot_expr(dot_all))
            }

            Expr::Concat(exprs) => {
                let mut hir_exprs = Vec::with_capacity(exprs.len());
                for e in exprs {
                    hir_exprs.push(self.translate_expr(e)?);
                }
                Ok(HirExpr::Concat(hir_exprs))
            }

            Expr::Alt(exprs) => {
                let mut hir_exprs = Vec::with_capacity(exprs.len());
                for e in exprs {
                    hir_exprs.push(self.translate_expr(e)?);
                }
                Ok(HirExpr::Alt(hir_exprs))
            }

            Expr::Repeat(rep) => self.translate_repeat(rep),

            Expr::Group(group) => self.translate_group(group),

            Expr::Class(class) => self.translate_class(class),

            Expr::Anchor(anchor) => {
                let hir_anchor = match anchor {
                    Anchor::StartOfString | Anchor::StartOfInput => {
                        self.props.has_anchors = true;
                        self.props.has_start_anchor = true;
                        HirAnchor::Start
                    }
                    // `$` and `\Z` — end of text OR just before a final newline
                    // (PCRE/Python). `HirAnchor::End` carries that semantics and
                    // stays a fast anchor (Shift-Or/DFA/JIT handle it directly).
                    Anchor::EndOfString | Anchor::EndOfInputBeforeNewline => {
                        self.props.has_anchors = true;
                        self.props.has_end_anchor = true;
                        HirAnchor::End
                    }
                    // `\z` — strict end of text (rare). Encoded as `(?![\s\S])`
                    // (no char ahead), reusing the lookaround machinery so `End`
                    // can carry the common before-newline `$` semantics.
                    Anchor::EndOfInput => {
                        self.props.has_anchors = true;
                        self.props.has_end_anchor = true;
                        self.props.has_lookaround = true;
                        return Ok(strict_end_of_text_lookahead());
                    }
                    Anchor::StartOfLine => {
                        self.props.has_anchors = true;
                        self.props.has_start_anchor = true;
                        self.props.has_multiline_anchors = true;
                        HirAnchor::StartLine
                    }
                    Anchor::EndOfLine => {
                        self.props.has_anchors = true;
                        self.props.has_end_anchor = true;
                        self.props.has_multiline_anchors = true;
                        HirAnchor::EndLine
                    }
                    Anchor::WordBoundary => {
                        self.props.has_word_boundary = true;
                        HirAnchor::WordBoundary
                    }
                    Anchor::NotWordBoundary => {
                        self.props.has_word_boundary = true;
                        HirAnchor::NotWordBoundary
                    }
                };
                Ok(HirExpr::Anchor(hir_anchor))
            }

            Expr::Lookaround(la) => self.translate_lookaround(la),

            Expr::Backref(n) => {
                self.props.has_backrefs = true;
                self.max_backref = self.max_backref.max(*n);
                Ok(HirExpr::Backref(*n))
            }

            Expr::UnicodeProperty { name, negated } => {
                self.translate_unicode_property(name, *negated)
            }

            Expr::PerlClass(kind) => self.translate_perl_class(*kind),

            Expr::LineBreak => self.build_line_break_expr(),

            // Identical to `.` with dot-all off, and deliberately built from
            // a literal `false` rather than `self.flags.dot_all` — `\N` stays
            // "no newline" even under `(?s)`, unlike `.`.
            Expr::AnyExceptNewline => Ok(self.build_dot_expr(false)),
        }
    }

    /// Translates a literal character to HIR.
    /// If case_insensitive flag is set, emits a class matching all case variants.
    fn translate_literal(&mut self, c: char) -> Result<HirExpr> {
        if self.flags.case_insensitive {
            // Get all case-equivalent code points
            let equivalents = unicode_data::case_fold_equivalents(c as u32);

            if equivalents.len() > 1 {
                // Multiple equivalents - emit a character class
                // Convert code points to ranges for the class
                let ranges: Vec<(u32, u32)> = equivalents.iter().map(|&cp| (cp, cp)).collect();

                return self.translate_ranges_to_hir(&ranges, false);
            }
            // Single code point (no case variants) - fall through to literal
        }

        // Standard literal: encode as UTF-8 bytes
        let mut bytes = [0u8; 4];
        let len = c.encode_utf8(&mut bytes).len();
        Ok(HirExpr::Literal(bytes[..len].to_vec()))
    }

    /// Translates a Perl shorthand class to HIR.
    fn translate_perl_class(&mut self, kind: PerlClassKind) -> Result<HirExpr> {
        // `\s`/`\S` always use the full Unicode `White_Space` set, matching the
        // Rust `regex`/Python defaults and the reference tokenizer engines (onig,
        // PCRE2+UCP) — e.g. `\s` includes U+00A0 and U+2000-U+200A. It's a small
        // set (~25 codepoints). `\h`/`\H` (horizontal whitespace) are always
        // Unicode-correct for the same reason and the same size class. `\w`/`\d`/`\b`
        // stay gated on Unicode mode (`(?u)`) since their Unicode forms are huge
        // and would bloat the byte engines.
        let unicode = self.flags.unicode
            || matches!(
                kind,
                PerlClassKind::Whitespace
                    | PerlClassKind::NotWhitespace
                    | PerlClassKind::HorizontalWhitespace
                    | PerlClassKind::NotHorizontalWhitespace
            );
        if unicode {
            self.translate_perl_class_unicode(kind)
        } else {
            self.translate_perl_class_ascii(kind)
        }
    }

    /// Translates a Perl class in ASCII mode.
    ///
    /// A negated class (`\W`, `\D`) is negated over *characters*, not bytes:
    /// `\W` means "a character other than `[A-Za-z0-9_]`", which must include
    /// non-ASCII code points like `é` or `世`, matched whole. It is built the
    /// same way as a negated ASCII class like `[^a]` (see `build_negated_class_expr`)
    /// via `build_ascii_or_non_ascii`, rather than a raw negated byte class,
    /// which would match a single continuation byte instead of a full
    /// codepoint.
    fn translate_perl_class_ascii(&self, kind: PerlClassKind) -> Result<HirExpr> {
        let (ranges, negated) = match kind {
            PerlClassKind::Digit => (vec![(b'0', b'9')], false),
            PerlClassKind::NotDigit => (vec![(b'0', b'9')], true),
            PerlClassKind::Word => (
                vec![(b'a', b'z'), (b'A', b'Z'), (b'0', b'9'), (b'_', b'_')],
                false,
            ),
            PerlClassKind::NotWord => (
                vec![(b'a', b'z'), (b'A', b'Z'), (b'0', b'9'), (b'_', b'_')],
                true,
            ),
            PerlClassKind::Whitespace => (
                vec![
                    (b' ', b' '),
                    (b'\t', b'\t'),
                    (b'\n', b'\n'),
                    (b'\r', b'\r'),
                    (0x0C, 0x0C),
                    (0x0B, 0x0B),
                ],
                false,
            ),
            PerlClassKind::NotWhitespace => (
                vec![
                    (b' ', b' '),
                    (b'\t', b'\t'),
                    (b'\n', b'\n'),
                    (b'\r', b'\r'),
                    (0x0C, 0x0C),
                    (0x0B, 0x0B),
                ],
                true,
            ),
            // The ASCII-only members of `\h`'s set (Tab and Space); the
            // remaining 16 code points are all non-ASCII. This arm is
            // unreachable in practice — `translate_perl_class`'s `unicode`
            // guard always routes `\h`/`\H` through
            // `translate_perl_class_unicode` instead, the same way it already
            // does for `\s`/`\S` — but is kept correct rather than left as a
            // trap, matching this match's existing precedent.
            PerlClassKind::HorizontalWhitespace => (vec![(b'\t', b'\t'), (b' ', b' ')], false),
            PerlClassKind::NotHorizontalWhitespace => (vec![(b'\t', b'\t'), (b' ', b' ')], true),
        };

        if negated {
            let surviving_ascii = merge_byte_ranges(complement_within_ascii(&ranges));
            return Ok(self.build_ascii_or_non_ascii(surviving_ascii));
        }

        Ok(HirExpr::Class(HirClass::new(ranges, negated)))
    }

    /// Translates a Perl class in Unicode mode.
    /// Uses the pre-computed PERL_WORD, PERL_DECIMAL, and PERL_SPACE tables
    /// which exactly match Perl/PCRE semantics from UCD.
    fn translate_perl_class_unicode(&mut self, kind: PerlClassKind) -> Result<HirExpr> {
        // Use the pre-computed Perl class tables from unicode_data
        let (ranges, negated): (&[(u32, u32)], bool) = match kind {
            PerlClassKind::Digit => (unicode_data::PERL_DECIMAL, false),
            PerlClassKind::NotDigit => (unicode_data::PERL_DECIMAL, true),
            PerlClassKind::Word => (unicode_data::PERL_WORD, false),
            PerlClassKind::NotWord => (unicode_data::PERL_WORD, true),
            PerlClassKind::Whitespace => (unicode_data::PERL_SPACE, false),
            PerlClassKind::NotWhitespace => (unicode_data::PERL_SPACE, true),
            PerlClassKind::HorizontalWhitespace => (HORIZONTAL_WHITESPACE, false),
            PerlClassKind::NotHorizontalWhitespace => (HORIZONTAL_WHITESPACE, true),
        };

        // `has_large_unicode_class` is set by the two builders that actually emit
        // a `UnicodeCpClass` (and by `translate_unicode_property` where it emits
        // one directly). Pre-judging it from the code-point count would flag `\s`
        // and `\S`, which now lower to small byte tries — and the flag routes a
        // pattern away from the DFA and Shift-Or engines.
        self.translate_ranges_to_hir(ranges, negated)
    }

    /// Converts code point ranges to HIR expression.
    ///
    /// The negated case is built straight from the exact merged code-point
    /// ranges via `build_negated_class_expr`, the same route `translate_class`
    /// uses for bracketed classes like `[^...]` — it skips the lossy
    /// byte-range/UTF-8-sequence detour entirely (pushing ranges through
    /// `compile_utf8_range`/`optimize_sequences` and then reconstructing the
    /// excluded set by decoding those sequences back through a hull-taking
    /// decode (`utf8_sequence_to_code_point_range`, since deleted), which
    /// over-approximates once `optimize_sequences` has merged sequences with
    /// partial continuation ranges — see `build_negated_class_expr`'s docs).
    fn translate_ranges_to_hir(&mut self, ranges: &[(u32, u32)], negated: bool) -> Result<HirExpr> {
        if negated {
            let mut sorted = ranges.to_vec();
            sorted.sort_by_key(|r| r.0);
            let merged = merge_codepoint_ranges(sorted);
            return Ok(self.build_negated_class_expr(&merged));
        }

        let mut byte_ranges: Vec<(u8, u8)> = Vec::new();
        let mut utf8_sequences: Vec<Utf8Sequence> = Vec::new();

        for &(start, end) in ranges {
            push_codepoint_range(start, end, &mut byte_ranges, &mut utf8_sequences);
        }

        byte_ranges.sort_by_key(|r| r.0);
        let merged_bytes = merge_byte_ranges(byte_ranges);
        let optimized_seqs = optimize_sequences(utf8_sequences);
        let merged_codepoints = merge_codepoint_ranges(ranges.to_vec());

        Ok(self.build_class_expr(merged_bytes, optimized_seqs, &merged_codepoints))
    }

    /// Translates a Unicode property to HIR.
    fn translate_unicode_property(&mut self, name: &str, negated: bool) -> Result<HirExpr> {
        let ranges = unicode_data::get_property(name)
            .ok_or_else(|| Error::new(ErrorKind::UnknownUnicodeProperty(name.to_string()), name))?;

        // Unicode properties with many code points cause DFA state explosion.
        // Use CodepointClass for large properties to avoid expanding UTF-8 automata.
        // Thresholds:
        //   - total code points > 500
        //   - any range covers > 500 code points
        //   - many disjoint ranges (> 50) cause excessive UTF-8 alternation branches
        // Negated properties (\P{...}) cover almost all of Unicode, so always use CodepointClass.
        let total_codepoints: u32 = ranges.iter().map(|(s, e)| e - s + 1).sum();
        let has_large_range = ranges.iter().any(|(s, e)| e - s > 500);
        let has_many_ranges = ranges.len() > 50;
        let is_large = negated || total_codepoints > 500 || has_large_range || has_many_ranges;

        if is_large {
            self.props.has_large_unicode_class = true;
            // Use CodepointClass for efficient runtime matching
            let cp_ranges: Vec<(u32, u32)> = ranges.to_vec();
            return Ok(HirExpr::UnicodeCpClass(CodepointClass::new(
                cp_ranges, negated,
            )));
        }

        // For small Unicode properties, expand to byte-level automata
        // which can be handled efficiently by DFA engines

        // Convert code point ranges to UTF-8 sequences
        let mut byte_ranges: Vec<(u8, u8)> = Vec::new();
        let mut utf8_sequences: Vec<Utf8Sequence> = Vec::new();

        for &(start, end) in ranges {
            push_codepoint_range(start, end, &mut byte_ranges, &mut utf8_sequences);
        }

        // Sort and merge byte ranges
        byte_ranges.sort_by_key(|r| r.0);
        let merged_bytes = merge_byte_ranges(byte_ranges);

        // Optimize UTF-8 sequences
        let optimized_seqs = optimize_sequences(utf8_sequences);

        // Build the final expression (non-negated path)
        let merged_codepoints = merge_codepoint_ranges(ranges.to_vec());
        Ok(self.build_class_expr(merged_bytes, optimized_seqs, &merged_codepoints))
    }

    /// Builds a negated Unicode class directly from codepoint ranges.
    /// This is more accurate than reconstructing ranges from UTF-8 sequences.
    #[allow(dead_code)]
    fn build_negated_unicode_from_ranges(&mut self, ranges: &[(u32, u32)]) -> HirExpr {
        // Compute the complement using utf8_automata
        let complement_sequences = compile_utf8_complement(ranges);

        // Note: We don't fall back to CodepointClass for negated classes.
        // Even if there are many sequences, the trie-based construction will share
        // common prefixes and be more efficient than CodepointClass (which requires PikeVM).
        // This allows LazyDFA and EagerDFA to handle negated Unicode classes.

        // Separate single-byte and multi-byte sequences from the complement
        let mut complement_bytes: Vec<(u8, u8)> = Vec::new();
        let mut complement_multibyte: Vec<Utf8Sequence> = Vec::new();

        for seq in complement_sequences {
            if seq.len() == 1 {
                complement_bytes.push(seq.ranges[0]);
            } else {
                complement_multibyte.push(seq);
            }
        }

        // Merge byte ranges
        complement_bytes.sort_by_key(|r| r.0);
        let merged_bytes = merge_byte_ranges(complement_bytes);

        // Build the expression using byte-level transitions (NOT negated - complement already computed)
        let mut alternatives: Vec<HirExpr> = Vec::new();

        if !merged_bytes.is_empty() {
            alternatives.push(HirExpr::Class(HirClass::new(merged_bytes, false)));
        }

        if !complement_multibyte.is_empty() {
            let trie_expr = self.build_utf8_trie(&complement_multibyte);
            alternatives.push(trie_expr);
        }

        match alternatives.len() {
            0 => HirExpr::Class(HirClass::new(vec![], false)), // Empty - matches nothing
            1 => alternatives.pop().unwrap(),
            _ => HirExpr::Alt(alternatives),
        }
    }

    /// Translates a repetition.
    fn translate_repeat(&mut self, rep: &Repeat) -> Result<HirExpr> {
        let expr = self.translate_expr(&rep.expr)?;
        // Track non-greedy quantifiers for engine selection
        if !rep.greedy {
            self.props.has_non_greedy = true;
        }
        Ok(HirExpr::Repeat(Box::new(HirRepeat {
            expr,
            min: rep.min,
            max: rep.max,
            greedy: rep.greedy,
        })))
    }

    /// Translates a group.
    fn translate_group(&mut self, group: &Group) -> Result<HirExpr> {
        // A `(?flags:...)` group applies its flags only to its body, so push them
        // while translating the inner expression, then restore.
        if let GroupKind::Flagged(flags) = &group.kind {
            let saved = self.flags;
            self.flags = *flags;
            let expr = self.translate_expr(&group.expr)?;
            self.flags = saved;
            return Ok(expr);
        }

        let expr = self.translate_expr(&group.expr)?;

        match &group.kind {
            GroupKind::Capturing(index) => {
                // Capture indices are 1-based, so capture_count = max index seen
                self.props.capture_count = self.props.capture_count.max(*index);
                Ok(HirExpr::Capture(Box::new(HirCapture {
                    index: *index,
                    name: None,
                    expr,
                })))
            }
            GroupKind::NamedCapturing { name, index } => {
                // Named groups also have numeric indices
                self.props.capture_count = self.props.capture_count.max(*index);
                self.props.named_groups.insert(name.clone(), *index);
                Ok(HirExpr::Capture(Box::new(HirCapture {
                    index: *index,
                    name: Some(name.clone()),
                    expr,
                })))
            }
            GroupKind::NonCapturing => Ok(expr),
            // Handled by the early return above; listed for exhaustiveness.
            GroupKind::Flagged(_) => Ok(expr),
        }
    }

    /// Translates a character class to HIR.
    ///
    /// For simple ASCII-only classes (code points 0-127), returns an `HirExpr::Class`.
    /// For Unicode classes with multi-byte UTF-8 sequences (any code point at or
    /// above U+0080), returns an alternation of concatenations representing the
    /// valid byte sequences.
    fn translate_class(&mut self, class: &Class) -> Result<HirExpr> {
        // Under `(?i)`, a character class matches all case variants of its members
        // (e.g. `(?i:[sdmt])` must match `M`). Single literals are folded in
        // `translate_literal`; classes are folded here by adding case-equivalent
        // code points. The complement (for `[^...]`) is taken afterward, so a
        // negated case-insensitive class excludes all variants too.
        let folded: Vec<ClassRange>;
        let ranges: &[ClassRange] = if self.flags.case_insensitive {
            let mut out = class.ranges.clone();
            for r in &class.ranges {
                let (s, e) = (r.start as u32, r.end as u32);
                // Skip very large ranges (already broad; folding adds nothing useful
                // and would be expensive to enumerate).
                if e.saturating_sub(s) >= 0x1000 {
                    continue;
                }
                // Rather than walking every codepoint in [s, e] and calling
                // `case_fold_equivalents` on each (O(range size)), binary-search
                // the sub-range of codepoints that actually participate in
                // folding. Every folding codepoint is either a `from` entry in
                // `CASE_FOLDING_SIMPLE` or a key in `CASE_FOLD_GROUPS`, and those
                // two sets are provably disjoint: Unicode simple case folding is
                // single-hop, so a `to` value never appears as a `from` (verified
                // across all 1512 current `CASE_FOLDING_SIMPLE` entries). That
                // means visiting both sorted tables' overlap with [s, e] finds
                // every codepoint in the range that has equivalents, without
                // touching the ones that don't. If a future UCD version ever
                // introduces a multi-hop fold, this assumption breaks and this
                // loop would silently skip codepoints that need folding.
                let push_variants = |cp: u32, out: &mut Vec<ClassRange>| {
                    for &fc in unicode_data::case_fold_equivalents(cp) {
                        if fc != cp {
                            if let Some(fch) = char::from_u32(fc) {
                                out.push(ClassRange::new(fch, fch));
                            }
                        }
                    }
                };

                let from_lo =
                    unicode_data::CASE_FOLDING_SIMPLE.partition_point(|&(from, _)| from < s);
                let from_hi =
                    unicode_data::CASE_FOLDING_SIMPLE.partition_point(|&(from, _)| from <= e);
                for &(cp, _) in &unicode_data::CASE_FOLDING_SIMPLE[from_lo..from_hi] {
                    push_variants(cp, &mut out);
                }

                let key_lo = unicode_data::CASE_FOLD_GROUPS.partition_point(|&(key, _, _)| key < s);
                let key_hi =
                    unicode_data::CASE_FOLD_GROUPS.partition_point(|&(key, _, _)| key <= e);
                for &(cp, _, _) in &unicode_data::CASE_FOLD_GROUPS[key_lo..key_hi] {
                    push_variants(cp, &mut out);
                }
            }
            folded = out;
            &folded
        } else {
            &class.ranges
        };

        // A negated class is built from the *complement* of its members, so the
        // UTF-8 expansion of the members themselves is never used — computing it
        // would be pure waste (`\p{L}` alone is 684 ranges pushed through
        // `compile_utf8_range` and `optimize_sequences`, all of it discarded).
        // The code points are needed either way, so both are gathered in one
        // pass over `ranges`: splitting them into two passes costs a second walk
        // of a list that is routinely several hundred entries long.
        let negated = class.negated;
        let mut codepoint_ranges: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
        let mut byte_ranges: Vec<(u8, u8)> = Vec::new();
        let mut utf8_sequences: Vec<Utf8Sequence> = Vec::new();
        for range in ranges {
            codepoint_ranges.push((range.start as u32, range.end as u32));
            if !negated {
                self.collect_class_ranges(range, &mut byte_ranges, &mut utf8_sequences);
            }
        }

        // Sort and merge codepoint ranges. This is the canonical, exact form of
        // the class — every downstream consumer that needs code points takes it
        // from here rather than re-deriving it.
        codepoint_ranges.sort_by_key(|r| r.0);
        let merged_codepoints = merge_codepoint_ranges(codepoint_ranges);

        // Whether this class is "large" is decided by how it actually lowers, not
        // by the mere presence of multi-byte members: the branches that emit a
        // codepoint node set the flag themselves. Setting it here as well would
        // push a class that lowered to a small trie off the DFA for no reason.

        let expr = if negated {
            self.build_negated_class_expr(&merged_codepoints)
        } else {
            // Sort and merge single-byte ranges
            byte_ranges.sort_by_key(|r| r.0);
            let merged_bytes = merge_byte_ranges(byte_ranges);

            // Optimize multi-byte sequences
            let optimized_seqs = optimize_sequences(utf8_sequences);

            self.build_class_expr(merged_bytes, optimized_seqs, &merged_codepoints)
        };

        // Store for potential use by CodepointClassMatcher
        self.current_class_codepoints = Some((merged_codepoints, class.negated));

        Ok(expr)
    }

    /// Collects byte ranges and UTF-8 sequences for a character range.
    fn collect_class_ranges(
        &self,
        range: &ClassRange,
        byte_ranges: &mut Vec<(u8, u8)>,
        utf8_sequences: &mut Vec<Utf8Sequence>,
    ) {
        push_codepoint_range(
            range.start as u32,
            range.end as u32,
            byte_ranges,
            utf8_sequences,
        );
    }

    /// Builds the HIR for a negated character class from the exact code points
    /// it excludes.
    ///
    /// The negated case only ever needs the *complement* of its members, so it
    /// is built straight from the merged code-point ranges the class was parsed
    /// into. That is both cheaper and more accurate than the old route of
    /// pushing the members through `compile_utf8_range`/`optimize_sequences`
    /// and then reconstructing the excluded set by decoding those sequences
    /// back (`utf8_sequence_to_code_point_range`, since deleted): that decode
    /// took the byte-range hull of a sequence, and once `optimize_sequences`
    /// has merged
    /// two sequences on a byte position whose successors are partial ranges,
    /// the hull is a strict *superset* of what the sequence encodes. For a
    /// negated class a superset means excluding characters the class never
    /// named. `build_class_expr` no longer has that branch — every negated
    /// class, bracketed or Perl shorthand, now routes through here instead.
    ///
    /// The three outcomes mirror `build_class_expr`'s non-negated ones, in the
    /// same order: an all-ASCII excluded set complements within ASCII
    /// (equivalent to that function's "no multi-byte sequences" test, since
    /// only ASCII code points ever reach its `byte_ranges`), otherwise the
    /// complement is lowered to a byte trie, and otherwise the class becomes a
    /// code-point node.
    fn build_negated_class_expr(&mut self, excluded: &[(u32, u32)]) -> HirExpr {
        if excluded.iter().all(|&(_, hi)| hi <= 0x7f) {
            let ascii: Vec<(u8, u8)> = excluded
                .iter()
                .map(|&(lo, hi)| (lo as u8, hi as u8))
                .collect();
            let surviving_ascii = merge_byte_ranges(complement_within_ascii(&ascii));
            return self.build_ascii_or_non_ascii(surviving_ascii);
        }

        if let Some(expr) = self.lower_complement_to_bytes(excluded) {
            return expr;
        }

        // Mark as large unicode class for engine selection.
        self.props.has_large_unicode_class = true;
        HirExpr::UnicodeCpClass(CodepointClass::new(excluded.to_vec(), true))
    }

    /// Builds the final HIR expression for a non-negated character class.
    ///
    /// Every negated class — bracketed (`[^...]`) or a Unicode-mode Perl
    /// shorthand (`\S`, `\W`, `\D`, `\H`) — is built by `build_negated_class_expr`
    /// instead, straight from the exact excluded code points; this function no
    /// longer has a negated caller (see `translate_ranges_to_hir`), so it no
    /// longer takes a `negated` parameter.
    fn build_class_expr(
        &mut self,
        byte_ranges: Vec<(u8, u8)>,
        utf8_sequences: Vec<Utf8Sequence>,
        exact_codepoints: &[(u32, u32)],
    ) -> HirExpr {
        // A class with multi-byte members becomes a codepoint node once its UTF-8
        // expansion stops being cheap. Below that bound the trie is the better
        // representation: it holds only *complete* sequences, so it still cannot
        // match at a partial codepoint, and unlike the codepoint node it does not
        // pin the whole pattern to the PikeVM — a small class like `\s` would
        // otherwise drag every pattern containing it onto the slowest engine.
        // Unless the pattern is pinned there regardless, in which case the trie
        // has nothing to win and the codepoint node is the cheaper node. Only a
        // class with multi-byte members has that choice to make: a pure-ASCII
        // one is a plain byte class either way, and routing it through a
        // codepoint node would strand it on the slower engines for nothing.
        if utf8_sequences.len() > MAX_TRIE_SEQUENCES
            || (self.engine_already_pinned && !utf8_sequences.is_empty())
        {
            return self.build_unicode_codepoint_class(exact_codepoints);
        }

        let mut alternatives: Vec<HirExpr> = Vec::new();

        if !byte_ranges.is_empty() {
            alternatives.push(HirExpr::Class(HirClass::new(byte_ranges, false)));
        }

        // Build multi-byte sequences as a trie to share common prefixes.
        // This dramatically reduces NFA state count for large Unicode classes.
        if !utf8_sequences.is_empty() {
            let trie_expr = self.build_utf8_trie(&utf8_sequences);
            alternatives.push(trie_expr);
        }

        // Return the appropriate expression
        match alternatives.len() {
            0 => {
                // Empty class - matches nothing
                // Return an empty class which will never match
                HirExpr::Class(HirClass::new(vec![], false))
            }
            1 => alternatives.pop().unwrap(),
            _ => HirExpr::Alt(alternatives),
        }
    }

    /// Expands "any character except `excluded`" into a byte-level automaton, or
    /// `None` when the expansion would be too large to be worth it.
    ///
    /// The bound is on the number of UTF-8 sequences, which is what the NFA and
    /// DFA actually pay for — a code-point count would reject `[^\s]`, whose
    /// complement spans most of Unicode yet expands to only a few sequences.
    ///
    /// Also `None` once the engine is pinned regardless — see
    /// [`pins_codepoint_engine`].
    fn lower_complement_to_bytes(&self, excluded: &[(u32, u32)]) -> Option<HirExpr> {
        if self.engine_already_pinned {
            return None;
        }
        let complement = compile_utf8_complement(excluded);
        if complement.len() > MAX_TRIE_SEQUENCES {
            return None;
        }

        // Single-byte sequences are ASCII and belong in a plain byte class; the
        // rest form the trie.
        let mut byte_ranges = Vec::new();
        let mut sequences = Vec::new();
        for seq in complement {
            match &seq.ranges[..] {
                [single] => byte_ranges.push(*single),
                _ => sequences.push(seq),
            }
        }

        let mut alternatives: Vec<HirExpr> = Vec::new();
        if !byte_ranges.is_empty() {
            alternatives.push(HirExpr::Class(HirClass::new(
                merge_byte_ranges(byte_ranges),
                false,
            )));
        }
        if !sequences.is_empty() {
            alternatives.push(self.build_utf8_trie(&sequences));
        }
        match alternatives.len() {
            0 => Some(HirExpr::Class(HirClass::new(vec![], false))),
            1 => alternatives.pop(),
            _ => Some(HirExpr::Alt(alternatives)),
        }
    }

    /// Builds `.` — "any single character", not "any single byte".
    ///
    /// Written the same way a negated ASCII class is (see `build_class_expr`):
    /// an alternation of "one surviving ASCII byte" and "any non-ASCII
    /// character", the second half being a trie of complete UTF-8 sequences.
    /// That keeps `.` an ordinary byte automaton — LazyDfa, EagerDfa, Shift-Or
    /// and the JIT all run it natively — while making it impossible for a match
    /// to start or end in the middle of a codepoint.
    ///
    /// Deliberately *not* a `HirExpr::UnicodeCpClass`: that node forces the
    /// tagged NFA (see `engine::selector::hir_uses_codepoint_class`), which
    /// would drag every pattern containing `.` off the byte engines that run it
    /// best — and onto the PikeVM whenever step extraction declines. The trie is
    /// three fixed sequences, so `has_large_unicode_class` stays unset too.
    ///
    /// `\n` is 0x0A, which is never a UTF-8 lead or continuation byte, so
    /// excluding it from the ASCII half alone gives the non-dot-all semantics.
    fn build_dot_expr(&mut self, dot_all: bool) -> HirExpr {
        let ascii: Vec<(u8, u8)> = if dot_all {
            vec![(0x00, 0x7f)]
        } else {
            vec![(0x00, 0x09), (0x0b, 0x7f)]
        };
        self.build_ascii_or_non_ascii(ascii)
    }

    /// Builds `\R` — any Unicode line-break sequence, matched as one unit:
    /// the two-character `\r\n` sequence if present, else any single
    /// line-break character (LF, VT, FF, CR, NEL, LS, PS).
    ///
    /// PCRE makes `\R` atomic, so once it commits to matching `\r\n` it can
    /// never back off to just the `\r`. This engine has no atomic groups yet,
    /// so the same observable behavior is built as an ordered alternation
    /// with the two-character branch listed first: both branches can start
    /// on `\r`, so `engine::selector::hir_has_alternation` detects the
    /// overlap and routes the pattern to the PikeVM, which honors
    /// leftmost-first branch priority and therefore always prefers the
    /// longer `\r\n` branch over the bare `\r` when both are viable.
    fn build_line_break_expr(&mut self) -> Result<HirExpr> {
        let crlf = HirExpr::Literal(vec![b'\r', b'\n']);
        let single = self.translate_ranges_to_hir(LINE_BREAK_SINGLE, false)?;
        Ok(HirExpr::Alt(vec![crlf, single]))
    }

    /// Builds "one byte from `ascii_ranges`, or any whole non-ASCII
    /// character" as a byte-level alternation.
    ///
    /// Shared shape behind every construct that is negated (or otherwise
    /// open-ended) over characters but must still compile to a byte
    /// automaton: `build_negated_class_expr`'s negated-ASCII branch (`[^a]`,
    /// and Unicode-mode `\S`/`\W`/`\D`/`\H`), `build_dot_expr` (`.`), and
    /// ASCII-mode negated Perl classes (`\W`,
    /// `\D`) in `translate_perl_class_ascii`. The non-ASCII half is a trie of
    /// complete UTF-8 sequences (see `any_non_ascii_character`), so a match
    /// can never start or end in the middle of a codepoint.
    fn build_ascii_or_non_ascii(&self, ascii_ranges: Vec<(u8, u8)>) -> HirExpr {
        let non_ascii = any_non_ascii_character();

        let mut alternatives = Vec::new();
        if !ascii_ranges.is_empty() {
            alternatives.push(HirExpr::Class(HirClass::new(ascii_ranges, false)));
        }
        if !non_ascii.is_empty() {
            alternatives.push(self.build_utf8_trie(&non_ascii));
        }
        match alternatives.len() {
            0 => HirExpr::Class(HirClass::new(vec![], false)),
            1 => alternatives.pop().unwrap(),
            _ => HirExpr::Alt(alternatives),
        }
    }

    /// Builds a trie-based HIR expression for UTF-8 sequences.
    /// This shares common prefixes to minimize NFA states.
    ///
    /// Leading ranges are split at every boundary they share before grouping, so
    /// the resulting branches start on pairwise-disjoint bytes. Two sequences
    /// compiled from different code-point ranges routinely share a leading byte
    /// without sharing a leading *range* (`C2-C3` and `C3-DF`); grouping on the
    /// range alone would leave two branches competing for `C3`, which reads as a
    /// real alternation and sends the whole pattern to the PikeVM.
    #[allow(clippy::only_used_in_recursion)]
    fn build_utf8_trie(&self, sequences: &[Utf8Sequence]) -> HirExpr {
        if sequences.is_empty() {
            return HirExpr::Empty;
        }

        // Group sequences by their first byte range, split into atoms so the
        // groups partition the byte space instead of overlapping.
        let atoms = leading_range_atoms(sequences);
        let mut groups: std::collections::BTreeMap<(u8, u8), Vec<Utf8Sequence>> =
            std::collections::BTreeMap::new();

        for seq in sequences {
            let Some(&(lo, hi)) = seq.ranges.first() else {
                continue;
            };
            for &atom in atoms.iter().filter(|(a, b)| *a >= lo && *b <= hi) {
                groups
                    .entry(atom)
                    .or_default()
                    .push(Utf8Sequence::new(&seq.ranges[1..]));
            }
        }

        // Build alternatives for each group
        let mut alternatives: Vec<HirExpr> = Vec::new();

        for ((lo, hi), suffixes) in groups {
            let first_class = HirExpr::Class(HirClass::new(vec![(lo, hi)], false));

            if suffixes.is_empty() || suffixes.iter().all(|s| s.ranges.is_empty()) {
                // Single-byte sequences or all suffixes are empty
                alternatives.push(first_class);
            } else {
                // Filter out empty suffixes and recurse
                let non_empty: Vec<_> = suffixes
                    .into_iter()
                    .filter(|s| !s.ranges.is_empty())
                    .collect();

                if non_empty.is_empty() {
                    alternatives.push(first_class);
                } else {
                    let suffix_expr = self.build_utf8_trie(&non_empty);
                    alternatives.push(HirExpr::Concat(vec![first_class, suffix_expr]));
                }
            }
        }

        match alternatives.len() {
            0 => HirExpr::Empty,
            1 => alternatives.pop().unwrap(),
            _ => HirExpr::Alt(alternatives),
        }
    }

    /// Builds a Unicode codepoint class for efficient matching.
    ///
    /// Negated classes never reach here: `build_negated_class_expr` handles
    /// those directly from exact codepoint ranges (see `translate_ranges_to_hir`
    /// and `translate_class`), so this function's sole caller (`build_class_expr`)
    /// only ever has non-negated members to lower.
    ///
    /// Takes the exact, already-merged code-point ranges the class was parsed
    /// from — not reconstructed by decoding `utf8_sequences` back through their
    /// byte-range hull. That decode (formerly `utf8_sequence_to_code_point_range`)
    /// takes the min/max byte of each position and treats the result as one
    /// contiguous interval, which over-approximates once `optimize_sequences`
    /// has merged sequences leaving more than one byte position with a partial
    /// range — for a non-negated class that means matching code points the
    /// class never named. Passing the exact ranges through from the caller
    /// (which already has them, from `merged_codepoints` / the `ranges`
    /// parameter / a Unicode property's table) sidesteps the lossy decode
    /// entirely.
    fn build_unicode_codepoint_class(&mut self, exact_codepoints: &[(u32, u32)]) -> HirExpr {
        // Mark as large unicode class for engine selection
        self.props.has_large_unicode_class = true;

        // Return as UnicodeCpClass - the Thompson compiler will handle this efficiently
        // Instead of expanding to thousands of byte-level alternations, we use a single
        // state that checks codepoint membership using binary search.
        HirExpr::UnicodeCpClass(CodepointClass::new(exact_codepoints.to_vec(), false))
    }

    /// Translates a lookaround.
    fn translate_lookaround(&mut self, la: &Lookaround) -> Result<HirExpr> {
        self.props.has_lookaround = true;
        let expr = self.translate_expr(&la.expr)?;
        let kind = match la.kind {
            LookaroundKind::PositiveLookahead => HirLookaroundKind::PositiveLookahead,
            LookaroundKind::NegativeLookahead => HirLookaroundKind::NegativeLookahead,
            LookaroundKind::PositiveLookbehind => HirLookaroundKind::PositiveLookbehind,
            LookaroundKind::NegativeLookbehind => HirLookaroundKind::NegativeLookbehind,
        };
        Ok(HirExpr::Lookaround(Box::new(HirLookaround { expr, kind })))
    }

    /// Checks if an HIR expression represents a simple Unicode character class.
    /// A simple class is one that can be efficiently matched by CodepointClassMatcher:
    /// - A single HirExpr::Class
    /// - An Alt of Class and/or Concat (representing UTF-8 byte sequences)
    ///
    /// This excludes patterns with quantifiers, backrefs, lookarounds, etc.
    fn is_simple_unicode_class(expr: &HirExpr) -> bool {
        match expr {
            // Simple byte class
            HirExpr::Class(_) => true,
            // Alternation of byte sequences (UTF-8 encoded character class)
            HirExpr::Alt(alts) => {
                alts.iter().all(|alt| {
                    match alt {
                        HirExpr::Class(_) => true,
                        HirExpr::Concat(parts) => {
                            // Concat of Literals/Classes represents multi-byte UTF-8 sequence
                            parts
                                .iter()
                                .all(|p| matches!(p, HirExpr::Class(_) | HirExpr::Literal(_)))
                        }
                        _ => false,
                    }
                })
            }
            _ => false,
        }
    }
}

impl Default for HirTranslator {
    fn default() -> Self {
        Self::new()
    }
}

/// Merges overlapping byte ranges.
fn merge_byte_ranges(mut ranges: Vec<(u8, u8)>) -> Vec<(u8, u8)> {
    if ranges.is_empty() {
        return ranges;
    }

    ranges.sort_by_key(|r| r.0);

    let mut merged = vec![ranges[0]];

    for range in ranges.into_iter().skip(1) {
        let last = merged.last_mut().unwrap();
        if range.0 <= last.1.saturating_add(1) {
            last.1 = last.1.max(range.1);
        } else {
            merged.push(range);
        }
    }

    merged
}

/// Merges overlapping or adjacent codepoint ranges.
fn merge_codepoint_ranges(mut ranges: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    if ranges.is_empty() {
        return ranges;
    }

    ranges.sort_by_key(|r| r.0);

    let mut merged = vec![ranges[0]];

    for range in ranges.into_iter().skip(1) {
        let last = merged.last_mut().unwrap();
        if range.0 <= last.1.saturating_add(1) {
            last.1 = last.1.max(range.1);
        } else {
            merged.push(range);
        }
    }

    merged
}

/// Splits a codepoint range into single ASCII bytes and/or UTF-8 sequences.
///
/// Codepoints are ASCII bytes only up to U+007F; a byte value equals its
/// codepoint only in that range. Anything at or above U+0080 must go through
/// `compile_utf8_range` so it is encoded as UTF-8 instead of being truncated
/// to a raw byte (e.g. U+00E9 'é' is the two bytes 0xC3 0xA9, not the single
/// byte 0xE9). Single-byte results from `compile_utf8_range` (pure ASCII)
/// are folded into `byte_ranges` alongside the direct ASCII portion.
/// Partitions the leading byte ranges of `sequences` into the coarsest set of
/// pairwise-disjoint ranges that refines all of them.
fn leading_range_atoms(sequences: &[Utf8Sequence]) -> Vec<(u8, u8)> {
    let mut cuts: Vec<u16> = Vec::with_capacity(sequences.len() * 2);
    for seq in sequences {
        if let Some(&(lo, hi)) = seq.ranges.first() {
            cuts.push(lo as u16);
            cuts.push(hi as u16 + 1);
        }
    }
    cuts.sort_unstable();
    cuts.dedup();

    let mut atoms = Vec::with_capacity(cuts.len());
    for pair in cuts.windows(2) {
        let (Some(&start), Some(&end)) = (pair.first(), pair.get(1)) else {
            continue;
        };
        // `end` is exclusive and both came from `u8` bounds, so the cast holds.
        if let (Ok(lo), Ok(hi)) = (u8::try_from(start), u8::try_from(end - 1)) {
            atoms.push((lo, hi));
        }
    }
    atoms
}

/// Largest UTF-8 expansion a class may have before it becomes a codepoint node
/// instead. Past this the trie costs more NFA states than the codepoint check
/// costs in engine restrictions; below it the byte automaton wins outright —
/// but only when the trie can actually buy a faster engine, which is what
/// [`pins_codepoint_engine`] decides.
const MAX_TRIE_SEQUENCES: usize = 64;

/// Whether something other than class lowering already confines the pattern to
/// an engine that runs codepoint nodes natively.
///
/// A byte trie costs NFA states and buys exactly one thing: eligibility for the
/// byte engines, which cannot execute a codepoint node. A lookaround or a
/// non-greedy quantifier pins the pattern to the tagged NFA or PikeVM, which run
/// codepoint nodes directly, so there the eligibility is unreachable and the
/// extra states are pure cost — the class stays a codepoint node.
///
/// Deliberately narrower than the full set of early returns in
/// [`select_engine_from_hir`](crate::engine::selector::select_engine_from_hir):
/// an alternation also routes to PikeVM, but its classes still feed literal
/// prefilters and the JIT's own selection path, so leaving those lowered to
/// bytes measures better than pinning them. Read off the AST because the HIR
/// equivalents are not populated yet. A lookaround pins the engine whatever it
/// contains, so its body needs no further inspection.
///
/// A backreference runs the other way. Selection tests it first and routes to
/// the backtracker, which cannot execute a codepoint node at all — so a
/// backreference anywhere makes byte lowering mandatory, whatever else the
/// pattern contains.
/// Whether the pattern is exactly one bracket class, allowing non-capturing
/// and flag-scoping groups around it. A capturing group is excluded because
/// the codepoint-class matcher reports group 0 only.
fn is_single_class(expr: &Expr) -> bool {
    match expr {
        Expr::Class(_) => true,
        Expr::Group(group) => {
            matches!(group.kind, GroupKind::NonCapturing | GroupKind::Flagged(_))
                && is_single_class(&group.expr)
        }
        Expr::Concat(exprs) | Expr::Alt(exprs) => {
            exprs.len() == 1 && exprs.iter().all(is_single_class)
        }
        _ => false,
    }
}

fn pins_codepoint_engine(expr: &Expr) -> bool {
    !contains_backref(expr) && pins_without_backref(expr)
}

fn pins_without_backref(expr: &Expr) -> bool {
    match expr {
        Expr::Alt(branches) => branches.iter().any(pins_without_backref),
        Expr::Lookaround(_) => true,
        Expr::Repeat(repeat) => !repeat.greedy || pins_without_backref(&repeat.expr),
        Expr::Concat(exprs) => exprs.iter().any(pins_without_backref),
        Expr::Group(group) => pins_without_backref(&group.expr),
        Expr::Backref(_)
        | Expr::Empty
        | Expr::Literal(_)
        | Expr::Class(_)
        | Expr::Anchor(_)
        | Expr::Dot
        | Expr::GraphemeCluster
        | Expr::UnicodeProperty { .. }
        | Expr::PerlClass(_)
        | Expr::LineBreak
        | Expr::AnyExceptNewline => false,
    }
}

/// Whether a backreference appears anywhere, including inside a lookaround
/// body — the backtracker compiles the whole pattern, lookarounds and all.
fn contains_backref(expr: &Expr) -> bool {
    match expr {
        Expr::Backref(_) => true,
        Expr::Alt(branches) => branches.iter().any(contains_backref),
        Expr::Concat(exprs) => exprs.iter().any(contains_backref),
        Expr::Repeat(repeat) => contains_backref(&repeat.expr),
        Expr::Group(group) => contains_backref(&group.expr),
        Expr::Lookaround(lookaround) => contains_backref(&lookaround.expr),
        Expr::Empty
        | Expr::Literal(_)
        | Expr::Class(_)
        | Expr::Anchor(_)
        | Expr::Dot
        | Expr::GraphemeCluster
        | Expr::UnicodeProperty { .. }
        | Expr::PerlClass(_)
        | Expr::LineBreak
        | Expr::AnyExceptNewline => false,
    }
}

fn push_codepoint_range(
    start_cp: u32,
    end_cp: u32,
    byte_ranges: &mut Vec<(u8, u8)>,
    utf8_sequences: &mut Vec<Utf8Sequence>,
) {
    if start_cp <= 127 && end_cp <= 127 {
        byte_ranges.push((start_cp as u8, end_cp as u8));
        return;
    }

    let utf8_start = if start_cp <= 127 {
        byte_ranges.push((start_cp as u8, 127));
        128
    } else {
        start_cp
    };

    for seq in compile_utf8_range(utf8_start, end_cp) {
        if seq.len() == 1 {
            byte_ranges.push(seq.ranges[0]);
        } else {
            utf8_sequences.push(seq);
        }
    }
}

/// "Any non-ASCII character", as UTF-8 byte sequences.
///
/// Three shapes rather than an exact enumeration of the scalar-value ranges.
/// The exact form has to carve out surrogates and overlong encodings, which
/// costs hundreds of DFA states per class — `[^x]+y` took seconds to compile —
/// and buys nothing here: the public API matches `&str`, so the haystack is
/// already valid UTF-8 and those sequences cannot occur in it. `C0`/`C1` are
/// still excluded because they cannot begin any well-formed character.
fn any_non_ascii_character() -> Vec<Utf8Sequence> {
    vec![
        Utf8Sequence::new(&[(0xc2, 0xdf), (0x80, 0xbf)]),
        Utf8Sequence::new(&[(0xe0, 0xef), (0x80, 0xbf), (0x80, 0xbf)]),
        Utf8Sequence::new(&[(0xf0, 0xf4), (0x80, 0xbf), (0x80, 0xbf), (0x80, 0xbf)]),
    ]
}

/// The ASCII bytes a negated class still admits: `0x00..=0x7f` minus `excluded`.
fn complement_within_ascii(excluded: &[(u8, u8)]) -> Vec<(u8, u8)> {
    let mut sorted = excluded.to_vec();
    sorted.sort_unstable();

    let mut out = Vec::new();
    let mut next = 0u16;
    for (lo, hi) in sorted {
        if lo as u16 > next {
            out.push((next as u8, lo - 1));
        }
        next = next.max(hi as u16 + 1);
    }
    if next <= 0x7f {
        out.push((next as u8, 0x7f));
    }
    out
}

/// Largest automaton one pattern may ask the engines to build.
///
/// Every engine compiles `{n,m}` by emitting the subexpression `m` times, so the
/// work a pattern costs is its *expanded* size, not its text length. Bounding
/// each `{n,m}` on its own is not enough because nested repetitions multiply:
/// `(?:a{1000}){1000}` spells out a million elements from two legal bounds.
///
/// Calibrated against compile time, which runs at roughly 14 us per element:
/// this bounds `Regex::new` at about 150 ms for the worst pattern it accepts.
/// It is far more permissive than the `regex` crate, whose default size limit
/// refuses `\w{1000}` outright. A caller who knows their own workload can raise
/// it with [`crate::RegexBuilder::size_limit`].
pub const DEFAULT_EXPANDED_SIZE: u32 = 10_000;

/// How many elements the engines will emit for this expression.
///
/// Invariant: every leaf costs at least what the NFA builders (`nfa::thompson`)
/// will actually allocate for it, so no node is ever free. This matters even for
/// nodes that duplicate no *text* — `Empty`, `Anchor`, `Backref` still cost one or
/// more NFA states per copy — because a zero-cost leaf lets `Repeat` multiply it
/// by any copy count and still report zero, letting `(?:(?:\b){N}){M}` sail past
/// the limit while emitting `N * M` states. When adding a new `HirExpr` variant,
/// give it the state count its builder allocates, never `0`.
///
/// Above the leaves, a class or a literal byte is one element, and a repetition
/// multiplies its body by the number of copies it forces. Saturating throughout,
/// so an overflowing product reports the ceiling and is rejected rather than
/// wrapping to a small number.
fn expanded_size(expr: &HirExpr) -> u32 {
    match expr {
        // 1 state: `nfa::thompson::build_empty`.
        HirExpr::Empty => 1,
        // 1 state: `nfa::thompson::build_anchor`.
        HirExpr::Anchor(_) => 1,
        // 2 states: `nfa::thompson::build_backref`.
        HirExpr::Backref(_) => 2,
        HirExpr::Class(_) | HirExpr::UnicodeCpClass(_) => 1,
        HirExpr::Literal(bytes) => u32::try_from(bytes.len()).unwrap_or(u32::MAX),
        HirExpr::Concat(exprs) | HirExpr::Alt(exprs) => exprs
            .iter()
            .fold(0u32, |total, e| total.saturating_add(expanded_size(e))),
        HirExpr::Capture(capture) => expanded_size(&capture.expr),
        HirExpr::Lookaround(look) => expanded_size(&look.expr),
        // An unbounded repetition is a loop, not a duplication: only the copies
        // the minimum forces are emitted, plus one for the loop body itself.
        HirExpr::Repeat(repeat) => {
            let copies = repeat.max.unwrap_or(repeat.min).max(1);
            expanded_size(&repeat.expr).saturating_mul(copies)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn test_translate_literal() {
        let ast = parse("abc").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        assert!(matches!(hir.expr, HirExpr::Concat(_)));
    }

    #[test]
    fn test_translate_class() {
        let ast = parse("[a-z]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        if let HirExpr::Class(cls) = hir.expr {
            assert_eq!(cls.ranges, vec![(b'a', b'z')]);
        } else {
            panic!("Expected Class");
        }
    }

    /// No leaf `HirExpr` may cost 0 elements: `Repeat` multiplies a body's
    /// `expanded_size` by its copy count, so a zero-cost leaf lets any number
    /// of copies collapse back to zero and evade `DEFAULT_EXPANDED_SIZE`
    /// entirely — which is exactly what let
    /// `(?:(?:\b){65535}){65535}` emit ~4.3e9 NFA states for free.
    ///
    /// A newly added variant cannot slip past this silently: `expanded_size`
    /// matches without a wildcard arm, so it fails to compile until the variant
    /// is given the state count its builder allocates.
    #[test]
    fn expanded_size_never_returns_zero_for_a_leaf() {
        let leaves = [
            HirExpr::Empty,
            HirExpr::Literal(vec![b'a']),
            HirExpr::Class(HirClass::new(vec![(b'a', b'z')], false)),
            HirExpr::UnicodeCpClass(CodepointClass::new(vec![(0x61, 0x7a)], false)),
            HirExpr::Anchor(HirAnchor::WordBoundary),
            HirExpr::Backref(1),
        ];

        for leaf in &leaves {
            assert!(
                expanded_size(leaf) > 0,
                "{leaf:?} must cost at least one NFA state, matching what \
                 nfa::thompson actually allocates for it"
            );
        }
    }

    #[test]
    fn test_merge_ranges() {
        let ranges = vec![(1, 3), (2, 5), (7, 9)];
        let merged = merge_byte_ranges(ranges);
        assert_eq!(merged, vec![(1, 5), (7, 9)]);
    }

    /// Whether any node in the tree is a code-point class, i.e. the class did
    /// not lower to a byte automaton.
    fn contains_codepoint_class(expr: &HirExpr) -> bool {
        match expr {
            HirExpr::UnicodeCpClass(_) => true,
            HirExpr::Concat(exprs) | HirExpr::Alt(exprs) => {
                exprs.iter().any(contains_codepoint_class)
            }
            HirExpr::Repeat(repeat) => contains_codepoint_class(&repeat.expr),
            HirExpr::Capture(capture) => contains_codepoint_class(&capture.expr),
            HirExpr::Lookaround(look) => contains_codepoint_class(&look.expr),
            HirExpr::Empty
            | HirExpr::Literal(_)
            | HirExpr::Class(_)
            | HirExpr::Anchor(_)
            | HirExpr::Backref(_) => false,
        }
    }

    #[test]
    fn test_translate_full_codepoint_range() {
        // [\x00-\xff] covers code points U+0000-U+00FF. Everything at or above
        // U+0080 is multi-byte UTF-8, not a raw byte, so this is an ASCII byte
        // class alternated with a trie of two-byte sequences — small enough to
        // stay a byte automaton rather than becoming a code-point node.
        let ast = parse("[\\x00-\\xff]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        assert!(
            !contains_codepoint_class(&hir.expr),
            "small class should lower to bytes, got {:?}",
            hir.expr
        );

        let re = crate::Regex::new("^[\\x00-\\xff]$").unwrap();
        assert!(re.is_match("\u{0}"));
        assert!(re.is_match("é"));
        assert!(!re.is_match("Ā"), "U+0100 is outside the class");
    }

    #[test]
    fn test_translate_high_codepoint_range() {
        // [\x80-\xff] covers U+0080-U+00FF, all above the ASCII cutoff, so it is
        // a trie of the two-byte sequences 0xC2 0x80 .. 0xC3 0xBF — never the raw
        // bytes 128-255, which are not characters.
        let ast = parse("[\\x80-\\xff]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        assert!(
            !contains_codepoint_class(&hir.expr),
            "small class should lower to bytes, got {:?}",
            hir.expr
        );

        let re = crate::Regex::new("^[\\x80-\\xff]$").unwrap();
        assert!(re.is_match("\u{80}"));
        assert!(re.is_match("ÿ"));
        assert!(!re.is_match("a"));
        assert!(!re.is_match("Ā"));
    }

    #[test]
    fn test_translate_unicode_class_greek() {
        // [α-ω] lowers to a trie of complete three-byte sequences, so it matches
        // only at code-point boundaries while staying on the byte engines.
        let ast = parse("[α-ω]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        assert!(
            !contains_codepoint_class(&hir.expr),
            "small Unicode range should lower to bytes, got {:?}",
            hir.expr
        );

        let re = crate::Regex::new("^[α-ω]$").unwrap();
        assert!(re.is_match("α"));
        assert!(re.is_match("ω"));
        assert!(!re.is_match("Α"), "uppercase alpha is outside the range");
        assert!(!re.is_match("a"));
    }

    #[test]
    fn test_translate_unicode_single_char() {
        // A single multi-byte member is just its own UTF-8 sequence.
        let ast = parse("[α]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        assert!(
            !contains_codepoint_class(&hir.expr),
            "single multi-byte char should lower to bytes, got {:?}",
            hir.expr
        );

        let re = crate::Regex::new("^[α]$").unwrap();
        assert!(re.is_match("α"));
        assert!(!re.is_match("β"));
    }

    #[test]
    fn test_translate_mixed_ascii_unicode() {
        // [a-zα-ω] should match both ASCII and Greek letters
        // Use the Regex API which selects the correct engine (PikeVM for CodepointClass)
        let re = crate::Regex::new("[a-zα-ω]").unwrap();

        // ASCII letters should match
        assert!(re.is_match("a"));
        assert!(re.is_match("z"));
        // Greek letters should match
        assert!(re.is_match("α"));
        assert!(re.is_match("ω"));
        // Non-matching characters
        assert!(!re.is_match("A"));
        assert!(!re.is_match("1"));
    }

    #[test]
    fn test_translate_emoji_class() {
        // [😀-😂] should match emoji correctly using UnicodeCpClass
        // Use the Regex API which selects the correct engine (PikeVM for CodepointClass)
        let re = crate::Regex::new("[😀-😂]").unwrap();

        assert!(re.is_match("😀"));
        assert!(re.is_match("😁"));
        assert!(re.is_match("😂"));
        // Should not match emoji outside the range
        assert!(!re.is_match("a"));
    }

    #[test]
    fn test_backref_validation() {
        // Valid backrefs
        let ast = parse(r"(a)\1").unwrap();
        let result = HirTranslator::new().translate(&ast);
        assert!(result.is_ok(), "Valid backref \\1 with 1 group should work");

        let ast = parse(r"(a)(b)\1\2").unwrap();
        let result = HirTranslator::new().translate(&ast);
        assert!(
            result.is_ok(),
            "Valid backrefs \\1\\2 with 2 groups should work"
        );

        // Invalid backrefs - reference non-existent groups
        let ast = parse(r"\1").unwrap();
        let result = HirTranslator::new().translate(&ast);
        assert!(result.is_err(), "Backref \\1 with no groups should fail");

        let ast = parse(r"(a)\2").unwrap();
        let result = HirTranslator::new().translate(&ast);
        assert!(result.is_err(), "Backref \\2 with only 1 group should fail");
    }

    #[test]
    fn test_named_groups_tracking() {
        // Test that named groups are tracked in props
        let ast = parse(r"(?<word>\w+)").unwrap();
        println!("AST: {:?}", ast);
        let hir = HirTranslator::new().translate(&ast).unwrap();
        println!("HIR props: {:?}", hir.props);
        println!("Named groups: {:?}", hir.props.named_groups);
        assert_eq!(hir.props.named_groups.len(), 1);
        assert_eq!(hir.props.named_groups.get("word"), Some(&1));

        // Python-style
        let ast = parse(r"(?P<foo>\d+)").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        assert_eq!(hir.props.named_groups.len(), 1);
        assert_eq!(hir.props.named_groups.get("foo"), Some(&1));

        // Multiple named groups
        let ast = parse(r"(?<a>\w)(?<b>\d)").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        assert_eq!(hir.props.named_groups.len(), 2);
        assert_eq!(hir.props.named_groups.get("a"), Some(&1));
        assert_eq!(hir.props.named_groups.get("b"), Some(&2));
    }

    /// The flag must track whether a code-point node was actually emitted.
    ///
    /// It routes a pattern away from the DFA and Shift-Or engines, so setting it
    /// for a class that lowered to a small byte trie is a large, silent
    /// slowdown: `\s+` was taking the TaggedNfa path rather than Shift-Or purely
    /// because of this flag.
    #[test]
    fn test_small_perl_classes_are_not_large_unicode() {
        for pattern in [r"\s", r"\S", r"\s+", r"\S+", r"\h", r"\H"] {
            let ast = parse(pattern).unwrap();
            let hir = HirTranslator::new().translate(&ast).unwrap();
            assert!(
                !hir.props.has_large_unicode_class,
                "{pattern} lowers to a byte trie and must not be flagged large"
            );
            assert!(
                !contains_codepoint_class(&hir.expr),
                "{pattern} should contain no code-point node"
            );
        }
    }

    #[test]
    fn test_large_unicode_class_detection() {
        // Unicode properties should be detected as large
        let ast = parse(r"\p{Han}").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        assert!(
            hir.props.has_large_unicode_class,
            "\\p{{Han}} should be detected as large unicode class"
        );

        // ASCII-only classes should NOT be detected as large
        let ast = parse(r"[a-z]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        assert!(
            !hir.props.has_large_unicode_class,
            "[a-z] should not be large"
        );

        // Multi-byte membership alone no longer makes a class "large": the Greek
        // range lowers to a small trie, and flagging it would push every pattern
        // containing it off the DFA for nothing.
        let ast = parse(r"[α-ω]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        assert!(
            !hir.props.has_large_unicode_class,
            "[α-ω] lowers to a small trie and should not be flagged large"
        );
    }

    /// Proves the binary-searched range enumeration in `translate_class`
    /// (Step 3 of the `case_fold_equivalents` fast path) produces exactly the
    /// same set of case-folded variants as the naive per-codepoint walk it
    /// replaced. This is the test that actually protects the optimization:
    /// it recomputes the old O(range size) approach independently and checks
    /// the two agree, rather than restating the new logic.
    #[test]
    fn test_case_fold_range_enumeration_matches_brute_force() {
        use std::collections::HashSet;

        /// Old approach: walk every codepoint in `[s, e]` individually.
        fn brute_force(s: u32, e: u32) -> HashSet<u32> {
            let mut out = HashSet::new();
            for cp in s..=e {
                for &fc in unicode_data::case_fold_equivalents(cp).iter() {
                    if fc != cp {
                        out.insert(fc);
                    }
                }
            }
            out
        }

        /// New approach: binary-search the sub-ranges of `CASE_FOLDING_SIMPLE`
        /// and `CASE_FOLD_GROUPS` that overlap `[s, e]`, mirroring the
        /// production code in `translate_class`.
        fn binary_searched(s: u32, e: u32) -> HashSet<u32> {
            let mut out = HashSet::new();

            let from_lo = unicode_data::CASE_FOLDING_SIMPLE.partition_point(|&(from, _)| from < s);
            let from_hi = unicode_data::CASE_FOLDING_SIMPLE.partition_point(|&(from, _)| from <= e);
            for &(cp, _) in &unicode_data::CASE_FOLDING_SIMPLE[from_lo..from_hi] {
                for &fc in unicode_data::case_fold_equivalents(cp).iter() {
                    if fc != cp {
                        out.insert(fc);
                    }
                }
            }

            let key_lo = unicode_data::CASE_FOLD_GROUPS.partition_point(|&(key, _, _)| key < s);
            let key_hi = unicode_data::CASE_FOLD_GROUPS.partition_point(|&(key, _, _)| key <= e);
            for &(cp, _, _) in &unicode_data::CASE_FOLD_GROUPS[key_lo..key_hi] {
                for &fc in unicode_data::case_fold_equivalents(cp).iter() {
                    if fc != cp {
                        out.insert(fc);
                    }
                }
            }

            out
        }

        let representative_ranges: &[(u32, u32, &str)] = &[
            (0x0041, 0x005A, "ASCII (A-Z)"),
            (0x0391, 0x03A9, "Greek (Α-Ω)"),
            (0x0410, 0x042F, "Cyrillic (А-Я)"),
            (0x0030, 0x0039, "no folding (0-9)"),
            // Straddles the U+03B8 (θ) group key, which sits between two
            // "from" entries (U+0398 -> U+03B8, U+03D1 -> U+03B8).
            (0x03B5, 0x03BB, "range straddling a group key"),
        ];

        for &(s, e, label) in representative_ranges {
            let brute = brute_force(s, e);
            let fast = binary_searched(s, e);
            assert_eq!(
                brute, fast,
                "case folding mismatch for {label} (U+{s:04X}-U+{e:04X}): \
                 brute-force and binary-searched enumeration disagree"
            );
        }
    }

    /// `translate_ranges_to_hir`'s negated case must exclude exactly the code
    /// points it is given, not the byte-range hull of whatever UTF-8 sequences
    /// they happen to encode to.
    ///
    /// This is the same gap-inducing range set as
    /// `test_negated_class_excludes_only_the_codepoints_it_names` in
    /// `tests/unicode/negated.rs` (U+0810-U+083F and U+0850-U+087F, whose
    /// sequences `E0 A0 90-BF` / `E0 A1 90-BF` merge into `E0 A0-A1 90-BF`, a
    /// hull that also spans U+0840-U+084F), but driven through
    /// `translate_ranges_to_hir` directly. That is the only route this crate
    /// has to a negated class built from an arbitrary code-point range set —
    /// its sole real caller, `translate_perl_class_unicode` (`\S`/`\W`/`\D`/
    /// `\H`), only ever passes fixed UCD-derived tables that do not merge
    /// lossily today, so a pattern-level test could not exercise this path's
    /// hull-decode hazard. Reverting `translate_ranges_to_hir`'s negated
    /// branch to route through `build_class_expr` instead of
    /// `build_negated_class_expr` makes this test fail by wrongly excluding
    /// U+0840.
    #[test]
    fn translate_ranges_to_hir_negated_excludes_only_named_codepoints() {
        let ranges = [(0x810u32, 0x83F), (0x850, 0x87F)];
        let mut translator = HirTranslator::new();
        let expr = translator.translate_ranges_to_hir(&ranges, true).unwrap();
        let hir = Hir {
            expr,
            props: translator.props.clone(),
        };
        let compiled = crate::engine::compile_from_hir(&hir).unwrap();

        let encode = |cp: u32| char::from_u32(cp).unwrap().to_string();

        assert!(
            compiled.is_match(encode(0x840).as_bytes()),
            "U+0840 lies between the excluded ranges and must not be excluded"
        );
        assert!(
            compiled.is_match(encode(0x84F).as_bytes()),
            "U+084F lies between the excluded ranges and must not be excluded"
        );

        for cp in [0x810u32, 0x820, 0x83F, 0x850, 0x86F, 0x87F] {
            assert!(
                !compiled.is_match(encode(cp).as_bytes()),
                "U+{cp:04X} is named by the class and must be excluded"
            );
        }

        // Either side of the whole span.
        assert!(compiled.is_match(encode(0x80F).as_bytes()));
        assert!(compiled.is_match(encode(0x880).as_bytes()));
    }

    /// The non-negated counterpart of the test above: `build_class_expr`'s
    /// `UnicodeCpClass` fallback (`build_unicode_codepoint_class`) must match
    /// exactly the code points it is given, not the byte-range hull of
    /// whatever UTF-8 sequences they happen to encode to.
    ///
    /// This is the same gap-inducing range set (U+0810-U+083F and
    /// U+0850-U+087F, whose sequences `E0 A0 90-BF` / `E0 A1 90-BF` merge into
    /// `E0 A0-A1 90-BF`, a hull that also spans U+0840-U+084F), but for a
    /// non-negated class a hull means *over-matching*: accepting the gap code
    /// points the class never named.
    ///
    /// That fallback is reached only once the class's optimized UTF-8
    /// sequences exceed `MAX_TRIE_SEQUENCES`, or once something else has
    /// already pinned the pattern to a codepoint-capable engine — this range
    /// set optimizes to only 2 sequences, far under the trie cap, so no
    /// pattern built through the public API reaches the fallback with it.
    /// This test drives `translate_ranges_to_hir` directly and pins the
    /// engine by hand (`engine_already_pinned = true`, mirroring what
    /// `pins_codepoint_engine` would set for e.g. a lookaround elsewhere in
    /// the same pattern) to force the fallback while keeping the range set
    /// itself small enough to hand-verify.
    #[test]
    fn translate_ranges_to_hir_unicode_cp_class_fallback_matches_only_named_codepoints() {
        let ranges = [(0x810u32, 0x83F), (0x850, 0x87F)];
        let mut translator = HirTranslator::new();
        translator.engine_already_pinned = true;
        let expr = translator.translate_ranges_to_hir(&ranges, false).unwrap();

        // Confirm the fallback was actually taken, not the small-trie path.
        assert!(
            matches!(expr, HirExpr::UnicodeCpClass(_)),
            "expected the UnicodeCpClass fallback to be taken with engine_already_pinned set"
        );

        let hir = Hir {
            expr,
            props: translator.props.clone(),
        };
        let compiled = crate::engine::compile_from_hir(&hir).unwrap();

        let encode = |cp: u32| char::from_u32(cp).unwrap().to_string();

        // The gap between the two named ranges: must NOT match — the old hull
        // decode wrongly swallowed it.
        assert!(
            !compiled.is_match(encode(0x840).as_bytes()),
            "U+0840 lies in the gap between the named ranges and must not match"
        );
        assert!(
            !compiled.is_match(encode(0x84F).as_bytes()),
            "U+084F lies in the gap between the named ranges and must not match"
        );

        // The named ranges themselves, including both edges: must match.
        for cp in [0x810u32, 0x820, 0x83F, 0x850, 0x86F, 0x87F] {
            assert!(
                compiled.is_match(encode(cp).as_bytes()),
                "U+{cp:04X} is named by the class and must match"
            );
        }

        // Either side of the whole span: must not match.
        assert!(!compiled.is_match(encode(0x80F).as_bytes()));
        assert!(!compiled.is_match(encode(0x880).as_bytes()));
    }
}
