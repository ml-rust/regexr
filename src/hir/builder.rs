//! HIR builder - translates AST to HIR.

use crate::error::{Error, ErrorKind, Result};
use crate::nfa::utf8_automata::{
    compile_utf8_complement, compile_utf8_range, optimize_sequences, Utf8Sequence,
};
use crate::parser::{
    Anchor, Ast, Class, ClassRange, Expr, Flags, Group, GroupKind, Lookaround, LookaroundKind,
    PerlClassKind, Repeat,
};

use super::unicode_data;

use super::{
    CodepointClass, Hir, HirAnchor, HirCapture, HirClass, HirExpr, HirLookaround,
    HirLookaroundKind, HirProps, HirRepeat,
};

/// Translator from AST to HIR.
pub struct HirTranslator {
    props: HirProps,
    flags: Flags,
    /// Maximum backreference index used in the pattern.
    max_backref: u32,
    /// Tracks codepoint ranges during class translation for potential fast matching.
    /// Set during translate_class, consumed by translate if pattern is a simple class.
    current_class_codepoints: Option<(Vec<(u32, u32)>, bool)>,
}

impl HirTranslator {
    /// Creates a new translator.
    pub fn new() -> Self {
        Self {
            props: HirProps::default(),
            flags: Flags::default(),
            max_backref: 0,
            current_class_codepoints: None,
        }
    }

    /// Translates an AST to HIR.
    pub fn translate(&mut self, ast: &Ast) -> Result<Hir> {
        self.flags = ast.flags;
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

        // Only use CodepointClassMatcher if the pattern is a single character class
        // (no quantifiers, no concatenation, no alternation at the top level).
        // Check that the expression is a Class or an Alt of byte sequences.
        if let Some((ranges, negated)) = self.current_class_codepoints.take() {
            // Only use codepoint_class if the root expression looks like a char class
            let is_simple_class = Self::is_simple_unicode_class(&expr);
            if is_simple_class {
                self.props.codepoint_class = Some(CodepointClass::new(ranges, negated));
            }
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

            Expr::Dot => {
                // Dot matches any byte except newline
                Ok(HirExpr::Class(HirClass::dot()))
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
                        HirAnchor::Start
                    }
                    Anchor::EndOfString | Anchor::EndOfInput => {
                        self.props.has_anchors = true;
                        HirAnchor::End
                    }
                    Anchor::StartOfLine => {
                        self.props.has_anchors = true;
                        HirAnchor::StartLine
                    }
                    Anchor::EndOfLine => {
                        self.props.has_anchors = true;
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
                    Anchor::EndOfInputBeforeNewline => {
                        self.props.has_anchors = true;
                        HirAnchor::End
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
        if self.flags.unicode {
            // Unicode mode - use full Unicode properties
            self.translate_perl_class_unicode(kind)
        } else {
            // ASCII mode - use ASCII-only ranges
            self.translate_perl_class_ascii(kind)
        }
    }

    /// Translates a Perl class in ASCII mode.
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
        };

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
        };

        // Unicode Perl classes can have many code points, causing DFA state explosion.
        // Mark large classes to fall back to PikeVM.
        // Negated classes (\D, \W, \S) cover almost all of Unicode, so always mark as large.
        let total_codepoints: u32 = ranges.iter().map(|(s, e)| e - s + 1).sum();
        let has_large_range = ranges.iter().any(|(s, e)| e - s > 500);
        if negated || total_codepoints > 1000 || has_large_range {
            self.props.has_large_unicode_class = true;
        }

        self.translate_ranges_to_hir(ranges, negated)
    }

    /// Converts code point ranges to HIR expression.
    fn translate_ranges_to_hir(&mut self, ranges: &[(u32, u32)], negated: bool) -> Result<HirExpr> {
        let mut byte_ranges: Vec<(u8, u8)> = Vec::new();
        let mut utf8_sequences: Vec<Utf8Sequence> = Vec::new();

        for &(start, end) in ranges {
            if start <= 127 && end <= 127 {
                byte_ranges.push((start as u8, end as u8));
            } else if start <= 127 {
                byte_ranges.push((start as u8, 127));
                let sequences = compile_utf8_range(128, end);
                for seq in sequences {
                    if seq.len() == 1 {
                        byte_ranges.push(seq.ranges[0]);
                    } else {
                        utf8_sequences.push(seq);
                    }
                }
            } else {
                let sequences = compile_utf8_range(start, end);
                for seq in sequences {
                    if seq.len() == 1 {
                        byte_ranges.push(seq.ranges[0]);
                    } else {
                        utf8_sequences.push(seq);
                    }
                }
            }
        }

        byte_ranges.sort_by_key(|r| r.0);
        let merged_bytes = merge_byte_ranges(byte_ranges);
        let optimized_seqs = optimize_sequences(utf8_sequences);

        // Any multi-byte UTF-8 sequences cause slow DFA materialization.
        // Mark as large unicode class to skip DFA JIT.
        if !optimized_seqs.is_empty() {
            self.props.has_large_unicode_class = true;
        }

        Ok(self.build_class_expr(merged_bytes, optimized_seqs, negated))
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
            // Only ASCII code points (0-127) can be treated as single bytes.
            // Code points 128-255 require 2-byte UTF-8 encoding (C2 80 to C3 BF).
            if start <= 127 && end <= 127 {
                // Pure ASCII range - single bytes
                byte_ranges.push((start as u8, end as u8));
            } else if start <= 127 {
                // Range starts in ASCII but extends beyond
                // Add ASCII portion as single bytes
                byte_ranges.push((start as u8, 127));
                // Use UTF-8 automata for the rest
                let sequences = compile_utf8_range(128, end);
                for seq in sequences {
                    if seq.len() == 1 {
                        byte_ranges.push(seq.ranges[0]);
                    } else {
                        utf8_sequences.push(seq);
                    }
                }
            } else {
                // Entirely non-ASCII - use UTF-8 automata
                let sequences = compile_utf8_range(start, end);
                for seq in sequences {
                    if seq.len() == 1 {
                        byte_ranges.push(seq.ranges[0]);
                    } else {
                        utf8_sequences.push(seq);
                    }
                }
            }
        }

        // Sort and merge byte ranges
        byte_ranges.sort_by_key(|r| r.0);
        let merged_bytes = merge_byte_ranges(byte_ranges);

        // Optimize UTF-8 sequences
        let optimized_seqs = optimize_sequences(utf8_sequences);

        // Build the final expression (non-negated path)
        Ok(self.build_class_expr(merged_bytes, optimized_seqs, false))
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
        }
    }

    /// Translates a character class to HIR.
    ///
    /// For simple byte-range classes (code points 0-255), returns an `HirExpr::Class`.
    /// For Unicode classes with multi-byte UTF-8 sequences, returns an alternation
    /// of concatenations representing the valid byte sequences.
    fn translate_class(&mut self, class: &Class) -> Result<HirExpr> {
        let mut byte_ranges: Vec<(u8, u8)> = Vec::new();
        let mut utf8_sequences: Vec<Utf8Sequence> = Vec::new();

        // Collect codepoint ranges for potential fast matching
        let mut codepoint_ranges: Vec<(u32, u32)> = Vec::new();
        for range in &class.ranges {
            codepoint_ranges.push((range.start as u32, range.end as u32));
            self.collect_class_ranges(range, &mut byte_ranges, &mut utf8_sequences);
        }

        // Sort and merge codepoint ranges
        codepoint_ranges.sort_by_key(|r| r.0);
        let merged_codepoints = merge_codepoint_ranges(codepoint_ranges);

        // Store for potential use by CodepointClassMatcher
        self.current_class_codepoints = Some((merged_codepoints, class.negated));

        // Sort and merge single-byte ranges
        byte_ranges.sort_by_key(|r| r.0);
        let merged_bytes = merge_byte_ranges(byte_ranges);

        // Optimize multi-byte sequences
        let optimized_seqs = optimize_sequences(utf8_sequences);

        // Any multi-byte UTF-8 sequences cause slow DFA materialization.
        // Mark as large unicode class to skip DFA JIT.
        if !optimized_seqs.is_empty() {
            self.props.has_large_unicode_class = true;
        }

        // Build the final expression
        let expr = self.build_class_expr(merged_bytes, optimized_seqs, class.negated);
        Ok(expr)
    }

    /// Collects byte ranges and UTF-8 sequences for a character range.
    fn collect_class_ranges(
        &self,
        range: &ClassRange,
        byte_ranges: &mut Vec<(u8, u8)>,
        utf8_sequences: &mut Vec<Utf8Sequence>,
    ) {
        let start_cp = range.start as u32;
        let end_cp = range.end as u32;

        // Special case: if both endpoints are in 0-255, treat as literal bytes
        // This preserves backward compatibility for patterns like [\x00-\xff]
        if start_cp <= 255 && end_cp <= 255 {
            byte_ranges.push((start_cp as u8, end_cp as u8));
            return;
        }

        // Use UTF-8 automata for the full range
        let sequences = compile_utf8_range(start_cp, end_cp);

        for seq in sequences {
            if seq.len() == 1 {
                // Single-byte sequence goes into byte_ranges
                byte_ranges.push(seq.ranges[0]);
            } else {
                // Multi-byte sequence
                utf8_sequences.push(seq);
            }
        }
    }

    /// Builds the final HIR expression for a character class.
    fn build_class_expr(
        &mut self,
        byte_ranges: Vec<(u8, u8)>,
        utf8_sequences: Vec<Utf8Sequence>,
        negated: bool,
    ) -> HirExpr {
        // If negated and we have multi-byte sequences, compute the complement
        if negated && !utf8_sequences.is_empty() {
            return self.build_negated_unicode_class(byte_ranges, utf8_sequences);
        }

        // With trie-based construction, we can handle larger Unicode classes efficiently
        // because common UTF-8 prefixes are shared. Only fall back to CodepointClass
        // for extremely large classes where even the trie would be slow.
        //
        // Note: CodepointClass requires PikeVM which is ~10x slower than LazyDFA.
        // The trie approach keeps everything in LazyDFA-compatible byte transitions.
        if utf8_sequences.len() > 5000 {
            return self.build_unicode_codepoint_class(byte_ranges, utf8_sequences, negated);
        }

        let mut alternatives: Vec<HirExpr> = Vec::new();

        // Add single-byte class if we have byte ranges
        if !byte_ranges.is_empty() {
            alternatives.push(HirExpr::Class(HirClass::new(byte_ranges, negated)));
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

    /// Builds a trie-based HIR expression for UTF-8 sequences.
    /// This shares common prefixes to minimize NFA states.
    fn build_utf8_trie(&self, sequences: &[Utf8Sequence]) -> HirExpr {
        if sequences.is_empty() {
            return HirExpr::Empty;
        }

        // Group sequences by their first byte range
        let mut groups: std::collections::BTreeMap<(u8, u8), Vec<Utf8Sequence>> =
            std::collections::BTreeMap::new();

        for seq in sequences {
            if seq.ranges.is_empty() {
                continue;
            }
            let first = seq.ranges[0];
            groups
                .entry(first)
                .or_default()
                .push(Utf8Sequence::new(seq.ranges[1..].to_vec()));
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

    /// Builds a negated Unicode class using CodepointClass.
    ///
    /// Uses CodepointClass for efficient runtime matching. The CodepointClass
    /// instruction decodes UTF-8 and checks codepoint membership directly,
    /// avoiding the need to materialize the full UTF-8 complement automaton.
    fn build_negated_unicode_class(
        &mut self,
        byte_ranges: Vec<(u8, u8)>,
        utf8_sequences: Vec<Utf8Sequence>,
    ) -> HirExpr {
        // Convert byte ranges and UTF-8 sequences to codepoint ranges
        let mut codepoint_ranges: Vec<(u32, u32)> = Vec::new();

        // Add codepoints from byte ranges (ASCII: 0-127)
        for (start, end) in &byte_ranges {
            codepoint_ranges.push((*start as u32, *end as u32));
        }

        // Convert UTF-8 sequences back to codepoint ranges
        for seq in &utf8_sequences {
            if let Some(range) = self.utf8_sequence_to_code_point_range(seq) {
                codepoint_ranges.push(range);
            }
        }

        // Sort and merge ranges
        codepoint_ranges.sort_by_key(|r| r.0);
        let merged = merge_codepoint_ranges(codepoint_ranges);

        // Mark as large unicode class for engine selection
        self.props.has_large_unicode_class = true;

        // Return as UnicodeCpClass with negated=true
        // The CodepointClass instruction handles negation directly
        HirExpr::UnicodeCpClass(CodepointClass::new(merged, true))
    }

    /// Builds a Unicode codepoint class for efficient matching.
    /// Works for both negated and non-negated large Unicode classes.
    fn build_unicode_codepoint_class(
        &mut self,
        byte_ranges: Vec<(u8, u8)>,
        utf8_sequences: Vec<Utf8Sequence>,
        negated: bool,
    ) -> HirExpr {
        // Mark as large unicode class for engine selection
        self.props.has_large_unicode_class = true;

        // Convert byte ranges and UTF-8 sequences back to code point ranges
        let mut code_point_ranges = Vec::new();

        // Add code points from byte ranges (these are in 0-255)
        for (start, end) in byte_ranges {
            code_point_ranges.push((start as u32, end as u32));
        }

        // Convert UTF-8 sequences back to code point ranges
        for seq in utf8_sequences {
            if let Some(range) = self.utf8_sequence_to_code_point_range(&seq) {
                code_point_ranges.push(range);
            }
        }

        // Sort and merge ranges
        code_point_ranges.sort_by_key(|r| r.0);
        let merged = merge_codepoint_ranges(code_point_ranges);

        // Return as UnicodeCpClass - the Thompson compiler will handle this efficiently
        // Instead of expanding to thousands of byte-level alternations, we use a single
        // state that checks codepoint membership using binary search.
        HirExpr::UnicodeCpClass(CodepointClass::new(merged, negated))
    }

    /// Attempts to convert a UTF-8 sequence back to a code point range.
    /// This is a best-effort approximation for sequences with variable ranges.
    fn utf8_sequence_to_code_point_range(&self, seq: &Utf8Sequence) -> Option<(u32, u32)> {
        // Decode the start and end code points from the byte ranges
        match seq.len() {
            1 => {
                let (start, end) = seq.ranges[0];
                Some((start as u32, end as u32))
            }
            2 => {
                // 2-byte UTF-8: 110xxxxx 10xxxxxx
                let (b1_start, b1_end) = seq.ranges[0];
                let (b2_start, b2_end) = seq.ranges[1];

                let start = (((b1_start & 0x1F) as u32) << 6) | ((b2_start & 0x3F) as u32);
                let end = (((b1_end & 0x1F) as u32) << 6) | ((b2_end & 0x3F) as u32);

                Some((start, end))
            }
            3 => {
                // 3-byte UTF-8: 1110xxxx 10xxxxxx 10xxxxxx
                let (b1_start, b1_end) = seq.ranges[0];
                let (b2_start, b2_end) = seq.ranges[1];
                let (b3_start, b3_end) = seq.ranges[2];

                let start = (((b1_start & 0x0F) as u32) << 12)
                    | (((b2_start & 0x3F) as u32) << 6)
                    | ((b3_start & 0x3F) as u32);
                let end = (((b1_end & 0x0F) as u32) << 12)
                    | (((b2_end & 0x3F) as u32) << 6)
                    | ((b3_end & 0x3F) as u32);

                Some((start, end))
            }
            4 => {
                // 4-byte UTF-8: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                let (b1_start, b1_end) = seq.ranges[0];
                let (b2_start, b2_end) = seq.ranges[1];
                let (b3_start, b3_end) = seq.ranges[2];
                let (b4_start, b4_end) = seq.ranges[3];

                let start = (((b1_start & 0x07) as u32) << 18)
                    | (((b2_start & 0x3F) as u32) << 12)
                    | (((b3_start & 0x3F) as u32) << 6)
                    | ((b4_start & 0x3F) as u32);
                let end = (((b1_end & 0x07) as u32) << 18)
                    | (((b2_end & 0x3F) as u32) << 12)
                    | (((b3_end & 0x3F) as u32) << 6)
                    | ((b4_end & 0x3F) as u32);

                Some((start, end))
            }
            _ => None,
        }
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

    #[test]
    fn test_merge_ranges() {
        let ranges = vec![(1, 3), (2, 5), (7, 9)];
        let merged = merge_byte_ranges(ranges);
        assert_eq!(merged, vec![(1, 5), (7, 9)]);
    }

    #[test]
    fn test_translate_full_byte_range() {
        // [\x00-\xff] should match all 256 byte values
        let ast = parse("[\\x00-\\xff]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        if let HirExpr::Class(cls) = hir.expr {
            assert_eq!(cls.ranges, vec![(0, 255)]);
        } else {
            panic!("Expected Class");
        }
    }

    #[test]
    fn test_translate_high_byte_range() {
        // [\x80-\xff] should match bytes 128-255
        let ast = parse("[\\x80-\\xff]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        if let HirExpr::Class(cls) = hir.expr {
            assert_eq!(cls.ranges, vec![(0x80, 0xff)]);
        } else {
            panic!("Expected Class");
        }
    }

    #[test]
    fn test_translate_unicode_class_greek() {
        // [α-ω] should produce alternation with 2-byte UTF-8 sequences
        let ast = parse("[α-ω]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();

        // Greek lowercase letters are 2-byte UTF-8 (U+03B1-U+03C9)
        // Should produce an Alt with multiple Concat sequences
        match hir.expr {
            HirExpr::Alt(alts) => {
                // All alternatives should be 2-byte sequences (Concat of 2 elements)
                for alt in &alts {
                    match alt {
                        HirExpr::Concat(parts) => {
                            assert_eq!(parts.len(), 2, "Greek letters are 2-byte UTF-8");
                        }
                        _ => panic!("Expected Concat for 2-byte UTF-8"),
                    }
                }
            }
            _ => panic!("Expected Alt for Unicode class, got {:?}", hir.expr),
        }
    }

    #[test]
    fn test_translate_unicode_single_char() {
        // [α] should produce a single 2-byte sequence
        let ast = parse("[α]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();

        // Single Greek letter - should be Concat of 2 literals/classes
        match hir.expr {
            HirExpr::Concat(parts) => {
                assert_eq!(parts.len(), 2);
            }
            _ => panic!("Expected Concat for single 2-byte char, got {:?}", hir.expr),
        }
    }

    #[test]
    fn test_translate_mixed_ascii_unicode() {
        // [a-zα-ω] should match both ASCII and Greek letters
        let ast = parse("[a-zα-ω]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();

        // Compile to NFA and verify it matches correctly (functional test)
        let nfa = crate::nfa::compile(&Hir {
            expr: hir.expr,
            props: hir.props,
        })
        .unwrap();

        let mut dfa = crate::dfa::LazyDfa::new(nfa);
        // ASCII letters should match
        assert!(dfa.is_match_bytes(b"a"));
        assert!(dfa.is_match_bytes(b"z"));
        // Greek letters should match
        assert!(dfa.is_match_bytes("α".as_bytes()));
        assert!(dfa.is_match_bytes("ω".as_bytes()));
        // Non-matching characters
        assert!(!dfa.is_match_bytes(b"A"));
        assert!(!dfa.is_match_bytes(b"1"));
    }

    #[test]
    fn test_translate_emoji_class() {
        // [😀-😂] should produce UTF-8 byte transitions that can match 4-byte emoji
        let ast = parse("[😀-😂]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();

        // Compile to NFA and verify it matches correctly
        let nfa = crate::nfa::compile(&Hir {
            expr: hir.expr,
            props: hir.props,
        })
        .unwrap();

        // The NFA should be able to match these emoji (functional test)
        let mut dfa = crate::dfa::LazyDfa::new(nfa);
        assert!(dfa.is_match_bytes("😀".as_bytes()));
        assert!(dfa.is_match_bytes("😁".as_bytes()));
        assert!(dfa.is_match_bytes("😂".as_bytes()));
        // Should not match emoji outside the range
        assert!(!dfa.is_match_bytes("a".as_bytes()));
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

        // Greek range uses multi-byte UTF-8, should be flagged to skip DFA JIT
        let ast = parse(r"[α-ω]").unwrap();
        let hir = HirTranslator::new().translate(&ast).unwrap();
        assert!(
            hir.props.has_large_unicode_class,
            "[α-ω] has multi-byte UTF-8, should skip DFA JIT"
        );
    }
}
