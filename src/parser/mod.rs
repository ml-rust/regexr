//! Regex parser module.
//!
//! Implements a recursive descent parser for regular expressions.

pub mod ast;
pub mod lexer;

pub use ast::*;
pub use lexer::{EscapeKind, Lexer, Token, TokenKind};

use crate::error::{Error, ErrorKind, Result, Span};
use crate::hir::unicode_data;

/// Parses a regex pattern into an AST.
pub fn parse(pattern: &str) -> Result<Ast> {
    let mut parser = Parser::new(pattern);
    parser.parse()
}

/// The regex parser.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    /// Current token.
    current: Token,
    /// Pattern being parsed.
    pattern: &'a str,
    /// Next capture group index.
    next_capture: u32,
    /// Total number of capture groups.
    capture_count: u32,
    /// Current flags.
    flags: Flags,
}

impl<'a> Parser<'a> {
    /// Creates a new parser.
    pub fn new(pattern: &'a str) -> Self {
        let mut lexer = Lexer::new(pattern);
        let current = lexer.next_token().unwrap_or(Token {
            kind: TokenKind::Eof,
            span: Span::point(0),
        });

        Self {
            lexer,
            current,
            pattern,
            next_capture: 1,
            capture_count: 0,
            flags: Flags::default(),
        }
    }

    /// Parses the pattern.
    pub fn parse(&mut self) -> Result<Ast> {
        let expr = self.parse_alternation()?;

        if !self.is_at_end() {
            return Err(Error::with_span(
                ErrorKind::UnexpectedChar(self.current_char().unwrap_or('?')),
                self.pattern,
                self.current.span,
            ));
        }

        Ok(Ast {
            expr,
            flags: self.flags,
        })
    }

    /// Advances to the next token.
    fn advance(&mut self) -> Result<Token> {
        let prev = std::mem::replace(&mut self.current, self.lexer.next_token()?);
        Ok(prev)
    }

    /// Returns true if we're at the end of input.
    fn is_at_end(&self) -> bool {
        matches!(self.current.kind, TokenKind::Eof)
    }

    /// Returns the current character if it's a literal.
    fn current_char(&self) -> Option<char> {
        match &self.current.kind {
            TokenKind::Literal(c) => Some(*c),
            _ => None,
        }
    }

    /// Checks if the current token matches the given kind.
    fn check(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.current.kind) == std::mem::discriminant(kind)
    }

    /// Consumes the current token if it matches, otherwise returns an error.
    fn expect(&mut self, kind: TokenKind) -> Result<Token> {
        if self.check(&kind) {
            self.advance()
        } else {
            Err(Error::with_span(
                ErrorKind::UnexpectedChar(self.current_char().unwrap_or('?')),
                self.pattern,
                self.current.span,
            ))
        }
    }

    /// Parses alternation (lowest precedence): a|b|c
    fn parse_alternation(&mut self) -> Result<Expr> {
        let mut left = self.parse_concat()?;

        if matches!(self.current.kind, TokenKind::Pipe) {
            let mut alternatives = vec![left];

            while matches!(self.current.kind, TokenKind::Pipe) {
                self.advance()?;
                alternatives.push(self.parse_concat()?);
            }

            left = Expr::Alt(alternatives);
        }

        Ok(left)
    }

    /// Parses concatenation: abc
    fn parse_concat(&mut self) -> Result<Expr> {
        let mut exprs = Vec::new();

        while !self.is_at_end() && !self.is_concat_terminator() {
            exprs.push(self.parse_repeat()?);
        }

        Ok(match exprs.len() {
            0 => Expr::Empty,
            1 => exprs.pop().unwrap(),
            _ => Expr::Concat(exprs),
        })
    }

    /// Returns true if the current token terminates concatenation.
    fn is_concat_terminator(&self) -> bool {
        matches!(
            self.current.kind,
            TokenKind::Pipe | TokenKind::CloseParen | TokenKind::Eof
        )
    }

    /// Parses repetition: a*, a+, a?, a{n,m}
    fn parse_repeat(&mut self) -> Result<Expr> {
        let expr = self.parse_atom()?;
        self.parse_quantifier(expr)
    }

    /// Parses a quantifier if present.
    fn parse_quantifier(&mut self, expr: Expr) -> Result<Expr> {
        let (min, max) = match &self.current.kind {
            TokenKind::Star => {
                self.advance()?;
                (0, None)
            }
            TokenKind::Plus => {
                self.advance()?;
                (1, None)
            }
            TokenKind::Question => {
                self.advance()?;
                (0, Some(1))
            }
            TokenKind::OpenBrace => {
                self.advance()?;
                let (min, max) = self.parse_repetition_range()?;
                self.expect(TokenKind::CloseBrace)?;
                (min, max)
            }
            _ => return Ok(expr),
        };

        // Check for non-greedy modifier (? after quantifier)
        let greedy = if matches!(self.current.kind, TokenKind::Question) {
            self.advance()?;
            false
        } else {
            true
        };

        // Check for nested quantifier (*, +, {n} after a quantifier)
        // Note: This comes AFTER handling non-greedy ?, so *? is allowed
        if matches!(
            self.current.kind,
            TokenKind::Star | TokenKind::Plus | TokenKind::Question | TokenKind::OpenBrace
        ) {
            return Err(Error::with_span(
                ErrorKind::NestedQuantifier,
                self.pattern,
                self.current.span,
            ));
        }

        Ok(Expr::Repeat(Box::new(Repeat::new(expr, min, max, greedy))))
    }

    /// Parses repetition range: {n}, {n,}, {n,m}
    fn parse_repetition_range(&mut self) -> Result<(u32, Option<u32>)> {
        // Parse first number (min)
        let mut min = 0u32;
        while let TokenKind::Digit(d) = self.current.kind {
            min = min.saturating_mul(10).saturating_add(d);
            self.advance()?;
        }

        // Check for {n}
        if matches!(self.current.kind, TokenKind::CloseBrace) {
            return Ok((min, Some(min)));
        }

        // Expect comma
        if !matches!(self.current.kind, TokenKind::Comma) {
            return Err(Error::with_span(
                ErrorKind::InvalidRepetition,
                self.pattern,
                self.current.span,
            ));
        }
        self.advance()?; // consume comma

        // Check for {n,}
        if matches!(self.current.kind, TokenKind::CloseBrace) {
            return Ok((min, None));
        }

        // Parse second number (max)
        let mut max = 0u32;
        let mut has_max = false;
        while let TokenKind::Digit(d) = self.current.kind {
            has_max = true;
            max = max.saturating_mul(10).saturating_add(d);
            self.advance()?;
        }

        if !has_max {
            return Err(Error::with_span(
                ErrorKind::InvalidRepetition,
                self.pattern,
                self.current.span,
            ));
        }

        if max < min {
            return Err(Error::with_span(
                ErrorKind::InvalidRepetition,
                self.pattern,
                self.current.span,
            ));
        }

        Ok((min, Some(max)))
    }

    /// Parses an atom (the smallest unit).
    fn parse_atom(&mut self) -> Result<Expr> {
        match &self.current.kind {
            TokenKind::Literal(c) => {
                let c = *c;
                self.advance()?;
                Ok(Expr::Literal(c))
            }
            TokenKind::Dot => {
                self.advance()?;
                Ok(Expr::Dot)
            }
            TokenKind::Caret => {
                self.advance()?;
                let anchor = if self.flags.multi_line {
                    Anchor::StartOfLine
                } else {
                    Anchor::StartOfString
                };
                Ok(Expr::Anchor(anchor))
            }
            TokenKind::Dollar => {
                self.advance()?;
                let anchor = if self.flags.multi_line {
                    Anchor::EndOfLine
                } else {
                    Anchor::EndOfString
                };
                Ok(Expr::Anchor(anchor))
            }
            TokenKind::OpenParen => self.parse_group(),
            TokenKind::OpenBracket => self.parse_class(),
            TokenKind::Escape(esc) => self.parse_escape(esc.clone()),
            TokenKind::Digit(d) => {
                let c = char::from_digit(*d, 10).unwrap();
                self.advance()?;
                Ok(Expr::Literal(c))
            }
            // These tokens are only special inside (?...) constructs,
            // but as regular atoms they are literal characters
            TokenKind::Equals => {
                self.advance()?;
                Ok(Expr::Literal('='))
            }
            TokenKind::Exclamation => {
                self.advance()?;
                Ok(Expr::Literal('!'))
            }
            TokenKind::LessThan => {
                self.advance()?;
                Ok(Expr::Literal('<'))
            }
            TokenKind::GreaterThan => {
                self.advance()?;
                Ok(Expr::Literal('>'))
            }
            TokenKind::Colon => {
                self.advance()?;
                Ok(Expr::Literal(':'))
            }
            TokenKind::Comma => {
                self.advance()?;
                Ok(Expr::Literal(','))
            }
            TokenKind::CloseParen => Err(Error::with_span(
                ErrorKind::UnmatchedCloseParen,
                self.pattern,
                self.current.span,
            )),
            TokenKind::Star | TokenKind::Plus | TokenKind::Question => Err(Error::with_span(
                ErrorKind::RepetitionOnNothing,
                self.pattern,
                self.current.span,
            )),
            _ => Err(Error::with_span(
                ErrorKind::UnexpectedChar(self.current_char().unwrap_or('?')),
                self.pattern,
                self.current.span,
            )),
        }
    }

    /// Parses an escape sequence.
    fn parse_escape(&mut self, esc: EscapeKind) -> Result<Expr> {
        self.advance()?;

        match esc {
            EscapeKind::Literal(c) => Ok(Expr::Literal(c)),
            EscapeKind::Newline => Ok(Expr::Literal('\n')),
            EscapeKind::CarriageReturn => Ok(Expr::Literal('\r')),
            EscapeKind::Tab => Ok(Expr::Literal('\t')),
            EscapeKind::FormFeed => Ok(Expr::Literal('\x0C')),
            EscapeKind::VerticalTab => Ok(Expr::Literal('\x0B')),
            EscapeKind::Null => Ok(Expr::Literal('\0')),
            EscapeKind::Hex(c) | EscapeKind::Unicode(c) => Ok(Expr::Literal(c)),

            EscapeKind::Digit => Ok(Expr::PerlClass(PerlClassKind::Digit)),
            EscapeKind::NotDigit => Ok(Expr::PerlClass(PerlClassKind::NotDigit)),
            EscapeKind::Word => Ok(Expr::PerlClass(PerlClassKind::Word)),
            EscapeKind::NotWord => Ok(Expr::PerlClass(PerlClassKind::NotWord)),
            EscapeKind::Whitespace => Ok(Expr::PerlClass(PerlClassKind::Whitespace)),
            EscapeKind::NotWhitespace => Ok(Expr::PerlClass(PerlClassKind::NotWhitespace)),

            EscapeKind::WordBoundary => Ok(Expr::Anchor(Anchor::WordBoundary)),
            EscapeKind::NotWordBoundary => Ok(Expr::Anchor(Anchor::NotWordBoundary)),
            EscapeKind::StartOfInput => Ok(Expr::Anchor(Anchor::StartOfInput)),
            EscapeKind::EndOfInput => Ok(Expr::Anchor(Anchor::EndOfInput)),
            EscapeKind::EndOfInputBeforeNewline => {
                Ok(Expr::Anchor(Anchor::EndOfInputBeforeNewline))
            }

            EscapeKind::Backref(n) => Ok(Expr::Backref(n)),

            EscapeKind::UnicodeProperty(name) => Ok(Expr::UnicodeProperty {
                name,
                negated: false,
            }),
            EscapeKind::NotUnicodeProperty(name) => Ok(Expr::UnicodeProperty {
                name,
                negated: true,
            }),
        }
    }

    /// Parses a group: (...), (?:...), (?=...), etc.
    fn parse_group(&mut self) -> Result<Expr> {
        let start_span = self.current.span;
        self.advance()?; // consume '('

        // Check for special group syntax
        if matches!(self.current.kind, TokenKind::Question) {
            self.advance()?;

            match &self.current.kind {
                // Non-capturing group (?:...)
                TokenKind::Colon => {
                    self.advance()?;
                    let expr = self.parse_alternation()?;
                    self.expect_close_paren(start_span)?;
                    Ok(Expr::Group(Box::new(Group {
                        expr,
                        kind: GroupKind::NonCapturing,
                    })))
                }

                // Named group (?<name>...) or lookbehind (?<=...) (?<!...)
                TokenKind::LessThan => {
                    self.advance()?;

                    // Check if it's lookbehind or named group
                    match &self.current.kind {
                        TokenKind::Equals => {
                            // (?<=...) positive lookbehind
                            self.advance()?;
                            let expr = self.parse_alternation()?;
                            self.expect_close_paren(start_span)?;
                            Ok(Expr::Lookaround(Box::new(Lookaround {
                                expr,
                                kind: LookaroundKind::PositiveLookbehind,
                            })))
                        }
                        TokenKind::Exclamation => {
                            // (?<!...) negative lookbehind
                            self.advance()?;
                            let expr = self.parse_alternation()?;
                            self.expect_close_paren(start_span)?;
                            Ok(Expr::Lookaround(Box::new(Lookaround {
                                expr,
                                kind: LookaroundKind::NegativeLookbehind,
                            })))
                        }
                        _ => {
                            // (?<name>...) named group
                            // The first character of the name is in self.current
                            let first_char = match &self.current.kind {
                                TokenKind::Literal(c) => *c,
                                _ => {
                                    return Err(Error::with_span(
                                        ErrorKind::InvalidGroup,
                                        self.pattern,
                                        self.current.span,
                                    ));
                                }
                            };
                            // Read the rest of the identifier
                            let rest = self.lexer.read_ident().unwrap_or_default();
                            let name = format!("{}{}", first_char, rest);
                            self.current = self.lexer.next_token()?;
                            self.expect(TokenKind::GreaterThan)?;
                            let expr = self.parse_alternation()?;
                            self.expect_close_paren(start_span)?;
                            let index = self.next_capture;
                            self.next_capture += 1;
                            self.capture_count += 1;
                            Ok(Expr::Group(Box::new(Group {
                                expr,
                                kind: GroupKind::NamedCapturing { name, index },
                            })))
                        }
                    }
                }

                // Lookahead (?=...) or (?!...)
                TokenKind::Equals => {
                    self.advance()?;
                    let expr = self.parse_alternation()?;
                    self.expect_close_paren(start_span)?;
                    Ok(Expr::Lookaround(Box::new(Lookaround {
                        expr,
                        kind: LookaroundKind::PositiveLookahead,
                    })))
                }
                TokenKind::Exclamation => {
                    self.advance()?;
                    let expr = self.parse_alternation()?;
                    self.expect_close_paren(start_span)?;
                    Ok(Expr::Lookaround(Box::new(Lookaround {
                        expr,
                        kind: LookaroundKind::NegativeLookahead,
                    })))
                }

                // Python-style named group (?P<name>...)
                TokenKind::Literal('P') => {
                    self.advance()?;
                    self.expect(TokenKind::LessThan)?;
                    // The first character of the name is now in self.current
                    let first_char = match &self.current.kind {
                        TokenKind::Literal(c) => *c,
                        _ => {
                            return Err(Error::with_span(
                                ErrorKind::InvalidGroup,
                                self.pattern,
                                self.current.span,
                            ));
                        }
                    };
                    // Read the rest of the identifier
                    let rest = self.lexer.read_ident().unwrap_or_default();
                    let name = format!("{}{}", first_char, rest);
                    self.current = self.lexer.next_token()?;
                    self.expect(TokenKind::GreaterThan)?;
                    let expr = self.parse_alternation()?;
                    self.expect_close_paren(start_span)?;
                    let index = self.next_capture;
                    self.next_capture += 1;
                    self.capture_count += 1;
                    Ok(Expr::Group(Box::new(Group {
                        expr,
                        kind: GroupKind::NamedCapturing { name, index },
                    })))
                }

                // Flags (?imsx-imsx) or (?imsx:...)
                TokenKind::Literal(c) if is_flag_char(*c) => {
                    self.parse_flags()?;

                    if matches!(self.current.kind, TokenKind::Colon) {
                        // (?flags:...)
                        self.advance()?;
                        let expr = self.parse_alternation()?;
                        self.expect_close_paren(start_span)?;
                        Ok(Expr::Group(Box::new(Group {
                            expr,
                            kind: GroupKind::NonCapturing,
                        })))
                    } else if matches!(self.current.kind, TokenKind::CloseParen) {
                        // (?flags) - just set flags
                        self.advance()?;
                        Ok(Expr::Empty)
                    } else {
                        Err(Error::with_span(
                            ErrorKind::InvalidGroup,
                            self.pattern,
                            self.current.span,
                        ))
                    }
                }

                _ => Err(Error::with_span(
                    ErrorKind::InvalidGroup,
                    self.pattern,
                    self.current.span,
                )),
            }
        } else {
            // Regular capturing group
            let index = self.next_capture;
            self.next_capture += 1;
            self.capture_count += 1;

            let expr = self.parse_alternation()?;
            self.expect_close_paren(start_span)?;

            Ok(Expr::Group(Box::new(Group {
                expr,
                kind: GroupKind::Capturing(index),
            })))
        }
    }

    /// Expects a closing parenthesis.
    fn expect_close_paren(&mut self, open_span: Span) -> Result<()> {
        if matches!(self.current.kind, TokenKind::CloseParen) {
            self.advance()?;
            Ok(())
        } else {
            Err(Error::with_span(
                ErrorKind::UnmatchedOpenParen,
                self.pattern,
                open_span,
            ))
        }
    }

    /// Parses inline flags.
    fn parse_flags(&mut self) -> Result<()> {
        let mut negating = false;

        loop {
            match &self.current.kind {
                TokenKind::Literal(c) => match c {
                    'i' => {
                        self.flags.case_insensitive = !negating;
                        self.advance()?;
                    }
                    'm' => {
                        self.flags.multi_line = !negating;
                        self.advance()?;
                    }
                    's' => {
                        self.flags.dot_all = !negating;
                        self.advance()?;
                    }
                    'x' => {
                        self.flags.extended = !negating;
                        self.advance()?;
                    }
                    'u' => {
                        self.flags.unicode = !negating;
                        self.advance()?;
                    }
                    _ => break,
                },
                TokenKind::Hyphen => {
                    negating = true;
                    self.advance()?;
                }
                _ => break,
            }
        }

        Ok(())
    }

    /// Parses a character class: [abc], [^abc], [a-z].
    fn parse_class(&mut self) -> Result<Expr> {
        let start_span = self.current.span;
        // Set in_class BEFORE advancing so the next token is lexed correctly
        self.lexer.set_in_class(true);
        self.advance()?; // consume '['

        // Check for negation
        let negated = if matches!(self.current.kind, TokenKind::Caret) {
            self.advance()?;
            true
        } else {
            false
        };

        let mut ranges = Vec::new();

        // Handle leading ] or -
        if matches!(self.current.kind, TokenKind::CloseBracket) {
            ranges.push(ClassRange::single(']'));
            self.advance()?;
        } else if matches!(self.current.kind, TokenKind::Hyphen) {
            ranges.push(ClassRange::single('-'));
            self.advance()?;
        }

        while !matches!(self.current.kind, TokenKind::CloseBracket | TokenKind::Eof) {
            let item = self.parse_class_item()?;

            match item {
                ClassItem::Char(start_char) => {
                    // Check for range
                    if matches!(self.current.kind, TokenKind::Hyphen) {
                        self.advance()?;

                        // Trailing hyphen
                        if matches!(self.current.kind, TokenKind::CloseBracket) {
                            ranges.push(ClassRange::single(start_char));
                            ranges.push(ClassRange::single('-'));
                            break;
                        }

                        let end_item = self.parse_class_item()?;
                        match end_item {
                            ClassItem::Char(end_char) => {
                                if start_char > end_char {
                                    return Err(Error::with_span(
                                        ErrorKind::InvalidClassRange {
                                            start: start_char,
                                            end: end_char,
                                        },
                                        self.pattern,
                                        self.current.span,
                                    ));
                                }
                                ranges.push(ClassRange::new(start_char, end_char));
                            }
                            ClassItem::Ranges(r) => {
                                // Can't have a range ending with a Perl class like [a-\d]
                                // Just add start_char, hyphen, and the ranges
                                ranges.push(ClassRange::single(start_char));
                                ranges.push(ClassRange::single('-'));
                                ranges.extend(r);
                            }
                            ClassItem::UnicodeProperty { name, negated: _ } => {
                                // Can't have a range ending with a Unicode property like [a-\p{P}]
                                // Just add start_char, hyphen, and expand the property
                                ranges.push(ClassRange::single(start_char));
                                ranges.push(ClassRange::single('-'));
                                if let Some(code_point_ranges) = unicode_data::get_property(&name) {
                                    for &(start, end) in code_point_ranges {
                                        if (0xD800..=0xDFFF).contains(&start) {
                                            continue;
                                        }
                                        let start = start.min(0x10FFFF);
                                        let end = end.min(0x10FFFF);
                                        if let (Some(s), Some(e)) =
                                            (char::from_u32(start), char::from_u32(end))
                                        {
                                            ranges.push(ClassRange::new(s, e));
                                        }
                                    }
                                } else {
                                    return Err(Error::with_span(
                                        ErrorKind::UnknownUnicodeProperty(name.clone()),
                                        self.pattern,
                                        self.current.span,
                                    ));
                                }
                            }
                        }
                    } else {
                        ranges.push(ClassRange::single(start_char));
                    }
                }
                ClassItem::Ranges(r) => {
                    // Perl class like \d, \w, \s - just add all ranges
                    ranges.extend(r);
                }
                ClassItem::UnicodeProperty { name, negated } => {
                    // Look up Unicode property and expand to ranges
                    if let Some(code_point_ranges) = unicode_data::get_property(&name) {
                        // Convert (u32, u32) code point ranges to ClassRange (char-based)
                        for &(start, end) in code_point_ranges {
                            // Skip surrogate range (U+D800-U+DFFF) since they're not valid chars
                            if (0xD800..=0xDFFF).contains(&start) {
                                continue;
                            }
                            // Clamp end to valid char range
                            let start = start.min(0x10FFFF);
                            let end = end.min(0x10FFFF);

                            // Handle ranges that span across surrogates
                            if start < 0xD800 && end > 0xDFFF {
                                // Split into two ranges: before and after surrogates
                                if let (Some(s), Some(e)) =
                                    (char::from_u32(start), char::from_u32(0xD7FF))
                                {
                                    ranges.push(ClassRange::new(s, e));
                                }
                                if let (Some(s), Some(e)) =
                                    (char::from_u32(0xE000), char::from_u32(end))
                                {
                                    ranges.push(ClassRange::new(s, e));
                                }
                            } else if start <= 0xD7FF && (0xD800..=0xDFFF).contains(&end) {
                                // Range ends in surrogates, truncate
                                if let (Some(s), Some(e)) =
                                    (char::from_u32(start), char::from_u32(0xD7FF))
                                {
                                    ranges.push(ClassRange::new(s, e));
                                }
                            } else if (0xD800..=0xDFFF).contains(&start) && end > 0xDFFF {
                                // Range starts in surrogates, start from after
                                if let (Some(s), Some(e)) =
                                    (char::from_u32(0xE000), char::from_u32(end))
                                {
                                    ranges.push(ClassRange::new(s, e));
                                }
                            } else if let (Some(s), Some(e)) =
                                (char::from_u32(start), char::from_u32(end))
                            {
                                ranges.push(ClassRange::new(s, e));
                            }
                        }

                        // Handle negation: compute complement for \P{} inside class
                        if negated {
                            // Compute complement: all characters NOT in the property ranges
                            // The ranges we just added are the positive ranges, we need their complement
                            let count_to_drain = code_point_ranges
                                .iter()
                                .filter(|&&(start, _)| {
                                    // Count how many ranges we actually added
                                    !(0xD800..=0xDFFF).contains(&start)
                                })
                                .map(|&(start, end)| {
                                    let start = start.min(0x10FFFF);
                                    let end = end.min(0x10FFFF);
                                    if start < 0xD800 && end > 0xDFFF {
                                        2
                                    } else {
                                        1
                                    }
                                })
                                .sum::<usize>();
                            let drain_start = ranges.len().saturating_sub(count_to_drain);
                            let positive_ranges: Vec<ClassRange> =
                                ranges.drain(drain_start..).collect();

                            // Compute complement of positive_ranges
                            // Sort and merge positive ranges first
                            let mut sorted: Vec<(u32, u32)> = positive_ranges
                                .iter()
                                .map(|r| (r.start as u32, r.end as u32))
                                .collect();
                            sorted.sort_by_key(|r| r.0);

                            // Merge overlapping ranges
                            let mut merged: Vec<(u32, u32)> = Vec::new();
                            for (start, end) in sorted {
                                if let Some(last) = merged.last_mut() {
                                    if start <= last.1 + 1 {
                                        last.1 = last.1.max(end);
                                        continue;
                                    }
                                }
                                merged.push((start, end));
                            }

                            // Compute complement (gaps between merged ranges)
                            let mut prev_end: u32 = 0;
                            for (start, end) in merged {
                                if prev_end < start {
                                    // Gap from prev_end to start-1
                                    let gap_start = prev_end;
                                    let gap_end = start - 1;
                                    // Add gap ranges, avoiding surrogates
                                    if gap_start < 0xD800 {
                                        let s = gap_start;
                                        let e = gap_end.min(0xD7FF);
                                        if s <= e {
                                            if let (Some(cs), Some(ce)) =
                                                (char::from_u32(s), char::from_u32(e))
                                            {
                                                ranges.push(ClassRange::new(cs, ce));
                                            }
                                        }
                                    }
                                    if gap_end > 0xDFFF {
                                        let s = gap_start.max(0xE000);
                                        let e = gap_end;
                                        if s <= e {
                                            if let (Some(cs), Some(ce)) =
                                                (char::from_u32(s), char::from_u32(e))
                                            {
                                                ranges.push(ClassRange::new(cs, ce));
                                            }
                                        }
                                    }
                                }
                                prev_end = end + 1;
                            }
                            // Add final gap from last range to max codepoint
                            if prev_end <= 0x10FFFF {
                                let gap_start = prev_end;
                                let gap_end = 0x10FFFF;
                                if gap_start < 0xD800 {
                                    let s = gap_start;
                                    let e = gap_end.min(0xD7FF);
                                    if s <= e {
                                        if let (Some(cs), Some(ce)) =
                                            (char::from_u32(s), char::from_u32(e))
                                        {
                                            ranges.push(ClassRange::new(cs, ce));
                                        }
                                    }
                                }
                                if gap_end > 0xDFFF {
                                    let s = gap_start.max(0xE000);
                                    let e = gap_end;
                                    if s <= e {
                                        if let (Some(cs), Some(ce)) =
                                            (char::from_u32(s), char::from_u32(e))
                                        {
                                            ranges.push(ClassRange::new(cs, ce));
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        return Err(Error::with_span(
                            ErrorKind::UnknownUnicodeProperty(name.clone()),
                            self.pattern,
                            self.current.span,
                        ));
                    }
                }
            }
        }

        self.lexer.set_in_class(false);

        if matches!(self.current.kind, TokenKind::Eof) {
            return Err(Error::with_span(
                ErrorKind::UnmatchedOpenBracket,
                self.pattern,
                start_span,
            ));
        }

        self.advance()?; // consume ']'

        if ranges.is_empty() {
            return Err(Error::with_span(
                ErrorKind::EmptyClass,
                self.pattern,
                start_span,
            ));
        }

        Ok(Expr::Class(Box::new(Class::new(ranges, negated))))
    }

    /// Parses a class item which can be a single char, a range, or a Perl class.
    /// Returns either a list of ranges (for Perl classes) or a single character.
    fn parse_class_item(&mut self) -> Result<ClassItem> {
        match &self.current.kind {
            TokenKind::Escape(esc) => {
                match esc {
                    // Perl character classes - expand to ranges
                    EscapeKind::Digit => {
                        self.advance()?;
                        Ok(ClassItem::Ranges(vec![ClassRange::new('0', '9')]))
                    }
                    EscapeKind::NotDigit => {
                        self.advance()?;
                        // [^\d] = everything except 0-9
                        Ok(ClassItem::Ranges(vec![
                            ClassRange::new('\x00', '/'),       // 0x00-0x2F
                            ClassRange::new(':', '\u{10FFFF}'), // 0x3A onwards
                        ]))
                    }
                    EscapeKind::Word => {
                        self.advance()?;
                        Ok(ClassItem::Ranges(vec![
                            ClassRange::new('a', 'z'),
                            ClassRange::new('A', 'Z'),
                            ClassRange::new('0', '9'),
                            ClassRange::single('_'),
                        ]))
                    }
                    EscapeKind::NotWord => {
                        self.advance()?;
                        // [^\w] = everything except [a-zA-Z0-9_]
                        Ok(ClassItem::Ranges(vec![
                            ClassRange::new('\x00', '/'),       // before '0'
                            ClassRange::new(':', '@'),          // between '9' and 'A'
                            ClassRange::new('[', '^'),          // between 'Z' and '_'
                            ClassRange::single('`'),            // between '_' and 'a'
                            ClassRange::new('{', '\u{10FFFF}'), // after 'z'
                        ]))
                    }
                    EscapeKind::Whitespace => {
                        self.advance()?;
                        Ok(ClassItem::Ranges(vec![
                            ClassRange::single(' '),
                            ClassRange::single('\t'),
                            ClassRange::single('\n'),
                            ClassRange::single('\r'),
                            ClassRange::single('\x0C'), // form feed
                            ClassRange::single('\x0B'), // vertical tab
                        ]))
                    }
                    EscapeKind::NotWhitespace => {
                        self.advance()?;
                        // [^\s] = everything except whitespace
                        Ok(ClassItem::Ranges(vec![
                            ClassRange::new('\x00', '\x08'),    // before \t
                            ClassRange::new('\x0E', '\x1F'),    // between \r and space
                            ClassRange::new('!', '\u{10FFFF}'), // after space
                        ]))
                    }
                    // Single character escapes
                    EscapeKind::Literal(c) => {
                        let c = *c;
                        self.advance()?;
                        Ok(ClassItem::Char(c))
                    }
                    EscapeKind::Newline => {
                        self.advance()?;
                        Ok(ClassItem::Char('\n'))
                    }
                    EscapeKind::CarriageReturn => {
                        self.advance()?;
                        Ok(ClassItem::Char('\r'))
                    }
                    EscapeKind::Tab => {
                        self.advance()?;
                        Ok(ClassItem::Char('\t'))
                    }
                    EscapeKind::FormFeed => {
                        self.advance()?;
                        Ok(ClassItem::Char('\x0C'))
                    }
                    EscapeKind::VerticalTab => {
                        self.advance()?;
                        Ok(ClassItem::Char('\x0B'))
                    }
                    EscapeKind::Null => {
                        self.advance()?;
                        Ok(ClassItem::Char('\0'))
                    }
                    EscapeKind::Hex(c) | EscapeKind::Unicode(c) => {
                        let c = *c;
                        self.advance()?;
                        Ok(ClassItem::Char(c))
                    }
                    // Unicode properties \p{...} and \P{...}
                    EscapeKind::UnicodeProperty(name) => {
                        let name = name.clone();
                        self.advance()?;
                        Ok(ClassItem::UnicodeProperty {
                            name,
                            negated: false,
                        })
                    }
                    EscapeKind::NotUnicodeProperty(name) => {
                        let name = name.clone();
                        self.advance()?;
                        Ok(ClassItem::UnicodeProperty {
                            name,
                            negated: true,
                        })
                    }
                    _ => Err(Error::with_span(
                        ErrorKind::InvalidEscape('?'),
                        self.pattern,
                        self.current.span,
                    )),
                }
            }
            TokenKind::Literal(c) => {
                let c = *c;
                self.advance()?;
                Ok(ClassItem::Char(c))
            }
            // Inside a character class (not at start), caret is a literal character
            TokenKind::Caret => {
                self.advance()?;
                Ok(ClassItem::Char('^'))
            }
            _ => Err(Error::with_span(
                ErrorKind::UnexpectedChar(self.current_char().unwrap_or('?')),
                self.pattern,
                self.current.span,
            )),
        }
    }
}

/// Item parsed from a character class.
enum ClassItem {
    /// A single character
    Char(char),
    /// Multiple ranges (from Perl classes like \d, \w, \s)
    Ranges(Vec<ClassRange>),
    /// Unicode property (will be expanded to ranges later)
    UnicodeProperty { name: String, negated: bool },
}

/// Returns true if the character is a valid flag.
fn is_flag_char(c: char) -> bool {
    matches!(c, 'i' | 'm' | 's' | 'x' | 'u')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal() {
        let ast = parse("abc").unwrap();
        assert!(matches!(ast.expr, Expr::Concat(_)));
    }

    #[test]
    fn test_alternation() {
        let ast = parse("a|b|c").unwrap();
        if let Expr::Alt(alts) = ast.expr {
            assert_eq!(alts.len(), 3);
        } else {
            panic!("Expected Alt");
        }
    }

    #[test]
    fn test_quantifiers() {
        let ast = parse("a*b+c?").unwrap();
        if let Expr::Concat(exprs) = ast.expr {
            assert!(matches!(exprs[0], Expr::Repeat(_)));
            assert!(matches!(exprs[1], Expr::Repeat(_)));
            assert!(matches!(exprs[2], Expr::Repeat(_)));
        } else {
            panic!("Expected Concat");
        }
    }

    #[test]
    fn test_repetition_range() {
        let ast = parse("a{2,5}").unwrap();
        if let Expr::Repeat(rep) = ast.expr {
            assert_eq!(rep.min, 2);
            assert_eq!(rep.max, Some(5));
        } else {
            panic!("Expected Repeat");
        }
    }

    #[test]
    fn test_character_class() {
        let ast = parse("[a-z]").unwrap();
        if let Expr::Class(cls) = ast.expr {
            assert!(!cls.negated);
            assert_eq!(cls.ranges.len(), 1);
            assert_eq!(cls.ranges[0].start, 'a');
            assert_eq!(cls.ranges[0].end, 'z');
        } else {
            panic!("Expected Class");
        }
    }

    #[test]
    fn test_negated_class() {
        let ast = parse("[^0-9]").unwrap();
        if let Expr::Class(cls) = ast.expr {
            assert!(cls.negated);
        } else {
            panic!("Expected Class");
        }
    }

    #[test]
    fn test_capturing_group() {
        let ast = parse("(abc)").unwrap();
        if let Expr::Group(g) = ast.expr {
            assert!(matches!(g.kind, GroupKind::Capturing(1)));
        } else {
            panic!("Expected Group");
        }
    }

    #[test]
    fn test_non_capturing_group() {
        let ast = parse("(?:abc)").unwrap();
        if let Expr::Group(g) = ast.expr {
            assert!(matches!(g.kind, GroupKind::NonCapturing));
        } else {
            panic!("Expected Group");
        }
    }

    #[test]
    fn test_lookahead() {
        let ast = parse("(?=abc)").unwrap();
        if let Expr::Lookaround(la) = ast.expr {
            assert!(matches!(la.kind, LookaroundKind::PositiveLookahead));
        } else {
            panic!("Expected Lookaround");
        }
    }

    #[test]
    fn test_escape_sequences() {
        let ast = parse(r"\d\w\s").unwrap();
        if let Expr::Concat(exprs) = ast.expr {
            assert_eq!(exprs.len(), 3);
            assert!(matches!(exprs[0], Expr::PerlClass(PerlClassKind::Digit)));
            assert!(matches!(exprs[1], Expr::PerlClass(PerlClassKind::Word)));
            assert!(matches!(
                exprs[2],
                Expr::PerlClass(PerlClassKind::Whitespace)
            ));
        } else {
            panic!("Expected Concat");
        }
    }

    #[test]
    fn test_anchors() {
        let ast = parse(r"^\w+$").unwrap();
        if let Expr::Concat(exprs) = ast.expr {
            assert!(matches!(exprs[0], Expr::Anchor(Anchor::StartOfString)));
            assert!(matches!(exprs[2], Expr::Anchor(Anchor::EndOfString)));
        } else {
            panic!("Expected Concat");
        }
    }

    #[test]
    fn test_error_unmatched_paren() {
        let err = parse("(abc").unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::UnmatchedOpenParen));
    }

    #[test]
    fn test_error_quantifier_on_nothing() {
        let err = parse("*abc").unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::RepetitionOnNothing));
    }

    #[test]
    fn test_error_nested_quantifier() {
        let err = parse("a**").unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::NestedQuantifier));
    }
}
