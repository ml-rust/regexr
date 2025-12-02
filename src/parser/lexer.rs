//! Lexer/tokenizer for regex patterns.

use crate::error::{Error, ErrorKind, Result, Span};

/// A token in the regex pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// The kind of token.
    pub kind: TokenKind,
    /// Position in the source pattern.
    pub span: Span,
}

/// The kind of token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    /// A literal character.
    Literal(char),
    /// Dot (.) - any character.
    Dot,
    /// Star (*) - zero or more.
    Star,
    /// Plus (+) - one or more.
    Plus,
    /// Question (?) - zero or one, or non-greedy modifier.
    Question,
    /// Pipe (|) - alternation.
    Pipe,
    /// Caret (^) - start anchor or class negation.
    Caret,
    /// Dollar ($) - end anchor.
    Dollar,
    /// Opening parenthesis.
    OpenParen,
    /// Closing parenthesis.
    CloseParen,
    /// Opening bracket.
    OpenBracket,
    /// Closing bracket.
    CloseBracket,
    /// Opening brace.
    OpenBrace,
    /// Closing brace.
    CloseBrace,
    /// Hyphen (-) in character class.
    Hyphen,
    /// Comma (,) in repetition.
    Comma,
    /// A digit (for repetition bounds and backrefs).
    Digit(u32),
    /// Escaped character.
    Escape(EscapeKind),
    /// Colon (:).
    Colon,
    /// Less than (<).
    LessThan,
    /// Greater than (>).
    GreaterThan,
    /// Equals (=).
    Equals,
    /// Exclamation (!).
    Exclamation,
    /// An identifier (for named groups).
    Ident(String),
    /// End of input.
    Eof,
}

/// An escape sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EscapeKind {
    /// Literal escaped character (e.g., \., \\).
    Literal(char),
    /// \d - digit.
    Digit,
    /// \D - non-digit.
    NotDigit,
    /// \w - word character.
    Word,
    /// \W - non-word character.
    NotWord,
    /// \s - whitespace.
    Whitespace,
    /// \S - non-whitespace.
    NotWhitespace,
    /// \b - word boundary.
    WordBoundary,
    /// \B - non-word boundary.
    NotWordBoundary,
    /// \A - start of input.
    StartOfInput,
    /// \z - end of input.
    EndOfInput,
    /// \Z - end of input before newline.
    EndOfInputBeforeNewline,
    /// \n - newline.
    Newline,
    /// \r - carriage return.
    CarriageReturn,
    /// \t - tab.
    Tab,
    /// \f - form feed.
    FormFeed,
    /// \v - vertical tab.
    VerticalTab,
    /// \0 - null.
    Null,
    /// \xHH - hex escape.
    Hex(char),
    /// \u{HHHH} - unicode escape.
    Unicode(char),
    /// \1, \2, etc. - backreference.
    Backref(u32),
    /// \p{PropertyName} - Unicode property.
    UnicodeProperty(String),
    /// \P{PropertyName} - negated Unicode property.
    NotUnicodeProperty(String),
}

/// The lexer for regex patterns.
pub struct Lexer<'a> {
    /// The source pattern.
    src: &'a str,
    /// Iterator over characters.
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    /// Current position in bytes.
    pos: usize,
    /// Whether we're inside a character class.
    in_class: bool,
}

impl<'a> Lexer<'a> {
    /// Creates a new lexer.
    pub fn new(src: &'a str) -> Self {
        Self {
            src,
            chars: src.char_indices().peekable(),
            pos: 0,
            in_class: false,
        }
    }

    /// Returns the source pattern.
    pub fn source(&self) -> &str {
        self.src
    }

    /// Peeks at the next character without consuming it.
    fn peek_char(&mut self) -> Option<char> {
        self.chars.peek().map(|(_, c)| *c)
    }

    /// Consumes the next character.
    fn next_char(&mut self) -> Option<(usize, char)> {
        let result = self.chars.next();
        if let Some((pos, c)) = result {
            self.pos = pos + c.len_utf8();
        }
        result
    }

    /// Sets whether we're inside a character class.
    pub fn set_in_class(&mut self, in_class: bool) {
        self.in_class = in_class;
    }

    /// Returns the next token.
    pub fn next_token(&mut self) -> Result<Token> {
        let (start, c) = match self.next_char() {
            Some(pair) => pair,
            None => {
                return Ok(Token {
                    kind: TokenKind::Eof,
                    span: Span::point(self.pos),
                });
            }
        };

        let kind = if self.in_class {
            self.lex_class_char(c, start)?
        } else {
            self.lex_char(c, start)?
        };

        Ok(Token {
            kind,
            span: Span::new(start, self.pos),
        })
    }

    /// Lexes a character outside a character class.
    fn lex_char(&mut self, c: char, start: usize) -> Result<TokenKind> {
        match c {
            '.' => Ok(TokenKind::Dot),
            '*' => Ok(TokenKind::Star),
            '+' => Ok(TokenKind::Plus),
            '?' => Ok(TokenKind::Question),
            '|' => Ok(TokenKind::Pipe),
            '^' => Ok(TokenKind::Caret),
            '$' => Ok(TokenKind::Dollar),
            '(' => Ok(TokenKind::OpenParen),
            ')' => Ok(TokenKind::CloseParen),
            '[' => Ok(TokenKind::OpenBracket),
            ']' => Ok(TokenKind::CloseBracket),
            '{' => Ok(TokenKind::OpenBrace),
            '}' => Ok(TokenKind::CloseBrace),
            ':' => Ok(TokenKind::Colon),
            '<' => Ok(TokenKind::LessThan),
            '>' => Ok(TokenKind::GreaterThan),
            '=' => Ok(TokenKind::Equals),
            '!' => Ok(TokenKind::Exclamation),
            ',' => Ok(TokenKind::Comma),
            '\\' => self.lex_escape(start),
            c if c.is_ascii_digit() => Ok(TokenKind::Digit(c.to_digit(10).unwrap())),
            c => Ok(TokenKind::Literal(c)),
        }
    }

    /// Lexes a character inside a character class.
    fn lex_class_char(&mut self, c: char, start: usize) -> Result<TokenKind> {
        match c {
            ']' => Ok(TokenKind::CloseBracket),
            '-' => Ok(TokenKind::Hyphen),
            '^' => Ok(TokenKind::Caret),
            '\\' => self.lex_escape(start),
            c => Ok(TokenKind::Literal(c)),
        }
    }

    /// Lexes an escape sequence.
    fn lex_escape(&mut self, start: usize) -> Result<TokenKind> {
        let c = match self.next_char() {
            Some((_, c)) => c,
            None => {
                return Err(Error::with_span(
                    ErrorKind::UnexpectedEof,
                    self.src,
                    Span::point(start),
                ));
            }
        };

        let escape = match c {
            // Character classes
            'd' => EscapeKind::Digit,
            'D' => EscapeKind::NotDigit,
            'w' => EscapeKind::Word,
            'W' => EscapeKind::NotWord,
            's' => EscapeKind::Whitespace,
            'S' => EscapeKind::NotWhitespace,

            // Anchors
            'b' => EscapeKind::WordBoundary,
            'B' => EscapeKind::NotWordBoundary,
            'A' => EscapeKind::StartOfInput,
            'z' => EscapeKind::EndOfInput,
            'Z' => EscapeKind::EndOfInputBeforeNewline,

            // Special characters
            'n' => EscapeKind::Newline,
            'r' => EscapeKind::CarriageReturn,
            't' => EscapeKind::Tab,
            'f' => EscapeKind::FormFeed,
            'v' => EscapeKind::VerticalTab,
            '0' => EscapeKind::Null,

            // Hex escape
            'x' => {
                let ch = self.lex_hex_escape(start)?;
                EscapeKind::Hex(ch)
            }

            // Unicode escape
            'u' => {
                let ch = self.lex_unicode_escape(start)?;
                EscapeKind::Unicode(ch)
            }

            // Unicode property
            'p' => {
                let name = self.lex_unicode_property(start)?;
                EscapeKind::UnicodeProperty(name)
            }

            // Negated Unicode property
            'P' => {
                let name = self.lex_unicode_property(start)?;
                EscapeKind::NotUnicodeProperty(name)
            }

            // Backreference
            c if c.is_ascii_digit() && c != '0' => {
                let n = self.lex_backref(c)?;
                EscapeKind::Backref(n)
            }

            // Literal escape
            '\\' | '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$'
            | '-' | '/' => EscapeKind::Literal(c),

            _ => {
                return Err(Error::with_span(
                    ErrorKind::InvalidEscape(c),
                    self.src,
                    Span::new(start, self.pos),
                ));
            }
        };

        Ok(TokenKind::Escape(escape))
    }

    /// Lexes a hex escape (\xHH).
    fn lex_hex_escape(&mut self, start: usize) -> Result<char> {
        let mut value = 0u32;

        for _ in 0..2 {
            let (_, c) = self.next_char().ok_or_else(|| {
                Error::with_span(
                    ErrorKind::InvalidHexEscape,
                    self.src,
                    Span::new(start, self.pos),
                )
            })?;

            let digit = c.to_digit(16).ok_or_else(|| {
                Error::with_span(
                    ErrorKind::InvalidHexEscape,
                    self.src,
                    Span::new(start, self.pos),
                )
            })?;

            value = value * 16 + digit;
        }

        char::from_u32(value).ok_or_else(|| {
            Error::with_span(
                ErrorKind::InvalidHexEscape,
                self.src,
                Span::new(start, self.pos),
            )
        })
    }

    /// Lexes a unicode escape (\u{HHHH} or \uHHHH).
    fn lex_unicode_escape(&mut self, start: usize) -> Result<char> {
        let braced = self.peek_char() == Some('{');

        if braced {
            self.next_char(); // consume '{'

            let mut value = 0u32;
            let mut count = 0;

            loop {
                match self.peek_char() {
                    Some('}') => {
                        self.next_char();
                        break;
                    }
                    Some(c) if c.is_ascii_hexdigit() => {
                        self.next_char();
                        let digit = c.to_digit(16).unwrap();
                        value = value * 16 + digit;
                        count += 1;
                        if count > 6 {
                            return Err(Error::with_span(
                                ErrorKind::InvalidUnicodeEscape,
                                self.src,
                                Span::new(start, self.pos),
                            ));
                        }
                    }
                    _ => {
                        return Err(Error::with_span(
                            ErrorKind::InvalidUnicodeEscape,
                            self.src,
                            Span::new(start, self.pos),
                        ));
                    }
                }
            }

            if count == 0 {
                return Err(Error::with_span(
                    ErrorKind::InvalidUnicodeEscape,
                    self.src,
                    Span::new(start, self.pos),
                ));
            }

            char::from_u32(value).ok_or_else(|| {
                Error::with_span(
                    ErrorKind::InvalidUnicodeEscape,
                    self.src,
                    Span::new(start, self.pos),
                )
            })
        } else {
            // \uHHHH format
            let mut value = 0u32;

            for _ in 0..4 {
                let (_, c) = self.next_char().ok_or_else(|| {
                    Error::with_span(
                        ErrorKind::InvalidUnicodeEscape,
                        self.src,
                        Span::new(start, self.pos),
                    )
                })?;

                let digit = c.to_digit(16).ok_or_else(|| {
                    Error::with_span(
                        ErrorKind::InvalidUnicodeEscape,
                        self.src,
                        Span::new(start, self.pos),
                    )
                })?;

                value = value * 16 + digit;
            }

            char::from_u32(value).ok_or_else(|| {
                Error::with_span(
                    ErrorKind::InvalidUnicodeEscape,
                    self.src,
                    Span::new(start, self.pos),
                )
            })
        }
    }

    /// Lexes a Unicode property escape (\p{Name} or \P{Name}).
    fn lex_unicode_property(&mut self, start: usize) -> Result<String> {
        // Expect opening brace
        match self.next_char() {
            Some((_, '{')) => {}
            _ => {
                return Err(Error::with_span(
                    ErrorKind::InvalidUnicodeProperty,
                    self.src,
                    Span::new(start, self.pos),
                ));
            }
        }

        let mut name = String::new();

        // Read property name until closing brace
        loop {
            match self.next_char() {
                Some((_, '}')) => break,
                Some((_, c)) if c.is_alphanumeric() || c == '_' || c == '-' => {
                    name.push(c);
                }
                _ => {
                    return Err(Error::with_span(
                        ErrorKind::InvalidUnicodeProperty,
                        self.src,
                        Span::new(start, self.pos),
                    ));
                }
            }
        }

        if name.is_empty() {
            return Err(Error::with_span(
                ErrorKind::InvalidUnicodeProperty,
                self.src,
                Span::new(start, self.pos),
            ));
        }

        Ok(name)
    }

    /// Lexes a backreference (\1, \12, etc.).
    fn lex_backref(&mut self, first: char) -> Result<u32> {
        let mut n = first.to_digit(10).unwrap();

        // Consume additional digits
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() {
                self.next_char();
                n = n * 10 + c.to_digit(10).unwrap();
            } else {
                break;
            }
        }

        Ok(n)
    }

    /// Reads an identifier (for named groups).
    pub fn read_ident(&mut self) -> Result<String> {
        let start = self.pos;
        let mut ident = String::new();

        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '_' {
                self.next_char();
                ident.push(c);
            } else {
                break;
            }
        }

        if ident.is_empty() {
            return Err(Error::with_span(
                ErrorKind::InvalidGroup,
                self.src,
                Span::point(start),
            ));
        }

        Ok(ident)
    }

    /// Reads a number (for repetition bounds).
    pub fn read_number(&mut self) -> Result<Option<u32>> {
        let mut n: u32 = 0;
        let mut has_digit = false;

        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() {
                self.next_char();
                has_digit = true;
                n = n.saturating_mul(10).saturating_add(c.to_digit(10).unwrap());
            } else {
                break;
            }
        }

        Ok(if has_digit { Some(n) } else { None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_all(src: &str) -> Result<Vec<TokenKind>> {
        let mut lexer = Lexer::new(src);
        let mut tokens = Vec::new();
        loop {
            let tok = lexer.next_token()?;
            if tok.kind == TokenKind::Eof {
                break;
            }
            tokens.push(tok.kind);
        }
        Ok(tokens)
    }

    #[test]
    fn test_simple_tokens() {
        let tokens = lex_all("a.b*c+d?").unwrap();
        assert_eq!(
            tokens,
            vec![
                TokenKind::Literal('a'),
                TokenKind::Dot,
                TokenKind::Literal('b'),
                TokenKind::Star,
                TokenKind::Literal('c'),
                TokenKind::Plus,
                TokenKind::Literal('d'),
                TokenKind::Question,
            ]
        );
    }

    #[test]
    fn test_escapes() {
        let tokens = lex_all(r"\d\w\s\n\t").unwrap();
        assert_eq!(
            tokens,
            vec![
                TokenKind::Escape(EscapeKind::Digit),
                TokenKind::Escape(EscapeKind::Word),
                TokenKind::Escape(EscapeKind::Whitespace),
                TokenKind::Escape(EscapeKind::Newline),
                TokenKind::Escape(EscapeKind::Tab),
            ]
        );
    }

    #[test]
    fn test_hex_escape() {
        let tokens = lex_all(r"\x41").unwrap();
        assert_eq!(tokens, vec![TokenKind::Escape(EscapeKind::Hex('A'))]);
    }

    #[test]
    fn test_unicode_escape() {
        let tokens = lex_all(r"\u{1F600}").unwrap();
        assert_eq!(tokens, vec![TokenKind::Escape(EscapeKind::Unicode('😀'))]);
    }

    #[test]
    fn test_backref() {
        let tokens = lex_all(r"\1\12").unwrap();
        assert_eq!(
            tokens,
            vec![
                TokenKind::Escape(EscapeKind::Backref(1)),
                TokenKind::Escape(EscapeKind::Backref(12)),
            ]
        );
    }

    #[test]
    fn test_invalid_escape() {
        let err = lex_all(r"\q").unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::InvalidEscape('q')));
    }

    #[test]
    fn test_unicode_property() {
        let tokens = lex_all(r"\p{Letter}").unwrap();
        assert_eq!(
            tokens,
            vec![TokenKind::Escape(EscapeKind::UnicodeProperty(
                "Letter".to_string()
            ))]
        );
    }

    #[test]
    fn test_negated_unicode_property() {
        let tokens = lex_all(r"\P{Number}").unwrap();
        assert_eq!(
            tokens,
            vec![TokenKind::Escape(EscapeKind::NotUnicodeProperty(
                "Number".to_string()
            ))]
        );
    }

    #[test]
    fn test_unicode_property_short() {
        let tokens = lex_all(r"\p{L}").unwrap();
        assert_eq!(
            tokens,
            vec![TokenKind::Escape(EscapeKind::UnicodeProperty(
                "L".to_string()
            ))]
        );
    }

    #[test]
    fn test_invalid_unicode_property_no_brace() {
        let err = lex_all(r"\pL").unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::InvalidUnicodeProperty));
    }

    #[test]
    fn test_invalid_unicode_property_empty() {
        let err = lex_all(r"\p{}").unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::InvalidUnicodeProperty));
    }
}
