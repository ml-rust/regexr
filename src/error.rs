//! Error types for the regex engine.

use std::fmt;

/// A specialized Result type for regex operations.
pub type Result<T> = std::result::Result<T, Error>;

/// An error that occurred during regex parsing or compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    pattern: String,
    span: Option<Span>,
}

/// The position in the pattern where an error occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Start byte offset (inclusive).
    pub start: usize,
    /// End byte offset (exclusive).
    pub end: usize,
}

impl Span {
    /// Creates a new span.
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Creates a span for a single position.
    pub fn point(pos: usize) -> Self {
        Self {
            start: pos,
            end: pos + 1,
        }
    }
}

/// The kind of error that occurred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    // Parse errors
    /// Unexpected end of pattern.
    UnexpectedEof,
    /// Unexpected character in pattern.
    UnexpectedChar(char),
    /// Unmatched opening parenthesis.
    UnmatchedOpenParen,
    /// Unmatched closing parenthesis.
    UnmatchedCloseParen,
    /// Unmatched opening bracket.
    UnmatchedOpenBracket,
    /// Unmatched closing bracket.
    UnmatchedCloseBracket,
    /// Invalid escape sequence.
    InvalidEscape(char),
    /// Invalid hex escape.
    InvalidHexEscape,
    /// Invalid unicode escape.
    InvalidUnicodeEscape,
    /// Invalid Unicode property (e.g., \p{InvalidName}).
    InvalidUnicodeProperty,
    /// Unknown Unicode property name.
    UnknownUnicodeProperty(String),
    /// Invalid repetition syntax.
    InvalidRepetition,
    /// Repetition quantifier on nothing.
    RepetitionOnNothing,
    /// Invalid character class range (e.g., z-a).
    InvalidClassRange {
        /// Start of the invalid range.
        start: char,
        /// End of the invalid range.
        end: char,
    },
    /// Empty character class.
    EmptyClass,
    /// Invalid group syntax.
    InvalidGroup,
    /// Invalid backreference.
    InvalidBackref(usize),
    /// Backreference to non-existent group.
    BackrefNotFound(usize),
    /// Nested quantifiers (e.g., a**).
    NestedQuantifier,

    // Compile errors
    /// Pattern too large.
    PatternTooLarge,
    /// Too many capture groups.
    TooManyCaptureGroups,
    /// Too many states in NFA.
    TooManyStates,

    // Runtime errors
    /// Match limit exceeded (to prevent ReDoS).
    MatchLimitExceeded,
    /// Stack overflow during matching.
    StackOverflow,

    // JIT errors
    /// JIT compilation failed.
    Jit(String),
}

impl Error {
    /// Creates a new error.
    pub fn new(kind: ErrorKind, pattern: impl Into<String>) -> Self {
        Self {
            kind,
            pattern: pattern.into(),
            span: None,
        }
    }

    /// Creates a new error with a span.
    pub fn with_span(kind: ErrorKind, pattern: impl Into<String>, span: Span) -> Self {
        Self {
            kind,
            pattern: pattern.into(),
            span: Some(span),
        }
    }

    /// Returns the error kind.
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// Returns the pattern that caused the error.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Returns the span where the error occurred, if available.
    pub fn span(&self) -> Option<Span> {
        self.span
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "regex error: {}", self.kind)?;

        if let Some(span) = self.span {
            write!(f, " at position {}", span.start)?;

            // Show context
            if !self.pattern.is_empty() {
                write!(f, "\n  pattern: {}", self.pattern)?;
                write!(f, "\n           ")?;
                for _ in 0..span.start {
                    write!(f, " ")?;
                }
                let len = (span.end - span.start).max(1);
                for _ in 0..len {
                    write!(f, "^")?;
                }
            }
        }

        Ok(())
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::UnexpectedEof => write!(f, "unexpected end of pattern"),
            ErrorKind::UnexpectedChar(c) => write!(f, "unexpected character '{}'", c),
            ErrorKind::UnmatchedOpenParen => write!(f, "unmatched '('"),
            ErrorKind::UnmatchedCloseParen => write!(f, "unmatched ')'"),
            ErrorKind::UnmatchedOpenBracket => write!(f, "unmatched '['"),
            ErrorKind::UnmatchedCloseBracket => write!(f, "unmatched ']'"),
            ErrorKind::InvalidEscape(c) => write!(f, "invalid escape sequence '\\{}'", c),
            ErrorKind::InvalidHexEscape => write!(f, "invalid hex escape sequence"),
            ErrorKind::InvalidUnicodeEscape => write!(f, "invalid unicode escape sequence"),
            ErrorKind::InvalidUnicodeProperty => write!(f, "invalid unicode property syntax"),
            ErrorKind::UnknownUnicodeProperty(name) => {
                write!(f, "unknown unicode property '{}'", name)
            }
            ErrorKind::InvalidRepetition => write!(f, "invalid repetition syntax"),
            ErrorKind::RepetitionOnNothing => write!(f, "quantifier on nothing"),
            ErrorKind::InvalidClassRange { start, end } => {
                write!(f, "invalid character class range '{}-{}'", start, end)
            }
            ErrorKind::EmptyClass => write!(f, "empty character class"),
            ErrorKind::InvalidGroup => write!(f, "invalid group syntax"),
            ErrorKind::InvalidBackref(n) => write!(f, "invalid backreference '\\{}'", n),
            ErrorKind::BackrefNotFound(n) => {
                write!(f, "backreference '\\{}' references non-existent group", n)
            }
            ErrorKind::NestedQuantifier => write!(f, "nested quantifiers are not allowed"),
            ErrorKind::PatternTooLarge => write!(f, "pattern too large"),
            ErrorKind::TooManyCaptureGroups => write!(f, "too many capture groups"),
            ErrorKind::TooManyStates => write!(f, "too many NFA states"),
            ErrorKind::MatchLimitExceeded => write!(f, "match limit exceeded"),
            ErrorKind::StackOverflow => write!(f, "stack overflow during matching"),
            ErrorKind::Jit(msg) => write!(f, "JIT compilation failed: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::with_span(ErrorKind::UnexpectedChar('?'), "a?*b", Span::new(2, 3));
        let msg = err.to_string();
        assert!(msg.contains("unexpected character"));
        assert!(msg.contains("position 2"));
    }

    #[test]
    fn test_error_kind_display() {
        assert_eq!(
            ErrorKind::InvalidClassRange {
                start: 'z',
                end: 'a'
            }
            .to_string(),
            "invalid character class range 'z-a'"
        );
    }
}
