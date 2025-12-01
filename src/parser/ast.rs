//! Abstract Syntax Tree definitions for regex patterns.

use std::fmt;

/// The root AST node representing a complete regex pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ast {
    /// The root expression.
    pub expr: Expr,
    /// Flags that apply to the entire pattern.
    pub flags: Flags,
}

/// A regex expression node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// Empty expression (matches empty string).
    Empty,
    /// A single literal character.
    Literal(char),
    /// Concatenation of expressions (ab).
    Concat(Vec<Expr>),
    /// Alternation of expressions (a|b).
    Alt(Vec<Expr>),
    /// Repetition with quantifier.
    Repeat(Box<Repeat>),
    /// A group (capturing or non-capturing).
    Group(Box<Group>),
    /// A character class (`[abc]`, `[^abc]`, `[a-z]`).
    Class(Box<Class>),
    /// An anchor (^, $, \b, etc.).
    Anchor(Anchor),
    /// A lookaround assertion.
    Lookaround(Box<Lookaround>),
    /// A backreference (\1, \2, etc.).
    Backref(u32),
    /// Dot - matches any character (except newline by default).
    Dot,
    /// Unicode property (\p{Letter}, \P{Number}, etc.).
    UnicodeProperty {
        /// The property name.
        name: String,
        /// Whether this is negated (\P{...}).
        negated: bool,
    },
    /// Perl shorthand class (\w, \d, \s, \W, \D, \S).
    PerlClass(PerlClassKind),
}

/// Perl shorthand character classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerlClassKind {
    /// \d - digit
    Digit,
    /// \D - non-digit
    NotDigit,
    /// \w - word character
    Word,
    /// \W - non-word character
    NotWord,
    /// \s - whitespace
    Whitespace,
    /// \S - non-whitespace
    NotWhitespace,
}

impl Expr {
    /// Returns true if this expression can match the empty string.
    pub fn is_nullable(&self) -> bool {
        match self {
            Expr::Empty => true,
            Expr::Literal(_) => false,
            Expr::Concat(exprs) => exprs.iter().all(|e| e.is_nullable()),
            Expr::Alt(exprs) => exprs.iter().any(|e| e.is_nullable()),
            Expr::Repeat(rep) => rep.min == 0 || rep.expr.is_nullable(),
            Expr::Group(g) => g.expr.is_nullable(),
            Expr::Class(_) => false,
            Expr::Anchor(_) => true, // Anchors don't consume input
            Expr::Lookaround(_) => true, // Lookarounds don't consume input
            Expr::Backref(_) => false, // Could be nullable, but conservative
            Expr::Dot => false,
            Expr::UnicodeProperty { .. } => false,
            Expr::PerlClass(_) => false,
        }
    }
}

/// A repetition expression (*, +, ?, {n,m}).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repeat {
    /// The expression being repeated.
    pub expr: Expr,
    /// Minimum repetitions.
    pub min: u32,
    /// Maximum repetitions (None = unbounded).
    pub max: Option<u32>,
    /// Whether the quantifier is greedy.
    pub greedy: bool,
}

impl Repeat {
    /// Creates a new repetition.
    pub fn new(expr: Expr, min: u32, max: Option<u32>, greedy: bool) -> Self {
        Self { expr, min, max, greedy }
    }

    /// Creates a * quantifier (0 or more).
    pub fn star(expr: Expr, greedy: bool) -> Self {
        Self::new(expr, 0, None, greedy)
    }

    /// Creates a + quantifier (1 or more).
    pub fn plus(expr: Expr, greedy: bool) -> Self {
        Self::new(expr, 1, None, greedy)
    }

    /// Creates a ? quantifier (0 or 1).
    pub fn question(expr: Expr, greedy: bool) -> Self {
        Self::new(expr, 0, Some(1), greedy)
    }
}

/// A group expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    /// The grouped expression.
    pub expr: Expr,
    /// The kind of group.
    pub kind: GroupKind,
}

/// The kind of group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupKind {
    /// A capturing group with its index.
    Capturing(u32),
    /// A named capturing group with name and index.
    NamedCapturing {
        /// The name of the group.
        name: String,
        /// The numeric index of the group (1-based).
        index: u32,
    },
    /// A non-capturing group (?:...).
    NonCapturing,
}

/// A character class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Class {
    /// The ranges in this class.
    pub ranges: Vec<ClassRange>,
    /// Whether this class is negated ([^...]).
    pub negated: bool,
}

impl Class {
    /// Creates a new character class.
    pub fn new(ranges: Vec<ClassRange>, negated: bool) -> Self {
        Self { ranges, negated }
    }

    /// Creates a class from a single character.
    pub fn from_char(c: char) -> Self {
        Self {
            ranges: vec![ClassRange::single(c)],
            negated: false,
        }
    }
}

/// A range within a character class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassRange {
    /// Start of the range (inclusive).
    pub start: char,
    /// End of the range (inclusive).
    pub end: char,
}

impl ClassRange {
    /// Creates a new range.
    pub fn new(start: char, end: char) -> Self {
        Self { start, end }
    }

    /// Creates a single-character range.
    pub fn single(c: char) -> Self {
        Self { start: c, end: c }
    }

    /// Returns true if this range contains the given character.
    pub fn contains(&self, c: char) -> bool {
        c >= self.start && c <= self.end
    }
}

/// An anchor assertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    /// Start of string (^).
    StartOfString,
    /// End of string ($).
    EndOfString,
    /// Start of line (^ with multiline).
    StartOfLine,
    /// End of line ($ with multiline).
    EndOfLine,
    /// Word boundary (\b).
    WordBoundary,
    /// Not word boundary (\B).
    NotWordBoundary,
    /// Start of input (\A).
    StartOfInput,
    /// End of input (\z).
    EndOfInput,
    /// End of input before final newline (\Z).
    EndOfInputBeforeNewline,
}

/// A lookaround assertion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lookaround {
    /// The expression inside the lookaround.
    pub expr: Expr,
    /// The kind of lookaround.
    pub kind: LookaroundKind,
}

/// The kind of lookaround.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookaroundKind {
    /// Positive lookahead (?=...).
    PositiveLookahead,
    /// Negative lookahead (?!...).
    NegativeLookahead,
    /// Positive lookbehind (?<=...).
    PositiveLookbehind,
    /// Negative lookbehind (?<!...).
    NegativeLookbehind,
}

/// Regex flags/modifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Flags {
    /// Case-insensitive matching (i).
    pub case_insensitive: bool,
    /// Multi-line mode (m): ^ and $ match line boundaries.
    pub multi_line: bool,
    /// Dot-all mode (s): . matches newlines.
    pub dot_all: bool,
    /// Extended mode (x): ignore whitespace and allow comments.
    pub extended: bool,
    /// Unicode mode (u): enable full Unicode matching.
    pub unicode: bool,
}

impl Flags {
    /// Creates new default flags.
    pub fn new() -> Self {
        Self::default()
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Empty => Ok(()),
            Expr::Literal(c) => write!(f, "{}", escape_char(*c)),
            Expr::Concat(exprs) => {
                for expr in exprs {
                    write!(f, "{}", expr)?;
                }
                Ok(())
            }
            Expr::Alt(exprs) => {
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, "|")?;
                    }
                    write!(f, "{}", expr)?;
                }
                Ok(())
            }
            Expr::Repeat(rep) => {
                let needs_group = matches!(rep.expr, Expr::Concat(_) | Expr::Alt(_));
                if needs_group {
                    write!(f, "(?:{})", rep.expr)?;
                } else {
                    write!(f, "{}", rep.expr)?;
                }
                match (rep.min, rep.max) {
                    (0, None) => write!(f, "*")?,
                    (1, None) => write!(f, "+")?,
                    (0, Some(1)) => write!(f, "?")?,
                    (n, None) => write!(f, "{{{},}}", n)?,
                    (n, Some(m)) if n == m => write!(f, "{{{}}}", n)?,
                    (n, Some(m)) => write!(f, "{{{},{}}}", n, m)?,
                }
                if !rep.greedy {
                    write!(f, "?")?;
                }
                Ok(())
            }
            Expr::Group(g) => {
                match &g.kind {
                    GroupKind::Capturing(_) => write!(f, "({})", g.expr),
                    GroupKind::NamedCapturing { name, .. } => write!(f, "(?<{}>{})", name, g.expr),
                    GroupKind::NonCapturing => write!(f, "(?:{})", g.expr),
                }
            }
            Expr::Class(cls) => {
                write!(f, "[")?;
                if cls.negated {
                    write!(f, "^")?;
                }
                for range in &cls.ranges {
                    if range.start == range.end {
                        write!(f, "{}", escape_char(range.start))?;
                    } else {
                        write!(f, "{}-{}", escape_char(range.start), escape_char(range.end))?;
                    }
                }
                write!(f, "]")
            }
            Expr::Anchor(a) => write!(f, "{}", a),
            Expr::Lookaround(la) => {
                let prefix = match la.kind {
                    LookaroundKind::PositiveLookahead => "?=",
                    LookaroundKind::NegativeLookahead => "?!",
                    LookaroundKind::PositiveLookbehind => "?<=",
                    LookaroundKind::NegativeLookbehind => "?<!",
                };
                write!(f, "({}{})", prefix, la.expr)
            }
            Expr::Backref(n) => write!(f, "\\{}", n),
            Expr::Dot => write!(f, "."),
            Expr::UnicodeProperty { name, negated } => {
                if *negated {
                    write!(f, "\\P{{{}}}", name)
                } else {
                    write!(f, "\\p{{{}}}", name)
                }
            }
            Expr::PerlClass(kind) => {
                match kind {
                    PerlClassKind::Digit => write!(f, "\\d"),
                    PerlClassKind::NotDigit => write!(f, "\\D"),
                    PerlClassKind::Word => write!(f, "\\w"),
                    PerlClassKind::NotWord => write!(f, "\\W"),
                    PerlClassKind::Whitespace => write!(f, "\\s"),
                    PerlClassKind::NotWhitespace => write!(f, "\\S"),
                }
            }
        }
    }
}

impl fmt::Display for Anchor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Anchor::StartOfString | Anchor::StartOfLine => write!(f, "^"),
            Anchor::EndOfString | Anchor::EndOfLine => write!(f, "$"),
            Anchor::WordBoundary => write!(f, "\\b"),
            Anchor::NotWordBoundary => write!(f, "\\B"),
            Anchor::StartOfInput => write!(f, "\\A"),
            Anchor::EndOfInput => write!(f, "\\z"),
            Anchor::EndOfInputBeforeNewline => write!(f, "\\Z"),
        }
    }
}

fn escape_char(c: char) -> String {
    match c {
        '\\' | '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$' => {
            format!("\\{}", c)
        }
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        c if c.is_ascii_graphic() || c == ' ' => c.to_string(),
        c => format!("\\u{{{:X}}}", c as u32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeat_constructors() {
        let lit = Expr::Literal('a');

        let star = Repeat::star(lit.clone(), true);
        assert_eq!(star.min, 0);
        assert_eq!(star.max, None);
        assert!(star.greedy);

        let plus = Repeat::plus(lit.clone(), false);
        assert_eq!(plus.min, 1);
        assert_eq!(plus.max, None);
        assert!(!plus.greedy);

        let question = Repeat::question(lit, true);
        assert_eq!(question.min, 0);
        assert_eq!(question.max, Some(1));
    }

    #[test]
    fn test_class_range() {
        let range = ClassRange::new('a', 'z');
        assert!(range.contains('m'));
        assert!(!range.contains('A'));
    }

    #[test]
    fn test_is_nullable() {
        assert!(Expr::Empty.is_nullable());
        assert!(!Expr::Literal('a').is_nullable());
        assert!(Expr::Repeat(Box::new(Repeat::star(Expr::Literal('a'), true))).is_nullable());
    }
}
