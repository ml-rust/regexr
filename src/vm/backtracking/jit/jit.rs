//! Backtracking JIT public API.
//!
//! This module contains the BacktrackingJit struct and the public compile function.

use crate::error::Result;
use crate::hir::Hir;

use dynasmrt::ExecutableBuffer;

use super::x86_64::BacktrackingCompiler;

// Platform-specific function pointer type
#[cfg(target_os = "windows")]
type MatchFn = unsafe extern "win64" fn(*const u8, usize, *mut i64) -> i64;
#[cfg(not(target_os = "windows"))]
type MatchFn = unsafe extern "sysv64" fn(*const u8, usize, *mut i64) -> i64;

/// A compiled backtracking regex.
pub struct BacktrackingJit {
    /// Executable code buffer (kept alive for the function pointer).
    #[allow(dead_code)]
    pub(super) code: ExecutableBuffer,
    /// Entry point for matching.
    pub(super) match_fn: MatchFn,
    /// Number of capture groups.
    pub(super) capture_count: u32,
}

impl BacktrackingJit {
    /// Returns whether the pattern matches anywhere in the input.
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.find(input).is_some()
    }

    /// Finds the first match, returning (start, end).
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        let num_slots = (self.capture_count as usize + 1) * 2;
        let mut captures: Vec<i64> = vec![-1; num_slots];

        let result = unsafe { (self.match_fn)(input.as_ptr(), input.len(), captures.as_mut_ptr()) };

        if result >= 0 {
            // Group 0 contains the full match
            let start = captures[0];
            let end = captures[1];
            if start >= 0 && end >= 0 {
                Some((start as usize, end as usize))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Returns capture groups for the first match.
    pub fn captures(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        let num_slots = (self.capture_count as usize + 1) * 2;
        let mut captures_buf: Vec<i64> = vec![-1; num_slots];

        let result =
            unsafe { (self.match_fn)(input.as_ptr(), input.len(), captures_buf.as_mut_ptr()) };

        if result >= 0 {
            let mut captures = Vec::with_capacity(self.capture_count as usize + 1);
            for i in 0..=self.capture_count as usize {
                let start = captures_buf[i * 2];
                let end = captures_buf[i * 2 + 1];
                if start >= 0 && end >= 0 {
                    captures.push(Some((start as usize, end as usize)));
                } else {
                    captures.push(None);
                }
            }
            Some(captures)
        } else {
            None
        }
    }

    /// Finds a match starting at or after the given position.
    pub fn find_at(&self, input: &[u8], start: usize) -> Option<(usize, usize)> {
        if start >= input.len() {
            return None;
        }
        // For now, search from start position
        let slice = &input[start..];
        self.find(slice).map(|(s, e)| (s + start, e + start))
    }

    /// Debug method to see raw results
    #[cfg(test)]
    pub fn debug_match(&self, input: &[u8]) -> (i64, Vec<i64>) {
        let num_slots = (self.capture_count as usize + 1) * 2;
        let mut captures: Vec<i64> = vec![-1; num_slots];

        let result = unsafe { (self.match_fn)(input.as_ptr(), input.len(), captures.as_mut_ptr()) };

        (result, captures)
    }
}

/// Compiles a HIR pattern to a backtracking JIT.
///
/// Returns an error for patterns that require complex backtracking within captures,
/// such as `(a+)\1` where the backref refers to a capture containing an unbounded
/// repetition. These patterns should fall back to PikeVM.
pub fn compile_backtracking(hir: &Hir) -> Result<BacktrackingJit> {
    // Note: We now support backrefs to captures with unbounded repetitions like (\w+)\1.
    // The greedy repetition code properly saves choice points and updates capture ends
    // during backtracking.

    let compiler = BacktrackingCompiler::new(hir)?;
    compiler.compile()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn compile_pattern(pattern: &str) -> Result<BacktrackingJit> {
        let ast = parse(pattern)?;
        let hir = translate(&ast)?;
        compile_backtracking(&hir)
    }

    #[test]
    fn test_literal_debug() {
        let jit = compile_pattern("hello").unwrap();
        let (result, caps) = jit.debug_match(b"hello");
        println!("result: {}, caps: {:?}", result, caps);
        assert!(result >= 0, "Expected match, got result={}", result);
    }

    #[test]
    fn test_literal() {
        let jit = compile_pattern("hello").unwrap();
        assert!(jit.is_match(b"hello"));
        assert!(jit.is_match(b"say hello world"));
        assert!(!jit.is_match(b"helo"));
    }

    #[test]
    fn test_simple_backref() {
        let jit = compile_pattern(r"(a)\1").unwrap();

        // Debug: check what we match
        let (result_aa, caps_aa) = jit.debug_match(b"aa");
        println!("(a)\\1 on 'aa': result={}, caps={:?}", result_aa, caps_aa);

        let (result_ab, caps_ab) = jit.debug_match(b"ab");
        println!("(a)\\1 on 'ab': result={}, caps={:?}", result_ab, caps_ab);

        let (result_a, caps_a) = jit.debug_match(b"a");
        println!("(a)\\1 on 'a': result={}, caps={:?}", result_a, caps_a);

        assert!(jit.is_match(b"aa"), "Should match 'aa'");
        assert!(!jit.is_match(b"ab"), "Should NOT match 'ab'");
        assert!(!jit.is_match(b"a"), "Should NOT match 'a'");
    }

    #[test]
    fn test_quoted_string() {
        let jit = compile_pattern(r#"(['"])[^'"]*\1"#).unwrap();

        let (r1, c1) = jit.debug_match(br#""hello""#);
        println!(r#"['"][^'"]*\1 on "hello": result={}, caps={:?}"#, r1, c1);

        let (r2, c2) = jit.debug_match(b"'world'");
        println!(r#"['"][^'"]*\1 on 'world': result={}, caps={:?}"#, r2, c2);

        let (r3, c3) = jit.debug_match(br#""mixed'"#);
        println!(r#"['"][^'"]*\1 on "mixed': result={}, caps={:?}"#, r3, c3);

        let (r4, c4) = jit.debug_match(b"'mixed\"");
        println!(r#"['"][^'"]*\1 on 'mixed": result={}, caps={:?}"#, r4, c4);

        assert!(jit.is_match(br#""hello""#), "Should match \"hello\"");
        assert!(jit.is_match(b"'world'"), "Should match 'world'");
        assert!(!jit.is_match(br#""mixed'"#), "Should NOT match \"mixed'");
        assert!(!jit.is_match(b"'mixed\""), "Should NOT match 'mixed\"");
    }

    #[test]
    fn test_alternation_backref() {
        let jit = compile_pattern(r"(a|b)\1").unwrap();

        let (result_aa, caps_aa) = jit.debug_match(b"aa");
        println!("(a|b)\\1 on 'aa': result={}, caps={:?}", result_aa, caps_aa);

        let (result_bb, caps_bb) = jit.debug_match(b"bb");
        println!("(a|b)\\1 on 'bb': result={}, caps={:?}", result_bb, caps_bb);

        let (result_ab, caps_ab) = jit.debug_match(b"ab");
        println!("(a|b)\\1 on 'ab': result={}, caps={:?}", result_ab, caps_ab);

        let (result_ba, caps_ba) = jit.debug_match(b"ba");
        println!("(a|b)\\1 on 'ba': result={}, caps={:?}", result_ba, caps_ba);

        assert!(jit.is_match(b"aa"), "Should match 'aa'");
        assert!(jit.is_match(b"bb"), "Should match 'bb'");
        assert!(!jit.is_match(b"ab"), "Should NOT match 'ab'");
        assert!(!jit.is_match(b"ba"), "Should NOT match 'ba'");
    }

    #[test]
    fn test_captures() {
        let jit = compile_pattern(r"(a)(b)\2\1").unwrap();
        let caps = jit.captures(b"abba").unwrap();
        assert_eq!(caps[0], Some((0, 4))); // Full match
        assert_eq!(caps[1], Some((0, 1))); // Group 1: "a"
        assert_eq!(caps[2], Some((1, 2))); // Group 2: "b"
    }
}
