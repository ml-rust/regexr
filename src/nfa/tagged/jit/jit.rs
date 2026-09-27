//! TaggedNfaJit - JIT-compiled Tagged NFA for single-pass capture extraction.
//!
//! This module provides the public API for JIT-compiled pattern matching
//! with capture group support.

use crate::error::Result;
use crate::hir::CodepointClass;
use crate::nfa::Nfa;
use crate::vm::{PikeVm, PikeVmContext};

use super::super::{
    analyze_liveness, LookaroundCache, NfaLiveness, PatternStep, TaggedNfa, TaggedNfaContext,
};

#[cfg(target_arch = "x86_64")]
use super::x86_64::TaggedNfaJitCompiler;

#[cfg(target_arch = "aarch64")]
use super::aarch64::TaggedNfaJitCompiler;

use dynasmrt::ExecutableBuffer;

/// Sentinel value returned by JIT code to indicate interpreter fallback.
pub const JIT_USE_INTERPRETER: i64 = -2;

// Platform-specific function pointer types for JIT code
// x86_64 Windows uses Microsoft x64 ABI
#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
type FindFn = unsafe extern "win64" fn(*const u8, usize, *mut TaggedNfaContext) -> i64;
#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
type CapturesFn =
    unsafe extern "win64" fn(*const u8, usize, *mut TaggedNfaContext, *mut i64) -> i64;

// x86_64 Unix uses System V AMD64 ABI
#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
type FindFn = unsafe extern "sysv64" fn(*const u8, usize, *mut TaggedNfaContext) -> i64;
#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
type CapturesFn =
    unsafe extern "sysv64" fn(*const u8, usize, *mut TaggedNfaContext, *mut i64) -> i64;

// ARM64 uses AAPCS64 on all platforms (extern "C")
#[cfg(target_arch = "aarch64")]
type FindFn = unsafe extern "C" fn(*const u8, usize, *mut TaggedNfaContext) -> i64;
#[cfg(target_arch = "aarch64")]
type CapturesFn = unsafe extern "C" fn(*const u8, usize, *mut TaggedNfaContext, *mut i64) -> i64;

/// A JIT-compiled Tagged NFA for single-pass capture extraction.
pub struct TaggedNfaJit {
    /// Executable buffer containing the JIT code.
    #[allow(dead_code)]
    code: ExecutableBuffer,
    /// Entry point for `find` (returns end position or -1, or -2 for interpreter fallback).
    find_fn: FindFn,
    /// Entry point for `captures` (writes to captures_out buffer, returns match end or -1/-2).
    /// Arguments: input_ptr, input_len, ctx, captures_out
    captures_fn: CapturesFn,
    /// Liveness analysis for sparse copying.
    liveness: NfaLiveness,
    /// The NFA (kept for reference, PikeVm is used for fallback).
    #[allow(dead_code)]
    nfa: Nfa,
    /// Number of capture groups.
    capture_count: u32,
    /// Number of NFA states.
    state_count: usize,
    /// Number of lookarounds (for cache sizing).
    lookaround_count: u32,
    /// Capture stride (slots per thread).
    stride: usize,
    /// Stored CodepointClasses for JIT code to reference.
    /// These must outlive the JIT code since their pointers are embedded in the generated assembly.
    #[allow(dead_code, clippy::vec_box)]
    codepoint_classes: Vec<Box<CodepointClass>>,
    /// Stored lookaround NFAs for JIT code to reference via helper functions.
    /// Index corresponds to the index stored in PatternStep::*Lookahead/*Lookbehind.
    #[allow(dead_code, clippy::vec_box)]
    lookaround_nfas: Vec<Box<Nfa>>,
    /// Whether find_fn needs context (false for simple patterns).
    /// When false, we skip the expensive context setup in find().
    find_needs_ctx: bool,
    /// Whether the pattern depends on absolute position or characters to the
    /// left of the match start: a start anchor (`^` / `\A`), `\b`/`\B`, or a
    /// lookbehind. `find_at(start>0)` slices the input (the JIT has no start
    /// offset), which would make the slice's first byte look like the start of
    /// text and hide any preceding context — wrong for these. When set, such
    /// calls go to the interpreter, which matches at an absolute position.
    needs_left_context: bool,
    /// Pre-extracted pattern steps for fast fallback matching.
    /// Used when JIT returns JIT_USE_INTERPRETER to avoid creating
    /// a new interpreter on every call.
    fallback_steps: Option<Vec<PatternStep>>,
    /// Cached context to avoid allocation on every call.
    /// Uses RwLock for interior mutability since find/captures take &self.
    cached_ctx: std::sync::RwLock<Option<TaggedNfaContext>>,
    /// Cached captures buffer to avoid allocation.
    cached_captures_buf: std::sync::RwLock<Vec<i64>>,
    /// PikeVm for interpreter fallback.
    fallback_vm: PikeVm,
    /// Cached PikeVm context for fallback.
    fallback_vm_ctx: std::sync::RwLock<PikeVmContext>,
}

impl TaggedNfaJit {
    /// Creates a new TaggedNfaJit from compiled components.
    #[allow(clippy::too_many_arguments, clippy::vec_box)]
    pub(super) fn new(
        code: ExecutableBuffer,
        find_fn: FindFn,
        captures_fn: CapturesFn,
        liveness: NfaLiveness,
        nfa: Nfa,
        capture_count: u32,
        state_count: usize,
        lookaround_count: u32,
        stride: usize,
        codepoint_classes: Vec<Box<CodepointClass>>,
        lookaround_nfas: Vec<Box<Nfa>>,
        find_needs_ctx: bool,
        fallback_steps: Option<Vec<PatternStep>>,
        needs_left_context: bool,
    ) -> Self {
        // Create PikeVm for fallback
        let fallback_vm = PikeVm::new(nfa.clone());
        let fallback_vm_ctx = std::sync::RwLock::new(fallback_vm.create_context());

        Self {
            code,
            find_fn,
            captures_fn,
            liveness,
            nfa,
            capture_count,
            state_count,
            lookaround_count,
            stride,
            codepoint_classes,
            lookaround_nfas,
            find_needs_ctx,
            fallback_steps,
            needs_left_context,
            cached_ctx: std::sync::RwLock::new(None),
            cached_captures_buf: std::sync::RwLock::new(Vec::new()),
            fallback_vm,
            fallback_vm_ctx,
        }
    }

    /// Returns whether the pattern matches the input.
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.find(input).is_some()
    }

    /// Finds the first match, returning (start, end).
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> {
        // Fast path: if find_fn doesn't need context, skip all context setup
        if !self.find_needs_ctx {
            // Debug timing to isolate JIT call overhead
            #[cfg(debug_assertions)]
            static CALL_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            #[cfg(debug_assertions)]
            static TOTAL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

            #[cfg(debug_assertions)]
            let t0 = std::time::Instant::now();

            let result =
                unsafe { (self.find_fn)(input.as_ptr(), input.len(), std::ptr::null_mut()) };

            #[cfg(debug_assertions)]
            {
                let ns = t0.elapsed().as_nanos() as u64;
                TOTAL_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
                let count = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                #[allow(clippy::manual_is_multiple_of)]
                if count % 10000 == 0 {
                    let total = TOTAL_NS.load(std::sync::atomic::Ordering::Relaxed);
                    eprintln!(
                        "[DEBUG] JIT fn call: {} calls, {}ns total, {}ns/call avg",
                        count,
                        total,
                        total / count
                    );
                }
            }

            // Check for interpreter fallback (happens for standalone lookahead patterns)
            if result == JIT_USE_INTERPRETER {
                // Use fast TaggedNfa if we have fallback_steps
                if let Some(ref steps) = self.fallback_steps {
                    return TaggedNfa::find_at_bounded(steps, &self.fallback_vm, input, 0);
                }
                // Otherwise fall back to PikeVm
                return self.fallback_vm.find(input);
            }

            return if result >= 0 {
                let start = (result as u64 >> 32) as usize;
                let end = (result as u64 & 0xFFFF_FFFF) as usize;
                Some((start, end))
            } else {
                None
            };
        }

        // Slow path: patterns that need context (backrefs, complex lookarounds, etc.)
        let mut ctx_ref = self.cached_ctx.write().unwrap();
        let ctx = ctx_ref.get_or_insert_with(|| {
            TaggedNfaContext::new(
                self.capture_count,
                self.state_count,
                self.lookaround_count as usize,
                256, // Initial size, will grow if needed
            )
        });

        // Ensure lookaround cache is large enough for this input
        if ctx.lookaround_cache.max_len < input.len() + 1 {
            ctx.lookaround_cache =
                LookaroundCache::new(self.lookaround_count as usize, input.len() + 1);
        }
        ctx.reset();

        let result = unsafe { (self.find_fn)(input.as_ptr(), input.len(), ctx) };

        if result == JIT_USE_INTERPRETER {
            // find_fn doesn't support this pattern (e.g., backrefs need capture tracking).
            // Use captures_fn instead, which has full JIT support including backrefs.
            // We just need the full match (group 0), not all captures.
            let num_slots = (self.capture_count as usize + 1) * 2;

            // Use cached captures buffer
            let mut captures_buf = self.cached_captures_buf.write().unwrap();
            if captures_buf.len() < num_slots {
                captures_buf.resize(num_slots, -1);
            }
            // Reset buffer
            for slot in captures_buf.iter_mut() {
                *slot = -1;
            }
            ctx.reset();

            let captures_result = unsafe {
                (self.captures_fn)(input.as_ptr(), input.len(), ctx, captures_buf.as_mut_ptr())
            };

            if captures_result == JIT_USE_INTERPRETER {
                // captures_fn also needs interpreter fallback
                return self.fallback_vm.find(input);
            }

            if captures_result >= 0 {
                // Group 0 is at slots [0, 1] = (start, end)
                let start = captures_buf[0];
                let end = captures_buf[1];
                if start >= 0 && end >= 0 {
                    return Some((start as usize, end as usize));
                }
            }
            return None;
        }

        if result >= 0 {
            // JIT returns (start << 32 | end)
            let start = (result as u64 >> 32) as usize;
            let end = (result as u64 & 0xFFFF_FFFF) as usize;
            Some((start, end))
        } else {
            None
        }
    }

    /// Returns capture groups for the first match.
    pub fn captures(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        self.captures_from(input, 0)
    }

    /// Returns capture groups for the first match starting at or after `start`.
    ///
    /// Patterns that read left context (`^`, `\b`, lookbehind) are routed to the
    /// interpreter, which matches at an absolute position with the full input
    /// visible; the rest can run on the suffix slice (see [`TaggedNfaJit::find_at`]).
    pub fn captures_from(&self, input: &[u8], start: usize) -> Option<Vec<Option<(usize, usize)>>> {
        if start > input.len() {
            return None;
        }
        if start > 0 && self.needs_left_context {
            return self.fallback_captures_from(input, start);
        }
        self.captures_in_suffix(&input[start..]).map(|caps| {
            caps.into_iter()
                .map(|slot| slot.map(|(s, e)| (start + s, start + e)))
                .collect()
        })
    }

    /// Extracts captures with the fallback PikeVm, searching from `from` and
    /// reusing the cached context.
    ///
    /// Searching is what makes this equivalent to `find`: the anchored
    /// `captures_with_context` matches only at the position it is handed, so a
    /// single call at `from` would answer "no match" for every pattern whose
    /// match starts later. The unanchored entry point does the search in one
    /// pass instead of one full VM run per candidate position.
    fn fallback_captures_from(
        &self,
        input: &[u8],
        from: usize,
    ) -> Option<Vec<Option<(usize, usize)>>> {
        let mut vm_ctx = self.fallback_vm_ctx.write().unwrap();
        self.fallback_vm
            .captures_unanchored_with_context(input, &mut vm_ctx, from)
    }

    /// Runs the capture-extracting JIT over `input`, which is either the whole
    /// haystack or a suffix of it; positions are relative to `input`.
    fn captures_in_suffix(&self, input: &[u8]) -> Option<Vec<Option<(usize, usize)>>> {
        // Get or create cached context
        let mut ctx_ref = self.cached_ctx.write().unwrap();
        let ctx = ctx_ref.get_or_insert_with(|| {
            TaggedNfaContext::new(
                self.capture_count,
                self.state_count,
                self.lookaround_count as usize,
                256,
            )
        });

        // Ensure lookaround cache is large enough for this input
        if ctx.lookaround_cache.max_len < input.len() + 1 {
            ctx.lookaround_cache =
                LookaroundCache::new(self.lookaround_count as usize, input.len() + 1);
        }
        ctx.reset();

        // Use cached captures buffer
        let num_slots = (self.capture_count as usize + 1) * 2;
        let mut captures_buf = self.cached_captures_buf.write().unwrap();
        if captures_buf.len() < num_slots {
            captures_buf.resize(num_slots, -1);
        }
        // Reset buffer
        for slot in captures_buf.iter_mut() {
            *slot = -1;
        }

        let result = unsafe {
            (self.captures_fn)(input.as_ptr(), input.len(), ctx, captures_buf.as_mut_ptr())
        };

        if result == JIT_USE_INTERPRETER {
            // This is the *normal* path for a pattern with no capture groups: the
            // captures JIT is a stub for those (there is nothing to track), so it
            // always requests the interpreter.
            return self.fallback_captures_from(input, 0);
        }

        if result >= 0 {
            let mut captures = Vec::with_capacity(self.capture_count as usize + 1);

            // Read captures from the buffer
            for i in 0..=self.capture_count as usize {
                let start_idx = i * 2;
                let end_idx = i * 2 + 1;
                let start = captures_buf[start_idx];
                let end = captures_buf[end_idx];
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

    /// Returns the liveness analysis for this NFA.
    pub fn liveness(&self) -> &NfaLiveness {
        &self.liveness
    }

    /// Returns the capture count.
    pub fn capture_count(&self) -> u32 {
        self.capture_count
    }

    /// Returns the capture stride (slots per thread).
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Finds a match beginning exactly at `pos`, returning (pos, end).
    ///
    /// The anchored counterpart of [`TaggedNfaJit::find_at`]. The generated code
    /// has no anchored entry point — `find_fn` scans forward from the pointer it
    /// is handed — so this runs the interpreter, which matches at an absolute
    /// position with the full input visible.
    pub(crate) fn match_at(&self, input: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos > input.len() {
            return None;
        }
        // Only start at UTF-8 codepoint boundaries (see `is_utf8_boundary`).
        if !crate::nfa::is_utf8_boundary(input, pos) {
            return None;
        }
        if let Some(ref steps) = self.fallback_steps {
            return TaggedNfa::match_at_bounded(steps, &self.fallback_vm, input, pos);
        }
        // `find_at`, not `find_from`: the match must begin at `pos`.
        self.fallback_vm.find_at(input, pos)
    }

    /// Finds a match starting at or after the given position.
    pub fn find_at(&self, input: &[u8], start: usize) -> Option<(usize, usize)> {
        if start > input.len() {
            return None;
        }
        if start == 0 {
            return self.find(input);
        }

        // The JIT has no start-offset parameter, so `find_at(start>0)` slices the
        // input and runs from the slice. That hides everything before `start`,
        // which is wrong for patterns anchored at the text start or that read
        // left context (`\b`, lookbehind): the slice's first byte would look like
        // the start of text. Route those to the interpreter, which matches at an
        // absolute position with the full input visible.
        if self.needs_left_context {
            if let Some(ref steps) = self.fallback_steps {
                return TaggedNfa::find_at_bounded(steps, &self.fallback_vm, input, start);
            }
            return self.fallback_vm.find_from(input, start);
        }

        // JIT supports start offset by passing sliced input pointer
        // and adjusting the returned positions
        let slice_ptr = unsafe { input.as_ptr().add(start) };
        let slice_len = input.len() - start;

        // Fast path: if find_fn doesn't need context
        if !self.find_needs_ctx {
            let result = unsafe { (self.find_fn)(slice_ptr, slice_len, std::ptr::null_mut()) };

            if result == JIT_USE_INTERPRETER {
                // Use fast TaggedNfa if we have fallback_steps
                if let Some(ref steps) = self.fallback_steps {
                    return TaggedNfa::find_at_bounded(steps, &self.fallback_vm, input, start);
                }
                return self.fallback_vm.find_from(input, start);
            }

            return if result >= 0 {
                let rel_start = (result as u64 >> 32) as usize;
                let rel_end = (result as u64 & 0xFFFF_FFFF) as usize;
                Some((start + rel_start, start + rel_end))
            } else {
                None
            };
        }

        // Slow path: patterns that need context
        let mut ctx_ref = self.cached_ctx.write().unwrap();
        let ctx = ctx_ref.get_or_insert_with(|| {
            TaggedNfaContext::new(
                self.capture_count,
                self.state_count,
                self.lookaround_count as usize,
                256,
            )
        });

        if ctx.lookaround_cache.max_len < slice_len + 1 {
            ctx.lookaround_cache =
                LookaroundCache::new(self.lookaround_count as usize, slice_len + 1);
        }
        ctx.reset();

        let result = unsafe { (self.find_fn)(slice_ptr, slice_len, ctx) };

        if result == JIT_USE_INTERPRETER {
            if let Some(ref steps) = self.fallback_steps {
                return TaggedNfa::find_at_bounded(steps, &self.fallback_vm, input, start);
            }
            return self.fallback_vm.find_from(input, start);
        }

        if result >= 0 {
            let rel_start = (result as u64 >> 32) as usize;
            let rel_end = (result as u64 & 0xFFFF_FFFF) as usize;
            Some((start + rel_start, start + rel_end))
        } else {
            None
        }
    }
}

/// Compiles an NFA to a Tagged NFA JIT.
pub fn compile_tagged_nfa(nfa: &Nfa) -> Result<TaggedNfaJit> {
    let liveness = analyze_liveness(nfa);
    compile_tagged_nfa_with_liveness(nfa.clone(), liveness)
}

/// Compiles an NFA with pre-computed liveness analysis.
pub fn compile_tagged_nfa_with_liveness(nfa: Nfa, liveness: NfaLiveness) -> Result<TaggedNfaJit> {
    TaggedNfaJitCompiler::compile(nfa, liveness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::translate;
    use crate::parser::parse;

    fn compile_jit(pattern: &str) -> TaggedNfaJit {
        let ast = parse(pattern).unwrap();
        let hir = translate(&ast).unwrap();
        let nfa = crate::nfa::compile(&hir).unwrap();
        compile_tagged_nfa(&nfa).expect("failed to JIT-compile pattern")
    }

    /// `match_at` answers about `pos` alone, unlike `find_at`, which searches
    /// forward from it.
    #[test]
    fn match_at_is_anchored_to_the_position() {
        // A greedy repeat next to a lookaround is deferred to the interpreter,
        // so this stays a single class to exercise the JIT wrapper itself.
        let jit = compile_jit(r"(?<=@)\w");
        let input: &[u8] = b"user @name";
        assert_eq!(jit.match_at(input, 0), None);
        assert_eq!(jit.match_at(input, 6), Some((6, 7)));
        assert_eq!(jit.find_at(input, 0), Some((6, 7)));
        assert_eq!(jit.match_at(input, input.len() + 1), None);
    }
}
