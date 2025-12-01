//! TaggedNfaJit - JIT-compiled Tagged NFA for single-pass capture extraction.
//!
//! This module provides the public API for JIT-compiled pattern matching
//! with capture group support.

use crate::error::Result;
use crate::hir::CodepointClass;
use crate::nfa::Nfa;

use super::super::{
    analyze_liveness, NfaLiveness, TaggedNfaContext, PatternStep,
    StepInterpreter, TaggedNfaInterpreter, LookaroundCache,
};
use super::x86_64::TaggedNfaJitCompiler;

use dynasmrt::ExecutableBuffer;

/// Sentinel value returned by JIT code to indicate interpreter fallback.
pub const JIT_USE_INTERPRETER: i64 = -2;

/// A JIT-compiled Tagged NFA for single-pass capture extraction.
pub struct TaggedNfaJit {
    /// Executable buffer containing the JIT code.
    #[allow(dead_code)]
    code: ExecutableBuffer,
    /// Entry point for `find` (returns end position or -1, or -2 for interpreter fallback).
    find_fn: unsafe extern "sysv64" fn(*const u8, usize, *mut TaggedNfaContext) -> i64,
    /// Entry point for `captures` (writes to captures_out buffer, returns match end or -1/-2).
    /// Arguments: input_ptr, input_len, ctx, captures_out
    captures_fn: unsafe extern "sysv64" fn(*const u8, usize, *mut TaggedNfaContext, *mut i64) -> i64,
    /// Liveness analysis for sparse copying.
    liveness: NfaLiveness,
    /// The NFA (for interpreter fallback until full JIT is implemented).
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
    #[allow(dead_code)]
    codepoint_classes: Vec<Box<CodepointClass>>,
    /// Stored lookaround NFAs for JIT code to reference via helper functions.
    /// Index corresponds to the index stored in PatternStep::*Lookahead/*Lookbehind.
    #[allow(dead_code)]
    lookaround_nfas: Vec<Box<Nfa>>,
    /// Whether find_fn needs context (false for simple patterns).
    /// When false, we skip the expensive context setup in find().
    find_needs_ctx: bool,
    /// Pre-extracted pattern steps for fast fallback matching.
    /// Used when JIT returns JIT_USE_INTERPRETER to avoid creating
    /// a new interpreter on every call.
    fallback_steps: Option<Vec<PatternStep>>,
    /// Cached context to avoid allocation on every call.
    /// Uses RwLock for interior mutability since find/captures take &self.
    cached_ctx: std::sync::RwLock<Option<TaggedNfaContext>>,
    /// Cached captures buffer to avoid allocation.
    cached_captures_buf: std::sync::RwLock<Vec<i64>>,
}

impl TaggedNfaJit {
    /// Creates a new TaggedNfaJit from compiled components.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        code: ExecutableBuffer,
        find_fn: unsafe extern "sysv64" fn(*const u8, usize, *mut TaggedNfaContext) -> i64,
        captures_fn: unsafe extern "sysv64" fn(*const u8, usize, *mut TaggedNfaContext, *mut i64) -> i64,
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
    ) -> Self {
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
            cached_ctx: std::sync::RwLock::new(None),
            cached_captures_buf: std::sync::RwLock::new(Vec::new()),
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

            let result = unsafe {
                (self.find_fn)(input.as_ptr(), input.len(), std::ptr::null_mut())
            };

            #[cfg(debug_assertions)]
            {
                let ns = t0.elapsed().as_nanos() as u64;
                TOTAL_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
                let count = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if count % 10000 == 0 {
                    let total = TOTAL_NS.load(std::sync::atomic::Ordering::Relaxed);
                    eprintln!("[DEBUG] JIT fn call: {} calls, {}ns total, {}ns/call avg",
                        count, total, total / count);
                }
            }

            // Check for interpreter fallback (happens for standalone lookahead patterns)
            if result == JIT_USE_INTERPRETER {
                // Use fast StepInterpreter if we have fallback_steps
                if let Some(ref steps) = self.fallback_steps {
                    return StepInterpreter::find(steps, input);
                }
                // Otherwise fall back to Thompson NFA simulation
                let interp = TaggedNfaInterpreter::new(&self.nfa, &self.liveness);
                return interp.find(input);
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
            ctx.lookaround_cache = LookaroundCache::new(
                self.lookaround_count as usize,
                input.len() + 1,
            );
        }
        ctx.reset();

        let result = unsafe {
            (self.find_fn)(input.as_ptr(), input.len(), ctx)
        };

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
                let interp = TaggedNfaInterpreter::new(&self.nfa, &self.liveness);
                return interp.find(input);
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
            ctx.lookaround_cache = LookaroundCache::new(
                self.lookaround_count as usize,
                input.len() + 1,
            );
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
            // Fall back to interpreter
            let interp = TaggedNfaInterpreter::new(&self.nfa, &self.liveness);
            return interp.captures(input);
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

    /// Finds a match starting at or after the given position.
    pub fn find_at(&self, input: &[u8], start: usize) -> Option<(usize, usize)> {
        // For start=0, use the JIT find function directly
        if start == 0 {
            return self.find(input);
        }
        // For non-zero start, use StepInterpreter if we have steps
        if let Some(ref steps) = self.fallback_steps {
            return StepInterpreter::find_at(steps, input, start);
        }
        // Fall back to Thompson NFA simulation
        let interp = TaggedNfaInterpreter::new(&self.nfa, &self.liveness);
        for pos in start..=input.len() {
            if let Some(caps) = interp.captures_at(input, pos) {
                if let Some(full_match) = caps.first().and_then(|c| *c) {
                    return Some(full_match);
                }
            }
        }
        None
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
