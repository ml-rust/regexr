# Architecture Overview

regexr is a regex engine with multiple execution backends optimized for different pattern types. This document describes the overall architecture and how engines are selected.

## Core Components

### 1. Compilation Pipeline

The regex compilation follows this pipeline:

```
Pattern String → AST → HIR → NFA → Engine-Specific Representation
```

- **AST (Abstract Syntax Tree)**: Initial parse tree from the pattern string
- **HIR (High-Level IR)**: Byte-oriented intermediate representation with pattern analysis
- **NFA (Non-deterministic Finite Automaton)**: State machine representation
- **Engine**: Execution backend selected based on pattern characteristics

### 2. Key Types

#### Public API (`src/lib.rs`)
- `Regex`: Main public API for pattern matching
- `RegexBuilder`: Builder for configuring compilation options
- `Match`: Represents a single match in the input
- `Captures`: Capture groups from a match

#### Internal Representation (`src/engine/executor.rs`)
- `CompiledRegex`: Internal compiled pattern with selected engine
- `CompiledInner`: Enum of all possible execution engines

#### Pattern Analysis (`src/hir/mod.rs`)
- `Hir`: High-level intermediate representation
- `HirProps`: Derived properties (backreferences, lookaround, anchors, etc.)
- `HirExpr`: Expression nodes in the HIR tree

#### State Machines
- `Nfa` (`src/nfa/state.rs`): NFA state machine
- `LazyDfa` (`src/dfa/lazy/`): On-demand DFA construction
- `EagerDfa` (`src/dfa/eager/`): Pre-materialized DFA

## Execution Engines

### Non-JIT Engines (Interpreted)

#### PikeVM (`src/vm/pike/`)
- Thread-based NFA simulation
- Required for patterns with lookaround
- Supports all regex features including backreferences
- Uses Thompson NFA construction with epsilon transitions

#### ShiftOr (`src/vm/shift_or/`)
- Bit-parallel NFA simulation using bitwise operations
- Fast for small patterns (≤64 NFA states)
- Cannot handle anchors or word boundaries
- Excellent for simple alternations and character classes

#### LazyDFA (`src/dfa/lazy/`)
- Builds DFA states on-demand during matching
- Caches states for reuse
- Falls back to NFA simulation for complex patterns
- Good general-purpose engine

#### EagerDFA (`src/dfa/eager/`)
- Pre-computes DFA states during compilation
- Used for patterns with word boundaries or anchors
- Faster startup than LazyDFA for bounded patterns

#### BacktrackingVm (`src/vm/backtracking/`)
- PCRE-style backtracking engine
- Required for patterns with backreferences
- Single-pass capture extraction
- Non-JIT version of BacktrackingJit

#### CodepointClassMatcher (`src/vm/codepoint_class.rs`)
- Optimized for single Unicode character class patterns
- Operates at codepoint level instead of byte level
- Fast for patterns like `\p{Greek}` or `[α-ω]`

### JIT Engines (Native Code Generation)

Available on x86-64 (Linux, macOS, Windows) and ARM64 (Linux, macOS) with the `jit` feature.

#### DFA JIT (`src/jit/`)
- Compiles DFA to native machine code
- Benefits from SIMD prefiltering for literal prefixes
- Fast path for most patterns without backreferences or lookaround
- Re-export hub in `src/jit/mod.rs`

#### BacktrackingJit (`src/vm/backtracking/jit/`)
- JIT-compiled backtracking engine
- Required for patterns with backreferences
- Uses native code for faster backtracking

#### TaggedNfa (`src/nfa/tagged/jit/`)
- JIT-compiled NFA with liveness analysis
- Used for patterns with lookaround or non-greedy quantifiers
- Efficient single-pass capture extraction
- Preserves NFA match preference

#### JitShiftOr (`src/vm/shift_or/jit/`)
- JIT-compiled bit-parallel matcher
- Optimized for patterns with alternations
- Used when no effective prefilter is available

## Engine Selection Logic

### HIR Properties Analysis

During HIR construction, the following properties are analyzed:

- `has_backrefs`: Pattern contains backreferences (`\1`, `\2`, etc.)
- `has_lookaround`: Pattern contains lookahead/lookbehind assertions
- `has_anchors`: Pattern contains `^` or `$`
- `has_word_boundary`: Pattern contains `\b` or `\B`
- `has_non_greedy`: Pattern contains non-greedy quantifiers (`*?`, `+?`, etc.)
- `has_large_unicode_class`: Pattern contains large Unicode classes that cause DFA state explosion
- `capture_count`: Number of capture groups
- `min_len`/`max_len`: Match length bounds

### Selection in Non-JIT Mode

```rust
if has_backrefs {
    BacktrackingVm
} else if has_lookaround || has_non_greedy {
    PikeVm
} else if is_single_codepoint_class {
    CodepointClassMatcher
} else if pattern_size <= 64 && !has_word_boundary && !has_anchors {
    ShiftOr
} else if has_word_boundary || has_anchors {
    EagerDfa
} else {
    LazyDfa
}
```

### Selection in JIT Mode

JIT mode adds prefilter effectiveness analysis:

```rust
if has_backrefs {
    BacktrackingJit
} else if has_lookaround {
    PikeVm  // Lookaround requires NFA semantics
} else if has_non_greedy {
    TaggedNfa  // NFA preserves match preference
} else if has_large_unicode_class {
    LazyDfa  // Avoids DFA state explosion
} else if !has_effective_prefilter && shift_or_compatible {
    JitShiftOr  // Bit-parallel efficient for alternations
} else {
    DFA JIT  // Benefits from SIMD candidate filtering
}
```

### Prefilter Effectiveness

A prefilter is considered "effective" when it has good selectivity:

- **SingleByte**: Searches for a single literal byte
- **Literal**: Searches for a literal string
- **Teddy**: SIMD-accelerated multi-pattern search (2-8 patterns)
- **AhoCorasick**: Multi-pattern search for larger sets

Patterns starting with alternations or character classes often have no effective prefilter, making bit-parallel engines like ShiftOr or JitShiftOr more suitable.

## Prefilter Architecture

The prefilter system (`src/literal/`) provides fast candidate scanning:

1. **Literal extraction**: Identifies required literals from the pattern
2. **Prefilter selection**: Chooses optimal prefilter based on literal count and selectivity
3. **Candidate filtering**: Uses SIMD when available to skip to match candidates
4. **Verification**: Full engine confirms matches at candidate positions

### Prefilter Types

- **SingleByte**: Uses `memchr` for single-byte search
- **Literal**: Uses `memmem` for multi-byte literal search
- **Teddy**: Custom SIMD implementation for 2-8 patterns (requires `simd` feature)
- **AhoCorasick**: Delegates to `aho-corasick` crate for larger sets
- **FullMatch**: Teddy variant that validates full matches without verification

## Memory Management

### Lazy Initialization

Several components use lazy initialization to avoid unnecessary allocations:

- **Capture NFA**: Only built when `captures()` is called
- **PikeVm context**: Pre-allocated storage for NFA simulation
- **LazyDFA states**: Built on-demand during matching

### RefCell Usage

`CompiledRegex` uses `RefCell` for interior mutability:

```rust
pub struct CompiledRegex {
    inner: CompiledInner,
    prefilter: Prefilter,
    capture_nfa: RefCell<Option<Nfa>>,
    capture_vm: RefCell<Option<PikeVm>>,
    capture_ctx: RefCell<Option<PikeVmContext>>,
    // ...
}
```

This allows lazy initialization without requiring `&mut self`, enabling the public API to use `&self` for all operations.

## Unicode Support

Unicode support is provided through generated tables from the Unicode Character Database:

- **Character classes**: `\p{Letter}`, `\p{Number}`, etc.
- **Scripts**: `\p{Greek}`, `\p{Cyrillic}`, etc.
- **Case folding**: Case-insensitive matching for Unicode characters
- **Word boundaries**: Unicode-aware `\b` and `\B`

Tables are generated by `scripts/update_unicode_tables.sh` and stored in `src/hir/unicode_data.rs`.

## Test Organization

Tests are organized by category in `tests/`:

- `api/`: Public API tests (matching, captures, replacement)
- `engines/`: Backend-specific tests (ShiftOr, LazyDFA, etc.)
- `features/`: Feature tests (lookaround, backreferences, case sensitivity)
- `patterns/`: Real-world patterns (email, URL, phone numbers)
- `unicode/`: Unicode property and script tests

## Build Features

### Feature Flag Organization

```toml
[features]
default = ["simd"]
simd = []
jit = ["dynasm", "dynasmrt"]
full = ["jit", "simd"]
```

- **default**: SIMD acceleration only
- **jit**: Adds JIT compilation (x86-64 and ARM64)
- **full**: Both JIT and SIMD

### Conditional Compilation

JIT engines use conditional compilation:

```rust
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod jit;
```

This ensures JIT code is only compiled on supported platforms (x86-64 and ARM64).

## Performance Considerations

### Design Goals

- Non-JIT mode should match or beat `regex` crate performance
- JIT mode should be competitive with `pcre2-jit`
- JIT provides significant speedup for patterns with effective prefilters

### Optimization Strategies

1. **Engine specialization**: Different engines for different pattern types
2. **Prefilter acceleration**: SIMD literal search when applicable
3. **Lazy compilation**: Build expensive structures only when needed
4. **Native code generation**: JIT compilation for hot paths
5. **Prefix optimization**: Trie-based alternation merging for tokenizers

### When to Use JIT

JIT compilation is beneficial when:

- Pattern will be matched many times (amortizes compilation cost)
- Pattern has effective prefilters (benefits from SIMD + native code)
- Maximum performance is required

JIT may not help when:

- Pattern is matched only a few times
- Pattern has no effective prefilter
- Pattern has large Unicode classes (LazyDFA is better)
