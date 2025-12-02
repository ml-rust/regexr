# Engine Structure Guidelines

This document describes the standard module structure for regex engines in regexr. Following this structure ensures consistency, maintainability, and easier extension (e.g., adding ARM64 JIT support).

## Overview

Engines are organized into three top-level categories:

```
src/
├── nfa/          # NFA construction and NFA-based engines
├── dfa/          # DFA-based engines (deterministic automata)
└── vm/           # Virtual machine engines (specialized matchers)
```

**Note**: NFA (Nondeterministic Finite Automaton) is a data structure. Thompson's algorithm is one way to *construct* an NFA from a regex. The NFA can then be *executed* using various algorithms (PikeVM, tagged simulation, etc.).

The `src/jit/` module serves as a **re-export hub** only - it re-exports JIT types from their canonical locations for backwards compatibility.

## Standard Engine Structure

Each engine follows a consistent directory structure with **required** and **optional** components:

```
src/{engine_type}/{engine_name}/
├── mod.rs              # [required] Module coordination and re-exports
├── engine.rs           # [required] Engine facade (public API)
├── interpreter/        # [required] Interpreter implementations (always available)
│   ├── mod.rs
│   └── *.rs
├── jit/                # [optional] JIT implementations (feature-gated)
│   ├── mod.rs
│   ├── {name}.rs       # JIT struct and public API
│   ├── x86_64.rs       # x86-64 code generation
│   ├── aarch64.rs      # ARM64 code generation
│   └── helpers.rs      # Extern helper functions for JIT
└── *.rs                # [optional] Engine-specific files as needed
```

### Engine-Specific Files

Each engine may have additional files based on its needs. Examples:

| Engine | Specific Files | Purpose |
|--------|---------------|---------|
| Tagged NFA | `liveness.rs` | Capture liveness analysis for sparse copying |
| Tagged NFA | `steps.rs` | Pattern step extraction for fast matching |
| Tagged NFA | `shared.rs` | ThreadWorklist, PatternStep types |
| Lazy DFA | `cache.rs` | State cache management |
| Shift-Or | `bitset.rs` | Bit manipulation utilities |

**Don't force unnecessary files** - only create what the engine actually needs.

## Example: Tagged NFA Engine

The Tagged NFA engine has capture tracking and lookaround support, so it needs extra files:

```
src/nfa/tagged/
├── mod.rs              # [required] Re-exports
├── engine.rs           # [required] TaggedNfaEngine facade
├── interpreter/        # [required] Pure Rust execution
│   ├── mod.rs
│   ├── step_interpreter.rs    # Fast step-based matching
│   └── nfa_interpreter.rs     # Full NFA simulation
├── jit/                # [optional] JIT execution
│   ├── mod.rs
│   ├── jit.rs          # TaggedNfaJit struct
│   ├── x86_64.rs       # x86-64 dynasm code
│   └── helpers.rs      # Extern helper functions
│
│   # Engine-specific files (not required for all engines):
├── shared.rs           # ThreadWorklist, PatternStep, TaggedNfaContext
├── liveness.rs         # Capture liveness analysis (for sparse copying)
└── steps.rs            # Pattern step extraction (for fast path)
```

A simpler engine like Shift-Or might only need:

```
src/vm/shift_or/
├── mod.rs
├── engine.rs
├── interpreter/
│   ├── mod.rs
│   └── matcher.rs
└── jit/
    ├── mod.rs
    ├── jit.rs
    └── x86_64.rs
```

## Feature Gating

### Always Available (no feature gate)
- `shared.rs` - Data structures used by both interpreter and JIT
- `liveness.rs` - Analysis passes
- `engine.rs` - Engine facade
- `interpreter/` - All interpreter implementations

### JIT-Gated
```rust
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod jit;
```

### Architecture-Specific JIT
```rust
// In jit/mod.rs
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

// Use the appropriate backend
#[cfg(target_arch = "x86_64")]
pub use x86_64::compile;

#[cfg(target_arch = "aarch64")]
pub use aarch64::compile;
```

## Module Responsibilities

### `mod.rs` (required)
- Module coordination
- Re-exports for public API
- Feature-gated submodule inclusion

```rust
// Include engine-specific modules as needed
pub mod interpreter;
mod engine;

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod jit;

// Re-exports
pub use interpreter::{Interpreter, ...};
pub use engine::Engine;
```

### `engine.rs` (required)
- Facade pattern - single entry point for executor.rs
- Owns the NFA/DFA and selects between interpreter/JIT
- Provides `is_match()`, `find()`, `captures()` methods

```rust
pub struct Engine {
    automaton: Automaton,
    // ... cached data
}

impl Engine {
    pub fn new(automaton: Automaton) -> Self { ... }
    pub fn is_match(&self, input: &[u8]) -> bool { ... }
    pub fn find(&self, input: &[u8]) -> Option<(usize, usize)> { ... }
}
```

### `interpreter/` (required)
- Pure Rust implementations
- No JIT dependencies
- Multiple implementations allowed (e.g., StepInterpreter for fast path, full simulation for complex patterns)

### `jit/` (optional)
- JIT compilation code
- Architecture-specific backends in separate files (x86_64.rs, aarch64.rs)
- Common JIT struct in `{name}.rs`
- Helper functions in `helpers.rs` if needed for extern calls

## Adding ARM64 Support

To add ARM64 JIT support for an engine:

1. Create `jit/aarch64.rs` with the ARM64 code generator
2. Update `jit/mod.rs` to conditionally include the new backend:

```rust
#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
use x86_64::Compiler;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
use aarch64::Compiler;
```

3. Ensure the JIT struct interface is the same across architectures
4. Share helper functions where possible (helpers.rs)

## Re-export Hub: src/jit/mod.rs

The `src/jit/mod.rs` module exists for backwards compatibility and convenience. It re-exports JIT types from their canonical locations:

```rust
// Re-export from canonical locations
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use crate::nfa::tagged::jit::{TaggedNfaJit, compile_tagged_nfa};

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use crate::dfa::lazy::jit::{DfaJit, compile_dfa};
```

**Important**: New JIT code should go in the engine's `jit/` subdirectory, not in `src/jit/`.

## Final Planned Structure

After complete migration, the engine structure will be:

```
src/
├── nfa/
│   ├── tagged/         # Tagged NFA (captures, lookaround, non-greedy)
│   └── ...             # NFA construction (state.rs, compiler.rs, etc.)
│
├── dfa/
│   ├── lazy/           # Lazy DFA (on-demand state construction)
│   └── eager/          # Eager DFA (precomputed states)
│
├── vm/
│   ├── pike/           # PikeVM (parallel NFA simulation)
│   ├── backtracking/   # Backtracking VM (backreferences)
│   └── shift_or/       # Shift-Or (bit-parallel matcher)
│
├── simd/               # SIMD acceleration (prefilters, memchr, Teddy)
│
├── jit/                # Re-export hub only (no implementations)
│
└── engine/
    └── executor.rs     # Engine selection and dispatch
```

## Detailed Module Structure

### NFA Module (`src/nfa/`)

The NFA module contains NFA construction (Thompson's algorithm) and NFA-based execution engines:

```
src/nfa/
├── mod.rs              # NFA types, compile() using Thompson's algorithm
├── state.rs            # NFA state machine data structure
├── compiler.rs         # Thompson NFA construction from HIR
├── utf8_automata.rs    # UTF-8 byte sequence handling
└── tagged/             # Tagged NFA engine (captures, lookaround)
    ├── shared.rs       # ThreadWorklist, PatternStep
    ├── liveness.rs     # Capture liveness analysis
    ├── engine.rs       # TaggedNfaEngine facade
    ├── interpreter/    # Pure Rust execution
    └── jit/            # JIT-compiled execution
```

### DFA Module (`src/dfa/`)

DFA-based engines convert NFA to deterministic automata:

```
src/dfa/
├── mod.rs
├── lazy/               # Lazy DFA (on-demand state construction)
│   ├── engine.rs
│   ├── interpreter/
│   └── jit/
└── eager/              # Eager DFA (precomputed states)
    ├── engine.rs
    ├── interpreter/
    └── jit/
```

### VM Module (`src/vm/`)

Virtual machine engines for specialized matching:

```
src/vm/
├── mod.rs
├── pike/               # PikeVM (parallel NFA simulation with captures)
│   ├── engine.rs
│   ├── interpreter/
│   └── jit/
├── backtracking/       # Backtracking VM (for backreferences)
│   ├── engine.rs
│   ├── interpreter/
│   └── jit/
└── shift_or/           # Shift-Or bit-parallel matcher
    ├── engine.rs
    ├── interpreter/
    └── jit/
```

## Terminology

| Term | Meaning |
|------|---------|
| **NFA** | Nondeterministic Finite Automaton - a data structure representing regex states |
| **DFA** | Deterministic Finite Automaton - each state has exactly one transition per input |
| **Thompson's Algorithm** | Algorithm to construct an NFA from a regex |
| **PikeVM** | Algorithm to execute NFA with parallel thread simulation |
| **Tagged NFA** | NFA execution with capture group tracking |
| **Lazy DFA** | DFA that builds states on-demand during matching |
| **Eager DFA** | DFA that precomputes all states before matching |

## Benefits of This Structure

1. **Maintainability**: Each engine is self-contained
2. **Extensibility**: Adding new architectures is straightforward
3. **Feature isolation**: Non-JIT builds don't pull in JIT code
4. **Clear ownership**: No ambiguity about where code belongs
5. **Testability**: Each component can be tested independently
6. **Documentation**: Structure is self-documenting

## Migration Status

Current engine migration progress:

| Engine | Location | Status | Notes |
|--------|----------|--------|-------|
| Tagged NFA | `src/nfa/tagged/` | ✅ Migrated | Captures, lookaround, non-greedy |
| Shift-Or | `src/vm/shift_or/` | ✅ Migrated | Bit-parallel matcher |
| Backtracking | `src/vm/backtracking/` | ✅ Migrated | Backreference support |
| PikeVM | `src/vm/pike/` | ✅ Migrated | Parallel NFA simulation, no JIT |
| Lazy DFA | `src/dfa/lazy/` | ✅ Migrated | On-demand state construction |
| Eager DFA | `src/dfa/eager/` | ✅ Migrated | Pre-computed states for fast matching |

### Migrated Engines

#### Shift-Or (`src/vm/shift_or/`)
```
src/vm/shift_or/
├── mod.rs              # Module coordination, tests
├── engine.rs           # ShiftOrEngine facade
├── shared.rs           # ShiftOr data structure, masks, follow sets
├── interpreter/
│   ├── mod.rs
│   └── matcher.rs      # ShiftOrInterpreter
└── jit/
    ├── mod.rs
    ├── jit.rs          # JitShiftOr struct and API
    └── x86_64.rs       # x86-64 code generation
```

#### Backtracking (`src/vm/backtracking/`)
```
src/vm/backtracking/
├── mod.rs              # Module coordination, tests
├── engine.rs           # BacktrackingEngine facade
├── shared.rs           # Op bytecode enum, helpers (decode_utf8, is_word_byte)
├── interpreter/
│   ├── mod.rs
│   └── vm.rs           # BacktrackingVm
└── jit/
    ├── mod.rs
    ├── jit.rs          # BacktrackingJit struct and API
    └── x86_64.rs       # x86-64 code generation
```

#### PikeVM (`src/vm/pike/`)
```
src/vm/pike/
├── mod.rs              # Module coordination, tests
├── engine.rs           # PikeVmEngine facade
├── shared.rs           # Thread, PikeVmContext, InstructionResult
└── interpreter/
    ├── mod.rs
    └── vm.rs           # PikeVm (no JIT backend)
```

#### Lazy DFA (`src/dfa/lazy/`)
```
src/dfa/lazy/
├── mod.rs              # Module coordination, tests
├── engine.rs           # LazyDfaEngine facade
├── shared.rs           # DfaState, CharClass, PositionContext, LazyDfaContext
└── interpreter/
    ├── mod.rs
    └── dfa.rs          # LazyDfa (no JIT backend currently)
```

#### Eager DFA (`src/dfa/eager/`)
```
src/dfa/eager/
├── mod.rs              # Module coordination, tests
├── engine.rs           # EagerDfaEngine facade
├── shared.rs           # StateMetadata, tagged state constants
└── interpreter/
    ├── mod.rs
    └── dfa.rs          # EagerDfa (no JIT backend currently)
```

## Migration Checklist

When migrating an existing engine to this structure:

- [ ] Create the directory structure
- [ ] Move shared types to `shared.rs`
- [ ] Extract interpreter code to `interpreter/`
- [ ] Extract JIT code to `jit/`
- [ ] Create engine facade in `engine.rs`
- [ ] Update `mod.rs` with proper re-exports
- [ ] Update `src/jit/mod.rs` to re-export from new location
- [ ] Delete old files from `src/jit/`
- [ ] Update `src/engine/executor.rs` imports
- [ ] Run full test suite
- [ ] Update documentation
