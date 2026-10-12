# Contributing to regexr

## Before You Start

Open an issue before any non-trivial change, such as a new engine or a change to match semantics. Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## Build and Test

Install [cargo-nextest](https://nexte.st) for the test runner. CI uses it.

```bash
cargo build
cargo build --features full                 # JIT and SIMD
cargo nextest run --no-fail-fast            # Default features
cargo nextest run --features full --no-fail-fast
cargo nextest run --no-default-features --no-fail-fast
```

Run the same checks as CI before you open a pull request:

```bash
cargo fmt --all --check
cargo clippy --features full --all-targets -- -D warnings
cargo doc --features full --no-deps
cargo test --doc --features full
cargo check --no-default-features --target wasm32-unknown-unknown
shellcheck scripts/ci/*.sh
```

The `rust-version` in `Cargo.toml` sets the minimum supported Rust version. Cargo refuses to build on an older toolchain.

## Unicode Tables

Regenerate the Unicode tables from the Unicode Character Database with:

```bash
./scripts/update_unicode_tables.sh          # Latest UCD
./scripts/update_unicode_tables.sh 17.0.0   # Specific version
```

The script writes the tables to `data/unicode_tables/` and consolidates them into `src/hir/unicode_data.rs`. Commit both.

## Test Layout

Put each test in the category that matches its subject:

- `tests/api/`: public API
- `tests/engines/`: one engine or the engine selection
- `tests/features/`: lookarounds, backreferences, case sensitivity
- `tests/patterns/`: real-world patterns, such as tokenizer patterns
- `tests/unicode/`: Unicode properties and scripts

## Pull Requests

A pull request must include:

- Tests for the change. Put the test in the category above.
- Passing `cargo fmt`, `cargo clippy`, and `cargo nextest run` with the default and `full` features.
- An entry in [CHANGELOG.md](CHANGELOG.md) for each user-visible change. The changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
