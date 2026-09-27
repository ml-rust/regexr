//! Shared types for the Eager DFA engine.
//!
//! Contains types used by both the interpreter and potentially a JIT backend.

use super::super::lazy::CharClass;

/// Tagged state encoding constants.
pub const TAG_MATCH: u32 = 1 << 30;
pub const TAG_DEAD: u32 = 1 << 31;
pub const STATE_MASK: u32 = !(TAG_MATCH | TAG_DEAD);
pub const DEAD_STATE: u32 = TAG_DEAD | STATE_MASK;

/// Per-state metadata for end assertion checking.
#[derive(Clone, Copy, Default)]
pub struct StateMetadata {
    /// Character class of the last byte consumed to reach this state.
    pub prev_class: CharClass,
    /// Whether this match state requires a word boundary assertion.
    pub needs_word_boundary: bool,
    /// Whether this match state requires a NOT word boundary assertion.
    pub needs_not_word_boundary: bool,
    /// Whether this match state requires end of text ($) assertion.
    pub needs_end_of_text: bool,
    /// Whether this match state requires end of line (multiline $) assertion.
    pub needs_end_of_line: bool,
    /// Whether a match is reachable here without passing an end assertion — see
    /// `DfaState::match_without_end_assertion`.
    pub match_without_end_assertion: bool,
}

/// The most materialization work `EagerDfa::from_lazy` will spend before
/// declining in favor of `LazyDfa`.
///
/// "Work" is the running sum of each materialized state's NFA-subset size
/// (`LazyDfa::get_state_subset_size`), accumulated as the BFS visits states —
/// not a state count. State count alone hides the real cost: for a pattern
/// like `(?:a?){n}`, Thompson construction chains ~4 states per copy via
/// epsilon edges, so DFA state `S_k` carries a live NFA subset of size
/// Θ(n−k). Materializing it unions Θ(n−k) closures of size Θ(n−k) each —
/// Θ((n−k)²) work — and summing that over the ~n states gives Θ(n³) total
/// compile time, even though the *state count* is only Θ(n). The subset-size
/// sum tracked here is Θ(n²): a cheap-to-compute proxy that still separates
/// the two shapes below by 10x.
///
/// Measured on release builds (`Regex::new` wall time):
///
/// | pattern         | n     | Σ subset size | compile time |
/// |------------------|-------|---------------|--------------|
/// | `a{n}` (chain)   | 50000 | ≈50,000       | ≈3.3s        |
/// | `(?:a?){n}`      | 500   | ≈125,000      | 975ms        |
/// | `(?:a?){n}`      | 750   | ≈281,250      | 3.8s         |
/// | `(?:a?){n}`      | 1000  | ≈500,000      | 10.3s        |
///
/// `a{50000}` is a chain: each state's live subset stays small (O(1)), so its
/// Σ is only ≈50,000 despite having far more states than any `(?:a?){n})`
/// case here — it must stay eager, since `tests/pattern_limits.rs::
/// the_expansion_limit_can_be_raised` depends on that. `(?:a?){1000}`'s Σ is
/// 10x higher despite an order of magnitude fewer states, and must be
/// declined.
///
/// `200_000` sits clearly above the chain's ≈50,000 and clearly below
/// `(?:a?){1000}`'s ≈500,000, while also falling between the `n=500`
/// (975ms, kept eager) and `n=750` (3.8s, declined) data points above.
pub const MATERIALIZATION_WORK_BUDGET: usize = 200_000;

/// `EagerDfa::from_lazy` declined to finish materializing because its
/// cumulative work crossed `MATERIALIZATION_WORK_BUDGET` — see that
/// constant's doc for the Θ(n³) reasoning. The caller should fall back to
/// `LazyDfa`, which computes the exact same states/transitions on demand
/// (same `compute_all_transitions_simple`/`epsilon_closure` code), just
/// spread out over the search instead of paid upfront.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EagerMaterializationBudgetExceeded;

/// An eager-DFA search stopped because its start-position loop in
/// `EagerDfa::find_from` walked past its scan budget.
///
/// Distinct from "no match": trying every remaining start position at up to
/// O(n) cost each would make the search quadratic on patterns whose failed
/// attempts each scan far (`(?s)a.*b` over a run of `a`), so the loop gives
/// up rather than paying for it. The eager DFA holds no NFA to search with,
/// so the caller re-runs the search from the same position on a linear-time
/// engine built for the same pattern — see the
/// [`EagerDfa::find_from`](super::interpreter::EagerDfa::find_from) doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EagerScanBudgetExceeded;

/// Check if a byte is a word character.
#[inline(always)]
pub fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
