//! Prefix optimization for large alternations.
//!
//! This module implements a trie-based optimization pass that merges common prefixes
//! in alternations of literals. This is critical for supporting large vocabularies
//! (like tokenizers with 100k+ tokens) without causing thread explosion in the NFA.
//!
//! # Example
//!
//! Before optimization:
//! ```text
//! (the|that|them|they)
//! ```
//! Creates 4 separate branches, each processing 't' independently.
//!
//! After optimization:
//! ```text
//! th(e(m|y)?|at)
//! ```
//! Creates a single path through 't' -> 'h', then branches.
//!
//! This reduces active threads from O(vocabulary_size) to O(token_length).

use super::{Hir, HirCapture, HirExpr};
use std::collections::HashMap;

/// Threshold for when to apply prefix optimization.
/// Smaller alternations don't benefit much from the overhead.
const MIN_LITERALS_FOR_OPTIMIZATION: usize = 4;

/// A trie node for building prefix trees from literals.
#[derive(Debug, Default)]
struct TrieNode {
    /// Children indexed by byte value.
    children: HashMap<u8, TrieNode>,
    /// If this node represents a complete literal, store the capture index (if any).
    /// None means this is just an intermediate node.
    is_terminal: bool,
    /// Original capture index if this literal was in a capture group.
    capture_index: Option<u32>,
    /// Original capture name if this literal was in a named capture group.
    capture_name: Option<String>,
}

impl TrieNode {
    fn new() -> Self {
        Self::default()
    }

    /// Inserts a literal into the trie.
    fn insert(&mut self, bytes: &[u8], capture_index: Option<u32>, capture_name: Option<String>) {
        let mut node = self;
        for &byte in bytes {
            node = node.children.entry(byte).or_insert_with(TrieNode::new);
        }
        node.is_terminal = true;
        node.capture_index = capture_index;
        node.capture_name = capture_name;
    }

    /// Converts the trie back to an optimized HIR expression.
    fn to_hir(&self) -> HirExpr {
        // If this is a terminal with no children, return empty
        if self.children.is_empty() {
            return if self.is_terminal {
                HirExpr::Empty
            } else {
                HirExpr::Empty
            };
        }

        // Collect all children
        let mut children: Vec<(u8, &TrieNode)> =
            self.children.iter().map(|(&b, n)| (b, n)).collect();
        children.sort_by_key(|(b, _)| *b);

        // If there's only one child, we can create a concat
        if children.len() == 1 {
            let (byte, child) = children[0];
            let child_hir = child.to_hir();
            let literal = HirExpr::Literal(vec![byte]);

            // If child is terminal and has children, we need concat
            if child.is_terminal && !child.children.is_empty() {
                // This is a case like "the" where "th" leads to "e" which is terminal
                // but also has children (like "them", "they")
                let child_expr = child.to_hir_with_optional_suffix();
                return HirExpr::Concat(vec![literal, child_expr]);
            }

            match child_hir {
                HirExpr::Empty => literal,
                HirExpr::Concat(mut parts) => {
                    parts.insert(0, literal);
                    HirExpr::Concat(parts)
                }
                other => HirExpr::Concat(vec![literal, other]),
            }
        } else {
            // Multiple children - create an alternation
            let alts: Vec<HirExpr> = children
                .iter()
                .map(|(byte, child)| {
                    let literal = HirExpr::Literal(vec![*byte]);
                    let child_hir = if child.is_terminal && !child.children.is_empty() {
                        child.to_hir_with_optional_suffix()
                    } else {
                        child.to_hir()
                    };

                    match child_hir {
                        HirExpr::Empty => literal,
                        HirExpr::Concat(mut parts) => {
                            parts.insert(0, literal);
                            HirExpr::Concat(parts)
                        }
                        other => HirExpr::Concat(vec![literal, other]),
                    }
                })
                .collect();

            if alts.len() == 1 {
                alts.into_iter().next().unwrap()
            } else {
                HirExpr::Alt(alts)
            }
        }
    }

    /// Creates HIR for a node that is both terminal and has children.
    /// This creates an optional suffix pattern.
    fn to_hir_with_optional_suffix(&self) -> HirExpr {
        if self.children.is_empty() {
            return HirExpr::Empty;
        }

        // The suffix is optional (this node is terminal)
        let suffix = self.to_hir();

        // Create an alternation between empty (terminal here) and the suffix
        HirExpr::Alt(vec![HirExpr::Empty, suffix])
    }
}

/// Optimizes an HIR by merging common prefixes in large alternations.
pub fn optimize_prefixes(hir: Hir) -> Hir {
    let expr = optimize_expr(hir.expr);
    Hir {
        expr,
        props: hir.props,
    }
}

fn optimize_expr(expr: HirExpr) -> HirExpr {
    match expr {
        HirExpr::Alt(variants) => optimize_alternation(variants),
        HirExpr::Concat(parts) => {
            let optimized: Vec<HirExpr> = parts.into_iter().map(optimize_expr).collect();
            HirExpr::Concat(optimized)
        }
        HirExpr::Repeat(rep) => HirExpr::Repeat(Box::new(super::HirRepeat {
            expr: optimize_expr(rep.expr),
            min: rep.min,
            max: rep.max,
            greedy: rep.greedy,
        })),
        HirExpr::Capture(cap) => HirExpr::Capture(Box::new(HirCapture {
            index: cap.index,
            name: cap.name,
            expr: optimize_expr(cap.expr),
        })),
        HirExpr::Lookaround(la) => HirExpr::Lookaround(Box::new(super::HirLookaround {
            expr: optimize_expr(la.expr),
            kind: la.kind,
        })),
        // These don't need optimization
        other => other,
    }
}

/// Optimizes an alternation by merging common prefixes.
fn optimize_alternation(variants: Vec<HirExpr>) -> HirExpr {
    // Separate literals from complex expressions
    let mut literals: Vec<(Vec<u8>, Option<u32>, Option<String>)> = Vec::new();
    let mut complex: Vec<HirExpr> = Vec::new();

    for variant in variants {
        match extract_literal(&variant) {
            Some((bytes, cap_idx, cap_name)) => {
                literals.push((bytes, cap_idx, cap_name));
            }
            None => {
                // Recursively optimize complex expressions
                complex.push(optimize_expr(variant));
            }
        }
    }

    // Only optimize if we have enough literals
    if literals.len() < MIN_LITERALS_FOR_OPTIMIZATION {
        // Put literals back as-is
        let mut result: Vec<HirExpr> = literals
            .into_iter()
            .map(|(bytes, cap_idx, cap_name)| {
                let lit = HirExpr::Literal(bytes);
                wrap_in_capture(lit, cap_idx, cap_name)
            })
            .collect();
        result.extend(complex);

        if result.len() == 1 {
            return result.into_iter().next().unwrap();
        }
        return HirExpr::Alt(result);
    }

    // Build trie from literals
    let mut trie = TrieNode::new();
    for (bytes, cap_idx, cap_name) in &literals {
        trie.insert(bytes, *cap_idx, cap_name.clone());
    }

    // Convert trie back to HIR
    let optimized_literals = trie.to_hir();

    // Combine with complex expressions
    if complex.is_empty() {
        optimized_literals
    } else {
        let mut result = vec![optimized_literals];
        result.extend(complex);
        HirExpr::Alt(result)
    }
}

/// Extracts a literal from an HIR expression if possible.
/// Returns the bytes, optional capture index, and optional capture name.
fn extract_literal(expr: &HirExpr) -> Option<(Vec<u8>, Option<u32>, Option<String>)> {
    match expr {
        HirExpr::Literal(bytes) => Some((bytes.clone(), None, None)),
        HirExpr::Capture(cap) => {
            if let HirExpr::Literal(bytes) = &cap.expr {
                Some((bytes.clone(), Some(cap.index), cap.name.clone()))
            } else {
                None
            }
        }
        HirExpr::Concat(parts) => {
            // Check if all parts are literals
            let mut result = Vec::new();
            for part in parts {
                match part {
                    HirExpr::Literal(bytes) => result.extend(bytes),
                    _ => return None,
                }
            }
            Some((result, None, None))
        }
        _ => None,
    }
}

/// Wraps an expression in a capture group if needed.
fn wrap_in_capture(expr: HirExpr, cap_idx: Option<u32>, cap_name: Option<String>) -> HirExpr {
    match cap_idx {
        Some(idx) => HirExpr::Capture(Box::new(HirCapture {
            index: idx,
            name: cap_name,
            expr,
        })),
        None => expr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::HirProps;

    fn lit(s: &str) -> HirExpr {
        HirExpr::Literal(s.as_bytes().to_vec())
    }

    fn alt(exprs: Vec<HirExpr>) -> HirExpr {
        HirExpr::Alt(exprs)
    }

    #[test]
    fn test_trie_simple() {
        let mut trie = TrieNode::new();
        trie.insert(b"the", None, None);
        trie.insert(b"that", None, None);
        trie.insert(b"them", None, None);
        trie.insert(b"they", None, None);

        let hir = trie.to_hir();
        // Should produce something like: t -> h -> (a -> t | e -> (Empty | m | y))
        // The exact structure depends on implementation, but it should be a valid HIR

        // Verify it's not just 4 separate branches
        match hir {
            HirExpr::Concat(parts) => {
                // First part should be 't'
                assert!(matches!(&parts[0], HirExpr::Literal(b) if b == b"t"));
            }
            _ => panic!("Expected Concat, got {:?}", hir),
        }
    }

    #[test]
    fn test_optimize_small_alternation() {
        // Small alternations should not be optimized
        let expr = alt(vec![lit("a"), lit("b"), lit("c")]);
        let hir = Hir {
            expr,
            props: HirProps::default(),
        };
        let optimized = optimize_prefixes(hir);

        // Should remain as Alt since only 3 literals
        assert!(matches!(optimized.expr, HirExpr::Alt(_)));
    }

    #[test]
    fn test_optimize_large_alternation() {
        // Large alternations should be optimized
        let expr = alt(vec![lit("the"), lit("that"), lit("them"), lit("they")]);
        let hir = Hir {
            expr,
            props: HirProps::default(),
        };
        let optimized = optimize_prefixes(hir);

        // Should be optimized to a concat starting with 't'
        match optimized.expr {
            HirExpr::Concat(parts) => {
                assert!(matches!(&parts[0], HirExpr::Literal(b) if b == b"t"));
            }
            _ => panic!("Expected optimized to Concat, got {:?}", optimized.expr),
        }
    }

    #[test]
    fn test_optimize_mixed() {
        // Mix of literals and complex expressions
        let expr = alt(vec![
            lit("the"),
            lit("that"),
            lit("them"),
            lit("they"),
            HirExpr::Repeat(Box::new(super::super::HirRepeat {
                expr: lit("x"),
                min: 1,
                max: None,
                greedy: true,
            })),
        ]);
        let hir = Hir {
            expr,
            props: HirProps::default(),
        };
        let optimized = optimize_prefixes(hir);

        // Should have an Alt with the optimized literals and the repeat
        assert!(matches!(optimized.expr, HirExpr::Alt(_)));
    }

    #[test]
    fn test_no_common_prefix() {
        // Literals with no common prefix
        let expr = alt(vec![
            lit("apple"),
            lit("banana"),
            lit("cherry"),
            lit("date"),
        ]);
        let hir = Hir {
            expr,
            props: HirProps::default(),
        };
        let optimized = optimize_prefixes(hir);

        // Should still be an Alt, but with individual starting bytes
        assert!(matches!(optimized.expr, HirExpr::Alt(_)));
    }

    #[test]
    fn test_partial_overlap() {
        // Some words share prefixes, some don't
        let expr = alt(vec![
            lit("test"),
            lit("testing"),
            lit("tested"),
            lit("tester"),
            lit("apple"),
            lit("application"),
        ]);
        let hir = Hir {
            expr,
            props: HirProps::default(),
        };
        let optimized = optimize_prefixes(hir);

        // Should be an Alt with two main branches: 't...' and 'a...'
        match optimized.expr {
            HirExpr::Alt(branches) => {
                assert_eq!(branches.len(), 2);
            }
            _ => panic!("Expected Alt with 2 branches, got {:?}", optimized.expr),
        }
    }
}
