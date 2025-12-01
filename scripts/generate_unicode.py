#!/usr/bin/env python3
"""
Generate unicode_data.rs from the pre-generated Unicode tables.

This script reads the ucd-generate output files in data/unicode_tables/
and generates a consolidated unicode_data.rs module for the HIR builder.

Usage:
    python scripts/generate_unicode.py                    # Default paths
    python scripts/generate_unicode.py --data-dir DIR     # Custom input
    python scripts/generate_unicode.py --output FILE      # Custom output
    python scripts/generate_unicode.py --verify           # Verify only, no write

The output is written to src/hir/unicode_data.rs by default.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "unicode_tables"
OUTPUT_FILE = PROJECT_ROOT / "src" / "hir" / "unicode_data.rs"


def parse_rust_ranges(content: str, const_name: str) -> list[tuple[int, int]]:
    """Parse a Rust const array of (char, char) or (u32, u32) tuples."""
    # Try (char, char) format first
    pattern = rf"pub const {const_name}:\s*&'static\s*\[\(char,\s*char\)\]\s*=\s*&\[(.*?)\];"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        array_content = match.group(1)
        ranges = []
        # Match patterns like ('A', 'Z') or ('\u{300}', 'ʹ')
        for m in re.finditer(r"\('([^']+)',\s*'([^']+)'\)", array_content):
            start_str, end_str = m.groups()
            start = parse_char(start_str)
            end = parse_char(end_str)
            if start is not None and end is not None:
                ranges.append((start, end))
        return ranges

    # Try (u32, u32) format
    pattern = rf"pub const {const_name}:\s*&'static\s*\[\(u32,\s*u32\)\]\s*=\s*&\[(.*?)\];"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        array_content = match.group(1)
        ranges = []
        # Match patterns like (48, 57) or (0x0030, 0x0039)
        for m in re.finditer(r"\((\d+|0x[0-9a-fA-F]+),\s*(\d+|0x[0-9a-fA-F]+)\)", array_content):
            start_str, end_str = m.groups()
            start = int(start_str, 16) if start_str.startswith('0x') else int(start_str)
            end = int(end_str, 16) if end_str.startswith('0x') else int(end_str)
            ranges.append((start, end))
        return ranges

    return []


def parse_char(s: str) -> int | None:
    """Parse a Rust char literal to a code point."""
    if s.startswith('\\u{') and s.endswith('}'):
        # Unicode escape: \u{HHHH}
        return int(s[3:-1], 16)
    elif s.startswith('\\x'):
        # Hex escape: \xHH
        return int(s[2:], 16)
    elif s.startswith('\\'):
        # Other escapes
        escapes = {'n': 0x0A, 'r': 0x0D, 't': 0x09, '\\': 0x5C, "'": 0x27}
        if len(s) == 2 and s[1] in escapes:
            return escapes[s[1]]
        return None
    elif len(s) == 1:
        return ord(s)
    else:
        # Multi-byte UTF-8 character represented directly
        return ord(s)


def parse_by_name(content: str) -> dict[str, str]:
    """Parse BY_NAME array to get category -> const name mapping."""
    pattern = r'pub const BY_NAME:.*?=\s*&\[(.*?)\];'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return {}

    mapping = {}
    for m in re.finditer(r'\("([^"]+)",\s*(\w+)\)', match.group(1)):
        name, const_name = m.groups()
        mapping[name] = const_name

    return mapping


def format_ranges(ranges: list[tuple[int, int]], indent: str = "    ") -> str:
    """Format ranges as Rust code."""
    if not ranges:
        return f"{indent}// Empty"

    lines = []
    for i, (start, end) in enumerate(ranges):
        if i % 8 == 0:
            if lines:
                lines[-1] = lines[-1].rstrip()
                lines.append("\n" + indent)
            else:
                lines.append(indent)
        lines.append(f"(0x{start:04X}, 0x{end:04X}), ")

    return "".join(lines).rstrip(", ")


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping and adjacent ranges."""
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def generate_rust_module(
    general_categories: dict[str, list[tuple[int, int]]],
    scripts: dict[str, list[tuple[int, int]]],
    bool_properties: dict[str, list[tuple[int, int]]],
    perl_word: list[tuple[int, int]],
    perl_decimal: list[tuple[int, int]],
    perl_space: list[tuple[int, int]],
    case_folding: list[tuple[int, int]] = None
) -> str:
    """Generate the complete Rust module."""
    if case_folding is None:
        case_folding = []

    lines = [
        "//! Unicode property data tables.",
        "//!",
        "//! This module is auto-generated from Unicode Character Database (UCD).",
        "//! DO NOT EDIT MANUALLY. Regenerate with: python scripts/generate_unicode.py",
        "//!",
        "//! Data source: data/unicode_tables/ (generated by ucd-generate)",
        "",
        "#![allow(clippy::unreadable_literal)]",
        "",
    ]

    # Generate General Category constants
    lines.append("// =============================================================================")
    lines.append("// General Categories")
    lines.append("// =============================================================================")
    lines.append("")

    for name in sorted(general_categories.keys()):
        ranges = general_categories[name]
        const_name = name.upper().replace(" ", "_").replace("-", "_")
        lines.append(f"/// Unicode General Category: {name}")
        lines.append(f"pub const GC_{const_name}: &[(u32, u32)] = &[")
        lines.append(format_ranges(ranges))
        lines.append("];")
        lines.append("")

    # Generate Script constants
    lines.append("// =============================================================================")
    lines.append("// Scripts")
    lines.append("// =============================================================================")
    lines.append("")

    for name in sorted(scripts.keys()):
        ranges = scripts[name]
        const_name = name.upper().replace(" ", "_").replace("-", "_")
        lines.append(f"/// Unicode Script: {name}")
        lines.append(f"pub const SCRIPT_{const_name}: &[(u32, u32)] = &[")
        lines.append(format_ranges(ranges))
        lines.append("];")
        lines.append("")

    # Generate Boolean Property constants (selected important ones)
    important_bool_props = [
        "Alphabetic", "Lowercase", "Uppercase", "White_Space",
        "Hex_Digit", "ASCII_Hex_Digit", "Emoji", "Emoji_Presentation",
        "Extended_Pictographic", "XID_Start", "XID_Continue",
        "ID_Start", "ID_Continue", "Pattern_Syntax", "Pattern_White_Space"
    ]

    lines.append("// =============================================================================")
    lines.append("// Boolean Properties")
    lines.append("// =============================================================================")
    lines.append("")

    for name in important_bool_props:
        if name in bool_properties:
            ranges = bool_properties[name]
            const_name = name.upper().replace(" ", "_").replace("-", "_")
            lines.append(f"/// Unicode Boolean Property: {name}")
            lines.append(f"pub const PROP_{const_name}: &[(u32, u32)] = &[")
            lines.append(format_ranges(ranges))
            lines.append("];")
            lines.append("")

    # Generate Perl classes
    lines.append("// =============================================================================")
    lines.append("// Perl Character Classes (for \\w, \\d, \\s in Unicode mode)")
    lines.append("// =============================================================================")
    lines.append("")

    lines.append("/// Perl \\w in Unicode mode (word characters)")
    lines.append("pub const PERL_WORD: &[(u32, u32)] = &[")
    lines.append(format_ranges(perl_word))
    lines.append("];")
    lines.append("")

    lines.append("/// Perl \\d in Unicode mode (decimal digits)")
    lines.append("pub const PERL_DECIMAL: &[(u32, u32)] = &[")
    lines.append(format_ranges(perl_decimal))
    lines.append("];")
    lines.append("")

    lines.append("/// Perl \\s in Unicode mode (whitespace)")
    lines.append("pub const PERL_SPACE: &[(u32, u32)] = &[")
    lines.append(format_ranges(perl_space))
    lines.append("];")
    lines.append("")

    # Generate Case Folding table
    if case_folding:
        lines.append("// =============================================================================")
        lines.append("// Case Folding (Simple)")
        lines.append("// =============================================================================")
        lines.append("")
        lines.append("/// Simple case folding table: maps uppercase to lowercase.")
        lines.append("/// Each entry is (from_codepoint, to_codepoint).")
        lines.append("pub const CASE_FOLDING_SIMPLE: &[(u32, u32)] = &[")
        # Format as pairs
        fold_lines = []
        for i, (from_cp, to_cp) in enumerate(case_folding):
            if i % 8 == 0:
                if fold_lines:
                    fold_lines[-1] = fold_lines[-1].rstrip()
                    fold_lines.append("\n    ")
                else:
                    fold_lines.append("    ")
            fold_lines.append(f"(0x{from_cp:04X}, 0x{to_cp:04X}), ")
        lines.append("".join(fold_lines).rstrip(", "))
        lines.append("];")
        lines.append("")

    # Generate lookup functions
    lines.extend(generate_lookup_functions(general_categories, scripts, bool_properties, case_folding))

    # Generate tests
    lines.extend(generate_tests())

    return "\n".join(lines)


def generate_lookup_functions(
    general_categories: dict[str, list[tuple[int, int]]],
    scripts: dict[str, list[tuple[int, int]]],
    bool_properties: dict[str, list[tuple[int, int]]],
    case_folding: list[tuple[int, int]] = None
) -> list[str]:
    """Generate the property lookup functions."""
    if case_folding is None:
        case_folding = []

    lines = [
        "// =============================================================================",
        "// Property Lookup Functions",
        "// =============================================================================",
        "",
        "/// Checks if a code point is in any of the given ranges using binary search.",
        "#[inline]",
        "pub fn in_ranges(cp: u32, ranges: &[(u32, u32)]) -> bool {",
        "    ranges.binary_search_by(|&(start, end)| {",
        "        if cp < start {",
        "            std::cmp::Ordering::Greater",
        "        } else if cp > end {",
        "            std::cmp::Ordering::Less",
        "        } else {",
        "            std::cmp::Ordering::Equal",
        "        }",
        "    }).is_ok()",
        "}",
        "",
        "/// Look up a Unicode property by name.",
        "/// Returns the ranges for the property, or None if not found.",
        "/// ",
        "/// Supports:",
        "/// - General Categories: Letter, Number, Punctuation, etc. (and subcategories)",
        "/// - Scripts: Latin, Greek, Han, Arabic, etc.",
        "/// - Boolean Properties: Alphabetic, Lowercase, Uppercase, etc.",
        "/// - Perl classes: word, digit, space",
        "pub fn get_property(name: &str) -> Option<&'static [(u32, u32)]> {",
        "    // Normalize: lowercase and remove underscores/hyphens/spaces",
        "    let normalized: String = name.chars()",
        "        .filter(|c| !matches!(c, '_' | '-' | ' '))",
        "        .flat_map(|c| c.to_lowercase())",
        "        .collect();",
        "",
        "    // Try General Categories first",
        "    if let Some(ranges) = get_general_category(&normalized) {",
        "        return Some(ranges);",
        "    }",
        "",
        "    // Try Scripts",
        "    if let Some(ranges) = get_script(&normalized) {",
        "        return Some(ranges);",
        "    }",
        "",
        "    // Try Boolean Properties",
        "    if let Some(ranges) = get_bool_property(&normalized) {",
        "        return Some(ranges);",
        "    }",
        "",
        "    // Try Perl classes",
        "    match normalized.as_str() {",
        '        "word" => Some(PERL_WORD),',
        '        "digit" => Some(PERL_DECIMAL),',
        '        "space" => Some(PERL_SPACE),',
        "        _ => None,",
        "    }",
        "}",
        "",
    ]

    # Generate General Category lookup
    lines.append("/// Look up a General Category by normalized name.")
    lines.append("fn get_general_category(name: &str) -> Option<&'static [(u32, u32)]> {")
    lines.append("    match name {")

    # Add short aliases
    gc_aliases = {
        "l": "letter", "lc": "casedletter", "lu": "uppercaseletter",
        "ll": "lowercaseletter", "lt": "titlecaseletter", "lm": "modifierletter",
        "lo": "otherletter", "m": "mark", "mn": "nonspacingmark",
        "mc": "spacingmark", "me": "enclosingmark", "n": "number",
        "nd": "decimalnumber", "nl": "letternumber", "no": "othernumber",
        "p": "punctuation", "pc": "connectorpunctuation", "pd": "dashpunctuation",
        "ps": "openpunctuation", "pe": "closepunctuation", "pi": "initialpunctuation",
        "pf": "finalpunctuation", "po": "otherpunctuation", "s": "symbol",
        "sm": "mathsymbol", "sc": "currencysymbol", "sk": "modifiersymbol",
        "so": "othersymbol", "z": "separator", "zs": "spaceseparator",
        "zl": "lineseparator", "zp": "paragraphseparator", "c": "other",
        "cc": "control", "cf": "format", "cs": "surrogate", "co": "privateuse",
        "cn": "unassigned"
    }

    for short, long in sorted(gc_aliases.items()):
        const_name = None
        for gc_name in general_categories.keys():
            if gc_name.lower().replace("_", "").replace(" ", "") == long:
                const_name = "GC_" + gc_name.upper().replace(" ", "_").replace("-", "_")
                break
        if const_name:
            lines.append(f'        "{short}" => Some({const_name}),')

    # Add full names
    for name in sorted(general_categories.keys()):
        normalized = name.lower().replace("_", "").replace(" ", "")
        const_name = "GC_" + name.upper().replace(" ", "_").replace("-", "_")
        lines.append(f'        "{normalized}" => Some({const_name}),')

    lines.append("        _ => None,")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # Generate Script lookup
    lines.append("/// Look up a Script by normalized name.")
    lines.append("fn get_script(name: &str) -> Option<&'static [(u32, u32)]> {")
    lines.append("    match name {")

    for name in sorted(scripts.keys()):
        normalized = name.lower().replace("_", "").replace(" ", "")
        const_name = "SCRIPT_" + name.upper().replace(" ", "_").replace("-", "_")
        lines.append(f'        "{normalized}" => Some({const_name}),')

    lines.append("        _ => None,")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # Generate Boolean Property lookup
    lines.append("/// Look up a Boolean Property by normalized name.")
    lines.append("fn get_bool_property(name: &str) -> Option<&'static [(u32, u32)]> {")
    lines.append("    match name {")

    important_bool_props = [
        "Alphabetic", "Lowercase", "Uppercase", "White_Space",
        "Hex_Digit", "ASCII_Hex_Digit", "Emoji", "Emoji_Presentation",
        "Extended_Pictographic", "XID_Start", "XID_Continue",
        "ID_Start", "ID_Continue", "Pattern_Syntax", "Pattern_White_Space"
    ]

    # Add common aliases (excluding duplicates that appear in full names)
    bool_aliases = {
        "alpha": "alphabetic", "lower": "lowercase", "upper": "uppercase",
        "wspace": "whitespace",
        "xdigit": "hexdigit"
    }

    for short, long in sorted(bool_aliases.items()):
        const_name = None
        for prop_name in important_bool_props:
            if prop_name.lower().replace("_", "") == long:
                const_name = "PROP_" + prop_name.upper().replace(" ", "_").replace("-", "_")
                break
        if const_name:
            lines.append(f'        "{short}" => Some({const_name}),')

    for name in important_bool_props:
        if name in bool_properties:
            normalized = name.lower().replace("_", "")
            const_name = "PROP_" + name.upper().replace(" ", "_").replace("-", "_")
            lines.append(f'        "{normalized}" => Some({const_name}),')

    lines.append("        _ => None,")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # Generate case folding lookup function if we have data
    if case_folding:
        lines.extend([
            "/// Simple case fold: returns the lowercase equivalent of a code point.",
            "/// Returns the input unchanged if no folding applies.",
            "#[inline]",
            "pub fn simple_case_fold(cp: u32) -> u32 {",
            "    match CASE_FOLDING_SIMPLE.binary_search_by_key(&cp, |&(from, _)| from) {",
            "        Ok(idx) => CASE_FOLDING_SIMPLE[idx].1,",
            "        Err(_) => cp,",
            "    }",
            "}",
            "",
            "/// Get all code points that fold to the same value as the given code point.",
            "/// Used for case-insensitive matching.",
            "/// Returns a Vec containing all equivalent code points (including the input).",
            "pub fn case_fold_equivalents(cp: u32) -> Vec<u32> {",
            "    // Get the canonical (folded) form",
            "    let folded = simple_case_fold(cp);",
            "    ",
            "    // Collect all code points that fold to this value",
            "    let mut equivalents = vec![folded];",
            "    ",
            "    // Find all entries that map to the same folded value",
            "    for &(from, to) in CASE_FOLDING_SIMPLE.iter() {",
            "        if to == folded && from != folded {",
            "            equivalents.push(from);",
            "        }",
            "    }",
            "    ",
            "    // Also add the original if different from folded",
            "    if cp != folded && !equivalents.contains(&cp) {",
            "        equivalents.push(cp);",
            "    }",
            "    ",
            "    equivalents.sort();",
            "    equivalents.dedup();",
            "    equivalents",
            "}",
            "",
        ])

    return lines


def generate_tests() -> list[str]:
    """Generate test functions."""
    return [
        "// =============================================================================",
        "// Tests",
        "// =============================================================================",
        "",
        "#[cfg(test)]",
        "mod tests {",
        "    use super::*;",
        "",
        "    #[test]",
        "    fn test_ascii_letter() {",
        "        assert!(in_ranges('A' as u32, get_property(\"Letter\").unwrap()));",
        "        assert!(in_ranges('z' as u32, get_property(\"Letter\").unwrap()));",
        "        assert!(!in_ranges('0' as u32, get_property(\"Letter\").unwrap()));",
        "    }",
        "",
        "    #[test]",
        "    fn test_greek() {",
        "        let greek = get_property(\"Greek\").unwrap();",
        "        assert!(in_ranges('α' as u32, greek));",
        "        assert!(in_ranges('ω' as u32, greek));",
        "        assert!(!in_ranges('a' as u32, greek));",
        "    }",
        "",
        "    #[test]",
        "    fn test_han() {",
        "        let han = get_property(\"Han\").unwrap();",
        "        assert!(in_ranges('中' as u32, han));",
        "        assert!(in_ranges('文' as u32, han));",
        "        assert!(!in_ranges('a' as u32, han));",
        "    }",
        "",
        "    #[test]",
        "    fn test_perl_word() {",
        "        let word = PERL_WORD;",
        "        assert!(in_ranges('a' as u32, word));",
        "        assert!(in_ranges('Z' as u32, word));",
        "        assert!(in_ranges('5' as u32, word));",
        "        assert!(in_ranges('_' as u32, word));",
        "        assert!(in_ranges('α' as u32, word)); // Greek letters are word chars",
        "        assert!(!in_ranges(' ' as u32, word));",
        "    }",
        "",
        "    #[test]",
        "    fn test_property_aliases() {",
        "        // Short aliases",
        "        assert!(get_property(\"L\").is_some());",
        "        assert!(get_property(\"N\").is_some());",
        "        assert!(get_property(\"P\").is_some());",
        "        // Long names",
        "        assert!(get_property(\"Letter\").is_some());",
        "        assert!(get_property(\"Number\").is_some());",
        "        // Case insensitive",
        "        assert!(get_property(\"LETTER\").is_some());",
        "        assert!(get_property(\"letter\").is_some());",
        "        // With underscores",
        "        assert!(get_property(\"Decimal_Number\").is_some());",
        "        assert!(get_property(\"DecimalNumber\").is_some());",
        "    }",
        "",
        "    #[test]",
        "    fn test_scripts() {",
        "        assert!(get_property(\"Latin\").is_some());",
        "        assert!(get_property(\"Greek\").is_some());",
        "        assert!(get_property(\"Han\").is_some());",
        "        assert!(get_property(\"Arabic\").is_some());",
        "        assert!(get_property(\"Cyrillic\").is_some());",
        "        assert!(get_property(\"Hiragana\").is_some());",
        "        assert!(get_property(\"Katakana\").is_some());",
        "    }",
        "",
        "    #[test]",
        "    fn test_emoji() {",
        "        let emoji = get_property(\"Emoji\").unwrap();",
        "        assert!(in_ranges('😀' as u32, emoji));",
        "    }",
        "",
        "    #[test]",
        "    fn test_case_folding() {",
        "        // Simple case fold",
        "        assert_eq!(simple_case_fold('A' as u32), 'a' as u32);",
        "        assert_eq!(simple_case_fold('Z' as u32), 'z' as u32);",
        "        assert_eq!(simple_case_fold('a' as u32), 'a' as u32); // unchanged",
        "        assert_eq!(simple_case_fold('0' as u32), '0' as u32); // unchanged",
        "        ",
        "        // Case fold equivalents",
        "        let equiv_a = case_fold_equivalents('A' as u32);",
        "        assert!(equiv_a.contains(&('a' as u32)));",
        "        assert!(equiv_a.contains(&('A' as u32)));",
        "        ",
        "        let equiv_lower = case_fold_equivalents('a' as u32);",
        "        assert!(equiv_lower.contains(&('a' as u32)));",
        "        assert!(equiv_lower.contains(&('A' as u32)));",
        "    }",
        "}",
        "",
    ]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate unicode_data.rs from Unicode Character Database tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_unicode.py
  python scripts/generate_unicode.py --data-dir /path/to/ucd-tables
  python scripts/generate_unicode.py --output /path/to/output.rs
  python scripts/generate_unicode.py --verify
        """
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory containing ucd-generate output files (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output file path (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify tables can be parsed without writing output"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print detailed statistics about Unicode coverage"
    )
    return parser.parse_args()


def print_stats(
    general_categories: dict[str, list[tuple[int, int]]],
    scripts: dict[str, list[tuple[int, int]]],
    bool_properties: dict[str, list[tuple[int, int]]],
    perl_word: list[tuple[int, int]],
    perl_decimal: list[tuple[int, int]],
    perl_space: list[tuple[int, int]]
):
    """Print detailed statistics about the Unicode tables."""

    def count_codepoints(ranges: list[tuple[int, int]]) -> int:
        return sum(end - start + 1 for start, end in ranges)

    print("\n" + "=" * 60)
    print("Unicode Tables Statistics")
    print("=" * 60)

    print(f"\nGeneral Categories: {len(general_categories)}")
    total_gc = 0
    for name in sorted(general_categories.keys()):
        count = count_codepoints(general_categories[name])
        total_gc += count
        print(f"  {name:30} {count:>8,} code points")

    print(f"\nScripts: {len(scripts)}")
    for name in sorted(scripts.keys()):
        count = count_codepoints(scripts[name])
        print(f"  {name:30} {count:>8,} code points")

    print(f"\nBoolean Properties: {len(bool_properties)}")
    for name in sorted(bool_properties.keys()):
        count = count_codepoints(bool_properties[name])
        print(f"  {name:30} {count:>8,} code points")

    print(f"\nPerl Character Classes:")
    print(f"  {'PERL_WORD':30} {count_codepoints(perl_word):>8,} code points")
    print(f"  {'PERL_DECIMAL':30} {count_codepoints(perl_decimal):>8,} code points")
    print(f"  {'PERL_SPACE':30} {count_codepoints(perl_space):>8,} code points")

    print("\n" + "=" * 60)


def run(data_dir: Path, output_file: Path, verify_only: bool = False, show_stats: bool = False):
    """Main entry point with configurable paths."""
    print(f"Reading Unicode data from {data_dir}")

    # Check required files exist
    required_files = [
        "general_category.rs",
        "script.rs",
        "property_bool.rs",
        "perl_word.rs",
        "perl_decimal.rs",
        "perl_space.rs"
    ]

    missing = [f for f in required_files if not (data_dir / f).exists()]
    if missing:
        print(f"Error: Missing required files: {missing}", file=sys.stderr)
        print(f"Run update_unicode_tables.sh to generate them.", file=sys.stderr)
        sys.exit(1)

    # Read general_category.rs
    gc_content = (data_dir / "general_category.rs").read_text()
    gc_by_name = parse_by_name(gc_content)

    # Read script.rs
    script_content = (data_dir / "script.rs").read_text()
    script_by_name = parse_by_name(script_content)

    # Read property_bool.rs for boolean properties
    bool_content = (data_dir / "property_bool.rs").read_text()
    bool_by_name = parse_by_name(bool_content)

    # Read perl classes
    perl_word_content = (data_dir / "perl_word.rs").read_text()
    perl_decimal_content = (data_dir / "perl_decimal.rs").read_text()
    perl_space_content = (data_dir / "perl_space.rs").read_text()

    # Parse General Categories
    general_categories = {}
    for name, const_name in gc_by_name.items():
        ranges = parse_rust_ranges(gc_content, const_name)
        if ranges:
            general_categories[name] = merge_ranges(ranges)
            print(f"  General Category: {name} ({len(ranges)} ranges)")

    # Parse Scripts
    scripts = {}
    for name, const_name in script_by_name.items():
        ranges = parse_rust_ranges(script_content, const_name)
        if ranges:
            scripts[name] = merge_ranges(ranges)
            print(f"  Script: {name} ({len(ranges)} ranges)")

    # Parse Boolean Properties
    bool_properties = {}
    for name, const_name in bool_by_name.items():
        ranges = parse_rust_ranges(bool_content, const_name)
        if ranges:
            bool_properties[name] = merge_ranges(ranges)
            print(f"  Boolean Property: {name} ({len(ranges)} ranges)")

    # Parse Perl classes (using actual const names from files)
    perl_word = parse_rust_ranges(perl_word_content, "PERL_WORD")
    perl_decimal = parse_rust_ranges(perl_decimal_content, "DECIMAL_NUMBER")
    perl_space = parse_rust_ranges(perl_space_content, "WHITE_SPACE")
    print(f"  Perl Word: {len(perl_word)} ranges")
    print(f"  Perl Decimal: {len(perl_decimal)} ranges")
    print(f"  Perl Space: {len(perl_space)} ranges")

    # Parse case folding table
    case_folding = []
    case_folding_path = data_dir / "case_folding_simple.rs"
    if case_folding_path.exists():
        case_folding_content = case_folding_path.read_text()
        case_folding = parse_rust_ranges(case_folding_content, "CASE_FOLDING_SIMPLE")
        print(f"  Case Folding: {len(case_folding)} mappings")

    # Show statistics if requested
    if show_stats:
        print_stats(
            general_categories, scripts, bool_properties,
            merge_ranges(perl_word), merge_ranges(perl_decimal), merge_ranges(perl_space)
        )

    if verify_only:
        print("\nVerification complete. No output written.")
        return

    # Generate output
    output = generate_rust_module(
        general_categories, scripts, bool_properties,
        merge_ranges(perl_word), merge_ranges(perl_decimal), merge_ranges(perl_space),
        case_folding
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(output)
    print(f"\nWritten to {output_file}")
    print(f"Total size: {len(output):,} bytes")


if __name__ == "__main__":
    args = parse_args()
    run(
        data_dir=args.data_dir,
        output_file=args.output,
        verify_only=args.verify,
        show_stats=args.stats
    )
