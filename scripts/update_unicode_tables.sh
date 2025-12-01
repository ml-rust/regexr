#!/bin/bash
#
# Update Unicode tables from the Unicode Character Database (UCD).
#
# This script:
# 1. Downloads the specified UCD version (default: latest)
# 2. Uses ucd-generate to create Rust source files
# 3. Runs generate_unicode.py to create the consolidated unicode_data.rs
#
# Requirements:
#   - curl
#   - unzip
#   - ucd-generate (cargo install ucd-generate)
#
# Usage:
#   ./scripts/update_unicode_tables.sh              # Use latest UCD version
#   ./scripts/update_unicode_tables.sh 16.0.0       # Use specific version
#   ./scripts/update_unicode_tables.sh --help       # Show help
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
UNICODE_TABLES_DIR="$DATA_DIR/unicode_tables"
TEMP_DIR="$DATA_DIR/.ucd-temp"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default UCD version (latest as of Nov 2024)
DEFAULT_UCD_VERSION="16.0.0"
UCD_VERSION="${1:-$DEFAULT_UCD_VERSION}"

show_help() {
    echo "Usage: $0 [UCD_VERSION]"
    echo ""
    echo "Update Unicode tables from the Unicode Character Database."
    echo ""
    echo "Arguments:"
    echo "  UCD_VERSION    Unicode version to download (default: $DEFAULT_UCD_VERSION)"
    echo ""
    echo "Examples:"
    echo "  $0              # Download UCD $DEFAULT_UCD_VERSION"
    echo "  $0 15.1.0       # Download UCD 15.1.0"
    echo "  $0 16.0.0       # Download UCD 16.0.0"
    echo ""
    echo "Available versions: https://www.unicode.org/Public/"
    exit 0
}

# Handle --help
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_help
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Unicode Tables Update Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "UCD Version: ${GREEN}$UCD_VERSION${NC}"
echo -e "Output Dir:  ${GREEN}$UNICODE_TABLES_DIR${NC}"
echo ""

# Check for required tools
check_requirements() {
    local missing=()

    if ! command -v curl &> /dev/null; then
        missing+=("curl")
    fi

    if ! command -v unzip &> /dev/null; then
        missing+=("unzip")
    fi

    if ! command -v ucd-generate &> /dev/null; then
        echo -e "${YELLOW}Warning: ucd-generate not found. Installing...${NC}"
        cargo install ucd-generate
    fi

    if ! command -v python3 &> /dev/null; then
        missing+=("python3")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo -e "${RED}Error: Missing required tools: ${missing[*]}${NC}"
        echo "Please install them and try again."
        exit 1
    fi
}

# Download UCD data - sets UCD_DIR global variable
download_ucd() {
    local version="$1"
    local ucd_url="https://www.unicode.org/Public/$version/ucd/UCD.zip"
    local dest_dir="$TEMP_DIR/ucd-$version"
    local zip_file="$TEMP_DIR/UCD-$version.zip"

    echo -e "${BLUE}[1/4] Downloading UCD $version...${NC}"

    mkdir -p "$TEMP_DIR"

    if [[ -d "$dest_dir" ]]; then
        echo "  Using cached UCD data at $dest_dir"
    else
        if [[ ! -f "$zip_file" ]]; then
            echo "  Downloading from $ucd_url"
            # Retry up to 3 times with exponential backoff
            local retries=3
            local delay=2
            local success=false
            for ((i=1; i<=retries; i++)); do
                if curl -L --connect-timeout 30 --retry 3 -o "$zip_file" "$ucd_url"; then
                    success=true
                    break
                else
                    echo -e "${YELLOW}  Download attempt $i failed. Retrying in ${delay}s...${NC}"
                    sleep $delay
                    delay=$((delay * 2))
                fi
            done
            if [[ "$success" != "true" ]]; then
                echo -e "${RED}Failed to download UCD after $retries attempts.${NC}"
                echo -e "${RED}Check your network connection and verify version $version exists at:${NC}"
                echo -e "${RED}  https://www.unicode.org/Public/${NC}"
                rm -f "$zip_file"
                exit 1
            fi
        fi

        echo "  Extracting..."
        mkdir -p "$dest_dir"
        unzip -q "$zip_file" -d "$dest_dir"
        echo -e "  ${GREEN}Downloaded UCD $version${NC}"
    fi

    # Set global variable instead of echoing (to avoid output capture issues)
    UCD_DIR="$dest_dir"
}

# Generate tables using ucd-generate
generate_tables() {
    local ucd_dir="$1"
    local output_dir="$2"

    echo -e "${BLUE}[2/4] Generating Unicode tables with ucd-generate...${NC}"

    mkdir -p "$output_dir"

    # General Category
    echo "  Generating general_category.rs..."
    ucd-generate general-category "$ucd_dir" > "$output_dir/general_category.rs"

    # Scripts
    echo "  Generating script.rs..."
    ucd-generate script "$ucd_dir" > "$output_dir/script.rs"

    # Script Extensions
    echo "  Generating script_extension.rs..."
    ucd-generate script-extension "$ucd_dir" > "$output_dir/script_extension.rs"

    # Boolean Properties
    echo "  Generating property_bool.rs..."
    ucd-generate property-bool "$ucd_dir" > "$output_dir/property_bool.rs"

    # Property Names (for reference)
    echo "  Generating property_names.rs..."
    ucd-generate property-names "$ucd_dir" > "$output_dir/property_names.rs"

    # Property Values (for reference)
    echo "  Generating property_values.rs..."
    ucd-generate property-values "$ucd_dir" > "$output_dir/property_values.rs"

    # Perl character classes
    echo "  Generating perl_word.rs..."
    ucd-generate perl-word "$ucd_dir" > "$output_dir/perl_word.rs"

    # perl_decimal: Extract Decimal_Number (Nd) from general-category
    echo "  Generating perl_decimal.rs..."
    ucd-generate general-category "$ucd_dir" --include decimalnumber > "$output_dir/perl_decimal.rs"

    # perl_space: Extract whitespace from property-bool
    echo "  Generating perl_space.rs..."
    ucd-generate property-bool "$ucd_dir" --include whitespace > "$output_dir/perl_space.rs"

    # Word/Sentence/Grapheme break properties (for future use)
    echo "  Generating word_break.rs..."
    ucd-generate word-break "$ucd_dir" > "$output_dir/word_break.rs"

    echo "  Generating sentence_break.rs..."
    ucd-generate sentence-break "$ucd_dir" > "$output_dir/sentence_break.rs"

    echo "  Generating grapheme_cluster_break.rs..."
    ucd-generate grapheme-cluster-break "$ucd_dir" > "$output_dir/grapheme_cluster_break.rs"

    # Age property
    echo "  Generating age.rs..."
    ucd-generate age "$ucd_dir" > "$output_dir/age.rs"

    # Case folding (for case-insensitive matching)
    echo "  Generating case_folding_simple.rs..."
    ucd-generate case-folding-simple "$ucd_dir" > "$output_dir/case_folding_simple.rs"

    # Generate mod.rs
    echo "  Generating mod.rs..."
    cat > "$output_dir/mod.rs" << 'EOF'
//! Unicode Character Database tables.
//!
//! These tables are auto-generated from the Unicode Character Database.
//! See the LICENSE-UNICODE file for the Unicode license.

pub mod age;
pub mod case_folding_simple;
pub mod general_category;
pub mod grapheme_cluster_break;
pub mod perl_decimal;
pub mod perl_space;
pub mod perl_word;
pub mod property_bool;
pub mod property_names;
pub mod property_values;
pub mod script;
pub mod script_extension;
pub mod sentence_break;
pub mod word_break;
EOF

    # Copy LICENSE
    echo "  Writing Unicode license..."
    cat > "$output_dir/LICENSE-UNICODE" << 'EOF'
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2025 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
EOF

    echo -e "  ${GREEN}Generated all tables${NC}"
}

# Run the Python script to consolidate tables
consolidate_tables() {
    echo -e "${BLUE}[3/4] Consolidating tables with generate_unicode.py...${NC}"

    cd "$PROJECT_ROOT"
    python3 scripts/generate_unicode.py

    echo -e "  ${GREEN}Generated src/hir/unicode_data.rs${NC}"
}

# Verify the generated code compiles
verify_build() {
    echo -e "${BLUE}[4/4] Verifying build...${NC}"

    cd "$PROJECT_ROOT"

    echo "  Running cargo check..."
    if cargo check 2>&1 | head -20; then
        echo -e "  ${GREEN}Build check passed${NC}"
    else
        echo -e "${RED}Build check failed. Please review generated code.${NC}"
        exit 1
    fi

    echo "  Running tests..."
    if cargo test 2>&1 | tail -10; then
        echo -e "  ${GREEN}Tests passed${NC}"
    else
        echo -e "${YELLOW}Some tests may have failed. Please review.${NC}"
    fi
}

# Clean up temporary files
cleanup() {
    if [[ -d "$TEMP_DIR" ]]; then
        echo ""
        echo -e "${YELLOW}Temporary files at $TEMP_DIR can be removed with:${NC}"
        echo "  rm -rf $TEMP_DIR"
    fi
}

# Main
main() {
    check_requirements

    # UCD_DIR is set as a global by download_ucd
    download_ucd "$UCD_VERSION"

    generate_tables "$UCD_DIR" "$UNICODE_TABLES_DIR"
    consolidate_tables
    verify_build
    cleanup

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Unicode tables updated successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "UCD Version: $UCD_VERSION"
    echo "Tables:      $UNICODE_TABLES_DIR"
    echo "Output:      src/hir/unicode_data.rs"
}

main
