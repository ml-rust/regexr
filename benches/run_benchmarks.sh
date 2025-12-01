#!/bin/bash
# Benchmark runner script for regexr
#
# This script runs benchmarks and optionally generates visualizations.

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  regexr Benchmark Suite${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Parse arguments
VISUALIZE=false
FEATURE_FLAGS=""
CONFIG_NAME="default"

while [[ $# -gt 0 ]]; do
    case $1 in
        --visualize)
            VISUALIZE=true
            shift
            ;;
        --base)
            FEATURE_FLAGS="--no-default-features --features simd"
            CONFIG_NAME="base (SIMD only)"
            shift
            ;;
        --full)
            FEATURE_FLAGS="--features full"
            CONFIG_NAME="full (SIMD + JIT)"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --visualize    Generate visualizations after benchmarks"
            echo "  --base         Run with regexr-base (SIMD only, no JIT)"
            echo "  --full         Run with regexr-full (SIMD + JIT)"
            echo "  --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                     # Run with default features"
            echo "  $0 --base              # Run with SIMD only"
            echo "  $0 --full --visualize  # Run with full features and generate charts"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Configuration:${NC} $CONFIG_NAME"
echo -e "${YELLOW}Feature flags:${NC} $FEATURE_FLAGS"
echo ""

# Check for PCRE2 library
echo -e "${BLUE}Checking dependencies...${NC}"
if ! pkg-config --exists libpcre2-8 2>/dev/null; then
    echo -e "${YELLOW}Warning: PCRE2 library not found. Some benchmarks may fail.${NC}"
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install libpcre2-dev"
    echo "  macOS:         brew install pcre2"
    echo "  Fedora/RHEL:   sudo dnf install pcre2-devel"
    echo ""
else
    PCRE2_VERSION=$(pkg-config --modversion libpcre2-8)
    echo -e "${GREEN}PCRE2 version: $PCRE2_VERSION${NC}"
fi

# Display system info
echo ""
echo -e "${BLUE}System Information:${NC}"
echo "Rust version: $(rustc --version)"
echo "Cargo version: $(cargo --version)"
echo "CPU: $(lscpu | grep 'Model name' | sed 's/Model name:[[:space:]]*//' || echo 'Unknown')"
echo "OS: $(uname -s) $(uname -r)"
echo ""

# Run benchmarks
echo -e "${BLUE}Running benchmarks...${NC}"
echo "This may take several minutes."
echo ""

cargo bench --bench competitors $FEATURE_FLAGS

echo ""
echo -e "${GREEN}Benchmarks complete!${NC}"
echo ""
echo "View detailed HTML reports at:"
echo "  file://$(pwd)/target/criterion/report/index.html"
echo ""

# Generate visualizations if requested
if [ "$VISUALIZE" = true ]; then
    echo -e "${BLUE}Generating visualizations...${NC}"

    # Check for Python and dependencies
    if ! command -v python3 &> /dev/null; then
        echo -e "${YELLOW}Warning: python3 not found. Skipping visualizations.${NC}"
        echo "Install Python 3 and run: python3 benches/visualize_results.py"
    else
        # Check if required packages are installed
        if python3 -c "import matplotlib, pandas, seaborn" 2>/dev/null; then
            python3 benches/visualize_results.py
            echo ""
            echo -e "${GREEN}Visualizations generated!${NC}"
            echo "View charts in: benches/results/visualizations/"
        else
            echo -e "${YELLOW}Warning: Required Python packages not found.${NC}"
            echo "Install with: pip install matplotlib pandas seaborn"
            echo "Then run: python3 benches/visualize_results.py"
        fi
    fi
fi

echo ""
echo -e "${GREEN}Done!${NC}"
