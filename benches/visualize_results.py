#!/usr/bin/env python3
"""
Benchmark results visualization tool.

Parses Criterion benchmark results and generates publication-quality charts
comparing regexr against competitor regex engines.

Requirements:
    pip install matplotlib pandas seaborn

Usage:
    python visualize_results.py
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns

# Use a colorblind-friendly palette
sns.set_palette("colorblind")
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 10


def parse_size(size: str) -> int:
    """Parse size string like '1KB', '10KB', '100KB' to numeric bytes for sorting."""
    match = re.match(r'^(\d+)([KMG]?)B?$', size, re.IGNORECASE)
    if not match:
        return 0
    value = int(match.group(1))
    unit = match.group(2).upper()
    multipliers = {'': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3}
    return value * multipliers.get(unit, 1)


def sort_sizes(sizes) -> list:
    """Sort size strings numerically (1KB < 10KB < 100KB)."""
    return sorted(sizes, key=parse_size)

# Output directory for visualizations
OUTPUT_DIR = Path(__file__).parent / "results" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Criterion benchmark output directory
CRITERION_DIR = Path(__file__).parent.parent / "target" / "criterion"


def parse_criterion_results() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Parse Criterion benchmark results from JSON files.

    Returns:
        Dict mapping: benchmark_name -> engine_name -> metric_name -> value
    """
    results = {}

    if not CRITERION_DIR.exists():
        print(f"Error: Criterion directory not found at {CRITERION_DIR}")
        print("Please run benchmarks first: cargo bench --bench competitors")
        return results

    # Walk through all benchmark directories
    for bench_dir in CRITERION_DIR.iterdir():
        if not bench_dir.is_dir():
            continue

        bench_name = bench_dir.name

        # Look for subdirectories (different engines)
        for engine_dir in bench_dir.iterdir():
            if not engine_dir.is_dir():
                continue

            engine_name = engine_dir.name
            estimates_file = engine_dir / "new" / "estimates.json"

            if not estimates_file.exists():
                continue

            try:
                with open(estimates_file) as f:
                    data = json.load(f)

                # Extract mean time in nanoseconds
                mean_ns = data.get("mean", {}).get("point_estimate", 0)

                # Store result
                if bench_name not in results:
                    results[bench_name] = {}
                if engine_name not in results[bench_name]:
                    results[bench_name][engine_name] = {}

                results[bench_name][engine_name]["mean_ns"] = mean_ns

            except Exception as e:
                print(f"Warning: Failed to parse {estimates_file}: {e}")

    return results


def extract_benchmark_info(bench_name: str) -> Tuple[str, str]:
    """Extract test type and size from benchmark name."""
    # Try slash-separated format first: test_type/size/engine or test_type/size
    parts = bench_name.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]

    # Try underscore format: TEST_TYPE_SIZE (e.g., ALTERNATION_100KB)
    # Look for size patterns like 1KB, 10KB, 100KB at the end
    size_pattern = re.match(r'^(.+?)_((?:\d+[KMG]?B))$', bench_name, re.IGNORECASE)
    if size_pattern:
        test_type = size_pattern.group(1)
        size = size_pattern.group(2)
        return test_type, size

    return bench_name, "unknown"


def create_throughput_comparison(results: Dict, output_file: str):
    """Create bar chart comparing throughput across engines, one row per benchmark+size."""

    # Organize data by test type and size
    data_points = []

    for bench_name, engines in results.items():
        test_type, size = extract_benchmark_info(bench_name)

        for engine, metrics in engines.items():
            mean_ns = metrics.get("mean_ns", 0)
            if mean_ns > 0:
                # Convert to operations per second (assuming single operation per benchmark)
                ops_per_sec = 1_000_000_000 / mean_ns
                data_points.append({
                    "Test": test_type.replace("_", " ").title(),
                    "Size": size,
                    "Engine": engine,
                    "Ops/sec": ops_per_sec
                })

    if not data_points:
        print("No data points found for throughput comparison")
        return

    df = pd.DataFrame(data_points)

    # Filter out entries with unknown size (compilation benchmarks are handled separately)
    df = df[df["Size"] != "unknown"]

    if df.empty:
        print("No data points with known sizes for throughput comparison")
        return

    # Get unique test types and sizes, sorted properly
    test_types = sorted(df["Test"].unique())
    sizes = sort_sizes(df["Size"].unique())
    engines = sorted(df["Engine"].unique())

    # Build list of (test, size) pairs in order: test1-1kb, test1-10kb, test1-100kb, test2-1kb, ...
    test_size_pairs = []
    for test in test_types:
        for size in sizes:
            test_size_pairs.append((test, size))

    # Create one subplot per test+size combination, single column
    n_charts = len(test_size_pairs)
    fig, axes = plt.subplots(n_charts, 1, figsize=(6, 4 * n_charts))

    if n_charts == 1:
        axes = [axes]

    # Get color palette for engines
    colors = sns.color_palette("colorblind", len(engines))

    for idx, (test, size) in enumerate(test_size_pairs):
        subset = df[(df["Test"] == test) & (df["Size"] == size)]

        # Create vertical bar chart with engines on x-axis
        engine_values = []
        for engine in engines:
            engine_data = subset[subset["Engine"] == engine]
            if len(engine_data) > 0:
                engine_values.append(engine_data["Ops/sec"].values[0])
            else:
                engine_values.append(0)

        x = range(len(engines))
        bars = axes[idx].bar(x, engine_values, color=colors, width=1.0)
        axes[idx].set_xticks([])
        axes[idx].set_xlim(-2.5, len(engines) + 1.5)
        axes[idx].set_ylabel("Operations/second")
        axes[idx].set_title(f"{test} - {size}")
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].legend(bars, engines, loc='upper right')
        axes[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K' if x >= 1e3 else f'{x:.0f}'))
        # Add top padding
        ymin, ymax = axes[idx].get_ylim()
        axes[idx].set_ylim(ymin, ymax * 1.3)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Created: {output_file}")
    plt.close()


def create_latency_comparison(results: Dict, output_file: str):
    """Create bar chart comparing latency across engines, one row per benchmark+size."""

    data_points = []

    for bench_name, engines in results.items():
        test_type, size = extract_benchmark_info(bench_name)

        for engine, metrics in engines.items():
            mean_ns = metrics.get("mean_ns", 0)
            if mean_ns > 0:
                # Convert to microseconds for readability
                mean_us = mean_ns / 1000
                data_points.append({
                    "Test": test_type.replace("_", " ").title(),
                    "Size": size,
                    "Engine": engine,
                    "Latency (μs)": mean_us
                })

    if not data_points:
        print("No data points found for latency comparison")
        return

    df = pd.DataFrame(data_points)

    # Filter out entries with unknown size (compilation benchmarks are handled separately)
    df = df[df["Size"] != "unknown"]

    if df.empty:
        print("No data points with known sizes for latency comparison")
        return

    # Get unique test types and sizes, sorted properly
    test_types = sorted(df["Test"].unique())
    sizes = sort_sizes(df["Size"].unique())
    engines = sorted(df["Engine"].unique())

    # Build list of (test, size) pairs in order: test1-1kb, test1-10kb, test1-100kb, test2-1kb, ...
    test_size_pairs = []
    for test in test_types:
        for size in sizes:
            test_size_pairs.append((test, size))

    # Create one subplot per test+size combination, single column
    n_charts = len(test_size_pairs)
    fig, axes = plt.subplots(n_charts, 1, figsize=(10, 3 * n_charts))

    if n_charts == 1:
        axes = [axes]

    # Get color palette for engines
    colors = sns.color_palette("colorblind", len(engines))

    for idx, (test, size) in enumerate(test_size_pairs):
        subset = df[(df["Test"] == test) & (df["Size"] == size)]

        # Create vertical bar chart with engines on x-axis
        engine_values = []
        for engine in engines:
            engine_data = subset[subset["Engine"] == engine]
            if len(engine_data) > 0:
                engine_values.append(engine_data["Latency (μs)"].values[0])
            else:
                engine_values.append(0)

        x = range(len(engines))
        axes[idx].bar(x, engine_values, color=colors)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(engines, rotation=45, ha='right')
        axes[idx].set_ylabel("Latency (μs)")
        axes[idx].set_title(f"{test} - {size}")
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Created: {output_file}")
    plt.close()


def create_speedup_heatmap(results: Dict, output_file: str):
    """Create heatmap showing speedup relative to baseline."""

    # Calculate speedup relative to 'regex' (the standard Rust regex)
    speedup_data = []

    for bench_name, engines in results.items():
        test_type, size = extract_benchmark_info(bench_name)

        # Skip compilation benchmarks (handled separately)
        if test_type.lower().startswith("compilation"):
            continue

        if "regex" not in engines:
            continue

        baseline_ns = engines["regex"].get("mean_ns", 0)
        if baseline_ns == 0:
            continue

        for engine, metrics in engines.items():
            if engine == "regex":
                continue

            mean_ns = metrics.get("mean_ns", 0)
            if mean_ns > 0:
                speedup = baseline_ns / mean_ns  # >1 means faster than baseline
                # Use "test_type/size" format, but just "test_type" if size is unknown
                test_label = f"{test_type}/{size}" if size != "unknown" else test_type
                speedup_data.append({
                    "Test": test_label,
                    "Engine": engine,
                    "Speedup": speedup
                })

    if not speedup_data:
        print("No data points found for speedup heatmap")
        return

    df = pd.DataFrame(speedup_data)

    # Pivot for heatmap
    pivot_df = df.pivot_table(index="Test", columns="Engine", values="Speedup")

    # Sort rows: group by test type, then by size (1KB, 10KB, 100KB)
    def sort_key(label):
        if "/" in label:
            test, size = label.rsplit("/", 1)
            return (test, parse_size(size))
        return (label, 0)

    sorted_index = sorted(pivot_df.index, key=sort_key)
    pivot_df = pivot_df.reindex(sorted_index)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot_df) * 0.4)))

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=1.0,
        vmin=0.5,
        vmax=2.0,
        ax=ax,
        cbar_kws={'label': 'Speedup vs. regex (>1 is faster)'}
    )

    ax.set_title("Performance Comparison - Speedup Relative to 'regex' Crate")
    ax.set_xlabel("Engine")
    ax.set_ylabel("Test / Input Size")

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Created: {output_file}")
    plt.close()


def create_compilation_time_chart(results: Dict, output_file: str):
    """Create chart comparing compilation times."""

    # Filter for compilation benchmarks
    comp_data = []

    for bench_name, engines in results.items():
        # Support both "compilation/pattern" and "compilation_pattern" formats
        if bench_name.startswith("compilation/"):
            pattern_type = bench_name.replace("compilation/", "")
        elif bench_name.lower().startswith("compilation_"):
            pattern_type = bench_name[len("compilation_"):]
        else:
            continue

        for engine, metrics in engines.items():
            mean_ns = metrics.get("mean_ns", 0)
            if mean_ns > 0:
                mean_us = mean_ns / 1000
                comp_data.append({
                    "Pattern": pattern_type.replace("_", " ").title(),
                    "Engine": engine,
                    "Time (μs)": mean_us
                })

    if not comp_data:
        print("No compilation benchmark data found")
        return

    df = pd.DataFrame(comp_data)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    patterns = sorted(df["Pattern"].unique())
    x = range(len(patterns))
    width = 0.15
    engines = sorted(df["Engine"].unique())

    for i, engine in enumerate(engines):
        engine_data = []
        for pattern in patterns:
            pattern_data = df[(df["Engine"] == engine) & (df["Pattern"] == pattern)]
            if len(pattern_data) > 0:
                engine_data.append(pattern_data["Time (μs)"].values[0])
            else:
                engine_data.append(0)

        ax.bar([p + width * i for p in x], engine_data, width, label=engine)

    ax.set_xlabel("Pattern Type")
    ax.set_ylabel("Compilation Time (μs) - lower is better")
    ax.set_title("Regex Compilation Time Comparison")
    ax.set_xticks([p + width * (len(engines) - 1) / 2 for p in x])
    ax.set_xticklabels(patterns)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Created: {output_file}")
    plt.close()


def create_summary_table(results: Dict, output_file: str):
    """Create a comprehensive summary table in CSV format."""

    data_points = []

    for bench_name, engines in results.items():
        test_type, size = extract_benchmark_info(bench_name)

        for engine, metrics in engines.items():
            mean_ns = metrics.get("mean_ns", 0)
            if mean_ns > 0:
                data_points.append({
                    "Benchmark": bench_name,
                    "Test Type": test_type,
                    "Size": size,
                    "Engine": engine,
                    "Mean (ns)": mean_ns,
                    "Mean (μs)": mean_ns / 1000,
                    "Mean (ms)": mean_ns / 1_000_000,
                })

    if not data_points:
        print("No data points found for summary table")
        return

    df = pd.DataFrame(data_points)

    # Sort by benchmark and engine
    df = df.sort_values(["Test Type", "Size", "Engine"])

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Created: {output_file}")

    # Also create a pivot table showing relative performance
    summary_file = output_file.replace(".csv", "_summary.txt")

    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BENCHMARK RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Group by test type
        for test_type in sorted(df["Test Type"].unique()):
            f.write(f"\n{test_type.upper()}\n")
            f.write("-" * 80 + "\n")

            test_df = df[df["Test Type"] == test_type]

            for size in sort_sizes(test_df["Size"].unique()):
                if size != "unknown":
                    f.write(f"\n  {size}:\n")
                else:
                    f.write("\n")
                size_df = test_df[test_df["Size"] == size]

                # Find fastest engine
                fastest = size_df.loc[size_df["Mean (ns)"].idxmin()]

                for _, row in size_df.iterrows():
                    speedup = fastest["Mean (ns)"] / row["Mean (ns)"]
                    marker = " (FASTEST)" if row["Engine"] == fastest["Engine"] else f" ({speedup:.2f}x slower)"
                    f.write(f"    {row['Engine']:15s}: {row['Mean (μs)']:10.2f} μs{marker}\n")

    print(f"Created: {summary_file}")


def main():
    """Main entry point."""
    print("Parsing Criterion benchmark results...")
    results = parse_criterion_results()

    if not results:
        print("\nNo benchmark results found!")
        print("Please run: cargo bench --bench competitors")
        return

    print(f"\nFound {len(results)} benchmark results")
    print(f"Generating visualizations in: {OUTPUT_DIR}\n")

    # Create all visualizations
    create_throughput_comparison(
        results,
        str(OUTPUT_DIR / "throughput_comparison.png")
    )

    create_latency_comparison(
        results,
        str(OUTPUT_DIR / "latency_comparison.png")
    )

    create_speedup_heatmap(
        results,
        str(OUTPUT_DIR / "speedup_heatmap.png")
    )

    create_compilation_time_chart(
        results,
        str(OUTPUT_DIR / "compilation_time.png")
    )

    create_summary_table(
        results,
        str(OUTPUT_DIR.parent / "detailed_results.csv")
    )

    print("\nVisualization complete!")
    print(f"View results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
