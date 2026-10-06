#!/usr/bin/env python3
"""
analyze_results.py — Benchmark Result Analyzer (Phase 17)

Reads results/benchmark_results.csv and produces:
  1. A rich console summary table
  2. Speedup analysis
  3. Per-operator performance comparison charts (if matplotlib available)
  4. A markdown summary table

Usage:
    python scripts/analyze_results.py [--csv PATH] [--plot]
"""

import sys
import csv
import os
import argparse
from collections import defaultdict

# ── CSV schema ────────────────────────────────────────────────────────────────
# operator, shape, kernel, avg_ms, min_ms, med_ms, tflops, speedup_vs_naive

def load_results(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                row = {
                    'operator': r['operator'],
                    'shape':    r['shape'],
                    'kernel':   r['kernel'],
                    'avg_ms':   float(r['avg_ms']),
                    'min_ms':   float(r['min_ms']),
                    'med_ms':   float(r['med_ms']),
                    'tflops':   float(r.get('tflops', 0)),
                }
                rows.append(row)
            except (ValueError, KeyError) as e:
                print(f"  [WARN] Skipping row: {e}")
    return rows

def compute_speedups(rows):
    """
    For each (operator, shape), compute speedup of each kernel vs the Naive baseline.
    Baseline = slowest kernel (typically V1-Naive or first entry).
    """
    groups = defaultdict(list)
    for r in rows:
        groups[(r['operator'], r['shape'])].append(r)

    enhanced = []
    for (op, shape), grp in groups.items():
        # Find naive baseline (max avg_ms = slowest)
        baseline = max(grp, key=lambda x: x['avg_ms'])
        baseline_ms = baseline['avg_ms']
        for r in grp:
            speedup = baseline_ms / r['avg_ms'] if r['avg_ms'] > 0 else 1.0
            enhanced.append({**r, 'speedup': speedup, 'baseline_ms': baseline_ms})
    return enhanced

def print_table(rows):
    """Print a rich ASCII table."""
    col_widths = {'operator': 12, 'shape': 20, 'kernel': 22,
                  'avg_ms': 10, 'min_ms': 10, 'tflops': 8, 'speedup': 8}
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths.values()) + "+"
    header = "| " + " | ".join(
        k.upper().center(v) for k, v in col_widths.items()
    ) + " |"

    print("\n")
    print("╔══ BENCHMARK RESULTS TABLE ══════════════════════════════════════════════════╗")
    print(sep)
    print(header)
    print(sep)

    current_op = None
    for r in sorted(rows, key=lambda x: (x['operator'], x['shape'], x['avg_ms'])):
        if r['operator'] != current_op:
            if current_op is not None:
                print(sep)
            current_op = r['operator']

        def fmt(k, v):
            w = col_widths[k]
            if isinstance(v, float):
                return f"{v:.3f}".center(w)
            return str(v).center(w)

        cells = [
            fmt('operator', r['operator']),
            fmt('shape',    r['shape']),
            fmt('kernel',   r['kernel']),
            fmt('avg_ms',   r['avg_ms']),
            fmt('min_ms',   r['min_ms']),
            fmt('tflops',   r.get('tflops', 0.0)),
            fmt('speedup',  r.get('speedup', 1.0)),
        ]
        print("| " + " | ".join(cells) + " |")

    print(sep)
    print("╚═════════════════════════════════════════════════════════════════════════════╝")

def print_speedup_summary(rows):
    """For each operator×shape, print best speedup vs slowest kernel."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r['operator'], r['shape'])].append(r)

    print("\n╔══ SPEEDUP SUMMARY ═══════════╗")
    for (op, shape), grp in sorted(groups.items()):
        slowest = max(grp, key=lambda x: x['avg_ms'])
        fastest = min(grp, key=lambda x: x['avg_ms'])
        speedup = slowest['avg_ms'] / fastest['avg_ms']
        cublas  = next((r for r in grp if 'cuBLAS' in r['kernel']), None)
        custom_fast = min((r for r in grp if 'cuBLAS' not in r['kernel']),
                          key=lambda x: x['avg_ms'])
        vs_cublas = (cublas['avg_ms'] / custom_fast['avg_ms']) if cublas else None

        print(f"\n  {op:12s} {shape:16s}")
        print(f"    Slowest : {slowest['kernel']:25s} {slowest['avg_ms']:.3f} ms")
        print(f"    Fastest : {fastest['kernel']:25s} {fastest['avg_ms']:.3f} ms")
        print(f"    Max speedup: {speedup:.2f}×")
        if vs_cublas:
            status = "✓ BEATS cuBLAS" if vs_cublas > 1.0 else f"cuBLAS still {1/vs_cublas:.2f}× faster"
            print(f"    Custom vs cuBLAS: {vs_cublas:.2f}× ({status})")
    print()

def write_markdown(rows, out_path="results/summary.md"):
    """Write a GitHub-flavoured Markdown summary table."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r['operator'], r['shape'])].append(r)

    with open(out_path, 'w') as f:
        f.write("# Benchmark Summary\n\n")
        f.write("| Operator | Shape | Naive (ms) | Optimized (ms) | cuBLAS (ms) | Opt Speedup | vs cuBLAS |\n")
        f.write("|----------|-------|-----------|----------------|-------------|-------------|----------|\n")

        for (op, shape), grp in sorted(groups.items()):
            naive   = max(grp, key=lambda x: x['avg_ms'])
            cublas  = next((r for r in grp if 'cuBLAS' in r['kernel']), None)
            custom  = min((r for r in grp if 'cuBLAS' not in r['kernel']),
                          key=lambda x: x['avg_ms'])
            opt_speedup = naive['avg_ms'] / custom['avg_ms']
            vs_cb = (cublas['avg_ms'] / custom['avg_ms']) if cublas else float('nan')
            cublas_str = 'N/A' if not cublas else f'{cublas["avg_ms"]:.3f}'
            vs_cb_str  = 'N/A' if not cublas else f'{vs_cb:.2f}x'
            f.write(f"| {op} | {shape} | {naive['avg_ms']:.3f} | {custom['avg_ms']:.3f} | "
                    f"{cublas_str} | "
                    f"{opt_speedup:.2f}x | {vs_cb_str} |\n")

    print(f"  Markdown summary written to: {out_path}")

def plot_results(rows, do_plot=False):
    if not do_plot:
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        operators = sorted(set(r['operator'] for r in rows))
        fig, axes = plt.subplots(1, len(operators), figsize=(18, 6))
        if len(operators) == 1:
            axes = [axes]

        for ax, op in zip(axes, operators):
            op_rows = [r for r in rows if r['operator'] == op]
            shapes = sorted(set(r['shape'] for r in op_rows))

            for shape in shapes:
                shape_rows = sorted(
                    [r for r in op_rows if r['shape'] == shape],
                    key=lambda x: x['avg_ms'], reverse=True
                )
                kernels = [r['kernel'] for r in shape_rows]
                times   = [r['avg_ms']  for r in shape_rows]
                ax.bar(range(len(kernels)), times, label=shape, alpha=0.7)
                ax.set_xticks(range(len(kernels)))
                ax.set_xticklabels(kernels, rotation=45, ha='right', fontsize=7)

            ax.set_title(f"{op} Latency (ms)")
            ax.set_ylabel("avg_ms")
            ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig("results/benchmark_chart.png", dpi=150)
        print("  Chart saved to results/benchmark_chart.png")
    except ImportError:
        print("  matplotlib not available; skipping chart generation.")

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CUDA kernel benchmark results")
    parser.add_argument("--csv",  default="results/benchmark_results.csv")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib charts")
    args = parser.parse_args()

    rows = load_results(args.csv)
    if not rows:
        print("[ERROR] No valid rows found in CSV.")
        sys.exit(1)

    enhanced = compute_speedups(rows)
    print_table(enhanced)
    print_speedup_summary(enhanced)
    write_markdown(enhanced)
    plot_results(enhanced, do_plot=args.plot)
