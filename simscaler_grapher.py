#!/usr/bin/env python3
"""
SimpleScalar sim-outorder Results Grapher
==========================================
Supports Part I, II, and III of COA/CA Project (Spring 2026)

Usage:
    python simscalar_grapher.py                          # Interactive mode
    python simscalar_grapher.py file1.txt file2.txt ...  # Pass files directly

The script will:
  - Auto-parse one or more sim-outorder result files
  - Let you label each file (e.g. "Small", "Medium", "Large", "Excessive")
  - Let you choose which graph(s) to generate
  - Save graphs as PNG files
"""

import re
import sys
import os
import matplotlib
matplotlib.use("Agg")          # works even without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  PARSER
# ──────────────────────────────────────────────────────────────────────────────

# Every stat we might want to plot
STAT_PATTERNS = {
    # Core performance
    "sim_num_insn":       r"sim_num_insn\s+([\d.]+)",
    "sim_num_refs":       r"sim_num_refs\s+([\d.]+)",
    "sim_num_loads":      r"sim_num_loads\s+([\d.]+)",
    "sim_num_stores":     r"sim_num_stores\s+([\d.]+)",
    "sim_num_branches":   r"sim_num_branches\s+([\d.]+)",
    "sim_cycle":          r"sim_cycle\s+([\d.]+)",
    "sim_IPC":            r"sim_IPC\s+([\d.]+)",
    "sim_CPI":            r"sim_CPI\s+([\d.]+)",
    "sim_inst_rate":      r"sim_inst_rate\s+([\d.]+)",
    "sim_exec_BW":        r"sim_exec_BW\s+([\d.]+)",
    "sim_IPB":            r"sim_IPB\s+([\d.]+)",

    # IFQ
    "ifq_occupancy":      r"ifq_occupancy\s+([\d.]+)",
    "ifq_rate":           r"ifq_rate\s+([\d.]+)",
    "ifq_latency":        r"ifq_latency\s+([\d.]+)",
    "ifq_full":           r"ifq_full\s+([\d.]+)",

    # RUU
    "ruu_occupancy":      r"ruu_occupancy\s+([\d.]+)",
    "ruu_full":           r"ruu_full\s+([\d.]+)",

    # LSQ
    "lsq_occupancy":      r"lsq_occupancy\s+([\d.]+)",
    "lsq_full":           r"lsq_full\s+([\d.]+)",

    # Slip
    "avg_sim_slip":       r"avg_sim_slip\s+([\d.]+)",

    # Branch predictor — generic (matches bimod / 2lev / comb / perfect)
    "bpred_lookups":      r"bpred[\w.]+\.lookups\s+([\d.]+)",
    "bpred_updates":      r"bpred[\w.]+\.updates\s+([\d.]+)",
    "bpred_addr_hits":    r"bpred[\w.]+\.addr_hits\s+([\d.]+)",
    "bpred_dir_hits":     r"bpred[\w.]+\.dir_hits\s+([\d.]+)",
    "bpred_misses":       r"bpred[\w.]+\.misses\s+([\d.]+)",
    "bpred_addr_rate":    r"bpred[\w.]+\.bpred_addr_rate\s+([\d.]+)",
    "bpred_dir_rate":     r"bpred[\w.]+\.bpred_dir_rate\s+([\d.]+)",
    "bpred_jr_rate":      r"bpred[\w.]+\.bpred_jr_rate\s+([\d.]+)",

    # DL1 cache
    "dl1_accesses":       r"dl1\.accesses\s+([\d.]+)",
    "dl1_hits":           r"dl1\.hits\s+([\d.]+)",
    "dl1_misses":         r"dl1\.misses\s+([\d.]+)",
    "dl1_miss_rate":      r"dl1\.miss_rate\s+([\d.]+)",
    "dl1_repl_rate":      r"dl1\.repl_rate\s+([\d.]+)",
    "dl1_wb_rate":        r"dl1\.wb_rate\s+([\d.]+)",

    # IL1 cache (separate il1 if configured)
    "il1_accesses":       r"il1\.accesses\s+([\d.]+)",
    "il1_hits":           r"il1\.hits\s+([\d.]+)",
    "il1_misses":         r"il1\.misses\s+([\d.]+)",
    "il1_miss_rate":      r"il1\.miss_rate\s+([\d.]+)",

    # DL2 cache
    "dl2_accesses":       r"dl2\.accesses\s+([\d.]+)",
    "dl2_hits":           r"dl2\.hits\s+([\d.]+)",
    "dl2_misses":         r"dl2\.misses\s+([\d.]+)",
    "dl2_miss_rate":      r"dl2\.miss_rate\s+([\d.]+)",

    # IL2 cache
    "il2_accesses":       r"il2\.accesses\s+([\d.]+)",
    "il2_hits":           r"il2\.hits\s+([\d.]+)",
    "il2_misses":         r"il2\.misses\s+([\d.]+)",
    "il2_miss_rate":      r"il2\.miss_rate\s+([\d.]+)",

    # ITLB / DTLB
    "itlb_miss_rate":     r"itlb\.miss_rate\s+([\d.]+)",
    "dtlb_miss_rate":     r"dtlb\.miss_rate\s+([\d.]+)",

    # Memory
    "mem_page_count":     r"mem\.page_count\s+([\d.]+)",

    # Config values (extracted from the options section)
    "fetch_ifqsize":      r"-fetch:ifqsize\s+(\d+)",
    "decode_width":       r"-decode:width\s+(\d+)",
    "issue_width":        r"-issue:width\s+(\d+)",
    "commit_width":       r"-commit:width\s+(\d+)",
    "ruu_size":           r"-ruu:size\s+(\d+)",
    "lsq_size":           r"-lsq:size\s+(\d+)",
    "bpred_bimod_size":   r"-bpred:bimod\s+(\d+)",
}

# Derived stats computed after parsing
DERIVED = {
    "dl1_hit_rate":  lambda s: 1 - s.get("dl1_miss_rate", 0),
    "il1_hit_rate":  lambda s: 1 - s.get("il1_miss_rate", 0),
    "dl2_hit_rate":  lambda s: 1 - s.get("dl2_miss_rate", 0),
    "bpred_miss_rate": lambda s: (
        s["bpred_misses"] / (s["bpred_updates"] or 1)
        if "bpred_misses" in s and "bpred_updates" in s else None
    ),
}


def parse_file(filepath):
    """Parse a sim-outorder result file and return a dict of stats."""
    with open(filepath, "r", errors="replace") as f:
        text = f.read()

    stats = {}
    for key, pattern in STAT_PATTERNS.items():
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            stats[key] = float(m.group(1))

    # Derived values
    for key, fn in DERIVED.items():
        try:
            val = fn(stats)
            if val is not None:
                stats[key] = val
        except Exception:
            pass

    return stats


# ──────────────────────────────────────────────────────────────────────────────
#  PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800",
          "#9C27B0", "#00BCD4", "#E91E63", "#795548"]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def style_ax(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(fontsize=9, framealpha=0.8)


def bar_chart(labels, values, title, xlabel, ylabel, outfile, color="#2196F3"):
    """Single-series bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color, edgecolor="white", linewidth=0.8, width=0.55)
    for bar, val in zip(bars, values):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:,.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    style_ax(ax, title, xlabel, ylabel, legend=False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Saved: {outfile}")


def grouped_bar_chart(group_labels, series_dict, title, xlabel, ylabel, outfile):
    """Multi-series grouped bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    n_groups = len(group_labels)
    n_series = len(series_dict)
    width = 0.7 / n_series
    x = np.arange(n_groups)

    for i, (name, values) in enumerate(series_dict.items()):
        offset = (i - n_series / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name,
                      color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=15, ha="right")
    style_ax(ax, title, xlabel, ylabel)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Saved: {outfile}")


def line_chart(x_vals, series_dict, title, xlabel, ylabel, outfile,
               x_is_numeric=True):
    """Multi-series line chart."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (name, y_vals) in enumerate(series_dict.items()):
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        if x_is_numeric:
            ax.plot(x_vals, y_vals, marker=marker, color=color,
                    linewidth=2, markersize=7, label=name)
        else:
            ax.plot(range(len(x_vals)), y_vals, marker=marker, color=color,
                    linewidth=2, markersize=7, label=name)
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(x_vals, rotation=15, ha="right")

    style_ax(ax, title, xlabel, ylabel)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Saved: {outfile}")


# ──────────────────────────────────────────────────────────────────────────────
#  GRAPH DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

GRAPH_MENU = {
    # ── PART I ──
    "1":  "Part I  │ Execution Time (sim_cycle) vs Memory Config",
    "2":  "Part I  │ CPI (sim_CPI) vs Memory Config",
    "3":  "Part I  │ IPC (sim_IPC) vs Memory Config",
    "4":  "Part I  │ Instruction Rate (sim_inst_rate) vs Memory Config",

    # ── Cache Stats ──
    "5":  "Cache   │ DL1 Miss Rate vs Memory Config",
    "6":  "Cache   │ DL1 Hit Rate  vs Memory Config",
    "7":  "Cache   │ IL1 Miss Rate vs Memory Config",
    "8":  "Cache   │ IL1 Hit Rate  vs Memory Config",
    "9":  "Cache   │ DL2 Miss Rate vs Memory Config (if DL2 present)",
    "10": "Cache   │ DL1 Accesses vs Memory Config",
    "11": "Cache   │ DL1 Hits vs Memory Config",
    "12": "Cache   │ DL1 Misses vs Memory Config",

    # ── PART II – Branch Predictor ──
    "13": "Part II │ Branch Predictor: Dir Hits vs Config",
    "14": "Part II │ Branch Predictor: Misses vs Config",
    "15": "Part II │ Branch Predictor: Addr Hits (BTB) vs Config",
    "16": "Part II │ Branch Predictor: Dir Rate vs Config",
    "17": "Part II │ Branch Predictor: Miss Rate vs Config",

    # ── PART III – Pipeline ──
    "18": "Part III│ IFQ Occupancy vs Config",
    "19": "Part III│ RUU Occupancy vs Config",
    "20": "Part III│ LSQ Occupancy vs Config",
    "21": "Part III│ IPC vs IFQ Size (auto-extracted from config)",
    "22": "Part III│ Total Cycles vs Issue Width (auto-extracted)",

    # ── Misc ──
    "23": "Misc    │ Number of Branches vs Config",
    "24": "Misc    │ ITLB Miss Rate vs Config",
    "25": "Misc    │ DTLB Miss Rate vs Config",

    # ── ALL ──
    "ALL": "Generate ALL graphs listed above",
}

STAT_FOR_GRAPH = {
    "1":  "sim_cycle",
    "2":  "sim_CPI",
    "3":  "sim_IPC",
    "4":  "sim_inst_rate",
    "5":  "dl1_miss_rate",
    "6":  "dl1_hit_rate",
    "7":  "il1_miss_rate",
    "8":  "il1_hit_rate",
    "9":  "dl2_miss_rate",
    "10": "dl1_accesses",
    "11": "dl1_hits",
    "12": "dl1_misses",
    "13": "bpred_dir_hits",
    "14": "bpred_misses",
    "15": "bpred_addr_hits",
    "16": "bpred_dir_rate",
    "17": "bpred_miss_rate",
    "18": "ifq_occupancy",
    "19": "ruu_occupancy",
    "20": "lsq_occupancy",
    "21": ("fetch_ifqsize", "sim_IPC"),
    "22": ("issue_width",   "sim_cycle"),
    "23": "sim_num_branches",
    "24": "itlb_miss_rate",
    "25": "dtlb_miss_rate",
}

YLABELS = {
    "sim_cycle":        "Execution Time (cycles)",
    "sim_CPI":          "Cycles Per Instruction (CPI)",
    "sim_IPC":          "Instructions Per Cycle (IPC)",
    "sim_inst_rate":    "Simulation Speed (insts/sec)",
    "dl1_miss_rate":    "DL1 Miss Rate",
    "dl1_hit_rate":     "DL1 Hit Rate",
    "il1_miss_rate":    "IL1 Miss Rate",
    "il1_hit_rate":     "IL1 Hit Rate",
    "dl2_miss_rate":    "DL2 Miss Rate",
    "dl1_accesses":     "DL1 Accesses",
    "dl1_hits":         "DL1 Hits",
    "dl1_misses":       "DL1 Misses",
    "bpred_dir_hits":   "Direction-Predicted Hits",
    "bpred_misses":     "Branch Predictor Misses",
    "bpred_addr_hits":  "Address-Predicted Hits (BTB)",
    "bpred_dir_rate":   "Branch Direction Prediction Rate",
    "bpred_miss_rate":  "Branch Miss Rate",
    "ifq_occupancy":    "Avg IFQ Occupancy (insts)",
    "ruu_occupancy":    "Avg RUU Occupancy (insts)",
    "lsq_occupancy":    "Avg LSQ Occupancy (insts)",
    "sim_num_branches": "Number of Branches",
    "itlb_miss_rate":   "ITLB Miss Rate",
    "dtlb_miss_rate":   "DTLB Miss Rate",
}


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def collect_files():
    """Get file paths from command-line args or prompt the user."""
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
        print(f"\nFiles from command line: {paths}")
        return paths

    print("\n" + "="*60)
    print("  SimpleScalar sim-outorder Results Grapher")
    print("="*60)
    print("Enter sim-result file paths one by one.")
    print("Press ENTER on an empty line when done.\n")
    paths = []
    while True:
        p = input(f"  File {len(paths)+1} path (or ENTER to finish): ").strip()
        if not p:
            if paths:
                break
            print("  Please enter at least one file.")
        elif not os.path.isfile(p):
            print(f"  ✗ File not found: {p}")
        else:
            paths.append(p)
    return paths


def collect_labels(paths):
    """Ask user to label each file."""
    print("\nLabel each file (e.g. Small / Medium / Large / Bimod-512 …)")
    labels = []
    for p in paths:
        default = os.path.splitext(os.path.basename(p))[0]
        label = input(f"  Label for '{os.path.basename(p)}' [{default}]: ").strip()
        labels.append(label if label else default)
    return labels


def choose_graphs():
    """Show menu and return chosen graph IDs."""
    print("\n" + "-"*60)
    print("  Available Graphs")
    print("-"*60)
    for key, desc in GRAPH_MENU.items():
        print(f"  [{key:>3}] {desc}")
    print("-"*60)
    print("  Enter graph numbers separated by commas, or 'ALL'")
    choice = input("  Your choice: ").strip().upper()

    if choice == "ALL":
        return list(STAT_FOR_GRAPH.keys())

    selected = []
    for part in choice.split(","):
        part = part.strip()
        if part in STAT_FOR_GRAPH:
            selected.append(part)
        else:
            print(f"  ✗ Unknown option ignored: {part}")
    return selected


def output_dir_choice():
    default = "graphs_output"
    d = input(f"\nSave graphs to folder [{default}]: ").strip()
    d = d if d else default
    os.makedirs(d, exist_ok=True)
    return d


def safe_val(stat, key):
    return stat.get(key, None)


def generate_graph(graph_id, labels, all_stats, out_dir, benchmark_name=""):
    """Generate a single graph given its ID."""
    spec = STAT_FOR_GRAPH[graph_id]
    title_suffix = f" — {benchmark_name}" if benchmark_name else ""

    # ── XY line graphs (graph 21, 22) ──
    if isinstance(spec, tuple):
        x_key, y_key = spec
        x_vals = [safe_val(s, x_key) for s in all_stats]
        y_vals = [safe_val(s, y_key) for s in all_stats]

        # Filter out None
        pairs = [(x, y, lbl) for x, y, lbl in zip(x_vals, y_vals, labels)
                 if x is not None and y is not None]
        if not pairs:
            print(f"  ⚠  Graph {graph_id}: no data found for '{x_key}' or '{y_key}' — skipped.")
            return

        pairs.sort(key=lambda t: t[0])
        xs, ys, lbls = zip(*pairs)
        xlabel_map = {
            "fetch_ifqsize": "IFQ Size (insts)",
            "issue_width":   "Issue Width (insts/cycle)",
            "decode_width":  "Decode Width",
        }
        ylabel_map = {
            "sim_IPC":   "IPC (Instructions Per Cycle)",
            "sim_cycle": "Execution Time (cycles)",
        }
        title_map = {
            ("fetch_ifqsize", "sim_IPC"):  "IFQ Size vs IPC",
            ("issue_width",   "sim_cycle"): "Issue Width vs Total Cycles",
        }

        title = title_map.get(spec, f"{x_key} vs {y_key}") + title_suffix
        outfile = os.path.join(out_dir, f"graph_{graph_id}_{x_key}_vs_{y_key}.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys, marker="o", color=COLORS[0], linewidth=2, markersize=8)
        for xi, yi, li in zip(xs, ys, lbls):
            ax.annotate(li, (xi, yi), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        style_ax(ax, title,
                 xlabel_map.get(x_key, x_key),
                 ylabel_map.get(y_key, y_key), legend=False)
        plt.tight_layout()
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓  Saved: {outfile}")
        return

    # ── Standard bar / grouped bar charts ──
    stat_key = spec
    values = [safe_val(s, stat_key) for s in all_stats]
    available = [(lbl, val) for lbl, val in zip(labels, values) if val is not None]

    if not available:
        print(f"  ⚠  Graph {graph_id}: stat '{stat_key}' not found in any file — skipped.")
        return

    lbls, vals = zip(*available)
    title = GRAPH_MENU[graph_id].split("│")[-1].strip() + title_suffix
    ylabel = YLABELS.get(stat_key, stat_key)
    outfile = os.path.join(out_dir, f"graph_{graph_id}_{stat_key}.png")

    bar_chart(list(lbls), list(vals), title,
              "Configuration", ylabel, outfile,
              color=COLORS[int(graph_id) % len(COLORS)])


def summary_table(labels, all_stats):
    """Print a quick summary table to console."""
    key_stats = ["sim_cycle", "sim_CPI", "sim_IPC", "sim_inst_rate",
                 "dl1_miss_rate", "dl1_hit_rate", "bpred_dir_rate", "bpred_misses"]
    col_w = max(len(l) for l in labels) + 2
    print("\n" + "="*70)
    print("  SUMMARY TABLE")
    print("="*70)
    header = f"{'Stat':<22}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-"*70)
    for k in key_stats:
        row_vals = [s.get(k) for s in all_stats]
        if all(v is None for v in row_vals):
            continue
        row = f"{k:<22}"
        for v in row_vals:
            if v is None:
                row += f"{'N/A':>{col_w}}"
            elif v > 1e6:
                row += f"{v:>{col_w-1},.0f} "
            elif v < 0.01:
                row += f"{v:>{col_w}.6f}"
            else:
                row += f"{v:>{col_w}.4f}"
        print(row)
    print("="*70)


def main():
    paths = collect_files()

    print("\nParsing files…")
    all_stats = []
    for p in paths:
        s = parse_file(p)
        all_stats.append(s)
        found = sum(1 for v in s.values() if v is not None)
        print(f"  ✓  {os.path.basename(p)}: {found} stats extracted")

    labels = collect_labels(paths)
    summary_table(labels, all_stats)

    benchmark_name = input("\nBenchmark name for graph titles (e.g. ijpeg, vortex) [optional]: ").strip()

    chosen = choose_graphs()
    if not chosen:
        print("No valid graphs selected. Exiting.")
        return

    out_dir = output_dir_choice()

    print(f"\nGenerating {len(chosen)} graph(s) into '{out_dir}/' …\n")
    for gid in chosen:
        generate_graph(gid, labels, all_stats, out_dir, benchmark_name)

    print(f"\n✅  Done! All graphs saved in '{out_dir}/'")
    print("    Open that folder to view your PNG files.\n")


# ──────────────────────────────────────────────────────────────────────────────
#  BATCH / PROGRAMMATIC API  (import and call directly from another script)
# ──────────────────────────────────────────────────────────────────────────────

def batch_generate(file_label_pairs, graph_ids="ALL", out_dir="graphs_output",
                   benchmark_name=""):
    """
    Non-interactive entry point.

    Parameters
    ----------
    file_label_pairs : list of (filepath, label) tuples
        e.g. [("simresults-small1.txt", "Small"),
               ("simresults-large1.txt", "Large")]
    graph_ids : "ALL" or list of str
        Which graphs to generate (see GRAPH_MENU keys).
    out_dir : str
        Output directory for PNGs.
    benchmark_name : str
        Appended to graph titles.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths  = [p for p, _ in file_label_pairs]
    labels = [l for _, l in file_label_pairs]
    all_stats = [parse_file(p) for p in paths]

    if graph_ids == "ALL":
        graph_ids = list(STAT_FOR_GRAPH.keys())

    for gid in graph_ids:
        generate_graph(gid, labels, all_stats, out_dir, benchmark_name)

    print(f"\n✅  Batch done! Graphs in '{out_dir}/'")
    return out_dir


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()