#!/usr/bin/env python3
"""
Visualize per-layer activation stats logged by gemma.c's warn_stats().

Usage:
    python plot_debug_log.py fp16_debug.csv
    python plot_debug_log.py fp16_debug.csv int8_debug.csv    # side-by-side comparison
    python plot_debug_log.py fp16_debug.csv --pos 0           # pick which token position to inspect
    python plot_debug_log.py fp16_debug.csv --out report.png  # save instead of showing interactively

The CSV format expected (written by the patched warn_stats in gemma.c):
    call_idx,name,layer,pos,min,max,mean,std,inf_cnt,nan_cnt

For each distinct stage name (e.g. "norm1", "att_out", "resid1", ...), this
draws mean +/- std as a shaded band across layers, with min/max as dashed
lines, so you can see at a glance where values start to blow up or diverge.
When two CSV files are given (e.g. one from an fp16 run, one from an int8
run of the same prompt), both are overlaid on the same axes per stage, so
you can spot exactly which layer/stage is the first to diverge between the
two runs -- this is generally the fastest way to localize a numerical bug
when one code path is known-good and the other isn't.
"""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd


def load_log(path):
    df = pd.read_csv(path)
    required = {"call_idx", "name", "layer", "pos", "min", "max", "mean", "std",
                "inf_cnt", "nan_cnt"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"error: {path} is missing expected columns: {missing}")
    return df


def pick_pos(df, requested):
    available = sorted(df["pos"].unique())
    if requested is not None:
        if requested not in available:
            sys.exit(f"error: pos={requested} not found in log. "
                      f"Available positions: {available[:10]}"
                      f"{' ...' if len(available) > 10 else ''}")
        return requested
    # Default to the smallest pos present (first token processed), which
    # gives one clean pass through every layer of the network.
    return available[0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="+", help="one or two debug CSV files")
    ap.add_argument("--pos", type=int, default=None,
                     help="which token position to inspect (default: smallest pos in the file)")
    ap.add_argument("--out", type=str, default=None,
                     help="save figure to this path instead of showing it interactively")
    args = ap.parse_args()

    if len(args.logs) > 2:
        sys.exit("error: pass at most two log files (for A/B comparison)")

    dfs = [load_log(p) for p in args.logs]
    labels = [p.rsplit("/", 1)[-1] for p in args.logs]

    pos = pick_pos(dfs[0], args.pos)
    dfs = [df[df["pos"] == pos].sort_values(["name", "layer"]) for df in dfs]

    # Preserve first-seen order of stage names so subplots follow the order
    # they actually occur in the forward pass, not alphabetical order.
    stage_order = list(dict.fromkeys(dfs[0]["name"]))
    n = len(stage_order)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows),
                              squeeze=False)
    colors = ["tab:blue", "tab:orange"]

    any_nan_or_inf = False

    for idx, stage in enumerate(stage_order):
        ax = axes[idx // ncols][idx % ncols]

        for df, label, color in zip(dfs, labels, colors):
            sub = df[df["name"] == stage]
            if sub.empty:
                continue

            ax.plot(sub["layer"], sub["mean"], color=color, label=f"{label} mean")
            ax.fill_between(sub["layer"], sub["mean"] - sub["std"],
                             sub["mean"] + sub["std"], color=color, alpha=0.2)
            ax.plot(sub["layer"], sub["min"], color=color, linestyle="--",
                     linewidth=0.8, alpha=0.6)
            ax.plot(sub["layer"], sub["max"], color=color, linestyle="--",
                     linewidth=0.8, alpha=0.6)

            bad = sub[(sub["inf_cnt"] > 0) | (sub["nan_cnt"] > 0)]
            if not bad.empty:
                any_nan_or_inf = True
                ax.scatter(bad["layer"], bad["mean"], color="red", zorder=5,
                           marker="x", s=60,
                           label=f"{label}: inf/nan present")

        ax.set_title(stage, fontsize=10)
        ax.set_xlabel("layer")
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    # Hide unused subplot slots if stage count isn't a multiple of ncols
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    title = f"Activation stats at pos={pos}"
    if len(dfs) == 2:
        title += f"  ({labels[0]} vs {labels[1]})"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if any_nan_or_inf:
        print("Warning: inf/nan detected in at least one stage -- marked with red X on the plot.")

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
