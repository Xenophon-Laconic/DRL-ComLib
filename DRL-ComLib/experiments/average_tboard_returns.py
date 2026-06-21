import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
from tbparse import SummaryReader  # pip install tbparse
import matplotlib.pyplot as plt
import math


"""
run with:

uv run average_tboard_returns.py \
  --logdir-base runs \
  --pattern "CartPole-v1__exp1_4actors_seed" \
  --tag "charts/episodic_return" \
  --output-csv averages/exp1_4actors_episode_return.csv \
  --output-png figures/exp1_4actors_episode_return.png \
  --title "Experiment 1: 4 actors" \
  --ylabel "Episode return" \
  --smooth-weight 0.99

"""

def tensorboard_smooth(values, weight=0.99):
    """
    TensorBoard-style exponential smoothing (approximate).
    `weight` in [0, 1): larger -> smoother (e.g. 0.99).
    """
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return vals

    last = vals[0]
    smoothed = [last]
    for x in vals[1:]:
        last = last * weight + (1.0 - weight) * x
        smoothed.append(last)
    return np.array(smoothed)


def load_runs(run_dirs, tag):
    """Load scalar data for a given tag from multiple TensorBoard runs."""
    dfs = []
    for d in run_dirs:
        d = Path(d)
        if not d.exists():
            print(f"Warning: {d} does not exist, skipping.")
            continue

        print(f"Loading {d}...")
        reader = SummaryReader(str(d))
        df = reader.scalars  # columns: wall_time, step, tag, value, dir_name
        df = df[df["tag"] == tag].copy()
        if df.empty:
            print(f"  No data for tag '{tag}' in {d}, skipping.")
            continue

        df = df[["step", "value"]].copy()
        df["run"] = d.name
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid runs found for the given tag.")

    return dfs


def align_and_average(dfs):
    """
    Align runs on 'step' and compute mean/std across runs.
    Uses inner join on steps that exist in all runs.
    """
    merged = None
    for i, df in enumerate(dfs):
        df = df.rename(columns={"value": f"value_run_{i}"})
        if merged is None:
            merged = df[["step", f"value_run_{i}"]]
        else:
            merged = pd.merge(
                merged,
                df[["step", f"value_run_{i}"]],
                on="step",
                how="inner",
            )

    value_cols = [c for c in merged.columns if c.startswith("value_run_")]
    merged["mean"] = merged[value_cols].mean(axis=1)
    merged["std"] = merged[value_cols].std(axis=1)
    return merged


import matplotlib.pyplot as plt
import matplotlib as mpl

# Global style config (put near the top of the file, once)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
})

def plot_mean_with_std(
    df,
    title,
    ylabel,
    output_png,
    smooth_weight=0.99,
    std_scale=1.0,
):
    """
    Plot smoothed mean ± smoothed (scaled) std band over steps and save to PNG.
    df must have columns: step, mean, std.
    """
    x = df["step"].values
    y = df["mean"].values
    y_std = df["std"].values

    # Apply TensorBoard-like smoothing
    y_smooth = tensorboard_smooth(y, weight=smooth_weight)
    std_smooth = tensorboard_smooth(y_std, weight=smooth_weight)

    lower = y_smooth - std_scale * std_smooth
    upper = y_smooth + std_scale * std_smooth

    # Colors: muted blue for mean, lighter blue for band
    line_color = "#1f77b4"      # classic matplotlib blue
    band_color = "#1f77b4"

    plt.figure(figsize=(6.5, 4.0))

    plt.plot(
        x,
        y_smooth,
        label="Mean episode return",
        color=line_color,
        linewidth=2.0,
    )
    plt.fill_between(
        x,
        lower,
        upper,
        color=band_color,
        alpha=0.18,
        label=f"±{std_scale:.1f} std dev",
    )

    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(title)

    # Light, unobtrusive grid
    plt.grid(True, which="both", axis="both", alpha=0.3, linestyle="--", linewidth=0.7)

    # Legend with subtle frame
    legend = plt.legend(frameon=True, loc="upper left")
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor("0.8")

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir-base",
        type=str,
        required=True,
        help="Base directory containing the run subdirectories",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Substring to select run directories, e.g. 'exp1_1actor_seed'",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="train/episode_return",
        help="TensorBoard scalar tag to average (default: train/episode_return)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to write the averaged CSV",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        required=True,
        help="Path to write the plot PNG",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Episode return vs environment steps",
        help="Plot title",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default="Episode return",
        help="Y-axis label",
    )
    parser.add_argument(
        "--smooth-weight",
        type=float,
        default=0.99,
        help="EMA smoothing weight in [0,1) (default: 0.99)",
    )
    args = parser.parse_args()

    base = Path(args.logdir_base)
    if not base.exists():
        raise RuntimeError(f"Base logdir '{base}' does not exist")

    run_dirs = sorted(
        d for d in base.iterdir()
        if d.is_dir() and args.pattern in d.name
    )

    if not run_dirs:
        raise RuntimeError(
            f"No run directories matching pattern '{args.pattern}' under {base}"
        )

    print("Found runs:")
    for d in run_dirs:
        print(" ", d)

    dfs = load_runs(run_dirs, args.tag)
    merged = align_and_average(dfs)

    # Save CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    print(f"Saved averaged data to {args.output_csv}")

    # Plot
    plot_mean_with_std(
        df=merged,
        title=args.title,
        ylabel=args.ylabel,
        output_png=args.output_png,
        smooth_weight=args.smooth_weight,
    )


if __name__ == "__main__":
    main()