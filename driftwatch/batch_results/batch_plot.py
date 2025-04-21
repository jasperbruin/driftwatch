"""usage:
python batch_plot.py --results_path /Users/jasper.bruin/Documents/driftwatch/driftwatch/data/2025-04-18_12-02-01/results.json \
                                      --out_dir
"""

import argparse
import json
import os
import pathlib
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _flatten_records(obj: dict) -> List[dict]:
    """Flatten the nested {dataset}->{model}->[records] structure."""
    rows = []
    for dataset, model_map in obj.items():
        for model, records in model_map.items():
            for rec in records:
                rec = rec.copy()
                rec["dataset"] = dataset
                rec["model"] = model
                rows.append(rec)
    return rows


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Create the efficiency summary table used in the paper."""
    return (
        df
        # collapse seeds / datasets / distance names – we want *overall* view
        .groupby("method", as_index=False)
        .agg(
            mean_final_similarity=("final_similarity", "mean"),
            mean_avg_overhead=("avg_overhead", "mean"),
            mean_time_taken=("time_taken", "mean"),
            mean_memory=("avg_memory_mb", "mean"),
        )
        .assign(
            speedup_vs_full=lambda d: d.loc[
                d.method == "no_pca", "mean_time_taken"
            ].iloc[0]
            / d["mean_time_taken"],
            memory_reduction=lambda d: d.loc[d.method == "no_pca", "mean_memory"].iloc[
                0
            ]
            / d["mean_memory"],
        )
        .rename(
            columns={
                "method": "Method",
                "mean_final_similarity": "DetectionAccuracy",  # alias for clarity
                "mean_avg_overhead": "AvgOverhead_s",
                "mean_time_taken": "TotalRuntime_s",
                "mean_memory": "AvgMem_MB",
                "mean_peak_memory": "PeakMem_MB",
            }
        )
        .sort_values("AvgMem_MB")
    )


def _plot_tradeoffs(df, output_dir):
    """Plots tradeoffs between two metrics (e.g., accuracy vs. memory usage) for different methods."""
    # Define figure size and layout
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define color palette
    palette = sns.color_palette("Set2", len(df["method"].unique()))

    # Plot each method
    for i, method in enumerate(df["method"].unique()):
        subset = df[df["method"] == method]
        ax.plot(
            subset["metric_x"],  # Replace with the x-axis metric
            subset["metric_y"],  # Replace with the y-axis metric
            marker="o",
            label=method,
            color=palette[i % len(palette)],
        )

    # Set title, labels, and grid
    ax.set_title(
        "Tradeoff Between Metric X and Metric Y", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Metric X", fontsize=14)
    ax.set_ylabel("Metric Y", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Style ticks
    ax.tick_params(axis="both", labelsize=12)

    # Add legend
    ax.legend(title="Method", fontsize=12, title_fontsize=13, loc="best")

    # Save and show the plot
    path_out = os.path.join(output_dir, "plot_tradeoffs.png")
    plt.tight_layout()
    plt.savefig(path_out, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"[Saved] {path_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_path",
        required=True,
        help="Path to results.json produced by the experiment script",
    )
    ap.add_argument(
        "--out_dir", default="figs", help="Directory to store CSV and PNG outputs"
    )
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results_path) as f:
        raw = json.load(f)
    rows = _flatten_records(raw)
    df = pd.DataFrame(rows)

    summary = _summarise(df)
    csv_path = out_dir / "compression_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[✓] Summary table written to {csv_path}")

    png_path = out_dir / "compression_tradeoffs.png"
    _plot_tradeoffs(summary, png_path)
    print(f"[✓] Figure written to {png_path}")


if __name__ == "__main__":
    main()
