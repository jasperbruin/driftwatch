import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define metric categorization
VECTOR_METRICS = [
    "cosine",
    "manhattan",
    "minkowski",
    "mahalanobis",
    "chebyshev",
    "canberra",
]
DISTRIBUTION_METRICS = [
    "wasserstein",
    "ks",
    "kl",
    "js",
    "hellinger",
    "bhattacharyya",
    "mmd",
]
DIST_IMPLEMENTATIONS = ["kll", "histogram"]


def plot_memory_usage(output_dir="analysis_results"):
    """Create visualizations of memory usage across different metrics."""
    # Find all memory metrics files
    memory_files = glob(os.path.join(output_dir, "memory_metrics_*.csv"))

    if not memory_files:
        print(f"No memory metrics files found in {output_dir}.")
        return

    # Load and combine all memory metrics
    dfs = []
    for file in memory_files:
        df = pd.read_csv(file)
        dfs.append(df)

    memory_data = pd.concat(dfs, ignore_index=True)

    # Sort metrics by peak memory usage
    memory_data = memory_data.sort_values(by="peak_memory_mb", ascending=False)

    # Define which metrics are vector-based vs distribution-based
    vector_metrics = [
        "cosine",
        "euclidean",
        "manhattan",
        "minkowski",
        "mahalanobis",
        "chebyshev",
        "canberra",
    ]
    distribution_metrics = [
        "wasserstein",
        "ks",
        "kl",
        "js",
        "hellinger",
        "bhattacharyya",
        "mmd",
    ]

    # Classify each metric
    memory_data["metric_type"] = memory_data["metric"].apply(
        lambda x: "Vector-based"
        if any(vm.lower() in x.lower() for vm in vector_metrics)
        else (
            "Distribution-based"
            if any(dm.lower() in x.lower() for dm in distribution_metrics)
            else "Other"
        )
    )

    # Plot 1: Memory usage by metric (peak)
    plt.figure(figsize=(14, 8))
    bars = plt.bar(memory_data["metric"], memory_data["peak_memory_mb"])

    # Color bars by metric type
    colors = {
        "Vector-based": "skyblue",
        "Distribution-based": "salmon",
        "Other": "lightgray",
    }
    for i, bar in enumerate(bars):
        bar.set_color(colors[memory_data.iloc[i]["metric_type"]])

    plt.title("Peak Memory Usage by Distance Metric", fontsize=14)
    plt.xlabel("Distance Metric", fontsize=12)
    plt.ylabel("Peak Memory Usage (MB)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)

    # Add legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[t]) for t in colors]
    plt.legend(handles, colors.keys(), title="Metric Type")

    plt.tight_layout()
    output_image = os.path.join(output_dir, "memory_usage_peak.png")
    plt.savefig(output_image)
    plt.close()
    print(f"Saved memory peak usage plot to {output_image}")

    # Plot 2: Memory progression (initialization -> baseline -> final)
    plt.figure(figsize=(14, 10))

    # Group by metric type and calculate means
    grouped = memory_data.groupby("metric_type")

    x = np.arange(3)  # init, baseline, final
    width = 0.25  # width of bars

    # Plot bars for each metric type
    i = 0
    for name, group in grouped:
        means = [
            group["init_memory_mb"].mean(),
            group["baseline_memory_mb"].mean(),
            group["final_memory_mb"].mean(),
        ]

        plt.bar(x + i * width, means, width, label=name, color=colors[name])
        i += 1

    plt.xlabel("Stage", fontsize=12)
    plt.ylabel("Memory Usage (MB)", fontsize=12)
    plt.title("Memory Usage Progression by Metric Type", fontsize=14)
    plt.xticks(x + width, ["Initialization", "Baseline", "Final"])
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_image = os.path.join(output_dir, "memory_progression.png")
    plt.savefig(output_image)
    plt.close()
    print(f"Saved memory progression plot to {output_image}")

    # Plot 3: Memory usage increase from init to final
    memory_data["memory_increase"] = (
        memory_data["final_memory_mb"] - memory_data["init_memory_mb"]
    )

    plt.figure(figsize=(14, 8))
    bars = plt.bar(memory_data["metric"], memory_data["memory_increase"])

    # Color bars by metric type
    for i, bar in enumerate(bars):
        bar.set_color(colors[memory_data.iloc[i]["metric_type"]])

    plt.title("Memory Growth During Drift Detection by Metric", fontsize=14)
    plt.xlabel("Distance Metric", fontsize=12)
    plt.ylabel("Memory Increase (MB)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(handles, colors.keys(), title="Metric Type")

    plt.tight_layout()
    output_image = os.path.join(output_dir, "memory_growth.png")
    plt.savefig(output_image)
    plt.close()
    print(f"Saved memory growth plot to {output_image}")

    # Save the combined memory metrics data
    combined_file = os.path.join(output_dir, "combined_memory_metrics.csv")
    memory_data.to_csv(combined_file, index=False)
    print(f"Saved combined memory metrics to {combined_file}")


def plot_drift_results(results_file, output_dir="baseline_results"):
    """Plot drift results from a single CSV file."""
    # Extract metric name from filename
    filename = os.path.basename(results_file)
    distance_metric = filename.replace("drift_detection_results_", "").replace(
        ".csv", ""
    )

    # Load results
    results_df = pd.read_csv(results_file)

    # Convert window_time to datetime
    results_df["window_time"] = pd.to_datetime(results_df["window_time"])

    # Get data for plotting
    window_times = results_df["window_time"]
    all_distances = results_df["drift_distance"]
    all_thresholds = results_df["threshold"]
    all_labels = results_df["drift_detected"]

    # Plot 1: Drift distance and threshold over window index
    plt.figure(figsize=(12, 6))
    plt.plot(all_distances, label="Drift Distance", color="blue")
    plt.plot(all_thresholds, label="Threshold", color="red", linestyle="--")
    plt.title(f"Embedding Drift Distance per Window ({distance_metric})")
    plt.xlabel("Window Index")
    plt.ylabel("Drift Distance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_image = os.path.join(
        output_dir, f"drift_distance_by_index_{distance_metric}.png"
    )
    plt.savefig(output_image)
    plt.close()
    print(f"Saved plot to {output_image}")

    # Plot 2: Drift distance over time with threshold and detected drift points
    plt.figure(figsize=(12, 6))
    plt.plot(window_times, all_distances, label="Drift Score", color="blue")
    plt.plot(
        window_times, all_thresholds, label="Threshold", color="red", linestyle="--"
    )

    # Highlight detected drift windows on the plot
    drift_times = [t for t, lbl in zip(window_times, all_labels) if lbl == 1]
    drift_scores = [d for d, lbl in zip(all_distances, all_labels) if lbl == 1]
    if drift_times:
        plt.scatter(
            drift_times,
            drift_scores,
            color="red",
            marker="o",
            s=50,
            label="Detected Drift",
        )

    # Optionally highlight a known drift period (if relevant)
    plt.axvspan(
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2023-01-01"),
        color="purple",
        alpha=0.1,
        label="Observed Drift (2021-2023)",
    )

    plt.title(f"Drift Scores Over Time ({distance_metric})", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Drift Score", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_image = os.path.join(
        output_dir, f"drift_score_over_time_{distance_metric}.png"
    )
    plt.savefig(output_image)
    plt.close()
    print(f"Saved plot to {output_image}")

    return {
        "metric": distance_metric,
        "window_times": window_times,
        "distances": all_distances,
    }


def get_metric_category(metric_name):
    """Determine the category and implementation of a metric."""
    metric_lower = metric_name.lower()

    # Check if it's a distribution metric with implementation specified
    for impl in DIST_IMPLEMENTATIONS:
        if f"_{impl}" in metric_lower:
            base_metric = metric_lower.split(f"_{impl}")[0]
            if any(dm.lower() in base_metric for dm in DISTRIBUTION_METRICS):
                return f"Distribution-based ({impl.upper()})"

    # Check if it's a vector-based metric
    if any(vm.lower() in metric_lower for vm in VECTOR_METRICS):
        return "Vector-based"

    # Check if it's a distribution metric without implementation specified (legacy data)
    if any(dm.lower() in metric_lower for dm in DISTRIBUTION_METRICS):
        return "Distribution-based (KLL)"  # Default to KLL for backward compatibility

    # Default category for unknown metrics
    return "Other"


def create_comparison_plot(results_data, output_dir="baseline_results"):
    """Create a comparison plot with separate subplots for different metric categories."""
    # Create figure with subplots for different metric categories
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

    # Group results by metric category
    vector_results = []
    kll_results = []
    histogram_results = []

    for result in results_data:
        metric_name = result["metric"].lower()

        # Categorize by implementation type
        if any(vm.lower() in metric_name for vm in VECTOR_METRICS):
            vector_results.append(result)
        elif "_histogram" in metric_name:
            histogram_results.append(result)
        elif any(dm.lower() in metric_name for dm in DISTRIBUTION_METRICS):
            kll_results.append(result)
        else:
            # For unknown metrics, default to vector-based
            vector_results.append(result)

    # Plot vector-based metrics in the first subplot
    for result in vector_results:
        # Normalize the drift scores to range [0, 1]
        distances = np.array(result["distances"])
        min_val = np.min(distances)
        max_val = np.max(distances)

        # Check to avoid division by zero if all values are the same
        if max_val == min_val:
            normalized_distances = np.zeros_like(distances)
        else:
            normalized_distances = (distances - min_val) / (max_val - min_val)

        ax1.plot(
            result["window_times"], normalized_distances, label=f"{result['metric']}"
        )

    # Add observed drift period to the first subplot
    ax1.axvspan(
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2023-01-01"),
        color="purple",
        alpha=0.1,
        label="Observed Drift (2021-2023)",
    )

    #
