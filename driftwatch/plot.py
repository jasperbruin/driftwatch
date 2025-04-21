import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_json_data(json_path):
    with open(json_path) as file:
        return json.load(file)


def flatten_data(results_data):
    records_list = []
    for dataset_name, model_dict in results_data.items():
        for model_name, records in model_dict.items():
            for r in records:
                method = r.get("method", "unknown")
                records_list.append(
                    {
                        "dataset": dataset_name,
                        "model_name": model_name,
                        "distance_name": r["distance_name"],
                        "distance_type": r.get("distance_type", "vector"),
                        "pca_applied": r.get("pca_applied", False),
                        "method": method,
                        "drift_strength": r["drift_strength"],
                        "final_similarity": r["final_similarity"],
                        "avg_overhead": r.get("avg_overhead", float("nan")),
                        "time_taken": r.get("time_taken", float("nan")),
                        "avg_memory_mb": r.get("avg_memory_mb", float("nan")),
                        "peak_memory_mb": r.get(
                            "peak_memory_mb", r.get("avg_memory_mb", float("nan"))
                        ),
                    }
                )
    return pd.DataFrame(records_list)


def plot_final_similarity(df, output_dir):
    """Produce a two-subplot figure:
      - Left subplot: Legacy metrics (mahalanobis + classical distance functions).
      - Right subplot: Distribution-based metrics (kl, js, hellinger, etc.).

    Each subplot shows how the normalized final similarity varies with drift_strength,
    with one line per method (e.g., no_pca, pca, kll_sketch). We first min–max normalize
    final_similarity per distance_name, then average across all distance_names in each
    category (legacy or distribution-based). This lets us compare trends between the
    two categories at a high level.
    """
    # 1) Min–max normalize final_similarity for each distance_name
    df_sim = df.copy()
    df_sim = df_sim.groupby(
        ["distance_name", "drift_strength", "method"], as_index=False
    )["final_similarity"].mean()

    df_sim["final_similarity_norm"] = 0.0
    for dist_name, group_data in df_sim.groupby("distance_name"):
        min_val = group_data["final_similarity"].min()
        max_val = group_data["final_similarity"].max()
        if max_val - min_val < 1e-12:
            # Handle edge case where all values are identical
            df_sim.loc[group_data.index, "final_similarity_norm"] = 0.0
        else:
            df_sim.loc[group_data.index, "final_similarity_norm"] = (
                group_data["final_similarity"] - min_val
            ) / (max_val - min_val)

    # 2) Separate the distance metrics into two categories
    legacy_set = {
        "mahalanobis",
        "euclidean",
        "manhattan",
        "minkowski",
        "chebyshev",
        "canberra",
    }
    dist_set = {"kl", "js", "hellinger", "bhattacharyya", "mmd", "wasserstein"}

    df_legacy = df_sim[df_sim["distance_name"].isin(legacy_set)].copy()
    df_dist = df_sim[df_sim["distance_name"].isin(dist_set)].copy()

    # If either subset is empty, handle gracefully
    if df_legacy.empty and df_dist.empty:
        print(
            "[Warning] Neither legacy nor distribution-based metrics found in the DataFrame."
        )
        return

    # 3) For each category, we average across all distance_names
    #    so we get a single trend line per method for that category.
    df_legacy_grouped = df_legacy.groupby(["method", "drift_strength"], as_index=False)[
        "final_similarity_norm"
    ].mean()
    df_dist_grouped = df_dist.groupby(["method", "drift_strength"], as_index=False)[
        "final_similarity_norm"
    ].mean()

    # 4) Plot them side-by-side
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Determine a consistent color palette based on the number of methods
    all_methods = np.unique(df_sim["method"])
    palette = sns.color_palette("husl", n_colors=len(all_methods))
    method_to_color = dict(zip(all_methods, palette))

    # -- Left subplot: Legacy metrics --
    if not df_legacy_grouped.empty:
        for method_name in all_methods:
            subset = df_legacy_grouped[df_legacy_grouped["method"] == method_name]
            if subset.empty:
                continue
            ax_left.plot(
                subset["drift_strength"],
                subset["final_similarity_norm"],
                marker="o",
                label=method_name,
                color=method_to_color[method_name],
            )
        ax_left.set_title("Distance-based Metrics (Averaged)")
        ax_left.set_xlabel("Drift Strength")
        ax_left.set_ylabel("Normalized Final Similarity")
        ax_left.grid(True)
        ax_left.legend()
    else:
        ax_left.set_title("No Legacy Metrics Found")
        ax_left.set_axis_off()

    # -- Right subplot: Distribution-based metrics --
    if not df_dist_grouped.empty:
        for method_name in all_methods:
            subset = df_dist_grouped[df_dist_grouped["method"] == method_name]
            if subset.empty:
                continue
            ax_right.plot(
                subset["drift_strength"],
                subset["final_similarity_norm"],
                marker="o",
                label=method_name,
                color=method_to_color[method_name],
            )
        ax_right.set_title("Distribution-Based Metrics (Averaged)")
        ax_right.set_xlabel("Drift Strength")
        ax_right.set_ylabel("Normalized Final Similarity")
        ax_right.grid(True)
        ax_right.legend()
    else:
        ax_right.set_title("No Distribution-Based Metrics Found")
        ax_right.set_axis_off()

    plt.tight_layout()

    # 5) Save the figure
    out_path = os.path.join(output_dir, "final_similarity_legacy_vs_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Saved] {out_path}")


def plot_relative_log_increase(df, output_dir):
    plot_data = []

    # Step 1: Compute relative log increase per row
    for _, row in df.iterrows():
        if row["drift_strength"] >= 0:
            base_similarity = df[
                (df["drift_strength"] == 0)
                & (df["distance_name"] == row["distance_name"])
                & (
                    df["method"] == row["method"]
                )  # Changed to use method instead of pca_applied
            ]["final_similarity"].mean()

            if base_similarity and base_similarity > 0:
                rel_log = np.log(row["final_similarity"] / base_similarity)
                plot_data.append(
                    {
                        "distance_name": row["distance_name"],
                        "drift_strength": row["drift_strength"],
                        "method": row["method"],  # Store method
                        "distance_type": row["distance_type"],
                        "rel_log_increase": rel_log,
                    }
                )

    df_plot = pd.DataFrame(plot_data)

    # Step 2: Normalize per (distance_name, method)
    df_normalized = []
    for (dist_name, method), group in df_plot.groupby(["distance_name", "method"]):
        df_avg = (
            group.groupby("drift_strength")["rel_log_increase"].mean().reset_index()
        )
        min_val = df_avg["rel_log_increase"].min()
        max_val = df_avg["rel_log_increase"].max()
        if max_val - min_val > 1e-12:
            df_avg["normalized"] = (df_avg["rel_log_increase"] - min_val) / (
                max_val - min_val
            )
        else:
            df_avg["normalized"] = 0.0
        df_avg["distance_name"] = dist_name
        df_avg["method"] = method
        df_avg["distance_type"] = group["distance_type"].iloc[0]
        df_normalized.append(df_avg)

    df_final = pd.concat(df_normalized, ignore_index=True)

    # Step 3: Create 2x3 subplot figure (vector methods on top, distribution on bottom)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)

    # Define which methods to show in each subplot position (row, column)
    # Row 0: Vector methods, Row 1: Distribution methods
    method_positions = {
        (0, 0): "no_pca",  # Vector-based normal
        (0, 1): "pca",  # Vector-based with PCA
        (0, 2): "kll_vector",  # Vector-based with KLL sketch
        (1, 0): "histogram",  # Distribution-based normal
        (1, 1): "pca_histogram",  # Distribution-based with PCA
        (1, 2): "kll_sketch",  # Distribution-based with KLL sketch
    }

    # Create titles for each subplot
    title_map = {
        "no_pca": "Vector-Based (Full Embedding Size)",
        "pca": "Vector-Based (PCA)",
        "kll_vector": "Vector-Based (KLL Vector)",
        "histogram": "Distribution-Based (Histogram Full Embedding Size)",
        "pca_histogram": "Distribution-Based (PCA Histogram)",
        "kll_sketch": "Distribution-Based (KLL Sketch)",
    }

    # Plot each subplot
    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            method = method_positions.get((i, j))

            if method:
                sub_df = df_final[df_final["method"] == method]

                for dist_name in sub_df["distance_name"].unique():
                    d = sub_df[sub_df["distance_name"] == dist_name]
                    if not d.empty:
                        ax.plot(
                            d["drift_strength"],
                            d["normalized"],
                            marker="o",
                            label=dist_name,
                        )

                ax.set_title(
                    title_map.get(method, f"Method: {method}"),
                    fontsize=12,
                    fontweight="bold",
                )
                ax.set_xlabel("Drift Strength")
                ax.grid(True)

                if j == 0:  # Left column gets y-axis label
                    ax.set_ylabel("Normalized Relative Increase")

                # Add legend to the rightmost plot in each row
                if j == 2:
                    ax.legend(title="Distance Metric", fontsize=9)

    # Add row labels
    fig.text(
        0.02,
        0.75,
        "Vector-Based Methods",
        fontsize=14,
        fontweight="bold",
        rotation=90,
        va="center",
    )
    fig.text(
        0.02,
        0.25,
        "Distribution-Based Methods",
        fontsize=14,
        fontweight="bold",
        rotation=90,
        va="center",
    )

    fig.suptitle(
        "Relative Increase vs Drift Strength by Method", fontsize=16, fontweight="bold"
    )
    plt.tight_layout(rect=[0.04, 0, 1, 0.96])

    path_out = os.path.join(output_dir, "plot_relative_log_increase_6subplots.png")
    plt.savefig(path_out, dpi=300)
    plt.close()
    print(f"[Saved] {path_out}")


def plot_avg_time(df, output_dir):
    df_time = df.groupby(["model_name", "method"], as_index=False)["time_taken"].mean()
    pivot_time = df_time.pivot(
        index="model_name", columns="method", values="time_taken"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("pastel", n_colors=len(pivot_time.columns))
    pivot_time.plot(kind="bar", logy=True, ax=ax, color=colors)

    ax.set_xlabel("Model Name", fontsize=12)
    ax.set_ylabel("Time Taken (s) [log scale]", fontsize=12)
    ax.set_title("Average Time Taken per Model", fontsize=14, fontweight="bold")


def plot_time_vs_similarity(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="time_taken",
        y="final_similarity",
        hue="method",
        style="distance_name",
    )
    plt.title("Scatter Plot of Time Taken vs. Final Similarity")
    plt.xlabel("Time Taken (s)")
    plt.ylabel("Final Similarity")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    path_scatter = os.path.join(output_dir, "scatter_time_vs_similarity.png")
    plt.savefig(path_scatter)
    plt.close()
    print(f"[Saved] {path_scatter}")


def plot_overhead_vs_size(df, output_dir):
    model_sizes = {
        "facebook/opt-125m": 125,
        "bigscience/bloomz-560m": 560,
        "layonsan/google-t5-small": 60,
        "openai-community/gpt2": 117,
        "distilbert-base-uncased": 66,
        "google/mobilebert-uncased": 25,
    }

    avg_overhead_per_model = {}
    for model, size in model_sizes.items():
        overheads = []
        if model in df["model_name"].unique():
            for method in df["method"].unique():
                avg_overhead = df[
                    (df["model_name"] == model) & (df["method"] == method)
                ]["avg_overhead"].mean()
                overheads.append(avg_overhead)
        avg_overhead_per_model[model] = np.nanmean(
            overheads
        )  # Handle NaN values safely

    sizes = [model_sizes[m] for m in avg_overhead_per_model.keys()]
    overheads = [avg_overhead_per_model[m] for m in avg_overhead_per_model.keys()]

    # Use seaborn for improved aesthetics
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        x=sizes,
        y=overheads,
        hue=list(avg_overhead_per_model.keys()),
        palette="tab10",
        s=100,
        edgecolor="black",
    )

    plt.xlabel("Model Size (Million Parameters)", fontsize=12)
    plt.ylabel("Average Overhead", fontsize=12)
    plt.title("Overhead Footprint vs Model Size", fontsize=14)

    # Annotate points with model names
    for model, x, y in zip(avg_overhead_per_model.keys(), sizes, overheads):
        plt.text(
            x,
            y,
            model.split("/")[-1],
            fontsize=10,
            ha="right",
            va="bottom",
            fontweight="bold",
        )

    # Save the plot
    path_overhead_size = os.path.join(output_dir, "scatter_overhead_vs_size.png")
    plt.savefig(path_overhead_size, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path_overhead_size}")


def plot_final_similarity_separate(df, output_dir):
    """Creates 12 subplots (3x4 grid), one per distance_name.
    Each plot shows three lines: PCA, No PCA, and KLL (Vector or Distribution).
    """
    # Step 1: Compute mean final_similarity for each group
    df_sim = df.groupby(["distance_name", "drift_strength", "method"], as_index=False)[
        "final_similarity"
    ].mean()

    # Step 2: Normalize final similarity within each distance metric
    df_sim["final_similarity_norm"] = 0.0
    for dist_name, group_data in df_sim.groupby("distance_name"):
        min_val = group_data["final_similarity"].min()
        max_val = group_data["final_similarity"].max()
        if max_val - min_val == 0:
            df_sim.loc[group_data.index, "final_similarity_norm"] = 0.0
        else:
            df_sim.loc[group_data.index, "final_similarity_norm"] = (
                group_data["final_similarity"] - min_val
            ) / (max_val - min_val)

    # Step 3: Prepare subplots
    unique_distances = sorted(df_sim["distance_name"].unique())
    num_dists = len(unique_distances)
    num_cols = 4
    num_rows = int(np.ceil(num_dists / num_cols))

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False
    )
    palette = sns.color_palette("Set2", 3)  # 3 colors: PCA, No PCA, KLL

    # Define method labels
    method_labels = {
        "pca": "Vector-Based (PCA)",
        "no_pca": "Vector-Based (Full Embedding Size)",
        "kll_vector": "Vector-Based (KLL Vector)",
        "kll_sketch": "Distribution-Based (KLL Sketch)",
    }

    # Step 4: Plot each distance
    for idx, dist_name in enumerate(unique_distances):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row][col]

        for i, method in enumerate(
            ["pca", "no_pca", "kll_vector"]
        ):  # Adjust to "kll_sketch" if needed
            subset = df_sim[
                (df_sim["distance_name"] == dist_name) & (df_sim["method"] == method)
            ]
            if not subset.empty:
                ax.plot(
                    subset["drift_strength"],
                    subset["final_similarity_norm"],
                    marker="o",
                    label=method_labels[method],
                    color=palette[i],
                )

        ax.set_title(dist_name)
        ax.set_xlabel("Drift Strength")
        ax.set_ylabel("Normalized Final Similarity")
        ax.grid(True)
        ax.legend()

    # Step 5: Remove unused axes
    for i in range(num_dists, num_rows * num_cols):
        fig.delaxes(axes[i // num_cols][i % num_cols])

    plt.tight_layout()
    path_sim = os.path.join(
        output_dir, "plot_final_similarity_pca_vs_no_pca_kll_by_distance.png"
    )
    plt.savefig(path_sim)
    plt.close()
    print(f"[Saved] {path_sim}")


def plot_avg_overhead_merged(df, output_dir):
    """Plots a single histogram showing the average overhead per method
    across all models (merged), sorted from high to low.
    """
    if "method" not in df.columns or "avg_overhead" not in df.columns:
        raise ValueError("DataFrame must include 'method' and 'avg_overhead' columns.")

    # Step 1: Compute average overhead per method across all models
    method_means = df.groupby("method", as_index=False)["avg_overhead"].mean()
    method_means_sorted = method_means.sort_values("avg_overhead", ascending=False)

    # Step 2: Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=method_means_sorted, x="method", y="avg_overhead", palette="viridis"
    )

    plt.yscale("log")
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Avg Overhead (s) [log scale]", fontsize=12)
    plt.title(
        "Average Overhead per Method (All Models Combined)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Step 3: Save
    out_path = os.path.join(output_dir, "plot_avg_overhead_merged.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


def plot_avg_memory_merged(df, output_dir):
    """Creates a single figure with two subplots sharing the same y-axis:
      - Left subplot: distribution-based methods
      - Right subplot: vector-based methods

    By using sharey=True, both subplots have the same vertical scale.
    """
    # Validate required columns
    required_cols = {"method", "avg_memory_mb", "distance_type"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must include columns: {required_cols}")

    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    # Define which methods are distribution-based vs. vector-based
    vector_methods = ["pca", "no_pca"]
    distribution_methods = [
        "histogram",
        "kll_sketch",
        "pca_histogram",
        "pca_kll_sketch",
    ]

    # Correct distance_type to override if necessary
    df["corrected_distance_type"] = df["distance_type"]
    df.loc[df["method"].isin(vector_methods), "corrected_distance_type"] = "vector"
    df.loc[df["method"].isin(distribution_methods), "corrected_distance_type"] = (
        "distribution"
    )

    # Create shorter labels for the plot
    label_map = {
        "pca_histogram": "Hist (PCA)",
        "pca_kll_sketch": "KLL (PCA)",
        "histogram": "Hist",
        "kll_sketch": "KLL",
        "pca": "Vector (PCA)",
        "no_pca": "Vector (No PCA)",
    }
    df["method_short"] = df["method"].map(label_map).fillna(df["method"])

    # Calculate mean memory usage per method
    method_means = df.groupby(
        ["method_short", "corrected_distance_type"], as_index=False
    )["avg_memory_mb"].mean()

    # Separate distribution-based and vector-based
    dist_df = method_means[
        method_means["corrected_distance_type"] == "distribution"
    ].copy()
    vec_df = method_means[method_means["corrected_distance_type"] == "vector"].copy()

    # Sort by descending memory usage for clarity
    dist_df.sort_values(by="avg_memory_mb", ascending=False, inplace=True)
    vec_df.sort_values(by="avg_memory_mb", ascending=False, inplace=True)

    # Create figure with two subplots that share the same y-axis
    fig, (ax_dist, ax_vec) = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 5), sharey=True
    )

    # --- Distribution-based subplot ---
    if not dist_df.empty:
        x_dist = range(len(dist_df))
        y_dist = dist_df["avg_memory_mb"].values
        labels_dist = dist_df["method_short"].values

        ax_dist.bar(x_dist, y_dist, color="C0")
        ax_dist.set_title("Distribution-based Methods", fontsize=12, fontweight="bold")
        ax_dist.set_xlabel("Method")
        ax_dist.set_ylabel("Avg Memory (MB)")
        ax_dist.set_xticks(x_dist)
        ax_dist.set_xticklabels(labels_dist, rotation=30, ha="right")

        # Annotate each bar with its value
        for i, val in enumerate(y_dist):
            ax_dist.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # --- Vector-based subplot ---
    if not vec_df.empty:
        x_vec = range(len(vec_df))
        y_vec = vec_df["avg_memory_mb"].values
        labels_vec = vec_df["method_short"].values

        ax_vec.bar(x_vec, y_vec, color="C1")
        ax_vec.set_title("Vector-based Methods", fontsize=12, fontweight="bold")
        ax_vec.set_xlabel("Method")
        # We do NOT set ax_vec.set_ylabel here since sharey=True
        ax_vec.set_xticks(x_vec)
        ax_vec.set_xticklabels(labels_vec, rotation=30, ha="right")

        # Annotate each bar with its value
        for i, val in enumerate(y_vec):
            ax_vec.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # Add a main title
    fig.suptitle("Average Memory Overhead by Method", fontsize=14, fontweight="bold")

    # Adjust spacing
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save to output
    out_path = os.path.join(output_dir, "plot_avg_memory_single_figure_subplots.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_peak_vs_avg_memory(df, output_dir):
    """Creates a scatter plot comparing peak vs average memory usage
    for different methods to visualize memory stability.
    """
    # Check if required columns exist
    if "method" not in df.columns or "avg_memory_mb" not in df.columns:
        print("Warning: Required columns for peak_vs_avg memory plot not found")
        return

    # Use peak_memory_mb if available, otherwise fall back to avg_memory_mb
    if "peak_memory_mb" not in df.columns:
        print("Warning: peak_memory_mb not found, skipping peak vs avg memory plot")
        return

    # Get mean values per method
    method_metrics = (
        df.groupby("method")
        .agg({"avg_memory_mb": "mean", "peak_memory_mb": "mean"})
        .reset_index()
    )

    # Calculate ratio of peak to average memory
    method_metrics["memory_ratio"] = (
        method_metrics["peak_memory_mb"] / method_metrics["avg_memory_mb"]
    )

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Main scatter plot
    scatter = plt.scatter(
        method_metrics["avg_memory_mb"],
        method_metrics["peak_memory_mb"],
        s=100,  # marker size
        c=method_metrics["memory_ratio"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
    )

    # Add a diagonal line representing peak=avg
    max_val = max(
        method_metrics["avg_memory_mb"].max(), method_metrics["peak_memory_mb"].max()
    )
    plt.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Peak = Average")

    # Add method labels
    for i, row in method_metrics.iterrows():
        plt.annotate(
            row["method"],
            (row["avg_memory_mb"], row["peak_memory_mb"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    # Add colorbar to show ratio
    cbar = plt.colorbar(scatter)
    cbar.set_label("Peak/Average Memory Ratio")

    plt.title("Peak vs Average Memory Usage by Method", fontsize=14, fontweight="bold")
    plt.xlabel("Average Memory (MB)", fontsize=12)
    plt.ylabel("Peak Memory (MB)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "plot_peak_vs_avg_memory.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


def generate_all_plots(json_path, output_dir):
    results_data = load_json_data(json_path)
    df = flatten_data(results_data)
    plot_final_similarity(df, output_dir)
    plot_relative_log_increase(df, output_dir)
    plot_avg_time(df, output_dir)
    plot_overhead_vs_size(df, output_dir)
    plot_final_similarity_separate(df, output_dir)
    plot_avg_overhead_merged(df, output_dir)
    plot_avg_memory_merged(df, output_dir)
    plot_peak_vs_avg_memory(df, output_dir)  # Add the new plot


def run_all_results(data_dir):
    """Finds all 'results.json' files in subdirectories under data_dir,
    and runs 'generate_all_plots' on each one.
    """
    for root, _, files in os.walk(data_dir):
        if "results.json" in files:
            json_path = os.path.join(root, "results.json")
            generate_all_plots(json_path, root)


if __name__ == "__main__":
    # Run the script on the data directory
    run_all_results("data")
