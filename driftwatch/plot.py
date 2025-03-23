import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns


def load_json_data(json_path):
    with open(json_path, "r") as file:
        return json.load(file)

def flatten_data(results_data):
    records_list = []
    for dataset_name, model_dict in results_data.items():
        for model_name, records in model_dict.items():
            for r in records:
                distance_type = r.get("distance_type", "vector")
                pca_applied = r.get("pca_applied", False)

                if distance_type == "vector":
                    method = "pca" if pca_applied else "no_pca"
                else:
                    method = "pca_kll_sketch" if pca_applied else "kll_sketch"

                records_list.append({
                    "dataset": dataset_name,
                    "model_name": model_name,
                    "distance_name": r["distance_name"],
                    "distance_type": distance_type,
                    "pca_applied": pca_applied,
                    "method": method,
                    "drift_strength": r["drift_strength"],
                    "final_similarity": r["final_similarity"],
                    "avg_overhead": r.get("avg_overhead", float('nan')),
                    "time_taken": r.get("time_taken", float('nan')),
                })
    return pd.DataFrame(records_list)


def plot_final_similarity(df, output_dir):
    """
    Produce a two-subplot figure:
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
    df_sim = df_sim.groupby(["distance_name", "drift_strength", "method"], as_index=False)["final_similarity"].mean()

    df_sim["final_similarity_norm"] = 0.0
    for dist_name, group_data in df_sim.groupby("distance_name"):
        min_val = group_data["final_similarity"].min()
        max_val = group_data["final_similarity"].max()
        if max_val - min_val < 1e-12:
            # Handle edge case where all values are identical
            df_sim.loc[group_data.index, "final_similarity_norm"] = 0.0
        else:
            df_sim.loc[group_data.index, "final_similarity_norm"] = (
                (group_data["final_similarity"] - min_val) / (max_val - min_val)
            )

    # 2) Separate the distance metrics into two categories
    legacy_set = {
        "mahalanobis", "euclidean", "manhattan", "minkowski", "chebyshev", "canberra"
    }
    dist_set = {
        "kl", "js", "hellinger", "bhattacharyya", "mmd", "wasserstein"
    }

    df_legacy = df_sim[df_sim["distance_name"].isin(legacy_set)].copy()
    df_dist = df_sim[df_sim["distance_name"].isin(dist_set)].copy()

    # If either subset is empty, handle gracefully
    if df_legacy.empty and df_dist.empty:
        print("[Warning] Neither legacy nor distribution-based metrics found in the DataFrame.")
        return

    # 3) For each category, we average across all distance_names
    #    so we get a single trend line per method for that category.
    df_legacy_grouped = (
        df_legacy.groupby(["method", "drift_strength"], as_index=False)["final_similarity_norm"]
        .mean()
    )
    df_dist_grouped = (
        df_dist.groupby(["method", "drift_strength"], as_index=False)["final_similarity_norm"]
        .mean()
    )

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
                marker='o',
                label=method_name,
                color=method_to_color[method_name]
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
                marker='o',
                label=method_name,
                color=method_to_color[method_name]
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


def plot_avg_overhead(df, output_dir):
    # Ensure required fields are present
    if "distance_type" not in df.columns or "pca_applied" not in df.columns:
        raise ValueError("DataFrame must include 'distance_type' and 'pca_applied' columns.")

    # Step 1: Aggregate
    df_grouped = (
        df.groupby(["model_name", "distance_type", "pca_applied"], as_index=False)
          .agg({"avg_overhead": "mean"})
    )

    # Step 2: Create combined method name for bar labels
    df_grouped["method_type"] = df_grouped.apply(
        lambda row: f"{row['distance_type']}_{'pca' if row['pca_applied'] else 'no_pca'}", axis=1
    )

    # Step 3: Pivot to have 4 bars per model
    pivot_df = df_grouped.pivot(index="model_name", columns="method_type", values="avg_overhead").fillna(0)

    # Step 4: Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("pastel", n_colors=4)
    pivot_df.plot(kind='bar', ax=ax, logy=True, color=colors)

    ax.set_xlabel("Model Name", fontsize=12)
    ax.set_ylabel("Avg Overhead (s) [log scale]", fontsize=12)
    ax.set_title("Average Overhead per Model (Vector vs Distribution, PCA vs No PCA)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    ax.legend(title="Method Type", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    path_overhead = os.path.join(output_dir, "plot_model_avg_overhead_4bars.png")
    plt.savefig(path_overhead, dpi=300)
    plt.close()
    print(f"[Saved] {path_overhead}")


def plot_relative_log_increase(df, output_dir):
    plot_data = []

    # Step 1: Compute relative log increase per row
    for _, row in df.iterrows():
        if row["drift_strength"] >= 0:
            base_similarity = df[
                (df["drift_strength"] == 0) &
                (df["distance_name"] == row["distance_name"]) &
                (df["pca_applied"] == row["pca_applied"])
            ]["final_similarity"].mean()

            if base_similarity and base_similarity > 0:
                rel_log = np.log(row["final_similarity"] / base_similarity)
                plot_data.append({
                    "distance_name": row["distance_name"],
                    "drift_strength": row["drift_strength"],
                    "pca_applied": row["pca_applied"],
                    "distance_type": row["distance_type"],
                    "rel_log_increase": rel_log
                })

    df_plot = pd.DataFrame(plot_data)

    # Step 2: Normalize per (distance_name, pca_applied)
    df_normalized = []
    for (dist_name, pca_flag), group in df_plot.groupby(["distance_name", "pca_applied"]):
        df_avg = group.groupby("drift_strength")["rel_log_increase"].mean().reset_index()
        min_val = df_avg["rel_log_increase"].min()
        max_val = df_avg["rel_log_increase"].max()
        if max_val - min_val > 1e-12:
            df_avg["normalized"] = (df_avg["rel_log_increase"] - min_val) / (max_val - min_val)
        else:
            df_avg["normalized"] = 0.0
        df_avg["distance_name"] = dist_name
        df_avg["pca_applied"] = pca_flag
        df_avg["distance_type"] = group["distance_type"].iloc[0]
        df_normalized.append(df_avg)

    df_final = pd.concat(df_normalized, ignore_index=True)

    # Step 3: Create 4-subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes = axes.flatten()

    title_map = {
        (False, "vector"): "Vector-Based (No PCA)",
        (True, "vector"): "Vector-Based (PCA)",
        (False, "distribution"): "Distribution-Based (No PCA)",
        (True, "distribution"): "Distribution-Based (PCA)",
    }

    for i, (pca_flag, dist_type) in enumerate(title_map.keys()):
        ax = axes[i]
        sub_df = df_final[
            (df_final["pca_applied"] == pca_flag) &
            (df_final["distance_type"] == dist_type)
        ]

        for dist_name in sub_df["distance_name"].unique():
            d = sub_df[sub_df["distance_name"] == dist_name]
            ax.plot(d["drift_strength"], d["normalized"], marker='o', label=dist_name)

        ax.set_title(title_map[(pca_flag, dist_type)], fontsize=12, fontweight='bold')
        ax.set_xlabel("Drift Strength")
        ax.grid(True)
        if i % 2 == 0:
            ax.set_ylabel("Normalized Relative Log Increase")
        if i == 3:
            ax.legend(title="Metric", fontsize=9)

    fig.suptitle("Relative Log Increase vs Drift Strength", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path_out = os.path.join(output_dir, "plot_relative_log_increase_4subplots.png")
    plt.savefig(path_out, dpi=300)
    plt.close()
    print(f"[Saved] {path_out}")




def plot_avg_time(df, output_dir):
    df_time = df.groupby(["model_name", "method"], as_index=False)["time_taken"].mean()
    pivot_time = df_time.pivot(index="model_name", columns="method", values="time_taken").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("pastel", n_colors=len(pivot_time.columns))
    pivot_time.plot(kind='bar', logy=True, ax=ax, color=colors)

    ax.set_xlabel("Model Name", fontsize=12)
    ax.set_ylabel("Time Taken (s) [log scale]", fontsize=12)
    ax.set_title("Average Time Taken per Model", fontsize=14, fontweight='bold')


def plot_time_vs_similarity(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="time_taken", y="final_similarity", hue="method", style="distance_name")
    plt.title("Scatter Plot of Time Taken vs. Final Similarity")
    plt.xlabel("Time Taken (s)")
    plt.ylabel("Final Similarity")
    plt.grid(True, linestyle='--', alpha=0.7)
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
                avg_overhead = df[(df["model_name"] == model) & (df["method"] == method)]["avg_overhead"].mean()
                overheads.append(avg_overhead)
        avg_overhead_per_model[model] = np.nanmean(overheads)  # Handle NaN values safely

    sizes = [model_sizes[m] for m in avg_overhead_per_model.keys()]
    overheads = [avg_overhead_per_model[m] for m in avg_overhead_per_model.keys()]

    # Use seaborn for improved aesthetics
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(x=sizes, y=overheads, hue=list(avg_overhead_per_model.keys()), palette="tab10", s=100,
                         edgecolor="black")

    plt.xlabel("Model Size (Million Parameters)", fontsize=12)
    plt.ylabel("Average Overhead", fontsize=12)
    plt.title("Overhead Footprint vs Model Size", fontsize=14)

    # Annotate points with model names
    for model, x, y in zip(avg_overhead_per_model.keys(), sizes, overheads):
        plt.text(x, y, model.split("/")[-1], fontsize=10, ha='right', va='bottom', fontweight='bold')

    # Save the plot
    path_overhead_size = os.path.join(output_dir, "scatter_overhead_vs_size.png")
    plt.savefig(path_overhead_size, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path_overhead_size}")


def plot_final_similarity_separate(df, output_dir):
    """
    Creates 12 subplots (3x4 grid), one per distance_name.
    Each plot shows two lines: PCA applied vs not.
    """

    # Step 1: Compute mean final_similarity for each group
    df_sim = df.groupby(["distance_name", "drift_strength", "pca_applied"], as_index=False)["final_similarity"].mean()

    # Step 2: Normalize final similarity within each distance metric
    df_sim["final_similarity_norm"] = 0.0
    for dist_name, group_data in df_sim.groupby("distance_name"):
        min_val = group_data["final_similarity"].min()
        max_val = group_data["final_similarity"].max()
        if max_val - min_val == 0:
            df_sim.loc[group_data.index, "final_similarity_norm"] = 0.0
        else:
            df_sim.loc[group_data.index, "final_similarity_norm"] = (
                (group_data["final_similarity"] - min_val) / (max_val - min_val)
            )

    # Step 3: Prepare subplots
    unique_distances = sorted(df_sim["distance_name"].unique())
    num_dists = len(unique_distances)
    num_cols = 4
    num_rows = int(np.ceil(num_dists / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False)
    palette = sns.color_palette("Set2", 2)  # 2 colors: PCA applied vs not

    # Step 4: Plot each distance
    for idx, dist_name in enumerate(unique_distances):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row][col]

        for i, pca_flag in enumerate([False, True]):
            subset = df_sim[(df_sim["distance_name"] == dist_name) & (df_sim["pca_applied"] == pca_flag)]
            if not subset.empty:
                label = "PCA" if pca_flag else "No PCA"
                ax.plot(
                    subset["drift_strength"],
                    subset["final_similarity_norm"],
                    marker='o',
                    label=label,
                    color=palette[i]
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
    path_sim = os.path.join(output_dir, "plot_final_similarity_pca_vs_no_pca_by_distance.png")
    plt.savefig(path_sim)
    plt.close()
    print(f"[Saved] {path_sim}")



def generate_all_plots(json_path, output_dir):
    results_data = load_json_data(json_path)
    df = flatten_data(results_data)
    plot_final_similarity(df, output_dir)
    plot_avg_overhead(df, output_dir)
    plot_relative_log_increase(df, output_dir)
    plot_avg_time(df, output_dir)
    plot_overhead_vs_size(df, output_dir)
    plot_final_similarity_separate(df, output_dir)

def run_all_results(data_dir):
    """
    Finds all 'results.json' files in subdirectories under data_dir,
    and runs 'generate_all_plots' on each one.
    """
    for root, _, files in os.walk(data_dir):
        if "results.json" in files:
            json_path = os.path.join(root, "results.json")
            generate_all_plots(json_path, root)


if __name__ == '__main__':
    # Run the script on the data directory
    run_all_results("data")