import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_all_plots(json_path, output_dir):
    """
    Generates and saves:
      1. Final similarity vs. drift strength (log scale), with lines for each method.
      2. Average overhead per model, pivoted by method.
      3. Average time taken per model, pivoted by method.
      4. Method trade-off scatter: average similarity vs. overhead.
    """
    # 1) Load the JSON data
    with open(json_path, "r") as file:
        results_data = json.load(file)

    # 2) Flatten into a DataFrame
    records_list = []
    for dataset_name, model_dict in results_data.items():
        for model_name, records in model_dict.items():
            for r in records:
                # Check if 'method' is provided. If not, fallback to pca/no_pca
                method_val = r.get("method", None)
                if not method_val:
                    method_val = "pca" if r.get("pca", False) else "no_pca"
                records_list.append({
                    "dataset": dataset_name,
                    "model_name": model_name,
                    "distance_name": r["distance_name"],
                    "drift_strength": r["drift_strength"],
                    "method": method_val,
                    "final_similarity": r["final_similarity"],
                    "avg_overhead": r.get("avg_overhead", float('nan')),
                    "time_taken": r.get("time_taken", float('nan')),
                })

    df = pd.DataFrame(records_list)

    # -----------------------------
    # 1) Final similarity vs. drift strength (log scale)
    # -----------------------------
    df_sim = df.groupby(["distance_name", "drift_strength", "method"], as_index=False)["final_similarity"].mean()

    unique_distances = df_sim["distance_name"].unique()
    num_dist = len(unique_distances)
    num_cols = 3
    num_rows = (num_dist + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False)
    axes = axes.flatten()

    for i, dist_name in enumerate(unique_distances):
        ax = axes[i]
        subset = df_sim[df_sim["distance_name"] == dist_name]
        for method_name in subset["method"].unique():
            sub_m = subset[subset["method"] == method_name]
            ax.plot(
                sub_m["drift_strength"],
                sub_m["final_similarity"],
                marker='o',
                label=method_name
            )
        ax.set_title(f"{dist_name} Dist. Similarity")
        ax.set_xlabel("Drift Strength")
        ax.set_ylabel("Final Similarity (log scale)")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()

    # Hide extra subplots
    for j in range(num_dist, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    path_sim = os.path.join(output_dir, "plot_final_similarity_methods.png")
    plt.savefig(path_sim)
    plt.close()
    print(f"[Saved] {path_sim}")

    # -----------------------------
    # 2) Average Overhead per model, pivoted by method
    # -----------------------------
    df_overhead = df.groupby(["model_name", "method"], as_index=False)["avg_overhead"].mean()
    pivot_overhead = df_overhead.pivot(index="model_name", columns="method", values="avg_overhead").fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_overhead.plot(kind='bar', logy=True, ax=ax, title="Average Overhead per Model (Log Scale)")
    ax.set_xlabel("Model Name")
    ax.set_ylabel("Avg Overhead (s) [log scale]")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path_overhead = os.path.join(output_dir, "plot_model_avg_overhead_methods.png")
    plt.savefig(path_overhead)
    plt.close()
    print(f"[Saved] {path_overhead}")

    # -----------------------------
    # 3) Average Time per model, pivoted by method
    # -----------------------------
    df_time = df.groupby(["model_name", "method"], as_index=False)["time_taken"].mean()
    pivot_time = df_time.pivot(index="model_name", columns="method", values="time_taken").fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_time.plot(kind='bar', logy=True, ax=ax, title="Average Time Taken per Model (Log Scale)")
    ax.set_xlabel("Model Name")
    ax.set_ylabel("Time Taken (s) [log scale]")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path_time = os.path.join(output_dir, "plot_model_time_taken_methods.png")
    plt.savefig(path_time)
    plt.close()
    print(f"[Saved] {path_time}")

    # -----------------------------
    # 4) Method trade-off scatter
    # -----------------------------
    # For each (model, method), plot average final_similarity vs. overhead
    df_trade = df.groupby(["model_name", "method"], as_index=False).agg(
        avg_similarity=("final_similarity", "mean"),
        avg_overhead=("avg_overhead", "mean")
    )

    # You could do custom logic if you want to measure 'gain' from no_pca to something else.
    # For simplicity, let's just plot raw values for each method.
    plt.figure(figsize=(8, 6))
    marker_sizes = np.sqrt(df_trade["avg_similarity"].clip(lower=1)) * 15
    method_codes = df_trade["method"].astype("category").cat.codes

    scatter = plt.scatter(
        df_trade["avg_similarity"], df_trade["avg_overhead"],
        s=marker_sizes, c=method_codes,
        cmap="rainbow", alpha=0.7, edgecolor="black"
    )
    plt.xlabel("Average Similarity")
    plt.ylabel("Average Overhead (s)")
    plt.title("Method Trade-off: Similarity vs Overhead")
    plt.grid(True)

    # Annotate with model_name + method
    for i, row in df_trade.iterrows():
        plt.annotate(f"{row['model_name']} ({row['method']})",
                     (row["avg_similarity"], row["avg_overhead"]),
                     fontsize=8)

    cbar = plt.colorbar(scatter, label="Method Code")
    path_scatter = os.path.join(output_dir, "plot_method_tradeoff.png")
    plt.savefig(path_scatter)
    plt.close()
    print(f"[Saved] {path_scatter}")


def run_all_results(data_dir):
    """
    Finds all 'results.json' files in subdirectories under data_dir,
    and runs 'generate_all_plots' on each one.
    """
    for root, _, files in os.walk(data_dir):
        if "results.json" in files:
            json_path = os.path.join(root, "results.json")
            generate_all_plots(json_path, root)

