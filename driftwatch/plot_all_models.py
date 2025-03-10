import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def generate_all_plots(json_path, output_dir):
    """
    Generates and saves:
      1. Final similarity vs. drift strength (log scale).
      2. Average overhead per model (PCA vs. non-PCA).
      3. Average time taken per model (PCA vs. non-PCA).
      4. PCA Effectiveness vs. Computational Cost (scatter plot).
    """
    # Load JSON data
    with open(json_path, "r") as file:
        results_data = json.load(file)

    # Flatten the JSON into a DataFrame
    data_list = []
    for dataset, models in results_data.items():
        for model_name, records in models.items():
            for record in records:
                data_list.append({
                    "model_name": model_name,
                    "distance_name": record["distance_name"],
                    "drift_strength": record["drift_strength"],
                    "pca": record["pca"],
                    "final_similarity": record["final_similarity"],
                    "avg_overhead": record.get("avg_overhead", float('nan')),
                    "time_taken": record.get("time_taken", float('nan'))
                })

    df = pd.DataFrame(data_list)

    # --- 1) Average Overhead per Model (PCA vs. Non‐PCA) ---

    df_agg_overhead = df.groupby(["model_name", "pca"], as_index=False)["avg_overhead"].mean()
    pivot_overhead = df_agg_overhead.pivot(index="model_name", columns="pca", values="avg_overhead")
    pivot_overhead.columns = ["PCA=False", "PCA=True"]
    pivot_overhead = pivot_overhead.fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_overhead.plot(kind='bar', logy=True, ax=ax, title="Average Overhead per Model (Log Scale) [PCA vs. Non‐PCA]")
    ax.set_xlabel("Model Name")
    ax.set_ylabel("Average Overhead (s) [log scale]")
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels
    plt.tight_layout()

    out_path_overhead = os.path.join(output_dir, "plot_model_avg_overhead.png")
    plt.savefig(out_path_overhead)
    plt.close()
    print(f"Saved overhead plot to {out_path_overhead}")

    # --- 2) Average Time Taken per Model (PCA vs. Non‐PCA) ---

    df_agg_time = df.groupby(["model_name", "pca"], as_index=False)["time_taken"].mean()
    pivot_time = df_agg_time.pivot(index="model_name", columns="pca", values="time_taken")
    pivot_time.columns = ["PCA=False", "PCA=True"]
    pivot_time = pivot_time.fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_time.plot(kind='bar', logy=True, ax=ax, title="Average Time Taken per Model (Log Scale) [PCA vs. Non‐PCA]")
    ax.set_xlabel("Model Name")
    ax.set_ylabel("Time Taken (s) [log scale]")
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels
    plt.tight_layout()

    out_path_time = os.path.join(output_dir, "plot_model_time_taken.png")
    plt.savefig(out_path_time)
    plt.close()
    print(f"Saved time-taken plot to {out_path_time}")

    # --- 3) PCA Effectiveness vs. Computational Cost (Scatter Plot) ---

    df_pca_effect = df.groupby(["model_name", "pca"]).agg(
        avg_similarity=("final_similarity", "mean"),
        avg_overhead=("avg_overhead", "mean")
    ).unstack()

    df_pca_effect["similarity_gain"] = (df_pca_effect["avg_similarity", True] / df_pca_effect[
        "avg_similarity", False]) - 1
    df_pca_effect["overhead_non_pca"] = df_pca_effect["avg_overhead", False]
    df_pca_effect["avg_similarity"] = (df_pca_effect["avg_similarity", True] + df_pca_effect[
        "avg_similarity", False]) / 2

    df_pca_effect = df_pca_effect.reset_index()
    df_pca_effect.columns = ['_'.join(map(str, col)).strip() for col in df_pca_effect.columns]

    df_pca_effect = df_pca_effect.rename(columns={
        "model_name_": "model_name",
        "avg_similarity_False": "avg_similarity_no_pca",
        "avg_similarity_True": "avg_similarity_pca",
        "avg_overhead_False": "overhead_no_pca",
        "avg_overhead_True": "overhead_pca",
        "similarity_gain_": "similarity_gain",
        "overhead_non_pca_": "overhead_non_pca"
    })

    df_pca_effect["similarity_gain"] = (df_pca_effect["avg_similarity_pca"] / df_pca_effect[
        "avg_similarity_no_pca"]) - 1
    df_pca_effect["overhead_non_pca"] = df_pca_effect["overhead_no_pca"]
    df_pca_effect["avg_similarity"] = (df_pca_effect["avg_similarity_pca"] + df_pca_effect["avg_similarity_no_pca"]) / 2

    df_pca_effect = df_pca_effect.dropna(subset=["similarity_gain", "overhead_non_pca", "avg_similarity"])

    marker_sizes = np.sqrt(df_pca_effect["avg_similarity"].clip(lower=1)) * 10

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df_pca_effect["similarity_gain"],
        df_pca_effect["overhead_non_pca"],
        s=marker_sizes,
        c=np.arange(len(df_pca_effect)),
        cmap="coolwarm",
        edgecolors="black",
        alpha=0.8
    )

    plt.axvline(x=0, color="gray", linestyle="--", label="No PCA Effect")
    plt.xlabel("PCA Similarity Gain (%)")
    plt.ylabel("Average Overhead (Non-PCA) [s]")
    plt.title("Trade-off Between PCA Effectiveness & Computation Cost")
    plt.colorbar(scatter, label="Model Index")
    plt.grid(True)

    for i, row in df_pca_effect.iterrows():
        plt.annotate(row["model_name"], (row["similarity_gain"], row["overhead_non_pca"]), fontsize=9)

    out_path_pca_tradeoff = os.path.join(output_dir, "plot_pca_tradeoff.png")
    plt.savefig(out_path_pca_tradeoff)
    plt.close()
    print(f"Saved PCA trade-off plot to {out_path_pca_tradeoff}")


def run_all_results(data_dir):
    """
    Finds all 'results.json' files in subdirectories under data_dir,
    and runs 'generate_all_plots' on each one.
    """
    for root, _, files in os.walk(data_dir):
        if "results.json" in files:
            json_path = os.path.join(root, "results.json")
            generate_all_plots(json_path, root)


