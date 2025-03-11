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
                method_val = r.get("method", "pca" if r.get("pca", False) else "no_pca")
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
    return pd.DataFrame(records_list)

def plot_final_similarity(df, output_dir):
    # Compute mean final_similarity at each group
    df_sim = df.groupby(["distance_name", "drift_strength", "method"], as_index=False)["final_similarity"].mean()

    # For each distance_name, min–max normalize the final_similarity
    df_sim["final_similarity_norm"] = 0.0
    for dist_name, group_data in df_sim.groupby("distance_name"):
        min_val = group_data["final_similarity"].min()
        max_val = group_data["final_similarity"].max()
        if max_val - min_val == 0:
            # Handle edge case where all values are the same
            df_sim.loc[group_data.index, "final_similarity_norm"] = 0.0
        else:
            df_sim.loc[group_data.index, "final_similarity_norm"] = (
                (group_data["final_similarity"] - min_val) / (max_val - min_val)
            )

    unique_distances = df_sim["distance_name"].unique()
    num_dist = len(unique_distances)
    num_cols = 3
    num_rows = (num_dist + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False)
    axes = axes.flatten()
    palette = sns.color_palette("husl", len(df_sim["method"].unique()))

    for i, dist_name in enumerate(unique_distances):
        ax = axes[i]
        subset = df_sim[df_sim["distance_name"] == dist_name]
        for j, method_name in enumerate(subset["method"].unique()):
            sub_m = subset[subset["method"] == method_name]
            ax.plot(
                sub_m["drift_strength"],
                sub_m["final_similarity_norm"],
                marker='o',
                label=method_name,
                color=palette[j]
            )
        ax.set_title(f"{dist_name} Dist. Similarity")
        ax.set_xlabel("Drift Strength")
        ax.set_ylabel("Normalized Final Similarity (0 - 1)")
        # ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()

    # Remove extra subplots
    for j in range(num_dist, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    path_sim = os.path.join(output_dir, "plot_final_similarity_methods.png")
    plt.savefig(path_sim)
    plt.close()
    print(f"[Saved] {path_sim}")


def plot_avg_overhead(df, output_dir):
    df_overhead = df.groupby(["model_name", "method"], as_index=False)["avg_overhead"].mean()
    pivot_overhead = df_overhead.pivot(index="model_name", columns="method", values="avg_overhead").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("pastel", n_colors=len(pivot_overhead.columns))
    pivot_overhead.plot(kind='bar', logy=True, ax=ax, color=colors)
    ax.set_xlabel("Model Name", fontsize=12)
    ax.set_ylabel("Avg Overhead (s) [log scale]", fontsize=12)
    ax.set_title("Average Overhead per Model", fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    ax.legend(title="Method", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    path_overhead = os.path.join(output_dir, "plot_model_avg_overhead_methods.png")
    plt.savefig(path_overhead, dpi=300)
    plt.close()
    print(f"[Saved] {path_overhead}")

def plot_relative_log_increase(df, output_dir):
    plot_data = []
    for _, row in df.iterrows():
        if row["drift_strength"] >= 0:
            base_similarity = df[(df["drift_strength"] == 0) & (df["method"] == row["method"]) & (df["distance_name"] == row["distance_name"])]["final_similarity"].mean()
            if base_similarity:
                relative_log_increase = np.log(row["final_similarity"] / base_similarity)
                plot_data.append((row["distance_name"], row["drift_strength"], relative_log_increase))

    aggregated_data = defaultdict(list)
    for distance_name, drift_strength, log_increase in plot_data:
        aggregated_data[distance_name].append((drift_strength, log_increase))

    aggregated_results = {}
    for distance_name, values in aggregated_data.items():
        df_agg = pd.DataFrame(values, columns=["drift_strength", "log_increase"])
        avg_log_increase = df_agg.groupby("drift_strength")["log_increase"].mean().reset_index()
        avg_log_increase["log_increase"] = (avg_log_increase["log_increase"] - avg_log_increase["log_increase"].min()) / (avg_log_increase["log_increase"].max() - avg_log_increase["log_increase"].min())
        aggregated_results[distance_name] = avg_log_increase

    plt.figure(figsize=(12, 6))
    for distance_name, df_agg in aggregated_results.items():
        sns.lineplot(data=df_agg, x="drift_strength", y="log_increase", label=distance_name, marker='o')
    plt.xlabel("Drift Strength", fontsize=12)
    plt.ylabel("Normalized Relative Log Increase", fontsize=12)
    plt.title("Aggregated Relative Increase of Final Similarity vs Drift Strength", fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    path_agg = os.path.join(output_dir, "plot_aggregated_relative_log_increase_normalized.png")
    plt.savefig(path_agg, dpi=300)
    plt.close()
    print(f"[Saved] {path_agg}")

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


def generate_all_plots(json_path, output_dir):
    results_data = load_json_data(json_path)
    df = flatten_data(results_data)
    plot_final_similarity(df, output_dir)
    plot_avg_overhead(df, output_dir)
    plot_relative_log_increase(df, output_dir)
    plot_avg_time(df, output_dir)
    plot_overhead_vs_size(df, output_dir)

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