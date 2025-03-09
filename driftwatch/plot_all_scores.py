import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import math


def plot_and_save(json_path, output_dir):
    """Plots and saves final similarity on a log scale with drift strength on x-axis."""

    # Load JSON data
    with open(json_path, "r") as file:
        results_data = json.load(file)

    # Flatten the JSON into a DataFrame
    data_list = []
    for dataset, models in results_data.items():
        for model, records in models.items():
            for record in records:
                data_list.append({
                    "distance_name": record["distance_name"],
                    "drift_strength": record["drift_strength"],
                    "pca": record["pca"],
                    "final_similarity": record["final_similarity"]
                })

    df = pd.DataFrame(data_list)

    # Aggregate to get mean final similarity per distance_name, drift_strength, and PCA status
    df_aggregated = df.groupby(["distance_name", "drift_strength", "pca"], as_index=False)["final_similarity"].mean()

    # Select first 6 unique distance names
    selected_distances = df_aggregated["distance_name"].unique()[:6]
    num_plots = len(selected_distances)
    num_cols = 3
    num_rows = 2

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), squeeze=False)
    axes = axes.flatten()

    for i, distance_name in enumerate(selected_distances):
        ax = axes[i]
        subset = df_aggregated[df_aggregated["distance_name"] == distance_name]

        for pca_status in [True, False]:
            subset_pca = subset[subset["pca"] == pca_status]
            ax.plot(
                subset_pca["drift_strength"],
                subset_pca["final_similarity"],
                marker='o',
                linestyle='-',
                label=f'PCA={pca_status}'
            )

        ax.set_title(f"Distance: {distance_name}")
        ax.set_xlabel("Drift Strength")
        ax.set_ylabel("Final Similarity (log scale)")
        ax.set_yscale("log")  # Set y-axis to log scale
        ax.legend()
        ax.grid()

    # Hide any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, "plot_log_similarity.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def process_all_scores(data_dir):
    """Finds all results.json files in data subdirectories and generates plots."""
    for root, _, files in os.walk(data_dir):
        if "results.json" in files:
            json_path = os.path.join(root, "results.json")
            plot_and_save(json_path, root)
