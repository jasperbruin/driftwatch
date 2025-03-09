import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_time_overhead(json_path, output_dir):
    """Plots time taken and overhead for each measurement, with PCA and non-PCA."""

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
                    "pca": record["pca"],
                    "time_taken": record["time_taken"],
                    "avg_overhead": record["avg_overhead"]
                })

    df = pd.DataFrame(data_list)

    # Aggregate time taken and overhead per distance_name and PCA status
    df_aggregated = df.groupby(["distance_name", "pca"], as_index=False).mean()

    # Plot Time Taken Bar Chart with Log Scale
    fig, ax = plt.subplots(figsize=(12, 6))
    distance_names = df_aggregated["distance_name"].unique()
    bar_width = 0.4
    x = np.arange(len(distance_names))

    for i, pca_status in enumerate([False, True]):
        subset = df_aggregated[df_aggregated["pca"] == pca_status]
        ax.bar(x + i * bar_width, subset["time_taken"], width=bar_width, label=f'PCA={pca_status}')

    ax.set_xlabel("Distance Name")
    ax.set_ylabel("Time Taken (s) [Log Scale]")
    ax.set_title("Average Time Taken per Distance Measure (Log Scale)")
    ax.set_yscale("log")  # Set log scale for y-axis
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(distance_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', which="both")

    plt.tight_layout()
    time_plot_path = os.path.join(output_dir, "plot_time_taken.png")
    plt.savefig(time_plot_path)
    plt.close()
    print(f"Saved time plot to {time_plot_path}")

    # Plot Average Overhead Bar Chart with Log Scale
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, pca_status in enumerate([False, True]):
        subset = df_aggregated[df_aggregated["pca"] == pca_status]
        ax.bar(x + i * bar_width, subset["avg_overhead"], width=bar_width, label=f'PCA={pca_status}')

    ax.set_xlabel("Distance Name")
    ax.set_ylabel("Average Overhead (s) [Log Scale]")
    ax.set_title("Average Overhead per Distance Measure (Log Scale)")
    ax.set_yscale("log")  # Set log scale for y-axis
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(distance_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', which="both")

    plt.tight_layout()
    overhead_plot_path = os.path.join(output_dir, "plot_avg_overhead.png")
    plt.savefig(overhead_plot_path)
    plt.close()
    print(f"Saved overhead plot to {overhead_plot_path}")


def process_all_json_time_overhead(data_dir):
    """Finds all results.json files in data subdirectories and generates time and overhead plots."""
    for root, _, files in os.walk(data_dir):
        if "results.json" in files:
            json_path = os.path.join(root, "results.json")
            plot_time_overhead(json_path, root)



