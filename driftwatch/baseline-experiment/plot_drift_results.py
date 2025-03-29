import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Configuration constants
OUTPUT_DIR = "baseline_results"

def create_output_directory():
    """Create the output directory for results if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def plot_drift_results(results_file):
    """Plot drift results from a single CSV file."""
    # Extract metric name from filename
    filename = os.path.basename(results_file)
    distance_metric = filename.replace("drift_detection_results_", "").replace(".csv", "")
    
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
    plt.plot(all_distances, label="Drift Distance", color='blue')
    plt.plot(all_thresholds, label="Threshold", color='red', linestyle="--")
    plt.title(f"Embedding Drift Distance per Window ({distance_metric})")
    plt.xlabel("Window Index")
    plt.ylabel("Drift Distance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_image = os.path.join(OUTPUT_DIR, f"drift_distance_by_index_{distance_metric}.png")
    plt.savefig(output_image)
    plt.close()
    print(f"Saved plot to {output_image}")

    # Plot 2: Drift distance over time with threshold and detected drift points
    plt.figure(figsize=(12, 6))
    plt.plot(window_times, all_distances, label="Drift Score", color="blue")
    plt.plot(window_times, all_thresholds, label="Threshold", color="red", linestyle="--")
    
    # Highlight detected drift windows on the plot
    drift_times = [t for t, lbl in zip(window_times, all_labels) if lbl == 1]
    drift_scores = [d for d, lbl in zip(all_distances, all_labels) if lbl == 1]
    if drift_times:
        plt.scatter(drift_times, drift_scores, color="red", marker="o", s=50, label="Detected Drift")
    
    # Optionally highlight a known drift period (if relevant)
    plt.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01"), color="purple", alpha=0.1, 
                label="Observed Drift (2021-2023)")
                
    plt.title(f"Drift Scores Over Time ({distance_metric})", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Drift Score", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_image = os.path.join(OUTPUT_DIR, f"drift_score_over_time_{distance_metric}.png")
    plt.savefig(output_image)
    plt.close()
    print(f"Saved plot to {output_image}")
    
    return {
        "metric": distance_metric,
        "window_times": window_times,
        "distances": all_distances
    }

def create_comparison_plot(results_data):
    """Create a comparison plot of drift scores across all metrics."""
    plt.figure(figsize=(14, 8))
    
    for result in results_data:
        plt.plot(result["window_times"], result["distances"], label=result["metric"])
    
    plt.title("Comparison of Drift Scores Across Distance Metrics", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Drift Score", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_image = os.path.join(OUTPUT_DIR, "drift_score_comparison.png")
    plt.savefig(output_image)
    plt.close()
    print(f"Saved comparison plot to {output_image}")

def main():
    """Main function to plot drift detection results."""
    # Ensure output directory exists
    create_output_directory()
    
    # Find all result files
    result_files = glob(os.path.join(OUTPUT_DIR, "drift_detection_results_*.csv"))
    
    if not result_files:
        print(f"No result files found in {OUTPUT_DIR}.")
        return
    
    print(f"Found {len(result_files)} result files.")
    
    # Plot each metric's results and collect data for comparison plot
    all_results = []
    for file in result_files:
        result = plot_drift_results(file)
        all_results.append(result)
    
    # Create comparison plot
    create_comparison_plot(all_results)
    
    print("All plots generated successfully.")

if __name__ == "__main__":
    main()
