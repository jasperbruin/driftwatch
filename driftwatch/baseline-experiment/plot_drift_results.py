import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def plot_drift_results(results_file, output_dir="baseline_results"):
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
    
    output_image = os.path.join(output_dir, f"drift_distance_by_index_{distance_metric}.png")
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
    
    output_image = os.path.join(output_dir, f"drift_score_over_time_{distance_metric}.png")
    plt.savefig(output_image)
    plt.close()
    print(f"Saved plot to {output_image}")
    
    return {
        "metric": distance_metric,
        "window_times": window_times,
        "distances": all_distances
    }

def create_comparison_plot(results_data, output_dir="baseline_results"):
    """Create a comparison plot with two subplots separating vector and distribution-based metrics."""
    # Define which metrics are vector-based vs distribution-based
    vector_metrics = ['cosine', 'euclidean', 'manhattan', 'minkowski', 'mahalanobis', 'chebyshev', 'canberra']
    distribution_metrics = ['wasserstein', 'ks', 'kl', 'js', 'hellinger', 'bhattacharyya', 'mmd']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Sort results into the two categories
    vector_results = []
    distribution_results = []
    
    for result in results_data:
        metric_name = result["metric"].lower()
        
        # Check if metric belongs to either category
        is_vector = any(vm in metric_name for vm in vector_metrics)
        is_distribution = any(dm in metric_name for dm in distribution_metrics)
        
        if is_vector:
            vector_results.append(result)
        elif is_distribution:
            distribution_results.append(result)
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
        
        ax1.plot(result["window_times"], normalized_distances, label=f"{result['metric']}")

    # Add observed drift period to the first subplot
    ax1.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01"), color="purple", alpha=0.1,
                label="Observed Drift (2021-2023)")

    # Plot distribution-based metrics in the second subplot
    for result in distribution_results:
        # Normalize the drift scores to range [0, 1]
        distances = np.array(result["distances"])
        min_val = np.min(distances)
        max_val = np.max(distances)
        
        # Check to avoid division by zero if all values are the same
        if max_val == min_val:
            normalized_distances = np.zeros_like(distances)
        else:
            normalized_distances = (distances - min_val) / (max_val - min_val)
        
        ax2.plot(result["window_times"], normalized_distances, label=f"{result['metric']}")
    
    # Add observed drift period to the second subplot
    ax2.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01"), color="purple", alpha=0.1,
                label="Observed Drift (2021-2023)")

    # Configure the first subplot (vector-based)
    ax1.set_title("Vector Distance Metrics", fontsize=14)
    ax1.set_ylabel("Drift Score", fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Configure the second subplot (distribution-based)
    ax2.set_title("Distribution-Based Metrics", fontsize=14)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Drift Score", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # Add a main title for the entire figure
    fig.suptitle("Comparison of Drift Scores by Metric Type", fontsize=16)
    plt.tight_layout()
    
    output_image = os.path.join(output_dir, "drift_score_comparison.png")
    plt.savefig(output_image)
    plt.close()
    print(f"Saved comparison plot to {output_image}")

def main():
    """Main function to plot drift detection results."""
    # Allow input dir as command line argument if desired
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "baseline_results"
    
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist.")
        return
        
    # Find all result files
    result_files = glob(os.path.join(output_dir, "drift_detection_results_*.csv"))
    
    if not result_files:
        print(f"No result files found in {output_dir}.")
        return
    
    print(f"Found {len(result_files)} result files.")
    
    # Plot each metric's results and collect data for comparison plot
    all_results = []
    for file in result_files:
        result = plot_drift_results(file, output_dir)
        all_results.append(result)
    
    # Create comparison plot
    create_comparison_plot(all_results, output_dir)
    
    print("All plots generated successfully.")

if __name__ == "__main__":
    main()
