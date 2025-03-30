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

def plot_memory_usage(output_dir="baseline_results"):
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
    vector_metrics = ['cosine', 'euclidean', 'manhattan', 'minkowski', 'mahalanobis', 'chebyshev', 'canberra']
    distribution_metrics = ['wasserstein', 'ks', 'kl', 'js', 'hellinger', 'bhattacharyya', 'mmd']
    
    # Classify each metric
    memory_data['metric_type'] = memory_data['metric'].apply(
        lambda x: 'Vector-based' if any(vm.lower() in x.lower() for vm in vector_metrics) 
        else ('Distribution-based' if any(dm.lower() in x.lower() for dm in distribution_metrics) 
              else 'Other')
    )
    
    # Plot 1: Memory usage by metric (peak)
    plt.figure(figsize=(14, 8))
    bars = plt.bar(memory_data['metric'], memory_data['peak_memory_mb'])
    
    # Color bars by metric type
    colors = {'Vector-based': 'skyblue', 'Distribution-based': 'salmon', 'Other': 'lightgray'}
    for i, bar in enumerate(bars):
        bar.set_color(colors[memory_data.iloc[i]['metric_type']])
    
    plt.title("Peak Memory Usage by Distance Metric", fontsize=14)
    plt.xlabel("Distance Metric", fontsize=12)
    plt.ylabel("Peak Memory Usage (MB)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=colors[t]) for t in colors]
    plt.legend(handles, colors.keys(), title="Metric Type")
    
    plt.tight_layout()
    output_image = os.path.join(output_dir, "memory_usage_peak.png")
    plt.savefig(output_image)
    plt.close()
    print(f"Saved memory peak usage plot to {output_image}")
    
    # Plot 2: Memory progression (initialization -> baseline -> final)
    plt.figure(figsize=(14, 10))
    
    # Group by metric type and calculate means
    grouped = memory_data.groupby('metric_type')
    
    x = np.arange(3)  # init, baseline, final
    width = 0.25  # width of bars
    
    # Plot bars for each metric type
    i = 0
    for name, group in grouped:
        means = [group['init_memory_mb'].mean(), 
                 group['baseline_memory_mb'].mean(), 
                 group['final_memory_mb'].mean()]
        
        plt.bar(x + i*width, means, width, label=name, color=colors[name])
        i += 1
    
    plt.xlabel('Stage', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.title('Memory Usage Progression by Metric Type', fontsize=14)
    plt.xticks(x + width, ['Initialization', 'Baseline', 'Final'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_image = os.path.join(output_dir, "memory_progression.png")
    plt.savefig(output_image)
    plt.close()
    print(f"Saved memory progression plot to {output_image}")
    
    # Plot 3: Memory usage increase from init to final
    memory_data['memory_increase'] = memory_data['final_memory_mb'] - memory_data['init_memory_mb']
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(memory_data['metric'], memory_data['memory_increase'])
    
    # Color bars by metric type
    for i, bar in enumerate(bars):
        bar.set_color(colors[memory_data.iloc[i]['metric_type']])
    
    plt.title("Memory Growth During Drift Detection by Metric", fontsize=14)
    plt.xlabel("Distance Metric", fontsize=12)
    plt.ylabel("Memory Increase (MB)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
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
    
    # Plot memory usage
    plot_memory_usage(output_dir)
    
    print("All plots generated successfully.")

if __name__ == "__main__":
    main()
