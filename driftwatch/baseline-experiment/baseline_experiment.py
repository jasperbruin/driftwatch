import os
import pandas as pd
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
from driftwatch.embedding_tracker import EmbeddingTracker
from driftwatch.metrics import get_available_metrics
from plot_drift_results import plot_drift_results, create_comparison_plot

# Configuration constants
THRESHOLD_MULTIPLIER = 2.5
AVAILABLE_DISTANCE_METRICS = get_available_metrics()  # Get all metrics from the metrics module
BASE_OUTPUT_DIR = "baseline_results"
N_WINDOWS = 300
BASELINE_FRACTION = 0.3
EMBEDDING_DIM = 64
EPOCHS = 2
BATCH_SIZE = 1024
THRESHOLD_WINDOW = 30
ADAPTIVE_UPDATE = False

def create_output_directory(base_dir=BASE_OUTPUT_DIR):
    """Create a timestamped output directory for results."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_dir, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def load_and_preprocess_data(filepath: str):
    """Load Amazon reviews data from JSON lines and preprocess for drift detection."""
    # Load data
    data = pd.read_json(filepath, lines=True)
    # Filter out neutral ratings and binarize the rating
    data = data[data["rating"] != 3]
    data["rating"] = data["rating"].apply(lambda x: 1 if x > 3 else 0)
    # Filter by timestamp if needed
    data = data[data["timestamp"] > '2015-01-01']
    # Set timestamp as index and sort chronologically
    data = data.set_index("timestamp")
    data.sort_index(inplace=True)
    return data

def train_deepfm_model(data: pd.DataFrame, sparse_features: list, target: str, 
                       embedding_dim: int = 64, epochs: int = 2, batch_size: int = 1024):
    """
    Train a DeepFM model on the given data for the binary classification task.
    Returns the trained model.
    """
    # Label encode categorical features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    
    # Define feature columns for DeepFM
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=embedding_dim)
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    
    # Prepare model input dict
    model_input = {name: data[name].values for name in feature_names}
    
    # Initialize DeepFM model
    # Decide on device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    
    # Train the model
    model.fit(model_input, data[target].values, batch_size=batch_size, epochs=epochs, verbose=3, validation_split=0.1)
    
    # Switch model to evaluation mode for inference
    model.eval()
    return model

def split_into_windows(data: pd.DataFrame, n_windows: int):
    """
    Split the data into n_windows sequential time-ordered segments (windows).
    Returns a list of dataframes, one per window.
    """
    # Use numpy array_split to split indices into roughly equal segments
    indices = np.array(data.index)  # numpy array of timestamps
    windows = []
    split_indices = np.array_split(indices, n_windows)
    for inds in split_indices:
        # Create each window DataFrame by selecting rows for those indices
        window_df = data.loc[inds]
        windows.append(window_df)
    return windows

def extract_embeddings(model, batch_df: pd.DataFrame, sparse_features: list):
    """
    Extract the concatenated embeddings for all samples in batch_df using the trained model.
    Returns a NumPy array of shape (num_samples, total_embedding_dim).
    """
    # Prepare model input for the batch
    batch_input = {feat: batch_df[feat].values for feat in sparse_features}
    
    # Forward pass (ensure no gradient computation)
    with torch.no_grad():
        _ = model.predict(batch_input)  # run through model to ensure embeddings are materialized
    
    # Collect embeddings from each feature's embedding layer
    embeddings_list = []
    for feat in sparse_features:
        # Get the full embedding matrix for this feature and pick the rows for our batch indices
        emb_matrix = model.embedding_dict[feat].weight.detach().cpu().numpy()
        indices = batch_input[feat]
        batch_emb = emb_matrix[indices]
        embeddings_list.append(batch_emb)
    
    # Concatenate embeddings from all features to form the complete representation
    embeddings = np.concatenate(embeddings_list, axis=1)
    return embeddings

def establish_baseline(embedding_tracker: EmbeddingTracker, windows: list, baseline_count: int):
    """
    Use the first baseline_count windows to establish the baseline distribution in the embedding tracker.
    Returns a dictionary with baseline drift distances and the initial threshold.
    """
    baseline_distances = []
    # Iterate over baseline windows
    for i in range(baseline_count):
        window_df = windows[i]
        emb = extract_embeddings(model, window_df, sparse_features)
        # If using vector-based distance, use the mean embedding; if distribution-based, use full embeddings
        emb_input = emb if embedding_tracker.is_distribution_mode else np.mean(emb, axis=0)
        if i == 0:
            # Initialize baseline in tracker with the first window
            embedding_tracker.update(emb_input)
            baseline_distances.append(0.0)  # no drift distance for the first window
        else:
            # Compute distance of this window's embedding distribution to current baseline
            drift_dist = embedding_tracker.compute_distance(emb_input)
            baseline_distances.append(float(drift_dist))
            # Update the baseline tracker with this window
            embedding_tracker.update(emb_input)
    
    # Calculate initial drift threshold as mean + k*std of baseline distances (excluding the very first 0.0)
    if len(baseline_distances) > 1:
        baseline_dist_array = np.array(baseline_distances[1:])  # exclude the initial 0.0
        initial_threshold = np.mean(baseline_dist_array) + THRESHOLD_MULTIPLIER * np.std(baseline_dist_array)
    else:
        initial_threshold = float(baseline_distances[0])  # if only one baseline window, use its (zero) distance
    
    # For baseline period, we consider no drift occurred, so labels are 0
    baseline_labels = [0] * len(baseline_distances)
    # Also prepare a threshold list for baseline windows (initial_threshold applied throughout baseline)
    baseline_thresholds = [initial_threshold] * len(baseline_distances)
    
    return {
        "distances": baseline_distances,
        "thresholds": baseline_thresholds,
        "labels": baseline_labels,
        "initial_threshold": initial_threshold
    }

def detect_drift(embedding_tracker: EmbeddingTracker, windows: list, baseline_count: int, initial_threshold: float,
                 threshold_window: int = 50, threshold_multiplier: float = 3.0, adaptive_update: bool = False):
    """
    Compute drift distances for windows beyond the baseline period and detect drift events.
    Uses a rolling window of past `threshold_window` drift scores to update the threshold.
    If adaptive_update is True, update the baseline tracker with windows that are NOT flagged as drift.
    Returns a dict with drift distances, thresholds, and drift labels for the post-baseline windows.
    """
    drift_distances = []
    thresholds = []
    drift_labels = []
    
    # Iterate through each window after the baseline period
    for j in range(baseline_count, len(windows)):
        window_df = windows[j]
        emb = extract_embeddings(model, window_df, sparse_features)
        emb_input = emb if embedding_tracker.is_distribution_mode else np.mean(emb, axis=0)
        
        # Compute the embedding distance to baseline distribution
        drift_dist = embedding_tracker.compute_distance(emb_input)
        drift_distances.append(float(drift_dist))
        
        # Compute dynamic threshold based on recent `threshold_window` distances
        if len(drift_distances) > threshold_window:
            recent = drift_distances[-threshold_window:]  # last N drift distances
            thresh = np.mean(recent) + threshold_multiplier * np.std(recent)
        else:
            thresh = initial_threshold
        thresholds.append(float(thresh))
        
        # Determine drift label for this window
        if drift_dist > thresh:
            drift_labels.append(1)  # drift detected
        else:
            drift_labels.append(0)  # no drift
            if adaptive_update:
                # Optionally update baseline tracker with this window if it's considered stable
                embedding_tracker.update(emb_input)
    
    return {
        "distances": drift_distances,
        "thresholds": thresholds,
        "labels": drift_labels
    }

def save_drift_results(output_dir, window_times, all_distances, all_thresholds, all_labels, distance_metric):
    """Save drift detection results to CSV."""
    # Save drift distances and thresholds to disk as a CSV
    results_df = pd.DataFrame({
        "window_time": window_times,
        "drift_distance": all_distances,
        "threshold": all_thresholds,
        "drift_detected": all_labels
    })
    
    output_file = os.path.join(output_dir, f"drift_detection_results_{distance_metric}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Drift results saved to '{output_file}'.")
    return output_file

def save_hyperparameters(output_dir):
    """Save all hyperparameters to a config.txt file in the results directory."""
    hyperparams = {
        "THRESHOLD_MULTIPLIER": THRESHOLD_MULTIPLIER,
        "AVAILABLE_DISTANCE_METRICS": ", ".join(AVAILABLE_DISTANCE_METRICS),
        "BASE_OUTPUT_DIR": BASE_OUTPUT_DIR,
        "N_WINDOWS": N_WINDOWS,
        "BASELINE_FRACTION": BASELINE_FRACTION,
        "EMBEDDING_DIM": EMBEDDING_DIM,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "THRESHOLD_WINDOW": THRESHOLD_WINDOW,
        "ADAPTIVE_UPDATE": ADAPTIVE_UPDATE,
        "EMBEDDING_TRACKER_ALPHA": 0.01,  # Alpha value used in EmbeddingTracker
        "MODEL_OPTIMIZER": "adam",
        "MODEL_LOSS": "binary_crossentropy",
        "SPARSE_FEATURES": ", ".join(sparse_features) if 'sparse_features' in globals() else "Not yet defined",
        "DATA_FILEPATH": "Amazon_Fashion.jsonl"
    }
    
    config_path = os.path.join(output_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write("# Drift Detection Experiment Configuration\n")
        f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for key, value in hyperparams.items():
            f.write(f"{key} = {value}\n")
    
    print(f"Hyperparameters saved to {config_path}")

def run_experiment_for_metric(distance_metric, windows, baseline_window_count, embedding_dim, sparse_features, model, output_dir):
    """Run the drift detection experiment for a specific distance metric."""
    print(f"\nRunning drift detection with {distance_metric} distance metric...")
    
    # Initialize EmbeddingTracker with the specific distance metric
    tracker = EmbeddingTracker(
        embedding_dim=embedding_dim * len(sparse_features),
        distance_name=distance_metric,
        alpha=0.01
    )
    
    # Establish baseline distribution
    baseline_result = establish_baseline(tracker, windows, baseline_window_count)
    initial_threshold = baseline_result["initial_threshold"]
    print(f"Initial drift threshold (baseline): {initial_threshold:.4f}")
    
    # Run drift detection on remaining windows
    drift_result = detect_drift(
        tracker, windows, baseline_window_count, initial_threshold,
        threshold_window=THRESHOLD_WINDOW, 
        threshold_multiplier=THRESHOLD_MULTIPLIER, 
        adaptive_update=ADAPTIVE_UPDATE
    )
    
    # Combine baseline and detection results
    all_distances = baseline_result["distances"] + drift_result["distances"]
    all_thresholds = baseline_result["thresholds"] + drift_result["thresholds"]
    all_labels = baseline_result["labels"] + drift_result["labels"]
    
    # Calculate timestamps for visualization
    window_times = [win.index.mean() for win in windows]
    
    # Save results to CSV
    results_file = save_drift_results(output_dir, window_times, all_distances, all_thresholds, all_labels, distance_metric)
    
    # Use imported plotting function instead of internal one
    plot_drift_results(results_file, output_dir)
    
    return {
        "metric": distance_metric,
        "window_times": window_times,
        "distances": all_distances,
        "thresholds": all_thresholds,
        "labels": all_labels
    }

def main():
    """Main function to run the drift detection experiment."""
    # Create timestamped output directory
    output_dir = create_output_directory()
    
    # Load and preprocess data
    data_filepath = "Amazon_Fashion.jsonl"
    data = load_and_preprocess_data(data_filepath)
    print(f"Loaded {len(data)} records. Data columns: {list(data.columns)}")
    
    # Define features and train model
    global sparse_features, model
    sparse_features = ["asin", "parent_asin", "user_id"]
    target = "rating"
    
    # Save hyperparameters to config file
    save_hyperparameters(output_dir)
    
    model = train_deepfm_model(
        data, sparse_features, target, 
        embedding_dim=EMBEDDING_DIM, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE
    )
    
    # Split data into windows
    windows = split_into_windows(data, N_WINDOWS)
    baseline_window_count = max(1, int(len(windows) * BASELINE_FRACTION))
    print(f"Total windows: {len(windows)}; Using first {baseline_window_count} windows as baseline.")
    
    # Run experiments for each distance metric
    all_results = []
    print(f"Running experiments with the following metrics: {AVAILABLE_DISTANCE_METRICS}")
    for metric in AVAILABLE_DISTANCE_METRICS:
        try:
            result = run_experiment_for_metric(
                metric, windows, baseline_window_count, 
                EMBEDDING_DIM, sparse_features, model, output_dir
            )
            all_results.append(result)
            print(f"Successfully completed experiment for {metric}")
        except Exception as e:
            print(f"Error running experiment for {metric}: {str(e)}")
    
    # Generate comparison plot using imported function
    create_comparison_plot(all_results, output_dir)
    
    print(f"\nAll experiments completed. Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
