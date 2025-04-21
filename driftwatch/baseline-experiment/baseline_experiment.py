import os
import time
import tracemalloc

import numpy as np
import pandas as pd
import torch
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
from results.plot_drift_results import (
    create_comparison_plot,
    plot_drift_results,
    plot_memory_usage,
)
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from embedding_tracker import EmbeddingTracker
from metrics import get_available_metrics

# Configuration constants
THRESHOLD_MULTIPLIER = 2.5
AVAILABLE_DISTANCE_METRICS = (
    get_available_metrics()
)  # Get all metrics from the metrics module
BASE_OUTPUT_DIR = "baseline_results"
N_WINDOWS = 300
BASELINE_FRACTION = 0.3
EMBEDDING_DIM = 64
EPOCHS = 2
BATCH_SIZE = 1024
THRESHOLD_WINDOW = 30
ADAPTIVE_UPDATE = False

# Define which metrics are vector-based vs distribution-based
VECTOR_METRICS = [
    "cosine",
    "euclidean",
    "manhattan",
    "minkowski",
    "mahalanobis",
    "chebyshev",
    "canberra",
]
DISTRIBUTION_METRICS = [
    "wasserstein",
    "ks",
    "kl",
    "js",
    "hellinger",
    "bhattacharyya",
    "mmd",
]

# Distribution implementations to test
DIST_IMPLEMENTATIONS = ["kll", "histogram"]


# Memory profiling functions
def start_memory_tracking():
    """Start tracking memory usage."""
    tracemalloc.start()
    return tracemalloc.take_snapshot()


def get_memory_usage(baseline_snapshot=None):
    """Take a memory snapshot and return statistics.
    If baseline_snapshot is provided, return difference from baseline.
    """
    snapshot = tracemalloc.take_snapshot()
    if baseline_snapshot:
        stats = snapshot.compare_to(baseline_snapshot, "lineno")
    else:
        stats = snapshot.statistics("lineno")

    # Get current and peak memory usage
    current, peak = tracemalloc.get_traced_memory()

    return {
        "snapshot": snapshot,
        "stats": stats,
        "current_memory": current / (1024 * 1024),  # MB
        "peak_memory": peak / (1024 * 1024),  # MB
    }


def stop_memory_tracking():
    """Stop memory tracking."""
    tracemalloc.stop()


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
    data = data[data["timestamp"] > "2015-01-01"]
    # Set timestamp as index and sort chronologically
    data = data.set_index("timestamp")
    data.sort_index(inplace=True)
    return data


def train_deepfm_model(
    data: pd.DataFrame,
    sparse_features: list,
    target: str,
    embedding_dim: int = 64,
    epochs: int = 2,
    batch_size: int = 1024,
):
    """Train a DeepFM model on the given data for the binary classification task.
    Returns the trained model and feature columns.
    """
    # Label encode categorical features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # Define feature columns for DeepFM
    fixlen_feature_columns = [
        SparseFeat(
            feat, vocabulary_size=data[feat].nunique(), embedding_dim=embedding_dim
        )
        for feat in sparse_features
    ]
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

    model = DeepFM(
        linear_feature_columns, dnn_feature_columns, task="binary", device=device
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])

    # Train the model
    model.fit(
        model_input,
        data[target].values,
        batch_size=batch_size,
        epochs=epochs,
        verbose=3,
        validation_split=0.1,
    )

    # Switch model to evaluation mode for inference
    model.eval()
    return model, linear_feature_columns, dnn_feature_columns


def split_into_windows(data: pd.DataFrame, n_windows: int):
    """Split the data into n_windows sequential time-ordered segments (windows).
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
    """Extract the concatenated embeddings for all samples in batch_df using the trained model.
    Returns a NumPy array of shape (num_samples, total_embedding_dim).
    """
    # Prepare model input for the batch
    batch_input = {feat: batch_df[feat].values for feat in sparse_features}

    # Forward pass (ensure no gradient computation)
    with torch.no_grad():
        _ = model.predict(
            batch_input
        )  # run through model to ensure embeddings are materialized

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


def evaluate_model_accuracy(
    model,
    window_df,
    sparse_features,
    target,
    linear_feature_columns,
    dnn_feature_columns,
):
    """Evaluate the model's prediction accuracy on a window of data.
    Returns accuracy score and AUC score.
    """
    # Prepare model input
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    model_input = {name: window_df[name].values for name in feature_names}

    # Get predictions
    with torch.no_grad():
        y_pred = model.predict(model_input)

    # Convert to binary predictions (threshold at 0.5)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(window_df[target].values, y_pred_binary)

    # Calculate AUC if possible (requires both classes to be present)
    try:
        auc = roc_auc_score(window_df[target].values, y_pred)
    except ValueError:
        # If only one class is present in the window, AUC is undefined
        auc = np.nan

    return accuracy, auc


def establish_baseline(
    embedding_tracker: EmbeddingTracker,
    windows: list,
    baseline_count: int,
    model=None,
    sparse_features=None,
    target=None,
    linear_feature_columns=None,
    dnn_feature_columns=None,
):
    """Use the first baseline_count windows to establish the baseline distribution in the embedding tracker.
    Returns a dictionary with baseline drift distances and the initial threshold.
    """
    baseline_distances = []
    baseline_accuracies = []
    baseline_aucs = []

    # Iterate over baseline windows
    for i in range(baseline_count):
        window_df = windows[i]
        emb = extract_embeddings(model, window_df, sparse_features)
        # If using vector-based distance, use the mean embedding; if distribution-based, use full embeddings
        emb_input = (
            emb if embedding_tracker.is_distribution_mode else np.mean(emb, axis=0)
        )

        # Evaluate model accuracy if model and target are provided
        if model is not None and target is not None:
            accuracy, auc = evaluate_model_accuracy(
                model,
                window_df,
                sparse_features,
                target,
                linear_feature_columns,
                dnn_feature_columns,
            )
            baseline_accuracies.append(float(accuracy))
            baseline_aucs.append(float(auc))

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
        baseline_dist_array = np.array(
            baseline_distances[1:]
        )  # exclude the initial 0.0
        initial_threshold = np.mean(
            baseline_dist_array
        ) + THRESHOLD_MULTIPLIER * np.std(baseline_dist_array)
    else:
        initial_threshold = float(
            baseline_distances[0]
        )  # if only one baseline window, use its (zero) distance

    # For baseline period, we consider no drift occurred, so labels are 0
    baseline_labels = [0] * len(baseline_distances)
    # Also prepare a threshold list for baseline windows (initial_threshold applied throughout baseline)
    baseline_thresholds = [initial_threshold] * len(baseline_distances)

    return {
        "distances": baseline_distances,
        "thresholds": baseline_thresholds,
        "labels": baseline_labels,
        "initial_threshold": initial_threshold,
        "accuracies": baseline_accuracies,
        "aucs": baseline_aucs,
    }


def detect_drift(
    embedding_tracker: EmbeddingTracker,
    windows: list,
    baseline_count: int,
    initial_threshold: float,
    threshold_window: int = 50,
    threshold_multiplier: float = 3.0,
    adaptive_update: bool = False,
    model=None,
    sparse_features=None,
    target=None,
    linear_feature_columns=None,
    dnn_feature_columns=None,
):
    """Compute drift distances for windows beyond the baseline period and detect drift events.
    Uses a rolling window of past `threshold_window` drift scores to update the threshold.
    If adaptive_update is True, update the baseline tracker with windows that are NOT flagged as drift.
    Returns a dict with drift distances, thresholds, and drift labels for the post-baseline windows.
    """
    drift_distances = []
    thresholds = []
    drift_labels = []
    accuracies = []
    aucs = []

    # Iterate through each window after the baseline period
    for j in range(baseline_count, len(windows)):
        window_df = windows[j]
        emb = extract_embeddings(model, window_df, sparse_features)
        emb_input = (
            emb if embedding_tracker.is_distribution_mode else np.mean(emb, axis=0)
        )

        # Evaluate model accuracy if model and target are provided
        if model is not None and target is not None:
            accuracy, auc = evaluate_model_accuracy(
                model,
                window_df,
                sparse_features,
                target,
                linear_feature_columns,
                dnn_feature_columns,
            )
            accuracies.append(float(accuracy))
            aucs.append(float(auc))

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
        "labels": drift_labels,
        "accuracies": accuracies,
        "aucs": aucs,
    }


def save_drift_results(
    output_dir,
    window_times,
    all_distances,
    all_thresholds,
    all_labels,
    distance_metric,
    memory_metrics=None,
    all_accuracies=None,
    all_aucs=None,
):
    """Save drift detection results to CSV with memory metrics and accuracy metrics."""
    # Create base results dictionary
    results_dict = {
        "window_time": window_times,
        "drift_distance": all_distances,
        "threshold": all_thresholds,
        "drift_detected": all_labels,
    }

    # Add accuracy metrics if provided
    if all_accuracies is not None:
        results_dict["accuracy"] = all_accuracies
    if all_aucs is not None:
        results_dict["auc"] = all_aucs

    # Save drift distances and thresholds to disk as a CSV
    results_df = pd.DataFrame(results_dict)

    output_file = os.path.join(
        output_dir, f"drift_detection_results_{distance_metric}.csv"
    )
    results_df.to_csv(output_file, index=False)
    print(f"Drift results saved to '{output_file}'.")

    # Save memory metrics
    if memory_metrics:
        memory_file = os.path.join(output_dir, f"memory_metrics_{distance_metric}.csv")
        memory_df = pd.DataFrame(
            {
                "metric": [distance_metric],
                "init_memory_mb": [memory_metrics["init_memory"]],
                "baseline_memory_mb": [memory_metrics["baseline_memory"]],
                "final_memory_mb": [memory_metrics["final_memory"]],
                "peak_memory_mb": [memory_metrics["peak_memory"]],
            }
        )
        memory_df.to_csv(memory_file, index=False)
        print(f"Memory metrics saved to '{memory_file}'.")

    return output_file


def run_experiment_for_metric(
    distance_metric,
    windows,
    baseline_window_count,
    embedding_dim,
    sparse_features,
    model,
    output_dir,
    linear_feature_columns,
    dnn_feature_columns,
    target="rating",
):
    """Run the drift detection experiment for a specific distance metric with memory profiling."""
    # Check if this is a distribution-based metric
    is_distribution_metric = any(
        dm.lower() in distance_metric.lower() for dm in DISTRIBUTION_METRICS
    )

    # For vector-based metrics, run once; for distribution-based metrics, run for each implementation
    implementations_to_run = (
        DIST_IMPLEMENTATIONS if is_distribution_metric else ["default"]
    )

    results = []

    for impl in implementations_to_run:
        # For distribution metrics, include implementation in the metric name
        metric_name = distance_metric
        if is_distribution_metric:
            metric_name = f"{distance_metric}_{impl}"

        print(f"\nRunning drift detection with {metric_name} distance metric...")

        # Start memory tracking
        baseline_snapshot = start_memory_tracking()

        # Initialize EmbeddingTracker with the specific distance metric and implementation
        distribution_impl = (
            impl if is_distribution_metric else "kll"
        )  # Default to KLL for vector metrics

        tracker = EmbeddingTracker(
            embedding_dim=embedding_dim * len(sparse_features),
            distance_name=distance_metric,
            alpha=0.01,
            distribution_impl=distribution_impl,
        )

        # Measure memory after tracker initialization
        init_memory = get_memory_usage(baseline_snapshot)
        print(
            f"Memory after tracker initialization: {init_memory['current_memory']:.2f} MB"
        )

        # Establish baseline distribution
        baseline_result = establish_baseline(
            tracker,
            windows,
            baseline_window_count,
            model=model,
            sparse_features=sparse_features,
            target=target,
            linear_feature_columns=linear_feature_columns,
            dnn_feature_columns=dnn_feature_columns,
        )
        initial_threshold = baseline_result["initial_threshold"]
        print(f"Initial drift threshold (baseline): {initial_threshold:.4f}")

        # Measure memory after baseline establishment
        baseline_memory = get_memory_usage(baseline_snapshot)
        print(
            f"Memory after baseline establishment: {baseline_memory['current_memory']:.2f} MB"
        )

        # Run drift detection on remaining windows
        drift_result = detect_drift(
            tracker,
            windows,
            baseline_window_count,
            initial_threshold,
            threshold_window=THRESHOLD_WINDOW,
            threshold_multiplier=THRESHOLD_MULTIPLIER,
            adaptive_update=ADAPTIVE_UPDATE,
            model=model,
            sparse_features=sparse_features,
            target=target,
            linear_feature_columns=linear_feature_columns,
            dnn_feature_columns=dnn_feature_columns,
        )

        # Measure memory after drift detection
        drift_memory = get_memory_usage(baseline_snapshot)
        print(
            f"Memory after drift detection: {drift_memory['current_memory']:.2f} MB (Peak: {drift_memory['peak_memory']:.2f} MB)"
        )

        # Combine baseline and detection results
        all_distances = baseline_result["distances"] + drift_result["distances"]
        all_thresholds = baseline_result["thresholds"] + drift_result["thresholds"]
        all_labels = baseline_result["labels"] + drift_result["labels"]

        # Combine accuracy metrics if available
        all_accuracies = None
        all_aucs = None
        if "accuracies" in baseline_result and "accuracies" in drift_result:
            all_accuracies = baseline_result["accuracies"] + drift_result["accuracies"]
            all_aucs = baseline_result["aucs"] + drift_result["aucs"]

        # Calculate timestamps for visualization
        window_times = [win.index.mean() for win in windows]

        # Store memory metrics
        memory_metrics = {
            "init_memory": init_memory["current_memory"],
            "baseline_memory": baseline_memory["current_memory"],
            "final_memory": drift_memory["current_memory"],
            "peak_memory": drift_memory["peak_memory"],
        }

        # Save results to CSV (including memory metrics and accuracy)
        results_file = save_drift_results(
            output_dir,
            window_times,
            all_distances,
            all_thresholds,
            all_labels,
            metric_name,
            memory_metrics,
            all_accuracies,
            all_aucs,
        )

        # Use imported plotting functions
        plot_drift_results(results_file, output_dir)

        # Stop memory tracking
        stop_memory_tracking()

        results.append(
            {
                "metric": metric_name,
                "window_times": window_times,
                "distances": all_distances,
                "thresholds": all_thresholds,
                "labels": all_labels,
                "accuracies": all_accuracies,
                "aucs": all_aucs,
                "memory_metrics": memory_metrics,
                "distribution_impl": distribution_impl
                if is_distribution_metric
                else "vector-based",
            }
        )

    return results


def main():
    """Main function to run the drift detection experiment."""
    # Create timestamped output directory
    output_dir = create_output_directory()

    # Load and preprocess data
    data_filepath = "data/Amazon_Fashion.jsonl"
    data = load_and_preprocess_data(data_filepath)
    print(f"Loaded {len(data)} records. Data columns: {list(data.columns)}")

    # Define features and train model
    sparse_features = ["asin", "parent_asin", "user_id"]
    target = "rating"

    # Save hyperparameters to config file
    # save_hyperparameters(output_dir)

    model, linear_feature_columns, dnn_feature_columns = train_deepfm_model(
        data,
        sparse_features,
        target,
        embedding_dim=EMBEDDING_DIM,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # Split data into windows
    windows = split_into_windows(data, N_WINDOWS)
    baseline_window_count = max(1, int(len(windows) * BASELINE_FRACTION))
    print(
        f"Total windows: {len(windows)}; Using first {baseline_window_count} windows as baseline."
    )

    # Run experiments for each distance metric
    all_results = []
    print(
        f"Running experiments with the following metrics: {AVAILABLE_DISTANCE_METRICS}"
    )
    for metric in AVAILABLE_DISTANCE_METRICS:
        try:
            results = run_experiment_for_metric(
                metric,
                windows,
                baseline_window_count,
                EMBEDDING_DIM,
                sparse_features,
                model,
                output_dir,
                linear_feature_columns,
                dnn_feature_columns,
                target=target,
            )
            all_results.extend(results)
            print(f"Successfully completed experiment for {metric}")
        except Exception as e:
            print(f"Error running experiment for {metric}: {str(e)}")

    # Generate comparison plots
    create_comparison_plot(all_results, output_dir)
    plot_memory_usage(output_dir)  # Memory usage plot

    print(f"\nAll experiments completed. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
