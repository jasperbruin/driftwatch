# drift_detection.py

import os
import time
import tracemalloc
from collections import defaultdict

import numpy as np

# 1. Import the KLL-Floats-Sketch
from tqdm import tqdm

from config import args
from embedding_tracker import (
    DISTRIBUTION_METRICS,
    VECTOR_DISTANCE_FUNCTIONS,
    EmbeddingTracker,
)
from plot import *
from utils import *


def run_distance_tracking(
    model,
    tokenizer,
    baseline_texts,
    test_texts,
    baseline_embs,
    pca,
    distance_name,
    pca_components,
    batch_size,
    device,
    realtime_update=False,
):
    """Main entry point for distance tracking experiments.
    This function demonstrates how to toggle between:
      - No dimensionality reduction
      - PCA-based dimensionality reduction
      - KLL-based dimensionality reduction
    and measure memory/time usage.
    """
    # 2. Decide which "approaches" to run for a given distance_name.
    if distance_name in DISTRIBUTION_METRICS or distance_name in ("wasserstein", "mmd"):
        # Distribution-based approaches
        approaches = ["kll_sketch", "histogram", "pca_kll_sketch", "pca_histogram"]
    else:
        # Vector-based approaches
        approaches = ["no_pca", "pca", "kll_vector"]

    all_results = []
    tracker_dict = {}

    # 3. Initialize the trackers for each approach.
    for method in approaches:
        if method in ["pca", "pca_kll_sketch", "pca_histogram"]:
            embedding_dim = pca_components
        elif method == "kll_vector":
            # Use the same dimension as the KLL 'k' parameter
            embedding_dim = args.get("kll_k", 8)
        else:
            embedding_dim = baseline_embs.shape[1]

        if method in ["no_pca", "pca", "kll_vector"]:
            distribution_impl = "none"
        elif method in ["kll_sketch", "pca_kll_sketch"]:
            distribution_impl = "kll"
        elif method in ["histogram", "pca_histogram"]:
            distribution_impl = "histogram"
        else:
            raise ValueError(f"Unknown method: {method}")

        tracker_dict[method] = EmbeddingTracker(
            embedding_dim=embedding_dim,
            alpha=0.01,
            distance_name=distance_name,
            k=args.get("kll_k", 20),
            num_bins=args.get("kll_bins", 20),
            distribution_impl=distribution_impl,
        )

    # 4. Update the tracker with baseline embeddings.
    for method in approaches:
        tracker = tracker_dict[method]

        for batch in batch_generator(baseline_texts, batch_size):
            emb = extract_embeddings(model, tokenizer, batch, device)

            if method in ["pca", "pca_kll_sketch", "pca_histogram"]:
                emb = pca.transform(emb)

            if method == "kll_vector":
                # Convert to size-(k) vectors
                emb = kll_transform(emb, k=args.get("kll_k", 8))

            tracker.update(emb)

    tracemalloc.start()

    # 5. Compute distance for the test data with each approach.
    for method in approaches:
        tracker = tracker_dict[method]
        distance_scores = []
        overhead_times = []
        memory_usages = []

        start_time = time.time()

        for batch in tqdm(batch_generator(test_texts, batch_size), leave=False):
            emb = extract_embeddings(model, tokenizer, batch, device)

            if method in ["pca", "pca_kll_sketch", "pca_histogram"]:
                emb = pca.transform(emb)

            if method == "kll_vector":
                emb = kll_transform(emb, k=args.get("kll_k", 8))

            overhead_start = time.time()
            dist = tracker.compute_distance(emb)
            overhead_end = time.time()

            current_mem, _ = tracemalloc.get_traced_memory()
            memory_usages.append(current_mem)

            distance_scores.append(dist)
            overhead_times.append(overhead_end - overhead_start)

        end_time = time.time()
        final_dist = distance_scores[-1] if distance_scores else 0.0
        total_time = end_time - start_time
        avg_overhead = np.mean(overhead_times) if overhead_times else 0.0
        avg_memory = np.mean(memory_usages) / (1024**2)  # MB

        all_results.append((method, final_dist, total_time, avg_overhead, avg_memory))

    tracemalloc.stop()

    return all_results


def run_experiments_for_model(
    model_name,
    baseline_texts,
    drift_texts,
    device,
    pca_components,
    batch_size,
    drift_strengths,
    baseline_embs,
    pca,
    seed=None,
):
    partial_results = []

    all_distance_names = (
        ["mahalanobis"]
        + list(VECTOR_DISTANCE_FUNCTIONS.keys())
        + ["kl", "js", "hellinger", "bhattacharyya", "mmd", "wasserstein"]
    )

    for distance_name in all_distance_names:
        if distance_name in DISTRIBUTION_METRICS or distance_name in (
            "wasserstein",
            "mmd",
        ):
            distance_type = "distribution"
        else:
            distance_type = "vector"

        for drift_strength in drift_strengths:
            drifted_texts = introduce_gradual_drift(
                drift_texts, fraction_shuffle=drift_strength
            )
            test_texts = baseline_texts + drifted_texts

            results = run_distance_tracking(
                model_name["model"],
                model_name["tokenizer"],
                baseline_texts,
                test_texts,
                baseline_embs,
                pca,
                distance_name,
                pca_components,
                batch_size,
                device,
            )

            for method, final_dist, total_time, avg_overhead, avg_memory in results:
                partial_results.append(
                    {
                        "distance_type": distance_type,
                        "distance_name": distance_name,
                        "drift_strength": drift_strength,
                        "pca_applied": method
                        in ["pca", "pca_kll_sketch", "pca_histogram"],
                        "method": method,
                        "final_similarity": final_dist,
                        "time_taken": total_time,
                        "avg_overhead": avg_overhead,
                        "avg_memory_mb": avg_memory,
                        "seed": seed,
                    }
                )

    return partial_results


def collect_data_single_seed(seed, args):
    set_seed(seed)
    device = get_device()
    print(f"[Seed={seed}] Using device:", device)

    results = defaultdict(list)
    for dataset_info in args["datasets"]:
        dataset_name, baseline_texts, drift_texts = load_and_split_texts(
            dataset_info, args["max_texts"]
        )

        for model_name in args["models"]:
            print(f"[Seed={seed}] --- Using Model: {model_name} ---")

            model, tokenizer, baseline_embs, pca = compute_baseline_embeddings_and_pca(
                model_name,
                baseline_texts,
                device,
                args["pca_components"],
                args["batch_size"],
            )

            model_details = {
                "model": model,
                "tokenizer": tokenizer,
            }

            partial_results = run_experiments_for_model(
                model_details,
                baseline_texts,
                drift_texts,
                device,
                args["pca_components"],
                args["batch_size"],
                args["drift_strengths"],
                baseline_embs,
                pca,
                seed=seed,
            )
            for r in partial_results:
                key = (dataset_name, model_name)
                results[key].append(r)

    return results


def collect_data_multiple_seeds():
    all_results = defaultdict(list)
    for seed in range(args["num_seeds"]):
        seed_results = collect_data_single_seed(seed, args)
        for key, records in seed_results.items():
            all_results[key].extend(records)
    print("\nAll seeds complete!")
    return all_results


def main():
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args["output_dir"], timestamp)

    results = collect_data_multiple_seeds()
    save_results(results, output_dir)

    run_all_results("data")


if __name__ == "__main__":
    main()
