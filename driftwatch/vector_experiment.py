import argparse
import os
import time
import tracemalloc
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from embedding_tracker import EmbeddingTracker
from metrics import VECTOR_DISTANCE_FUNCTIONS
from utils import (
    batch_generator,
    compute_baseline_embeddings_and_pca,
    extract_embeddings,
    get_device,
    introduce_gradual_drift,
    kll_transform,
    load_and_split_texts,
    save_results,
    set_seed,
)


def parse_args():
    """Get configuration from config.py and allow command-line arguments to override"""
    from config import args as config_args

    parser = argparse.ArgumentParser(
        description="Vector-based Drift Detection Experiment"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=config_args["models"],
        help="List of model names to evaluate",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[d["name"] for d in config_args["datasets"]],
        help="Names of datasets to use",
    )
    parser.add_argument(
        "--max_texts",
        type=int,
        default=config_args["max_texts"],
        help="Maximum number of texts to process per dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config_args["batch_size"],
        help="Batch size for processing",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=config_args["pca_components"],
        help="Number of PCA components",
    )
    parser.add_argument(
        "--kll_k",
        type=int,
        default=config_args.get("kll_k", 50),
        help="KLL parameter k (output dimension)",
    )
    parser.add_argument(
        "--drift_strengths",
        type=float,
        nargs="+",
        default=config_args["drift_strengths"],
        help="Drift strength values to test",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(config_args["output_dir"], "vector_experiment"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=config_args["num_seeds"],
        help="Number of random seeds to run",
    )

    return parser.parse_args()


def run_vector_experiment(
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
    kll_k,
):
    """Run vector-based distance tracking experiment with different dimension reduction approaches."""
    # For vector-based approaches, we only use these methods
    approaches = ["no_pca", "pca", "kll_vector"]

    all_results = []
    tracker_dict = {}

    # Initialize the trackers for each approach
    for method in approaches:
        if method == "pca":
            embedding_dim = pca_components
        elif method == "kll_vector":
            embedding_dim = kll_k
        else:  # no_pca
            embedding_dim = baseline_embs.shape[1]

        tracker_dict[method] = EmbeddingTracker(
            embedding_dim=embedding_dim,
            alpha=0.01,
            distance_name=distance_name,
            k=kll_k,
            distribution_impl="none",  # This is a vector-based approach
        )

    # Update the trackers with baseline embeddings
    for method in approaches:
        tracker = tracker_dict[method]

        for batch in batch_generator(baseline_texts, batch_size):
            emb = extract_embeddings(model, tokenizer, batch, device)

            if method == "pca":
                emb = pca.transform(emb)
            elif method == "kll_vector":
                emb = kll_transform(emb, k=kll_k)

            tracker.update(emb)

    # Start memory tracking
    tracemalloc.start()

    # Compute distance for test data with each approach
    for method in approaches:
        tracker = tracker_dict[method]
        distance_scores = []
        overhead_times = []
        memory_usages = []
        peak_memory_usages = []

        start_time = time.time()

        for batch in tqdm(batch_generator(test_texts, batch_size), leave=False):
            emb = extract_embeddings(model, tokenizer, batch, device)

            if method == "pca":
                emb = pca.transform(emb)
            elif method == "kll_vector":
                emb = kll_transform(emb, k=kll_k)

            # Clear tracemalloc statistics before distance computation
            tracemalloc.clear_traces()

            # Measure overhead time for distance computation
            overhead_start = time.time()
            dist = tracker.compute_distance(emb)
            overhead_end = time.time()

            # Track memory usage - now getting both current and peak
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            memory_usages.append(current_mem)
            peak_memory_usages.append(peak_mem)

            distance_scores.append(dist)
            overhead_times.append(overhead_end - overhead_start)

        end_time = time.time()
        final_dist = distance_scores[-1] if distance_scores else 0.0
        total_time = end_time - start_time
        avg_overhead = np.mean(overhead_times) if overhead_times else 0.0
        avg_memory = np.mean(memory_usages) / (1024**2)  # MB
        avg_peak_memory = np.mean(peak_memory_usages) / (1024**2)  # MB

        all_results.append(
            (method, final_dist, total_time, avg_overhead, avg_memory, avg_peak_memory)
        )

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
    kll_k,
    seed=None,
):
    partial_results = []

    # Only using vector-based distance functions for this experiment
    vector_distance_names = ["mahalanobis"] + list(VECTOR_DISTANCE_FUNCTIONS.keys())

    for distance_name in vector_distance_names:
        for drift_strength in drift_strengths:
            drifted_texts = introduce_gradual_drift(
                drift_texts, fraction_shuffle=drift_strength
            )
            test_texts = baseline_texts + drifted_texts

            results = run_vector_experiment(
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
                kll_k,
            )

            for result_tuple in results:
                if len(result_tuple) == 6:  # Updated version with peak memory
                    (
                        method,
                        final_dist,
                        total_time,
                        avg_overhead,
                        avg_memory,
                        avg_peak_memory,
                    ) = result_tuple
                    peak_memory = avg_peak_memory
                else:  # Backward compatibility
                    method, final_dist, total_time, avg_overhead, avg_memory = (
                        result_tuple
                    )
                    peak_memory = avg_memory

                partial_results.append(
                    {
                        "distance_type": "vector",
                        "distance_name": distance_name,
                        "drift_strength": drift_strength,
                        "pca_applied": method == "pca",
                        "kll_applied": method == "kll_vector",
                        "method": method,
                        "final_similarity": final_dist,
                        "time_taken": total_time,
                        "avg_overhead": avg_overhead,
                        "avg_memory_mb": avg_memory,
                        "peak_memory_mb": peak_memory,
                        "seed": seed,
                    }
                )

    return partial_results


def collect_data_single_seed(seed, args):
    set_seed(seed)
    device = get_device()
    print(f"[Seed={seed}] Using device:", device)

    results = defaultdict(list)

    for dataset_name in args.datasets:
        # Create a dataset info structure for each dataset
        dataset_info = {
            "name": dataset_name,
            "config": None,
            "split": "train",
            "text_column": "text",
        }

        dataset_name, baseline_texts, drift_texts = load_and_split_texts(
            dataset_info, args.max_texts
        )

        for model_name in args.models:
            print(f"[Seed={seed}] --- Using Model: {model_name} ---")

            model, tokenizer, baseline_embs, pca = compute_baseline_embeddings_and_pca(
                model_name,
                baseline_texts,
                device,
                args.pca_components,
                args.batch_size,
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
                args.pca_components,
                args.batch_size,
                args.drift_strengths,
                baseline_embs,
                pca,
                args.kll_k,
                seed=seed,
            )

            for r in partial_results:
                key = (dataset_name, model_name)
                results[key].append(r)

    return results


def collect_data_multiple_seeds(args):
    all_results = defaultdict(list)
    for seed in range(args.num_seeds):
        seed_results = collect_data_single_seed(seed, args)
        for key, records in seed_results.items():
            all_results[key].extend(records)
    print("\nAll seeds complete!")
    return all_results


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Add timestamp to output directory
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, timestamp)

    print("Vector-based Drift Detection Experiment")
    print("======================================")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Output directory: {output_dir}")
    print(f"PCA components: {args.pca_components}")
    print(f"KLL k: {args.kll_k}")
    print(f"Drift strengths: {args.drift_strengths}")

    results = collect_data_multiple_seeds(args)
    save_results(results, output_dir)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
