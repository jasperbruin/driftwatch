import random
import numpy as np
import time
import os
import json
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
from sklearn.decomposition import PCA
from embedding_tracker import EmbeddingTracker, DISTANCE_FUNCTIONS
from utils import set_seed, extract_embeddings, batch_generator, get_device, introduce_gradual_drift
from config import args
from collections import defaultdict
from plot_time_overhead import process_all_json_time_overhead
from plot_all_scores import process_all_scores

def collect_data_single_seed(seed):
    """Run the entire process once for a given seed."""
    set_seed(seed)

    # Detect device
    device = get_device()
    print(f"[Seed={seed}] Using device:", device)

    results = defaultdict(list)

    for dataset_info in args["datasets"]:
        dataset_name = dataset_info["name"]
        dataset_config = dataset_info["config"]
        dataset_split = dataset_info["split"]
        text_col = dataset_info["text_column"]

        print(f"[Seed={seed}] === Loading dataset: {dataset_name} ===")
        ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
        texts = list(ds[text_col])
        random.shuffle(texts)
        if args["max_texts"] > 0 and len(texts) > args["max_texts"]:
            texts = texts[:args["max_texts"]]

        half_point = len(texts) // 2
        baseline_texts = texts[:half_point]
        drift_texts = texts[half_point:]

        for model_name in args["models"]:
            print(f"[Seed={seed}] --- Using Model: {model_name} ---")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            model.eval()

            # Precompute baseline embeddings
            baseline_embs = np.concatenate([extract_embeddings(model, tokenizer, b, device) for b in batch_generator(baseline_texts, args["batch_size"])], axis=0)

            # Fit PCA on the baseline embeddings
            pca = PCA(n_components=args["pca_components"])
            pca.fit(baseline_embs)

            key = (dataset_name, model_name)
            all_distance_names = ["mahalanobis"] + list(DISTANCE_FUNCTIONS.keys())

            for distance_name in all_distance_names:
                for drift_strength in args["drift_strengths"]:
                    drifted_texts = introduce_gradual_drift(drift_texts, fraction_shuffle=drift_strength)
                    test_texts = baseline_texts + drifted_texts

                    for use_pca in [False, True]:
                        embedding_dim = args["pca_components"] if use_pca else baseline_embs.shape[1]
                        tracker = EmbeddingTracker(embedding_dim, alpha=0.01, distance_name=distance_name)
                        distance_scores = []
                        time_overhead = []

                        # Initialize tracker with baseline distribution
                        for batch in batch_generator(baseline_texts, args["batch_size"]):
                            emb = extract_embeddings(model, tokenizer, batch, device)
                            emb = pca.transform(emb) if use_pca else emb
                            tracker.update(emb.mean(axis=0))

                        start_time = time.time()
                        for batch in tqdm(batch_generator(test_texts, args["batch_size"]), leave=False):
                            emb = extract_embeddings(model, tokenizer, batch, device)
                            emb = pca.transform(emb) if use_pca else emb
                            mean_emb = emb.mean(axis=0)
                            overhead_start = time.time()
                            dist = tracker.compute_distance(mean_emb)
                            overhead_end = time.time()
                            distance_scores.append(dist)
                            time_overhead.append(overhead_end - overhead_start)
                            tracker.update(mean_emb)
                        end_time = time.time()

                        final_dist = distance_scores[-1] if distance_scores else 0.0
                        total_time = end_time - start_time
                        avg_overhead = np.mean(time_overhead)

                        results[key].append({
                            "distance_name": distance_name,
                            "drift_strength": drift_strength,
                            "pca": use_pca,
                            "final_similarity": final_dist,
                            "time_taken": total_time,
                            "avg_overhead": avg_overhead,
                            "seed": seed,
                        })

    return results


def collect_data_multiple_seeds():
    all_results = defaultdict(list)

    for seed in range(args["num_seeds"]):
        seed_results = collect_data_single_seed(seed)
        for key, records in seed_results.items():
            all_results[key].extend(records)

    print("\nAll seeds complete!")
    return all_results


def save_results(results, output_dir):
    if not results:
        print("No results to save.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Convert defaultdict to a regular dictionary and convert tuple keys to separate dataset and model keys
    results_dict = {}
    for (dataset, model), records in results.items():
        if dataset not in results_dict:
            results_dict[dataset] = {}
        results_dict[dataset][model] = records

    # Convert numpy.float32 to native Python float
    def convert_to_native(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj

    results_dict = convert_to_native(results_dict)

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as json_file:
        json.dump(results_dict, json_file, indent=4)
    print(f"Results saved to {json_path}")

    args_path = os.path.join(output_dir, "args.txt")
    with open(args_path, "w") as f:
        for key, value in args.items():
            f.write(f"{key}: {value}\n")
    print(f"Arguments saved to {args_path}")

def main():
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args["output_dir"], timestamp)

    results = collect_data_multiple_seeds()
    save_results(results, output_dir)

    process_all_json_time_overhead("data")
    process_all_scores("data")

if __name__ == "__main__":
    main()