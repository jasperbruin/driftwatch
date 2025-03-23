import time
from tqdm import tqdm

from embedding_tracker import EmbeddingTracker, VECTOR_DISTANCE_FUNCTIONS, DISTRIBUTION_METRICS
from utils import *
from config import *
from plot import *

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
    realtime_update=False
):
    """
    Main entry point for distance tracking experiments.
    """

    if distance_name in DISTRIBUTION_METRICS or distance_name in ("wasserstein", "mmd"):
        approaches = ["kll_sketch", "pca_kll_sketch"]
    else:
        approaches = ["no_pca", "pca"]

    all_results = []
    tracker_dict = {}

    for method in approaches:
        if method in ["pca", "pca_kll_sketch"]:
            embedding_dim = pca_components
        else:
            embedding_dim = baseline_embs.shape[1]

        tracker_dict[method] = EmbeddingTracker(
            embedding_dim=embedding_dim,
            alpha=0.01,
            distance_name=distance_name,
            k=args.get("kll_k", 20),
            num_bins=args.get("kll_bins", 20)
        )

    for method in approaches:
        tracker = tracker_dict[method]
        for batch in batch_generator(baseline_texts, batch_size):
            emb = extract_embeddings(model, tokenizer, batch, device)
            if method in ["pca", "pca_kll_sketch"]:
                emb = pca.transform(emb)
            tracker.update(emb)

    for method in approaches:
        tracker = tracker_dict[method]
        distance_scores = []
        overhead_times = []

        start_time = time.time()
        for batch in tqdm(batch_generator(test_texts, batch_size), leave=False):
            emb = extract_embeddings(model, tokenizer, batch, device)
            if method in ["pca", "pca_kll_sketch"]:
                emb = pca.transform(emb)

            overhead_start = time.time()
            dist = tracker.compute_distance(emb)
            overhead_end = time.time()

            distance_scores.append(dist)
            overhead_times.append(overhead_end - overhead_start)

            if realtime_update:
                tracker.update(emb)

        end_time = time.time()
        final_dist = distance_scores[-1] if distance_scores else 0.0
        total_time = end_time - start_time
        avg_overhead = np.mean(overhead_times) if overhead_times else 0.0

        all_results.append((method, final_dist, total_time, avg_overhead))

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
    seed=None
):
    partial_results = []
    all_distance_names = ["mahalanobis"] + list(VECTOR_DISTANCE_FUNCTIONS.keys()) \
                         + ["kl", "js", "hellinger", "bhattacharyya", "mmd", "wasserstein"]

    for distance_name in all_distance_names:
        if distance_name in DISTRIBUTION_METRICS or distance_name in ("wasserstein", "mmd"):
            distance_type = "distribution"
        else:
            distance_type = "vector"

        for drift_strength in drift_strengths:
            drifted_texts = introduce_gradual_drift(drift_texts, fraction_shuffle=drift_strength)
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

            for (method, final_dist, total_time, avg_overhead) in results:
                partial_results.append({
                    "distance_type": distance_type,
                    "distance_name": distance_name,
                    "drift_strength": drift_strength,
                    "pca_applied": method in ["pca", "pca_kll_sketch"],
                    "final_similarity": final_dist,
                    "time_taken": total_time,
                    "avg_overhead": avg_overhead,
                    "seed": seed,
                })

    return partial_results


def collect_data_single_seed(seed, args):
    set_seed(seed)
    device = get_device()
    print(f"[Seed={seed}] Using device:", device)

    results = defaultdict(list)
    for dataset_info in args["datasets"]:
        dataset_name, baseline_texts, drift_texts = load_and_split_texts(dataset_info, args["max_texts"])

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
                seed=seed
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
