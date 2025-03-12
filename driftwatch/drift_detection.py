import time
from tqdm import tqdm

from embedding_tracker import *
from utils import *
from config import *
from plot import *

from kll_transform import KLLTransformer


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
):
    """
    Initialize the EmbeddingTracker with the baseline distribution and measure
    distance on the test_texts for three approaches:
    1) no_pca
    2) pca
    3) kll_sketch
    """
    approaches = ["no_pca", "pca", "kll_sketch"]
    all_results = []

    kll_transformer = KLLTransformer(k=args["kll_k"])
    kll_transformer.fit(baseline_embs)  # uses all baseline embeddings

    for method in approaches:
        # Decide embedding_dim
        if method == "pca":
            embedding_dim = pca_components
        else:
            embedding_dim = baseline_embs.shape[1]

        # Create an EmbeddingTracker
        tracker = EmbeddingTracker(
            embedding_dim=embedding_dim,
            alpha=0.01,
            distance_name=distance_name
        )

        distance_scores = []
        time_overhead = []

        for batch in batch_generator(baseline_texts, batch_size):
            emb = extract_embeddings(model, tokenizer, batch, device)

            if method == "pca":
                emb = pca.transform(emb)
            elif method == "kll_sketch":
                emb = kll_transformer.transform(emb)

            # Update with the mean of the batch
            tracker.update(emb.mean(axis=0))


        start_time = time.time()
        for batch in tqdm(batch_generator(test_texts, batch_size), leave=False):
            emb = extract_embeddings(model, tokenizer, batch, device)

            if method == "pca":
                emb = pca.transform(emb)
            elif method == "kll_sketch":
                emb = kll_transformer.transform(emb)

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

        # Record results
        all_results.append(
            (method, final_dist, total_time, avg_overhead)
        )

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
    pca
):

    partial_results = []
    all_distance_names = ["mahalanobis"] + list(DISTANCE_FUNCTIONS.keys())

    for distance_name in all_distance_names:
        for drift_strength in drift_strengths:
            # Introduce drift
            drifted_texts = introduce_gradual_drift(drift_texts, fraction_shuffle=drift_strength)
            test_texts = baseline_texts + drifted_texts

            # Actually run the tracker for each approach
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
            # results is a list of (method, final_distance, total_time, avg_overhead)
            for (method, final_dist, total_time, avg_overhead) in results:
                partial_results.append({
                    "distance_name": distance_name,
                    "drift_strength": drift_strength,
                    "method": method,  # Instead of a boolean 'pca', we store the string approach
                    "final_similarity": final_dist,
                    "time_taken": total_time,
                    "avg_overhead": avg_overhead,
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

            # Compute baseline embeddings and fit PCA
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

            # Run experiments
            partial_results = run_experiments_for_model(
                model_details,
                baseline_texts,
                drift_texts,
                device,
                args["pca_components"],
                args["batch_size"],
                args["drift_strengths"],
                baseline_embs,
                pca
            )

            # Attach metadata and store
            for r in partial_results:
                r["seed"] = seed
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
