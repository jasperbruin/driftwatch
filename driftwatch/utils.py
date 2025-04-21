import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from datasketches import kll_floats_sketch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

from config import args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_embeddings(model, tokenizer, texts, device):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encodings = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        if model.config.model_type == "t5":
            decoder_input_ids = torch.zeros(
                (input_ids.shape[0], 1), dtype=torch.long, device=device
            )
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
        else:
            outputs = model(input_ids, attention_mask=attention_mask)

        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
            if model.config.model_type in [
                "gpt2",
                "gpt_neo",
                "opt",
                "mistral",
                "falcon",
                "bloom",
            ]:
                return hidden_states[:, -1, :].cpu().numpy()
            elif model.config.model_type in ["t5", "mbart"]:
                return hidden_states.mean(dim=1).cpu().numpy()
            elif (
                "bert" in model.config.model_type
                or "electra" in model.config.model_type
            ):
                return hidden_states[:, 0, :].cpu().numpy()
        return None


def batch_generator(data, batch_size=32):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def introduce_gradual_drift(text_list, fraction_shuffle=0.5):
    new_texts = []
    for txt in text_list:
        words = txt.split()
        if len(words) < 2:
            new_texts.append(txt)
            continue
        k = int(len(words) * fraction_shuffle)
        if k < 1:
            new_texts.append(txt)
            continue
        indices = list(range(len(words)))
        random.shuffle(indices)
        shuffle_indices = indices[:k]
        to_shuffle = [words[i] for i in shuffle_indices]
        random.shuffle(to_shuffle)
        for i, idx in enumerate(shuffle_indices):
            words[idx] = to_shuffle[i]
        new_texts.append(" ".join(words))
    return new_texts


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon (Metal Performance Shaders)
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    else:
        return torch.device("cpu")  # Fallback to CPU


def load_and_split_texts(dataset_info, max_texts):
    """Load dataset, shuffle, and split into baseline_texts and drift_texts."""
    dataset_name = dataset_info["name"]
    dataset_config = dataset_info["config"]
    dataset_split = dataset_info["split"]
    text_col = dataset_info["text_column"]

    print(f"=== Loading dataset: {dataset_name} ===")
    ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    texts = list(ds[text_col])
    random.shuffle(texts)

    if max_texts > 0 and len(texts) > max_texts:
        texts = texts[:max_texts]

    half_point = len(texts) // 2
    baseline_texts = texts[:half_point]
    drift_texts = texts[half_point:]

    return dataset_name, baseline_texts, drift_texts


def compute_baseline_embeddings_and_pca(
    model_name, baseline_texts, device, pca_components, batch_size
):
    """Compute baseline embeddings and fit PCA (if needed) on them."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Precompute baseline embeddings
    baseline_embs_list = []
    for batch in batch_generator(baseline_texts, batch_size):
        emb = extract_embeddings(model, tokenizer, batch, device)
        baseline_embs_list.append(emb)
    baseline_embs = np.concatenate(baseline_embs_list, axis=0)

    # Fit PCA on the baseline embeddings
    pca = PCA(n_components=pca_components)
    pca.fit(baseline_embs)

    return model, tokenizer, baseline_embs, pca


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


def kll_transform(embeddings, k=8, sketch_k=None):
    """Optimized KLL-based transform that converts each embedding vector
    into k quantiles of its values. This yields a new embedding of size k
    for every original row in 'embeddings'.

    Parameters
    ----------
    - embeddings: numpy array of shape (n_samples, n_features)
    - k: number of quantiles to extract (output dimension)
    - sketch_k: KLL accuracy parameter (if None, defaults to k*2)

    Returns
    -------
    - transformed embeddings of shape (n_samples, k)

    """
    if sketch_k is None:
        sketch_k = min(
            k * 2, 200
        )  # Higher accuracy for sketch, capped at reasonable value

    n_samples = embeddings.shape[0]
    transformed = np.zeros((n_samples, k), dtype=np.float32)
    quantile_points = np.linspace(0, 1, k)

    for i in range(n_samples):
        # Create a new sketch for each embedding vector
        sketch = kll_floats_sketch(sketch_k)
        # Update with all values at once if supported
        sketch.update(np.asarray(embeddings[i], dtype=np.float32))
        transformed[i] = sketch.get_quantiles(quantile_points.tolist())

    return transformed
