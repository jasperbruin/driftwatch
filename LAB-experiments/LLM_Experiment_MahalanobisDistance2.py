#%%
!pip install torch transformers accelerate datasets scikit-learn numpy evidently
!pip install huggingface-hub
#%%
# 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
!huggingface-cli login --token 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
#%%
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
import os


#################################
# Parse Arguments
#################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run long embedding drift experiments.")
    parser.add_argument("--model_name", type=str,
                        default="openai-community/gpt2",
                        help="HuggingFace model name")
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str,
                        default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split")
    parser.add_argument("--max_texts", type=int, default=30000,
                        help="Max number of texts to use from dataset")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Alpha for running mean/cov updates")
    parser.add_argument("--drift_threshold_std", type=float, default=3.0,
                        help="Number of std devs for threshold")
    parser.add_argument("--window_size", type=int, default=50,
                        help="Rolling window size for threshold computation")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results and plots")

    # If running in a Jupyter/Colab environment, return default args
    args, unknown = parser.parse_known_args()
    return args


args = parse_args()

#################################
# Setup
#################################
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

#################################
# Load Model and Tokenizer
#################################
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
model.to(device)
model.eval()
print("Model ready on device:", device)

#################################
# Load Dataset with Simulated Drifts
#################################
dataset = load_dataset(args.dataset_name, args.dataset_config,
                       split=args.split)
texts = dataset["text"]
if args.max_texts > 0 and args.max_texts < len(texts):
    texts = texts[:args.max_texts]
print(f"Original dataset loaded: {len(texts)} texts")

# Introduce simulated drifts
drift_start = len(texts) // 3
drift_end = 2 * len(texts) // 3

# Replace a portion of the texts with new content to simulate a drift
for i in range(drift_start, drift_end):
    texts[
        i] = "This is a simulated drift event. The content has changed significantly."

print(
    f"Simulated drift introduced between indices {drift_start} and {drift_end}.")
#%%
#################################
# Utility Functions
#################################
def batch_generator(data, batch_size=32):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def extract_embeddings(model, tokenizer, texts, device):
    encodings = tokenizer(texts, return_tensors="pt", padding=True,
                          truncation=True)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model.transformer(input_ids, attention_mask=attention_mask,
                                    output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state
        avg_embeddings = last_hidden_state.mean(dim=1)
    return avg_embeddings.mean(dim=0).cpu().numpy()
#%%
#################################
# Drift Detection Class
#################################
class EmbeddingTracker:
    def __init__(self, embedding_dim, alpha=0.01):
        self.alpha = alpha
        self.mean = np.zeros((embedding_dim,))
        self.cov = np.eye(embedding_dim)
        self.count = 0

    def update(self, embedding):
        if self.count == 0:
            self.mean = embedding
            self.cov = np.eye(len(embedding))
        else:
            self.mean = (1 - self.alpha) * self.mean + self.alpha * embedding
            diff = embedding - self.mean
            self.cov = (1 - self.alpha) * self.cov + self.alpha * np.outer(
                diff, diff)
        self.count += 1

    def mahalanobis_distance(self, embedding):
        diff = embedding - self.mean
        cov_inv = np.linalg.pinv(self.cov)
        return np.sqrt(diff.T @ cov_inv @ diff)
#%%
#################################
# Main Experiment Loop
#################################

print("Starting drift detection...")
dummy_embedding = extract_embeddings(model, tokenizer, ["Hello world!"], device)
embedding_dim = len(dummy_embedding)
tracker = EmbeddingTracker(embedding_dim, alpha=args.alpha)

distances = []
thresholds = []
drift_points = []  # To store the batch indices where drifts are detected

for batch_index, batch_texts in enumerate(batch_generator(texts, batch_size=args.batch_size)):
    current_embedding = extract_embeddings(model, tokenizer, batch_texts, device)
    tracker.update(current_embedding)
    M_t = tracker.mahalanobis_distance(current_embedding)
    distances.append(M_t)

    # Compute threshold after we have enough history
    if len(distances) > args.window_size:
        recent_dists = distances[-args.window_size:]
        mean_dist = np.mean(recent_dists)
        std_dist = np.std(recent_dists)
        threshold = mean_dist + args.drift_threshold_std * std_dist
        thresholds.append(threshold)

        if M_t > threshold:
            print(f"Batch {batch_index}: Drift detected! M_t = {M_t:.4f}, threshold = {threshold:.4f}")
            print(f"Inputs in this batch:\n{batch_texts}\n")
            drift_points.append(batch_index)  # Record drift point
    else:
        thresholds.append(float('inf'))
#%%
#################################
# Save Results
#################################
distances_path = os.path.join(args.output_dir, "distances.npy")
thresholds_path = os.path.join(args.output_dir, "thresholds.npy")
drift_points_path = os.path.join(args.output_dir, "drift_points.npy")
np.save(distances_path, distances)
np.save(thresholds_path, thresholds)
np.save(drift_points_path, drift_points)
print(f"Distances saved to {distances_path}")
print(f"Thresholds saved to {thresholds_path}")
print(f"Drift points saved to {drift_points_path}")
#%%
#################################
# Visualization
#################################
plt.figure(figsize=(12, 6))
plt.plot(distances, label="Mahalanobis Distance")
plt.plot(thresholds, label="Dynamic Threshold", linestyle="--")

# Highlight detected drift points
if drift_points:
    plt.scatter(drift_points, [distances[i] for i in drift_points], color="red", label="Detected Drifts", zorder=5)

# Mark simulated drift region
plt.axvline(x=drift_start // args.batch_size, color="r", linestyle=":", label="Drift Start")
plt.axvline(x=drift_end // args.batch_size, color="g", linestyle=":", label="Drift End")

plt.xlabel("Batch Index")
plt.ylabel("Mahalanobis Distance")
plt.title("Drift Detection Over Batches with Simulated Drift")
plt.legend()

plot_path = os.path.join(args.output_dir, "drift_detection_with_drift_points.png")
plt.savefig(plot_path)
plt.show()
print(f"Plot saved to {plot_path}")
#%%
plt.figure(figsize=(12, 6))
plt.plot(distances, label="Mahalanobis Distance")
plt.plot(thresholds, label="Dynamic Threshold", linestyle="--")
plt.axvline(x=drift_start // args.batch_size, color="r", linestyle=":",
            label="Drift Start")
plt.axvline(x=drift_end // args.batch_size, color="g", linestyle=":",
            label="Drift End")
plt.xlabel("Batch Index")
plt.ylabel("Mahalanobis Distance")
plt.title("Drift Detection Over Batches with Simulated Drift")
plt.legend()

plot_path = os.path.join(args.output_dir,
                         "drift_detection_with_simulated_drift.png")
plt.savefig(plot_path)
plt.show()
print(f"Plot saved to {plot_path}")
#%%
import seaborn as sns

# Mask to identify drift points
drift_mask = distances > thresholds

pre_drift_distances = distances[~drift_mask]
post_drift_distances = distances[drift_mask]

# Plot density comparison
plt.figure(figsize=(12, 6))
sns.kdeplot(pre_drift_distances, label="Pre-Drift", color="blue", fill=True, alpha=0.6)
sns.kdeplot(post_drift_distances, label="Post-Drift", color="red", fill=True, alpha=0.6)
plt.xlabel("Mahalanobis Distance")
plt.ylabel("Density")
plt.title("Density Comparison: Pre-Drift vs Post-Drift")
plt.legend()
plt.savefig("density_comparison.png")
plt.show()
#%%
