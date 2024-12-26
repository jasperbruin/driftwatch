#%%
!pip install numpy pandas torch scikit-learn tensorflow deepctr-torch
#%%
#################################
# Import Libraries
#################################
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
import matplotlib.pyplot as plt
import os
#%%
#################################
# Load and Preprocess Data
#################################
# Load and preprocess data
data = pd.read_json("experiment_1/Amazon_Fashion.jsonl", lines=True)
data = data[data["rating"] != 3]
data = data[data["timestamp"] > '2015-01-01']
data["year"] = data["timestamp"].apply(lambda x: x.year)
data["month"] = data["timestamp"].apply(lambda x: x.month)
data["day"] = data["timestamp"].apply(lambda x: x.day)
data["rating"] = data["rating"].transform(lambda x: 1 if x > 3 else 0)

# Set index to timestamp
data = data.set_index("timestamp")
data.sort_index(inplace=True)
#%%
#################################
# Define Ground Truth Functionality
#################################
# Calculate rolling mean and threshold
rolling_mean = data["rating"].rolling("30D").mean()
rolling_std = data["rating"].rolling("30D").std()
significance_threshold = rolling_mean - 1.96 * rolling_std

# Add ground truth drift flag based on threshold
data["drift_flag"] = (data["rating"].rolling("30D").mean() < significance_threshold).fillna(False)

#################################
# Visualize Data and Mark Observed Drift
#################################
plt.figure(figsize=(14, 7))

# Plot ratings over time
plt.plot(data.index, data["rating"], label="Ratings", alpha=0.5)
# Plot rolling mean
plt.plot(data.index, rolling_mean, label="Rolling Mean (30D)", color="orange")
# Plot significance threshold
plt.plot(data.index, significance_threshold, label="Threshold (Mean - 1.96*Std)", color="red", linestyle="--")

# Highlight drift periods
plt.fill_between(
    data.index,
    rolling_mean,
    significance_threshold,
    where=data["drift_flag"],
    color="red",
    alpha=0.2,
    label="Drift Detected",
)

# Highlight observed drift (2021–2023)
plt.axvspan(
    pd.Timestamp("2021-01-01"), 
    pd.Timestamp("2023-12-12"), 
    color="purple", 
    alpha=0.1, 
    label="Observed Drift (2021-2024)"
)

# Add titles and legend
plt.title("Ratings Over Time with Drift Detection", fontsize=16)
plt.xlabel("Timestamp", fontsize=12)
plt.ylabel("Rating", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()
#%%
# print data features
print(data.columns)

sparse_features = ["asin", "parent_asin", "user_id"]
#%%
#################################
# Utility Functions
#################################
def compute_mahalanobis_distance(mean, cov_inv, embedding):
    diff = embedding - mean
    return np.sqrt(diff.T @ cov_inv @ diff)

class DriftDetector:
    def __init__(self, embedding_dim, alpha=0.01):
        self.alpha = alpha
        self.mean = np.zeros(embedding_dim)
        self.cov = np.eye(embedding_dim)
        self.count = 0

    def update(self, embedding):
        if self.count == 0:
            self.mean = embedding
            self.cov = np.eye(len(embedding))
        else:
            self.mean = (1 - self.alpha) * self.mean + self.alpha * embedding
            diff = embedding - self.mean
            self.cov = (1 - self.alpha) * self.cov + self.alpha * np.outer(diff, diff)
        self.count += 1

    def detect_drift(self, embedding):
        return compute_mahalanobis_distance(self.mean, np.linalg.pinv(self.cov), embedding)

#################################
# Model Training (DeepFM)
#################################
# Suppose you have some target and sparse_features defined:
target = "rating"
sparse_features = ["year", "month", "day"]  # example sparse features

# Label encoding
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

# Define feature columns
embedding_dim = 8
fixlen_feature_columns = [
    SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
    for feat in sparse_features
]
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# Prepare model input
model_input = {name: data[name] for name in sparse_features}

# Decide on device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
model.compile("adam", metrics=["AUC"], loss='binary_crossentropy')

# Train the model (using some fraction of data as data_sample)
data_sample = data.sample(frac=0.5, random_state=42)  # example sampling
history = model.fit(
    {name: data_sample[name] for name in sparse_features},
    data_sample[target].values,
    batch_size=1024,
    epochs=2,
    verbose=2,
    validation_split=0.1
)

#################################
# Mahalanobis Drift Detection Setup
#################################
window_size = 10
threshold_multiplier = 3
alpha = 0.01

tracker = DriftDetector(embedding_dim, alpha=alpha)
drift_distances = []
thresholds = []
predicted_drift_labels = []

# To map each batch back to its ground truth drift label, we need indexing:
n_batches = 300
batch_list = np.array_split(data_sample, n_batches)

with torch.no_grad():
    for batch_index, batch_df in enumerate(batch_list):
        # Prepare batch input
        batch_input = {key: batch_df[key].values for key in sparse_features}
        
        # Get model's embeddings or outputs
        embeddings = model.predict(batch_input)
        mean_embedding = np.mean(embeddings, axis=0)
        
        # Update and compute distance
        tracker.update(mean_embedding)
        drift_distance = tracker.detect_drift(mean_embedding)
        drift_distances.append(drift_distance)
        
        # Dynamic threshold
        if len(drift_distances) > window_size:
            recent_dists = drift_distances[-window_size:]
            mean_dist = np.mean(recent_dists)
            std_dist = np.std(recent_dists)
            threshold = mean_dist + threshold_multiplier * std_dist
        else:
            threshold = (
                np.mean(drift_distances) +
                threshold_multiplier * np.std(drift_distances)
            )
        thresholds.append(threshold)
        
        # Predicted drift (batch level)
        if drift_distance > threshold:
            predicted_drift_labels.append(1)
        else:
            predicted_drift_labels.append(0)

        if predicted_drift_labels[-1] == 1:
            print(f"Batch {batch_index}: Drift detected! Distance = {drift_distance:.4f}, Threshold = {threshold:.4f}")



#%%
# import confusion matrix
from sklearn.metrics import confusion_matrix

#################################
# Compute Ground Truth (Batch-Level) and FPR
#################################
# We need a ground truth drift label per batch.  
# One strategy: if >50% of rows in the batch_df have drift_flag == True, 
# then label the entire batch as True. Otherwise False.

actual_drift_labels = []
for batch_df in batch_list:
    batch_flags = batch_df["drift_flag"]
    # If the majority in this batch is drift, label as drift
    drift_fraction = batch_flags.mean()
    if drift_fraction > 0.5:
        actual_drift_labels.append(1)
    else:
        actual_drift_labels.append(0)

# Now compute confusion matrix
y_true = np.array(actual_drift_labels)
y_pred = np.array(predicted_drift_labels)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# False Positive Rate
fpr = fp / (fp + tn)
print("Confusion Matrix:")
print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")
print(f"False Positive Rate (FPR): {fpr:.4f}")

#################################
# Save Results
#################################
distances_path = "drift_distances_dl.npy"
thresholds_path = "drift_thresholds_dl.npy"
np.save(distances_path, drift_distances)
np.save(thresholds_path, thresholds)
print("Drift detection results saved.")
#%%
