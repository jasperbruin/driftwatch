#%%
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
from scipy.stats import scoreatpercentile
from datetime import datetime
#%%
#=====================================================
# Step 1: Load Data and Create Realistic Reference/Current Sets
#=====================================================
data = pd.read_json("Amazon_Fashion.jsonl", lines=True)


#%%
# Filter and preprocess
reviews = pd.read_json("Amazon_Fashion.jsonl", lines=True)
reviews = reviews[reviews["rating"] != 3]
reviews = reviews[reviews["timestamp"] > "2015-01-01"]
reviews["year"] = reviews["timestamp"].apply(lambda x: x.year)
reviews["month"] = reviews["timestamp"].apply(lambda x: x.month)
reviews["day"] = reviews["timestamp"].apply(lambda x: x.day)

reviews.head()

# Sort by time
data.sort_values(by="timestamp", inplace=True)

# Define a reference period (e.g., 2015-2018) and a current period (e.g., 2019-2020)
#create reference dataframe
reference_data = reviews[reviews["timestamp"] > "2015-01-01"]
reference_data = reference_data[reference_data["timestamp"] < "2020-01-01"]
#create evaluation dataframe
evaluation_data = reviews[reviews["timestamp"] >= "2020-01-01"]
evaluation_data = evaluation_data.set_index(evaluation_data["timestamp"])
evaluation_data.sort_index(inplace = True)

# remove products that appear after ref
evaluation_data_with_old_products = evaluation_data[evaluation_data["parent_asin"].isin(reference_data["parent_asin"])]
#%%
#create rolling window
rolling = evaluation_data_with_old_products["rating"].rolling("30D").mean()

# plot rolling window
import matplotlib.pyplot as plt
plt.plot(rolling)

#%%
#=====================================================
# Step 2: Feature Engineering and Model Training on Reference Data Only
#=====================================================

#encode features in the reviews for use in ml model
sparse_features = ["parent_asin", "user_id","year","month","day"]
for feat in sparse_features:
        lbe = LabelEncoder()
        reviews[feat] = lbe.fit_transform(reviews[feat])
#%%



#%%
#=====================================================
# Step 3: Extract Embeddings
#=====================================================
def extract_embeddings(model, data_dict):
    """Extract combined embeddings from the trained model."""
    model.eval()
    with torch.no_grad():
        embeddings = model.embedding_dict
        # Extract embeddings for each sparse feature and concatenate
        feat_embeddings = [embeddings[feat](
            torch.tensor(data_dict[feat].values, device=device))
                           for feat in sparse_features]
        combined = torch.cat(feat_embeddings, axis=1)
    return combined.cpu().numpy()


reference_embeddings = extract_embeddings(model, reference_data)
current_embeddings = extract_embeddings(model, current_data)
#%%
#=====================================================
# (Optional) Step 4: Dimensionality Reduction using PCA
#=====================================================

# compute best n_components for PCA
n_components = min(reference_embeddings.shape[0], reference_embeddings.shape[1])

print(f"Number of components: {n_components}")

pca = PCA(n_components=n_components)  # choose a dimension that captures most variance
reference_emb_reduced = pca.fit_transform(reference_embeddings)
current_emb_reduced = pca.transform(current_embeddings)
#%%
#=====================================================
# Step 5: Implement MMD Calculation
#=====================================================
def rbf_kernel(x, y, gamma=None):
    """Compute the RBF kernel between arrays x and y.
    gamma = 1/(2*sigma^2). If gamma is None, uses 1/num_features heuristic."""
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    xx = np.sum(x * x, axis=1)
    yy = np.sum(y * y, axis=1)
    xy = np.dot(x, y.T)
    Kx = np.reshape(xx, (-1, 1))
    Ky = np.reshape(yy, (1, -1))
    dist = Kx - 2 * xy + Ky
    return np.exp(-gamma * dist)


def mmd(x, y, gamma=None):
    """Compute the Maximum Mean Discrepancy (MMD) between x and y."""
    Kxx = rbf_kernel(x, x, gamma=gamma)
    Kyy = rbf_kernel(y, y, gamma=gamma)
    Kxy = rbf_kernel(x, y, gamma=gamma)

    # MMD^2 = E[Kxx] + E[Kyy] - 2E[Kxy]
    return np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)


#%%

#=====================================================
# Step 6: Compute Baseline MMD Distribution from Reference Data for Thresholding
#=====================================================
# We can bootstrap from the reference data to see what MMD values we get comparing random splits of the reference.
rng = np.random.default_rng(42)
num_bootstrap = 50
mmd_values = []
N = len(reference_emb_reduced)

for _ in range(num_bootstrap):
    idx1 = rng.choice(N, size=N // 2, replace=False)
    idx2 = rng.choice(N, size=N // 2, replace=False)
    x1 = reference_emb_reduced[idx1]
    x2 = reference_emb_reduced[idx2]
    mmd_val = mmd(x1, x2)
    mmd_values.append(mmd_val)

# Determine a threshold based on a high percentile of no-drift MMD distribution
threshold = scoreatpercentile(mmd_values, 99)  # 99th percentile, for example
#%%

#=====================================================
# Step 7: Compute MMD Between Reference and Current Data
#=====================================================
drift_score = mmd(reference_emb_reduced, current_emb_reduced)

print("Drift Score (MMD):", drift_score)
print("Threshold:", threshold)

if drift_score > threshold:
    print("Drift detected!")
else:
    print("No significant drift detected.")

#=====================================================
# Additional Notes:
# - Consider training a self-supervised embedding model or using another method for embeddings
#   that is not tied to the prediction task.
# - You can also update gamma for RBF kernels or experiment with different kernels.
# - Adjust the time windows for your reference/current sets as appropriate for your use case.
# - Longer training or more complex embeddings might improve sensitivity to drift.
# - PCA dimensionality can be tuned. Sometimes using no dimensionality reduction might be appropriate,
#   though it can be expensive for MMD in very high dimensions.
# - Consider repeating the drift calculation over rolling windows of current data to identify
#   when drift first appears.