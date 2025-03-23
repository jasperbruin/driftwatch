import numpy as np
from datasketches import kll_floats_sketch

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

def minkowski_distance(x, y, p=3):
    return np.sum(np.abs(x - y) ** p) ** (1.0 / p)

def chebyshev_distance(x, y):
    return np.max(np.abs(x - y))

def canberra_distance(x, y):
    numerator = np.abs(x - y)
    denominator = np.abs(x) + np.abs(y) + 1e-12
    return np.sum(numerator / denominator)

VECTOR_DISTANCE_FUNCTIONS = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "minkowski": lambda u, v: minkowski_distance(u, v, p=3),
    "chebyshev": chebyshev_distance,
    "canberra": canberra_distance,
}

def kl_divergence(p, q, eps=1e-12):
    """KL(p||q) for discrete distributions p, q (already normalized)."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log(p / q))

def jensen_shannon(p, q, eps=1e-12):
    """Jensen-Shannon divergence (symmetric) for discrete distributions p, q."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)

def hellinger_distance(p, q, eps=1e-12):
    """Hellinger distance for discrete distributions p, q."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p /= p.sum()
    q /= q.sum()
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))

def bhattacharyya_distance(p, q, eps=1e-12):
    """Bhattacharyya distance for discrete distributions p, q."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p /= p.sum()
    q /= q.sum()
    bc = np.sum(np.sqrt(p * q))
    return -np.log(bc + eps)

DISTRIBUTION_METRICS = {
    "kl": kl_divergence,
    "js": jensen_shannon,
    "hellinger": hellinger_distance,
    "bhattacharyya": bhattacharyya_distance,
    # "wasserstein" and "mmd" handled separately
}

def approx_wasserstein_1d(bin_edges1, pmf1, bin_edges2, pmf2):
    """
    Approximate 1D Wasserstein distance between two discrete distributions.
    We unify edges, build CDFs, and numerically integrate |CDF1 - CDF2|.
    """
    pmf1 /= (pmf1.sum() + 1e-12)
    pmf2 /= (pmf2.sum() + 1e-12)

    cdf1 = np.cumsum(pmf1)
    cdf2 = np.cumsum(pmf2)

    # Combine and sort all edges
    all_edges = np.union1d(bin_edges1, bin_edges2)
    all_edges.sort()

    def cdf_lookup(edges, cdf, x):
        idx = np.searchsorted(edges, x, side='right') - 1
        if idx < 0:
            return 0.0
        if idx >= len(cdf):
            return 1.0
        return cdf[idx]

    distance = 0.0
    for i in range(len(all_edges) - 1):
        left = all_edges[i]
        right = all_edges[i + 1]
        midpoint = 0.5 * (left + right)

        cdf_val_1 = cdf_lookup(bin_edges1, cdf1, midpoint)
        cdf_val_2 = cdf_lookup(bin_edges2, cdf2, midpoint)
        distance += abs(cdf_val_1 - cdf_val_2) * (right - left)

    return distance

# RBF kernel for MMD in 1D
def _rbf_kernel_1d(x, y, sigma):
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    XX = (X*X).sum(axis=1, keepdims=True)
    YY = (Y*Y).sum(axis=1, keepdims=True)
    dists = XX + YY.T - 2 * np.dot(X, Y.T)
    return np.exp(-dists / (2 * sigma**2))

def _mmd_1d_from_bins(bin_edges, pmf1, pmf2, kernel='rbf', sigma=1.0):
    """
    Approximate MMD in 1D using the bin centers as "points", with
    pmf1, pmf2 as their weights.
    """
    pmf1 /= (pmf1.sum() + 1e-12)
    pmf2 /= (pmf2.sum() + 1e-12)

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    Kxx = _rbf_kernel_1d(centers, centers, sigma)
    Exx = 0.0
    for i in range(len(centers)):
        for j in range(len(centers)):
            Exx += pmf1[i] * pmf1[j] * Kxx[i, j]

    Kyy = _rbf_kernel_1d(centers, centers, sigma)
    Eyy = 0.0
    for i in range(len(centers)):
        for j in range(len(centers)):
            Eyy += pmf2[i] * pmf2[j] * Kyy[i, j]

    Kxy = _rbf_kernel_1d(centers, centers, sigma)
    Exy = 0.0
    for i in range(len(centers)):
        for j in range(len(centers)):
            Exy += pmf1[i] * pmf2[j] * Kxy[i, j]

    return Exx + Eyy - 2 * Exy

class EmbeddingTracker:
    def __init__(
        self,
        embedding_dim,
        alpha=0.01,
        distance_name="mahalanobis",
        k=50,
        num_bins=50,
        distribution_impl="kll",
    ):
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.distance_name = distance_name
        self.k = k
        self.num_bins = num_bins
        self.distribution_impl = distribution_impl

        self.is_distribution_mode = (
            distance_name in DISTRIBUTION_METRICS
            or distance_name in ("wasserstein", "mmd")
        )

        self.mean = np.zeros((embedding_dim,), dtype=np.float64)
        self.var_diag = np.ones((embedding_dim,), dtype=np.float64)
        self.count = 0

        self.kll_sketches = []
        self.hist_counts = []
        self.hist_min = np.full(embedding_dim, np.inf)
        self.hist_max = np.full(embedding_dim, -np.inf)

        if self.is_distribution_mode:
            if self.distribution_impl == "kll":
                self.kll_sketches = [kll_floats_sketch(k) for _ in range(embedding_dim)]
            elif self.distribution_impl == "histogram":
                self.hist_counts = [np.zeros(num_bins, dtype=np.float64) for _ in range(embedding_dim)]
            else:
                raise ValueError(f"Unsupported distribution_impl: {distribution_impl}")

    def update(self, embeddings):
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]

        if self.is_distribution_mode:
            if self.distribution_impl == "kll":
                for dim_idx in range(self.embedding_dim):
                    temp_sketch = kll_floats_sketch(self.k)
                    temp_sketch.update(np.asarray(embeddings[:, dim_idx], dtype=np.float32))
                    self.kll_sketches[dim_idx].merge(temp_sketch)

            elif self.distribution_impl == "histogram":
                mins = embeddings.min(axis=0)
                maxs = embeddings.max(axis=0)
                self.hist_min = np.minimum(self.hist_min, mins)
                self.hist_max = np.maximum(self.hist_max, maxs)

                for dim_idx in range(self.embedding_dim):
                    col_vals = embeddings[:, dim_idx]
                    range_span = max(self.hist_max[dim_idx] - self.hist_min[dim_idx], 1e-12)
                    bin_width = range_span / self.num_bins
                    bin_indices = ((col_vals - self.hist_min[dim_idx]) / bin_width).astype(int)
                    bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
                    np.add.at(self.hist_counts[dim_idx], bin_indices, 1.0)

        else:
            batch_mean = embeddings.mean(axis=0)
            if self.count == 0:
                self.mean = batch_mean
            else:
                self.mean = (1 - self.alpha) * self.mean + self.alpha * batch_mean

            if self.distance_name == "mahalanobis":
                diff = batch_mean - self.mean
                self.var_diag = (1 - self.alpha) * self.var_diag + self.alpha * (diff ** 2)

            self.count += 1

    def compute_distance(self, embeddings):
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]

        if not self.is_distribution_mode:
            if self.distance_name == "mahalanobis":
                diff = embeddings.mean(axis=0) - self.mean
                epsilon = 1e-12
                return float(np.sqrt(np.sum(diff ** 2 / (self.var_diag + epsilon))))
            elif self.distance_name in VECTOR_DISTANCE_FUNCTIONS:
                dist_fn = VECTOR_DISTANCE_FUNCTIONS[self.distance_name]
                return float(dist_fn(self.mean, embeddings.mean(axis=0)))
            else:
                raise ValueError(f"Unknown legacy distance: {self.distance_name}")

        if self.distribution_impl == "kll":
            new_sketches = [kll_floats_sketch(self.k) for _ in range(self.embedding_dim)]
            for dim_idx in range(self.embedding_dim):
                new_sketches[dim_idx].update(np.asarray(embeddings[:, dim_idx], dtype=np.float32))

        elif self.distribution_impl == "histogram":
            new_counts = [np.zeros(self.num_bins, dtype=np.float64) for _ in range(self.embedding_dim)]
            for dim_idx in range(self.embedding_dim):
                col_vals = embeddings[:, dim_idx]
                range_span = max(self.hist_max[dim_idx] - self.hist_min[dim_idx], 1e-12)
                bin_width = range_span / self.num_bins
                bin_indices = ((col_vals - self.hist_min[dim_idx]) / bin_width).astype(int)
                bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
                np.add.at(new_counts[dim_idx], bin_indices, 1.0)
        else:
            raise ValueError(f"Unsupported distribution_impl: {self.distribution_impl}")

        dim_distances = []
        for dim_idx in range(self.embedding_dim):
            if self.distribution_impl == "kll":
                base_sketch = self.kll_sketches[dim_idx]
                new_sketch = new_sketches[dim_idx]
                bin_edges_b, pmf_b = self._sketch_to_hist(base_sketch, self.num_bins)
                bin_edges_n, pmf_n = self._sketch_to_hist(new_sketch, self.num_bins)
                all_edges = np.union1d(bin_edges_b, bin_edges_n)
                all_edges.sort()
                pmf_b = self._resample_pmf(base_sketch, all_edges)
                pmf_n = self._resample_pmf(new_sketch, all_edges)
            else:
                base_min = self.hist_min[dim_idx]
                base_max = self.hist_max[dim_idx]
                if base_min >= base_max:
                    dim_distances.append(0.0)
                    continue
                bin_edges_b = np.linspace(base_min, base_max, self.num_bins + 1)
                pmf_b = self.hist_counts[dim_idx] / (self.hist_counts[dim_idx].sum() + 1e-12)
                pmf_n = new_counts[dim_idx] / (new_counts[dim_idx].sum() + 1e-12)
                all_edges = bin_edges_b

            if self.distance_name in DISTRIBUTION_METRICS:
                dist_fn = DISTRIBUTION_METRICS[self.distance_name]
                dim_dist = dist_fn(pmf_b, pmf_n)
            elif self.distance_name == "wasserstein":
                dim_dist = approx_wasserstein_1d(all_edges[:-1], pmf_b, all_edges[:-1], pmf_n)
            elif self.distance_name == "mmd":
                dim_dist = _mmd_1d_from_bins(all_edges, pmf_b, pmf_n, kernel='rbf', sigma=1.0)
            else:
                raise ValueError(f"Unknown distribution-based metric: {self.distance_name}")
            dim_distances.append(dim_dist)

        return float(np.mean(dim_distances)) if dim_distances else 0.0

    def _sketch_to_hist(self, sketch, num_bins):
        if sketch.is_empty():
            return np.array([0, 1]), np.array([1.0])
        min_val = sketch.get_min_value()
        max_val = sketch.get_max_value()
        if abs(max_val - min_val) < 1e-12:
            return np.array([min_val, max_val + 1e-12]), np.array([1.0])
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        cdf_vals = [sketch.get_rank(edge) for edge in bin_edges]
        pmf = np.clip(np.diff(cdf_vals), 0, 1.0)
        return bin_edges, pmf

    def _resample_pmf(self, sketch, edges):
        if sketch.is_empty():
            return np.array([1.0]) if len(edges) > 1 else np.array([])
        cdf_vals = [sketch.get_rank(e) for e in edges]
        return np.clip(np.diff(cdf_vals), 0, 1.0)
