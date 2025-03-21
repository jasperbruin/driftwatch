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

LEGACY_DISTANCE_FUNCTIONS = {
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
    # TODO: handle "wasserstein" and "mmd" separately since they require different computations.
}


def approx_wasserstein_1d(bin_edges1, pmf1, bin_edges2, pmf2):
    """
    Approximate 1D Wasserstein distance between two discrete distributions:
    (bin_edges1, pmf1) and (bin_edges2, pmf2).
    We unify edges, build CDFs, and numerically integrate |CDF1 - CDF2|.
    """
    pmf1 /= (pmf1.sum() + 1e-12)
    pmf2 /= (pmf2.sum() + 1e-12)

    cdf1 = np.cumsum(pmf1)
    cdf2 = np.cumsum(pmf2)

    # Combine and sort all edges
    all_edges = np.unique(np.concatenate([bin_edges1, bin_edges2]))
    all_edges.sort()

    def cdf_lookup(edges, cdf, x):
        # For x < edges[0], cdf=0; for x >= edges[-1], cdf=1
        idx = np.searchsorted(edges, x, side='right') - 1
        if idx < 0:
            return 0.0
        if idx >= len(cdf):
            return 1.0
        return cdf[idx]

    # We'll sample midpoints of each sub-interval in the unified edge grid
    # and integrate the difference in stepwise fashion.
    distance = 0.0
    for i in range(len(all_edges) - 1):
        left = all_edges[i]
        right = all_edges[i + 1]
        midpoint = 0.5 * (left + right)

        cdf_val_1 = cdf_lookup(bin_edges1, cdf1, midpoint)
        cdf_val_2 = cdf_lookup(bin_edges2, cdf2, midpoint)
        distance += abs(cdf_val_1 - cdf_val_2) * (right - left)

    return distance

def _mmd_1d_from_hist(p, q, kernel='rbf', sigma=1.0):
    """
    A naive 1D MMD approximation from discrete PMFs p, q.
    We treat each bin index as a "point" and weight by p, q.
    This is purely illustrative; real MMD is often computed on samples directly.
    """
    p /= (p.sum() + 1e-12)
    q /= (q.sum() + 1e-12)

    # Indices as "points"
    N = len(p)
    idx = np.arange(N).reshape(-1, 1).astype(float)

    def rbf_kernel(X, Y, s):
        XX = (X*X).sum(axis=1, keepdims=True)
        YY = (Y*Y).sum(axis=1, keepdims=True)
        dists = XX + YY.T - 2 * np.dot(X, Y.T)
        return np.exp(-dists / (2 * s**2))

    Kxx = rbf_kernel(idx, idx, sigma)
    Exx = (p.reshape(-1, 1) * p.reshape(1, -1) * Kxx).sum()

    Kyy = rbf_kernel(idx, idx, sigma)
    Eyy = (q.reshape(-1, 1) * q.reshape(1, -1) * Kyy).sum()

    Kxy = rbf_kernel(idx, idx, sigma)
    Exy = (p.reshape(-1, 1) * q.reshape(1, -1) * Kxy).sum()

    return Exx + Eyy - 2 * Exy

class EmbeddingTracker:
    """
    A tracker that can operate in:
    1) Legacy vector-based mode (mean/var + Euclidean, Mahalanobis, etc.), or
    2) Distribution-based mode using KLL sketches for each dimension.

    The mode is decided by `distance_name`:
      - If distance_name in LEGACY_DISTANCE_FUNCTIONS or "mahalanobis", legacy mode.
      - Otherwise, if distance_name is in DISTRIBUTION_METRICS or is "wasserstein"/"mmd",
        we switch to KLL-based distribution mode.
    """
    def __init__(
        self,
        embedding_dim,
        alpha=0.01,
        distance_name="mahalanobis",
        k=50,
        num_bins=50
    ):
        """
        :param embedding_dim: Dimensionality of embeddings.
        :param alpha: (For legacy distances) exponential moving average factor.
        :param distance_name: Which metric to compute.
        :param k: K param for dimension-wise KLL sketches (distribution-based).
        :param num_bins: Number of bins for histogram approximation from KLL.
        """
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.distance_name = distance_name
        self.k = k
        self.num_bins = num_bins

        self.is_distribution_mode = (
            distance_name in DISTRIBUTION_METRICS
            or distance_name in ("wasserstein", "mmd")
        )

        # Legacy statistics
        self.mean = np.zeros((embedding_dim,), dtype=np.float64)
        self.var_diag = np.ones((embedding_dim,), dtype=np.float64)
        self.count = 0

        # KLL sketches for distribution mode
        self.kll_sketches = [
            kll_floats_sketch(k) for _ in range(embedding_dim)
        ]

    def update(self, embeddings):
        """
        Update the tracker with a batch of embeddings.

        In distribution-based mode:
          - Merge each dimension's values into that dimension's KLL sketch.

        In legacy vector mode:
          - Update the EMA of the mean and diagonal variance (if needed).
        """
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]

        if self.is_distribution_mode:
            # Merge data into dimension-wise KLL
            for dim_idx in range(self.embedding_dim):
                col_vals = embeddings[:, dim_idx]
                for val in col_vals:
                    self.kll_sketches[dim_idx].update(val)
        else:
            # Legacy approach: keep updating mean/var
            batch_mean = embeddings.mean(axis=0)
            if self.count == 0:
                self.mean = batch_mean
            else:
                self.mean = (1 - self.alpha) * self.mean + self.alpha * batch_mean

            if self.distance_name == "mahalanobis":
                diff = batch_mean - self.mean
                self.var_diag = (1 - self.alpha) * self.var_diag + self.alpha * (diff**2)

            self.count += 1

    def compute_distance(self, embeddings):
        """
        Compute the distance between the tracker's "baseline" (mean or KLL sketches)
        and the new data in 'embeddings'.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]

        # 1) Legacy vector-based distances
        if not self.is_distribution_mode:
            if self.distance_name == "mahalanobis":
                diff = embeddings.mean(axis=0) - self.mean
                epsilon = 1e-12
                return float(np.sqrt(np.sum(diff**2 / (self.var_diag + epsilon))))

            elif self.distance_name in LEGACY_DISTANCE_FUNCTIONS:
                dist_fn = LEGACY_DISTANCE_FUNCTIONS[self.distance_name]
                return float(dist_fn(self.mean, embeddings.mean(axis=0)))

            else:
                raise ValueError(f"Unknown legacy distance: {self.distance_name}")

        new_sketches = [kll_floats_sketch(self.k) for _ in range(self.embedding_dim)]
        for dim_idx in range(self.embedding_dim):
            col_vals = embeddings[:, dim_idx]
            for val in col_vals:
                new_sketches[dim_idx].update(val)

        # dimension-wise distances
        dim_distances = []
        for dim_idx in range(self.embedding_dim):
            base_sketch = self.kll_sketches[dim_idx]
            new_sketch = new_sketches[dim_idx]

            bin_edges_b, pmf_b = self._sketch_to_hist(base_sketch, self.num_bins)
            bin_edges_n, pmf_n = self._sketch_to_hist(new_sketch, self.num_bins)

            if self.distance_name in DISTRIBUTION_METRICS:
                # e.g. KL, JS, Hellinger, Bhattacharyya
                dist_fn = DISTRIBUTION_METRICS[self.distance_name]
                pmf_b /= (pmf_b.sum() + 1e-12)
                pmf_n /= (pmf_n.sum() + 1e-12)
                dim_dist = dist_fn(pmf_b, pmf_n)
                dim_distances.append(dim_dist)

            elif self.distance_name == "wasserstein":
                # 1D univariate Wasserstein
                dim_dist = approx_wasserstein_1d(bin_edges_b, pmf_b, bin_edges_n, pmf_n)
                dim_distances.append(dim_dist)

            elif self.distance_name == "mmd":
                # Naive 1D MMD approach
                pmf_b /= (pmf_b.sum() + 1e-12)
                pmf_n /= (pmf_n.sum() + 1e-12)
                dim_dist = _mmd_1d_from_hist(pmf_b, pmf_n, kernel='rbf', sigma=1.0)
                dim_distances.append(dim_dist)
            else:
                raise ValueError(f"Unknown distribution-based metric: {self.distance_name}")

        if not dim_distances:
            return 0.0
        return float(np.mean(dim_distances))

    def _sketch_to_hist(self, sketch, num_bins):
        """
        Convert a KLL sketch to a histogram: (bin_edges, pmf).
        - If the sketch is empty or has effectively the same min/max, handle gracefully.
        - We query rank() at each bin edge to approximate the CDF, then diff -> pmf.
        """
        if sketch.is_empty():
            # Return dummy single bin so we don't break the distance computation
            return np.array([0, 1]), np.array([1.0])

        min_val = sketch.get_min_value()
        max_val = sketch.get_max_value()
        if abs(max_val - min_val) < 1e-12:
            # All the data is essentially the same value
            return np.array([min_val, max_val + 1e-12]), np.array([1.0])

        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        cdf_vals = [sketch.get_rank(edge) for edge in bin_edges]
        cdf_vals = np.array(cdf_vals)
        pmf = np.diff(cdf_vals)
        pmf = np.clip(pmf, 0, 1.0)
        return bin_edges, pmf