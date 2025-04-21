import numpy as np
from datasketches import kll_floats_sketch

from metrics import (
    DISTRIBUTION_METRICS,
    VECTOR_DISTANCE_FUNCTIONS,
    _mmd_1d_from_bins,
    approx_wasserstein_1d,
)


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

        self.mean = np.zeros(embedding_dim, dtype=np.float64)
        self.var_diag = np.ones(embedding_dim, dtype=np.float64)
        self.count = 0

        # For distribution-based modes:
        self.kll_sketches = []
        self.hist_counts = []
        # We'll store per-dim min/max to define fixed bin edges once (based on baseline data)
        self.min_vals = np.full(embedding_dim, np.inf)
        self.max_vals = np.full(embedding_dim, -np.inf)

        # Cache of "baseline" PMF for each dimension (KLL or hist), so we don't recalc repeatedly
        self.cached_baseline_pmfs = [None] * embedding_dim
        self.cached_bin_edges = [None] * embedding_dim

        if self.is_distribution_mode:
            if self.distribution_impl == "kll":
                # Create one KLL sketch per dimension. We'll update them directly each time
                self.kll_sketches = [kll_floats_sketch(k) for _ in range(embedding_dim)]
            elif self.distribution_impl == "histogram":
                self.hist_counts = [
                    np.zeros(num_bins, dtype=np.float64) for _ in range(embedding_dim)
                ]
            else:
                raise ValueError(f"Unsupported distribution_impl: {distribution_impl}")

    def update(self, embeddings):
        """Called once for each batch that belongs to the 'baseline' distribution,
        or if you're doing real-time updates. We'll accumulate data in either
        KLL sketches or histograms. Also track per-dimension min/max for bin edges.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]

        if self.is_distribution_mode:
            # Update distribution-based trackers
            # 1) Update min/max
            mins = embeddings.min(axis=0)
            maxs = embeddings.max(axis=0)
            self.min_vals = np.minimum(self.min_vals, mins)
            self.max_vals = np.maximum(self.max_vals, maxs)

            if self.distribution_impl == "kll":
                # Directly update each dimension's KLL with the entire column
                for dim_idx in range(self.embedding_dim):
                    self.kll_sketches[dim_idx].update(
                        np.asarray(embeddings[:, dim_idx], dtype=np.float32)
                    )
            elif self.distribution_impl == "histogram":
                for dim_idx in range(self.embedding_dim):
                    col_vals = embeddings[:, dim_idx]
                    # We'll update the histogram in place
                    range_span = max(
                        self.max_vals[dim_idx] - self.min_vals[dim_idx], 1e-12
                    )
                    bin_width = range_span / self.num_bins
                    bin_indices = (
                        (col_vals - self.min_vals[dim_idx]) / bin_width
                    ).astype(int)
                    bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
                    np.add.at(self.hist_counts[dim_idx], bin_indices, 1.0)

            # Clear cached PMFs because the baseline distribution changed
            self.cached_baseline_pmfs = [None] * self.embedding_dim
            self.cached_bin_edges = [None] * self.embedding_dim

        else:
            # Legacy vector-based approach
            batch_mean = embeddings.mean(axis=0)
            if self.count == 0:
                self.mean = batch_mean
            else:
                self.mean = (1 - self.alpha) * self.mean + self.alpha * batch_mean

            if self.distance_name == "mahalanobis":
                diff = batch_mean - self.mean
                self.var_diag = (1 - self.alpha) * self.var_diag + self.alpha * (
                    diff**2
                )

            self.count += 1

    def compute_distance(self, embeddings):
        """Compare a new batch's distribution to the baseline distribution
        (which is stored in self.kll_sketches or self.hist_counts).
        """
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]

        # For vector-based distances, just do the usual
        if not self.is_distribution_mode:
            return self._compute_vector_distance(embeddings)

        # For distribution-based: get a distribution for the new batch
        if self.distribution_impl == "kll":
            # Build a temporary KLL for each dimension
            new_klls = [kll_floats_sketch(self.k) for _ in range(self.embedding_dim)]

            for dim_idx in range(self.embedding_dim):
                new_klls[dim_idx].update(
                    np.asarray(embeddings[:, dim_idx], dtype=np.float32)
                )

        else:  # histogram
            new_counts = [
                np.zeros(self.num_bins, dtype=np.float64)
                for _ in range(self.embedding_dim)
            ]
            for dim_idx in range(self.embedding_dim):
                col_vals = embeddings[:, dim_idx]
                range_span = max(self.max_vals[dim_idx] - self.min_vals[dim_idx], 1e-12)
                bin_width = range_span / self.num_bins
                bin_indices = ((col_vals - self.min_vals[dim_idx]) / bin_width).astype(
                    int
                )
                bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
                np.add.at(new_counts[dim_idx], bin_indices, 1.0)

        # Now compute the distance dimension-by-dimension
        dim_distances = []
        for dim_idx in range(self.embedding_dim):
            if self.distribution_impl == "kll":
                # Get baseline PMF once, if cached
                baseline_pmf, edges = self._get_baseline_pmf_kll(dim_idx)
                # Get new PMF in one shot
                new_pmf = self._kll_to_pmf(new_klls[dim_idx], edges)

            else:
                baseline_pmf, edges = self._get_baseline_pmf_hist(dim_idx)
                new_pmf = new_counts[dim_idx]
                new_pmf = new_pmf / (new_pmf.sum() + 1e-12)

            # Compute the distance using whichever metric
            if self.distance_name in DISTRIBUTION_METRICS:
                dist_fn = DISTRIBUTION_METRICS[self.distance_name]
                dim_dist = dist_fn(baseline_pmf, new_pmf)
            elif self.distance_name == "wasserstein":
                dim_dist = approx_wasserstein_1d(edges, baseline_pmf, new_pmf)
            elif self.distance_name == "mmd":
                dim_dist = _mmd_1d_from_bins(edges, baseline_pmf, new_pmf)
            else:
                raise ValueError(
                    f"Unknown distribution-based metric: {self.distance_name}"
                )

            dim_distances.append(dim_dist)

        return float(np.mean(dim_distances)) if dim_distances else 0.0

    def _compute_vector_distance(self, embeddings):
        # For non-distribution distances (e.g. mahalanobis, euclidean, etc.)
        if self.distance_name == "mahalanobis":
            diff = embeddings.mean(axis=0) - self.mean
            epsilon = 1e-12
            return float(np.sqrt(np.sum(diff**2 / (self.var_diag + epsilon))))
        elif self.distance_name in VECTOR_DISTANCE_FUNCTIONS:
            dist_fn = VECTOR_DISTANCE_FUNCTIONS[self.distance_name]
            return float(dist_fn(self.mean, embeddings.mean(axis=0)))
        else:
            raise ValueError(f"Unknown vector-based distance: {self.distance_name}")

    def _get_baseline_pmf_kll(self, dim_idx):
        """Return the baseline PMF and the bin edges for dimension dim_idx,
        caching them so we don't recompute each time.
        """
        if self.cached_baseline_pmfs[dim_idx] is not None:
            return self.cached_baseline_pmfs[dim_idx], self.cached_bin_edges[dim_idx]

        # Build edges from [min_vals[dim_idx], max_vals[dim_idx]]
        minv = self.min_vals[dim_idx]
        maxv = self.max_vals[dim_idx]
        if maxv - minv < 1e-12:
            # Degenerate case
            edges = np.array([minv, minv + 1e-12])
            pmf = np.array([1.0])
        else:
            edges = np.linspace(minv, maxv, self.num_bins + 1)
            pmf = self._kll_to_pmf(self.kll_sketches[dim_idx], edges)

        # Cache
        self.cached_baseline_pmfs[dim_idx] = pmf
        self.cached_bin_edges[dim_idx] = edges
        return pmf, edges

    def _kll_to_pmf(self, sketch, edges):
        """Convert a KLL sketch to a discrete PMF across the given edges in one call,
        using sketch.get_pmf(split_points).
        We exclude the very first and last edges from 'split_points' because KLL
        will return an array of length len(split_points)+1.
        """
        if sketch.is_empty():
            return np.array([1.0]) if len(edges) > 1 else np.array([])

        # The library expects interior split points, so we skip edges[0] and edges[-1].
        # Then get_pmf(...) returns an array of length = (num_bins), matching our intervals.
        if len(edges) <= 2:
            # Degenerate (min==max case)
            return np.array([1.0])

        split_points = edges[1:-1]  # interior bin boundaries
        # get_pmf() returns e.g. shape = (len(split_points)+1,)
        pmf_array = sketch.get_pmf(split_points.tolist())

        # Ensure it's a NumPy array
        pmf_array = np.array(pmf_array, dtype=np.float64)
        return pmf_array / (pmf_array.sum() + 1e-12)

    def _get_baseline_pmf_hist(self, dim_idx):
        """For histogram-based baseline, just build normalized counts and a matching edge array.
        Cache it to avoid re-normalizing each time.
        """
        if self.cached_baseline_pmfs[dim_idx] is not None:
            return self.cached_baseline_pmfs[dim_idx], self.cached_bin_edges[dim_idx]

        minv = self.min_vals[dim_idx]
        maxv = self.max_vals[dim_idx]
        if maxv - minv < 1e-12:
            edges = np.array([minv, minv + 1e-12])
            pmf = np.array([1.0])
        else:
            edges = np.linspace(minv, maxv, self.num_bins + 1)
            counts = self.hist_counts[dim_idx]
            pmf = counts / (counts.sum() + 1e-12)

        self.cached_baseline_pmfs[dim_idx] = pmf
        self.cached_bin_edges[dim_idx] = edges
        return pmf, edges
