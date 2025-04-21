import math
import numpy as np
from kll_sketch import KLL


class DistanceFunctions:
    """Encapsulates all distance functions, including Mahalanobis handling."""

    @staticmethod
    def euclidean(x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    @staticmethod
    def manhattan(x, y):
        return np.sum(np.abs(x - y))

    @staticmethod
    def minkowski(x, y, p=3):
        return np.sum(np.abs(x - y) ** p) ** (1.0 / p)

    @staticmethod
    def chebyshev(x, y):
        return np.max(np.abs(x - y))

    @staticmethod
    def canberra(x, y):
        numerator = np.abs(x - y)
        denominator = np.abs(x) + np.abs(y) + 1e-12  # Avoid division by zero
        return np.sum(numerator / denominator)

    @staticmethod
    def mahalanobis(x, mean, cov_inv):
        diff = x - mean
        return np.sqrt(diff.T @ cov_inv @ diff)

    DISTANCE_FUNCTIONS = {
        "euclidean": euclidean.__func__,
        "manhattan": manhattan.__func__,
        "minkowski": lambda u, v: DistanceFunctions.minkowski(u, v, p=3),
        "chebyshev": chebyshev.__func__,
        "canberra": canberra.__func__,
    }


class EmbeddingTracker:
    """
    A tracker that maintains:
    - An exponential moving average (EMA) of the embedding mean
    - Optionally an EMA-based covariance for Mahalanobis distance
      *with optional diagonal or block-diagonal approximations*.
    - An optional KLL sketch for rank-based statistics of drift distances.
    """

    def __init__(
        self,
        embedding_dim,
        alpha=0.01,
        distance_name="mahalanobis",
        use_kll=False,
        kll_k=200,
        cov_approx_mode="diagonal",
        block_size=32,
    ):
        """
        :param embedding_dim: Dimension of the embeddings
        :param alpha: Smoothing factor for EMA updates
        :param distance_name: "mahalanobis" or one of the DISTANCE_FUNCTIONS keys
        :param use_kll: If True, maintain a KLL sketch of drift distances
        :param kll_k: The 'k' parameter for KLL (sketch size)
        :param cov_approx_mode: "full", "diagonal", or "block"
        :param block_size: if using "block" approximation, each block is block_size x block_size
        """
        self.alpha = alpha
        self.distance_name = distance_name
        self.use_kll = use_kll
        self.cov_approx_mode = cov_approx_mode.lower()
        self.block_size = block_size

        # Basic stats for mean/cov updates
        self.mean = np.zeros((embedding_dim,))
        self.count = 0

        # Build or initialize the covariance structures if we're doing Mahalanobis
        if distance_name == "mahalanobis":
            if self.cov_approx_mode == "full":
                self.cov = np.eye(embedding_dim)
            elif self.cov_approx_mode == "diagonal":
                self.var_diag = np.ones(embedding_dim)
            elif self.cov_approx_mode == "block":
                self.blocks = []
                self.block_indices = []
                start = 0
                while start < embedding_dim:
                    end = min(start + block_size, embedding_dim)
                    size = end - start
                    self.blocks.append(np.eye(size))
                    self.block_indices.append((start, end))
                    start += block_size
            else:
                raise ValueError(f"Unknown cov_approx_mode: {self.cov_approx_mode}")

            self.distance_fn = self._mahalanobis_distance

        else:
            # For non-mahalanobis, look up from the DistanceFunctions class
            if distance_name not in DistanceFunctions.DISTANCE_FUNCTIONS:
                raise ValueError(f"Unknown distance_name: {distance_name}")

            dist_fn = DistanceFunctions.DISTANCE_FUNCTIONS[distance_name]

            def closure_for_dict_fn(embedding):
                return dist_fn(self.mean, embedding)

            self.distance_fn = closure_for_dict_fn

        # If requested, set up the KLL sketch
        if self.use_kll:
            self.distance_kll = KLL(kll_k)

    def _mahalanobis_distance(self, embedding):
        """Compute Mahalanobis distance between `embedding` and `self.mean`."""
        diff = embedding - self.mean

        if self.cov_approx_mode == "full":
            cov_inv = np.linalg.pinv(self.cov)
            return DistanceFunctions.mahalanobis(embedding, self.mean, cov_inv)

        elif self.cov_approx_mode == "diagonal":
            safe_var = np.where(self.var_diag > 1e-12, self.var_diag, 1e-12)
            return np.sqrt(np.sum((diff**2) / safe_var))

        elif self.cov_approx_mode == "block":
            total = 0.0
            for i, (start, end) in enumerate(self.block_indices):
                d_block = diff[start:end]
                block_inv = np.linalg.pinv(self.blocks[i])
                total += d_block.T @ block_inv @ d_block
            return math.sqrt(total)

        else:
            raise ValueError(f"Unknown cov_approx_mode: {self.cov_approx_mode}")

    def compute_distance(self, embedding):
        """Compute distance using the assigned `self.distance_fn`."""
        return self.distance_fn(embedding)

    def update(self, embedding):
        if self.count == 0:
            self.mean = embedding.copy()
            if self.distance_name == "mahalanobis":
                if self.cov_approx_mode == "full":
                    self.cov = np.eye(len(embedding))
                elif self.cov_approx_mode == "diagonal":
                    self.var_diag = np.ones_like(embedding)
                elif self.cov_approx_mode == "block":
                    for i, (start, end) in enumerate(self.block_indices):
                        size = end - start
                        self.blocks[i] = np.eye(size)
        else:
            self.mean = (1 - self.alpha) * self.mean + self.alpha * embedding

            if self.distance_name == "mahalanobis":
                diff_new = embedding - self.mean
                if self.cov_approx_mode == "full":
                    outer = np.outer(diff_new, diff_new)
                    self.cov = (1 - self.alpha) * self.cov + self.alpha * outer

                elif self.cov_approx_mode == "diagonal":
                    self.var_diag = (1 - self.alpha) * self.var_diag + self.alpha * (
                        diff_new**2
                    )

                elif self.cov_approx_mode == "block":
                    for i, (start, end) in enumerate(self.block_indices):
                        d_block = diff_new[start:end]
                        outer_block = np.outer(d_block, d_block)
                        self.blocks[i] = (1 - self.alpha) * self.blocks[
                            i
                        ] + self.alpha * outer_block

        self.count += 1

        if self.use_kll:
            dist = self.compute_distance(embedding)
            self.distance_kll.insert(dist)

    def get_distance_quantile(self, q):
        """Return approximate q-th quantile from the KLL sketch."""
        if not self.use_kll:
            raise ValueError("KLL not enabled in this tracker. Set use_kll=True.")

        cdf_vals = self.distance_kll.cdf()
        if not cdf_vals:
            return 0.0

        for value, c_prob in cdf_vals:
            if c_prob >= q:
                return value
        return cdf_vals[-1][0]
