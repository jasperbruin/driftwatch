import numpy as np
from kll_sketch import KLL

class KLLTransformer:
    """
    Dimension-wise rank-based transform using KLL sketches.
    - fit(X): builds one KLL per dimension from the baseline embeddings X.
    - transform(X): for each dimension, returns the approximate CDF value
      i.e., rank(x) / total_count, preserving the distribution shape.
    """
    def __init__(self, k=50):
        """
        :param k: base buffer size for each KLL. Tune as needed for accuracy/speed.
        """
        self.k = k
        self.kll_sketches = None
        self.total_counts = None

    def fit(self, X):
        """
        X is shape (num_samples, embedding_dim).
        We create embedding_dim KLL sketches (one per column).
        """
        num_samples, embedding_dim = X.shape
        self.kll_sketches = [KLL(self.k) for _ in range(embedding_dim)]
        self.total_counts = [0] * embedding_dim

        # Feed each dimension's column into its KLL
        for dim_index in range(embedding_dim):
            col_values = X[:, dim_index]
            for val in col_values:
                self.kll_sketches[dim_index].insert(val)
            self.total_counts[dim_index] = self.kll_sketches[dim_index].size

        return self

    def transform(self, X):
        """
        Convert each entry X[i, dim] into rank(X[i, dim]) / total_count for that dimension.
        """
        num_samples, embedding_dim = X.shape
        X_out = np.zeros_like(X, dtype=np.float32)

        for dim_index in range(embedding_dim):
            kll = self.kll_sketches[dim_index]
            total_size = float(self.total_counts[dim_index])

            # For each value in this dimension, compute approximate rank
            for i in range(num_samples):
                val = X[i, dim_index]
                # rank() returns the #elements <= val
                approx_rank = kll.rank(val)
                # scale to [0,1]
                X_out[i, dim_index] = approx_rank / max(1.0, total_size)

        return X_out
