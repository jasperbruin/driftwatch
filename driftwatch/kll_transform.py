import numpy as np
from datasketches import kll_floats_sketch

class KLLTransformer:
    """
    Dimension-wise rank-based transformation using KLL (K-Quantile) sketches.
    """

    def __init__(self, k=50):
        """
        :param k: Parameter that determines the accuracy and size of the sketch.
        """
        # Ensure k is within the valid range of [8, 65535]
        if not isinstance(k, int) or k < 8 or k > 65535:
            raise ValueError("k must be an integer between 8 and 65,535.")

        self.k = k
        self.kll_sketches = None

    def fit(self, X):
        """
        Fit the transformer to the data X (shape = (num_samples, embedding_dim)).
        Creates one KLL sketch per column in X.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array, e.g., shape (num_samples, embedding_dim).")

        num_samples, embedding_dim = X.shape
        if embedding_dim == 0:
            raise ValueError("X must have at least one column.")
        if num_samples == 0:
            pass  # Allow fitting on empty datasets, though it may not be useful

        # Create a separate KLL sketch for each dimension
        self.kll_sketches = [kll_floats_sketch(self.k) for _ in range(embedding_dim)]

        # Feed each column's data into the corresponding sketch
        for dim_index in range(embedding_dim):
            col_values = X[:, dim_index]
            for val in col_values:
                self.kll_sketches[dim_index].update(val)

        return self  # scikit-learn convention

    def transform(self, X):
        """
        Transform the data X by converting each value into its approximate rank in [0, 1].
        """
        if self.kll_sketches is None:
            raise AttributeError("The KLLTransformer has not been fitted. Call fit() before transform().")

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        num_samples, embedding_dim = X.shape

        if embedding_dim != len(self.kll_sketches):
            raise ValueError(
                f"Inconsistent dimension in transform. Expected {len(self.kll_sketches)} columns, but got {embedding_dim}."
            )

        X_out = np.zeros_like(X, dtype=np.float32)

        for dim_index, kll in enumerate(self.kll_sketches):
            for i in range(num_samples):
                val = X[i, dim_index]
                approx_rank = kll.get_rank(val)
                X_out[i, dim_index] = approx_rank

        return X_out
