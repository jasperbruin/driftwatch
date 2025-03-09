import numpy as np

class EmbeddingTracker:
    """
    A tracker that maintains:
    - An exponential-moving-average (EMA) of the embedding mean
    - For Mahalanobis: A diagonal approximation to the covariance
    - A selected distance metric from one of the DISTANCE_FUNCTIONS
    """
    def __init__(self, embedding_dim, alpha=0.01, distance_name="mahalanobis"):
        self.alpha = alpha
        self.distance_name = distance_name
        self.mean = np.zeros((embedding_dim,))
        self.count = 0
        self.var_diag = np.ones((embedding_dim,))

    def update(self, embedding):
        """Update the running mean and (if Mahalanobis) the diagonal of the covariance."""
        if self.count == 0:
            self.mean = embedding
            if self.distance_name == "mahalanobis":
                self.var_diag = np.ones_like(embedding)
        else:
            self.mean = (1 - self.alpha) * self.mean + self.alpha * embedding

            if self.distance_name == "mahalanobis":
                diff = embedding - self.mean
                self.var_diag = (1 - self.alpha) * self.var_diag + self.alpha * (diff ** 2)

        self.count += 1

    def compute_distance(self, embedding):
        """Compute distance between 'embedding' and the tracker's current mean."""
        if self.distance_name == "mahalanobis":
            diff = embedding - self.mean
            epsilon = 1e-12
            return np.sqrt(np.sum(diff**2 / (self.var_diag + epsilon)))
        else:
            dist_fn = DISTANCE_FUNCTIONS[self.distance_name]
            return dist_fn(self.mean, embedding)


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

DISTANCE_FUNCTIONS = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "minkowski": lambda u, v: minkowski_distance(u, v, p=3),  # example p=3
    "chebyshev": chebyshev_distance,
    "canberra": canberra_distance,
}
