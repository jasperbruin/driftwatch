import numpy as np
from scipy.spatial.distance import euclidean, cosine, mahalanobis, cityblock, minkowski, chebyshev
from scipy.stats import wasserstein_distance, entropy
import scipy.spatial.distance as scipy_dist

# Define vector distance functions
VECTOR_DISTANCE_FUNCTIONS = {
    "euclidean": euclidean,
    "cosine": cosine,
    "mahalanobis": lambda u, v, VI: mahalanobis(u, v, VI),
    "manhattan": cityblock,
    "minkowski": minkowski,
    "chebyshev": chebyshev,
    "canberra": scipy_dist.canberra
}

# Distribution metrics
def kl_divergence(p, q):
    """Calculate KL divergence between two distributions."""
    return entropy(p, q)

def js_divergence(p, q):
    """Calculate Jensen-Shannon divergence between two distributions."""
    # Add small epsilon to avoid log(0)
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def hellinger_distance(p, q):
    """Calculate Hellinger distance between two distributions."""
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

def bhattacharyya_distance(p, q):
    """Calculate Bhattacharyya distance between two distributions."""
    return -np.log(np.sum(np.sqrt(p * q)))

def approx_wasserstein_1d(p, q):
    """Calculate 1D Wasserstein distance between distributions."""
    return wasserstein_distance(p, q)

def _mmd_1d_from_bins(p, q, sigma=1.0):
    """Maximum Mean Discrepancy for 1D distributions from binned data."""
    # Simple implementation of MMD using Gaussian kernel
    p_grid, q_grid = np.meshgrid(p, q)
    K_pp = np.exp(-((p_grid - p_grid.T) ** 2) / (2 * sigma ** 2))
    K_qq = np.exp(-((q_grid - q_grid.T) ** 2) / (2 * sigma ** 2))
    K_pq = np.exp(-((p_grid - q_grid.T) ** 2) / (2 * sigma ** 2))
    
    return np.mean(K_pp) + np.mean(K_qq) - 2 * np.mean(K_pq)

# Dictionary of distribution metrics
DISTRIBUTION_METRICS = {
    "kl": kl_divergence,
    "js": js_divergence,
    "hellinger": hellinger_distance,
    "bhattacharyya": bhattacharyya_distance,
    "wasserstein": approx_wasserstein_1d,
    "mmd": _mmd_1d_from_bins
}

def get_available_metrics():
    """
    Returns a list of all available distance metrics.
    
    Returns:
        list: Names of all available distance metrics (both vector-based and distribution-based)
    """
    # Combine all available metrics from vector and distribution metrics
    all_metrics = list(VECTOR_DISTANCE_FUNCTIONS.keys()) + list(DISTRIBUTION_METRICS.keys())
    return all_metrics
