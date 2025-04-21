import numpy as np
import scipy.spatial.distance as scipy_dist
from scipy.spatial.distance import (
    chebyshev,
    cityblock,
    cosine,
    euclidean,
    mahalanobis,
    minkowski,
)
from scipy.stats import entropy, wasserstein_distance

# Define vector distance functions
VECTOR_DISTANCE_FUNCTIONS = {
    "euclidean": euclidean,
    "cosine": cosine,
    "mahalanobis": lambda u, v, VI: mahalanobis(u, v, VI),
    "manhattan": cityblock,
    "minkowski": minkowski,
    "chebyshev": chebyshev,
    "canberra": scipy_dist.canberra,
}


# Distribution metrics
def kl_divergence(p, q):
    """Calculate KL divergence between two distributions."""
    # Add small epsilon to avoid log(0)
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10

    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)

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
    # Add small epsilon and normalize
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    p = p / np.sum(p)
    q = q / np.sum(q)

    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def bhattacharyya_distance(p, q):
    """Calculate Bhattacharyya distance between two distributions."""
    # Add small epsilon and normalize
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    p = p / np.sum(p)
    q = q / np.sum(q)

    return -np.log(np.sum(np.sqrt(p * q)))


def approx_wasserstein_1d(edges, p, q):
    """Calculate 1D Wasserstein distance between distributions.

    Args:
        edges: bin edges for the distributions
        p: first distribution (probabilities)
        q: second distribution (probabilities)

    """
    # For wasserstein distance we need to convert PMFs to CDFs
    # and use the bin centers as the values
    bin_centers = (edges[1:] + edges[:-1]) / 2

    # Normalize distributions
    p = np.array(p) / (np.sum(p) + 1e-10)
    q = np.array(q) / (np.sum(q) + 1e-10)

    return wasserstein_distance(bin_centers, bin_centers, p, q)


def _mmd_1d_from_bins(edges, p, q, sigma=1.0):
    """Maximum Mean Discrepancy for 1D distributions from binned data.

    Args:
        edges: bin edges for the distributions
        p: first distribution (probabilities)
        q: second distribution (probabilities)
        sigma: kernel bandwidth parameter

    """
    # Get bin centers as locations
    bin_centers = (edges[1:] + edges[:-1]) / 2

    # Normalize distributions
    p = np.array(p) / (np.sum(p) + 1e-10)
    q = np.array(q) / (np.sum(q) + 1e-10)

    # Create weighted samples
    p_samples = np.repeat(bin_centers, np.round(p * 1000).astype(int))
    q_samples = np.repeat(bin_centers, np.round(q * 1000).astype(int))

    # If we don't have enough samples, return 0
    if len(p_samples) < 2 or len(q_samples) < 2:
        return 0.0

    # Compute kernel matrices
    p_grid, q_grid = np.meshgrid(p_samples, q_samples)
    K_pp = np.exp(-((p_samples[:, None] - p_samples[None, :]) ** 2) / (2 * sigma**2))
    K_qq = np.exp(-((q_samples[:, None] - q_samples[None, :]) ** 2) / (2 * sigma**2))
    K_pq = np.exp(-((p_samples[:, None] - q_samples[None, :]) ** 2) / (2 * sigma**2))

    return np.mean(K_pp) + np.mean(K_qq) - 2 * np.mean(K_pq)


# Dictionary of distribution metrics
DISTRIBUTION_METRICS = {
    "kl": kl_divergence,
    "js": js_divergence,
    "hellinger": hellinger_distance,
    "bhattacharyya": bhattacharyya_distance,
    # wasserstein and mmd are handled specially in EmbeddingTracker.compute_distance
}


def get_available_metrics():
    """Returns a list of all available distance metrics.

    Returns:
        list: Names of all available distance metrics (both vector-based and distribution-based)

    """
    # Combine all available metrics from vector and distribution metrics
    all_metrics = (
        list(VECTOR_DISTANCE_FUNCTIONS.keys())
        + list(DISTRIBUTION_METRICS.keys())
        + ["wasserstein", "mmd"]
    )
    return all_metrics
