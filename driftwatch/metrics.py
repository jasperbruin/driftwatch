import numpy as np

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

def approx_wasserstein_1d(bin_edges, pmf1, pmf2):
    """
    Approximate 1D Wasserstein distance between two discrete distributions.
    We unify bin_edges, build CDFs, and numerically integrate |CDF1 - CDF2|.
    """
    # pmf1 & pmf2 should be aligned to the same bin_edges
    pmf1 = pmf1 / (pmf1.sum() + 1e-12)
    pmf2 = pmf2 / (pmf2.sum() + 1e-12)

    cdf1 = np.cumsum(pmf1)
    cdf2 = np.cumsum(pmf2)

    distance = 0.0
    for i in range(len(bin_edges) - 1):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        distance += abs(cdf1[i] - cdf2[i]) * (right - left)

    return distance

def _rbf_kernel_1d(x, y, sigma):
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    XX = (X*X).sum(axis=1, keepdims=True)
    YY = (Y*Y).sum(axis=1, keepdims=True)
    dists = XX + YY.T - 2 * np.dot(X, Y.T)
    return np.exp(-dists / (2 * sigma**2))

def _mmd_1d_from_bins(bin_edges, pmf1, pmf2, kernel='rbf', sigma=1.0):
    """
    Approximate MMD in 1D using the bin centers as points, with
    pmf1, pmf2 as their weights.
    """
    pmf1 = pmf1 / (pmf1.sum() + 1e-12)
    pmf2 = pmf2 / (pmf2.sum() + 1e-12)

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    Kxx = _rbf_kernel_1d(centers, centers, sigma)
    Kyy = Kxx  # same dimension
    Kxy = Kxx  # re-use shape, but we'll index differently

    # Compute E[k(x,x)] under pmf1
    Exx = 0.0
    for i in range(len(centers)):
        for j in range(len(centers)):
            Exx += pmf1[i] * pmf1[j] * Kxx[i, j]

    # Compute E[k(y,y)] under pmf2
    Eyy = 0.0
    for i in range(len(centers)):
        for j in range(len(centers)):
            Eyy += pmf2[i] * pmf2[j] * Kyy[i, j]

    # Compute E[k(x,y)] under pmf1, pmf2
    Exy = 0.0
    for i in range(len(centers)):
        for j in range(len(centers)):
            Exy += pmf1[i] * pmf2[j] * Kxy[i, j]

    return Exx + Eyy - 2 * Exy