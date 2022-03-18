import numpy as np
from .normalizations import sum_normalization
import itertools


# Entropy weighting
def entropy_weighting(X, types):
    """
    Calculate criteria weights using objective Entropy weighting method.

    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
        
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    # normalize the decision matrix with sum_normalization method from normalizations as for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    pij = sum_normalization(X, criteria_type)
    # Transform negative values in decision matrix X to positive values
    pij = np.abs(pij)
    m, n = np.shape(pij)
    H = np.zeros((m, n))

    # Calculate entropy
    for j, i in itertools.product(range(n), range(m)):
        if pij[i, j]:
            H[i, j] = pij[i, j] * np.log(pij[i, j])

    h = np.sum(H, axis = 0) * (-1 * ((np.log(m)) ** (-1)))

    # Calculate degree of diversification
    d = 1 - h

    # Set w as the degree of importance of each criterion
    w = d / (np.sum(d))
    return w