import itertools
import numpy as np
from .normalizations import sum_normalization


# Entropy weighting
def entropy_weighting(matrix):
    """
    Calculate criteria weights using objective Entropy weighting method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        
    Returns
    --------
        ndarray
            Vector of criteria weights

    Examples
    ----------
    >>> weights = entropy_weighting(matrix)
    """
    # normalize the decision matrix with sum_normalization method from normalizations as for profit criteria
    types = np.ones(np.shape(matrix)[1])
    pij = sum_normalization(matrix, types)
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