import numpy as np
from .mcda_method import MCDA_method

class SPOTIS(MCDA_method):
    def __init__(self):
        """Create SPOTIS method object.
        """
        pass

    def __call__(self, matrix, weights, types, bounds):
        """Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.

        Parameters
        ----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Criteria types. Profit criteria are represented by 1 and cost by -1.
            bounds: ndarray
                Bounds contain minimum and maximum values of each criterion. Minimum and maximum cannot be the same.

        Returns
        -------
            ndrarray
                Preference values of each alternative. The best alternative has the lowest preference value. 
        """
        SPOTIS._verify_input_data(matrix, weights, types)
        return SPOTIS._spotis(matrix, weights, types, bounds)

    @staticmethod
    def _spotis(matrix, weights, types, bounds):
        # Determine Ideal Solution Point (ISP)
        isp = np.zeros(matrix.shape[1])
        isp[types == 1] = bounds[1, types == 1]
        isp[types == -1] = bounds[0, types == -1]

        # Calculate normalized distances
        norm_matrix = np.abs(matrix - isp) / np.abs(bounds[1, :] - bounds[0, :])
        # Calculate the normalized weighted average distance
        D = np.sum(weights * norm_matrix, axis = 1)
        return D
