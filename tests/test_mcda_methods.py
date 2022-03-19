from evo_spotis.mcda_methods import SPOTIS
from evo_spotis.additions import rank_preferences

import unittest
import numpy as np

# Test for SPOTIS method
class Test_SPOTIS(unittest.TestCase):

    def test_spotis(self):
        """Test based on paper Dezert, J., Tchamova, A., Han, D., & Tacnet, J. M. (2020, July). 
        The SPOTIS rank reversal free method for multi-criteria decision-making support. 
        In 2020 IEEE 23rd International Conference on Information Fusion (FUSION) (pp. 1-8). 
        IEEE."""


        matrix = np.array([[15000, 4.3, 99, 42, 737],
                 [15290, 5.0, 116, 42, 892],
                 [15350, 5.0, 114, 45, 952],
                 [15490, 5.3, 123, 45, 1120]])

        weights = np.array([0.2941, 0.2353, 0.2353, 0.0588, 0.1765])
        types = np.array([-1, -1, -1, 1, 1])
        bounds_min = np.array([14000, 3, 80, 35, 650])
        bounds_max = np.array([16000, 8, 140, 60, 1300])
        bounds = np.vstack((bounds_min, bounds_max))


        method = SPOTIS()
        test_result = method(matrix, weights, types, bounds)
        real_result = np.array([0.4779, 0.5781, 0.5558, 0.5801])
        self.assertEqual(list(np.round(test_result, 3)), list(np.round(real_result, 3)))


# Test for rank preferences
class Test_Rank_preferences(unittest.TestCase):

    def test_rank_preferences(self):
        """Test based on paper Dezert, J., Tchamova, A., Han, D., & Tacnet, J. M. (2020, July). 
        The SPOTIS rank reversal free method for multi-criteria decision-making support. 
        In 2020 IEEE 23rd International Conference on Information Fusion (FUSION) (pp. 1-8). 
        IEEE."""

        pref = np.array([0.4779, 0.5781, 0.5558, 0.5801])
        test_result = rank_preferences(pref , reverse = False)
        real_result = np.array([1, 3, 2, 4])
        self.assertEqual(list(test_result), list(real_result))


def main():
    test_spotis = Test_SPOTIS()
    test_spotis.test_spotis()

    test_rank_preferences = Test_Rank_preferences()
    test_rank_preferences.test_rank_preferences()


if __name__ == '__main__':
    main()