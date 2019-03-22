import unittest
import numpy as np
from fit1d.outliers.outlier_by_distance import RemoveOutlierByDistance


class TestOutlierByDist(unittest.TestCase):
    def test_empty(self):
        outlier_obj = RemoveOutlierByDistance()
        self.assertEqual([outlier_obj.max_error, outlier_obj.min_points], [1, 10])

    def test_find_outliers_min_points_exceeds_vector(self):
        outlier_obj = RemoveOutlierByDistance(min_points=6, max_error=1)
        error_vec = np.array([0, 1, 0, 0, 3])
        indexes = outlier_obj.find_outliers(error_vec)
        self.assertEqual(indexes, [])

    def test_find_outliers_max_error_exceeds_vector(self):
        outlier_obj = RemoveOutlierByDistance(min_points=2, max_error=4)
        error_vec = np.array([0, 1, 0, 0, 3, 1, 2, 1, 2, 0, 0, 1, 2, 1])
        indexes = outlier_obj.find_outliers(error_vec)
        self.assertEqual(indexes, [])

    def test_find_outliers_empty(self):
        outlier_obj = RemoveOutlierByDistance()
        error_vec = np.array([0, 1, 0, 0, 3, 1, 2, 1, 2, 0, 0, 1, 2, 1])
        indexes = outlier_obj.find_outliers(error_vec)
        self.assertEqual(indexes, [4])

    def test_find_outliers_picks_first_if_tie(self):
        outlier_obj = RemoveOutlierByDistance()
        error_vec = np.array([0, 1, 0, 0, 3, 1, 2, 1, 2, 0, 0, 1, 2, 1, 3])
        indexes = outlier_obj.find_outliers(error_vec)
        self.assertEqual(indexes, [4])

    def test_find_outliers_max_error(self):
        outlier_obj = RemoveOutlierByDistance(max_error=1)
        error_vec = np.array([0, 1, 0, 0, 3, 1, 2, 1, 2, 0, 0, 1, 2, 1])
        indexes = outlier_obj.find_outliers(error_vec)
        self.assertEqual(indexes, [4])

    def test_find_outliers_min_points(self):
        outlier_obj = RemoveOutlierByDistance(min_points=3)
        error_vec = np.array([0, 1, 0, 0, 3])
        indexes = outlier_obj.find_outliers(error_vec)
        self.assertEqual(indexes, [4])
