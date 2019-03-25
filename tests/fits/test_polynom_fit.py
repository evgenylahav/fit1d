import unittest
import numpy as np
from fit1d.outliers.outlier_by_distance import RemoveOutlierByDistance
from fit1d.fits.polynom_fit import PolynomFit
from fit1d.common.outlier import OutLierMock


class TestPolyFit(unittest.TestCase):
    def test_empty(self):
        obj = PolynomFit()
        obj._fit_data.x = np.array([-2, -1, 0, 1, 2])
        obj._fit_data.y = np.array([4, 1, 0, 1, 4])
        obj.fit(obj._fit_data.x, obj._fit_data.y)
        expected_model = {"coefficients": [1, 0, 0]}
        self.assertEqual(obj.get_fit_data().model, expected_model)

    def test_calc_fit_linear(self):
        outlier_obj = RemoveOutlierByDistance()
        obj = PolynomFit(outlier=outlier_obj, remove_outliers=False, degree=1)
        obj._fit_data.x = np.array([0, 1, 2, 3])
        obj._fit_data.y = np.array([1, 1, 0, 1])
        obj._calc_fit()
        expected_model = {"coefficients": [-0.1, 0.9]}
        self.assertEqual(obj.get_fit_data().model, expected_model)

    def test_calc_fit_parab(self):
        outlier_obj = RemoveOutlierByDistance()
        obj = PolynomFit(outlier=outlier_obj, remove_outliers=False, degree=2)
        obj._fit_data.x = np.array([-2, -1, 0, 1, 2])
        obj._fit_data.y = np.array([4, 1, 0, 1, 4])
        obj.fit(obj._fit_data.x, obj._fit_data.y)
        expected_model = {"coefficients": [1, 0, 0]}
        self.assertEqual(obj.get_fit_data().model, expected_model)

    def test_calc_eval_parab_smiling(self):
        outlier_obj = RemoveOutlierByDistance()
        obj = PolynomFit(outlier=outlier_obj, remove_outliers=False, degree=2)
        obj._fit_data.x = np.array([-2, -1, 0, 1, 2])
        obj._fit_data.y = np.array([4, 1, 0, 1, 4])
        obj._calc_fit()
        obj._calc_eval()
        self.assertEqual(obj.get_fit_data().y_fit.any(), obj.get_fit_data().y.any())

    def test_calc_eval_parab_sad(self):
        outlier_obj = RemoveOutlierByDistance()
        obj = PolynomFit(outlier=outlier_obj, remove_outliers=False, degree=2)
        obj._fit_data.x = np.array([-2, -1, 0, 1, 2])
        obj._fit_data.y = np.array([-4, -1, 0, -1, -4])
        obj._calc_fit()
        obj._calc_eval()
        self.assertEqual(obj.get_fit_data().y_fit.any(), obj.get_fit_data().y.any())


