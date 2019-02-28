import unittest
import numpy as np
from collections import namedtuple
from fit1d.model import Model, ModelMock
from fit1d.outlier import OutLier, OutLierMock
from fit1d.fit1d import Fit1D, Fit1DMock


class TestFit1D(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(-15.0, 15.0, num=50)
        self.y = self.x
        self.model = ModelMock({"param1": 5.5})
        self.outlier = OutLierMock({'param1': 10, 'param2': 5.4, 'param3': 'hello'})
        self.fit1d = Fit1DMock(model=self.model, outlier=self.outlier, exclude_outliers_removal=False)

    def test_fit(self):

        fit = self.fit1d.fit(self.x, self.y)
        fit_results = namedtuple('fit_results', ['a', 'b', 'c'])
        fit_ref = fit_results(a=5, b=5.6, c="hello")
        # print(fit == fit_ref)
        self.assertTrue(fit == fit_ref)

    def test_evaluate_with_model(self):
        e_val = self.fit1d.eval_with_model(self.x, self.model)
        e_ref = np.array([1, 2, 3, 4])
        self.assertTrue(np.array_equal(e_val, e_ref))

    def test_eval(self):
        e_val = self.fit1d.eval(self.x)
        e_ref = np.array([1, 2, 3, 4])
        self.assertTrue(np.array_equal(e_val, e_ref))

    def test_remove_outliers(self):
        o_val = self.fit1d.remove_outlier(self.x, self.y)
        o_ref = [1, 2, 3]
        self.assertListEqual(o_val, o_ref)

    def test_calc_err(self):
        x = np.array([1, 10, 100, 1000])
        y = np.array([10, -10, 100, 2000])
        dif_ref = np.array([-9, 20, 0, -1000])
        dif_val = self.fit1d.calc_error(y=x, y_fit=y)
        self.assertTrue(np.array_equal(dif_ref, dif_val))

    def test_calc_rms(self):
        x = np.array([1, 10, 100, 1000])
        y = np.array([10, -10, 100, 2000])
        rms_ref = sum(np.array([-9, 20, 0, -1000]) ** 2)
        rms_val = self.fit1d.calc_rms(residual=x-y)
        self.assertTrue(np.array_equal(rms_ref, rms_val))

    def test_get_model(self):
        model_ref = self.model
        self.assertTrue(model_ref == self.fit1d.get_model())

