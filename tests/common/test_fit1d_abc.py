import unittest
import numpy as np
from fit1d.common.model import ModelMock
from fit1d.common.outlier import OutLierMock
from fit1d.common.fit1d import Fit1DMock, FitData


class TestFit1D(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1, 2, 3, 4])
        self.y = np.array([10, 20, 30, 40])
        self.model = ModelMock({"param1": 5.5})
        self.outlier = OutLierMock({'param1': 10, 'param2': 5.4, 'param3': 'hello'})
        self.fit1d = Fit1DMock(outlier=self.outlier, remove_outliers=False)

    def test_fit(self):
        fit = self.fit1d.fit(self.x, self.y)
        fit_ref = FitData(x=self.x,
                          y=self.y,
                          y_fit=np.array([11, 22, 33, 44]),
                          model=self.model,
                          error_vector=np.array([-1, -2, -3, -4]),
                          rms=sum((np.array([-1, -2, -3, -4]) ** 2) / 4) ** 0.5)
        self.assertTrue(fit == fit_ref)

    def test_fit_with_removing_outliers(self):
        fit1d = Fit1DMock(outlier=self.outlier, remove_outliers=True)
        fit = fit1d.fit(self.x, self.y)
        fit_ref = FitData(x=np.array([1, 3, 4]),
                          y=np.array([10, 30, 40]),
                          y_fit=np.array([11, 33, 44]),
                          model=self.model,
                          error_vector=np.array([-1, -3, -4]),
                          rms=sum((np.array([-1, -3, -4]) ** 2) / 3) ** 0.5)
        self.assertTrue(fit == fit_ref)

    def test_eval_without_inputs(self):
        y_eval = self.fit1d.eval()
        y_ref = np.array([11, 22, 33, 44])
        self.assertTrue(np.array_equal(y_eval, y_ref))

    def test_eval_with_x_only(self):
        y_eval = self.fit1d.eval(self.x)
        y_ref = np.array([11, 22, 33, 44])
        self.assertTrue(np.array_equal(y_eval, y_ref))

    def test_eval_with_all_inputs(self):
        y_eval = self.fit1d.eval(self.x, self.model)
        y_ref = np.array([11, 22, 33, 44])
        self.assertTrue(np.array_equal(y_eval, y_ref))

    def test_fit_data_after_eval(self):
        y_fit = self.fit1d.eval(self.x, self.model)
        y_fit_ref = np.array([11, 22, 33, 44])
        self.assertTrue(np.array_equal(y_fit, y_fit_ref))

