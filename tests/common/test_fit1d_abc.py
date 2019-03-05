import unittest
import numpy as np
from fit1d.common.model import ModelMock
from fit1d.common.outlier import OutLierMock
from fit1d.common.fit1d import Fit1DMock, FitResults


class TestFit1D(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1,2, 3, 4])
        self.y = np.array([10,20, 30, 40])
        self.model = ModelMock({"param1": 5.5})
        self.outlier = OutLierMock({'param1': 10, 'param2': 5.4, 'param3': 'hello'})
        self.fit1d = Fit1DMock(model=self.model, outlier=self.outlier, remove_outliers=False)

    def test_fit_results_init(self):
        fit_ref = FitResults(model=self.model,
                             error_vector=np.array([1, 10, 100, 1000]),
                             rms=5.4,
                             outlier=self.outlier)

        self.assertTrue(isinstance(fit_ref, FitResults))

    def test_fit_results_equal(self):
        fit_res1 = FitResults(model=self.model,
                              error_vector=np.array([1, 10, 100, 1000]),
                              rms=5.4,
                              outlier=self.outlier)
        fit_res2 = FitResults(model=self.model,
                              error_vector=np.array([1, 10, 100, 1000]),
                              rms=5.4,
                              outlier=self.outlier)

        self.assertTrue(fit_res1 == fit_res2)

    def test_fit_results_nonequal(self):
        fit_res1 = FitResults(model=self.model,
                              error_vector=np.array([1, 100, 100, 1000]),
                              rms=5.4,
                              outlier=self.outlier)
        fit_res2 = FitResults(model=self.model,
                              error_vector=np.array([1, 10, 100, 1000]),
                              rms=5.4,
                              outlier=self.outlier)

        self.assertFalse(fit_res1 == fit_res2)


    def test_fit(self):
        fit = self.fit1d.fit(self.x, self.y)
        fit_ref = FitResults(model=self.model,
                             error_vector=np.array([-1, -2, -3, -4]),
                             rms=self.fit1d.calc_rms(np.array([1, 2, 3, 4])),
                             outlier=[])
        self.assertTrue(fit == fit_ref)

    def test_evaluate_with_model(self):
        e_val = self.fit1d.eval(self.x, self.model)
        e_ref = np.array([1, 2, 3, 4])
        self.assertTrue(np.array_equal(e_val, e_ref))

    def test_eval(self):
        e_val = self.fit1d.eval(self.x)
        e_ref = np.array([1, 2, 3, 4])
        self.assertTrue(np.array_equal(e_val, e_ref))

    def test_remove_outliers(self):
        o_val = self.fit1d._remove_outlier(self.x, self.y)
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
        residual = self.fit1d.calc_error(y=x, y_fit=y)
        rms_ref = (sum(residual ** 2) / len(residual)) ** 0.5
        rms_val = self.fit1d.calc_rms(residual=residual)
        self.assertTrue(np.array_equal(rms_ref, rms_val))

    def test_get_model(self):
        model_ref = self.model
        self.assertTrue(model_ref == self.fit1d.get_model())

