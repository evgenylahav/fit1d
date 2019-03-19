import unittest
import numpy as np
from fit1d.common.model import ModelMock
from fit1d.common.fit1d import FitData


class TestFitData(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1,2, 3, 4])
        self.y = np.array([10,20, 30, 40])
        self.model = ModelMock({"param1": 5.5})

    def test_fit_data_empty_init(self):
        fit_ref = FitData()
        self.assertTrue(isinstance(fit_ref, FitData))

    def test_fit_data_partial_init(self):
        fit_ref = FitData(model=self.model, x=self.x)
        self.assertTrue(isinstance(fit_ref, FitData))

    def test_fit_data_full_init(self):
        fit_ref = FitData(model=self.model, x=self.x, y=self.y,
                          y_fit=self.y, error_vector=self.x, rms=0.5)
        self.assertTrue(isinstance(fit_ref, FitData))

    def test_fit_results_equal(self):
        fit_res1 = FitData(model=self.model, x=self.x, y=self.y,
                           y_fit=self.y, error_vector=self.x, rms=0.5)
        fit_res2 = FitData(model=self.model, x=self.x, y=self.y,
                           y_fit=self.y, error_vector=self.x, rms=0.5)
        self.assertTrue(fit_res1 == fit_res2)

    def test_fit_results_non_equal(self):
        fit_res1 = FitData(model=self.model, x=self.x, y=self.y,
                           y_fit=self.y, error_vector=self.x-1, rms=0.5)
        fit_res2 = FitData(model=self.model, x=self.x, y=self.y,
                           y_fit=self.y, error_vector=self.x, rms=0.5)

        self.assertFalse(fit_res1 == fit_res2)
