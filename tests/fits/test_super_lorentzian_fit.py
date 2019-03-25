import numpy as np
from unittest import TestCase
from fit1d.fits.super_lorentzian_fit import SuperLorentzianFit
from fit1d.outliers.outlier_by_distance import RemoveOutlierByDistance


class TestSuperLorentzian(TestCase):
    def setUp(self):
        outlier = RemoveOutlierByDistance
        self.x = np.linspace(-100, 100, 500)
        c = [30, 20, 10, 50]
        self.c = c
        self.y = c[1] + c[0]/(1 + ((self.x - c[2])/c[3])**40)
        self.fitter = SuperLorentzianFit(outlier=outlier)
        self.fitter.fit(self.x, self.y)
        self.fitter.eval(self.x)


    def test_init(self):
        self.assertIsInstance(self.fitter, SuperLorentzianFit)

    def test_superlorentzian_func(self):
        for ind, val in enumerate(SuperLorentzianFit._superlorentzian(self.x, self.c)):
            self.assertEqual(val, self.y[ind])

    def test_calc_eval(self):
        for ind, val in enumerate(self.fitter._fit_data.y_fit):
            self.assertAlmostEqual(val, self.y[ind], places=3)

    def test_fit(self):
        for ind, val in enumerate(self.fitter._fit_data.model.dump_model_to_list()):
            self.assertAlmostEqual(val, self.c[ind], places=3)
