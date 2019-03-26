import unittest
from fit1d.fits.gaussian_fit import GaussianFit
from fit1d.models.gaussian_model import GaussianModel
from fit1d.common.outlier import OutLierMock
import numpy as np


class TestGaussianFit(unittest.TestCase):
    def test_gaussian_fit(self):
        gf = GaussianFit(OutLierMock)
        coef = [1, 0, 1, 0]
        x = np.linspace(-5, 5)
        y = coef[0] * np.exp(-(x - coef[1]) ** 2 / 2 / coef[2] ** 2) + coef[3]

        res = gf.fit(x, y)

        for ind, val in enumerate(res.model.dump_model_to_list()):
            self.assertAlmostEqual(val, coef[ind])

    def test_guassian_eval(self):
        gf = GaussianFit(OutLierMock)
        gm = GaussianModel()

        coef = [1, 0, 1, 0]
        gm.create_model_from_list(coef)
        x = np.linspace(-5, 5)
        y = coef[0] * np.exp(-(x - coef[1]) ** 2 / 2 / coef[2] ** 2) + coef[3]
        res = gf.eval(x, gm)

        for ind, val in enumerate(res):
            self.assertAlmostEqual(val, y[ind])
