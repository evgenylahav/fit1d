import unittest
import math
from fit1d.gaussian import Gaussian
import numpy as np


class TestGaussian(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-15.0, 15.0, num=50)
        self.model = {'amplitude': 10.0, 'mean': 1.5, 'sigma': 0.8}
        self.y = self.model['amplitude'] * math.e ** (-0.5 * ((self.x - self.model['mean']) / self.model['sigma']) ** 2)

    def test_fit(self):
        model_out = Gaussian.fit(self.x, self.y)
        for key, value in model_out.items():
            self.assertAlmostEqual(value, self.model[key], 4)

    def test_eval(self):
        y = Gaussian.eval(self.model, self.x)
        self.assertTrue(np.array_equal(y, self.y))