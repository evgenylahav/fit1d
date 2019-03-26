import unittest
from fit1d.models.gaussian_model import GaussianModel


class TestGaussianModel(unittest.TestCase):
    def test_default_constructor(self):
        g_model = GaussianModel()
        self.assertEqual(
            g_model._model_parameters,
            {'amplitude': 1, 'mu': 0, 'sigma': 1, 'bias': 0})

    def test_inputs_constructor(self):
        g_model = GaussianModel(3, 3, 3, 3)
        self.assertEqual(
            g_model._model_parameters,
            {'amplitude': 3, 'mu': 3, 'sigma': 3, 'bias': 3})

    def test_to_from_list(self):
        g_model = GaussianModel()
        g_model.create_model_from_list([4, 4, 4, 4])
        self.assertEqual(
            g_model.dump_model_to_list(),
            [4, 4, 4, 4])
