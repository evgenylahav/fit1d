import unittest
from fit1d.outlier import OutLierMock
from fit1d.model import ModelMock
import numpy as np


def fit():
    return 'abc'


def evaluate():
    return 123


class TestOutliers(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-15.0, 15.0, num=50)
        self.y = self.x
        self.model = ModelMock({"param1": 5.5})

    def test_outliers_mock(self):
        d_ref = [1, 2, 3]
        o = OutLierMock({'param1': 10, 'param2': 5.4, 'param3': 'hello'})
        d_out = o.find_outliers(self.x, self.y, self.model)
        self.assertListEqual(d_ref, d_out)

    def test_outliers_mock_with_functions(self):
        d_ref = [1, 2, 3]
        o = OutLierMock({'param1': 10, 'param2': 5.4, 'param3': 'hello'}, fit, evaluate)
        d_out = o.find_outliers(self.x, self.y, self.model)
        self.assertListEqual(d_ref, d_out)

    def test_outliers_mock_fit_output(self):
        o = OutLierMock({'param1': 10, 'param2': 5.4, 'param3': 'hello'}, fit, evaluate)
        self.assertEqual(o._fit(), 'abc')

    def test_outliers_mock_eval_output(self):
        o = OutLierMock({'param1': 10, 'param2': 5.4, 'param3': 'hello'}, fit, evaluate)
        self.assertEqual(o._eval(), 123)
