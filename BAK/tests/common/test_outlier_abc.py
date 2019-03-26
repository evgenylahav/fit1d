import unittest
from fit1d.common.outlier import OutLierMock
from fit1d.common.model import ModelMock
import numpy as np


class TestOutliers(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-15.0, 15.0, num=50)
        self.y = self.x
        self.model = ModelMock({"param1": 5.5})

    def test_outliers_mock(self):
        d_ref = [1]
        o = OutLierMock({'param1': 10, 'param2': 5.4, 'param3': 'hello'})
        d_out = o.find_outliers(np.array([9, 0, 3]))
        self.assertListEqual(d_ref, d_out)

    def test_outliers_mock_run_twice(self):
        d_ref = []
        d_out = [6, 8]
        o = OutLierMock({'param1': 10, 'param2': 5.4, 'param3': 'hello'})
        for i in range(2):
            d_out = o.find_outliers(np.array([9, 0, 3]))

        self.assertListEqual(d_ref, d_out)
