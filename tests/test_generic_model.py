import unittest
from fit1d.generic_model import GenericModelMock
import numpy as np


class TestGenericModel(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-15.0, 15.0, num=50)
        self.y = self.x

    def test_fit(self):
        d_ref = {'a': 5, 'b': "hello"}
        d_out = GenericModelMock.fit(self.x, self.y)
        self.assertDictEqual(d_ref, d_out)

    def test_eval(self):
        d = {'a': 5, 'b': "hello"}
        nd_out = GenericModelMock.eval(d, self.x)
        nd_ref = np.array([1, 2, 3, 4])
        self.assertTrue(np.array_equal(nd_out, nd_ref))
