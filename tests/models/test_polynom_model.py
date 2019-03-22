import unittest
from fit1d.models.polynom_model import PolynomModel


class TestPolyModel(unittest.TestCase):

    def test_empty(self):
        with self.assertRaises(ValueError):
            PolynomModel()

    def test_create_model_from_list(self):
        obj = PolynomModel([2, 5, 7])
        self.assertEqual(obj.create_model_from_list([2, 5, 7]), obj)

    def test_dump_model_to_list(self):
        obj = PolynomModel([2, 5, 7])
        self.assertEqual(obj.dump_model_to_list(), [2, 5, 7])
