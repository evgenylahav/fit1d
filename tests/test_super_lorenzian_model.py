from unittest import TestCase
from fit1d.models.super_lorenzian_model import SuperLorentzianModel


class TestSuperLorenzianModel(TestCase):

    def test_create_model_from_list(self):
        sl = SuperLorentzianModel([0, 1, 2, 3])
        self.assertEqual(sl._model_parameters, {'c1': 0, 'c2': 1, 'c3': 2, 'c4': 3})

    def test_dump_model_to_list(self):
        sl = SuperLorentzianModel([0, 0, 0, 0])
        sl.create_model_from_list([3, 2, 1, 9])
        self.assertEqual(sl.dump_model_to_list(), [3, 2, 1, 9])
