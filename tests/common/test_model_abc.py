import unittest
from fit1d.common.model import Model, ModelMock


class TestModel(unittest.TestCase):
    def setUp(self):
        self.m = ModelMock({"param1": 5.5, "param2": "hello"})

    def test_model_instance(self):
        self.assertTrue(isinstance(self.m, Model))

    def test_equality(self):
        new_m = ModelMock({"param1": 5.5, "param2": "hello"})
        self.assertTrue(self.m == new_m)

    def test_key_inequality(self):
        new_m = ModelMock({"param10": 5.5, "param2": "hello"})
        self.assertFalse(self.m == new_m)

    def test_value_inequality(self):
        new_m = ModelMock({"param1": 20.1, "param2": "hello"})
        self.assertFalse(self.m == new_m)

    def test_from_list(self):
        ref_list = [10, 50.2]
        new_m = ModelMock({"param1": 10, "param2": 50.2})
        self.m.create_model_from_list(ref_list)
        self.assertTrue(new_m == self.m)

    def test_to_list(self):
        ref_list = [5.5, "hello"]
        self.assertListEqual(ref_list, self.m.dump_model_to_list())

    def test_from_dict(self):
        ref_dict = {"param1": "10", "param2": 50.2}
        new_m = ModelMock(ref_dict)
        self.m.create_model_from_dict(ref_dict)
        self.assertTrue(new_m == self.m)

    def test_to_dict(self):
        new_m = ModelMock({"param1": "10", "param2": 50.2})
        ref_dict = {"param1": "10", "param2": 50.2}
        self.assertDictEqual(ref_dict, new_m.dump_model_to_dict())

