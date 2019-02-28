import unittest
from fit1d.model import Model, ModelMock


class TestModel(unittest.TestCase):
    def test_model_instance(self):
        m = ModelMock({"param1": 5.5})
        self.assertTrue(isinstance(m, Model))
