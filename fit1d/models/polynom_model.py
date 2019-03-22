from typing import List
from fit1d.common.model import Model


class PolynomModel(Model):
    def __init__(self, model_parameters: List[float]=[]):
        if not model_parameters:
            raise ValueError('model_parameters cannot be empty')
        self._model_parameters["coefficients"] = model_parameters

    def create_model_from_list(self, model_list: List[float]):
        self.__init__(model_list)
        return self

    def dump_model_to_list(self):
        return self._model_parameters["coefficients"]
