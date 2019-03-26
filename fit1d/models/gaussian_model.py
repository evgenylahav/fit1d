from typing import List
from fit1d.common.model import Model



class GaussianModel(Model):

    def __init__(self, amplitude: float = 1, mu: float = 0, sigma: float = 1, bias: float = 0):
        self._model_parameters = \
            {"amplitude": amplitude, "mu": mu, "sigma": sigma, "bias": bias}

    def create_model_from_list(self, model_list: List[float]):
        self._model_parameters = \
            {"amplitude": model_list[0],
             "mu": model_list[1],
             "sigma": model_list[2],
             "bias": model_list[3]}

    def dump_model_to_list(self) -> object:
        return [self._model_parameters["amplitude"],
                self._model_parameters["mu"],
                self._model_parameters["sigma"],
                self._model_parameters["bias"]]

