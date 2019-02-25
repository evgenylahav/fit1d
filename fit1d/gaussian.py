import math
from typing import Dict, Any
import numpy as np
from fit1d.generic_model import GenericModel
from scipy.optimize import fmin


class Gaussian(GenericModel):

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray, outlairremoval = False, **kwargs) -> Dict[str, float]:
        # Amplitude * e ^ (-0.5 * ((x - mu) / sigma) ^ 2)

        initial_values = np.random.rand(3, 1)
        output_model = fmin(Gaussian._err_model, initial_values, args=(x, y))
        return {'amplitude': output_model[0], 'mean': output_model[1], 'sigma': output_model[2]}

    @staticmethod
    def eval(model: Dict[str, Any], x: np.ndarray) -> np.ndarray:
        model_vec = [model['amplitude'], model['mean'], model['sigma']]
        return Gaussian._model(model_vec, x)

    @staticmethod
    def _model(model_vec, x_vec):
        return model_vec[0] * math.e ** (-0.5 * ((x_vec - model_vec[1]) / model_vec[2]) ** 2)

    @staticmethod
    def _err_model(model_vec, x_vec, y_vec):
        return (abs((Gaussian._model(model_vec, x_vec) - y_vec))).sum()

    def _outliers_removal(self) -> Dict[str, float]:
        iterator = 1
        while (iterator<self.max_iteration and len(self.x) > self.min_data_length):
            model = self.fit()
            self.remove_max_error(model)
            iterator += 1
        return model