from abc import ABC, abstractmethod
import numpy as np
from collections import namedtuple
from typing import Any, Dict, NamedTuple, List
from fit1d.model import Model
from fit1d.outlier import OutLier


class Fit1D(ABC):
    _model: Model
    _outlier: OutLier
    _exclude_outliers_removal: bool

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> NamedTuple:
        pass

    @abstractmethod
    def eval_with_model(self, x: np.ndarray, model: Model) -> np.ndarray:
        pass

    def eval(self, x: np.ndarray) -> np.ndarray:
        return self.eval_with_model(x, self._model)

    def remove_outlier(self, x: np.ndarray, y: np.ndarray) -> List[int]:
        return self._outlier.remove_outliers(x, y, self._model)

    @staticmethod
    def calc_error(y: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
        return y - y_fit

    @staticmethod
    def calc_rms(residual: np.ndarray) -> float:
        return sum(residual ** 2)

    def get_model(self) -> Model:
        return self._model


class Fit1DMock(Fit1D):
    def __init__(self, model: Model, outlier: OutLier, exclude_outliers_removal: bool):
        self._model = model
        self._outlier = outlier
        self._exclude_outliers_removal = exclude_outliers_removal

    def fit(self, x, y):
        fit_results = namedtuple('fit_results', ['a', 'b', 'c'])
        return fit_results(a=5, b=5.6, c="hello")

    def eval_with_model(self, x, model):
        return np.array([1, 2, 3, 4])
