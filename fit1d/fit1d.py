"""
fit1d package is designed to provide an organized toolbox for different types of
1D fits that can be performed.
It is easy to add new fits and other functionalities
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List
from fit1d.model import Model
from fit1d.outlier import OutLier


class FitResults:
    """
    FitResults is a data storage class, that is used to get the output of the
    fit method in the Fit1D class
    """
    _model: Model
    _outlier: OutLier
    _error_vector: np.ndarray
    _rms: float

    def __init__(self, model: Model, error_vector: np.ndarray, rms: float, outlier: OutLier):
        """
        The data class storage contains model object, error vector, rms and outlier object
        """
        self._model = model
        self._error_vector = error_vector
        self._rms = rms
        self._outlier = outlier

    def __eq__(self, other):
        return self._model == other._model and\
               np.array_equal(self._error_vector, other._error_vector) and\
               self._rms == other._rms and \
               self._outlier == other._outlier


class Fit1D(ABC):
    """
    This is the main class of the fit1d package. It is used to allow the user to execute
    fit and eval methods, in addition to calc_RMS and calc_error static services.
    The properties of this class are the _model and _outlier objects and a _remove_outliers
    boolean
    """
    _model: Model
    _outlier: OutLier
    _remove_outliers: bool

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> FitResults:
        pass

    @abstractmethod
    def eval(self, x: np.ndarray, model: Model = None) -> np.ndarray:
        pass

    def _remove_outlier(self, x: np.ndarray, y: np.ndarray) -> List[int]:
        return self._outlier.remove_outliers(x, y, self._model)

    @staticmethod
    def calc_error(y: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
        return y - y_fit

    @staticmethod
    def calc_rms(residual: np.ndarray) -> float:
        return (sum(residual ** 2) / len(residual)) ** 0.5

    def get_model(self) -> Model:
        return self._model


class Fit1DMock(Fit1D):
    """ Mock class. Used only for tests """
    def __init__(self, model: Model, outlier: OutLier, remove_outliers: bool):
        self._model = model
        self._outlier = outlier
        self._remove_outliers = remove_outliers

    def fit(self, x, y):
        return FitResults(model=self._model,
                          error_vector=np.array([1, 10, 100, 1000]),
                          rms=5.4,
                          outlier=self._outlier)

    def eval(self, x: np.ndarray, model: Model = None) -> np.ndarray:
        if model is None:
            model = self._model

        return np.array([1, 2, 3, 4])
