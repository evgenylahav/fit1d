"""
fit1d package is designed to provide an organized toolbox for different types of
1D fits that can be performed.
It is easy to add new fits and other functionalities
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List,Tuple
from fit1d.common.model import Model
from fit1d.common.outlier import OutLier


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
    The properties of this class are the _model and _outlier objects and a _use_remove_outliers
    boolean
    """
    _model: Model
    _outlier: OutLier
    _use_remove_outliers: bool
    _x: np.ndarray
    _y: np.ndarray
    _outlier_index: np.ndarray

    def fit(self, x: np.ndarray, y: np.ndarray) -> FitResults:
        if self._use_remove_outliers:
            outliers = list()
            self._model = self._fit_model(x, y)
        else:
            self._model, outliers = self._remove_outlier(x, y)

        return self._calc_fit_results(x, y, outliers)

    @abstractmethod
    def _fit_model(self, x: np.ndarray, y: np.ndarray) -> Model:
        pass

    @abstractmethod
    def eval(self, x: np.ndarray, model: Model = None) -> np.ndarray:
        pass

    def _calc_fit_results(self, x, y, outliers):
        y_fit = self.eval(x)
        errors = self.calc_error(y, y_fit)
        rms = self.calc_rms(errors)
        return FitResults(model=self._model,
                          error_vector=errors,
                          rms=rms,
                          outlier=outliers)

    def _remove_outlier(self, x: np.ndarray, y: np.ndarray) -> Tuple[Model, List[int]]:
        outliers = list()

        index = [1]
        while len(index) > 1:
            self._model = self._fit_model(x, y)
            y_fit = self.eval(x)
            error = self.calc_error(y, y_fit)
            index = self._outlier.find_outliers(error)
            outliers.append(index)
            np.delete(x, index)
            np.delete(y, index)
        return self._model, outliers

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
        self._use_remove_outliers = remove_outliers

    def _fit_model(self, x: np.ndarray, y: np.ndarray) -> Model:
        return self._model

    def eval(self, x: np.ndarray, model: Model = None) -> np.ndarray:
        if model is None:
            model = self._model

        return np.array([11, 22, 33, 44])
