"""
fit1d package is designed to provide an organized toolbox for different types of
1D fits that can be performed.
It is easy to add new fits and other functionalities
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List,Tuple
from fit1d.common.model import Model, ModelMock
from fit1d.common.outlier import OutLier
from fit1d.common.fit_data import FitData


class Fit1D(ABC):
    """
    This is the main class of the fit1d package. It is used to allow the user to execute
    fit and eval methods, in addition to calc_RMS and calc_error static services.
    The properties of this class are the _model and _outlier objects and a _use_remove_outliers
    boolean
    """
    _outlier: OutLier
    _use_remove_outliers: bool
    _fit_data: FitData

    # interface methods
    def fit(self, x: np.ndarray, y: np.ndarray) -> FitData:
        self._fit_data.x = x
        self._fit_data.y = y
        if self._use_remove_outliers:
            self._remove_outlier()
        else:
            self._calc_fit_and_update_fit_data()

        return self._fit_data

    def eval(self, x: np.ndarray = None, model: Model = None) -> np.ndarray:
        if x is not None:
            self._fit_data.x = x
        if model is not None:
            self._fit_data.model = model
        self._calc_eval()
        return self._fit_data.y_fit

    def calc_error(self):
        """
        calc error vector , update _fit_data
        :return:
        """
        if self._fit_data.y is not None and self._fit_data.y_fit is not None:
            self._fit_data.error_vector = self._fit_data.y - self._fit_data.y_fit

    def calc_rms(self):
        if self._fit_data.error_vector is not None:
            self._fit_data.rms = (sum(self._fit_data.error_vector ** 2) / len(self._fit_data.error_vector)) ** 0.5

    def get_fit_data(self) -> FitData:
        return self._fit_data

    # abstract methods
    @abstractmethod
    def _calc_fit(self):
        """
        abstractmethod:
        run fit calculation of the data update model in _fit_data.model
        :return: Null
        """
        pass

    @abstractmethod
    def _calc_eval(self):
        """
        abstractmethod:
        subclass calculate model eval for inner x and model
        update _fit_data.y_fit
        :return: Void
        """
        pass

    # internal methods
    def _update_fit_data(self):
        self._calc_eval()
        self.calc_error()
        self.calc_rms()

    def _remove_outlier(self):
        while True:
            self._calc_fit_and_update_fit_data()
            indexes_to_remove = self._outlier.find_outliers(self._fit_data.error_vector)
            if len(indexes_to_remove) == 0:
                break
            else:
                self._remove_indexes(indexes_to_remove)

    def _remove_indexes(self, ind):
        self._fit_data.x = np.delete(self._fit_data.x, ind)
        self._fit_data.y = np.delete(self._fit_data.y, ind)

    def _calc_fit_and_update_fit_data(self):
        self._calc_fit()
        self._update_fit_data()


class Fit1DMock(Fit1D):
    """ Mock class. Used only for tests """
    def __init__(self, outlier: OutLier, remove_outliers: bool):
        self._fit_data = FitData()
        self._outlier = outlier
        self._use_remove_outliers = remove_outliers

    def _calc_fit(self):
        self._fit_data.model = ModelMock({"param1": 5.5})

    def _calc_eval(self) -> np.ndarray:
        if self._fit_data.y is None or len(self._fit_data.y) == 4:
            self._fit_data.y_fit = np.array([11, 22, 33, 44])
        else:
            self._fit_data.y_fit = np.array([11, 33, 44])

