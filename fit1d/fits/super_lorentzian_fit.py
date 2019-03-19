"""Super Lorentzian Fit Module"""
import numpy as np
from scipy.optimize import fmin
from fit1d.common.fit1d import Fit1D, FitData
from fit1d.common.fit1d import OutLier


class SuperLorentzianFit(Fit1D):
    """Implement the Super Lorentzian Fit class"""
    def __init__(self, outlier: OutLier, use_remove_outliers: bool = False):
        self._use_remove_outliers = use_remove_outliers
        self._outlier = outlier
        self._model = SuperLorentzianModel()

    def _calc_fit(self):
        """
        calc the fit parameters
        :return: Void
        """
        # Initial guess:
        c = [0.0] * 4
        c[0] = max(self._fit_data.y)
        c[1] = float(np.mean(self._fit_data.y))
        c[2] = float(np.mean(self._fit_data.x))
        c[3] = float(np.argmax())
        pass

    def _calc_eval(self) -> np.ndarray:
        """
        Eval the model
        :return: np.ndarray
        """
        return self._superlorentzian(self._fit_data.x, self._model.dump_model_to_list())



    @staticmethod
    def _superlorentzian(x: np.ndarray, c: list) -> np.ndarray:
        return c[1] + c[0]/(1 + (x-c[2])/c[3])**40


class SuperLorentzianModel():
    pass