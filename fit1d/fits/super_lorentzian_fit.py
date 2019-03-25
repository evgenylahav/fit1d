"""Super Lorentzian Fit Module"""
from typing import List
from typing import Type
import numpy as np
from scipy.optimize import fmin
from fit1d.common.fit1d import Fit1D, OutLier, FitData
from fit1d.models.super_lorenzian_model import SuperLorentzianModel
from fit1d.outliers.outlier_by_distance import RemoveOutlierByDistance


class SuperLorentzianFit(Fit1D):
    """Implement the Super Lorentzian Fit class"""

    def __init__(self, outlier: Type[OutLier] = RemoveOutlierByDistance(),
                 use_remove_outliers: bool = False):
        """ init the class"""
        self._use_remove_outliers = use_remove_outliers
        self._outlier = outlier
        self._fit_data = FitData()

    def _calc_fit(self) -> None:
        """
        calc the fit parameters
        :return: Void
        """
        # Initial guess:
        p_0 = np.ndarray((4,))
        p_0[1] = np.mean(self._fit_data.y)
        p_0[0] = max(self._fit_data.y) - p_0[1]
        p_0[2] = -self._fit_data.x[np.argmax(self._fit_data.y)]
        p_0[3] = abs(self._fit_data.x[np.argmax(np.diff(self._fit_data.y))]
                     - self._fit_data.x[np.argmin(np.diff(self._fit_data.y))])/2
        params = fmin(lambda p: np.sum((self._fit_data.y
                                        - self._superlorentzian(self._fit_data.x, p))**2), p_0)
        self._fit_data.model = SuperLorentzianModel(list(params))

    def _calc_eval(self):
        """
        Eval the model
        :return: np.ndarray
        """
        params = self._fit_data.model.dump_model_to_list()
        eval_result = self._superlorentzian(self._fit_data.x, params)
        self._fit_data.y_fit = eval_result

    @staticmethod
    def _superlorentzian(x_values: np.ndarray, param: List[float]) -> np.ndarray:
        """the super lorentzian formula"""
        return param[1] + param[0]/(1 + ((x_values - param[2])/param[3])**40)
