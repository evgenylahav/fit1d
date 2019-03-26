from fit1d.common.fit1d import Fit1D
from fit1d.common.fit_data import FitData
from scipy.optimize import leastsq
from fit1d.common.outlier import OutLier, OutLierMock
from fit1d.models.gaussian_model import GaussianModel
import numpy as np


class GaussianFit(Fit1D):

    def __init__(self, outlier: OutLier, use_remove_outliers: bool = False):
        """ init the class"""
        self._use_remove_outliers = use_remove_outliers
        self._outlier = outlier
        self._fit_data = FitData()

    def _calc_fit(self):
        gm = GaussianModel()

        init_params = self.initial_conditions()

        params = leastsq(self.error_func, init_params)
        gm.create_model_from_list(params[0])  # Final findings
        self._fit_data.model = gm

    def _calc_eval(self):
        model = self._fit_data.model.dump_model_to_list()
        self._fit_data.y_fit = model[0] * np.exp(-(self._fit_data.x-model[1])**2/2/model[2]**2) + model[3]

    def error_func(self, params):
        gm = GaussianModel()
        gm.create_model_from_list(params)
        return self._fit_data.y - self.eval(self._fit_data.x, gm)

    def initial_conditions(self):
        x_data = self._fit_data.x
        y_data = self._fit_data.y
        min_y = min(y_data)
        max_y = max(y_data)

        params = np.ndarray((4,))
        params[0] = max_y - min_y

        max_ind = np.where(y_data == max_y)
        if len(max_ind[0]) > 1:
            max_ind = max_ind[0][0]

        params[1] = x_data[max_ind]
        params[3] = min_y
        tmp1 = 1 / np.exp(1) * params[0] + params[3]
        temp_index = [i for i, val in enumerate(y_data) if val >= tmp1]
        params[2] = (x_data[temp_index[-1]] - x_data[temp_index[0]]) / 2

        return params


if __name__ == '__main__':
    gf = GaussianFit(OutLierMock)
    gm = GaussianModel()
    x = np.linspace(-5, 5)
    y = 1*np.exp(-(x-0)**2/2/1**2) + 0
    gm.create_model_from_list([1, 0, 1, 0])
    res = gf.eval(x, gm)

    print('Noach')

