import numpy as np
from fit1d.common.outlier import OutLier
from fit1d.common.fit_data import FitData
from fit1d.common.fit1d import Fit1D
from fit1d.outliers.outlier_by_distance import RemoveOutlierByDistance


class PolynomFit(Fit1D):
        def __init__(self, outlier: OutLier=RemoveOutlierByDistance(), remove_outliers: bool =False, degree: int =2):
            self._fit_data = FitData()
            self._outlier = outlier
            self._use_remove_outliers = remove_outliers
            self.degree = degree

        def _calc_fit(self):
            polynom_coefficients = np.polynomial.polynomial.polyfit(self._fit_data.x, self._fit_data.y, self.degree)
            polynom_coefficients = [round(coef, 5) for coef in polynom_coefficients] #round each coef to 5 digits after .
            polynom_coefficients.reverse()
            self._fit_data.model = {"coefficients": polynom_coefficients}

        def _calc_eval(self):
            self._fit_data.y_fit = np.polyval(self._fit_data.model.get("coefficients"), self._fit_data.x)

