import numpy as np
from fit1d.common.model import Model


class FitData:
    """
    FitResults is a data storage class, that is used to get the output of the
    fit method in the Fit1D class
    """
    model: Model
    x: np.ndarray
    y: np.ndarray
    y_fit: np.ndarray
    error_vector: np.ndarray
    rms: float

    def __init__(self,
                 model: Model = None,
                 x: np.ndarray = None,
                 y: np.ndarray = None,
                 y_fit: np.ndarray = None,
                 error_vector: np.ndarray = None,
                 rms: float = None):
        """
        The data class storage contains model object, error vector, rms and outlier object
        """
        self.model = model
        self.x = x
        self.y = y
        self.y_fit = y_fit
        self.error_vector = error_vector
        self.rms = rms

    def __eq__(self, other):
        return self.model == other.model and\
               np.array_equal(self.error_vector, other.error_vector) and\
               np.array_equal(self.x, other.x) and\
               np.array_equal(self.y, other.y) and\
               np.array_equal(self.y_fit, other.y_fit) and\
               self.rms == other.rms
