"""
This class is an common class that will be used for defining different types of
outliers removal handling.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any


class OutLier(ABC):
    """
    The OutLier class contains the parameters for outlier and an common method
    to be implemented separately for each outliers removal function.
    """
    _outlier_parameters: Dict[str, Any] = dict()

    @abstractmethod
    def find_outliers(self, error: np.ndarray) -> List[int]:
        pass


class OutLierMock(OutLier):
    """ Mock class. Used only for tests. """
    DEFAULT_PARAMS = {'max error': 4, 'min points': 20}

    def __init__(self, outlier_parameters=DEFAULT_PARAMS):
        self.counter = 0
        self._outlier_parameters = outlier_parameters

    def find_outliers(self, error: np.ndarray) -> List[int]:
        if self.counter == 0:
            self.counter += 1
            return [1]
        else:
            return []

