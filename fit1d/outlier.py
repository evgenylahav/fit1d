from abc import ABC, abstractmethod
from fit1d.model import Model
import numpy as np
from typing import Dict, List, Any


class OutLier(ABC):
    _outlier_parameters: Dict[str, Any] = dict()

    @abstractmethod
    def remove_outliers(self, x: np.ndarray, y: np.ndarray, model: Model) -> List[int]:
        pass


class OutLierMock(OutLier):
    def __init__(self, outlier_parameters: Dict[str, Any]):
        self._outlier_parameters = outlier_parameters

    def remove_outliers(self, x: np.ndarray, y: np.ndarray, model: Model) -> List[int]:
        return [1, 2, 3]
