from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict


class GenericModel(ABC):
    max_iteration : int
    
    @staticmethod
    @abstractmethod
    def fit(x: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def eval(model: Dict[str, Any], x: np.ndarray) -> np.ndarray:
        pass


class GenericModelMock(GenericModel):
    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        return {'a': 5, 'b': "hello"}

    @staticmethod
    def eval(model: Dict[str, Any], x: np.ndarray) -> np.ndarray:
        return np.array([1, 2, 3, 4])
