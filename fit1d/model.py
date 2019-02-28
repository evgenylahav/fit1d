from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Model(ABC):
    _model_parameters: Dict[str, float] = dict()

    def __eq__(self, other):
        return self._model_parameters == other._model_parameters

    def from_list(self) -> 'Model':
        pass

    def to_list(self) -> List[Any]:
        pass

    def from_dict(self) -> 'Model':
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass


class ModelMock(Model):
    def __init__(self, model_parameters):
        self._model_parameters = model_parameters
