"""
The model class is an common class used to define models for different types of fits.
Each fit will need to implement its own model class
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Union


class Model(ABC):
    """
    The Model class is used to contain a dictionary of model parameters and provides
    a few services to handle model objects easily.
    """
    _model_parameters: Dict[str, Union[List[float], float]] = dict()

    def __eq__(self, other):
        return self._model_parameters == other._model_parameters

    @abstractmethod
    def create_model_from_list(self, model_list: List[float]):
        pass

    @abstractmethod
    def dump_model_to_list(self) -> List[float]:
        pass

    def create_model_from_dict(self, model_dict: Dict[str, Union[List[float], float]]):
        self._model_parameters = model_dict

    def dump_model_to_dict(self) -> Dict[str, Union[List[float], float]]:
        return self._model_parameters


class ModelMock(Model):
    """ Mock class. Used only for tests """
    def __init__(self, model_parameters):
        self._model_parameters = model_parameters

    def create_model_from_list(self, model_list: List[float]):
        self._model_parameters["param1"] = model_list[0]
        self._model_parameters["param2"] = model_list[1]

    def dump_model_to_list(self):
        out_list = list()
        out_list.append(self._model_parameters["param1"])
        out_list.append(self._model_parameters["param2"])
        return out_list

