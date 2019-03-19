from fit1d.common.model import Model
from typing import List


class SuperLorenzianModel(Model):

    def __init__(self, model_list: List[float]):
        self.create_model_from_list(model_list)

    def create_model_from_list(self, model_list: List[float]):
        self._model_parameters['c1'] = model_list[0]
        self._model_parameters['c2'] = model_list[1]
        self._model_parameters['c3'] = model_list[2]
        self._model_parameters['c4'] = model_list[3]

    def dump_model_to_list(self) -> List[float]:
        out_list = list()
        out_list.append(self._model_parameters['c1'])
        out_list.append(self._model_parameters['c2'])
        out_list.append(self._model_parameters['c3'])
        out_list.append(self._model_parameters['c4'])
        return out_list

