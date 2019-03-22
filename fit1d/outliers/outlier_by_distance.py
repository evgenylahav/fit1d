import numpy as np
from fit1d.common.outlier import OutLier
import math
from typing import List, Union


class RemoveOutlierByDistance(OutLier):

    def __init__(self, max_error: Union[int, float] = 1, min_points: int = 10):
        self.max_error = max_error
        self.min_points = min_points

    def find_outliers(self, error_vec: np.ndarray) -> List[int]:
        if self.min_points <= len(error_vec) and max(error_vec) >= self.max_error:
            return [np.argmax(error_vec)]
        return []
