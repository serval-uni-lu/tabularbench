from abc import abstractmethod

import numpy as np
import numpy.typing as npt


class Metric:
    @abstractmethod
    def compute(
        self, y_true: npt.NDArray[np.generic], y_score: npt.NDArray[np.generic]
    ) -> npt.NDArray[np.generic]:
        pass
