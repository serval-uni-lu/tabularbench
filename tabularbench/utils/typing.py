"""This module containts type aliases for this project.
"""

from typing import Union

import numpy as np
import numpy.typing as npt

NDNumber = npt.NDArray[Union[np.float_, np.int_]]
NDFloat = npt.NDArray[np.float_]
NDInt = npt.NDArray[np.int_]
NDBool = npt.NDArray[np.bool_]
