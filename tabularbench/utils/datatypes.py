import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from tabularbench.utils.typing import NDNumber


def dict2obj(d: Dict[str, Any]) -> Any:
    dumped_data = json.dumps(d)
    result = json.loads(
        dumped_data, object_hook=lambda x: SimpleNamespace(**x)
    )
    return result


def to_torch_number(
    tensor: Union[torch.Tensor, NDNumber, pd.DataFrame]
) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        return tensor

    if isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor).float()

    if isinstance(tensor, pd.DataFrame):
        return torch.from_numpy(tensor.values).float()

    if isinstance(tensor, pd.Series):
        return torch.from_numpy(tensor.values).float()

    raise NotImplementedError(f"Unsupported type: {type(tensor)}")


def to_numpy_number(tensor: Union[torch.Tensor, NDNumber]) -> NDNumber:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()

    elif isinstance(tensor, np.ndarray):
        return tensor

    raise NotImplementedError(f"Unsupported type: {type(tensor)}")


def binary_to_2dim(tensor: torch.Tensor) -> torch.Tensor:
    if (tensor.ndim == 1) or (tensor.shape[1] == 1):
        tensor = torch.column_stack((1 - tensor, tensor))

    return tensor
