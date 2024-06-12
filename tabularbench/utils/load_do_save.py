import json
import logging
import os.path
from pathlib import Path
from types import FunctionType
from typing import Any, Dict, Optional

import h5py
import joblib
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

cache = []
hdf5_dataset_key = "default"
logger = logging.getLogger(__name__)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        obj = json.load(f)
    return obj


def save_json(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_hdf5(path: str) -> npt.NDArray[Any]:

    try:
        with h5py.File(path, "r") as f:
            return f[hdf5_dataset_key][()]
    except:
        logger.error(f"Error loading {path}")


def save_hdf5(obj: Any, path: str) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset(hdf5_dataset_key, data=obj, compression="gzip")


type_to_fun_mapper = [
    {
        "type": np.ndarray,
        "load": load_hdf5,
        "save": save_hdf5,
    },
    {"type": BaseEstimator, "load": joblib.load, "save": joblib.dump},
    {"type": dict, "load": load_json, "save": save_json},
]


def find_mapper(out_type: Any) -> Dict[str, Any]:
    for mapper in type_to_fun_mapper:
        mapper_type = mapper.get("type")
        if issubclass(out_type, mapper_type):
            return mapper
    raise NotImplementedError


def load_do_save(
    path: str,
    executable: FunctionType,
    return_type: Optional[str] = None,
    verbose: bool = False,
    **kwargs: Dict[str, Any],
) -> Any:
    if return_type is None:
        return_type = executable.__annotations__["return"]

    mapper = find_mapper(return_type)
    if os.path.exists(path):
        load = mapper["load"]
        out = load(path)
        if verbose:
            print(f"{path} loaded.")
        return out
    else:
        save = mapper["save"]
        obj = executable(**kwargs)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save(obj, path)
        if verbose:
            print(f"{path} saved.")
        return obj
