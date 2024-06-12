from functools import lru_cache
from typing import Optional, Tuple, Union

import numpy as np


@lru_cache(maxsize=512)
def get_feature_index_str(
    feature_names: Optional[Tuple[str]], feature_id: str
):
    # print("Should be here only few times")
    if feature_names is None:
        raise ValueError(
            f"Feature names not provided. "
            f"Impossible to convert {feature_id} to index"
        )

    feature_names = np.array(feature_names)
    index = np.where(feature_names == feature_id)[0]

    if len(index) <= 0:
        raise IndexError(f"{feature_id} is not in {feature_names}")

    return index[0]


def get_feature_index(
    feature_names: Optional[Tuple[str]], feature_id: Union[int, str]
) -> int:
    if isinstance(feature_id, str):
        return get_feature_index_str(feature_names, feature_id)
    else:
        return feature_id
