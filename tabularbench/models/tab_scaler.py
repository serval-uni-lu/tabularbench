from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder

from tabularbench.utils.datatypes import to_numpy_number, to_torch_number
from tabularbench.utils.typing import NDInt, NDNumber

logger = logging.getLogger(__name__)


def round_func_BPDA(input):
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out


def get_cat_idx(x_type: Union[pd.Series, npt.NDArray[np.str_]]) -> List[int]:
    return np.where(x_type == "cat")[0].tolist()


def softargmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    .. deprecated:: 1.0.0
       Use `mlc.transformer.tab_scaler.soft_where_max`
       and `mlc.transformer.tab_scaler.inverse_transform` instead.


    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dim : int, optional
        Dimension on which the argmax are computed, by default -1

    Returns
    -------
    torch.Tensor
        The tensor of argmax values.
    """
    # crude: assumes max value is unique
    # Can cause rounding errors
    beta = 100.0
    xx = beta * x
    sm = torch.nn.functional.softmax(xx, dim=dim)
    indices = torch.arange(x.shape[dim]).to(x.device)
    y = torch.mul(indices, sm)
    result = torch.sum(y, dim)
    return result


def soft_where_max(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # crude: assumes max value is unique
    # Can cause rounding errors
    beta = 100.0
    xx = beta * x
    out = torch.nn.functional.softmax(xx, dim=dim)
    return out


def differentiable_indexing(
    indexes: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Returns the index of the reference tensor that matches the indexes tensor.

    Parameters
    ----------
    indexes : torch.Tensor
        Indexes tensor.
    reference : torch.Tensor
        Reference tensor.
    Returns
    -------
    torch.Tensor
        The tensor of indexed values.
    """
    mul = torch.mul(reference, indexes)
    out = torch.sum(mul, dim=-1)
    return out


def _handle_zeros_in_scale_torch(
    scale: torch.Tensor,
    copy: bool = True,
    constant_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Set scales of near constant features to 1.
    The goal is to avoid division by very small or zero values.
    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.
    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """
    # if we are fitting on 1D arrays, scale might be a scalar
    if scale.dim() == 0:
        if scale == 0.0:
            scale = torch.tensor(1.0)
        return scale
    else:
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * torch.finfo(scale.dtype).eps

        if copy:
            # New array to avoid side-effects
            scale = scale.clone()
        scale[constant_mask] = 1.0
        return scale


def process_cat_idx_params(
    x: torch.Tensor,
    cat_idx: Optional[List[int]] = None,
    x_type: Optional[Union[pd.Series, npt.NDArray[np.str_]]] = None,
) -> Tuple[List[int], List[int]]:
    if cat_idx is None:
        if x_type is not None:
            cat_idx = get_cat_idx(x_type)

    nb_features = x.shape[1]
    num_idx = [*range(nb_features)]

    if cat_idx is None:
        cat_idx = []
    else:
        num_idx = [e for e in num_idx if e not in cat_idx]

    return num_idx, cat_idx


@dataclass
class ScalerData:
    x_min: Optional[torch.Tensor] = None
    x_max: Optional[torch.Tensor] = None
    x_mean: Optional[torch.Tensor] = None
    x_std: Optional[torch.Tensor] = None
    categories: List[NDInt] = field(default_factory=list)
    cat_idx: List[int] = field(default_factory=list)
    num_idx: List[int] = field(default_factory=list)

    def save(self, path: str) -> None:
        out = {
            "x_min": (
                self.x_min.numpy().tolist() if self.x_min is not None else None
            ),
            "x_max": (
                self.x_max.numpy().tolist() if self.x_max is not None else None
            ),
            "x_mean": (
                self.x_mean.numpy().tolist()
                if self.x_mean is not None
                else None
            ),
            "x_std": (
                self.x_std.numpy().tolist() if self.x_std is not None else None
            ),
            "categories": [e.tolist() for e in self.categories],
            "cat_idx": self.cat_idx,
            "num_idx": self.num_idx,
        }

        json.dump(out, open(path, "w"))

    @staticmethod
    def load(path: str) -> ScalerData:
        data = json.load(open(path, "r"))
        return ScalerData(
            x_min=(
                torch.tensor(data["x_min"])
                if data["x_min"] is not None
                else None
            ),
            x_max=(
                torch.tensor(data["x_max"])
                if data["x_max"] is not None
                else None
            ),
            x_mean=(
                torch.tensor(data["x_mean"])
                if data["x_mean"] is not None
                else None
            ),
            x_std=(
                torch.tensor(data["x_std"])
                if data["x_std"] is not None
                else None
            ),
            categories=[torch.tensor(e) for e in data["categories"]],
            cat_idx=data["cat_idx"],
            num_idx=data["num_idx"],
        )


class TabScaler:
    def __init__(
        self,
        num_scaler: str = "min_max",
        one_hot_encode: bool = True,
        out_min: float = 0.0,
        out_max: float = 1.0,
    ) -> None:
        # Params
        self.num_scaler = num_scaler
        self.one_hot_encode = one_hot_encode
        self.out_min = out_min
        self.out_max = out_max

        # Internal
        self.fitted = False
        self.x_min: Optional[torch.Tensor] = None
        self.x_max: Optional[torch.Tensor] = None
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.categories: List[NDInt] = []
        self.cat_idx: List[int] = []
        self.num_idx: List[int] = []

        # Check params
        if self.out_min >= self.out_max:
            raise ValueError("out_min must be smaller than out_max")

    def do_one_hot_encode(self) -> bool:
        return self.one_hot_encode and len(self.cat_idx) > 0

    def fit(
        self,
        x: Union[torch.Tensor, NDNumber],
        cat_idx: Optional[List[int]] = None,
        x_type: Union[pd.Series, npt.NDArray[np.str_]] = None,
    ) -> TabScaler:
        if isinstance(x, pd.DataFrame):
            return self.fit(to_torch_number(x.values), cat_idx, x_type)

        if isinstance(x, np.ndarray):
            return self.fit(to_torch_number(x), cat_idx, x_type)

        # Process feature types
        self.num_idx, self.cat_idx = process_cat_idx_params(x, cat_idx, x_type)

        # Numerical features
        if self.num_scaler == "min_max":
            self.x_min = torch.min(x[:, self.num_idx], dim=0)[0]
            self.x_max = torch.max(x[:, self.num_idx], dim=0)[0]
            if self.x_min is None or self.x_max is None:
                raise ValueError("No numerical features to scale")
            self.min_max_scale = _handle_zeros_in_scale_torch(
                self.x_max - self.x_min
            )

        elif self.num_scaler == "standard":
            self.mean = torch.mean(x[:, self.num_idx], dim=0)
            self.std = _handle_zeros_in_scale_torch(
                torch.std(x[:, self.num_idx], dim=0)
            )

        elif self.num_scaler == "none":
            pass

        else:
            raise NotImplementedError

        # Categorical features
        if len(self.cat_idx) > 0:
            ohe = OneHotEncoder(sparse=False)
            ohe.fit(x.numpy()[:, self.cat_idx])
            self.categories = ohe.categories_

        self.fitted = True

        return self

    def fit_metadata(self, metadata: pd.DataFrame) -> TabScaler:
        # Process feature types
        self.num_idx, self.cat_idx = process_cat_idx_params(
            torch.from_numpy(
                np.array([metadata["min"].to_numpy().astype(float)])
            ),
            None,
            metadata["type"],
        )

        if self.num_scaler == "min_max":
            self.x_min = torch.tensor(
                metadata["min"].values.astype(np.float_)[self.num_idx],
                dtype=torch.float,
            )
            self.x_max = torch.tensor(
                metadata["max"].values.astype(np.float_)[self.num_idx],
                dtype=torch.float,
            )
            if self.x_min is None or self.x_max is None:
                raise ValueError("No numerical features to scale")
            self.min_max_scale = _handle_zeros_in_scale_torch(
                self.x_max - self.x_min
            )

        elif self.num_scaler == "standard":
            self.mean = torch.tensor(
                metadata["mean"].values.astype(np.float_)[self.num_idx],
                dtype=torch.float,
            )
            self.std = torch.tensor(
                metadata["std"].values.astype(np.float_)[self.num_idx],
                dtype=torch.float,
            )

        elif self.num_scaler == "none":
            pass

        else:
            raise NotImplementedError

        # Categorical features
        if len(self.cat_idx) > 0:
            self.categories = [
                np.arange(
                    int(metadata["min"].values[i]),
                    int(metadata["max"].values[i]) + 1,
                )
                for i in self.cat_idx
            ]

        self.fitted = True

        return self

    def fit_scaler_data(self, scaler_data: ScalerData) -> TabScaler:
        self.num_idx = scaler_data.num_idx
        self.cat_idx = scaler_data.cat_idx
        self.x_min = scaler_data.x_min
        self.x_max = scaler_data.x_max
        self.mean = scaler_data.x_mean
        self.std = scaler_data.x_std
        self.categories = scaler_data.categories
        self.fitted = True

        if self.x_min is not None and self.x_max is not None:
            self.min_max_scale = _handle_zeros_in_scale_torch(
                self.x_max - self.x_min
            )

        return self

    def get_scaler_data(self) -> ScalerData:
        return ScalerData(
            num_idx=self.num_idx,
            cat_idx=self.cat_idx,
            x_min=self.x_min,
            x_max=self.x_max,
            x_mean=self.mean,
            x_std=self.std,
            categories=self.categories,
        )

    def transform(
        self,
        x_in: Union[torch.Tensor, NDNumber],
        cat_encode_method: str = "elu",
    ) -> Union[torch.Tensor, NDNumber]:
        # logger.info(f"{x_in.dtype}")
        if not self.fitted:
            raise ValueError("Must fit scaler before transforming data")

        if isinstance(x_in, pd.DataFrame):
            return to_numpy_number(
                self.transform(to_torch_number(x_in.values))
            )

        if isinstance(x_in, np.ndarray):
            return to_numpy_number(self.transform(to_torch_number(x_in)))
        x = x_in.clone()

        # Numerical features

        if self.num_scaler == "min_max":
            x[:, self.num_idx] = (
                (x[:, self.num_idx].clone() - self.x_min.to(x.device))
                / self.min_max_scale.to(x.device)
            ) * (self.out_max - self.out_min) + self.out_min

        elif self.num_scaler == "standard":
            x[:, self.num_idx] = (
                x[:, self.num_idx] - self.mean.to(x.device)
            ) / self.std.to(x.device)

        elif self.num_scaler == "none":
            pass

        else:
            raise NotImplementedError

        if self.do_one_hot_encode():
            list_encoded = []

            for i, idx in enumerate(self.cat_idx):
                categories = self.categories[i]
                num_categories = len(categories)

                coder = torch.tensor(categories, dtype=torch.float32)
                codes = x[:, idx].clone()

                rows = x.shape[0]

                dummies = torch.broadcast_to(coder, (rows, num_categories))
                coded = codes.repeat_interleave(num_categories).reshape(
                    rows, num_categories
                )

                if cat_encode_method == "elu":
                    encoded = elu_encode(coded, dummies)
                elif cat_encode_method == "power":
                    encoded = power_encode(coded, dummies)
                else:
                    raise NotImplementedError

                list_encoded.append(encoded)

            x = torch.concat([x[:, self.num_idx]] + list_encoded, dim=1)

        return x

    def inverse_transform(
        self, x_in: Union[torch.Tensor, NDNumber], is_cat_binary: bool = False
    ) -> Union[torch.Tensor, NDNumber]:
        if not self.fitted:
            raise ValueError("Must fit scaler before transforming data")

        device: Union[str, torch.device] = "cpu"
        if isinstance(x_in, np.ndarray):
            return to_numpy_number(
                self.inverse_transform(to_torch_number(x_in))
            )
        else:
            device = x_in.device

        x = x_in.clone()

        # Categorical features
        if self.do_one_hot_encode():
            start_idx = len(self.num_idx)

            decoded_list = []
            for i, categories in enumerate(self.categories):
                num_categories = len(categories)
                end_idx = start_idx + num_categories

                decoded = x[:, start_idx:end_idx]
                if not is_cat_binary:
                    decoded = soft_where_max(decoded, dim=1)
                decoded = differentiable_indexing(
                    decoded,
                    torch.from_numpy(categories).type(x.dtype).to(device),
                )
                decoded_list.append(decoded)
                start_idx = end_idx

            out: List[torch.Tensor] = []

            # First numerical features
            out.append(x[:, : self.cat_idx[0]])
            last_num_used = self.cat_idx[0]

            for i, idx in enumerate(self.cat_idx):
                out.append(decoded_list[i].reshape(-1, 1))
                if i < len(self.cat_idx) - 1:
                    num_to_add = self.cat_idx[i + 1] - idx - 1
                else:
                    num_to_add = len(self.num_idx) - last_num_used
                end_idx = last_num_used + num_to_add
                out.append(x[:, last_num_used:end_idx])
                last_num_used = end_idx

            x = torch.concat(out, dim=1)

        # Numerical features
        if self.num_scaler == "min_max":
            x[:, self.num_idx] = (x[:, self.num_idx] - self.out_min) / (
                self.out_max - self.out_min
            )
            x[:, self.num_idx] = x[:, self.num_idx] * self.min_max_scale.to(
                x.device
            ) + self.x_min.to(x.device)

        elif self.num_scaler == "standard":
            x[:, self.num_idx] = x[:, self.num_idx] * self.std + self.mean

        elif self.num_scaler == "none":
            pass

        else:
            raise NotImplementedError

        return x if isinstance(x_in, np.ndarray) else x.to(device)

    def get_transorm_nn(self) -> nn.Module:
        return Transform(self)

    def transform_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask_out = [mask[self.num_idx]]
        for i, e in enumerate(self.cat_idx):
            mask_out.append(mask[e].repeat(len(self.categories[i])))
        return torch.cat(mask_out, dim=0)

    def get_transformed_num_features(
        self,
    ) -> int:
        out = len(self.num_idx)
        if self.do_one_hot_encode():
            out += sum([len(e) for e in self.categories])
        else:
            out += len(self.cat_idx)
        return out

    def save(self, path: str) -> None:
        joblib.dump(self.get_scaler_data(), path)

    def load(self, path: str) -> None:
        self.fit_scaler_data(joblib.load(path))


def elu_encode(coded: torch.Tensor, dummies: torch.Tensor) -> torch.Tensor:
    diff = coded - dummies.to(coded.device)
    diff[diff > 0] = -diff[diff > 0]
    elu = torch.nn.ELU()
    array_bef = elu(diff) + 0.55
    encoded = round_func_BPDA(array_bef)
    return encoded


def power_encode(coded: torch.Tensor, dummies: torch.Tensor) -> torch.Tensor:
    diff_pos = (coded - dummies) ** 2
    diff_neg = 1 - diff_pos
    encoded = torch.maximum(diff_neg, torch.zeros_like(diff_neg))
    return encoded


class Transform(nn.Module):
    def __init__(self, scaler: TabScaler):
        self.scaler = scaler
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return to_torch_number(self.scaler.transform(x))
