import warnings
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from tabularbench.constraints.constraints import Constraints
from tabularbench.constraints.constraints_fixer import ConstraintsFixer
from tabularbench.constraints.relation_constraint import (
    EqualConstraint,
    Feature,
)
from tabularbench.utils.typing import NDNumber


def mutate(x_original: NDNumber, x_mutation: NDNumber) -> None:
    if x_original.shape[:-1] != x_mutation.shape[:-1]:
        raise ValueError(
            f"X_original has shape: {x_original.shape}, "
            f"X_mutation has shape {x_mutation.shape}. "
            f"Shapes must be equal until index -1."
        )


def compute_distance(x_1: NDNumber, x_2: NDNumber, norm: Any) -> NDNumber:
    if norm in ["inf", np.inf, "Linf", "linf"]:
        distance = np.linalg.norm(x_1 - x_2, ord=np.inf, axis=-1)
    elif norm in ["2", 2, "L2", "l2"]:
        distance = np.linalg.norm(x_1 - x_2, ord=2, axis=-1)
    else:
        raise NotImplementedError

    return distance


def cut_in_batch(
    arr: NDArray[Any],
    n_desired_batch: int = 1,
    batch_size: Optional[int] = None,
) -> List[NDArray[Any]]:
    if batch_size is None:
        n_batch = min(n_desired_batch, len(arr))
    else:
        n_batch = np.ceil(len(arr) / batch_size)
    batches_i = np.array_split(np.arange(arr.shape[0]), n_batch)

    return [arr[batch_i] for batch_i in batches_i]


def fix_types(
    x_clean: torch.Tensor, x_adv: torch.Tensor, types: pd.Series
) -> torch.Tensor:
    x_adv = x_adv.clone()
    int_indices = np.where(types == "int")[0]

    if len(int_indices) == 0:
        return x_adv

    x_adv_ndim = x_adv.ndim

    if x_clean.ndim == 2:
        x_clean = x_clean.unsqueeze(1)
    if x_adv_ndim == 2:
        x_adv = x_adv.unsqueeze(1)

    int_perturbation = x_adv[:, :, int_indices] - x_clean[:, :, int_indices]

    int_perturbation_fixed = torch.fix(int_perturbation)

    x_adv[:, :, int_indices] = (
        x_clean[:, :, int_indices] + int_perturbation_fixed
    )

    cat_indices = np.where(types == "cat")[0]
    x_adv[:, :, cat_indices] = torch.round(x_adv[:, :, cat_indices])

    if x_adv_ndim == 2:
        x_adv = x_adv[:, 0, :]

    return x_adv


def fix_immutable(
    x_clean: torch.Tensor, x_adv: torch.Tensor, mutable: pd.Series
) -> torch.Tensor:
    x_adv = x_adv.clone()
    immutable_indices = np.where(~mutable)[0]

    if len(immutable_indices) == 0:
        return x_adv

    x_adv_ndim = x_adv.ndim

    if x_clean.ndim == 2:
        x_clean = x_clean.unsqueeze(1)
    if x_adv_ndim == 2:
        x_adv = x_adv.unsqueeze(1)

    if not torch.isclose(
        x_clean[:, :, immutable_indices], x_adv[:, :, immutable_indices]
    ).all():
        warnings.warn("Mutable indices are not equal nor close.")
        print("Mutable indices are not equal")
        print("Fixing mutable indices")

    x_adv[:, :, immutable_indices] = x_clean[:, :, immutable_indices]

    if x_adv_ndim == 2:
        x_adv = x_adv[:, 0, :]

    return x_adv


def fix_equality_constraints(
    constraints: Constraints, x_adv: torch.Tensor, fix_constraints_ijcai=False
) -> torch.Tensor:
    if constraints.relation_constraints is None:
        return x_adv
    constraints_to_fix = [
        c
        for c in constraints.relation_constraints
        if (
            isinstance(c, EqualConstraint)
            and isinstance(c.left_operand, Feature)
        )
    ]

    if fix_constraints_ijcai:
        constraints_to_fix = [
            c
            for c in constraints.relation_constraints
            if (
                isinstance(c, EqualConstraint)
                and isinstance(c.left_operand, Feature)
                and (c.left_operand.feature_id == "installment")
            )
        ]

    constraints_fixer = ConstraintsFixer(
        guard_constraints=constraints_to_fix,
        fix_constraints=constraints_to_fix,
        feature_names=constraints.feature_names,
    )

    x_adv = constraints_fixer.fix(x_adv)

    return x_adv
