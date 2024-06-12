from typing import Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from tabularbench.constraints.constraints_backend_executor import (
    ConstraintsExecutor,
)
from tabularbench.constraints.pytorch_backend import PytorchBackend
from tabularbench.constraints.relation_constraint import AndConstraint
from tabularbench.utils.typing import NDInt


def get_x_attack(
    x: pd.DataFrame,
    y: NDInt,
    constraints,
    model,
    filter_correct,
    filter_class,
    subset=0,
) -> Tuple[pd.DataFrame, NDInt]:
    if filter_class is not None:
        filter_l = y == filter_class
        x, y = x[filter_l], y[filter_l]

    if filter_correct:
        y_pred = model.predict(x)
        filter_l = y_pred == y
        x, y = x[filter_l], y[filter_l]

    constraints_executor = ConstraintsExecutor(
        AndConstraint(constraints.relation_constraints),
        PytorchBackend(),
        feature_names=constraints.feature_names,
    )
    constraints_val = constraints_executor.execute(torch.Tensor(x))
    constraints_ok_filter = constraints_val <= 1e-9
    constraints_ok_mean = constraints_ok_filter.float().mean()
    # print(f"Constraints ok: {constraints_ok_mean * 100:.2f}%")

    if constraints_ok_mean < 1:
        filter_l = constraints_ok_filter.nonzero(as_tuple=True)[0].numpy()
        x, y = x.iloc[filter_l], y[filter_l]

    if subset > 0 and subset < len(y):
        _, x, _, y = train_test_split(
            x, y, test_size=subset, stratify=y, random_state=42
        )

    return x, y
