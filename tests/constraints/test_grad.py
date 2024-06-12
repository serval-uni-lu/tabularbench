import pytest
import torch

from tabularbench.constraints.constraints_backend_executor import (
    ConstraintsExecutor,
)
from tabularbench.constraints.pytorch_backend import PytorchBackend
from tabularbench.datasets.dataset_factory import get_dataset


# Parametrize with the dataset
@pytest.mark.parametrize(
    "dataset_name",
    [
        ("lcld_time"),
        ("ctu_13_neris"),
        ("url"),
        ("wids"),
        ("malware"),
    ],
)
def test_constraints_grad(dataset_name: str) -> None:
    ds = get_dataset(dataset_name)
    constraints = ds.get_constraints()

    x, y = ds.get_x_y()
    x_metadata = ds.get_metadata(only_x=True)
    x = torch.tensor(x.values, dtype=torch.float32)[:1000]

    for c in constraints.relation_constraints:
        x_l = x.clone()
        constraints_executor = ConstraintsExecutor(
            c,
            PytorchBackend(),
            feature_names=x_metadata["feature"].to_list(),
        )

        x_l.requires_grad = True
        cost = constraints_executor.execute(x_l)
        grad = torch.autograd.grad(
            cost.sum(),
            x_l,
        )[0]
        assert not torch.isnan(grad).any()
        assert not torch.isinf(grad).any()
