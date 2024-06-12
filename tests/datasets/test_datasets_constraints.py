import numpy as np
import pytest

from tabularbench.constraints.constraints_checker import ConstraintChecker
from tabularbench.datasets import dataset_factory


@pytest.mark.parametrize(
    "dataset_name, tolerance, input_proportion",
    [
        ("lcld_time", 0.01, 0.001),
        ("ctu_13_neris", 0.0, 0.001),
        ("url", 0.0, 0.0),
        ("wids", 0.0, 0.015),
        ("malware", 1e-10, 0.0),
    ],
)
def test_constraints(dataset_name, tolerance, input_proportion):
    dataset = dataset_factory.get_dataset(dataset_name)
    x, _ = dataset.get_x_y()
    constraints_checker = ConstraintChecker(
        dataset.get_constraints(), tolerance
    )
    out = constraints_checker.check_constraints(x.to_numpy(), x.to_numpy())
    assert (1 - np.mean(out)) <= input_proportion
