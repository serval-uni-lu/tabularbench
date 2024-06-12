from dataclasses import dataclass
from typing import Union

import torch

from tabularbench.benchmark.model_utils import load_model_and_weights
from tabularbench.datasets.dataset_factory import get_dataset
from tabularbench.metrics.compute import compute_metric
from tabularbench.metrics.metric_factory import create_metric
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.utils.datatypes import to_torch_number


@dataclass
class BenchmarkSettings:
    n_input: int = 10
    device: str = "cpu"


def benchmark(
    dataset: str,
    model: str,
    distance: str,
    constraints: bool,
    settings: BenchmarkSettings = None,
) -> Union[float, float, float]:

    # Alliases
    bms = settings

    if bms is None:
        bms = BenchmarkSettings()

    # load everything that needs to be loaded
    ds = get_dataset(dataset)
    x, y = ds.get_x_y()
    metadata = ds.get_metadata(only_x=True)

    scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
    scaler.fit(to_torch_number(x), x_type=metadata["type"])

    model_arch = model.split("_")[0]
    model_training = "_".join(model.split("_")[1:])

    model_o = load_model_and_weights(
        ds.name, model_arch, model_training, metadata, scaler, bms.device
    )

    metric = create_metric("accuracy")

    clean_acc = compute_metric(
        model_o,
        metric,
        x,
        y,
    )

    # attack

    # evaluate

    # return

    return clean_acc, 5
