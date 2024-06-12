from typing import Callable, Dict, List, Union

from tabularbench.datasets.dataset import Dataset
from tabularbench.datasets.samples.airlines import (
    datasets as airlines_datasets,
)
from tabularbench.datasets.samples.ctu_13_neris import (
    datasets as ctu_13_neris_datasets,
)
from tabularbench.datasets.samples.electricity import (
    datasets as electricity_datasets,
)
from tabularbench.datasets.samples.lcld import datasets as lcld_datasets
from tabularbench.datasets.samples.malware import datasets as malware_datasets
from tabularbench.datasets.samples.url import datasets as url_datasets
from tabularbench.datasets.samples.wids import datasets as wids_datasets

datasets: List[Dict[str, Union[str, Callable[[], Dataset]]]] = (
    lcld_datasets
    + ctu_13_neris_datasets
    + electricity_datasets
    + airlines_datasets
    + url_datasets
    + malware_datasets
    + wids_datasets
)


def load_dataset(dataset_name: str) -> Dataset:
    if dataset_name not in [e["name"] for e in datasets]:
        raise NotImplementedError("Dataset not available.")

    return [e["fun_create"]() for e in datasets if e["name"] == dataset_name][
        0
    ]


def list_datasets() -> List[str]:
    return [e["name"] for e in datasets]
