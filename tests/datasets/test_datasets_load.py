import pytest

from tabularbench.datasets import dataset_factory


@pytest.mark.parametrize(
    "dataset_name",
    [
        "lcld_time",
        "ctu_13_neris",
        "url",
        "wids",
        "malware",
    ],
)
def test_load(dataset_name: str) -> None:
    dataset = dataset_factory.get_dataset(dataset_name)
    x, _ = dataset.get_x_y()
    metadata = dataset.get_metadata(only_x=True)
    assert x.shape[1] == metadata.shape[0]
