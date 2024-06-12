from typing import Dict, List

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from tabularbench.constraints.relation_constraint import (
    BaseRelationConstraint,
    Feature,
)
from tabularbench.datasets.dataset import (
    CsvDataSource,
    Dataset,
    DataSource,
    DefaultIndexSorter,
    DownloadFileDataSource,
    Splitter,
    Task,
)


class WidsSplitter(Splitter):
    def get_splits(self, dataset: Dataset) -> Dict[str, npt.NDArray[np.int_]]:
        _, y = dataset.get_x_y()
        i = np.arange(len(y))

        i_train, i_test = train_test_split(
            i,
            random_state=100,
            shuffle=True,
            stratify=y[i],
            test_size=0.2,
        )
        i_train, i_val = train_test_split(
            i_train,
            random_state=200,
            shuffle=True,
            stratify=y[i_train],
            test_size=0.2,
        )
        return {"train": i_train, "val": i_val, "test": i_test}


def get_relation_constraints(
    metadata: DataSource,
) -> List[BaseRelationConstraint]:

    # features = metadata.load_data()["feature"].to_list()

    # Maybe update the index with feature names in the future
    g_min_max = []
    for i in range(33, 94, 2):
        g_min_max.append(Feature(i + 1) <= Feature(i))
    return g_min_max


def create_dataset() -> Dataset:
    data_source = DownloadFileDataSource(
        url="https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
        "thibault_simonetto_uni_lu/"
        "ESKij8CqW6dAlIe3bW-6rk0BCyDZ7B1gLYI3OTLdlH2wGg?download=1",
        file_data_source=CsvDataSource(path="./data/datasets/wids/wids.csv"),
    )
    metadata_source = DownloadFileDataSource(
        url="https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
        "thibault_simonetto_uni_lu/"
        "EWZgana5BVJNgvCiM1I-Y7MBhXOw7MrxyjGWMg2FRnMCBA?download=1",
        file_data_source=CsvDataSource(
            path="./data/datasets/wids/wids_metadata.csv"
        ),
    )
    tasks = [
        Task(
            name="hospital_death",
            task_type="classification",
            evaluation_metric="mcc",
        )
    ]
    sorter = DefaultIndexSorter()
    splitter = WidsSplitter()
    relation_constraints = get_relation_constraints(metadata_source)

    wids = Dataset(
        name="wids",
        data_source=data_source,
        metadata_source=metadata_source,
        tasks=tasks,
        sorter=sorter,
        splitter=splitter,
        relation_constraints=relation_constraints,
    )
    return wids


datasets = [
    {
        "name": "wids",
        "fun_create": create_dataset,
    }
]
