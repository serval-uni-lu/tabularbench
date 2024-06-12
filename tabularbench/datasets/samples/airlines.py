import pandas as pd

from tabularbench.datasets.dataset import (
    CsvDataSource,
    Dataset,
    DateSorter,
    DefaultSplitter,
    Task,
    TaskProcessor,
)


class RegressionClassificationProcessor(TaskProcessor):
    def transform(self, task_data: pd.Series) -> pd.Series:
        return (task_data > 0).astype(int)


def create_dataset(is_classification: bool) -> Dataset:
    data_source = CsvDataSource(
        path="./data/tabularbench/airlines/flight_delay_ord.csv"
    )
    metadata_source = CsvDataSource(
        path="./data/tabularbench/airlines/flight_delay_ord_metadata.csv"
    )
    sorter = DateSorter(date_col="Date")
    splitter = DefaultSplitter()

    if is_classification:
        name = "flight_delay_ord_classification"
        task = Task(
            name="ArrDelay",
            task_type="classification",
            evaluation_metric="f1_score",
        )
    else:
        name = "flight_delay_ord_regression"
        task = Task(
            name="ArrDelay",
            task_type="regression",
            evaluation_metric="rmse",
            task_processor=RegressionClassificationProcessor(),
        )

    dataset = Dataset(
        name=name,
        data_source=data_source,
        metadata_source=metadata_source,
        tasks=[task],
        sorter=sorter,
        splitter=splitter,
        relation_constraints=[],
    )

    return dataset


datasets = [
    {
        "name": "flight_delay_ord_classification",
        "fun_create": lambda x: create_dataset(is_classification=True),
    },
    {
        "name": "flight_delay_ord_regression",
        "fun_create": lambda x: create_dataset(is_classification=False),
    },
]
