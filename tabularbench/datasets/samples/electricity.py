from tabularbench.datasets.dataset import (
    CsvDataSource,
    Dataset,
    DateSorter,
    DefaultSplitter,
    Task,
)


def create_dataset() -> Dataset:
    dataset = Dataset(
        name="electricity",
        data_source=CsvDataSource(
            path="./data/tabularbench/electricity/electricity.csv"
        ),
        metadata_source=CsvDataSource(
            path="./data/tabularbench/electricity/electricity_metadata.csv"
        ),
        tasks=[
            Task(
                name="price_up",
                task_type="classification",
                evaluation_metric="f1_score",
            )
        ],
        sorter=DateSorter(date_col="date_time"),
        splitter=DefaultSplitter(),
        relation_constraints=[],
    )
    return dataset


datasets = [
    {
        "name": "electricity",
        "fun_create": create_dataset,
    },
]
