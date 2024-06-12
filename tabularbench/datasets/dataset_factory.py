import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tabularbench.constraints.relation_constraint import BaseRelationConstraint
from tabularbench.datasets.dataset import (
    CsvDataSource,
    Dataset,
    DataSource,
    DateSorter,
    DownloadFileDataSource,
    FileDataSource,
    Hdf5DataSource,
    SQLDataSource,
    Task,
)
from tabularbench.datasets.samples import load_dataset


def get_tasks(config: List[Dict[str, Any]]) -> List[Task]:
    return [Task(**e) for e in config]


def get_sorter(config: Dict[str, Any]) -> Optional[DateSorter]:
    if "date" in config.keys():
        return DateSorter(config.get("date"))
    else:
        return None


extension_to_file_loader = {".csv": CsvDataSource, ".hdf5": Hdf5DataSource}


def get_filedatasource(path: str) -> FileDataSource:
    ext = Path(path).suffix
    if ext not in extension_to_file_loader:
        raise NotImplementedError
    return extension_to_file_loader[ext](path)


def get_fun_preprocess(module_name: str) -> Optional[Any]:
    return importlib.import_module(module_name).preprocess


def get_sql_datasource(config: Dict[str, Any]) -> DataSource:

    fun_preprocess = None
    if config.get("preprocess_module") is not None:
        fun_preprocess = get_fun_preprocess(config["preprocess_module"])

    return SQLDataSource(
        config["connection_string"],
        config["query"],
        fun_preprocess,
    )


def get_datasource(config: Dict[str, Any]) -> DataSource:
    if "download" in config.keys():
        download = config["download"]

        return DownloadFileDataSource(
            url=download.get("url"),
            file_data_source=get_filedatasource(download.get("cache")),
        )
    if "file" in config.keys():
        file = config["file"]
        return get_filedatasource(file.get("path"))

    if "sql" in config.keys():
        return get_sql_datasource(config["sql"])


def get_dataset(config: Union[Dict[str, Any], str]) -> Dataset:
    if isinstance(config, str):
        config = {"name": config}
    return get_dataset_from_config(config)


def get_dataset_from_config(config: Dict[str, Any]) -> Dataset:

    name = config.get("name")

    # If it is a known dataset
    if len(config.keys()) == 1:
        return load_dataset(name)

    data_source = get_datasource(config["data"].get("source"))
    metadata_source = get_datasource(config["metadata"].get("source"))
    tasks = get_tasks(config["tasks"])
    sorter = get_sorter(config["sorter"])
    splitter = None
    relation_constraints: List[BaseRelationConstraint] = []
    return Dataset(
        name=name,
        data_source=data_source,
        metadata_source=metadata_source,
        tasks=tasks,
        sorter=sorter,
        splitter=splitter,
        relation_constraints=relation_constraints,
    )
