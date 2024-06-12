from __future__ import annotations

import abc
import os
from abc import ABC
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from tabularbench.constraints.constraints import (
    Constraints,
    get_constraints_from_metadata,
)
from tabularbench.constraints.constraints_checker import ConstraintChecker
from tabularbench.constraints.relation_constraint import BaseRelationConstraint


class SplitterMissingError(Exception):
    pass


class DataSource(ABC):
    @abc.abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass


class FileDataSource(DataSource, ABC):
    def __init__(self, path: str):
        self.path = path

    def get_path(self) -> str:
        return self.path


class CsvDataSource(FileDataSource):
    def __init__(self, path: str, **kwargs: Dict[str, Any]) -> None:
        self.pandas_params = kwargs
        super(CsvDataSource, self).__init__(path)

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path, **self.pandas_params, low_memory=False)


class Hdf5DataSource(FileDataSource):
    def __init__(self, path: str, **kwargs: Dict[str, Any]) -> None:
        self.pandas_params = kwargs
        super().__init__(path)

    def load_data(self) -> pd.DataFrame:
        data = pd.read_hdf(self.path, **self.pandas_params, low_memory=False)
        if isinstance(data, pd.DataFrame):
            return data
        else:
            raise NotImplementedError


class SQLDataSource(DataSource):
    def __init__(
        self,
        connection_string: str,
        query: str,
        preprocess_fun: Optional[
            Callable[[pd.DataFrame], pd.DataFrame]
        ] = None,
    ) -> None:
        self.connection_string = connection_string
        self.query = query
        self.preprocess_fun = preprocess_fun

    def load_data(self) -> pd.DataFrame:
        df = pd.read_sql(self.query, self.connection_string, index_col="index")
        if self.preprocess_fun is not None:
            df = self.preprocess_fun(df)
        return df


class DownloadFileDataSource(DataSource):
    def __init__(
        self,
        url: str,
        file_data_source: FileDataSource,
        overwrite: bool = False,
    ) -> None:
        self.url = url
        self.file_data_source = file_data_source
        self.overwrite = overwrite

    def load_data(self) -> pd.DataFrame:
        path = self.file_data_source.get_path()
        if self.overwrite or (not os.path.exists(path)):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            data = requests.get(self.url)
            with open(path, "wb") as file:
                file.write(data.content)
        return self.file_data_source.load_data()


class TaskProcessor(ABC):
    @abc.abstractmethod
    def transform(self, task_data: pd.Series) -> pd.Series:
        pass


class Task:
    def __init__(
        self,
        name: Union[int, str],
        task_type: str,
        evaluation_metric: str,
        task_processor: TaskProcessor = None,
    ):
        self.name = name
        self.type = task_type
        self.evaluation_metric = evaluation_metric
        self.task_processor = task_processor


class Sorter(ABC):
    @abc.abstractmethod
    def get_index(
        self, data: pd.DataFrame
    ) -> Union[pd.Series, npt.NDArray[np.int_]]:
        pass


class DefaultIndexSorter(Sorter):
    def get_index(
        self, data: pd.DataFrame
    ) -> Union[pd.Series, npt.NDArray[np.int_]]:
        return np.arange(len(data))


class DateSorter(Sorter):
    def __init__(self, date_col: str) -> None:
        self.date_col = date_col

    def get_index(
        self, data: pd.DataFrame
    ) -> Union[pd.Series, npt.NDArray[np.int_]]:
        return pd.to_datetime(data[self.date_col])


class Splitter(ABC):
    @abc.abstractmethod
    def get_splits(self, dataset: Dataset) -> Dict[str, npt.NDArray[np.int_]]:
        pass


class DefaultSplitter(Splitter):
    def __init__(
        self, test_size: Union[float, int] = 0.2, random_seed: int = 42
    ) -> None:
        self.test_size = test_size
        self.random_seed = random_seed

    def get_splits(self, dataset: Dataset) -> Dict[str, npt.NDArray[np.int_]]:
        _, y = dataset.get_x_y()
        i = np.arange(len(y))
        i_train, i_test = train_test_split(
            i,
            random_state=self.random_seed,
            shuffle=True,
            stratify=y[i],
            test_size=self.test_size,
        )
        i_train, i_val = train_test_split(
            i_train,
            random_state=self.random_seed,
            shuffle=True,
            stratify=y[i_train],
            test_size=self.test_size,
        )
        return {"train": i_train, "val": i_val, "test": i_test}


class Dataset:
    def __init__(
        self,
        name: str,
        data_source: DataSource,
        metadata_source: DataSource,
        tasks: List[Task],
        sorter: Sorter,
        splitter: Union[Splitter, None],
        relation_constraints: List[BaseRelationConstraint] = None,
    ):

        self.name = name
        self.data_source = data_source
        self.metadata_source = metadata_source
        self.splitter = splitter
        self.tasks = tasks
        self.sorter = sorter
        self.relation_constraints = relation_constraints

        self.data: Optional[pd.DataFrame] = None
        self.metadata: Optional[pd.DataFrame] = None

    def get_name(self) -> str:
        return self.name

    def set_data(self, df: pd.DataFrame, filter_constraints: bool = False):
        data = df.copy()
        if filter_constraints:
            checker = ConstraintChecker(self.get_constraints(), tolerance=0.01)
            check = checker.check_constraints(self.data, data)
            data = data[check]
        self.data = data

    def get_data(self) -> pd.DataFrame:
        if self.data is None:
            self.data = self.data_source.load_data()
        return self.data.copy()

    def get_metadata(self, only_x: bool = False) -> pd.DataFrame:
        if self.metadata is None:
            self.metadata = self.metadata_source.load_data()
        metadata = self.metadata.copy()
        if only_x:
            x_col = self.get_x_y()[0].columns
            metadata = metadata[metadata["feature"].isin(x_col)]

        return metadata

    def get_ddpm_transformations(self):

        return {
            "seed": 0,
            "normalization": "quantile",
            "num_nan_policy": "__none__",
            "cat_nan_policy": "__none__",
            "cat_min_frequency": "__none__",
            "cat_encoding": "__none__",
            "y_policy": "default",
        }

    def get_x_y(
        self, keep_date: bool = False
    ) -> Tuple[
        pd.DataFrame, Union[npt.NDArray[np.int_], npt.NDArray[np.float_]]
    ]:
        data = self.get_data()
        y = []
        for task in self.tasks:
            task_data = data[task.name]
            if task.task_processor is not None:
                task_data = task.task_processor.transform(task_data)
            data = data.drop(columns=task.name)
            y.append(task_data)

        y_np = np.array(y)
        if y_np.shape[0] == 1:
            y_np = y_np.ravel()

        if isinstance(self.sorter, DateSorter) and (not keep_date):
            data = data.drop(columns=self.sorter.date_col)

        return data, y_np

    def get_x_y_t(self, keep_date: bool = False) -> Tuple[
        pd.DataFrame,
        Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        Union[pd.Series, npt.NDArray[np.int_]],
    ]:
        x, y = self.get_x_y(keep_date=True)
        t = self.sorter.get_index(x)
        if isinstance(self.sorter, DateSorter) and (not keep_date):
            x = x.drop(columns=self.sorter.date_col)
        return x, y, t

    def get_splits(self) -> Dict[str, npt.NDArray[np.int_]]:
        if self.splitter is None:
            raise SplitterMissingError
        return self.splitter.get_splits(self)

    def get_constraints(self) -> Constraints:
        metadata = self.get_metadata()
        col_filter = self.get_x_y()[0].columns.to_list()
        relation_constraints = self.relation_constraints
        if relation_constraints is None:
            relation_constraints = []
        return get_constraints_from_metadata(
            metadata, relation_constraints, col_filter
        )
