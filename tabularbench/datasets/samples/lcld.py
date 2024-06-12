from typing import Dict, List

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split

from tabularbench.constraints.relation_constraint import (
    BaseRelationConstraint,
    Constant,
    EqualConstraint,
    Feature,
    SafeDivision,
)
from tabularbench.datasets.dataset import (
    CsvDataSource,
    Dataset,
    DateSorter,
    DownloadFileDataSource,
    Splitter,
    Task,
)


class LcldSplitterTime(Splitter):
    def get_splits(self, dataset: Dataset) -> Dict[str, npt.NDArray[np.int_]]:
        _, _, t = dataset.get_x_y_t()
        i_train = np.where(
            (pd.to_datetime("2013-01-01") <= t)
            & (t <= pd.to_datetime("2015-06-30"))
        )[0]
        i_val = np.where(
            (pd.to_datetime("2015-07-01") <= t)
            & (t <= pd.to_datetime("2015-12-31"))
        )[0]
        i_test = np.where(
            (pd.to_datetime("2016-01-01") <= t)
            & (t <= pd.to_datetime("2017-12-31"))
        )[0]
        return {"train": i_train, "val": i_val, "test": i_test}


class LcldSplitterIid(LcldSplitterTime):
    def get_splits(self, dataset: Dataset) -> Dict[str, npt.NDArray[np.int_]]:
        _, y, _ = dataset.get_x_y_t()
        splits = super(LcldSplitterIid, self).get_splits(dataset)
        merged = np.concatenate(
            [splits.get("train"), splits.get("val"), splits.get("test")]
        )
        i_train, i_test = train_test_split(
            merged,
            random_state=100,
            shuffle=True,
            stratify=y[merged],
            test_size=splits["test"].shape[0],
        )
        i_train, i_val = train_test_split(
            i_train,
            random_state=200,
            shuffle=True,
            stratify=y[i_train],
            test_size=splits["val"].shape[0],
        )
        return {"train": i_train, "val": i_val, "test": i_test}


def get_relation_constraints() -> List[BaseRelationConstraint]:

    int_rate = Feature("int_rate") / Constant(1200)
    term = Feature("term")
    installment = Feature("loan_amnt") * (
        (int_rate * ((Constant(1) + int_rate) ** term))
        / ((Constant(1) + int_rate) ** term - Constant(1))
    )

    # g1 = (Feature("installment") - installment <= Constant(0.01)) and (
    #     Feature("installment") - installment >= Constant(-0.01)
    # )

    g1 = EqualConstraint(Feature("installment"), installment, Constant(0.01))

    g2 = Feature("open_acc") <= Feature("total_acc")

    g3 = Feature("pub_rec_bankruptcies") <= Feature("pub_rec")

    g4 = (Feature("term") == Constant(36)) | (Feature("term") == Constant(60))

    g5 = Feature("ratio_loan_amnt_annual_inc") == (
        Feature("loan_amnt") / Feature("annual_inc")
    )

    g6 = Feature("ratio_open_acc_total_acc") == (
        Feature("open_acc") / Feature("total_acc")
    )

    # g7 was diff_issue_d_earliest_cr_line
    # g7 is not necessary in v2
    # issue_d and d_earliest cr_line are replaced
    # by month_since_earliest_cr_line

    g8 = Feature("ratio_pub_rec_month_since_earliest_cr_line") == (
        Feature("pub_rec") / Feature("month_since_earliest_cr_line")
    )

    g9 = Feature(
        "ratio_pub_rec_bankruptcies_month_since_earliest_cr_line"
    ) == (
        Feature("pub_rec_bankruptcies")
        / Feature("month_since_earliest_cr_line")
    )

    g10 = Feature("ratio_pub_rec_bankruptcies_pub_rec") == SafeDivision(
        Feature("pub_rec_bankruptcies"), Feature("pub_rec"), Constant(-1)
    )

    return [g1, g2, g3, g4, g5, g6, g8, g9, g10]


def is_linked_bool(list_params: List[bool]) -> bool:
    saw_false = False

    for e in list_params:
        saw_false = saw_false or (not e)
        if saw_false and e:
            return False
    return True


def create_dataset(
    split_date: bool, date_1317: bool, simulate_date: bool
) -> Dataset:

    if not is_linked_bool([split_date, date_1317, simulate_date]):
        raise NotImplementedError

    tasks = [
        Task(
            name="charged_off",
            task_type="classification",
            evaluation_metric="f1_score",
        )
    ]
    sorter = DateSorter(date_col="issue_d")
    data_source = DownloadFileDataSource(
        url="https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
        "thibault_simonetto_uni_lu/"
        "EeKpMXnQNo9CuRaLcHNjiX0B4Tf2H_HV3OKmlqvwbZZ-aA?download=1",
        file_data_source=CsvDataSource(
            path="./data/datasets/lcld_v2/lcld_v2.csv"
        ),
    )
    metadata_source = DownloadFileDataSource(
        url="https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
        "thibault_simonetto_uni_lu/"
        "EfIRH-UBrW1BiYwvSjlyZIIBFbOiALcxUzdZ5T11qhILlw?download=1",
        file_data_source=CsvDataSource(
            path="./data/datasets/lcld_v2/lcld_v2_metadata.csv"
        ),
    )

    if split_date:
        name = "lcld_time"
        splitter = LcldSplitterTime()
    else:
        name = "lcld_v2_iid"
        splitter = LcldSplitterIid()

    if date_1317:
        name = "lcld_201317_time"

        data_source = DownloadFileDataSource(
            url="https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
            "thibault_simonetto_uni_lu/"
            "EYRPThLbct9DlYcWsKUbRlgBES9QO2Nk8hZs1cmrQ7tzMQ?download=1",
            file_data_source=CsvDataSource(
                path="./data/datasets/lcld_201317/lcld_201317.csv"
            ),
        )
        metadata_source = DownloadFileDataSource(
            url="https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
            "thibault_simonetto_uni_lu/"
            "EUS4y7I7-FBOrMKV7tb7tmsBUCHdAkCBKXaqEu4V-YuZeQ?download=1",
            file_data_source=CsvDataSource(
                path="./data/datasets/lcld_201317/lcld_201317_metadata.csv"
            ),
        )

        if simulate_date:
            name = "lcld_201317_ds_time"

            data_source = DownloadFileDataSource(
                url="https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
                "thibault_simonetto_uni_lu/"
                "EVMe8n0hiEJCuf3Y1oO7BmwBBgL0uzAAizboz-aSlcxwBA?download=1",
                file_data_source=CsvDataSource(
                    path="./data/datasets/lcld_201317_ds/lcld_201317_ds.csv"
                ),
            )
            metadata_source = DownloadFileDataSource(
                url="https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
                "thibault_simonetto_uni_lu/"
                "EZ9ttVvRzSZPjGtcnmxUwYEBZsxhByGttYAU9GGzX5kosQ?download=1",
                file_data_source=CsvDataSource(
                    path="./data/datasets/lcld_201317_ds/lcld_201317_ds_metadata.csv"
                ),
            )

    dataset = Dataset(
        name=name,
        data_source=data_source,
        metadata_source=metadata_source,
        tasks=tasks,
        sorter=sorter,
        splitter=splitter,
        relation_constraints=get_relation_constraints(),
    )
    return dataset


# V2 for backward compatibility
datasets = [
    {
        "name": "lcld_iid",
        "fun_create": lambda: create_dataset(False, False, False),
    },
    {
        "name": "lcld_v2_iid",
        "fun_create": lambda: create_dataset(False, False, False),
    },
    {
        "name": "lcld_time",
        "fun_create": lambda: create_dataset(True, False, False),
    },
    {
        "name": "lcld_v2_time",
        "fun_create": lambda: create_dataset(True, False, False),
    },
    {
        "name": "lcld_201317_time",
        "fun_create": lambda: create_dataset(True, True, False),
    },
    {
        "name": "lcld_201317_ds_time",
        "fun_create": lambda: create_dataset(True, True, True),
    },
]
