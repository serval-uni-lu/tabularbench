from typing import Dict, List

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from tabularbench.constraints.relation_constraint import BaseRelationConstraint
from tabularbench.constraints.relation_constraint import Constant as Co
from tabularbench.constraints.relation_constraint import Feature as Fe
from tabularbench.constraints.relation_constraint import SafeDivision, Value
from tabularbench.datasets.dataset import (
    CsvDataSource,
    Dataset,
    DataSource,
    DefaultIndexSorter,
    DownloadFileDataSource,
    Splitter,
    Task,
)


class Ctu13Splitter(Splitter):
    def get_splits(self, dataset: Dataset) -> Dict[str, npt.NDArray[np.int_]]:
        _, y = dataset.get_x_y()
        i = np.arange(len(y))
        i_train = i[:143046]
        i_test = i[143046:]
        i_train, i_val = train_test_split(
            i_train,
            random_state=1319,
            shuffle=True,
            stratify=y[i_train],
            test_size=0.2,
        )
        return {"train": i_train, "val": i_val, "test": i_test}


def get_relation_constraints(
    metadata: DataSource,
) -> List[BaseRelationConstraint]:

    features = metadata.load_data()["feature"].to_list()

    def get_feature_family(features_l: List[str], family: str) -> List[str]:
        return list(filter(lambda el: el.startswith(family), features_l))

    def sum_list_feature(features_l: List[str]) -> Value:
        out: Value = Fe(features_l[0])
        for el in features_l[1:]:
            out = out + Fe(el)
        return out

    def sum_feature_family(features_l: List[str], family: str) -> Value:
        return sum_list_feature(get_feature_family(features_l, family))

    g1 = (
        sum_feature_family(features, "icmp_sum_s_")
        + sum_feature_family(features, "udp_sum_s_")
        + sum_feature_family(features, "tcp_sum_s_")
    ) == (
        sum_feature_family(features, "bytes_in_sum_s_")
        + sum_feature_family(features, "bytes_out_sum_s_")
    )

    g2 = (
        sum_feature_family(features, "icmp_sum_d_")
        + sum_feature_family(features, "udp_sum_d_")
        + sum_feature_family(features, "tcp_sum_d_")
    ) == (
        sum_feature_family(features, "bytes_in_sum_d_")
        + sum_feature_family(features, "bytes_out_sum_d_")
    )

    g_packet_size = []
    for e in ["s", "d"]:
        # -1 cause ignore last OTHER features
        bytes_outs = get_feature_family(features, f"bytes_out_sum_{e}_")[:-1]
        pkts_outs = get_feature_family(features, f"pkts_out_sum_{e}_")[:-1]
        if len(bytes_outs) != len(pkts_outs):
            raise Exception("len(bytes_out) != len(pkts_out)")

        # Tuple of list to list of tuples
        for byte_out, pkts_out in list(zip(bytes_outs, pkts_outs)):
            g = SafeDivision(Fe(byte_out), Fe(pkts_out), Co(0.0)) <= Co(1500)
            g_packet_size.append(g)

    g_min_max_sum = []
    for e_1 in ["bytes_out", "pkts_out", "duration"]:
        for port in [
            "1",
            "3",
            "8",
            "10",
            "21",
            "22",
            "25",
            "53",
            "80",
            "110",
            "123",
            "135",
            "138",
            "161",
            "443",
            "445",
            "993",
            "OTHER",
        ]:
            for e_2 in ["d", "s"]:
                g_min_max_sum.extend(
                    [
                        Fe(f"{e_1}_max_{e_2}_{port}")
                        <= Fe(f"{e_1}_sum_{e_2}_{port}"),
                        Fe(f"{e_1}_min_{e_2}_{port}")
                        <= Fe(f"{e_1}_sum_{e_2}_{port}"),
                        Fe(f"{e_1}_min_{e_2}_{port}")
                        <= Fe(f"{e_1}_max_{e_2}_{port}"),
                    ]
                )

    return [g1, g2] + g_packet_size + g_min_max_sum


def create_dataset() -> Dataset:
    data_source = DownloadFileDataSource(
        url="https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
        "thibault_simonetto_uni_lu/"
        "Ed5Wox3GpUtChu5ZYzMfACEB3PDRE3fMloUw04MSNiAkTQ?download=1",
        file_data_source=CsvDataSource(
            path="./data/tabularbench/ctu_13_neris/ctu_13_neris.csv"
        ),
    )
    metadata_source = DownloadFileDataSource(
        url="https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
        "thibault_simonetto_uni_lu/"
        "ESz5WJPXeF1Fv_i3vVk9cY8B-fqbrYuaaUy_e_iypmZFbQ?download=1",
        file_data_source=CsvDataSource(
            path="./data/tabularbench/ctu_13_neris/ctu_13_neris_metadata.csv"
        ),
    )
    tasks = [
        Task(
            name="is_botnet",
            task_type="classification",
            evaluation_metric="f1_score",
        )
    ]
    sorter = DefaultIndexSorter()
    splitter = Ctu13Splitter()
    relation_constraints = get_relation_constraints(metadata_source)

    ctu_13_neris = Dataset(
        name="ctu_13_neris",
        data_source=data_source,
        metadata_source=metadata_source,
        tasks=tasks,
        sorter=sorter,
        splitter=splitter,
        relation_constraints=relation_constraints,
    )
    return ctu_13_neris


datasets = [
    {
        "name": "ctu_13_neris",
        "fun_create": create_dataset,
    }
]
