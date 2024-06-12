from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from tabularbench.attacks.capgd.capgd import CAPGD
from tabularbench.attacks.cpgd.cpgd import CPGD
from tabularbench.dataloaders.dataloader import BaseDataLoader
from tabularbench.datasets.dataset import Dataset
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.models.torch_models import BaseModelTorch
from tabularbench.utils.datatypes import to_torch_number
from tabularbench.utils.format import check_same_format

attacks = {"cpgd": CPGD, "capgd": CAPGD}


def get_normal_collate(custom_loader):
    def custom_collate(batch):
        collated = torch.utils.data.dataloader.default_collate(batch)
        augment = custom_loader.augment % 10
        y_type = collated[1].dtype

        if augment == 1:
            x, y = (
                collated[0][: len(collated[0]) // 2],
                collated[1][: len(collated[0]) // 2],
            )
            x_orig, y_orig = (
                collated[0][len(collated[0]) // 2 :],
                collated[1][len(collated[0]) // 2 :],
            )
        if augment == 2:
            x, y = collated[0], collated[1]
            x_orig, y_orig = x.clone().to(x.device), y.clone().to(x.device)
        else:
            x, y = collated[0], collated[1]
            x_orig, y_orig = torch.Tensor([]).to(x.device), torch.Tensor(
                []
            ).to(x.device)

        x_augmented, y_augmented = get_data(custom_loader, x.shape[0])
        x_augmented = x_augmented.to(x.device)
        y_augmented = y_augmented.to(x.device)

        # For the hot fix, here we add it just so it has the good shape to enter the scaler (or exit)
        if custom_loader.mlc_dataset.name == "ctu_13_neris":
            x_augmented = torch.cat(
                [x_augmented, x_orig[:, -1].unsqueeze(1)], dim=1
            )

        if (len(y_orig.shape) == 2) and (y_orig.shape[1] == 2):
            y_augmented = torch.column_stack([1 - y_augmented, y_augmented])

        if custom_loader.dataloader.scaler is not None:
            x_augmented = custom_loader.dataloader.scaler.transform(
                x_augmented
            )
            # For the hot fix, here we reset the value cause if we need it scaled
            # It means it was already scaled in the x_orig and we scaled the scaled,
            # So reset
            if custom_loader.mlc_dataset.name == "ctu_13_neris":
                x_augmented[:, -1] = x_orig[:, -1]

        check_same_format(x_orig, x_augmented)

        y_augmented = y_augmented.type(y_type)

        return torch.cat(
            [x_augmented.to(x.device), x_orig.to(x.device)], 0
        ), torch.cat([y_augmented.to(x.device), y_orig.to(x.device)], 0)

    return custom_collate


def get_adversarial_collate(attack, custom_loader):
    def custom_collate(batch):

        collated = get_normal_collate(custom_loader)(batch)
        if attacks is None:
            return collated

        x = collated[0].to(attack.device)
        y = collated[1].to(attack.device)

        n = len(x) // 4
        x_clean = torch.cat([x[:n], x[2 * n : 3 * n]])
        x_to_attack = torch.cat([x[n : 2 * n], x[3 * n :]])
        y_adv = torch.cat([y[n : 2 * n], y[3 * n :]])
        others = [
            torch.cat([e[:n], e[2 * n : 3 * n], e[n : 2 * n], e[3 * n :]])
            for e in collated[1:]
        ]

        y_adv = (
            y_adv
            if custom_loader.dataloader.transform_y is None
            else y_adv[:, 1].long()
        )

        if custom_loader.dataloader.scaler is None:
            adv_x = attack(x_to_attack, y_adv)
        else:
            adv_x = custom_loader.dataloader.scaler.transform(
                attack(
                    custom_loader.dataloader.scaler.inverse_transform(
                        x_to_attack
                    ),
                    y_adv,
                )
            )

        check_same_format(x_clean, adv_x)
        return torch.cat([x_clean, adv_x]), *others

    return custom_collate


def get_path_in(root: str, n: int, dataset: str, gan_model: str) -> str:
    return f"{root}/{gan_model}/{dataset}/synthetic_{n}.parquet"


def generate(
    tensor_in: torch.Tensor,
    root: str,
    n: int,
    dataset: str,
    gan_model: str,
    task_name: str,
) -> torch.Tensor:
    print(f"Generating {n}")
    path = get_path_in(root, n, dataset, gan_model)
    if not Path(path).exists():
        return df

    df = pd.read_parquet(path)

    label_col = task_name
    if label_col not in df.columns:
        label_col = "target"
    x = df
    y = x.pop(label_col)
    x, y = to_torch_number(x), to_torch_number(y)
    xy = torch.cat([x, y.unsqueeze(1)], dim=1)

    return torch.cat([tensor_in, xy])


def keep_data(custom_loader: GanFileDataLoader, n: int):
    while len(custom_loader.synthetic_data) < n:
        print(f"Loading {custom_loader.synthetic_batch}")
        previous_len = len(custom_loader.synthetic_data)
        custom_loader.synthetic_data = generate(
            custom_loader.synthetic_data,
            custom_loader.synthetic_root_path,
            custom_loader.synthetic_batch,
            custom_loader.mlc_dataset.name,
            custom_loader.gan_name,
            custom_loader.mlc_dataset.tasks[0].name,
        )
        if previous_len == len(custom_loader.synthetic_data):
            if custom_loader.synthetic_batch == 0:
                print(
                    get_path_in(
                        custom_loader.synthetic_root_path,
                        custom_loader.synthetic_batch,
                        custom_loader.mlc_dataset.name,
                        custom_loader.gan_name,
                    )
                )
                raise FileNotFoundError
            custom_loader.synthetic_batch = 0
        custom_loader.synthetic_batch += 1


def get_data(
    custom_loader: GanFileDataLoader, n: int
) -> Tuple[pd.DataFrame, pd.Series]:
    keep_data(custom_loader, n)

    n_samples = n

    xy_out = custom_loader.synthetic_data[:n_samples]
    custom_loader.synthetic_data = custom_loader.synthetic_data[n_samples:]

    return xy_out[:, :-1], xy_out[:, -1]


class GanFileDataLoader(BaseDataLoader):

    def __init__(
        self,
        dataset: Dataset = None,
        scaler=None,
        model: BaseModelTorch = None,
        attack="pgd",
        attack_params={"eps": 0.3, "norm": "L2"},
        move_all_to_model_device: bool = True,
        filter_constraints: bool = False,
        augment: int = 11,
        ratio: float = 0.5,
        verbose=False,
        device=None,
        synthetic_root_path="",
        gan_name="",
        mlc_dataset=None,
    ) -> None:
        self.num_workers = 0
        self.dataset = dataset
        self.device = (
            device
            if device is not None
            else (model.device if hasattr(model, "device") else "cpu")
        )
        self.model = model.wrapper_model
        self.move_all_to_model_device = move_all_to_model_device
        self.custom_scaler = None

        constraints = dataset.get_constraints()
        fix_equality_constraints_end = True
        if not attack.startswith("c-"):
            constraints.relation_constraints = None
            fix_equality_constraints_end = False

        self.attack = (
            attacks.get(attack, CPGD)(
                constraints=constraints,
                scaler=scaler,
                model=self.model,
                fix_equality_constraints_end=fix_equality_constraints_end,
                fix_equality_constraints_iter=False,
                model_objective=model.predict_proba,
                **attack_params,
                verbose=verbose,
            )
            if "cpgd" in attack
            else None
        )

        self.filter_constraints = filter_constraints

        # augment: 0- replace dataset with synthetic, 1/10-replace half of dataset with synthetic, 2/20- concatenate dataset with synthetic
        self.augment = augment
        self.ratio = ratio
        self.scaler = scaler
        self.synthetic_batch = 0
        self.synthetic_data = torch.Tensor()
        self.batch_size = 0
        self.synthetic_root_path = synthetic_root_path
        self.gan_name = gan_name
        self.mlc_dataset = mlc_dataset

    def set_tensors(self, tensors, scaler):
        self.dataset.tensors = tensors
        self.custom_scaler = scaler

    def get_dataloader(
        self, x: np.ndarray, y: np.ndarray, train: bool, batch_size: int
    ) -> DataLoader[Tuple[torch.Tensor, ...]]:
        # Create dataset
        x = torch.Tensor(x).float()
        y = torch.Tensor(y).long()
        if self.move_all_to_model_device:
            x = x.to(self.device)
            y = y.to(self.device)

        self.dataset = TensorDataset(x, y)
        self.batch_size = batch_size

        # Create dataloader
        if train:

            collate_fn = (
                get_adversarial_collate(attack=self.attack, custom_loader=self)
                if self.attack is not None
                else get_normal_collate(custom_loader=self)
            )
            self.dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
            )

        else:
            self.dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        self.dataloader.scaler = None
        self.dataloader.transform_y = None
        return self.dataloader
