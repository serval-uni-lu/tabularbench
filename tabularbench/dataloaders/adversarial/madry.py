import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tabularbench.attacks.capgd.capgd import CAPGD
from tabularbench.attacks.cpgd.cpgd import CPGD
from tabularbench.dataloaders.dataloader import BaseDataLoader
from tabularbench.datasets.dataset import Dataset
from tabularbench.models.torch_models import BaseModelTorch
from tabularbench.utils.format import check_same_format

attacks = {"cpgd": CPGD, "capgd": CAPGD}


def get_adversarial_collate(attack, augment, custom_loader):

    def custom_collate(batch):
        collated = torch.utils.data.dataloader.default_collate(batch)
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


class MadryATDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset: Dataset = None,
        scaler=None,
        model: BaseModelTorch = None,
        attack="pgd",
        attack_params={"eps": 0.3, "norm": "L2"},
        move_all_to_model_device: bool = True,
        filter_constraints: bool = False,
        augment: int = 1,
        verbose=False,
    ) -> None:
        self.num_workers = 0
        self.dataset = dataset
        self.device = model.device
        self.model = model.wrapper_model
        self.move_all_to_model_device = move_all_to_model_device
        self.custom_scaler = None

        constraints = dataset.get_constraints()
        fix_equality_constraints_end = True
        if not attack.startswith("c-"):
            constraints.relation_constraints = None
            fix_equality_constraints_end = False

        self.attack = attacks.get(attack, CPGD)(
            constraints=constraints,
            scaler=scaler,
            model=self.model,
            fix_equality_constraints_end=fix_equality_constraints_end,
            fix_equality_constraints_iter=False,
            model_objective=model.predict_proba,
            **attack_params,
            verbose=verbose,
        )
        self.filter_constraints = filter_constraints
        self.augment = augment
        self.scaler = scaler

    def set_tensors(self, tensors, scaler):
        self.dataset.tensors = tensors
        self.custom_scaler = scaler

    def get_dataloader(
        self, x: np.ndarray, y: np.ndarray, train: bool, batch_size: int
    ) -> DataLoader[Tuple[torch.Tensor, ...]]:
        # Create dataset
        x_scaled = torch.Tensor(self.scaler.transform(x)).float()
        x = torch.Tensor(x).float()
        y = torch.Tensor(y).long()
        if self.move_all_to_model_device:
            x = x.to(self.device)
            y = y.to(self.device)

        self.dataset = TensorDataset(x, y)

        # Create dataloader
        if train:
            self.dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=get_adversarial_collate(
                    self.attack, self.augment, self
                ),
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
