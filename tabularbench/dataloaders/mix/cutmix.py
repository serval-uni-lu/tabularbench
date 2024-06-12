import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tabularbench.attacks.capgd.capgd import CAPGD
from tabularbench.attacks.cpgd.cpgd import CPGD
from tabularbench.dataloaders.dataloader import BaseDataLoader
from tabularbench.datasets.dataset import Dataset
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.models.torch_models import BaseModelTorch
from tabularbench.utils.format import check_same_format

attacks = {"cpgd": CPGD, "capgd": CAPGD}


def get_normal_collate(custom_loader):
    def custom_collate(batch):
        synthetic_label = custom_loader.synthetic_label
        ratio = custom_loader.ratio
        model_label = custom_loader.model_label

        collated = torch.utils.data.dataloader.default_collate(batch)

        augment = custom_loader.augment % 10

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

        index_combinations = np.random.randint(0, len(x), len(x) * 2).reshape(
            2, -1
        )

        split_index = int(ratio * x.shape[1])
        feature_combinations = [
            x[index_combinations[0, :], :split_index],
            x[index_combinations[1, :], split_index:],
        ]
        label_combinations = torch.cat(
            [
                y[index_combinations[0, :]].unsqueeze(0),
                y[index_combinations[1, :]].unsqueeze(0),
            ],
            0,
        )
        x_combined = torch.concat(feature_combinations, 1)
        y_combined = y

        if synthetic_label == "intersection":
            y_combined = label_combinations.prod(0)
        elif synthetic_label == "union":
            y_combined = label_combinations.max(0)[0]
        elif synthetic_label == "ratio":
            y_combined = label_combinations[0] * ratio + label_combinations[
                1
            ] * (1 - ratio)
        elif synthetic_label == "model":
            y_pred = custom_loader.model_label(x_combined)
            if y_pred.shape[1] == 2:
                y_combined = y_pred.argmax(1)
            else:
                y_combined = y_pred

        check_same_format(x_combined, x_orig)

        return torch.cat([x_combined, x_orig], 0), torch.cat(
            [y_combined, y_orig], 0
        )

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


class CutmixDataLoader(BaseDataLoader):

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
        synthetic_label: str = "intersection",
        ratio: float = 0.5,
        model_label: BaseModelTorch = None,
        verbose=False,
        device=None,
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
            attacks.get(attack, CAPGD)(
                constraints=constraints,
                scaler=scaler,
                model=self.model,
                fix_equality_constraints_end=fix_equality_constraints_end,
                fix_equality_constraints_iter=False,
                model_objective=model.predict_proba,
                **attack_params,
                verbose=verbose,
            )
            if "pgd" in attack
            else None
        )

        self.filter_constraints = filter_constraints

        # augment: 0- replace dataset with synthetic, 1/10-replace half of dataset with synthetic, 2/20- concatenate dataset with synthetic
        self.augment = augment

        # synthetic_label
        # "intersection": for every label pair y1, y2, create label y as y = y1*y2
        # "union": for every label pair y1, y2, create label y as y = max(1,y1+y2)
        # "ratio": for every label pair y1, y2, create label y as y = y1*ratio+y2*ratio
        # "model": for every synthetic input x, create label y as model(x)
        self.synthetic_label = synthetic_label
        self.ratio = ratio
        self.model_label = (
            model_label.wrapper_model if model_label is not None else None
        )

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


if __name__ == "__main__":

    from mlc.datasets import dataset_factory
    from mlc.models.tabsurvey.tabtransformer import TabTransformer

    dataset_name = "lcld_v2_iid"
    ds = dataset_factory.get_dataset(dataset_name)
    x, y = ds.get_x_y()
    splits = ds.get_splits()
    x_test = x.iloc[splits["test"]]
    y_test = y[splits["test"]]
    x_train = x.iloc[splits["train"]]
    y_train = y[splits["train"]]

    batch_size = 32
    metadata = ds.get_metadata(only_x=True)

    scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
    scaler.fit(
        torch.tensor(x.values, dtype=torch.float32), x_type=metadata["type"]
    )

    sc = TabTransformer(
        "regression", metadata, batch_size, 10, False, 0.1, 2, scaler=scaler
    )

    # if attack="" : only run cutmix augmentation
    # if attack="cpgd or pgd": run cutmix+adv augmentation
    synthetic_label = "model"  # "intersection" "ratio" "model"
    attack = "pgd"  # ""
    ratio = 0.5
    dataloader_class = CutmixDataLoader(
        dataset=ds,
        model=sc,
        scaler=scaler,
        attack=attack,
        model_label=sc,
        synthetic_label=synthetic_label,
        ratio=ratio,
    )
    train_dataloader = dataloader_class.get_dataloader(
        x_train.values, y_train, batch_size=batch_size, train=True
    )

    for batch in train_dataloader:
        print(batch[0].shape)
