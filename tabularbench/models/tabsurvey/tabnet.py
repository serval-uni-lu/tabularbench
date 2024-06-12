import os
import pickle
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from tabularbench.models.model import Model
from tabularbench.models.tab_scaler import ScalerData, TabScaler
from tabularbench.models.tabsurvey.tabnet_lib.tab_model import (
    TabNetClassifier,
    TabNetRegressor,
)
from tabularbench.models.torch_models import BaseModelTorch
from tabularbench.utils.io import parent_exists
from tabularbench.utils.load_do_save import load_json, save_json

"""
    TabNet: Attentive Interpretable Tabular Learning
    (https://arxiv.org/pdf/1908.07442.pdf)

    See the implementation: https://github.com/dreamquark-ai/tabnet
"""


class NetworkWrapper(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = x[0]
        return x


class TabNet(BaseModelTorch):
    def __init__(
        self,
        objective: str,
        x_metadata: pd.DataFrame,
        num_classes: int,
        n_d: int = 8,
        n_steps: int = 3,
        gamma: float = 1.0,
        cat_emb_dim: int = 1,
        n_independent: int = 1,
        n_shared: int = 1,
        momentum: float = 0.001,
        mask_type: str = "sparsemax",
        name: str = "tabnet",
        scaler: Optional[TabScaler] = None,
        **kwargs,
    ):
        super().__init__(
            objective=objective,
            x_metadata=x_metadata,
            num_classes=num_classes,
            n_d=n_d,
            n_steps=n_steps,
            gamma=gamma,
            cat_emb_dim=cat_emb_dim,
            n_independent=n_independent,
            n_shared=n_shared,
            momentum=momentum,
            mask_type=mask_type,
            name=name,
            scaler=scaler,
            **kwargs,
        )
        self.scaler = TabScaler(num_scaler="min_max", one_hot_encode=False)
        if scaler is not None:
            self.scaler.fit_scaler_data(scaler.get_scaler_data())
        else:
            self.scaler.fit_metadata(x_metadata)
        self.num_classes = num_classes

        # Paper recommends to be n_d and n_a the same
        self.n_a = n_d
        self.n_d = n_d

        self.num_features = x_metadata.shape[0]
        self.cat_idx = np.where(x_metadata["type"] == "cat")[0].tolist()
        self.num_idx = list(set(range(self.num_features)) - set(self.cat_idx))
        self.cat_dims = [
            (int(x_metadata.iloc[i]["max"]) + 1) for i in self.cat_idx
        ]

        if self.objective == "regression":
            self.model = TabNetRegressor()
            self.metric = ["rmse"]
        elif self.objective == "classification" or self.objective == "binary":
            self.model = TabNetClassifier(
                cat_idxs=self.cat_idx,
                cat_dims=self.cat_dims,
                n_a=n_d,
                n_d=n_d,
                n_steps=n_steps,
                gamma=gamma,
                cat_emb_dim=cat_emb_dim,
                n_independent=n_independent,
                n_shared=n_shared,
                momentum=momentum,
                mask_type=mask_type,
                device_name=self.device,
            )
            if self.num_classes == 2:
                self.metric = ["auc"]
            else:
                self.metric = ["balanced_accuracy"]

        self.model.virtual_batch_size = 128
        self.model.output_dim = self.num_classes
        self.model.input_dim = self.num_features
        if self.scaler is not None:
            self.model.input_dim = self.scaler.get_transformed_num_features()
        self.model._set_network()

        self.wrapper_model = nn.Sequential(
            self.scaler.get_transorm_nn(),
            NetworkWrapper(self.model.network),
        )

    def fit(self, X, y, X_val=None, y_val=None, custom_train_dataloader=None):

        if X_val is None:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if X_val is not None and isinstance(X_val, torch.Tensor):
            X_val = X_val.numpy()

        if self.objective == "regression":
            y, y_val = y.reshape(-1, 1), y_val.reshape(-1, 1)

        X = X.astype(np.float32)

        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        if (y.ndim == 1) or (y.shape[1] == 1):
            y = torch.column_stack((1 - y, y))
        if (y_val.ndim == 1) or (y_val.shape[1] == 1):
            y_val = torch.column_stack((1 - y_val, y_val))

        y = y.float()
        y_val = y_val.float()

        self.set_loss_y(y)

        if self.scaler is not None:
            X = self.scaler.transform(X)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)

        self.model.fit(
            X,
            y[:, 1],
            eval_set=[(X_val, y_val[:, 1])],
            eval_name=["eval"],
            eval_metric=self.metric,
            max_epochs=self.epochs,
            patience=self.early_stopping_rounds,
            batch_size=self.batch_size,
            loss_fn=self.loss_func,
            weights=0,
            # scaler=self.scaler,
            compute_importance=False,
            custom_train_dataloader=custom_train_dataloader,
            warm_start=True,
        )
        history = self.model.history
        # self.save_model(filename_extension="best")

        return history["loss"], history["eval_" + self.metric[0]]

    def predict_helper(self, X):
        X = np.array(X, dtype=np.float32)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        if self.objective == "regression":
            return self.model.predict(X)
        elif self.objective == "classification" or self.objective == "binary":
            return self.model.predict_proba(X)

    def old_save(self, path: str) -> None:
        kwargs_save = deepcopy(self.constructor_kwargs)
        if "x_metadata" in kwargs_save:
            kwargs_save.pop("x_metadata")

        if "scaler" in kwargs_save:
            kwargs_save.pop("scaler")

        args_path = f"{path}/args.json"
        save_json(kwargs_save, parent_exists(args_path))

        weigths_path = f"{path}/weights.pickle"
        with open(parent_exists(weigths_path), "wb") as f:
            pickle.dump(self.model, f)
        # if self.scaler is None:
        #     torch.save(self.model.state_dict(), parent_exists(weigths_path))
        # else:
        #     torch.save(self.model[1].state_dict(),
        #     parent_exists(weigths_path))

    @classmethod
    def old_load_class(
        cls, path: str, **kwargs: Dict[str, Any]
    ) -> BaseModelTorch:
        args_path = f"{path}/args.json"
        weigths_path = f"{path}/weights.pickle"

        load_kwargs = load_json(args_path)
        args = {**load_kwargs, **kwargs}
        model = cls(**args)

        with open(weigths_path, "rb") as f:
            model.model = pickle.load(f)

        return model

    def save(self, path: str) -> None:
        kwargs_save = deepcopy(self.constructor_kwargs)
        if "x_metadata" in kwargs_save:
            kwargs_save.pop("x_metadata")

        if "scaler" in kwargs_save:
            kwargs_save.pop("scaler")

        if self.scaler is not None:
            scaler_path = f"{path}/scaler.json"
            self.scaler.get_scaler_data().save(parent_exists(scaler_path))

        args_path = f"{path}/args.json"
        save_json(kwargs_save, parent_exists(args_path))

        weigths_path = f"{path}/weights.pt"

        torch.save(
            self.model.network.state_dict(), parent_exists(weigths_path)
        )

    @classmethod
    def load_class(cls, path: str, **kwargs: Dict[str, Any]) -> BaseModelTorch:
        args_path = f"{path}/args.json"
        weigths_path = f"{path}/weights.pt"
        scaler_path = f"{path}/scaler.json"

        if ("scaler" not in kwargs) or (kwargs["scaler"] is None):
            if os.path.exists(scaler_path):
                scaler = TabScaler()
                scaler_data = ScalerData.load(scaler_path)
                scaler.fit_scaler_data(scaler_data)
                kwargs["scaler"] = scaler

        load_kwargs = load_json(args_path)
        args = {**load_kwargs, **kwargs}
        model = cls(**args)

        model.model.network.load_state_dict(
            torch.load(weigths_path, map_location=torch.device("cpu"))
        )

        return model

    def get_model_size(self):
        # To get the size, the model has be trained for at least one epoch
        model_size = sum(
            t.numel()
            for t in self.model.network.parameters()
            if t.requires_grad
        )
        return model_size

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        params = {
            "n_d": trial.suggest_int("n_d", 8, 64),
            "n_steps": trial.suggest_int("n_steps", 3, 10),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "cat_emb_dim": trial.suggest_int("cat_emb_dim", 1, 3),
            "n_independent": trial.suggest_int("n_independent", 1, 5),
            "n_shared": trial.suggest_int("n_shared", 1, 5),
            "momentum": trial.suggest_float("momentum", 0.001, 0.4, log=True),
            "mask_type": trial.suggest_categorical(
                "mask_type", ["sparsemax", "entmax"]
            ),
        }
        return params

    def get_logits(self, x: torch.Tensor, with_grad: bool) -> torch.Tensor:
        if not with_grad:
            with torch.no_grad():
                # with_grad = True to avoid recursion
                return self.get_logits(x, with_grad=True)

        preds = self.wrapper_model(x.to(self.device))
        if self.objective == "binary":
            preds = torch.sigmoid(preds)
        return preds

    @staticmethod
    def get_default_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            "n_d": 8,
            "n_steps": 3,
            "gamma": 1.0,
            "cat_emb_dim": 1,
            "n_independent": 1,
            "n_shared": 1,
            "momentum": 0.001,
            "mask_type": "sparsemax",
        }
        return params

    def attribute(self, X: np.ndarray, y: np.ndarray, stategy=""):
        """Generate feature attributions for the model input.
        Only strategy are supported: default ("")
        Return attribution in the same shape as X.
        """
        X = np.array(X, dtype=np.float32)
        attributions = self.model.explain(
            torch.tensor(X, dtype=torch.float32)
        )[0]
        return attributions

    def parameters(self):
        print("TabNet parameters")
        return self.wrapper_model[-1].parameters()

    @property
    def training(self):
        return self.wrapper_model.training

    @staticmethod
    def get_name() -> str:
        return "tabnet"


models: List[Tuple[str, Type[Model]]] = [("tabnet", TabNet)]
