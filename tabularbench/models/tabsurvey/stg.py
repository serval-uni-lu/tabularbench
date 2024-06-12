import os
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parameter import Parameter

from tabularbench.models.model import Model
from tabularbench.models.tab_scaler import ScalerData, TabScaler
from tabularbench.models.torch_models import BaseModelTorch
from tabularbench.utils.io import parent_exists
from tabularbench.utils.load_do_save import load_json, save_json

from .stg_lib import STG as STGModel

"""
    Feature Selection using Stochastic Gates (https://arxiv.org/abs/1810.04247)

    Code adapted from: https://github.com/runopti/stg
"""


class WrapperModel(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        in_dict = {"input": x}

        out_dict = self.model(in_dict)

        return out_dict["prob"]

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return super().parameters(recurse)


class STG(BaseModelTorch):
    def __init__(
        self,
        objective: str,
        x_metadata: pd.DataFrame,
        num_classes,
        batch_size,
        learning_rate: float = 1e-4,
        lam=0.1,
        hidden_dims=None,  # Default [400, 200] in constructor
        name: str = "stg",
        scaler: Optional[TabScaler] = None,
        **kwargs,
    ):
        super().__init__(
            objective=objective,
            x_metadata=x_metadata,
            learning_rate=learning_rate,
            lam=lam,
            hidden_dims=hidden_dims,
            name=name,
            scaler=scaler,
            batch_size=batch_size,
            num_classes=num_classes,
            **kwargs,
        )

        if hidden_dims is None:
            [400, 200]
        self.num_classes = num_classes
        self.x_metadata = x_metadata
        self.num_features = x_metadata.shape[0]
        self.cat_idx = np.where(x_metadata["type"] == "cat")[0].tolist()
        self.num_idx = list(set(range(self.num_features)) - set(self.cat_idx))
        self.cat_dims = [
            (int(x_metadata.iloc[i]["max"]) + 1) for i in self.cat_idx
        ]
        self.scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
        if scaler is not None:
            self.scaler.fit_scaler_data(scaler.get_scaler_data())
        else:
            self.scaler.fit_metadata(x_metadata)
        input_dim = self.scaler.get_transformed_num_features()

        task = (
            "classification" if self.objective == "binary" else self.objective
        )
        out_dim = 2 if self.objective == "binary" else self.num_classes

        self.model = STGModel(
            task_type=task,
            input_dim=input_dim,
            output_dim=out_dim,
            activation="tanh",
            sigma=0.5,
            optimizer="SGD",
            feature_selection=True,
            random_state=1,
            device=self.device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lam=lam,
            hidden_dims=hidden_dims,
            scaler=self.scaler,
        )  # hidden_dims=[500, 50, 10],

        self.wrapper_model = nn.Sequential(
            self.scaler.get_transorm_nn(),
            WrapperModel(self.model._model),
        )

    def save(self, path: str) -> None:
        kwargs_save = deepcopy(self.constructor_kwargs)
        if "x_metadata" in kwargs_save:
            kwargs_save.pop("x_metadata")

        if "scaler" in kwargs_save:
            kwargs_save.pop("scaler")

        args_path = f"{path}/args.json"
        save_json(kwargs_save, parent_exists(args_path))

        if self.scaler is not None:
            scaler_path = f"{path}/scaler.json"
            self.scaler.get_scaler_data().save(parent_exists(scaler_path))

        weigths_path = f"{path}/weights.pt"
        if self.scaler is None:
            torch.save(
                self.model._model.state_dict(), parent_exists(weigths_path)
            )
        else:
            torch.save(
                self.model._model.state_dict(), parent_exists(weigths_path)
            )

    def get_logits(self, x: torch.Tensor, with_grad: bool) -> torch.Tensor:
        if not with_grad:
            with torch.no_grad():
                # with_grad = True to avoid recursion
                return self.get_logits(x, with_grad=True)

        preds = self.wrapper_model(x.to(self.device))
        if self.objective == "binary":
            preds = torch.sigmoid(preds)
        return preds

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
        if model.scaler is None:
            model.model._model.load_state_dict(
                torch.load(weigths_path, map_location=torch.device("cpu")),
                strict=False,
            )
        else:
            model.model._model.load_state_dict(
                torch.load(weigths_path, map_location=torch.device("cpu")),
                strict=False,
            )

        return model

    def fit(self, X, y, X_val=None, y_val=None, custom_train_dataloader=None):
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        if X_val is None:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        if X_val is not None and isinstance(X_val, torch.Tensor):
            X_val = X_val.numpy()

        X, X_val = X.astype("float"), X_val.astype("float")

        if self.objective == "regression":
            y, y_val = y.reshape(-1, 1), y_val.reshape(-1, 1)

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
            X_val = self.scaler.transform(X_val)

        loss, val_loss = self.model.fit(
            X,
            y.numpy().astype(float),
            nr_epochs=self.epochs,
            valid_X=X_val,
            valid_y=y_val.numpy().astype(float),
            print_interval=1,
            loss=self.loss_func,
            custom_train_dataloader=custom_train_dataloader,
        )  # self.args.logging_period # early_stop=True

        return loss, val_loss

    def predict_helper(self, X):
        if isinstance(X, list):
            X = torch.tensor(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def get_model_size(self):
        model_size = sum(
            t.numel()
            for t in self.model._model.parameters()
            if t.requires_grad
        )
        return model_size

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 1e-1, log=True
            ),
            "lam": trial.suggest_float("lam", 1e-3, 10, log=True),
            # Change also the number and size of the hidden_dims?
            "hidden_dims": trial.suggest_categorical(
                "hidden_dims",
                [[500, 50, 10], [60, 20], [500, 500, 10], [500, 400, 20]],
            ),
        }
        return params

    @staticmethod
    def get_default_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            "learning_rate": 1e-4,
            "lam": 0.1,
            "hidden_dims": [60, 20],
        }

        return params

    def parameters(self):
        # print("STG parameters")
        return self.wrapper_model[-1].parameters()

    @property
    def training(self):
        return self.wrapper_model.training

    @staticmethod
    def get_name() -> str:
        return "stg"


models: List[Tuple[str, Type[Model]]] = [("stg", STG)]
