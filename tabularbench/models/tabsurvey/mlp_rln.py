from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tabularbench.dataloaders.fast_dataloader import FastTensorDataLoader
# from tabularbench.domain_generatlization.groups import get_random_groups
from tabularbench.models.model import Model
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.models.torch_models import BaseModelTorch
from tabularbench.utils.datatypes import binary_to_2dim, to_torch_number
from tabularbench.utils.typing import NDFloat, NDNumber

"""
    Custom implementation for the standard multi-layer perceptron
"""


class TORCHRLN(BaseModelTorch):
    def __init__(
        self,
        objective: str,
        x_metadata: pd.DataFrame,
        batch_size: int,
        epochs: int,
        early_stopping_rounds: int,
        num_classes: int,
        n_layers: int,
        hidden_dim: int,
        norm: int = 1,
        theta: float = -7.5,
        name: str = "torchrln",
        scaler: Optional[TabScaler] = None,
        **kwargs: Any,
    ) -> None:

        # Parameters
        self.objective = objective
        self.x_metadata = x_metadata
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.norm = norm
        self.theta = theta

        # Super call

        # Generate super call
        super().__init__(
            objective=objective,
            x_metadata=x_metadata,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_rounds=early_stopping_rounds,
            num_classes=num_classes,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            norm=norm,
            theta=theta,
            name=name,
            scaler=scaler,
            **kwargs,
        )

        # Compatibility

        self.experiment = None

        # self.dataset = args.dataset
        lr = self.learning_rate
        self.scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
        if scaler is not None:
            self.scaler.fit_scaler_data(scaler.get_scaler_data())

        # Generated
        self.num_features = (
            x_metadata.shape[0]
            if scaler is None
            else scaler.get_transformed_num_features()
        )

        self.model = MLP_ModelRLN(
            n_layers=self.n_layers,
            input_dim=self.num_features,
            hidden_dim=self.hidden_dim,
            output_dim=self.num_classes,
            task=self.objective,
        )
        layer = self.model.layers[0]
        if self.scaler is not None:
            self.model = nn.Sequential(
                self.scaler.get_transorm_nn(), self.model
            )
            layer = self.model[1].layers[0]

        self.wrapper_model = self.model
        self.to_device()

        # layer = (
        #     self.model.module.layers[0]
        #     if hasattr(self.model, "module")
        #     else self.model.layers[0]
        # )
        self.rln_callback = RLNCallback(
            layer,
            norm=self.norm,
            avg_reg=self.theta,
            learning_rate=lr,
        )

    def parameters(self):
        return self.model.parameters()

    @property
    def training(self):
        return self.model.training

    # def fit(
    #     self,
    #     x: NDFloat,
    #     y: NDNumber,
    #     x_val: Optional[NDFloat] = None,
    #     y_val: Optional[NDNumber] = None,
    #     custom_train_dataloader: DataLoader = None,
    #     custom_val_dataloader: DataLoader = None,
    # ) -> None:
    #     x = np.array(x, dtype=np.float32)

    #     self.rln_callback.on_train_begin()
    #     if x_val is None:
    #         x, x_val, y, y_val = train_test_split(
    #             x, y, test_size=0.2, random_state=42, stratify=y
    #         )
    #     x_val = np.array(x_val, dtype=np.float32)
    #     if not self.scaler.fitted:
    #         self.scaler.fit(x)

    #     if self.scaler is not None:
    #         previous_model = self.model
    #         self.model = self.model[1]
    #         x = self.scaler.transform(x)
    #         x_val = self.scaler.transform(x_val)

    #         out = super(TORCHRLN, self).fit(
    #             x,
    #             y,
    #             x_val,
    #             y_val,
    #             custom_train_dataloader=custom_train_dataloader,
    #             custom_val_dataloader=custom_val_dataloader,
    #             scaler=self.scaler,
    #         )
    #         self.model = previous_model
    #         return out
    #     return super(TORCHRLN, self).fit(x, y, x_val, y_val)

    def fit(
        self,
        x: NDFloat,
        y: NDNumber,
        x_val: Optional[NDFloat] = None,
        y_val: Optional[NDNumber] = None,
        custom_train_dataloader=None,
        custom_val_dataloader=None,
        scaler=None,
    ):
        # if reset_weight:
        #     self.reset_all_weights()

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if x_val is None:
            x, x_val, y, y_val = train_test_split(
                x, y, test_size=0.2, random_state=42, stratify=y
            )

        x = to_torch_number(x)
        x_val = to_torch_number(x_val)
        y = to_torch_number(y)
        y_val = to_torch_number(y_val)

        y = binary_to_2dim(y)
        y_val = binary_to_2dim(y_val)

        if not self.scaler.fitted:
            self.scaler.fit(x)

        self.set_loss_y(y)
        loss_func = self.loss_func

        # y = y.float()
        # y_val = y_val.float()

        # train_loader = self.data_loader.get_dataloader(
        #     x, y, True, self.batch_size
        # )

        if self.dg_group_method == "random":
            groups = get_random_groups(len(x), self.num_groups)
            groups_val = get_random_groups(len(x_val), self.num_groups)
        elif self.dg_group_method == "time":
            groups = (
                torch.tensor(np.arange(self.num_groups))
                .int()
                .repeat(len(x) // self.num_groups + 1)[: len(x)]
            )
            groups_val = (
                torch.tensor(np.arange(self.num_groups))
                .int()
                .repeat(len(x_val) // self.num_groups + 1)[: len(x_val)]
            )

        if custom_train_dataloader:
            if hasattr(custom_train_dataloader, "dataset"):
                custom_train_dataloader.dataset.tensors = x.to(
                    self.device
                ), y.to(self.device)
            else:
                custom_train_dataloader.tensors = x.to(self.device), y.to(
                    self.device
                )
            train_loader = custom_train_dataloader
            train_loader.transform_y = True
        else:
            train_loader = FastTensorDataLoader(
                x.to(self.device),
                y.to(self.device),
                groups.to(self.device),
                batch_size=self.batch_size,
                shuffle=True,
            )

        if custom_val_dataloader:
            custom_val_dataloader.dataset.tensors = (x_val.to(self.device),)
            y_val.to(self.device)
            val_loader = custom_val_dataloader
            train_loader.scaler = scaler
        else:
            val_loader = FastTensorDataLoader(
                x_val.to(self.device),
                y_val.to(self.device),
                groups_val.to(self.device),
                batch_size=self.val_batch_size,
                shuffle=False,
            )

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        for epoch in range(self.epochs):
            print(f"Epoch {epoch}.", flush=True)
            # for i, (batch_x, batch_y, batch_g) in enumerate(train_loader):
            for i, (batch_x, batch_y) in enumerate(train_loader):
                out = self.model(batch_x.to(self.device))

                if (
                    self.objective == "regression"
                    or self.objective == "binary"
                ):
                    out = out.squeeze()
                if callable(getattr(loss_func, "update", None)):
                    loss_func.update(batch_x, batch_y)

                loss = loss_func(out, batch_y.to(self.device))
                loss_history.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Early Stopping
            if self.early_stopping_rounds > 0:
                val_loss = 0.0
                val_dim = 0
                with torch.no_grad():
                    for val_i, (
                        batch_val_x,
                        batch_val_y,
                        batch_val_g,
                    ) in enumerate(val_loader):
                        out = self.model(batch_val_x.to(self.device))

                        if (
                            self.objective == "regression"
                            or self.objective == "binary"
                        ):
                            out = out.squeeze()
                        if callable(getattr(loss_func, "update", None)):
                            loss_func.update(batch_val_x, batch_val_y)
                        val_loss += loss_func(
                            out, batch_val_y.to(self.device), g=batch_val_g
                        ).item()
                        val_dim += 1

                val_loss /= val_dim
                val_loss_history.append(val_loss)

                # logger.debug("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))
                # logger.info("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_val_loss_idx = epoch

                    # Save the currently best model
                    self.save_best_weights()

                if min_val_loss_idx + self.early_stopping_rounds < epoch:
                    # logger.debug(
                    #     "Validation loss has not improved for %d steps!"
                    #     % self.early_stopping_rounds
                    # )
                    # logger.debug("Early stopping applies.")
                    break

        self.model.__hash__()
        # Load best model
        if self.early_stopping_rounds > 0:
            self.load_best_weights()
        return loss_history, val_loss_history

    def predict_helper(
        self,
        x: Union[NDFloat, torch.Tensor, pd.DataFrame],
        load_all_gpu: bool = False,
    ) -> NDFloat:
        x = np.array(x, dtype=float)
        return super().predict_helper(x, load_all_gpu=load_all_gpu)

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 10, 100),
            "n_layers": trial.suggest_int("n_layers", 2, 5),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.0005, 0.001
            ),
            "norm": trial.suggest_categorical("norm", [1, 2]),
            "theta": trial.suggest_int("theta", -12, -8),
        }
        return params

    @staticmethod
    def get_default_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            "hidden_dim": 10,
            "n_layers": 2,
            "learning_rate": 0.001,
            "norm": 1,
            "theta": -7.5,
        }

        return params

    @staticmethod
    def get_name() -> str:
        return "torchrln"


class MLP_ModelRLN(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        task: str,
    ):
        super().__init__()

        self.task = task

        self.layers = nn.ModuleList()

        # Input Layer (= first hidden layer)
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Hidden Layers (number specified by n_layers)
        self.layers.extend(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)]
        )

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(x))

        # Use ReLU as activation for all hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))

        # No activation function on the output
        x = self.output_layer(x)

        if self.task == "classification":
            x = F.softmax(x, dim=1)

        return x


class RLNCallback(object):
    def __init__(
        self,
        layer: nn.Linear,
        norm: int = 1,
        avg_reg: float = -7.5,
        learning_rate: float = 6e5,
    ):
        """
        An implementation of Regularization Learning, described in https://arxiv.org/abs/1805.06440, as a Keras
        callback.
        :param layer: The Keras layer to which we apply regularization learning.
        :param norm: Norm of the regularization. Currently supports only l1 and l2 norms. Best results were obtained
        with l1 norm so far.
        :param avg_reg: The average regularization coefficient, Theta in the paper.
        :param learning_rate: The learning rate of the regularization coefficients, nu in the paper. Note that since we
        typically have many weights in the network, and we optimize the coefficients in the log scale, optimal learning
        rates tend to be large, with best results between 10^4-10^6.
        """
        super(RLNCallback, self).__init__()
        self._layer = layer

        self._prev_weights: Optional[torch.Tensor] = None
        self._weights: Optional[torch.Tensor] = None
        self._prev_regularization: Optional[torch.Tensor] = None
        self._avg_reg = avg_reg
        self._shape = torch.t(self._layer.weight).shape
        self._lambdas = DataFrame(np.ones(self._shape) * self._avg_reg)
        self._lr = learning_rate
        assert norm in [1, 2], "Only supporting l1 and l2 norms at the moment"
        self.norm = norm

    def on_train_begin(self, logs: Any = None) -> None:
        self._update_values()

    def on_batch_end(self, logs: Any = None) -> None:
        self._prev_weights = self._weights
        self._update_values()
        gradients = self._weights - self._prev_weights

        # Calculate the derivatives of the norms of the weights
        if self.norm == 1:
            norms_derivative = np.sign(self._weights)
        else:
            norms_derivative = self._weights * 2

        if self._prev_regularization is not None:
            # This is not the first batch, and we need to update the lambdas
            lambda_gradients = gradients.multiply(self._prev_regularization)
            self._lambdas -= self._lr * lambda_gradients

            # Project the lambdas onto the simplex \sum(lambdas) = Theta
            translation = self._avg_reg - self._lambdas.mean().mean()
            self._lambdas += translation

        # Clip extremely large lambda values to prevent overflow
        max_lambda_values = np.log(
            np.abs(self._weights / norms_derivative)
        ).fillna(np.inf)
        self._lambdas = self._lambdas.clip(upper=max_lambda_values)

        # Update the weights
        regularization = norms_derivative.multiply(np.exp(self._lambdas))
        self._weights -= regularization

        with torch.no_grad():
            self._layer.weight = nn.Parameter(
                torch.t(torch.Tensor(self._weights.values))
            )

        self._prev_regularization = regularization

    def _update_values(self) -> None:
        self._weights = DataFrame(torch.t(self._layer.weight.cpu().detach()))


class TORCHRLNDG(TORCHRLN):
    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 10, 100),
            "n_layers": trial.suggest_int("n_layers", 2, 5),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.0005, 0.001
            ),
            "norm": trial.suggest_categorical("norm", [1, 2]),
            "theta": trial.suggest_int("theta", -12, -8),
            "dg_num_groups": trial.suggest_int("dg_num_groups", 2, 10),
            "dg_group_method": trial.suggest_categorical(
                "dg_group_method", ["random", "time"]
            ),
        }
        return params

    @staticmethod
    def get_default_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            "hidden_dim": 10,
            "n_layers": 2,
            "learning_rate": 0.001,
            "norm": 1,
            "theta": -7.5,
            "dg_num_groups": 4,
            "dg_group_method": "random",
        }

        return params

    @staticmethod
    def get_name() -> str:
        return "torchrln_dg"


models: List[Tuple[str, Type[Model]]] = [
    ("torchrln", TORCHRLN),
    ("torchrln_dg", TORCHRLNDG),
]
