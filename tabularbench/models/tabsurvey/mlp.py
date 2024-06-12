from turtle import pd
from typing import Any, Dict, Optional, Union

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F

from tabularbench.models.tab_scaler import TabScaler
from tabularbench.models.torch_models import BaseModelTorch
from tabularbench.utils.typing import NDFloat, NDNumber

"""
    Custom implementation for the standard multi-layer perceptron
"""


class MLP(BaseModelTorch):
    def __init__(
        self,
        objective: str,
        n_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        scaler: Optional[TabScaler] = None,
        **kwargs: Any,
    ):

        self.objective = objective
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.scaler = scaler

        self.model = MLPModel(
            n_layers=self.n_layers,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            task=self.objective,
        )

        if self.scaler is not None:
            self.model = nn.Sequential(
                self.scaler.get_transorm_nn(), self.model
            )

        super().__init__(
            objective=objective,
            n_layers=n_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            scaler=scaler,
            **kwargs,
        )

        self.to_device()

    def fit(
        self,
        x: NDFloat,
        y: NDNumber,
        x_val: Optional[NDFloat] = None,
        y_val: Optional[NDNumber] = None,
    ):
        x = np.array(x, dtype=np.float)
        x_val = np.array(x_val, dtype=np.float)

        return super().fit(x, y, x_val, y_val)

    def predict_helper(
        self,
        x: Union[NDFloat, torch.Tensor, pd.DataFrame],
        load_all_gpu: bool = False,
    ):
        x = np.array(x, dtype=np.float)
        return super().predict_helper(x, load_all_gpu)

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ):
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 10, 100),
            "n_layers": trial.suggest_int("n_layers", 2, 5),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.0005, 0.001
            ),
        }
        return params


class MLPModel(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, task):
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

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        # Use ReLU as activation for all hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))

        # No activation function on the output
        x = self.output_layer(x)

        if self.task == "classification":
            x = F.softmax(x, dim=1)

        return x
