import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from tabularbench.dataloaders.fast_dataloader import FastTensorDataLoader
from tabularbench.models.model import Model
from tabularbench.models.tab_scaler import ScalerData, TabScaler
from tabularbench.models.torch_models import BaseModelTorch
from tabularbench.utils.io import parent_exists
from tabularbench.utils.load_do_save import load_json, save_json

"""
    VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain
    (https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html)

    Custom implementation using PyTorch.
    See the original implementation using
    Tensorflow: https://github.com/jsyoon0823/VIME
"""
logger = logging.getLogger(__name__)


class VIME(BaseModelTorch):
    def __init__(
        self,
        objective: str,
        x_metadata: pd.DataFrame,
        batch_size: int,
        epochs: int,
        early_stopping_rounds: int,
        num_classes: int,
        p_m: float = 0.3,
        K: int = 3,
        alpha: float = 2,
        beta: float = 1,
        val_batch_size: int = 100000,
        device="cpu",
        name="saint",
        scaler: Optional[TabScaler] = None,
        **kwargs: Any,
    ) -> None:

        self.objective = objective
        self.x_metadata = x_metadata
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.num_classes = num_classes
        self.device = device

        self.p_m = p_m
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.val_batch_size = val_batch_size

        self.scaler = TabScaler(num_scaler="min_max", one_hot_encode=False)
        if scaler is not None:
            self.scaler.fit_scaler_data(scaler.get_scaler_data())
        else:
            self.scaler.fit_metadata(x_metadata)

        # Generate super call
        super().__init__(
            objective=objective,
            x_metadata=x_metadata,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_rounds=early_stopping_rounds,
            num_classes=num_classes,
            p_m=p_m,
            K=K,
            alpha=alpha,
            beta=beta,
            val_batch_size=val_batch_size,
            name=name,
            scaler=self.scaler,
            **kwargs,
        )

        logger.info(f"On Device: {self.device}")

        self.cat_idx = np.where(x_metadata["type"] == "cat")[0].tolist()
        self.num_features = x_metadata.shape[0]
        self.cat_idx = np.where(x_metadata["type"] == "cat")[0].tolist()
        self.num_idx = list(set(range(self.num_features)) - set(self.cat_idx))
        self.cat_dims = [
            (int(x_metadata.iloc[i]["max"]) + 1) for i in self.cat_idx
        ]

        self.model_self = VIMESelf(self.num_features).to(self.device)
        self.model_semi = VIMESemi(
            self.objective, self.num_features, self.num_classes
        ).to(self.device)

        self.encoder_layer = self.model_self.input_layer

        self.model = nn.Sequential(self.encoder_layer, self.model_semi)
        if hasattr(self, "scaler") and (self.scaler is not None):
            self.model = nn.Sequential(
                self.scaler.get_transorm_nn(), self.model
            )

        self.wrapper_model = self.model

        self.experiment = None

    def parameters(self):
        return self.model.parameters()

    @property
    def training(self):
        return self.model.training

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        custom_train_dataloader=None,
        custom_val_dataloader=None,
    ):

        X = np.array(X, dtype=np.float32)
        if X_val is not None:
            X_val = np.array(X_val, dtype=np.float32)

        X = self.scaler.transform(np.array(X, dtype=np.float32))
        if X_val is None:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=True, stratify=y
            )

        X_val = self.scaler.transform(np.array(X_val, dtype=np.float32))

        X_unlab = np.concatenate([X, X_val], axis=0)

        self.fit_self(X_unlab, p_m=self.p_m, alpha=self.alpha)

        # if self.args.data_parallel:
        #     self.encoder_layer = self.model_self.module.input_layer
        # else:
        # self.encoder_layer = self.model_self.input_layer
        if self.device == "cuda":
            torch.cuda.synchronize()

        loss_history, val_loss_history = self.fit_semi(
            X,
            y,
            X,
            X_val,
            y_val,
            p_m=self.p_m,
            K=self.K,
            beta=self.beta,
            custom_train_dataloader=custom_train_dataloader,
            custom_val_dataloader=custom_val_dataloader,
            scaler=self.scaler,
        )
        self.load_best_weights()
        return loss_history, val_loss_history

    def get_logits(self, x: torch.Tensor, with_grad: bool) -> torch.Tensor:
        if not with_grad:
            with torch.no_grad():
                # with_grad = True to avoid recursion
                return self.get_logits(x, with_grad=True)

        preds = self.model(x)
        if self.objective == "binary":
            preds = torch.sigmoid(preds)
        return preds

    def predict_helper(self, x, keep_grad=False):
        self.model.eval()

        x = np.array(x, dtype=float)
        x = torch.tensor(x).float().to(self.device)

        # test_dataset = TensorDataset(x)
        # test_loader = DataLoader(
        #     dataset=test_dataset,
        #     batch_size=self.val_batch_size,
        #     shuffle=False,
        #     num_workers=2,
        # )

        # logger.info(x.dtype)

        test_loader = FastTensorDataLoader(
            x, batch_size=self.val_batch_size, shuffle=False
        )

        predictions = []
        for batch_x in test_loader:
            preds = self.get_logits(batch_x[0], with_grad=keep_grad)
            predictions.append(preds.cpu())

        return np.concatenate(predictions)

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ):
        params = {
            "p_m": trial.suggest_float("p_m", 0.1, 0.9),
            "alpha": trial.suggest_float("alpha", 0.1, 10),
            "K": trial.suggest_categorical("K", [2, 3, 5, 10, 15, 20]),
            "beta": trial.suggest_float("beta", 0.1, 10),
        }
        return params

    @staticmethod
    def get_default_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            "p_m": 0.3,
            "K": 3,
            "alpha": 2,
            "beta": 1,
        }
        return params

    def get_non_tunable_params(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_rounds": self.early_stopping_rounds,
            "num_classes": self.num_classes,
            "class_weight": self.class_weight,
        }

    def fit_self(self, X, p_m=0.3, alpha=2):
        optimizer = optim.RMSprop(self.model_self.parameters(), lr=0.001)
        loss_func_mask = nn.BCELoss()
        loss_func_feat = nn.MSELoss()

        # X.to(self.device)
        # X = torch.tensor(X).float()
        # print(X.device)

        m_unlab = mask_generator(p_m, X)
        m_label, x_tilde = pretext_generator(m_unlab, X)
        X = torch.tensor(X).float()

        x_tilde = torch.tensor(x_tilde).float()
        m_label = torch.tensor(m_label).float()

        # If the dataset is relatively small, we can load it to the GPU
        if X.shape[1] < 10000:
            x_tilde = x_tilde.to(self.device)
            m_label = m_label.to(self.device)
            X = X.to(self.device)

        train_loader = FastTensorDataLoader(
            x_tilde,
            m_label,
            X,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(10):
            logger.debug(f"Self epoch{epoch}")
            for i, (batch_X, batch_mask, batch_feat) in enumerate(
                train_loader
            ):
                logger.debug(f"Self supervised batch {i}")
                out_mask, out_feat = self.model_self(batch_X.to(self.device))

                loss_mask = loss_func_mask(
                    out_mask, batch_mask.to(self.device)
                )
                loss_feat = loss_func_feat(
                    out_feat, batch_feat.to(self.device)
                )

                loss = loss_mask + loss_feat * alpha

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        logger.info("Fitted encoder")

    def fit_semi(
        self,
        X,
        y,
        x_unlab,
        X_val=None,
        y_val=None,
        p_m=0.3,
        K=3,
        beta=1,
        custom_train_dataloader=None,
        custom_val_dataloader=None,
        scaler=None,
    ):
        X = torch.tensor(X).float()
        y = torch.tensor(y)
        x_unlab = torch.tensor(x_unlab).float()

        X_val = torch.tensor(X_val).float()
        y_val = torch.tensor(y_val)

        self.set_loss_y(y)

        y = y.float()
        y_val = y_val.float()

        if (y.ndim == 1) or (y.shape[1] == 1):
            y = torch.column_stack((1 - y, y))
        if (y_val.ndim == 1) or (y_val.shape[1] == 1):
            y_val = torch.column_stack((1 - y_val, y_val))

        loss_func_supervised = self.loss_func

        optimizer = optim.AdamW(self.model_semi.parameters())

        if custom_train_dataloader:
            tensors_x, tensors_y, tensors_x_unlab = (
                X.to(self.device),
                y.to(self.device),
                x_unlab.to(self.device),
            )
            if hasattr(custom_train_dataloader, "dataset"):
                custom_train_dataloader.dataset.tensors = (
                    tensors_x,
                    tensors_y,
                    tensors_x_unlab,
                )
            else:
                custom_train_dataloader.set_tensors(
                    tensors_x, tensors_y, tensors_x_unlab
                )

            train_loader = custom_train_dataloader
            train_loader.scaler = scaler
            train_loader.transform_y = True
        else:
            if X.shape[1] < 10000:
                X = X.to(self.device)
                y = y.to(self.device)
                x_unlab = x_unlab.to(self.device)
            train_loader = FastTensorDataLoader(
                X,
                y,
                x_unlab,
                batch_size=self.batch_size,
                shuffle=True,
            )

        if custom_val_dataloader:
            custom_val_dataloader.dataset.tensors = (
                X_val.to(self.device),
                y_val.to(self.device),
            )
            val_loader = custom_val_dataloader
            val_loader.scaler = scaler
        else:
            if X.shape[1] < 10000:
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
            val_loader = FastTensorDataLoader(
                X_val,
                y_val,
                batch_size=self.batch_size,
                shuffle=False,
            )

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        n_batch = int(np.ceil(X.shape[0] / self.batch_size))
        # Even if K=20 cutting in 5 shall fit in the GPU memory
        n_batch_per_precompute = int(np.ceil(n_batch / min(K, 5)))
        # print("--------")
        # print(X.shape)
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}", flush=True)

            for i, (batch_X, batch_y) in enumerate(train_loader):
                # print(f"Batch {i}.", flush=True)
                i_relative_k = i % n_batch_per_precompute
                if i_relative_k == 0:
                    logger.info(f"Precompute {i}")
                    x_for_k = x_unlab[
                        i
                        * self.batch_size : (i + n_batch_per_precompute)
                        * self.batch_size
                    ]
                    mask = torch_mask_generator(p_m, x_for_k.repeat(K, 1))
                    _, unlab_tmp = prepare_pretext_generator(
                        mask, x_for_k, self.batch_size, K
                    )
                    unlab_tmp = unlab_tmp.reshape(K, x_for_k.shape[0], -1)
                    # print(unlab_tmp.shape)

                batch_X_encoded = self.encoder_layer(batch_X.to(self.device))
                y_hat = self.model_semi(batch_X_encoded)

                yv_hats = torch.empty(K, batch_X.shape[0], self.num_classes)
                for rep in range(K):
                    batch_unlab_tmp = unlab_tmp[rep][
                        (i_relative_k)
                        * self.batch_size : (i_relative_k + 1)
                        * self.batch_size
                    ]

                    if batch_X.shape[0] > batch_unlab_tmp.shape[0]:
                        repeat = int(
                            np.ceil(
                                batch_X.shape[0] / batch_unlab_tmp.shape[0]
                            )
                        )
                        batch_unlab_tmp = batch_unlab_tmp.repeat(repeat, 1)
                        batch_unlab_tmp = batch_unlab_tmp[: batch_X.shape[0]]
                    batch_unlab_encoded = self.encoder_layer(
                        batch_unlab_tmp.to(self.device)
                    )
                    yv_hat = self.model_semi(batch_unlab_encoded)
                    yv_hats[rep] = yv_hat

                if (
                    self.objective == "regression"
                    or self.objective == "binary"
                ):
                    y_hat = y_hat.squeeze()

                y_loss = loss_func_supervised(y_hat, batch_y.to(self.device))
                yu_loss = torch.mean(torch.var(yv_hats, dim=0))
                loss = y_loss + beta * yu_loss
                loss_history.append(loss.item())
                if self.experiment is not None:
                    self.experiment.log_metric("train_loss", loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Early Stopping
            val_loss = 0.0
            val_dim = 0
            with torch.no_grad():
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                    batch_val_X_encoded = self.encoder_layer(
                        batch_val_X.to(self.device)
                    )
                    y_hat = self.model_semi(batch_val_X_encoded)

                    if (
                        self.objective == "regression"
                        or self.objective == "binary"
                    ):
                        y_hat = y_hat.squeeze()

                    val_loss += loss_func_supervised(
                        y_hat, batch_val_y.to(self.device)
                    ).item()
                    val_dim += 1

            val_loss /= val_dim
            val_loss_history.append(val_loss)

            if self.experiment is not None:
                self.experiment.log_metric("validation_loss", val_loss)

            logger.info("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                self.save_best_weights()

                # self.save_model(filename_extension="best", directory="tmp")

            if min_val_loss_idx + self.early_stopping_rounds < epoch:
                logger.info("Early stopping applies.")
                break

        return loss_history, val_loss_history

    def save(self, path: str) -> None:
        kwargs_save = deepcopy(self.constructor_kwargs)
        kwargs_save.pop("x_metadata")

        if "scaler" in kwargs_save:
            kwargs_save.pop("scaler")
        args_path = f"{path}/args.json"
        save_json(kwargs_save, parent_exists(args_path))

        if self.scaler is not None:
            scaler_path = f"{path}/scaler.json"
            self.scaler.get_scaler_data().save(parent_exists(scaler_path))

        weigths_path = f"{path}/weights.pt"
        torch.save(self.model_self.state_dict(), parent_exists(weigths_path))

        semi_weights_path = f"{path}/semi_weights.pt"
        torch.save(
            self.model_semi.state_dict(), parent_exists(semi_weights_path)
        )

    @classmethod
    def load_class(cls, path: str, **kwargs: Dict[str, Any]) -> BaseModelTorch:

        args_path = f"{path}/args.json"
        weigths_path = f"{path}/weights.pt"
        semi_weights_path = f"{path}/semi_weights.pt"
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
        model.model_self.load_state_dict(
            torch.load(weigths_path, map_location=torch.device("cpu"))
        )
        model.model_semi.load_state_dict(
            torch.load(semi_weights_path, map_location=torch.device("cpu"))
        )

        return model

    def save_best_weights(self) -> None:
        for path, model in zip(
            [f"./tmp/{self.uuid}_best.pt", f"./tmp/{self.uuid}_semi_best.pt"],
            [self.model_self, self.model_semi],
        ):
            torch.save(model.state_dict(), parent_exists(path))

    def load_best_weights(self) -> None:
        for path, model in zip(
            [f"./tmp/{self.uuid}_best.pt", f"./tmp/{self.uuid}_semi_best.pt"],
            [self.model_self, self.model_semi],
        ):
            model.load_state_dict(torch.load(path))

    def get_model_size(self):
        self_size = sum(
            t.numel() for t in self.model_self.parameters() if t.requires_grad
        )
        semi_size = sum(
            t.numel() for t in self.model_semi.parameters() if t.requires_grad
        )
        return self_size + semi_size

    @staticmethod
    def get_name() -> str:
        return "vime"


class VIMESelf(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, input_dim)

        self.mask_layer = nn.Linear(input_dim, input_dim)
        self.feat_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        out_mask = torch.sigmoid(self.mask_layer(x))
        out_feat = torch.sigmoid(self.feat_layer(x))

        return out_mask, out_feat


class VIMESemi(nn.Module):
    def __init__(
        self, objective, input_dim, output_dim, hidden_dim=100, n_layers=5
    ):
        super().__init__()
        self.objective = objective

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        self.layers.extend(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)]
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        for layer in self.layers:
            x = F.relu(layer(x))

        out = self.output_layer(x)

        if self.objective == "classification":
            out = F.softmax(out, dim=1)

        return out


"""
    VIME code copied from: https://github.com/jsyoon0823/VIME
"""


def mask_generator(p_m, x):
    mask = np.random.binomial(1, p_m, x.shape)
    return mask


def torch_mask_generator(p_m, x):
    return torch.bernoulli(torch.full(x.shape, p_m).to(x.device))


def pretext_generator(m, x):
    # Parameters
    no, dim = x.shape
    # Convert the input numpy array to a PyTorch tensor
    # x = torch.tensor(x)

    # Randomly (and column-wise) shuffle data
    # Randomly (and column-wise) shuffle data

    x_bar = x.copy()

    np.apply_along_axis(np.random.shuffle, 0, x_bar)
    # for i in range(arr.shape[1]):
    #     np.random.shuffle(arr[:, i])
    # x_bar = arr

    # x_bar = np.zeros([no, dim])
    # for i in range(dim):
    #     idx = np.random.permutation(no)
    #     x_bar[:, i] = x[idx, i]

    # Corrupt samples
    # x_bar = torch.tensor(x_bar).float()
    x_tilde = x * (1 - m) + x_bar * m
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde


def prepare_permut(x):
    x_bar = x.copy()
    np.apply_along_axis(np.random.shuffle, 0, x_bar)
    return x_bar


def prepare_pretext_generator(m, x, batch_size, K):

    n_batch = np.ceil(x.shape[0] / batch_size)
    x_batch = np.array_split(x.cpu().numpy(), n_batch) * K

    x_bar = np.concatenate(
        Parallel(n_jobs=max(10, joblib.cpu_count()))(
            delayed(prepare_permut)(e) for e in x_batch
        )
    )

    x_bar = torch.tensor(x_bar).float().to(x.device)

    x = x.repeat(K, 1)
    x_tilde = x * (1 - m) + x_bar * m
    m_new = (1 * (x != x_tilde)).float()

    return m_new, x_tilde


models: List[Tuple[str, Type[Model]]] = [("vime", VIME)]
