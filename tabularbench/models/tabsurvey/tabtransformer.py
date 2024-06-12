from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from torch import einsum, nn

from tabularbench.dataloaders.fast_dataloader import FastTensorDataLoader
from tabularbench.models.model import Model
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.models.torch_models import BaseModelTorch
from tabularbench.utils.datatypes import to_torch_number
from tabularbench.utils.io import parent_exists
from tabularbench.utils.load_do_save import load_json, save_json
from tabularbench.utils.typing import NDFloat, NDNumber

"""
    TabTransformer: Tabular Data Modeling
    Using Contextual Embeddings (https://arxiv.org/abs/2012.06678)

    Code adapted from: https://github.com/lucidrains/tab-transformer-pytorch
"""


class ScalerModule(nn.Module):
    def __init__(self, scaler: TabScaler):
        super().__init__()
        self.scaler = scaler
        if scaler.num_scaler != "min_max":
            raise ValueError(
                f"Scaler {scaler.num_scaler} not supported for TabTransformer"
            )
        if not scaler.one_hot_encode:
            raise ValueError(
                f"One hot encoding {scaler.num_scaler} "
                f"not supported for TabTransformer"
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(x.shape)
        if self.scaler.do_one_hot_encode():
            x_categ = x[:, self.scaler.cat_idx].int()
        else:
            x_categ = None
        x_cont = x
        x_cont = self.scaler.transform(x_cont)
        x_cont = x_cont[:, : len(self.scaler.num_idx)]

        return (
            x_categ,
            x_cont,
        )


class TabTransformer(BaseModelTorch):
    def __init__(
        self,
        objective: str,
        x_metadata: pd.DataFrame,
        batch_size: int,
        epochs: int,
        early_stopping_rounds: int,
        learning_rate: float,
        num_classes: int,
        name: str = "tabtransformer",
        dim: int = 32,
        depth: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
        val_batch_size: int = 100000,
        class_weight: Union[str, Dict[int, str]] = None,
        force_device: str = None,
        weight_decay: int = 0,
        scaler: Optional[TabScaler] = None,
        **kwargs: Any,
    ):

        super().__init__(
            objective=objective,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_rounds=early_stopping_rounds,
            learning_rate=learning_rate,
            x_metadata=x_metadata,
            num_classes=num_classes,
            name=name,
            dim=dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
            val_batch_size=val_batch_size,
            class_weight=class_weight,
            force_device=force_device,
            weight_decay=weight_decay,
            scaler=scaler,
            **kwargs,
        )

        self.num_features = x_metadata.shape[0]
        self.cat_idx = np.where(x_metadata["type"] == "cat")[0].tolist()
        self.num_idx = list(set(range(self.num_features)) - set(self.cat_idx))
        self.cat_dims = [
            (int(x_metadata.iloc[i]["max"]) + 1) for i in self.cat_idx
        ]
        self.num_continuous = self.num_features - len(self.cat_idx)
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.weight_decay = weight_decay

        print("ANSWER")
        print(scaler.one_hot_encode)
        if scaler is not None:
            self.scaler = TabScaler(num_scaler="min_max", one_hot_encode=False)
            self.scaler.fit_scaler_data(scaler.get_scaler_data())

        self.model: nn.Module = TabTransformerModel(
            categories=self.cat_dims,  # tuple (or list?) containing the number of unique values in each category
            num_continuous=self.num_continuous,  # number of continuous values
            dim_out=self.num_classes,
            mlp_act=nn.ReLU(),  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            attn_dropout=self.dropout,
            ff_dropout=self.dropout,
            mlp_hidden_mults=(4, 2),
        )

        if scaler is not None:
            self.model = nn.Sequential(ScalerModule(scaler), self.model)
        else:
            print("No scaler! fit method may fail")

        # For compatibility
        self.experiment = None

        self.uuid = str(uuid4())
        self.wrapper_model = self.model
        self.to_device()

    def save(self, path: str) -> None:
        kwargs_save = deepcopy(self.constructor_kwargs)
        if "x_metadata" in kwargs_save:
            kwargs_save.pop("x_metadata")

        if "scaler" in kwargs_save:
            kwargs_save.pop("scaler")

        args_path = f"{path}/args.json"
        save_json(kwargs_save, parent_exists(args_path))

        weigths_path = f"{path}/weights.pt"
        if self.scaler is None:
            torch.save(self.model.state_dict(), parent_exists(weigths_path))
        else:
            torch.save(self.model[1].state_dict(), parent_exists(weigths_path))

    @classmethod
    def load_class(cls, path: str, **kwargs: Dict[str, Any]) -> BaseModelTorch:

        args_path = f"{path}/args.json"
        weigths_path = f"{path}/weights.pt"

        load_kwargs = load_json(args_path)
        args = {**load_kwargs, **kwargs}
        model = cls(**args)
        if model.scaler is None:
            model.model.load_state_dict(
                torch.load(weigths_path, map_location=torch.device("cpu"))
            )
        else:
            model.model[1].load_state_dict(
                torch.load(weigths_path, map_location=torch.device("cpu"))
            )

        return model

    def fit(
        self,
        x: NDFloat,
        y: NDNumber,
        x_val: Optional[NDFloat] = None,
        y_val: Optional[NDNumber] = None,
        num_workers: int = 2,
        custom_train_dataloader=None,
        custom_val_dataloader=None,
    ) -> None:

        learning_rate = 10**self.learning_rate
        weight_decay = 10**self.weight_decay
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

        # For some reason this has to be set explicitly to work with categorical data
        x = np.array(x, dtype=float)
        x_val = np.array(x_val, dtype=float)

        x = torch.tensor(x).float()
        x_val = torch.tensor(x_val).float()

        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        if (y.ndim == 1) or (y.shape[1] == 1):
            y = torch.column_stack((1 - y, y))
        if (y_val.ndim == 1) or (y_val.shape[1] == 1):
            y_val = torch.column_stack((1 - y_val, y_val))

        y = y.float()
        y_val = y_val.float()

        train_loader = (
            FastTensorDataLoader(
                x.to(self.device),
                y.to(self.device),
                batch_size=self.batch_size,
                shuffle=True,
            )
            if custom_train_dataloader is None
            else custom_train_dataloader
        )

        val_loader = (
            FastTensorDataLoader(
                x_val.to(self.device),
                y_val.to(self.device),
                batch_size=self.batch_size,
                shuffle=False,
            )
            if custom_val_dataloader is None
            else custom_val_dataloader
        )

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        self.set_loss_y(y)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch}.", flush=True)
            for i, (batch_x, batch_y) in enumerate(train_loader):
                out = self.model(batch_x)
                if (
                    self.objective == "regression"
                    or self.objective == "binary"
                ):
                    out = out.squeeze()

                loss = self.loss_func(out, batch_y)
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
                for val_i, (batch_val_x, batch_val_y) in enumerate(val_loader):

                    out = self.model(batch_val_x)

                    if (
                        self.objective == "regression"
                        or (
                            self.objective == "binary"
                            and self.num_classes == 1
                        )
                    ) and len(batch_val_x) > 1:
                        out = out.squeeze()

                    val_loss += self.loss_func(out, batch_val_y).item()
                    val_dim += 1
            val_loss /= val_dim
            val_loss_history.append(val_loss)
            if self.experiment is not None:
                self.experiment.log_metric("validation_loss", val_loss)

            print(
                "Epoch %d: Train Loss %.5f Val Loss %.5f"
                % (epoch, loss, val_loss)
            )

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_best_weights()

            if min_val_loss_idx + self.early_stopping_rounds < epoch:
                print(
                    "Validation loss has not improved for %d steps!"
                    % self.early_stopping_rounds
                )
                print("Early stopping applies.")
                break

        self.load_best_weights()
        return loss_history, val_loss_history

    def save_best_weights(self) -> None:
        filename = parent_exists(f"./tmp/{self.uuid}_best.pt")
        torch.save(self.model.state_dict(), filename)

    def load_best_weights(self) -> None:
        filename = f"./tmp/{self.uuid}_best.pt"
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)

    def parameters(self):
        return self.model.parameters()

    @property
    def training(self):
        return self.model.training

    def get_logits(self, x: torch.Tensor, with_grad: bool) -> torch.Tensor:
        if not with_grad:
            with torch.no_grad():
                # with_grad = True to avoid recursion
                return self.get_logits(x, with_grad=True)

        preds = self.model(x.to(self.device))
        if self.objective == "binary":
            preds = torch.sigmoid(preds)
        return preds

    def predict_helper(self, x, keep_grad=False):
        self.model.eval()

        x = np.array(x, dtype=float)
        x = torch.tensor(x).float().to(self.device)

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
    ) -> Dict[str, Any]:
        params = {
            "dim": trial.suggest_categorical(
                "dim", [32, 64, 128, 256]
            ),  # dimension, paper set at 32
            "depth": trial.suggest_categorical(
                "depth", [1, 2, 3, 6, 12]
            ),  # depth, paper recommended 6
            "heads": trial.suggest_categorical(
                "heads", [2, 4, 8]
            ),  # heads, paper recommends 8
            "weight_decay": trial.suggest_int(
                "weight_decay", -6, -1
            ),  # x = 10 ^ u
            "learning_rate": trial.suggest_int(
                "learning_rate", -6, -3
            ),  # x = 10 ^ u
            "dropout": trial.suggest_categorical(
                "dropout", [0, 0.1, 0.2, 0.3, 0.4, 0.5]
            ),
        }
        return params

    def attribute(self, x: np.ndarray, y: np.ndarray, strategy=""):
        """Generate feature attributions for the model input.
        Two strategies are supported: default ("") or "diag".
        The default strategie takes the sum
        over a column of the attention map,
        while "diag" returns only the diagonal
        (feature attention to itself)
        of the attention map.
        return array with the same shape as x.
        The number of columns is equal
        to the number of categorical values in x.
        """
        x = np.array(x, dtype=np.float)
        # Unroll and Rerun until first attention stage.

        x = torch.tensor(x).float().to(self.device)
        test_loader = FastTensorDataLoader(
            x, batch_size=self.val_batch_size, shuffle=False
        )

        attentions_list = []
        with torch.no_grad():
            for batch_x in test_loader:
                x_categ = (
                    batch_x[0][:, self.cat_idx].int() if self.cat_idx else None
                )
                # x_cont = batch_x[0][:, self.num_idx].to(self.device)
                batch_x[0][:, self.num_idx]
                if x_categ is not None:
                    x_categ += self.model.categories_offset
                    # Tranformer
                    x = self.model.transformer.embeds(x_categ)

                    # Prenorm.
                    x = self.model.transformer.layers[0][0].fn.norm(x)

                    # Attention
                    active_transformer = self.model.transformer.layers[0][
                        0
                    ].fn.fn
                    h = active_transformer.heads
                    q, k, v = active_transformer.to_qkv(x).chunk(3, dim=-1)
                    q, k, v = map(
                        lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
                        (q, k, v),
                    )
                    sim = (
                        einsum("b h i d, b h j d -> b h i j", q, k)
                        * active_transformer.scale
                    )
                    attn = sim.softmax(dim=-1)
                    if strategy == "diag":
                        print(attn.shape)
                        attentions_list.append(attn.diagonal(0, 2, 3))
                    else:
                        attentions_list.append(attn.sum(dim=1))
                else:
                    raise ValueError(
                        "Attention can only be computed for categorical values in TabTransformer."
                    )
            attentions_list = torch.cat(attentions_list).sum(dim=1)
        return attentions_list.numpy()


####################################################################################################################
#
#  TabTransformer code from
#  https://github.com/lucidrains/tab-transformer-pytorch/blob/main/tab_transformer_pytorch/tab_transformer_pytorch.py
#  adapted to work without categorical data
#
#####################################################################################################################

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v)
        )
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


# transformer


class Transformer(nn.Module):
    def __init__(
        self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout
    ):
        super().__init__()
        self.embeds = nn.Embedding(
            num_tokens, dim
        )  # (Embed the categorical features.)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim,
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=attn_dropout,
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, dropout=ff_dropout))
                        ),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        max_shapes = x.max(1)[0]
        if torch.any(max_shapes >= self.embeds.num_embeddings):
            print("error embedding", max_shapes)
            x = torch.clamp(x, 0, self.embeds.num_embeddings - 1)

        x = self.embeds(x)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x


# mlp


class MLP(nn.Module):
    def __init__(self, dims: List[int], act: Optional[nn.Module] = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                self.dim_out = dim_out
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)

        # Added for multiclass output!
        if self.dim_out > 1:
            x = torch.softmax(x, dim=1)
        return x


# main class


class TabTransformerModel(nn.Module):
    def __init__(
        self,
        *,
        categories: List[int],
        num_continuous: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int = 16,
        dim_out: int = 1,
        mlp_hidden_mults: Tuple[int, int] = (4, 2),
        mlp_act: Optional[nn.Module] = None,
        num_special_tokens: int = 2,
        continuous_mean_std: Optional[torch.Tensor] = None,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert all(
            map(lambda n: n > 0, categories)
        ), "number of each category must be positive"

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(
            torch.tensor(list(categories)), (1, 0), value=num_special_tokens
        )  # Prepend num_special_tokens.
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer("categories_offset", categories_offset)

        # continuous

        if exists(continuous_mean_std):
            assert continuous_mean_std.shape == (
                num_continuous,
                2,
            ), f"continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively"
        self.register_buffer("continuous_mean_std", continuous_mean_std)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous

        # transformer

        self.transformer = Transformer(
            num_tokens=total_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

        # mlp to logits

        input_size = (dim * self.num_categories) + num_continuous
        logit = input_size // 8

        hidden_dimensions = list(map(lambda t: logit * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(
        self, x_categ: torch.Tensor, x_cont: torch.Tensor = "undefined"
    ) -> torch.Tensor:

        # Because we are using sequential the output of the first layer is in a tuple, henced we unpack.
        # It's ugly but it works.
        if x_cont == "undefined":
            x_categ, x_cont = x_categ

        # Adaptation to work without categorical data
        if x_categ is not None:
            assert x_categ.shape[-1] == self.num_categories, (
                f"you must pass in {self.num_categories} "
                f"values for your categories input"
            )
            # x_categ += self.categories_offset
            x = self.transformer(x_categ + self.categories_offset)
            flat_categ = x.flatten(1)

        assert x_cont.shape[1] == self.num_continuous, (
            f"you must pass in {self.num_continuous} "
            f"values for your continuous input"
        )

        if exists(self.continuous_mean_std):
            mean, std = self.continuous_mean_std.unbind(dim=-1)
            x_cont = (x_cont - mean) / std

        normed_cont = self.norm(to_torch_number(x_cont))

        # Adaptation to work without categorical data
        if x_categ is not None:
            x = torch.cat((flat_categ, normed_cont), dim=-1)
        else:
            x = normed_cont

        return self.mlp(x)


models: List[Tuple[str, Type[Model]]] = [("tabtransformer", TabTransformer)]
