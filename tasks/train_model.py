import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import List, Optional

import optuna
import torch
from optuna.trial import TrialState

from tabularbench.dataloaders import get_custom_dataloader
from tabularbench.datasets.dataset_factory import load_dataset
from tabularbench.metrics.compute import compute_metric
from tabularbench.metrics.metric_factory import create_metric
from tabularbench.models.model_factory import load_model
from tabularbench.models.tab_scaler import TabScaler

# Torch config to avoid crash on HPC
torch.multiprocessing.set_sharing_strategy("file_system")


@dataclass
class TrainParams:
    subset: int = 0
    train_batch_size: int = 1024
    val_batch_size: int = 2048
    epochs: int = 0
    verbose: int = 0
    device: str = "cpu"
    seed: int = 42


def train_save_model(
    dataset_name: str = "lcld_v2_iid",
    model_name: str = "tabtransformer",
    dataloader: str = None,
    train_params: Optional[TrainParams] = None,
) -> None:

    if dataloader is None:
        dataloader = "default"

    if train_params is None:
        train_params = TrainParams()
    print(
        f"Training model {model_name} "
        f"with {dataloader} "
        f"traininng method on dataset {dataset_name}."
    )

    print(
        "Train hyperparameter optimization for {} on {}".format(
            model_name, dataset_name
        )
    )

    dataset = load_dataset(dataset_name)
    metadata = dataset.get_metadata(only_x=True)
    common_model_params = {
        "x_metadata": metadata,
        "objective": "classification",
        "use_gpu": True,
        "batch_size": train_params.train_batch_size,
        "num_classes": 2,
        "early_stopping_rounds": 5,
        "val_batch_size": train_params.val_batch_size,
        "class_weight": "balanced",
        "custom_dataloader": dataloader,
        "epochs": train_params.epochs,
        "dataset": dataset_name,
    }

    x, y = dataset.get_x_y()
    splits = dataset.get_splits()

    x_train = x.iloc[splits["train"]].values
    y_train = y[splits["train"]]

    if train_params.subset > 0:
        x_train = x_train[: train_params.subset]
        y_train = y_train[: train_params.subset]

    model_class = load_model(model_name)
    args = {
        **common_model_params,
        "model_name": model_name,
        "dataset": dataset_name,
        "early_stopping_rounds": train_params.epochs,
        "num_splits": 5,
        "seed": train_params.seed,
        "shuffle": True,
        "metrics": ["auc"],
    }
    if model_name == "torchrln":
        args["weight_decay"] = 0

    study_name = f"{model_name}_{dataset_name}"
    storage_name = (
        f"sqlite:///data/model_parameters/{model_name}/{study_name}.db"
    )
    print(f"Using DB file: {storage_name}")

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )

    n_completed = len(study.get_trials(states=(TrialState.COMPLETE,)))
    if n_completed < 1:
        print("No parameters")
        raise Exception

    scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
    scaler.fit(
        torch.tensor(x.values, dtype=torch.float32), x_type=metadata["type"]
    )

    best_args = {**args}

    best_trial = sorted(
        study.trials, key=lambda d: d.value if d.values is not None else -1
    )[-1]
    print(
        f"Best trial parameters: {best_trial.params} with best performance: {best_trial.value}"
    )
    best_args = {**best_args, **study.best_trial.params}

    best_args["x_metadata"] = None

    best_args = {
        **best_args,
        **best_trial.params,
        "custom_dataloader": dataloader,
    }

    model = model_class(
        **{
            **args,
            **best_trial.params,
            "early_stopping_rounds": args["epochs"],
            "force_device": train_params.device,
        },
        scaler=scaler,
    )

    x_test = x.iloc[splits["test"]]
    y_test = y[splits["test"]]

    custom_train_dataloader = get_custom_dataloader(
        dataloader,
        dataset,
        model,
        scaler,
        {},
        verbose=train_params.verbose,
        x=x_train,
        y=y_train,
        train=True,
        batch_size=train_params.train_batch_size,
    )
    model.fit(
        x_train,
        y_train,
        x_test,
        y_test,
        custom_train_dataloader=custom_train_dataloader,
    )

    sc = create_metric("auc")
    metric_val = compute_metric(model, sc, x_test, y_test)
    print(metric_val)

    save_path = os.path.join(
        ".",
        "data",
        "models",
        dataset_name,
        "{}_{}_{}.model".format(model_name, dataset_name, dataloader),
    )
    model.save(save_path)


if __name__ == "__main__":
    train_params = TrainParams(
        subset=2000,
        train_batch_size=1000,
        val_batch_size=1000,
        verbose=0,
        epochs=2,
        device="cpu",
    )

    # for dataset in ["ctu_13_neris", "url", "lcld_v2_iid", "malware", "wids"]:
    for dataset in ["lcld_v2_iid"]:

        # for models in ["tabtransformer", "torchrln", "stg", "tabnet", "vime"]:
        for models in ["vime"]:

            # for dataloader in ["default","cutmix_madry", "madry", "cutmix", ]:
            for dataloader in ["madry"]:

                train_save_model(
                    dataset_name=dataset,
                    model_name=models,
                    dataloader=dataloader,
                    train_params=train_params,
                )


if False:
    parser = ArgumentParser(
        description="Training with Hyper-parameter optimization"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        type=str,
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--val_batch_size", type=int, default=2048)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--custom_dataloader",
        type=str,
        default="default",
    )

    args = parser.parse_args()

    train_params = TrainParams(
        subset=args.subset,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        verbose=args.verbose,
        epochs=args.epochs,
        device=args.device,
    )

    train_save_model(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        custom_dataloader=args.custom_dataloader,
        train_params=train_params,
    )
