import json
import pathlib
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from joblib import Parallel
from sklearn.model_selection import KFold, StratifiedKFold

from tabularbench.logging import XP
from tabularbench.logging.setup import delayed_with_logging
from tabularbench.metrics.compute import compute_metric
from tabularbench.metrics.metric_factory import create_metric
from tabularbench.utils.typing import NDNumber


def cut_in_batch(
    arr: npt.NDArray[Any],
    n_desired_batch: int = 1,
    batch_size: Optional[int] = None,
) -> List[npt.NDArray[Any]]:
    if batch_size is None:
        n_batch = min(n_desired_batch, len(arr))
    else:
        n_batch = np.ceil(len(arr) / batch_size)
    batches_i = np.array_split(np.arange(arr.shape[0]), n_batch)

    return [arr[batch_i] for batch_i in batches_i]


def dict2obj(d: Dict[str, Any]) -> Any:
    dumped_data = json.dumps(d)
    result = json.loads(
        dumped_data, object_hook=lambda x: SimpleNamespace(**x)
    )
    return result


def parent_exists(path: str) -> str:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


class Timer:
    def __init__(self):
        self.save_times = []
        self.start_time = 0

    def start(self):
        self.start_time = time.process_time()

    def end(self):
        end_time = time.process_time()
        self.save_times.append(end_time - self.start_time)

    def get_average_time(self):
        return np.mean(self.save_times)


def cross_validation_one(args, X, y, train_index, test_index, metric, model):
    sc = create_metric(metric)
    train_timer = Timer()
    test_timer = Timer()

    experiment = XP(vars(args))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    if args.objective == "binary":
        y_train = np.column_stack([1 - y_train, y_train])
        y_test = np.column_stack([1 - y_test, y_test])

    # Create a new unfitted version of the model
    curr_model = model.clone()
    curr_model.set_experiment(experiment)

    # Train model
    train_timer.start()
    curr_model.fit(X_train, y_train, X_test, y_test)  # X_val, y_val)
    train_timer.end()

    # Test model
    test_timer.start()
    metric_val = compute_metric(curr_model, sc, X_test, y_test)
    test_timer.end()

    print(metric_val)
    experiment.log_metric(sc.metric_name, metric_val)

    sce = create_metric("precision")
    metric_val_e = compute_metric(curr_model, sce, X_test, y_test)
    experiment.log_metric("test_{}".format(sce.metric_name), metric_val_e)
    print("test_{}".format(sce.metric_name))
    print(f"Precision {metric_val_e}")

    sce = create_metric("recall")
    metric_val_e = compute_metric(curr_model, sce, X_test, y_test)
    experiment.log_metric("test_{}".format(sce.metric_name), metric_val_e)
    print(f"Recall {metric_val_e}")
    experiment.end()

    return (
        metric_val,
        train_timer.get_average_time(),
        test_timer.get_average_time(),
    )


def cross_validation(model, X, y, metric, args, n_jobs=1):

    if args.objective == "regression":
        kf = KFold(
            n_splits=args.num_splits,
            shuffle=args.shuffle,
            random_state=args.seed,
        )
    elif args.objective == "classification" or args.objective == "binary":
        kf = StratifiedKFold(
            n_splits=args.num_splits,
            shuffle=args.shuffle,
            random_state=args.seed,
        )
    else:
        raise NotImplementedError(
            "Objective" + args.objective + "is not yet implemented."
        )

    out = Parallel(n_jobs=n_jobs)(
        delayed_with_logging(cross_validation_one)(
            args, X, y, train_index, test_index, metric, model
        )
        for i, (train_index, test_index) in enumerate(kf.split(X, y))
    )
    metric_val = np.array([e[0] for e in out]).mean()
    train_timer = np.array([e[1] for e in out]).mean()
    test_timer = np.array([e[2] for e in out]).mean()

    return metric_val, (train_timer, test_timer)


def round_func_BPDA(input):
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out


def to_torch_number(
    tensor: Union[torch.Tensor, NDNumber, pd.DataFrame]
) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        return tensor

    if isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor).float()

    if isinstance(tensor, pd.DataFrame):
        return torch.from_numpy(tensor.values).float()

    if isinstance(tensor, pd.Series):
        return torch.from_numpy(tensor.values).float()

    raise NotImplementedError(f"Unsupported type: {type(tensor)}")


def to_numpy_number(tensor: Union[torch.Tensor, NDNumber]) -> NDNumber:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()

    elif isinstance(tensor, np.ndarray):
        return tensor

    raise NotImplementedError(f"Unsupported type: {type(tensor)}")


def binary_to_2dim(tensor: torch.Tensor) -> torch.Tensor:
    if (tensor.ndim == 1) or (tensor.shape[1] == 1):
        tensor = torch.column_stack((1 - tensor, tensor))

    return tensor
