from typing import List, Union

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import OneHotEncoder

from tabularbench.metrics.metric import Metric
from tabularbench.metrics.metrics import PredClassificationMetric
from tabularbench.models.model import Model
from tabularbench.utils.typing import NDFloat, NDNumber


def default_model_prediction(model: Model, x: NDFloat) -> NDFloat:
    if model.objective in ["regression"]:
        return model.predict(x).astype(np.float_)
    if model.objective in ["binary", "classification"]:
        return model.predict_proba(x)
    raise NotImplementedError


def compute_metric(
    model: Model,
    metric: Metric,
    x: NDFloat,
    y: NDNumber,
) -> npt.NDArray[np.generic]:
    return compute_metrics(model, metric, x, y)


def compute_metrics(
    model: Model,
    metrics: Union[Metric, List[Metric]],
    x: NDFloat,
    y: NDNumber,
) -> NDFloat:
    if isinstance(metrics, Metric):
        return compute_metrics(model, [metrics], x, y)[0]

    y_score = default_model_prediction(model, x)

    return compute_metrics_from_scores(metrics, y, y_score)


def compute_metrics_from_scores(
    metrics: Union[Metric, List[Metric]],
    y: NDNumber,
    y_score: NDFloat,
) -> NDFloat:
    if isinstance(metrics, Metric):
        return compute_metrics_from_scores([metrics], y, y_score)[0]

    y_2d = OneHotEncoder(sparse_output=False).fit_transform(y[:, None])

    y_pred = np.argmax(y_score, axis=1)
    out = []
    for metric in metrics:
        if isinstance(metric, PredClassificationMetric):
            out.append(metric.compute(y, y_pred))
        else:
            out.append(metric.compute(y_2d, y_score))

    return np.array(out)
