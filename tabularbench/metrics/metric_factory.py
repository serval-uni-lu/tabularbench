from typing import Any, Dict, Union

from tabularbench.metrics.metric import Metric
from tabularbench.metrics.metrics import (
    PredClassificationMetric,
    ProbaClassificationMetric,
    RegressionMetric,
    str2pred_classification_metric,
    str2proba_classification_metric,
    str2regression_metric,
)


def create_metric(config: Union[str, Dict[str, Any]]) -> Metric:
    if isinstance(config, str):
        config = {"name": config}
    name = config.get("name")
    if name in str2pred_classification_metric:
        return PredClassificationMetric(name)
    elif name in str2proba_classification_metric:
        return ProbaClassificationMetric(name)
    elif name in str2regression_metric:
        return RegressionMetric(name)
    else:
        raise NotImplementedError(f"Metric {name} is not yet implemented.")
