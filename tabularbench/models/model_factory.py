import importlib
from typing import Any, Dict, List, Tuple, Type, Union

from tabularbench.models.aliases import ALIASES
from tabularbench.models.model import Model
from tabularbench.models.tabsurvey import models as tabsurvey_models

models: List[Tuple[str, Type[Model]]] = tabsurvey_models


def get_model_from_config(config: Dict[str, Any]) -> Type[Model]:
    name = config["name"]

    # If it is a known dataset
    if len(config.keys()) == 1:
        return load_model(name)

    # Else load and do
    models_l = importlib.import_module(config["source"]).models
    models_out = list(filter(lambda e: e[0] in [name], models_l))
    models_out = [e[1] for e in models_out]

    if len(models_out) != 1:
        raise NotImplementedError("At least one model is not available.")
    return models_out[0]


def get_model(config: Union[Dict[str, Any], str]) -> Type[Model]:
    if isinstance(config, str):
        config = {"name": config}
    return get_model_from_config(config)


def load_model(model_name: str) -> Type[Model]:
    return load_models(model_name)[0]


def get_from_alias_model_names(names: List[str]) -> List[str]:
    return [ALIASES.get(e, e) for e in names]


def load_models(model_names: Union[str, List[str]]) -> List[Type[Model]]:

    if isinstance(model_names, str):
        model_names = [model_names]

    model_names = get_from_alias_model_names(model_names)
    models_out = list(filter(lambda e: e[0] in model_names, models))
    models_out = [e[1] for e in models_out]

    if len(models_out) != len(model_names):
        raise NotImplementedError(
            f"At least one model of {model_names} is not available."
        )

    return models_out
