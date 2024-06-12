import pandas as pd

from tabularbench.dataloaders.aliases import ALIASES as TRAINING_ALIASES
from tabularbench.models.aliases import ALIASES as ARCH_ALIASES
from tabularbench.models.model import Model
from tabularbench.models.model_factory import load_model
from tabularbench.models.tab_scaler import TabScaler


def load_model_and_weights(
    dataset_name: str,
    model_arch: str,
    training_name: str,
    metadata: pd.DataFrame,
    scaler: TabScaler,
    device: str,
) -> Model:

    # Load model
    model_class = load_model(model_arch)
    model_arch = ARCH_ALIASES.get(model_arch, model_arch)
    training_name = TRAINING_ALIASES.get(training_name, training_name)

    weight_path = f"./data/models/{dataset_name}/{model_arch}_{dataset_name}_{training_name}.model"

    force_device = device if device != "" else None
    model = model_class.load_class(
        weight_path,
        x_metadata=metadata,
        scaler=scaler,
        force_device=force_device,
    )

    return model
