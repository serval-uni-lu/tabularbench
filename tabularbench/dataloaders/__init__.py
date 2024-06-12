import numpy as np

from tabularbench.dataloaders.adversarial.madry import MadryATDataLoader
from tabularbench.dataloaders.default import DefaultDataLoader
from tabularbench.dataloaders.generative.gan_file import GanFileDataLoader
from tabularbench.dataloaders.mix.cutmix import CutmixDataLoader

SYNTHETIC_ROOT_PATH = "./data/synthetic/tabgan/"


def get_custom_dataloader(
    custom_dataloader: str,
    dataset,
    model,
    scaler,
    custom_args,
    verbose: int,
    x: np.ndarray,
    y: np.ndarray,
    train: bool,
    batch_size: int,
):

    DATALOADERS = {
        "madry": (MadryATDataLoader, {"verbose": verbose}),
        "subset": (DefaultDataLoader, {"subset": 0.1}),
        "cutmix": (
            CutmixDataLoader,
            {
                "synthetic_label": "intersection",
                "attack": "",
                "ratio": 0.5,
                "augment": 12,
            },
        ),
        "cutmix_madry": (
            CutmixDataLoader,
            {
                "synthetic_label": "intersection",
                "attack": "pgd",
                "ratio": 0.5,
                "augment": 12,
            },
        ),
        "ctgan": (
            GanFileDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "gan_name": "ctgan",
                "mlc_dataset": dataset,
                "attack": "",
            },
        ),
        "ctgan_madry": (
            GanFileDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "gan_name": "ctgan",
                "mlc_dataset": dataset,
                "attack": "pgd",
            },
        ),
        "tablegan": (
            GanFileDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "gan_name": "tablegan",
                "mlc_dataset": dataset,
                "attack": "",
            },
        ),
        "tablegan_madry": (
            GanFileDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "gan_name": "tablegan",
                "mlc_dataset": dataset,
                "attack": "pgd",
            },
        ),
        "tvae": (
            GanFileDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "gan_name": "tvae",
                "mlc_dataset": dataset,
                "attack": "",
            },
        ),
        "tvae_madry": (
            GanFileDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "gan_name": "tvae",
                "mlc_dataset": dataset,
                "attack": "pgd",
            },
        ),
        "wgan": (
            GanFileDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "gan_name": "wgan",
                "mlc_dataset": dataset,
                "attack": "",
            },
        ),
        "wgan_madry": (
            GanFileDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "gan_name": "wgan",
                "mlc_dataset": dataset,
                "attack": "pgd",
            },
        ),
        "goggle": (
            GanFileDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "gan_name": "goggle",
                "mlc_dataset": dataset,
                "attack": "",
            },
        ),
        "goggle_madry": (
            GanFileDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "gan_name": "goggle",
                "mlc_dataset": dataset,
                "attack": "pgd",
            },
        ),
    }
    dataloader = None
    dataloader_call, args = DATALOADERS.get(custom_dataloader, (None, {}))
    if dataloader_call is None:
        return None

    print(f"Using model type {type(model)}.")
    args = {
        **args,
        **custom_args,
        "dataset": dataset,
        "model": model,
        "scaler": scaler,
    }
    dataloader = dataloader_call(**args).get_dataloader(
        x=x, y=y, train=train, batch_size=batch_size
    )
    return dataloader
