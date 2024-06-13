import numpy as np

from tabularbench.dataloaders.adversarial.madry import MadryATDataLoader
from tabularbench.dataloaders.default import DefaultDataLoader
from tabularbench.dataloaders.generative.ctgan import CTGanDataLoader
from tabularbench.dataloaders.generative.gan_file import GanFileDataLoader
from tabularbench.dataloaders.generative.goggle import GoggleDataLoader
from tabularbench.dataloaders.generative.tablegan import TableGanDataLoader
from tabularbench.dataloaders.generative.tvae import TVAEDataLoader
from tabularbench.dataloaders.generative.wgan import WGanDataLoader
from tabularbench.dataloaders.mix.cutmix import CutmixDataLoader

SYNTHETIC_ROOT_PATH = "./data/synthetic/"


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
            CTGanDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "mlc_dataset": dataset,
                "attack": "",
            },
        ),
        "ctgan_madry": (
            CTGanDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "mlc_dataset": dataset,
                "attack": "pgd",
            },
        ),
        "tablegan": (
            TableGanDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "mlc_dataset": dataset,
                "attack": "",
            },
        ),
        "tablegan_madry": (
            TableGanDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "mlc_dataset": dataset,
                "attack": "pgd",
            },
        ),
        "tvae": (
            TVAEDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "mlc_dataset": dataset,
                "attack": "",
            },
        ),
        "tvae_madry": (
            TVAEDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "mlc_dataset": dataset,
                "attack": "pgd",
            },
        ),
        "wgan": (
            WGanDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "mlc_dataset": dataset,
                "attack": "",
            },
        ),
        "wgan_madry": (
            WGanDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "mlc_dataset": dataset,
                "attack": "pgd",
            },
        ),
        "goggle": (
            GoggleDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
                "mlc_dataset": dataset,
                "attack": "",
            },
        ),
        "goggle_madry": (
            GoggleDataLoader,
            {
                "ratio": 0.5,
                "augment": 12,
                "synthetic_root_path": SYNTHETIC_ROOT_PATH,
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
