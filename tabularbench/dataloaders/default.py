import numpy as np
import torch

from tabularbench.dataloaders.dataloader import BaseDataLoader
from tabularbench.dataloaders.fast_dataloader import FastTensorDataLoader


class DefaultDataLoader(BaseDataLoader):
    def __init__(
        self,
        scaler=None,
        model=None,
        seed: int = 42,
        subset: float = 1,
        move_all_to_model_device: bool = True,
        **kwargs,
    ) -> None:

        self.subset = subset
        self.seed = seed
        self.scaler = scaler
        if model is not None:
            self.device = model.device
            self.model = model.wrapper_model
        self.move_all_to_model_device = move_all_to_model_device

    def get_dataloader(
        self, x: np.ndarray, y: np.ndarray, train: bool, batch_size: int
    ) -> FastTensorDataLoader:

        x = torch.Tensor(x).float()
        y = torch.Tensor(y).long()

        if self.subset < 1:
            dataset_size = len(x)
            split = int(np.floor(self.subset * dataset_size))
            x = x[0:split]
            y = y[0:split]

        if self.move_all_to_model_device:
            x = x.to(self.device)
            y = y.to(self.device)

        # Create dataloader
        if train:

            self.dataloader = FastTensorDataLoader(
                x, y, batch_size=batch_size, shuffle=True
            )

        else:
            self.dataloader = FastTensorDataLoader(
                x, y, batch_size=batch_size, shuffle=False
            )
        self.dataloader.scaler = None
        self.dataloader.transform_y = None

        return self.dataloader
