from abc import abstractmethod
from typing import Tuple

import torch
from torch.utils.data import DataLoader


class BaseDataLoader:
    @abstractmethod
    def get_dataloader(
        self, x: torch.Tensor, y: torch.Tensor, train: bool, batch_size: int
    ) -> DataLoader[Tuple[torch.Tensor, ...]]:
        raise NotImplementedError
