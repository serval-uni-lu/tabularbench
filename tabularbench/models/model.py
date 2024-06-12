from __future__ import annotations

import abc
from abc import ABC
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from torch.utils.data import DataLoader

from tabularbench.utils.typing import NDFloat, NDNumber


class Model(ABC):
    def __init__(self, **kwargs: Any) -> None:
        self.name = kwargs["name"]
        self.objective = kwargs["objective"]
        self.constructor_kwargs = kwargs
        self.model: Optional[Any] = None

    def predict(
        self, x: npt.NDArray[np.float_]
    ) -> Union[npt.NDArray[np.int_], npt.NDArray[np.float_]]:
        """
        Returns the regression value or the concrete classes of binary /
        multi-class-classification tasks.
        (Save predictions to self.predictions)

        :param x: test data
        :return: predicted values / classes of test data (Shape N x 1)
        """

        if self.constructor_kwargs["objective"] == "regression":
            predictions = self.model.predict(x)
        elif (
            self.constructor_kwargs["objective"] == "classification"
            or self.constructor_kwargs["objective"] == "binary"
        ):
            prediction_probabilities = self.predict_proba(x)
            predictions = np.argmax(prediction_probabilities, axis=1)
        else:
            raise NotImplementedError

        return predictions

    def predict_proba(
        self, x: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """
        Only implemented for binary / multi-class-classification tasks.
        Returns the probability distribution over the classes C.
        (Save probabilities to self.prediction_probabilities)

        :param x: test data
        :return: probabilities for the classes (Shape N x C)
        """

        prediction_probabilities = self.model.predict_proba(x)

        # If binary task returns only probability for the true class,
        # adapt it to return (N x 2)
        if prediction_probabilities.shape[1] == 1:
            prediction_probabilities = np.concatenate(
                (
                    1 - prediction_probabilities,
                    prediction_probabilities,
                ),
                1,
            )
        return prediction_probabilities

    def fit(
        self,
        x: NDFloat,
        y: NDNumber,
        x_val: Optional[NDFloat] = None,
        y_val: Optional[NDNumber] = None,
        num_workers: int = 2,
        custom_train_dataloader: DataLoader = None,
        custom_val_dataloader: DataLoader = None,
    ) -> None:
        self.model.fit(x, y)

    def clone(self) -> Model:
        return self.__class__(**self.constructor_kwargs)

    @abc.abstractmethod
    def load(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def save(self, path: str) -> None:
        pass
