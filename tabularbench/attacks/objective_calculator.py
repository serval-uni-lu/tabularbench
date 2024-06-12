from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import joblib
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from tabularbench.attacks.utils import compute_distance
from tabularbench.constraints.constraints import Constraints
from tabularbench.constraints.constraints_checker import ConstraintChecker
from tabularbench.utils.typing import NDBool, NDInt, NDNumber

np.set_printoptions(threshold=sys.maxsize)


@dataclass
class ObjectiveMeasure:
    misclassification: NDNumber
    distance: NDNumber
    constraints: NDNumber

    def __getitem__(self, key: int) -> ObjectiveMeasure:
        return ObjectiveMeasure(*[e[key] for e in self.__dict__.values()])


@dataclass
class ObjectiveRespected:
    misclassification: NDBool
    distance: NDBool
    constraints: NDBool
    m_and_d: NDBool
    m_and_c: NDBool
    d_and_c: NDBool
    mdc: NDBool

    def __getitem__(self, key: int) -> ObjectiveRespected:
        return ObjectiveRespected(*[e[key] for e in self.__dict__.values()])


class ObjectiveCalculator:
    def __init__(
        self,
        classifier: Callable[[NDNumber], NDNumber],
        constraints: Constraints,
        thresholds: Dict[str, float],
        norm: str = "inf",
        fun_distance_preprocess: Callable[[NDNumber], NDNumber] = lambda x: x,
    ) -> None:
        """Calculate the objectives satisfaction according to a model
        and a set of constraints.
        This version is using cache, therefore you should pass the
        parameters recompute=False whenever your input are similar to
        previous call to avoid unnecessary computation.


        Parameters
        ----------
        classifier : _type_
            The tager classifier.
        constraints : Constraints
            The set of constraints.
        thresholds : dict
            Dictionary containing a float value for the
            "misclassfication" and  "distance" key.
        norm : _type_, optional
            Norm to compute the distance, by default np.inf.
        fun_distance_preprocess : _type_, optional
            function used to preprocess input before the distance metric
            calculation, typically the n-1 first steps of an n step
            classification Pipeline, by default lambdax:x.
        """
        self.classifier = classifier
        self.constraints = constraints
        self.norm = norm
        self.fun_distance_preprocess = fun_distance_preprocess

        self.thresholds = thresholds.copy()

        # if isinstance(self.thresholds["misclassification"], float):
        #     self.thresholds["misclassification"] = np.array(
        #         [
        #             1 - self.thresholds["misclassification"],
        #             self.thresholds["misclassification"],
        #         ]
        #     )

        if thresholds.get("misclassification") is not None:
            raise NotImplementedError(
                "misclassification threshold is not yet implemented in this version."
            )

        if "constraints" not in self.thresholds:
            self.thresholds["constraints"] = 0.0
        self.objectives_eval: Optional[ObjectiveMeasure] = None
        self.objectives_respected: Optional[ObjectiveRespected] = None

    def set_cache_objectives_eval(
        self, objectives_eval: ObjectiveMeasure
    ) -> None:
        self.objectives_eval = objectives_eval

    def compute_objectives_eval(
        self, x_clean: NDNumber, y_clean: NDInt, x_adv: NDNumber
    ) -> ObjectiveMeasure:
        constraints_checker = ConstraintChecker(
            self.constraints, self.thresholds["constraints"]
        )

        def parallel_fun(local_x_clean, local_x_adv):
            out = 1 - constraints_checker.check_constraints(
                local_x_clean[np.newaxis, :], local_x_adv
            )
            return out

        constraint_violation = np.array(
            Parallel(n_jobs=-1)(
                delayed(parallel_fun)(x_clean[i], x_adv[i])
                for i in range(len(x_clean))
            )
        )
        # constraint_violation = np.array(
        #     [
        #         1
        #         - constraints_checker.check_constraints(
        #             x_clean[i][np.newaxis, :], x_adv[i]
        #         )
        #         for i in tqdm(range(len(x_clean)))
        #     ]
        # )

        # Misclassification
        y_clean = np.repeat(y_clean, x_adv.shape[1], axis=0)
        torch.set_num_threads(joblib.cpu_count())

        N_SPLIT = (
            200
            if x_adv.reshape(-1, x_adv.shape[-1]).shape[0] > 200
            else x_adv.reshape(-1, x_adv.shape[-1]).shape[0]
        )

        for_classification = np.array_split(
            x_adv.reshape(-1, x_adv.shape[-1]), N_SPLIT
        )
        classification = np.concatenate(
            [self.classifier(e) for e in for_classification]
        )

        label_mask = np.zeros(classification.shape)
        label_mask[np.arange(len(y_clean)), y_clean] = 1

        correct_logit = np.max(label_mask * classification, axis=1)
        wrong_logit = np.max((1.0 - label_mask) * classification, axis=1)

        classification = correct_logit - wrong_logit

        classification = classification.reshape(*x_adv.shape[:-1])

        x_clean_distance = self.fun_distance_preprocess(x_clean)
        x_adv_shape = x_adv.shape

        for_distance = np.array_split(
            x_adv.reshape(-1, x_adv.shape[-1]), N_SPLIT
        )

        x_adv_distance = np.concatenate(
            [self.fun_distance_preprocess(e) for e in for_distance]
        )

        x_adv_distance = x_adv_distance.reshape((*x_adv_shape[:-1], -1))
        distance = np.array(
            [
                compute_distance(
                    x_clean_distance[i][np.newaxis, :],
                    x_adv_distance[i],
                    self.norm,
                )
                for i in range(len(x_clean))
            ]
        )

        return ObjectiveMeasure(classification, distance, constraint_violation)

    def get_objectives_eval(
        self,
        x_clean: NDNumber,
        y_clean: NDInt,
        x_adv: NDNumber,
        recompute: bool = True,
    ) -> ObjectiveMeasure:
        if (self.objectives_eval is None) or recompute:
            self.objectives_eval = self.compute_objectives_eval(
                x_clean, y_clean, x_adv
            )
        return self.objectives_eval

    def compute_objectives_respected(
        self, objectives_eval: ObjectiveMeasure, y_clean: NDInt
    ) -> ObjectiveRespected:

        constraints_respected = objectives_eval.constraints <= 0

        misclassified = objectives_eval.misclassification <= 0

        distance = objectives_eval.distance <= self.thresholds["distance"]

        return ObjectiveRespected(
            misclassification=misclassified,
            distance=distance,
            constraints=constraints_respected,
            m_and_d=misclassified * distance,
            m_and_c=misclassified * constraints_respected,
            d_and_c=distance * constraints_respected,
            mdc=misclassified * distance * constraints_respected,
        )

    def get_objectives_respected(
        self,
        x_clean: NDNumber,
        y_clean: NDInt,
        x_adv: NDNumber,
        recompute: bool = True,
    ) -> ObjectiveRespected:
        self.objectives_respected = None
        if self.objectives_respected is None or recompute:
            objectives_eval = self.get_objectives_eval(
                x_clean, y_clean, x_adv, recompute
            )
            self.objectives_respected = self.compute_objectives_respected(
                objectives_eval, y_clean
            )
        return self.objectives_respected

    def get_objectives_respected_clean_mask(
        self,
        x_clean: NDNumber,
        y_clean: NDInt,
        x_adv: NDNumber,
        recompute: bool = True,
    ) -> ObjectiveRespected:
        objectives_respected = self.get_objectives_respected(
            x_clean, y_clean, x_adv, recompute
        )
        at_least_one = ObjectiveRespected(
            misclassification=np.max(
                objectives_respected.misclassification, axis=1
            ),
            distance=np.max(objectives_respected.distance, axis=1),
            constraints=np.max(objectives_respected.constraints, axis=1),
            m_and_c=np.max(objectives_respected.m_and_c, axis=1),
            m_and_d=np.max(objectives_respected.m_and_d, axis=1),
            d_and_c=np.max(objectives_respected.d_and_c, axis=1),
            mdc=np.max(objectives_respected.mdc, axis=1),
        )
        return at_least_one

    def get_success_rate(
        self,
        x_clean: NDNumber,
        y_clean: NDInt,
        x_adv: NDNumber,
        recompute: bool = True,
    ) -> ObjectiveRespected:

        if x_adv.ndim == 2 and (x_clean.shape == x_adv.shape):
            x_adv = np.expand_dims(x_adv, 1)

        at_least_one = self.get_objectives_respected_clean_mask(
            x_clean, y_clean, x_adv, recompute
        )
        success_rate = [np.mean(e) for e in at_least_one.__dict__.values()]

        return ObjectiveRespected(*success_rate)

    def get_successful_attacks(
        self,
        x_clean: NDNumber,
        y_clean: NDInt,
        x_adv: NDNumber,
        preferred_metrics: str = "misclassification",
        order: str = "asc",
        max_inputs: int = -1,
        recompute: bool = True,
    ) -> NDNumber:

        indexes = self.get_successful_attacks_indexes(
            x_clean,
            y_clean,
            x_adv,
            preferred_metrics,
            order,
            max_inputs,
            recompute=recompute,
        )

        return x_adv[indexes]

    def get_unsuccessful_attacks_clean_indexes(
        self,
        x_clean: NDNumber,
        y_clean: NDInt,
        x_adv: NDNumber,
        recompute: bool = True,
    ) -> NDInt:

        index_not_ok = np.arange(x_clean.shape[0])
        index_not_ok = index_not_ok[
            ~self.get_objectives_respected_clean_mask(
                x_clean, y_clean, x_adv, recompute
            ).mdc
        ]

        return index_not_ok

    def get_successful_attacks_clean_indexes(
        self,
        x_clean: NDNumber,
        y_clean: NDInt,
        x_adv: NDNumber,
        recompute: bool = True,
    ):
        index_ok = np.arange(x_clean.shape[0])
        index_ok = index_ok[
            self.get_objectives_respected_clean_mask(
                x_clean, y_clean, x_adv, recompute
            ).mdc
        ]

        return index_ok

    def get_successful_attacks_indexes(
        self,
        x_clean: NDNumber,
        y_clean: NDInt,
        x_adv: NDNumber,
        preferred_metrics: str = "misclassification",
        order: str = "asc",
        max_inputs: int = -1,
        recompute: bool = True,
    ) -> Tuple[NDInt, ...]:
        if max_inputs == -1:
            max_inputs = x_adv.shape[1]

        objectives_measures = self.get_objectives_eval(
            x_clean, y_clean, x_adv, recompute=recompute
        )
        objectives_respected = self.get_objectives_respected(
            x_clean, y_clean, x_adv, recompute=False
        )
        objectives_mdc = objectives_respected.mdc

        metric = objectives_measures.__dict__[preferred_metrics]
        if order == "desc":
            metric = -metric

        indices = select_k_best(
            metric,
            objectives_mdc,
            max_inputs,
        )

        return indices

    def reset_objectives_respected(self) -> None:
        self.objectives_respected = None

    def reset_objectives_eval(self) -> None:
        self.objectives_eval = None


def select_k_best(
    metric: NDNumber, filter_ok: NDBool, k: int
) -> Tuple[NDInt, ...]:
    # Find the indices of valid elements based on the filter

    # for simulation
    # filter_ok = np.random.rand(*filter_ok.shape) < 0.5
    # metric = np.column_stack((metric, metric - 1, metric - 2))
    # filter_ok = np.column_stack((filter_ok, filter_ok - 1, filter_ok - 2))

    filter_best = np.zeros(metric.shape, dtype=np.bool_)

    # Replace invalid elements with infinity
    filtered_metric = np.where(filter_ok, metric, np.inf)

    # Find the indices of the k smallest values for each B dimension
    top_k_indices = np.argpartition(filtered_metric, k - 1, axis=1)[:, :k]

    first_axis = np.repeat(
        np.arange(top_k_indices.shape[0]), top_k_indices.shape[1]
    )

    filter_best[first_axis, top_k_indices.flatten()] = True

    out = np.where(filter_best * filter_ok)

    return out
