from typing import Optional, Union

import numpy as np
from pymoo.core.problem import Problem

from tabularbench.attacks.utils import compute_distance
from tabularbench.constraints.constraints import (
    Constraints,
    get_feature_min_max,
)
from tabularbench.constraints.constraints_backend_executor import (
    ConstraintsExecutor,
)
from tabularbench.constraints.numpy_backend import NumpyBackend
from tabularbench.constraints.relation_constraint import AndConstraint
from tabularbench.models.model import Model
from tabularbench.utils.datatypes import to_numpy_number
from tabularbench.utils.typing import NDNumber

NB_OBJECTIVES = 3


def get_nb_objectives() -> int:
    return NB_OBJECTIVES


class AdversarialProblem(Problem):
    def __init__(
        self,
        x_clean: NDNumber,
        y_clean: int,
        classifier: Model,
        constraints: Constraints,
        fun_distance_preprocess=lambda x: x,
        norm: Optional[Union[str, int]] = None,
    ) -> None:
        # Parameters
        self.x_clean = x_clean
        self.y_clean = y_clean
        self.classifier = classifier
        self.constraints = constraints
        self.fun_distance_preprocess = fun_distance_preprocess
        self.norm = norm

        # Optional parameters
        self.norm = norm

        # Caching
        self.x_clean_distance = self.fun_distance_preprocess(
            x_clean.reshape(1, -1)
        )[0]

        # Computed attributes
        xl, xu = get_feature_min_max(constraints, x_clean)
        xl, xu = (
            xl[constraints.mutable_features],
            xu[self.constraints.mutable_features],
        )

        super().__init__(
            n_var=self.constraints.mutable_features.sum(),
            n_obj=get_nb_objectives(),
            n_constr=0,
            xl=xl,
            xu=xu,
        )

    def get_x_clean(self):
        return self.x_clean

    def _obj_misclassify(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.classifier, "predict_proba"):
            y_pred = self.classifier.predict_proba(x)[:, self.y_clean]
        else:
            y_pred = self.classifier(x)[:, self.y_clean]

        return y_pred

    def _obj_distance(self, x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
        return compute_distance(x_1, x_2, self.norm)

    def _calculate_constraints(self, x):
        if (self.constraints.relation_constraints is None) or (
            len(self.constraints.relation_constraints) == 0
        ):
            return np.zeros(x.shape[0])
        executor = ConstraintsExecutor(
            AndConstraint(self.constraints.relation_constraints),
            NumpyBackend(),
            feature_names=self.constraints.feature_names,
        )

        return executor.execute(x)

    def _evaluate(self, x, out, *args, **kwargs):

        # print("Evaluate")

        # Sanity check
        if (x - self.xl < 0).sum() > 0:
            print("Lower than lower bound.")

        if (x - self.xu > 0).sum() > 0:
            print("Lower than lower bound.")

        # --- Prepare necessary representation of the samples

        # Retrieve original representation

        x_adv = np.repeat(self.x_clean.reshape(1, -1), x.shape[0], axis=0)
        x_adv[:, self.constraints.mutable_features] = x

        obj_misclassify = self._obj_misclassify(x_adv)

        obj_distance = self._obj_distance(
            self.fun_distance_preprocess(x_adv), self.x_clean_distance
        )

        obj_constraints = self._calculate_constraints(x_adv)

        F = [to_numpy_number(obj_misclassify), obj_distance, obj_constraints]

        # --- Output
        out["F"] = np.column_stack(F)
