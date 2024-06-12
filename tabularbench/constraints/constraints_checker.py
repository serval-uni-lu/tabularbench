from typing import Any

import numpy as np
import numpy.typing as npt

from tabularbench.constraints.constraints import (
    Constraints,
    get_feature_min_max,
)
from tabularbench.constraints.constraints_backend_executor import (
    ConstraintsExecutor,
)
from tabularbench.constraints.numpy_backend import NumpyBackend
from tabularbench.constraints.relation_constraint import AndConstraint
from tabularbench.utils.typing import NDBool, NDFloat


class ConstraintChecker:
    def __init__(self, constraints: Constraints, tolerance: float = 0.0):
        self.constraints = constraints
        self.tolerance = tolerance

    def _check_relationship_constraints(
        self, x_adv: npt.NDArray[Any]
    ) -> NDBool:

        no_constraints = (self.constraints.relation_constraints is None) or (
            len(self.constraints.relation_constraints) == 0
        )
        if no_constraints:
            return np.zeros(x_adv.shape[0]) <= self.tolerance

        constraints_executor = ConstraintsExecutor(
            AndConstraint(self.constraints.relation_constraints),
            backend=NumpyBackend(),
            feature_names=self.constraints.feature_names,
        )
        out = constraints_executor.execute(x_adv)

        # for i, e in enumerate(self.constraints.relation_constraints):
        #     constraints_executor = ConstraintsExecutor(
        #         AndConstraint([e, e]),
        #         backend=NumpyBackend(),
        #         feature_names=self.constraints.feature_names,
        #     )
        #     tmp = constraints_executor.execute(x_adv)
        #     if tmp.mean() > 0:
        #         print(f"{i}: {tmp}")

        if not isinstance(out, np.ndarray):
            raise ValueError(
                "ConstraintExecutor did not return a numpy array."
            )
        return out <= self.tolerance

    def _check_boundary_constraints(
        self, x: NDFloat, x_adv: NDFloat
    ) -> NDBool:
        xl, xu = get_feature_min_max(self.constraints, x)

        # Hard fix for wids, both value are acceptable in theory,
        # but keep like this to guarantee running for wids and,
        # reproducability for others.

        float_tolerance = np.finfo(np.float32).eps
        if x.shape[1] == 108:
            float_tolerance = 1000 * float_tolerance

        # print(float_tolerance)
        xl_ok, xu_ok = np.min((xl - float_tolerance) <= x_adv, axis=1), np.min(
            (xu + float_tolerance) >= x_adv, axis=1
        )

        # if xl_ok.mean() < 1:
        #     print("MIN")
        #     loc = np.where((xl - np.finfo(np.float32).eps) > x_adv)
        #     print(np.where((xl - np.finfo(np.float32).eps) > x_adv))
        #     print(np.abs((xl - x_adv.min(0))).max())
        #     print("PB down")
        #     # print(f"XLLLL {xl}")
        #     print(xl[0, loc[1]], x_adv[0, loc[1]])
        #     print(xl[0, loc[1]] - x_adv[0, loc[1]])
        #     exit(0)
        # if xu_ok.mean() < 1:
        #     print("Max")
        #     loc = np.where((xu + np.finfo(np.float32).eps) < x_adv)
        #     print(loc)
        #     print(np.abs((xu - x_adv.max(0))).max())
        #     print("PB Up")
        #     # print(f"XLLLL {xu}")
        #     print(xu[0, loc[1]], x_adv[0, loc[1]])
        #     print(xu[0, loc[1]] - x_adv[0, loc[1]])
        #     exit(0)
        return xl_ok * xu_ok

    def _check_type_constraints(self, x_adv: NDFloat) -> NDBool:
        int_type_mask = self.constraints.feature_types != "real"
        if int_type_mask.sum() > 0:
            type_ok = np.min(
                (x_adv[:, int_type_mask] == np.round(x_adv[:, int_type_mask])),
                axis=1,
            )
        else:
            type_ok = np.ones(shape=x_adv.shape[:-1], dtype=np.bool_)
        return type_ok

    def _check_mutable_constraints(self, x: NDFloat, x_adv: NDFloat) -> NDBool:
        immutable_mask = ~self.constraints.mutable_features
        if immutable_mask.sum() > 0:
            mutable_ok = np.min(
                (x[:, immutable_mask] == x_adv[:, immutable_mask]), axis=1
            )
        else:
            mutable_ok = np.ones(shape=x_adv.shape[:-1], dtype=np.bool_)
        return mutable_ok

    def check_constraints(self, x: NDFloat, x_adv: NDFloat) -> NDFloat:
        constraints = np.array(
            [
                self._check_relationship_constraints(x_adv),
                self._check_boundary_constraints(x, x_adv),
                self._check_type_constraints(x_adv),
                self._check_mutable_constraints(x, x_adv),
            ]
        )

        # print("CONSTRAINTS CHECKER ")
        # print(f"RELATION: {constraints[0].mean()}")
        # print(f"BOUND: {constraints[1].mean()}")
        # print(f"TYPE: {constraints[2].mean()}")
        # print(f"MUTABLE: {constraints[3].mean()}")

        # if constraints[1].mean() < 1.:
        #     print("BOUND FAIL")
        #     traceback.print_stack()
        #     print(self.constraints.lower_bounds)
        #     print(x_adv)
        #     exit(0)

        # if constraints[2].mean() < 1.:
        #     print("TYPE FAIL")
        #     traceback.print_stack()
        #     exit(0)

        # if constraints[3].mean() < 1.:
        #     print("MUTABLE FAIL")
        #     traceback.print_stack()
        #     exit(0)

        # if constraints[0].mean() < 1.:
        #     print("RELATION FAIL")
        #     traceback.print_stack()
        #     exit(0)

        constraints = np.min(constraints, axis=0)
        return constraints
