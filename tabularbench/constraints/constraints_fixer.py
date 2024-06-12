from typing import List, Optional, Union

import numpy as np
import torch

from tabularbench.constraints.constraints_backend_executor import (
    ConstraintsExecutor,
)
from tabularbench.constraints.pytorch_backend import PytorchBackend
from tabularbench.constraints.relation_constraint import (
    BaseRelationConstraint,
    EqualConstraint,
    Feature,
)
from tabularbench.constraints.utils import get_feature_index
from tabularbench.utils.datatypes import to_numpy_number, to_torch_number
from tabularbench.utils.typing import NDNumber


class ConstraintsFixer:
    def __init__(
        self,
        guard_constraints: List[BaseRelationConstraint],
        fix_constraints: List[EqualConstraint],
        feature_names: Optional[List[str]] = None,
    ):
        self.guard_constraints = guard_constraints
        self.fix_constraints = fix_constraints
        self.feature_names = (
            tuple(feature_names) if (feature_names is not None) else None
        )
        for c in self.fix_constraints:
            if not isinstance(c, EqualConstraint):
                raise ValueError("Fix constraints must be an EqualConstraint.")
            else:
                if not isinstance(c.left_operand, Feature):
                    raise ValueError(
                        "Left operand of fix constraints must be a Feature."
                    )

    def fix(
        self, x: Union[torch.Tensor, NDNumber]
    ) -> Union[torch.Tensor, NDNumber]:

        if isinstance(x, np.ndarray):
            return to_numpy_number(self.fix(to_torch_number(x)))

        x = x.clone()
        # print(f"CONSTRAINTS to fix  {len(self.fix_constraints)}")
        for i in range(len(self.fix_constraints)):
            guard_c = self.guard_constraints[i]
            fix_c = self.fix_constraints[i]

            if not isinstance(fix_c.left_operand, Feature):
                raise ValueError(
                    "Left operand of fix constraints must be a Feature."
                )

            # Index of inputs that shall be updated
            # according the guard constraints,
            # if none then update all.
            if guard_c is not None:
                executor = ConstraintsExecutor(
                    guard_c, PytorchBackend(), self.feature_names
                )
                to_update = executor.execute(x) > 0
            else:
                to_update = torch.ones(x.shape[0], dtype=torch.bool)

            # print(f"FIXING {i}: {to_update.float().mean()}")
            if torch.any(to_update):
                # Index of the feature to update.
                # Ignore warning, this is checked in the constructor.
                index = get_feature_index(
                    self.feature_names, fix_c.left_operand.feature_id
                )
                # Value to be update.
                # Known warning, not supposed to evaluate without constraint.
                executor = ConstraintsExecutor(
                    fix_c.right_operand, PytorchBackend(), self.feature_names
                )
                new_value = executor.execute(x[to_update])
                # print(new_value)

                x[to_update, index] = new_value.float()

        return x
