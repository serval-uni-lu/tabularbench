from abc import abstractmethod
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import torch

from tabularbench.constraints.backend import Backend
from tabularbench.constraints.pytorch_backend import PytorchBackend
from tabularbench.constraints.relation_constraint import (
    AndConstraint,
    BaseRelationConstraint,
    Constant,
    ConstraintsNode,
    Count,
    EqualConstraint,
    Feature,
    LessConstraint,
    LessEqualConstraint,
    Log,
    ManySum,
    MathOperation,
    OrConstraint,
    SafeDivision,
    Value,
)
from tabularbench.constraints.utils import get_feature_index
from tabularbench.utils.typing import NDNumber

EPS: npt.NDArray[Any] = np.array(0.000001)


class ConstraintsVisitor:
    """Abstract Visitor Class"""

    @abstractmethod
    def visit(self, item: ConstraintsNode) -> Any:
        pass

    @abstractmethod
    def execute(self) -> Any:
        pass


class BackendConstraintsVisitor(ConstraintsVisitor):
    def __init__(
        self,
        constraint: Union[BaseRelationConstraint, Value],
        x: Union[NDNumber, torch.Tensor],
        backend: Backend,
        feature_names: Optional[npt.ArrayLike] = None,
    ) -> None:
        self.constraint = constraint
        self.x = x
        self.backend = backend
        self.feature_names = (
            tuple(feature_names) if (feature_names is not None) else None
        )

    def visit(self, constraint_node: ConstraintsNode) -> Any:
        # ------------ Values
        if isinstance(constraint_node, Constant):
            return self.backend.constant(constraint_node.constant)

        elif isinstance(constraint_node, Feature):
            feature_index = get_feature_index(
                self.feature_names, constraint_node.feature_id
            )
            return self.backend.feature(self.x, feature_index)

        elif isinstance(constraint_node, MathOperation):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = (
                constraint_node.right_operand.accept(self).to(
                    left_operand.device
                )
                if isinstance(left_operand, torch.Tensor)
                else constraint_node.right_operand.accept(self)
            )
            operator = constraint_node.operator
            return self.backend.math_operation(
                operator, left_operand, right_operand
            )

        elif isinstance(constraint_node, SafeDivision):
            dividend = constraint_node.dividend.accept(self)
            divisor = constraint_node.divisor.accept(self)
            fill_value = constraint_node.fill_value.accept(self)
            return self.backend.safe_division(dividend, divisor, fill_value)

        elif isinstance(constraint_node, Log):
            operand = constraint_node.operand.accept(self)
            if constraint_node.safe_value is not None:
                safe_value = constraint_node.safe_value.accept(self)
            else:
                safe_value = None

            return self.backend.log(operand, safe_value)

        elif isinstance(constraint_node, ManySum):
            operands = [e.accept(self) for e in constraint_node.operands]

            if isinstance(self.backend, PytorchBackend) and all(
                isinstance(element, Feature)
                for element in constraint_node.operands
            ):
                feature_ids = [c.feature_id for c in constraint_node.operands]
                feature_ids = [
                    get_feature_index(self.feature_names, f)
                    for f in feature_ids
                ]
                x_l = self.x[:, feature_ids]
                return self.backend.many_sum(x_l)

            return self.backend.many_sum(operands)

        # ------------ Constraints

        # ------ Binary
        elif isinstance(constraint_node, OrConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]
            return self.backend.or_constraint(operands)

        elif isinstance(constraint_node, AndConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]
            return self.backend.and_constraint(operands)

        # ------ Comparison
        elif isinstance(constraint_node, LessEqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            return self.backend.less_equal_constraint(
                left_operand, right_operand
            )

        elif isinstance(constraint_node, LessConstraint):
            left_operand = constraint_node.left_operand.accept(
                self
            ) + self.backend.constant(EPS)
            right_operand = constraint_node.right_operand.accept(self)
            return self.backend.less_constraint(left_operand, right_operand)

        elif isinstance(constraint_node, EqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            if constraint_node.tolerance is not None:
                tolerance = constraint_node.tolerance.accept(self)
            else:
                tolerance = None
            return self.backend.equal_constraint(
                left_operand, right_operand, tolerance
            )

            # ------ Extension

        elif isinstance(constraint_node, Count):
            operands = [e.accept(self) for e in constraint_node.operands]
            inverse = constraint_node.inverse
            return self.backend.count(operands, inverse)

        else:
            raise NotImplementedError

    def execute(self) -> torch.Tensor:
        return self.constraint.accept(self)


class ConstraintsExecutor:
    def __init__(
        self,
        constraint: Union[BaseRelationConstraint, Value],
        backend: Backend,
        feature_names: Optional[npt.ArrayLike] = None,
    ):
        self.constraint = constraint
        self.backend = backend
        self.feature_names = feature_names

    def execute(
        self, x: Union[NDNumber, torch.Tensor]
    ) -> Union[NDNumber, torch.Tensor]:
        if isinstance(self.backend, PytorchBackend) and isinstance(
            x, torch.Tensor
        ):
            self.backend.set_device(x.device)

        visitor = BackendConstraintsVisitor(
            self.constraint, x, self.backend, self.feature_names
        )
        return visitor.execute()
