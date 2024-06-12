import typing
from abc import abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
import torch

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
)
from tabularbench.constraints.utils import get_feature_index

EPS: torch.Tensor = torch.tensor(0.000001)


class ConstraintsVisitor:
    """Abstract Visitor Class"""

    @abstractmethod
    def visit(self, item: ConstraintsNode) -> Any:
        pass

    @abstractmethod
    def execute(self) -> Any:
        pass


class PytorchConstraintsVisitor(ConstraintsVisitor):

    str_operator_to_result = {
        "+": lambda left, right: left + right,
        "-": lambda left, right: left - right,
        "*": lambda left, right: left * right,
        "/": lambda left, right: left / right,
        "**": lambda left, right: left**right,
        "%": lambda left, right: left % right,
    }

    def __init__(
        self,
        constraint: BaseRelationConstraint,
        feature_names: npt.ArrayLike = None,
    ) -> None:
        self.constraint = constraint
        self.feature_names = feature_names

    @staticmethod
    def get_zeros_torch(
        operands: typing.List["torch.Tensor"],
    ) -> "torch.Tensor":
        i = np.argmax([op.ndim for op in operands])
        return torch.zeros(operands[i].shape, dtype=operands[i].dtype)

    def visit(
        self, constraint_node: ConstraintsNode
    ) -> Callable[[torch.Tensor], torch.Tensor]:

        # ------------ Values
        if isinstance(constraint_node, Constant):
            constant = constraint_node.constant

            def process(x: torch.Tensor) -> torch.Tensor:
                return torch.tensor([constant])

            return process

        elif isinstance(constraint_node, Feature):
            feature_index = get_feature_index(
                self.feature_names, constraint_node.feature_id
            )

            def process(x: torch.Tensor) -> torch.Tensor:
                return x[:, feature_index]

            return process

        elif isinstance(constraint_node, MathOperation):
            operator = constraint_node.operator
            if not (operator in self.str_operator_to_result):
                raise NotImplementedError

            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)

            operator_function = self.str_operator_to_result[operator]

            def process(x: torch.Tensor) -> torch.Tensor:
                return operator_function(left_operand(x), right_operand(x))

            return process

        elif isinstance(constraint_node, SafeDivision):
            dividend = constraint_node.dividend.accept(self)
            divisor = constraint_node.divisor.accept(self)
            fill_value = constraint_node.fill_value.accept(self)

            def process(x: torch.Tensor) -> torch.Tensor:
                return torch.where(
                    divisor(x) != 0,
                    torch.div(dividend(x), divisor(x)),
                    fill_value(x),
                )

            return process

        elif isinstance(constraint_node, Log):
            operand = constraint_node.operand.accept(self)
            if constraint_node.safe_value is not None:
                safe_value = constraint_node.safe_value.accept(self)

                def process(x: torch.Tensor) -> torch.Tensor:
                    return torch.where(
                        operand(x) > 0, torch.log(operand(x)), safe_value(x)
                    )

                return process

            def process(x: torch.Tensor) -> torch.Tensor:
                return torch.log(operand(x))

            return process

        elif isinstance(constraint_node, ManySum):
            operands = [e.accept(self) for e in constraint_node.operands]

            def process(x: torch.Tensor) -> torch.Tensor:
                return torch.sum(
                    torch.stack([op(x) for op in operands]), dim=0
                )

            return process

        # ------------ Constraints

        # ------ Binary
        elif isinstance(constraint_node, OrConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]

            def process(x: torch.Tensor) -> torch.Tensor:
                return torch.min(
                    torch.stack([op(x) for op in operands]), dim=0
                ).values

            return process

        elif isinstance(constraint_node, AndConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]

            def process(x: torch.Tensor) -> torch.Tensor:
                return torch.sum(
                    torch.stack([op(x) for op in operands]), dim=0
                )

            return process

        # ------ Comparison TODO: continue here
        elif isinstance(constraint_node, LessEqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)

            def process(x: torch.Tensor) -> torch.Tensor:
                zeros = self.get_zeros_torch(
                    [left_operand(x), right_operand(x)]
                )
                substraction = left_operand(x) - right_operand(x)
                bound_zero = torch.max(
                    torch.stack([zeros, substraction]), dim=0
                )[0]
                return bound_zero

            return process

        elif isinstance(constraint_node, LessConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)

            def process(x: torch.Tensor) -> torch.Tensor:
                zeros = self.get_zeros_torch(
                    [left_operand(x), right_operand(x)]
                )
                substraction = (left_operand(x) + EPS) - right_operand(x)
                bound_zero = torch.max(
                    torch.stack([zeros, substraction]), dim=0
                )[0]
                return bound_zero

            return process

        elif isinstance(constraint_node, EqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)

            def process(x: torch.Tensor) -> torch.Tensor:
                return torch.abs(left_operand(x) - right_operand(x))

            return process

        # ------ Extension

        elif isinstance(constraint_node, Count):
            operands = [e.accept(self) for e in constraint_node.operands]
            if constraint_node.inverse:

                def process(x: torch.Tensor) -> torch.Tensor:
                    return torch.sum(
                        torch.stack([(op(x) != 0).float() for op in operands]),
                        dim=0,
                    )

                return process
            else:

                def process(x: torch.Tensor) -> torch.Tensor:
                    return torch.sum(
                        torch.stack([(op(x) == 0).float() for op in operands]),
                        dim=0,
                    )

                return process

        else:
            raise NotImplementedError

    def execute(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.constraint.accept(self)


class PytorchConstraintsParser:
    def __init__(
        self,
        constraint: BaseRelationConstraint,
        feature_names: npt.ArrayLike = None,
    ):
        self.constraint = constraint
        self.feature_names = feature_names
        self.process: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def parse(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self.process is None:
            visitor = PytorchConstraintsVisitor(
                self.constraint, self.feature_names
            )
            self.process = visitor.execute()

        return self.process

    def execute(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.parse()(x)
