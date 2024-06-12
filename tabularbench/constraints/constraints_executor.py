import typing
from abc import abstractmethod
from typing import Any, Optional, Union

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
    Value,
)
from tabularbench.constraints.utils import get_feature_index

EPS: npt.NDArray[Any] = np.array(0.000001)


class ConstraintsVisitor:
    """Abstract Visitor Class"""

    @abstractmethod
    def visit(self, item: ConstraintsNode) -> Any:
        pass

    @abstractmethod
    def execute(self) -> Any:
        pass


class NumpyConstraintsVisitor(ConstraintsVisitor):
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
        constraint: Union[BaseRelationConstraint, Value],
        x: npt.NDArray[Any],
        feature_names: npt.ArrayLike = None,
    ) -> None:
        self.constraint = constraint
        self.x = x
        self.feature_names = feature_names

    @staticmethod
    def get_zeros_np(
        operands: typing.List[npt.NDArray[Any]],
    ) -> npt.NDArray[Any]:
        i = np.argmax([op.ndim for op in operands])
        return np.zeros(operands[i].shape, dtype=operands[i].dtype)

    def visit(self, constraint_node: ConstraintsNode) -> npt.NDArray[Any]:
        # ------------ Values
        if isinstance(constraint_node, Constant):
            return np.array(constraint_node.constant)

        elif isinstance(constraint_node, Feature):
            feature_index = get_feature_index(
                self.feature_names, constraint_node.feature_id
            )
            return self.x[:, feature_index]

        elif isinstance(constraint_node, MathOperation):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            operator = constraint_node.operator
            if operator in self.str_operator_to_result:
                return self.str_operator_to_result[operator](
                    left_operand, right_operand
                )
            else:
                raise NotImplementedError

        elif isinstance(constraint_node, SafeDivision):
            dividend = constraint_node.dividend.accept(self)
            divisor = constraint_node.divisor.accept(self)
            fill_value = constraint_node.fill_value.accept(self)
            return np.divide(
                dividend,
                divisor,
                out=np.full_like(dividend, fill_value),
                where=divisor != 0,
            )

        elif isinstance(constraint_node, Log):
            operand = constraint_node.operand.accept(self)
            if constraint_node.safe_value is not None:
                safe_value = constraint_node.safe_value.accept(self)
                return np.log(
                    operand,
                    out=np.full_like(operand, fill_value=safe_value),
                    where=(operand > 0),
                )
            return np.log(operand)

        elif isinstance(constraint_node, ManySum):
            operands = [e.accept(self) for e in constraint_node.operands]
            return np.sum(operands, axis=0)

        # ------------ Constraints

        # ------ Binary
        elif isinstance(constraint_node, OrConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]
            return np.min(operands, axis=0)

        elif isinstance(constraint_node, AndConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]
            return np.sum(operands, axis=0)

        # ------ Comparison
        elif isinstance(constraint_node, LessEqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            zeros = self.get_zeros_np([left_operand, right_operand])
            return np.max([zeros, (left_operand - right_operand)], axis=0)

        elif isinstance(constraint_node, LessConstraint):
            left_operand = constraint_node.left_operand.accept(self) + EPS
            right_operand = constraint_node.right_operand.accept(self)
            zeros = self.get_zeros_np([left_operand, right_operand])
            return np.max([zeros, (left_operand - right_operand)], axis=0)

        elif isinstance(constraint_node, EqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            return np.abs(left_operand - right_operand)

        # ------ Extension

        elif isinstance(constraint_node, Count):
            operands = [e.accept(self) for e in constraint_node.operands]
            if constraint_node.inverse:
                operands = [(op != 0).astype(float) for op in operands]
            else:
                operands = [(op == 0).astype(float) for op in operands]

            return np.sum(np.array(operands), axis=0)

        else:
            raise NotImplementedError

    def execute(self) -> npt.NDArray[Any]:
        return self.constraint.accept(self)


class NumpyConstraintsExecutor:
    def __init__(
        self,
        constraint: Union[BaseRelationConstraint, Value],
        feature_names: npt.ArrayLike = None,
    ):
        self.constraint = constraint
        self.feature_names = feature_names

    def execute(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        visitor = NumpyConstraintsVisitor(
            self.constraint, x, self.feature_names
        )
        return visitor.execute()


class TextConstraintsVisitor(ConstraintsVisitor):
    str_operator_to_result = {
        "+": lambda left, right: left + right,
        "-": lambda left, right: left - right,
        "*": lambda left, right: left * right,
        "/": lambda left, right: left / right,
        "**": lambda left, right: left**right,
    }

    def __init__(
        self,
        constraint: BaseRelationConstraint,
    ) -> None:
        self.constraint = constraint

    def visit(self, constraint_node: ConstraintsNode) -> str:
        # ------------ Values
        if isinstance(constraint_node, Constant):
            return str(constraint_node.constant)

        elif isinstance(constraint_node, Feature):
            return f"F({str(constraint_node.feature_id)})"

        elif isinstance(constraint_node, MathOperation):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            operator = constraint_node.operator
            return f"({left_operand} {operator} {right_operand})"

        elif isinstance(constraint_node, SafeDivision):
            dividend = constraint_node.dividend.accept(self)
            divisor = constraint_node.divisor.accept(self)
            fill_value = constraint_node.fill_value.accept(self)
            return f"SafeDiv({dividend}, {divisor}, {fill_value})"

        # ------------ Constraints

        # ------ Binary
        elif isinstance(constraint_node, OrConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]
            out = " OR ".join(operands)
            return f"({out})"

        elif isinstance(constraint_node, AndConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]
            out = " AND ".join(operands)
            return f"({out})"

        # ------ Comparison
        elif isinstance(constraint_node, LessEqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            return f"({left_operand} <= {right_operand})"

        elif isinstance(constraint_node, LessConstraint):
            left_operand = constraint_node.left_operand.accept(self) + EPS
            right_operand = constraint_node.right_operand.accept(self)
            return f"({left_operand} < {right_operand})"

        elif isinstance(constraint_node, EqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            return f"({left_operand} == {right_operand})"

        elif isinstance(constraint_node, ManySum):
            operands = [e.accept(self) for e in constraint_node.operands]
            out = " + ".join(operands)
            out = f"({out})"
            return out
        else:
            raise ConstraintNotImplementedError(constraint_node)

    def execute(self) -> str:
        return self.constraint.accept(self)


class ConstraintNotImplementedError(NotImplementedError):
    def __init__(
        self, constraint: ConstraintsNode, message: Optional[str] = None
    ) -> None:
        self.constraint = constraint
        if message is None:
            message = (
                f"Support for constraint Constraint of type "
                f"{type(self.constraint)} not implemented."
            )
        super().__init__(message)


class TextConstraintsExecutor:
    def __init__(
        self,
        constraint: BaseRelationConstraint,
    ):
        self.constraint = constraint

    def execute(self) -> str:
        visitor = TextConstraintsVisitor(self.constraint)
        return visitor.execute()


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
        x: "torch.Tensor",
        feature_names: npt.ArrayLike = None,
    ) -> None:
        self.constraint = constraint
        self.x = x
        self.feature_names = feature_names

    @staticmethod
    def get_zeros_np(
        operands: typing.List["torch.Tensor"],
    ) -> "torch.Tensor":
        i = np.argmax([op.ndim for op in operands])
        return torch.zeros(operands[i].shape, dtype=operands[i].dtype)

    def visit(self, constraint_node: ConstraintsNode) -> "torch.Tensor":
        # ------------ Values
        if isinstance(constraint_node, Constant):
            return torch.from_numpy(np.array(constraint_node.constant))

        elif isinstance(constraint_node, Feature):
            feature_index = get_feature_index(
                self.feature_names, constraint_node.feature_id
            )
            return self.x[:, feature_index]

        elif isinstance(constraint_node, MathOperation):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            operator = constraint_node.operator
            if operator in self.str_operator_to_result:
                return self.str_operator_to_result[operator](
                    left_operand, right_operand
                )
            else:
                raise NotImplementedError

        elif isinstance(constraint_node, SafeDivision):
            dividend = constraint_node.dividend.accept(self)
            divisor = constraint_node.divisor.accept(self)
            fill_value = constraint_node.fill_value.accept(self)
            return torch.where(
                divisor != 0, torch.div(dividend, divisor), fill_value
            )

        elif isinstance(constraint_node, Log):
            operand = constraint_node.operand.accept(self)
            if constraint_node.safe_value is not None:
                safe_value = constraint_node.safe_value.accept(self)
                return torch.where(operand > 0, torch.log(operand), safe_value)

            return torch.log(operand)

        elif isinstance(constraint_node, ManySum):
            operands = [e.accept(self) for e in constraint_node.operands]
            return torch.sum(torch.stack(operands), dim=0)

        # ------------ Constraints

        # ------ Binary
        elif isinstance(constraint_node, OrConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]
            return torch.min(torch.stack(operands), dim=0).values

        elif isinstance(constraint_node, AndConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]
            local_sum = operands[0]
            for i in range(1, len(operands)):
                local_sum = local_sum + operands[i]
            return local_sum

        # ------ Comparison
        elif isinstance(constraint_node, LessEqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            zeros = self.get_zeros_np([left_operand, right_operand])
            my_sub = left_operand - right_operand
            my_test = torch.stack([zeros, my_sub])
            return torch.max(my_test, dim=0)[0]

        elif isinstance(constraint_node, LessConstraint):
            left_operand = constraint_node.left_operand.accept(self) + EPS
            right_operand = constraint_node.right_operand.accept(self)
            zeros = self.get_zeros_np([left_operand, right_operand])
            return torch.max(
                torch.stack([zeros, (left_operand - right_operand)]), dim=0
            )

        elif isinstance(constraint_node, EqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            return torch.abs(left_operand - right_operand)

            # ------ Extension

        elif isinstance(constraint_node, Count):
            operands = [e.accept(self) for e in constraint_node.operands]
            if constraint_node.inverse:
                operands = [(op != 0).float() for op in operands]
            else:
                operands = [(op == 0).float() for op in operands]
            return torch.sum(torch.stack(operands), dim=0)

        else:
            raise NotImplementedError

    def execute(self) -> "torch.Tensor":
        return self.constraint.accept(self)


class PytorchConstraintsExecutor:
    def __init__(
        self,
        constraint: BaseRelationConstraint,
        feature_names: npt.ArrayLike = None,
    ):
        self.constraint = constraint
        self.feature_names = feature_names

    def execute(self, x: "torch.Tensor") -> "torch.Tensor":
        visitor = PytorchConstraintsVisitor(
            self.constraint, x, self.feature_names
        )
        return visitor.execute()
