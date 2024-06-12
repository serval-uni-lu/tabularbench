from __future__ import annotations

import typing
from typing import Any, List, Optional, Union


def _check_min_operands_length(
    expected: int, operands: List[typing.Any]
) -> None:
    if len(operands) < expected:
        raise ValueError(
            f"Operands length={len(operands)}, expected: >= {expected}."
        )


class ConstraintsNode:
    def accept(self, visitor: typing.Any) -> typing.Any:
        return visitor.visit(self)


# ------------ Values
class Value(ConstraintsNode):
    def __add__(self, other: Value) -> ManySum:
        if isinstance(self, ManySum):
            return ManySum(self.operands + [other])
        elif isinstance(other, ManySum):
            return ManySum([self] + other.operands)
        else:
            return ManySum([self, other])

    def __sub__(self, other: Value) -> MathOperation:
        return MathOperation("-", self, other)

    def __mul__(self, other: Value) -> MathOperation:
        return MathOperation("*", self, other)

    def __truediv__(self, other: Value) -> MathOperation:
        return MathOperation("/", self, other)

    def __pow__(self, power: Value, modulo: Any = None) -> MathOperation:
        return MathOperation("**", self, power)

    def __mod__(self, other: Value) -> MathOperation:
        return MathOperation("%", self, other)

    def __lt__(self, other: Value) -> BaseRelationConstraint:
        return LessConstraint(self, other)

    def __le__(self, other: Value) -> BaseRelationConstraint:
        return LessEqualConstraint(self, other)

    def __eq__(self, other: object) -> typing.Any:
        if isinstance(other, Value):
            return EqualConstraint(self, other)
        else:
            print(type(other))
            raise ValueError


class Constant(Value):
    def __init__(self, constant: Union[int, float]):
        self.constant = constant


class Feature(Value):
    def __init__(self, feature_id: Union[str, int]):
        self.feature_id = feature_id


# ------------
class MathOperation(Value):
    def __init__(
        self, operator: str, left_operand: Value, right_operand: Value
    ):
        self.operator = operator
        self.left_operand = left_operand
        self.right_operand = right_operand


class SafeDivision(Value):
    def __init__(self, dividend: Value, divisor: Value, fill_value: Value):
        self.dividend = dividend
        self.divisor = divisor
        self.fill_value = fill_value


class ManySum(Value):
    def __init__(self, operands: List[Value]) -> None:
        self.operands = operands


class Log(Value):
    def __init__(self, operand: Value, safe_value: Value = None):
        self.operand = operand
        self.safe_value = safe_value


# ------------ Constraints


class BaseRelationConstraint(ConstraintsNode):
    def __or__(self, other: BaseRelationConstraint) -> OrConstraint:
        return OrConstraint([self, other])

    def __and__(self, other: BaseRelationConstraint) -> AndConstraint:
        return AndConstraint([self, other])


# ------ Binary


class OrConstraint(BaseRelationConstraint):
    def __init__(self, operands: List[BaseRelationConstraint]):
        _check_min_operands_length(2, operands)
        self.operands = operands


class AndConstraint(BaseRelationConstraint):
    def __init__(self, operands: List[BaseRelationConstraint]):
        _check_min_operands_length(2, operands)
        self.operands = operands


# ----- Comparison


class LessEqualConstraint(BaseRelationConstraint):
    def __init__(self, left_operand: Value, right_operand: Value):
        self.left_operand = left_operand
        self.right_operand = right_operand


class LessConstraint(BaseRelationConstraint):
    def __init__(self, left_operand: Value, right_operand: Value):
        self.left_operand = left_operand
        self.right_operand = right_operand


class EqualConstraint(BaseRelationConstraint):
    def __init__(
        self,
        left_operand: Value,
        right_operand: Value,
        tolerance: Optional[Value] = None,
    ):
        self.left_operand = left_operand
        self.right_operand = right_operand
        self.tolerance = tolerance


# ----- Extension


class Count(Value):
    def __init__(
        self, operands: List[BaseRelationConstraint], inverse: bool = False
    ) -> None:
        _check_min_operands_length(1, operands)
        self.operands = operands
        self.inverse = inverse
