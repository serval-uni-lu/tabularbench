from typing import List, Optional, Union

import numpy as np

from tabularbench.constraints.backend import Backend
from tabularbench.utils.typing import NDNumber


class NumpyBackend(Backend):
    def __init__(self, eps: float = 0.000001) -> None:
        self.eps = np.array(eps)

    def get_eps(self) -> NDNumber:
        return self.eps

    def get_zeros(self, operands: List[NDNumber]) -> NDNumber:
        i = np.argmax([op.shape[-1] for op in operands])
        return np.zeros(operands[i].shape, dtype=operands[i].dtype)

    # Values
    def constant(self, value: Union[int, float]) -> NDNumber:
        return np.array([value])

    def feature(self, x: NDNumber, feature_id: int) -> NDNumber:
        return x[:, feature_id]

    # Math operations

    def math_operation(
        self,
        operator: str,
        left_operand: NDNumber,
        right_operand: NDNumber,
    ) -> NDNumber:
        if operator == "+":
            return left_operand + right_operand
        elif operator == "-":
            return left_operand - right_operand
        elif operator == "*":
            return left_operand * right_operand
        elif operator == "/":
            return left_operand / right_operand
        elif operator == "**":
            return left_operand**right_operand
        elif operator == "%":
            return left_operand % right_operand
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def safe_division(
        self,
        dividend: NDNumber,
        divisor: NDNumber,
        safe_value: NDNumber,
    ) -> NDNumber:
        return np.divide(
            dividend,
            divisor,
            out=np.full_like(dividend, safe_value),
            where=divisor != 0,
        )

    def log(
        self, operand: NDNumber, safe_value: Optional[NDNumber] = None
    ) -> NDNumber:
        if safe_value is not None:
            return np.log(
                operand,
                out=np.full_like(operand, fill_value=safe_value),
                where=(operand > 0),
            )
        return np.log(operand)

    def many_sum(self, operands: List[NDNumber]) -> NDNumber:

        zeros = self.get_zeros(operands)
        ops = []
        for op in operands:
            if op.shape[-1] == 1:
                ops.append(zeros + op)
            else:
                ops.append(op)
        return np.sum(ops, axis=0)

    def less_equal_constraint(
        self, left_operand: NDNumber, right_operand: NDNumber
    ) -> NDNumber:

        zeros = self.get_zeros([left_operand, right_operand])
        substraction = left_operand - right_operand
        bound_zero = np.max(np.stack([zeros, substraction]), axis=0)

        return bound_zero

    def less_constraint(
        self, left_operand: NDNumber, right_operand: NDNumber
    ) -> NDNumber:
        zeros = self.get_zeros([left_operand, right_operand])
        substraction = (left_operand + self.eps) - right_operand
        bound_zero = np.max(np.stack([zeros, substraction]), axis=0)
        return bound_zero

    def equal_constraint(
        self,
        left_operand: NDNumber,
        right_operand: NDNumber,
        tolerance: Optional[NDNumber] = None,
    ) -> NDNumber:
        abs_diff = np.abs(left_operand - right_operand)

        if tolerance is None:
            return abs_diff

        zeros = self.get_zeros([abs_diff, tolerance])
        substraction = abs_diff - tolerance
        bound_zero = np.max(np.stack([zeros, substraction]), axis=0)
        return bound_zero

    def or_constraint(self, operands: List[NDNumber]) -> NDNumber:
        return np.min(np.stack(operands), axis=0)

    def and_constraint(self, operands: List[NDNumber]) -> NDNumber:
        return np.sum(np.stack(operands), axis=0)

    def count(self, operands: List[NDNumber], inverse: bool) -> NDNumber:
        if inverse:
            return np.sum(
                np.stack([(op != 0).astype(float) for op in operands]),
                axis=0,
            )
        else:
            return np.sum(
                np.stack([(op == 0).astype(float) for op in operands]),
                axis=0,
            )
