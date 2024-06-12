from typing import List, Optional, Union

import numpy as np
import torch

from tabularbench.constraints.backend import Backend


class PytorchBackend(Backend):
    def __init__(
        self, eps: float = 0.000001, device: Union[str, torch.device] = "cpu"
    ) -> None:
        self.eps = torch.tensor(eps)
        self.device = device

    def set_device(self, device: Union[str, torch.device]) -> None:
        self.device = device

    def get_device(self) -> Union[str, torch.device]:
        return self.device

    def get_eps(self) -> torch.Tensor:
        return self.eps

    def get_zeros(self, operands: List[torch.Tensor]) -> torch.Tensor:
        i = np.argmax([op.shape[-1] for op in operands])
        return torch.zeros(operands[i].shape, dtype=operands[i].dtype).to(
            operands[i].device
        )

    # Values
    def constant(self, value: Union[int, float]) -> torch.Tensor:
        return torch.tensor(np.array([value])).to(self.device)

    def feature(self, x: torch.Tensor, feature_id: int) -> torch.Tensor:
        return x[:, feature_id]

    # Math operations

    def math_operation(
        self,
        operator: str,
        left_operand: torch.Tensor,
        right_operand: torch.Tensor,
    ) -> torch.Tensor:
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
        dividend: torch.Tensor,
        divisor: torch.Tensor,
        safe_value: torch.Tensor,
    ) -> torch.Tensor:
        # This new version does not break gradient computation

        out = torch.full_like(dividend, fill_value=float(safe_value))
        mask = torch.abs(divisor) > self.eps
        out[mask] = dividend[mask] / divisor[mask]
        if torch.isinf(out).float().sum() > 0:
            for _ in range(12):
                print("------------------------ INF ------------------------")
        return out

    def log(
        self, operand: torch.Tensor, safe_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if safe_value is not None:
            out = torch.full_like(operand, fill_value=float(safe_value))
            mask = torch.abs(operand) > self.eps
            out[mask] = torch.log(operand[mask])
        else:
            out = torch.log(operand)
        return out

    def many_sum(self, operands: List[torch.Tensor]) -> torch.Tensor:
        if isinstance(operands, torch.Tensor):
            return operands.sum(dim=1)

        zeros = self.get_zeros(operands)
        parsed_operands = []
        for op in operands:
            if op.shape[-1] == 1:
                parsed_operands.append(zeros + op)
            else:
                parsed_operands.append(op.to(operands[0].device))

        return torch.sum(torch.stack(parsed_operands), dim=0)

    def less_equal_constraint(
        self, left_operand: torch.Tensor, right_operand: torch.Tensor
    ) -> torch.Tensor:
        zeros = self.get_zeros([left_operand, right_operand]).to(
            left_operand.device
        )
        substraction = left_operand - right_operand
        bound_zero = torch.max(torch.stack([zeros, substraction]), dim=0)[0]

        return bound_zero

    def less_constraint(
        self, left_operand: torch.Tensor, right_operand: torch.Tensor
    ) -> torch.Tensor:
        zeros = self.get_zeros([left_operand, right_operand])
        substraction = (left_operand + self.eps) - right_operand
        bound_zero = torch.max(torch.stack([zeros, substraction]), dim=0)[0]
        return bound_zero

    def equal_constraint(
        self,
        left_operand: torch.Tensor,
        right_operand: torch.Tensor,
        tolerance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        abs_diff = torch.abs(left_operand - right_operand)
        if tolerance is None:
            return abs_diff
        zeros = self.get_zeros([abs_diff, tolerance])
        substraction = abs_diff - tolerance
        bound_zero = torch.max(torch.stack([zeros, substraction]), dim=0)[0]
        return bound_zero

    def or_constraint(self, operands: List[torch.Tensor]) -> torch.Tensor:
        return torch.min(torch.stack(operands), dim=0).values

    def and_constraint(self, operands: List[torch.Tensor]) -> torch.Tensor:
        return torch.sum(torch.stack(operands), dim=0)

    def count(
        self, operands: List[torch.Tensor], inverse: bool
    ) -> torch.Tensor:
        if inverse:
            return torch.sum(
                torch.stack([(op != 0).float() for op in operands]),
                dim=0,
            )
        else:
            return torch.sum(
                torch.stack([(op == 0).float() for op in operands]),
                dim=0,
            )
