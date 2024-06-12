from abc import abstractmethod
from typing import Any, List, Optional, Union


class Backend:

    # Values

    @abstractmethod
    def get_zeros(self, operands: Any) -> Any:
        pass

    @abstractmethod
    def constant(self, value: Union[int, float]) -> Any:
        pass

    @abstractmethod
    def feature(self, x: Any, feature_id: int) -> Any:
        pass

    # Math operations
    @abstractmethod
    def math_operation(
        self, operator: str, left_operand: Any, right_operand: Any
    ) -> Any:
        pass

    @abstractmethod
    def safe_division(
        self, dividend: Any, divisor: Any, safe_value: Any
    ) -> Any:
        pass

    @abstractmethod
    def log(self, operand: Any, safe_value: Any = None) -> Any:
        pass

    @abstractmethod
    def many_sum(self, operands: List[Any]) -> Any:
        pass

    # Comparisons
    @abstractmethod
    def less_equal_constraint(
        self, left_operand: Any, right_operand: Any
    ) -> Any:
        pass

    @abstractmethod
    def equal_constraint(
        self,
        left_operand: Any,
        right_operand: Any,
        tolerance: Optional[Any] = None,
    ) -> Any:
        pass

    @abstractmethod
    def less_constraint(self, left_operand: Any, right_operand: Any) -> Any:
        pass

    @abstractmethod
    def or_constraint(self, operands: List[Any]) -> Any:
        pass

    @abstractmethod
    def and_constraint(self, operands: List[Any]) -> Any:
        pass

    # Extension

    @abstractmethod
    def count(self, operands: List[Any], inverse: bool) -> Any:
        pass
