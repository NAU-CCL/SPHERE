from abc import ABC, abstractmethod


class FunctionalParam(ABC):
    """An abstract class for functional parameters."""
    @abstractmethod
    def get_current_value(self, t: int) -> float:
        """Get the current value of the parameter."""
        raise NotImplementedError("Subclasses should implement this method.")


class ConstantParam(FunctionalParam):
    """A constant-valued parameter."""
    def __init__(self, value: float):
        self._value = value

    def get_current_value(self, t: int) -> float:
        return self._value


class StepBeta(FunctionalParam):
    """A step-function version of beta."""
    def __init__(self, high_val: float = 0.3, low_val: float = 0.1, period: int = 30, start_high: bool = True):
        self.high = high_val
        self.low = low_val
        self.period = period
        self.current = high_val if start_high else low_val

    def get_current_value(self, t: int) -> float:
        """Get the current value of beta at time t."""
        if t % self.period == 0:
            self._switch_value()
        return self.current

    def _switch_value(self) -> None:
        self.current = self.low if self.current == self.high else self.high

