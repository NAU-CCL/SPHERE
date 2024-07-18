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


class StepFunctionParam(FunctionalParam):
    """A step-function version of beta that takes a list of values to cycle through."""

    def __init__(self, values: list[float], period: int = 30):
        """
        Args:
            values: List of beta values to cycle through.
            period: The number of time steps for each value.
        """
        if not values:
            raise ValueError("Values list must not be empty.")

        self.values = values
        self.period = period
        self.current_index = 0
        self.current = values[self.current_index]

    def get_current_value(self, t: int) -> float:
        """Get the current value of beta at time t."""
        if t % self.period == self.period - 1:
            self._switch_value()
        return self.current

    def _switch_value(self) -> None:
        """Switch the current value to the next one in the list."""
        self.current_index = (self.current_index + 1) % len(self.values)
        self.current = self.values[self.current_index]
