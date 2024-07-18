"""
The model parameters. An abstract base class is defined.
Each concrete subclass defines that model's unique parameters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple
import jax.numpy as jnp


@dataclass
class Parameters(ABC):
    pass


@dataclass
class SIRParameters(Parameters):
    beta: float
    gamma: float


@dataclass
class SIRHParameters(Parameters):
    beta: float
    gamma: float
    delta: float
    mu: float


@dataclass
class LorenzParameters(Parameters):
    sigma: float
    rho: float
    beta: float


class FunctionalParameter(ABC):
    @abstractmethod
    def get_current_value(self, *args: Any, **kwargs: Any) -> float:
        """Get the current value of the parameter."""
        raise NotImplementedError("Subclasses should implement this method.")


class BetaFunction(FunctionalParameter):
    @abstractmethod
    def get_current_value(self, t: int) -> float:
        """Get the current value of beta at time t."""
        raise NotImplementedError("Subclasses should implement this method.")


class StepBeta(BetaFunction):
    def __init__(self, high_val: float = 0.3, low_val: float = 0.1, period: int = 30, start_high: bool = True):
        self.high = high_val
        self.low = low_val
        self.period = period
        self.current = high_val if start_high else low_val

    def get_current_value(self, t: int) -> float:
        """Get the current value of beta at time t."""
        if t % self.period == 0:
            self.switch_value()
        return self.current

    def switch_value(self) -> None:
        """Switch the current value between high and low."""
        self.current = self.low if self.current == self.high else self.high


