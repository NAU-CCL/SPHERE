from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax.numpy as jnp

from sphere.model.abstract.parameters import Parameters


class Transition(ABC):
    """A base class for defining state transition functions.

    All implementations should have a deterministic transition function.

    Some models may also define a stochastic component. Default is False,
    which indicates that the model is not stochastic.
    """
    params: Parameters

    def __init__(self, params: Parameters, is_stochastic: bool = False) -> \
            None:
        self.params = params
        self.is_stochastic = is_stochastic

    @abstractmethod
    def deterministic(self, state: jnp.ndarray, t: int) -> jnp.ndarray:
        """Defines the state transition function for deterministic models."""
        pass

    def stochastic(self, state: jnp.ndarray, t: int) -> None:
        """A stochastic component for state transition functions."""
        return None


class SIRTransition(Transition):
    def __init__(self, params: Parameters, is_stochastic: bool = False) -> None:
        super().__init__(params=params)

    def deterministic(self, state: jnp.ndarray, t: int) -> jnp.ndarray:
        S, I, R = state
        beta, gamma = self.params.beta, self.params.gamma

        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I

        return jnp.array([dS, dI, dR])


class Lorenz63Transition(Transition):
    def __init__(self, params: Parameters, is_stochastic: bool = False) -> None:
        super().__init__(params=params)

    def deterministic(self, state: jnp.ndarray, t: int) -> jnp.ndarray:
        x, y, z = state
        sigma, rho, beta = self.params.sigma, self.params.rho, self.params.beta

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return jnp.array([dx, dy, dz])