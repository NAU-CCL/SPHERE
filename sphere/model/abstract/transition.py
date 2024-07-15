from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax.numpy as jnp

from sphere.model.abstract.parameters import Parameters


class Transition(ABC):
    """
    A base class for defining state transition functions.
    """
    def __init__(self, params: Parameters, is_stochastic: bool = False) -> \
            None:
        self.params = params


class DeterministicTransition(Transition):
    @abstractmethod
    def function(self):
        pass


class StochasticTransition(Transition):
    def drift(self):
        pass

    def diffusion(self):
        pass


class SIRTransition(DeterministicTransition):
    def __init__(self, params: Parameters) -> None:
        super().__init__(params=params)

    def function(self, state: jnp.ndarray, t: int) -> jnp.ndarray:
        S, I, R = state
        beta, gamma = self.params.beta, self.params.gamma

        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I

        return jnp.array([dS, dI, dR])


class Lorenz63Transition(DeterministicTransition):
    def __init__(self, params: Parameters) -> None:
        super().__init__(params=params)

    def function(self, state: jnp.ndarray, t: int) -> jnp.ndarray:
        x, y, z = state
        sigma, rho, beta = self.params.sigma, self.params.rho, self.params.beta

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return jnp.array([dx, dy, dz])
