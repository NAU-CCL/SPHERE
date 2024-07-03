from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class Transition(ABC):
    """A base class for defining state transition functions.

    """
    @abstractmethod
    def deterministic_func(self, state):
        pass

    def stochastic_func(self, state):
        return None


class SIRTransition(Transition):
    def deterministic_func(self, state: jnp.ndarray, t: int) -> jnp.ndarray:
        S, I, R = state
        beta, gamma = self.params.beta, self.params.gamma

        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I

        return jnp.array([dS, dI, dR])