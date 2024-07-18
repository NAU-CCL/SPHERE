"""
An abstract base class for solving a system through a given time step. 
"""

from abc import ABC, abstractmethod
from typing import Dict, Callable, List

from jax.typing import ArrayLike
from jax import Array

from sphere.model.abstract.transition import Transition, SIRTransition


def validate_input(delta_t: float) -> None:
    if delta_t <= 0:
        raise ValueError(f"Delta_t must be greater than zero! Delta_t "
                         f"was {delta_t}.")


class Solver(ABC):
    """
    Abstract base class providing an interface for calling one-step solvers for ODEs, SDEs, etc. 
    
    Implements a single method which is used in the Transition object when solving the system 
    for one discrete time step.
    """
    def __init__(self, delta_t: float, transition: Transition) -> None:
        validate_input(delta_t)
        self.delta_t = delta_t
        self.transition = transition  # initialized in Model.post_init()

    @abstractmethod
    def step(self, state: ArrayLike, dt: int) -> Array:
        """Solves the system described by func for a single discrete time step.

        Args:
            state: The state of the system at time t, a JAX or NumPy Array, used in func.
            dt: Time increment for the step.
        Returns:
            The state of the system at time t+1, a JAX Array.
        """
        pass

    @staticmethod
    def _validate_input(self, delta_t: float):
        if delta_t <= 0:
            raise ValueError(f"Delta_t must be greater than zero! Delta_t was {delta_t}.")


class EulerMaruyamaSolver(Solver):
    def step(self, transition, state, dt):
        drift = transition.drift(state) * dt
        diffusion = transition.diffusion(state) * jnp.sqrt(dt) * jax.random.normal(size=state.shape)
        return state + drift + diffusion
