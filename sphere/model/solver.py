"""
An abstract base class for solving a system through a given time step. 
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array, random
from jax.typing import ArrayLike

from sphere.model.transition import DeterministicTransition, Transition


class Solver(ABC):
    """
    Abstract base class providing an interface for calling one-step solvers for ODEs, SDEs, etc.

    Implements a single method which is used in the Transition object when solving the system
    for one discrete time step.
    """

    def __init__(self, transition: Transition) -> None:
        self.transition = transition

    @abstractmethod
    def step(self, state: ArrayLike, dt: float, t: float) -> Array:
        """Solves the system described by func for a single discrete time step.

        Args:
            state: The state of the system at time t, a JAX or NumPy Array, used in func.
            dt: Time increment for the step.
            t: Current discrete time step.
        Returns:
            jax.Array: The updated state of the system.
        """
        raise NotImplementedError("Subclass must implement this method.")


class DeterministicSolver(Solver, ABC):
    pass


class StochasticSolver(Solver, ABC):
    pass


class EulerSolver(DeterministicSolver):
    def step(self, state: ArrayLike, dt: float, t: float) -> Array:
        """
        Advances the state of the model by one time step using the Euler method.

        Args:
            state: The current state of the system.
            dt: Time increment for the step.
            t: The current time step.

        Returns:
            jax.Array: The updated state of the system.
        """
        drift = self.transition.drift(state, t) * dt
        return state + drift


class EulerMaruyamaSolver(StochasticSolver):

    def step(self, state: ArrayLike, dt: float, t: float) -> Array:
        drift = self.transition.drift(state) * dt
        diffusion = (
            self.transition.diffusion(state)
            * jnp.sqrt(dt)
            * random.normal(size=state.shape)
        )
        return state + drift + diffusion
