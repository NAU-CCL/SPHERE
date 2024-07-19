"""
An abstract base class for solving a system through a given time step. 
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array, random
from jax.typing import ArrayLike

from sphere.model.transition import DeterministicTransition, Transition


def validate_input(delta_t: float) -> None:
    if delta_t <= 0:
        raise ValueError(
            f"Delta_t must be greater than zero! Delta_t " f"was {delta_t}."
        )


class Solver(ABC):
    """
    Abstract base class providing an interface for calling one-step solvers for ODEs, SDEs, etc.

    Implements a single method which is used in the Transition object when solving the system
    for one discrete time step.
    """

    def __init__(self, transition: Transition) -> None:
        self.transition = transition

    @abstractmethod
    def step(self, state: ArrayLike, dt: int) -> Array:
        """Solves the system described by func for a single discrete time step.

        Args:
            state: The state of the system at time t, a JAX or NumPy Array, used in func.
            dt: Time increment for the step.
        Returns:
            The state of the system at time t+1, a JAX Array.
        """
        raise NotImplementedError("Subclass must implement this method.")

    @staticmethod
    def _validate_input(self, delta_t: float):
        if delta_t <= 0:
            raise ValueError(
                f"Delta_t must be greater than zero! Delta_t was {delta_t}."
            )

    @abstractmethod
    def is_stochastic(self) -> bool:
        raise NotImplementedError("Subclass must implement this method.")


class DeterministicSolver(Solver):
    pass


class StochasticSolver(Solver):
    pass


class EulerSolver(DeterministicSolver):

    def step(self, state, dt, t) -> Array:
        """
        Advances the state of the model by one time step using the Euler method.

        Args:
            model: The model that provides the drift (and optionally diffusion) functions.
            state: The current state of the system.
            dt: The time step size.
            t: The current time step.

        Returns:
            jax.Array: The updated state of the system.
        """
        drift = self.transition.drift(state, t) * dt
        return state + drift


class EulerMaruyamaSolver(StochasticSolver):

    def step(self, transition: Transition, state: ArrayLike, dt: float) -> Array:
        drift = self.transition.drift(state) * dt
        diffusion = (
            self.transition.diffusion(state)
            * jnp.sqrt(dt)
            * random.normal(size=state.shape)
        )
        return state + drift + diffusion
