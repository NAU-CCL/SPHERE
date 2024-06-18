"""
This is an abstract base class for all models.
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from sphere.model.abstract.parameters import Parameters
from sphere.model.abstract.solver import Solver


class Model(ABC):

    params: Parameters
    solver: Solver

    def __init__(self, params: Parameters, solver: Solver):
        self.params = params
        self.solver = solver

    @abstractmethod
    def state_transition(self, state: ArrayLike, t: int) -> Array:
        pass

    @abstractmethod
    def observation(self, state: ArrayLike, t: int) -> Array:
        pass

    def run(self, time_steps: int) -> Array:
        """Run the model for the specified number of time steps.

        The run method is implemented explicitly as its logic is fairly simple, essentially looping
        over the model transition method for the specified number of timesteps.

        Args:
            time_steps: A python integer representing the number of discrete time steps for
            which to run the model.

        Returns:
            A JAX Array representing the state of the system across all the
            time steps specified by the user. The shape of the return array should
            have the shape (N,time_steps), where N represents the number of state
            variables.

        Raises:
            Raises no exceptions, enforcement of correct arguments should be enforced at
            model creation.
        """

        t_span = (0, time_steps)
        t_eval = jnp.linspace(t_span[0], t_span[1], time_steps)
        sol = self.solver.solve(
            self.state_transition, self.params.initial_state, t_span, t_eval
        )
        return sol
