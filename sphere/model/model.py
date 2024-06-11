"""
This is an abstract base class for all models.
"""

import jax.numpy as jnp


from abc import ABC, abstractmethod
from sphere.model.parameters import Parameters
from sphere.model.solvers import ODESolver
from jax.typing import ArrayLike

class Model(ABC):
    def __init__(self, params: Parameters, solver: ODESolver):
        self.params = params
        self.solver = solver

    @abstractmethod
    def state_transition(self, state: jnp.ndarray, t: int) -> jnp.ndarray:
        pass

    @abstractmethod
    def observation(self, state: jnp.ndarray) -> jnp.ndarray:
        pass

    def run(self, time_steps: int) -> jnp.ndarray:
        """
        Run the model for the specified number of time steps.

        Return: An array of the system states at each time point.
        """
        t_span = (0, time_steps)
        t_eval = jnp.linspace(t_span[0], t_span[1], 10000)
        sol = self.solver.solve(
            self.state_transition, self.params.initial_state, t_span, t_eval
        )
        return sol
