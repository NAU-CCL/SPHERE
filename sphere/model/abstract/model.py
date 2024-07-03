"""
This is an abstract base class for all models.
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from sphere.model.abstract.parameters import Parameters
from sphere.model.abstract.solver import Solver
from sphere.model.abstract.transition import Transition


class Model(ABC):

    params: Parameters
    solver: Solver

    def __init__(self, params: Parameters, solver: Solver):
        self.params = params
        self.solver = solver

    def __post_init__(self):
        self.transition = Transition(self.params)
        self.solver.function = self.state_transition

    @abstractmethod
    def state_transition(self, state: ArrayLike, t: int) -> Array:
        pass

    @abstractmethod
    def observation(self, state: ArrayLike, t: int) -> Array:
        pass

    def run(self, x0: ArrayLike, t0: int, t_final: int) -> Array:
        """Run the model from t0 to t_final.

        The run method is implemented explicitly as its logic is fairly simple, essentially looping
        over the model transition method for the specified number of time steps.

        Args:
            x0: The initial state of the system at time t0, a JAX or NumPy Array, used in func.
            t0: The initial time step.
            t_final: The final time step.

        Returns:
            A JAX Array representing the state of the system across all the
            time steps specified by the user. The shape of the return array should
            have the shape (time_steps, N), where N represents the number of state
            variables.

        Raises:
            Raises no exceptions, enforcement of correct arguments should be enforced at
            model creation.
        """
        x0 = jnp.array(x0)
        num_steps = int((t_final - t0) / self.solver.delta_t) + 1
        x_t = x0

        t = t0

        # Initialize an array to store the states at each time step
        trajectory = jnp.zeros((num_steps, *x0.shape))
        trajectory = trajectory.at[0].set(x0)

        for i in range(1, num_steps):
            x_t = self.solver.solve_one_step(x_t, t)
            t += self.solver.delta_t
            trajectory = trajectory.at[i].set(x_t)

        return trajectory