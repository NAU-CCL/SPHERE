"""A simple implementation of a solver using one-step Euler. """

from typing import Callable, Dict

from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp

from sphere.model.abstract.solver import Solver
from sphere.model.abstract.model import Model


class EulerSolver(Solver):
    """
        An implementation of solver, takes an additional argument to fully specify the 
        Euler step, delta_t. 
    """

    delta_t: float

    req_keys = ['func']

    def __init__(self, delta_t: float) -> None:
        super().__init__(delta_t=delta_t)

    def solve_one_step(self, x_t: ArrayLike, t: int, function: Callable) -> (
            Array):
        """Solves the system described by func for a single discrete time step
        using one-step Euler.

        Args:
            function:
            x_t: The state of the system at time t, a JAX or NumPy Array, used in func.
            t: The current discrete time step of the system, discretization schemes are left 
            up to the user, used in func. 
        Returns:
            The state of the system at time t+1, a JAX Array. Note, regardless of whether x_t was a
            JAX or NumPy Array, the return will always be a JAX Array.
        """
        return x_t + self.delta_t * function(x_t, t)

    def solve(self, x0: ArrayLike, t0: int, t_final: int) -> jnp.ndarray:
        """Solves the system over multiple time steps using the Euler method and returns
        the state at each time step.

        Args:
            x0: The initial state of the system at time t0, a JAX or NumPy Array, used in func.
            t0: The initial time step.
            t_final: The final time step.
        Returns:
            An array containing the state of the system at each time step, a JAX Array.
        """
        num_steps = int((t_final - t0) / self.delta_t) + 1
        x_t = x0
        t = t0

        # Initialize an array to store the states at each time step
        trajectory = jnp.zeros((num_steps, *x0.shape))
        trajectory = trajectory.at[0].set(x0)

        for i in range(1, num_steps):
            x_t = self.solve_one_step(x_t, t, The)
            t += self.delta_t
            trajectory = trajectory.at[i].set(x_t)

        return trajectory
