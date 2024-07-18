"""A simple implementation of a solver using one-step Euler. """

from jax import Array
from jax.typing import ArrayLike

from sphere.model.abstract.transition import Transition
from sphere.model.solver import Solver


class EulerSolver(Solver):
    """
    An implementation of solver, takes an additional argument to fully specify the
    Euler step, delta_t.
    """

    delta_t: float

    req_keys = ["func"]

    def __init__(self, delta_t: float, transition: Transition) -> None:
        super().__init__(delta_t=delta_t, transition=transition)

    def solve_one_step(self, x_t: ArrayLike, t: int) -> Array:
        """Solves the system described by func for a single discrete time step
        using one-step Euler.

        Args:
            function: The RHS function that defines the system.
            x_t: The state of the system at time t, a JAX or NumPy Array, used in func.
            t: The current discrete time step of the system, discretization schemes are left
            up to the user, used in func.
        Returns:
            The state of the system at time t+1, a JAX Array. Note, regardless of whether x_t was a
            JAX or NumPy Array, the return will always be a JAX Array.
        """
        return x_t + self.delta_t * self.transition.deterministic(state=x_t, t=t)
