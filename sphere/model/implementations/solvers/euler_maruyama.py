"""A simple implementation of a solver using fixed tau leaping. """

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from sphere.model.abstract.transition import Transition
from sphere.model.solver import Solver


class EulerMaruyamaSolver(Solver):
    """
    An implementation of solver, using the Euler-Maruyama method for SDEs.

    Takes two additional arguments given by delta_t, the euler time-step,
    and PRNG_key, the key to pass to JAX random for the Brownian draws.
    """

    def __init__(self, delta_t: float, transition: Transition, prng_key: Array) -> None:
        super().__init__(delta_t=delta_t, transition=transition)
        self.prng_key = prng_key

    def step(self, x_t: ArrayLike, t: int) -> Array:
        """Solves the system described by func for a single discrete time step
        using tau leaping.

        Args:
            function: A function describing the transition rule for the system
             of interest, arguments are (x_t, t). Return type is a JAX Array of the same shape as x_t.
            x_t: The state of the system at time t, a JAX or NumPy Array, used in func.
            t: The current discrete time step of the system, discretization schemes are left
             up to the user, used in func.
        Returns:
            The state of the system at time t+1, a JAX Array. Note, regardless of whether x_t was a
            JAX or NumPy Array, the return will always be a JAX Array.
        """
        return (
            x_t
            + self.transition.deterministic(x_t, t) * self.delta_t
            + self.transition.stochastic(x_t, t)
            * np.random.normal(scale=jnp.sqrt(self.delta_t), size=x_t.shape)
        )
