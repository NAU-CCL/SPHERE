"""A simple implementation of a solver using fixed tau leaping. """

from typing import Callable,Dict

from jax.typing import ArrayLike
from jax import Array

from jax import random

import jax.numpy as jnp

from sphere.model.abstract.solver import Solver


class EulerMaruyamaSolver(Solver):
    """
        An implementation of solver, using the Euler-Maruyama method for SDEs. 

        Takes two additional arguments given by delta_t, the euler time-step, 
        and PRNG_key, the key to pass to JAX random for the Brownian draws. 
    """

    delta_t: float

    prng_key: Array

    req_keys = ['drift','diffusion']

    def __init__(self, delta_t: float, prng_key: Array, args: Dict[str,Callable]) -> None:
        super().__init__(args = args)

        self.delta_t = delta_t
        self.prng_key = prng_key

    def solve(
        self,
        x_t: ArrayLike,
        t: int
    ) -> Array:
        """Solves the system described by func for a single discrete time step
        using tau leaping. 

        Args:
            func: A function describing the transition rule for the system of interest, arguments 
            are (x_t, t). Return type is a JAX Array of the same shape as x_t. 
            x_t: The state of the system at time t, a JAX or NumPy Array, used in func. 
            t: The current discrete time step of the system, discretization schemes are left 
            up to the user, used in func. 
        Returns:
            The state of the system at time t+1, a JAX Array. Note, regardless of whether x_t was a
            JAX or NumPy Array, the return will always be a JAX Array.

        """

        return x_t + self.args['drift'](x_t,t) * self.delta_t + \
        self.args['diffusion'](x_t,t) * jnp.sqrt(self.delta_t) * \
        random.normal(key = self.prng_key,shape = x_t.shape)