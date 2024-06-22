"""A simple implementation of a solver using fixed tau leaping. """

from typing import Callable

from jax.typing import ArrayLike, DTypeLike
from jax import Array

from jax import random

from sphere.model.abstract.solver import Solver


class TauLeapSolver(Solver):
    """
        An implementation of solver, uses the approximate stochastic method of 
        tau-leaping described by Gillespie et. al. 

        Takes two additional arguments given by tau, analogous to delta_t in the one
        step Euler method, and PRNG_key, the key to pass to JAX random for the poisson draws. 
    """

    tau: float
    prng_key: Array

    def __init__(self, tau: float, prng_key: Array) -> None:
        super().__init__()

        if(tau <= 0):
            raise ValueError(f"Tau must be greater than zero! Tau was {tau}. ")

        self.tau = tau
        self.prng_key = prng_key

    def solve(
        self,
        func: Callable[[ArrayLike, float], Array],
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

        return x_t + random.poisson(key=self.prng_key,
                                    lam=self.tau * func(x_t, t))