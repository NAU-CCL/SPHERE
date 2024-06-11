"""
An abstract base class and concrete implementations of ODE solvers.
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple

import jax.numpy as jnp
from jax.experimental.ode import odeint


class ODESolver(ABC):
    @abstractmethod
    def solve(
        self,
        func: Callable,
        y0: Tuple,
        t_span: Tuple[float, float],
        t_eval: jnp.ndarray,
    ) -> jnp.ndarray:
        pass


class JAXSolver(ODESolver):
    def solve(
        self,
        func: Callable,
        y0: jnp.ndarray,
        t_span: Tuple[float, float],
        t_eval: jnp.ndarray,
    ) -> jnp.ndarray:

        sol = odeint(func, y0, t_eval)
        return jnp.asarray(sol)
