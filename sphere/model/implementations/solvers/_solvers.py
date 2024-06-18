"""
An abstract base class and concrete implementations of ODE solvers.
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple

import jax.numpy as jnp
from jax import Array
from jax.experimental.ode import odeint
from jax.typing import ArrayLike


class ODESolver(ABC):
    @abstractmethod
    def solve(
        self,
        func: Callable[[ArrayLike, float], ArrayLike],
        y0: ArrayLike,
        t_span: Tuple[float, float],
        t_eval: ArrayLike,
    ) -> Array:
        pass


class JAXSolver(ODESolver):
    """
    JAX-based Dormand-Prince ODE integration with adaptive step size.
    """

    def solve(
        self,
        func: Callable,
        y0: ArrayLike,
        t_span: Tuple[float, float],
        t_eval: ArrayLike,
    ) -> jnp.ndarray:

        sol = odeint(func, y0, t_eval)
        return jnp.asarray(sol)
