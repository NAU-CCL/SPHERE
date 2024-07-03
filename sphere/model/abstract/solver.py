"""
An abstract base class for solving a system through a given time step. 
"""

from abc import ABC, abstractmethod
from typing import Dict, Callable, List

from jax.typing import ArrayLike
from jax import Array

from sphere.model.abstract.transition import Transition, SIRTransition


def validate_input(delta_t: float) -> None:
    if delta_t <= 0:
        raise ValueError(f"Delta_t must be greater than zero! Delta_t "
                         f"was {delta_t}.")


class Solver(ABC):
    """
    Abstract base class providing an interface for calling one-step solvers for ODEs, SDEs, etc. 
    
    Implements a single method which is used in the Transition object when solving the system 
    for one discrete time step. Hidden Markov Models are observed through a map 
    at discrete time intervals, hence the motivation for single stepping, even if the 
    underlying system is continuous. 

    """
    req_keys: List[str]

    def __init__(self, delta_t: float, transition: Transition) -> None:
        validate_input(delta_t)
        self.delta_t = delta_t
        self.transition = transition  # initialized in Model.post_init()

    @abstractmethod
    def solve_one_step(
            self,
            x_t: ArrayLike,
            t: int
    ) -> Array:
        """Solves the system described by func for a single discrete time step.

        Args:
            function:
            x_t: The state of the system at time t, a JAX or NumPy Array, used in func.
            t: The current discrete time step of the system, discretization schemes are left 
            up to the user, used in func. 
        Returns:
            The state of the system at time t+1, a JAX Array. Note, regardless of whether x_t was a
            JAX or NumPy Array, the return will always be a JAX Array.
        """
