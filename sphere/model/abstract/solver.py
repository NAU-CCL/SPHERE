"""
An abstract base class for solving a system through a given time step. 
"""

from abc import ABC, abstractmethod
from typing import Callable,Generator

from jax.typing import ArrayLike
from jax import Array

class Solver(ABC):

    """
    Abstract base class providing an interface for calling one-step solvers for ODEs, SDEs, etc. 
    
    Implements a single method which is used in the Transition object when solving the system 
    for one discrete time step. Hidden Markov Models are observed through a map 
    at discrete time intervals, hence the motivation for single stepping, even if the 
    underlying system is continuous. 

    """

    @abstractmethod
    def solve(
        self,
        func: Callable[[ArrayLike,float],Array],
        x_t: ArrayLike,
        t: int
    ) -> Array:       
        """Solves the system described by func for a single discrete time step.

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



    

