"""A simple implementation of a solver using one-step Euler. """

from typing import Callable

from jax.typing import ArrayLike
from jax import Array

from sphere.model.abstract.solver import Solver

class EulerSolver(Solver): 
    """
        An implementation of solver, takes an additional argument to fully specify the 
        Euler step, delta_t. 
    """

    delta_t:float

    def __init__(self,delta_t:float) -> None:
        super().__init__()

        if(delta_t <= 0):
            raise ValueError(f"Delta_t must be greater than zero! Delta_t was {delta_t}")

        self.delta_t = delta_t


    def solve(
    self,
    func: Callable[[ArrayLike,float],Array],
    x_t: ArrayLike,
    t: int
) -> Array:  
        """Solves the system described by func for a single discrete time step
        using one-step Euler.

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

        return x_t + self.delta_t * func(x_t,t)     
