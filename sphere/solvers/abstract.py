from abc import ABC, abstractmethod

from jax import Array
from jax._src.basearray import ArrayLike
from jax.random import PRNGKey

from sphere.transition.abstract import Transition

KeyArray = Array


class Solver(ABC):
    """
    Abstract base class providing an interface for calling one-step solvers for ODEs, SDEs, etc.

    Implements a single method which is used in the Model class when solving
    the system for one discrete time step.
    """

    def __init__(self, transition: Transition) -> None:
        self.transition = transition

    @abstractmethod
    def step(self, state: ArrayLike, dt: float, t: float) -> Array:
        """Solves the system described by self.transition for a single
        discrete time step.

        Args:
            state: The state of the system at time t, a JAX or NumPy Array, used in func.
            dt: Time increment for the step.
            t: Current discrete time step.

        Returns:
            jax.Array: The updated state of the system.
        """
        raise NotImplementedError("Subclass must implement this method.")


class DeterministicSolver(Solver, ABC):
    pass


class StochasticSolver(Solver, ABC):
    def __init__(self, transition: Transition) -> None:
        super().__init__(transition)
        self.key = PRNGKey(0)
