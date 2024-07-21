from abc import ABC, abstractmethod

from jax import Array
from jax._src.basearray import ArrayLike

from sphere.parameters.parameters import Parameters

KeyArray = Array  # type checking for key, as done in jax source code


class Transition(ABC):
    """
    A base class for defining state transition functions.
    """

    def __init__(self, params: Parameters) -> None:
        self.params = params


class DeterministicTransition(Transition):
    @abstractmethod
    def drift(self, state: ArrayLike, t: float):
        raise NotImplementedError("Subclass must implement this method")


class StochasticTransition(Transition):
    def __init__(self, params: Parameters) -> None:
        super().__init__(params)

    @abstractmethod
    def drift(self, state: ArrayLike, t: float):
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def diffusion(self, state):
        raise NotImplementedError("Subclass must implement this method")
