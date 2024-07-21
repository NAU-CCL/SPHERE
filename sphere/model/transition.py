from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import random, Array
from jax.typing import ArrayLike

from sphere.model.parameters import Parameters, SIRParameters

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
    def __init__(self, params: Parameters, key: KeyArray) -> None:
        super().__init__(params)
        self.key = key

    @abstractmethod
    def drift(self, state: ArrayLike, t: float):
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def diffusion(self, state):
        raise NotImplementedError("Subclass must implement this method")


class DeterministicSIR(DeterministicTransition):
    def __init__(self, params: SIRParameters) -> None:
        super().__init__(params)

    def drift(self, state: ArrayLike, t: float) -> jnp.ndarray:
        self.params.update_all(t)
        beta = self.params.beta
        gamma = self.params.gamma
        N = self.params.population

        S, I, R = state
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return jnp.array([dS, dI, dR])


class StochasticSIR(StochasticTransition):
    def __init__(self, params: Parameters) -> None:
        super().__init__(params=params)

    def drift(self, state: ArrayLike, t: float) -> jnp.ndarray:
        self.params.update_all(t)
        beta = self.params.beta
        gamma = self.params.gamma
        N = self.params.population

        S, I, R = state
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return jnp.array([dS, dI, dR])

    def diffusion(self, state: ArrayLike, t: float):
        S, I, R = state
        dS = jnp.sqrt(self.params.beta * S * I / self.params.population)
        dI = jnp.sqrt(self.params.beta * S * I / self.params.population +
                      self.params.gamma * I)
        dR = jnp.sqrt(self.params.gamma * I)
        return jnp.array([dS, dI, dR])


class Lorenz63Transition(DeterministicTransition):
    def __init__(self, params: Parameters) -> None:
        super().__init__(params=params)

    def drift(self, state: jnp.ndarray, t: float) -> jnp.ndarray:
        x, y, z = state
        sigma, rho, beta = self.params.sigma, self.params.rho, self.params.beta

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return jnp.array([dx, dy, dz])
