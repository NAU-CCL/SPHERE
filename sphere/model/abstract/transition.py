from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array, random
from jax.typing import ArrayLike

from sphere.model.parameters import Parameters, SIRParameters
from sphere.model.solver import Solver

KeyArray = jax.Array  # type checking for key, as done in jax source code


class Transition(ABC):
    """
    A base class for defining state transition functions.
    """

    def __init__(self, params: Parameters, solver: Solver) -> None:
        self.params = params
        self.solver = solver

    @abstractmethod
    def step(self, state, dt):
        pass


class DeterministicTransition(Transition):
    @abstractmethod
    def drift(self, state: ArrayLike, t: int):
        raise NotImplementedError("Subclass must implement this method")


class StochasticTransition(Transition):
    def __init__(self, params: Parameters, solver: Solver, key: KeyArray) -> None:
        super().__init__(params, solver)
        self.key = key

    @abstractmethod
    def drift(self, state: ArrayLike, t: int):
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def diffusion(self, state):
        raise NotImplementedError("Subclass must implement this method")

    def step(self, state, dt):
        drift = self.drift(state) * dt
        diffusion = (
            self.diffusion(state)
            * jnp.sqrt(dt)
            * random.normal(key=self.key, shape=3)
        )
        return state + drift + diffusion


class DeterministicSIR(DeterministicTransition):
    def __init__(self, params: SIRParameters, solver: Solver) -> None:
        super().__init__(params, solver)

    def drift(self, state: ArrayLike, t: int) -> jnp.ndarray:
        beta = self.params.beta.get_current_state(t)
        gamma = self.params.gamma.get_current_state(t)
        N = self.params.population

        S, I, R = state
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return jnp.array([dS, dI, dR])


class StochasticSIR(StochasticTransition):
    def __init__(self, params: Parameters, solver: Solver) -> None:
        super().__init__(params=params, solver=solver)

    def drift(self, state: ArrayLike, t: int):
        beta = self.params.beta.get_current_state(t)
        gamma = self.params.gamma.get_current_state(t)
        N = self.params.population

        S, I, R = state
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return jnp.array([dS, dI, dR])

    def diffusion(self, state: ArrayLike, t: int):
        S, I, R = state
        dS = jnp.sqrt(self.beta * S * I / self.N)
        dI = jnp.sqrt(self.beta * S * I / self.N + self.gamma * I)
        dR = jnp.sqrt(self.gamma * I)
        return jnp.array([dS, dI, dR])


class Lorenz63Transition(DeterministicTransition):
    def __init__(self, params: Parameters) -> None:
        super().__init__(params=params)

    def function(self, state: jnp.ndarray, t: int) -> jnp.ndarray:
        x, y, z = state
        sigma, rho, beta = self.params.sigma, self.params.rho, self.params.beta

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return jnp.array([dx, dy, dz])
