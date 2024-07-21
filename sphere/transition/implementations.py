from jax import numpy as jnp
from jax._src.basearray import ArrayLike

from sphere.parameters.parameters import Parameters, SIRParameters
from sphere.transition.abstract import (DeterministicTransition,
                                        StochasticTransition)


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

    def diffusion(self, state: ArrayLike, t: float) -> jnp.ndarray:
        S, I, R = state
        N = self.params.population
        beta = self.params.beta
        gamma = self.params.gamma

        # Noise terms for each compartment
        dS = jnp.sqrt(beta * S * I / N)
        dI = jnp.sqrt(beta * S * I / N + gamma * I)
        dR = jnp.sqrt(gamma * I)

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
