from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax

from sphere.model.parameters import Parameters
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
    def drift(self):
        raise NotImplementedError("Subclass must implement this method")

    def step(self, state, dt):
        drift = self.drift(state) * dt
        return state + drift


class StochasticTransition(Transition):
    def __init__(self, params: Parameters, solver: Solver, key: KeyArray) -> None:
        super().__init__(params, solver)
        self.key = key

    @abstractmethod
    def drift(self, state):
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def diffusion(self, state):
        raise NotImplementedError("Subclass must implement this method")

    def step(self, state, dt):
        drift = self.drift(state) * dt
        diffusion = (self.diffusion(state) * jnp.sqrt(dt) *
                     jax.random.normal(key=self.key, shape=3))
        return state + drift + diffusion


class DeterministicSIR(DeterministicTransition):
    def __init__(self, params: Parameters, solver: Solver) -> None:
        super().__init__(params=params, solver=solver)

    def function(self, state: jnp.ndarray, t: int) -> jnp.ndarray:
        S, I, R = state
        beta, gamma = self.params.beta, self.params.gamma

        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I

        return jnp.array([dS, dI, dR])


class StochasticSIR(StochasticTransition):
    def __init__(self, params: Parameters, solver: Solver) -> None:
        super().__init__(params=params, solver=solver)

    def drift(self, state):
        S, I, R = state
        dS = -self.params.beta * S * I / self.N
        dI = self.beta * S * I / self.N - self.gamma * I
        dR = self.gamma * I
        return np.array([dS, dI, dR])

    def diffusion(self, state):
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
