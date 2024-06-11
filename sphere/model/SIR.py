import jax.numpy as jnp

from sphere.model.model import Model
from sphere.model.parameters import SIRParameters
from sphere.model.solvers import ODESolver


class SIRModel(Model):
    def __init__(self, params: SIRParameters, solver: ODESolver):
        super().__init__(params, solver)

    def state_transition(self, state: jnp.ndarray, t) -> jnp.ndarray:
        S, I, R = state
        beta, gamma = self.params.beta, self.params.gamma

        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I

        return jnp.array([dS, dI, dR])

    def observation(self, state: jnp.ndarray) -> jnp.ndarray:
        return state