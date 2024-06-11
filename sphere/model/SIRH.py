import jax.numpy as jnp

from sphere.model.model import Model
from sphere.model.parameters import SIRHParameters
from sphere.model.solvers import ODESolver


class SIRHModel(Model):
    """
    An SIRH model, where a hospitalization (H) compartment is added to
    the standard SIR model.
    """
    def __init__(self, params: SIRHParameters, solver: ODESolver):
        super().__init__(params, solver)

    def state_transition(self, state: jnp.ndarray, t: int) -> jnp.ndarray:
        S, I, R, H = state

        beta, gamma, delta, mu = (
            self.params.beta,
            self.params.gamma,
            self.params.delta,
            self.params.mu,
        )

        dS = -beta * S * I
        dI = beta * S * I - gamma * I - delta * I
        dR = gamma * I
        dH = delta * I - mu * H

        return jnp.array([dS, dI, dR, dH])

    def observation(self, state: jnp.ndarray) -> jnp.ndarray:
        return state[3] # observe hospitalizations
