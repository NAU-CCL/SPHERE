import jax.numpy as jnp

from sphere.model.model import Model
from sphere.model.parameters import LorenzParameters
from sphere.model.solvers import ODESolver


class LorenzModel(Model):
    def __init__(self, params: LorenzParameters, solver: ODESolver):
        super().__init__(params, solver)

    def state_transition(self, state: jnp.ndarray, t) -> jnp.ndarray:
        x, y, z = state
        sigma, rho, beta = self.params.sigma, self.params.rho, self.params.beta

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        new_state = jnp.array([dx, dy, dz])
        return new_state

    def observation(self, state: jnp.ndarray) -> jnp.ndarray:
        return state
