import jax.numpy as jnp

from sphere_old.model.abstract.model import Model
from sphere_old.model.implementations.solvers._solvers import ODESolver
from sphere.model.parameters import LorenzParameters


class LorenzModel(Model):
    """
    A model for the Lorenz 63 System:
    https://en.wikipedia.org/wiki/Lorenz_system
    """

    def __init__(self, params: LorenzParameters, solver: ODESolver):
        super().__init__(params, solver)

    def state_transition(self, state: jnp.ndarray, t: int) -> jnp.ndarray:
        x, y, z = state
        sigma, rho, beta = self.params.sigma, self.params.rho, self.params.beta

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return jnp.array([dx, dy, dz])

    def observation(self, state: jnp.ndarray, t: int) -> jnp.ndarray:
        return state
