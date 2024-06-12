from jax import Array
from jax.typing import ArrayLike

import jax.numpy as jnp

from sphere.model.abstract.model import Model
from sphere.model.abstract.parameters import LorenzParameters
from sphere.model.abstract.solvers import ODESolver


class GenModel(Model):
    """
    A model for the Lorenz System:
    https://en.wikipedia.org/wiki/Lorenz_system
    """
    def __init__(self, params: LorenzParameters):
        super().__init__(params,None)

    def state_transition(self,state:ArrayLike,t: int) -> str:
        
        dt = 0.01

        x, y, z = state
        sigma, rho, beta = self.params.sigma, self.params.rho, self.params.beta

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return state + jnp.array([dx, dy, dz]) * dt

    def observation(self, state: jnp.ndarray,t:int) -> jnp.ndarray:
        return state
