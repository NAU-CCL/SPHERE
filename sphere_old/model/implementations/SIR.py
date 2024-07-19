import jax.numpy as jnp

from sphere_old.model.abstract.model import Model
from sphere.model.solver import Solver
from sphere.model.parameters import SIRParameters
from sphere.model.transition import DeterministicSIR


class SIRDeterministicModel(Model):
    """
    A standard SIR compartmental model.
    """

    def __init__(self, params: SIRParameters, solver: Solver) -> None:
        super().__init__(params, solver)

    def initialize_transition(self):
        self.transition = DeterministicSIR(self.params, self.solver)

    def observation(self, state: jnp.ndarray, t: int) -> jnp.ndarray:
        return state


class SIRStochasticModel(Model):
