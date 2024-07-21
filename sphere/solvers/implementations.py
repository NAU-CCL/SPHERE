from jax import Array, numpy as jnp, random
from jax._src.basearray import ArrayLike

from sphere.solvers.abstract import DeterministicSolver, StochasticSolver


class EulerSolver(DeterministicSolver):
    def step(self, state: ArrayLike, dt: float, t: float) -> Array:
        """
        Advances the state of the model by one time step using the Euler method.

        Args:
            state: The current state of the system.
            dt: Time increment for the step.
            t: The current time step.

        Returns:
            jax.Array: The updated state of the system.
        """
        drift = self.transition.drift(state, t) * dt
        return state + drift


class EulerMaruyamaSolver(StochasticSolver):

    def step(self, state: ArrayLike, dt: float, t: float) -> Array:
        """
        Advances the state of the model by one time step using the
        Euler-Maruyama method.

        Args:
            state: The current state of the system.
            dt: Time increment for the step.
            t: The current time step.

        Returns:
            jax.Array: The updated state of the system.
        """
        drift = self.transition.drift(state, t) * dt
        self.key, subkey = random.split(self.key)
        diffusion = (
            self.transition.diffusion(state, t)
            * jnp.sqrt(dt)
            * random.normal(shape=state.shape, key=subkey) * 0.0001
            # TODO: Parameterize this noise in a principled manner.
            #  Remove the `* 0.0001`.
        )
        return state + drift + diffusion
