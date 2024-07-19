"""
This is an abstract base class for all models.
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from sphere.model.transition import Transition
from sphere.model.parameters import Parameters
from sphere.model.solver import Solver


class Model(ABC):

    def __init__(self, params: Parameters, solver: Solver):
        self.params = params
        self.solver = solver
        self.transition: Transition

    @abstractmethod
    def initialize_transition(self):
        """This function will assign a transition object to a model.

        Example:
            An SIRModel would grab the SIRTransition.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def observation(self, state: ArrayLike, t: int) -> Array:
        pass

    def run(self, x0: ArrayLike, t0: int, t_final: int, dt: float) -> Array:
        """Run the model from t0 to t_final.

        The run method is implemented explicitly as its logic is fairly simple, essentially looping
        over the model transition method for the specified number of time steps.

        Args:
            x0: The initial state of the system at time t0, a JAX or NumPy Array, used in func.
            t0: The initial time step.
            t_final: The final time step.

        Returns:
            A JAX Array representing the state of the system across all the
            time steps specified by the user. The shape of the return array should
            have the shape (time_steps, N), where N represents the number of state
            variables.

        Raises:
            Raises no exceptions, enforcement of correct arguments should be enforced at
            model creation.
        """
        x0 = jnp.array(x0)

        num_output_points = int((t_final - t0)) + 1
        trajectory = jnp.zeros((num_output_points, *x0.shape))
        output_times = jnp.arange(t0, t_final + 1)  # Integer time steps

        state = x0
        current_time = t0
        output_index = 0

        # Collect the initial state
        trajectory[output_index] = state

        for target_time in output_times[1:]:
            while current_time < target_time:
                state = self.solver.step(state, self.transition)
                current_time += self.solver.delta_t
            trajectory[output_index] = state
            output_index += 1

        return trajectory
