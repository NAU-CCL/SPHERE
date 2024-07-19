import jax.numpy as jnp
from sphere.model.solver import DeterministicSolver, StochasticSolver
from sphere.model.transition import StochasticTransition, DeterministicTransition, StochasticSIR, DeterministicSIR, Lorenz63Transition
from sphere.model.parameters import SIRParameters, Lorenz63Parameters

from jax.typing import ArrayLike
from jax import Array


class Model:
    def __init__(self, params, solver, output):
        self.params = params
        self.solver = solver
        self.output = output
        self._validate_model_solver_compatibility()
        self._validate_params()

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
        self._validate_dt(dt)
        x0 = jnp.array(x0)

        num_output_points = int((t_final - t0)) + 1
        trajectory = jnp.zeros((num_output_points, *x0.shape))
        output_times = jnp.arange(t0, t_final + 1)

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

    @staticmethod
    def _validate_dt(delta_t: float):
        if delta_t <= 0:
            raise ValueError(
                f"Delta_t must be greater than zero! Delta_t was {delta_t}."
            )
        
    def _validate_model_solver_compatibility(self):
        if isinstance(self.solver, DeterministicSolver) and isinstance(
            self.solver.transition, StochasticTransition
        ):
            raise TypeError(
                "Cannot use a deterministic solver with a stochastic transition."
            )
        if isinstance(self.solver, StochasticSolver) and isinstance(
            self.solver.transition, DeterministicTransition
        ):
            raise TypeError(
                "Cannot use a stochastic solver with a deterministic transition."
            )

    def _validate_params(self):
        if isinstance(self.solver.transition, (DeterministicSIR, StochasticSIR)) and not isinstance(
            self.params, SIRParameters
        ):
            raise TypeError(
                "Parameters must be of type SIRParameters for SIRTransition."
            )
        if isinstance(self.solver.transition, Lorenz63Transition) and not isinstance(
            self.params, Lorenz63Parameters
        ):
            raise TypeError(
                "Parameters must be of type Lorenz63Parameters for Lorenz63Transition."
            )
