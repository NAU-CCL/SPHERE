import jax.numpy as jnp
from sphere.model.solver import DeterministicSolver, StochasticSolver
from sphere.model.transition import (
    StochasticTransition,
    DeterministicTransition,
    StochasticSIR,
    DeterministicSIR,
    Lorenz63Transition,
)
from sphere.model.parameters import SIRParameters, Lorenz63Parameters

from jax.typing import ArrayLike
from jax import Array


class Model:
    def __init__(self, solver, output):
        self.solver = solver
        self.output = output

    def run(self, x0: ArrayLike, t0: int, t_final: int, dt: float) -> None:
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
        self.output.store(state)

        # We run for one discrete time step: t to t+1.
        # Depending on dt, this may involve intermediary updates.
        for target_time in output_times[1:]:
            while current_time < target_time:
                state = self.solver.step(state=state, dt=dt, t=current_time)
                current_time += dt
            self.output.store(state)
            output_index += 1

        print(
            "Model.run() was successful. Data is accessible at "
            "Model.output.states. Plot the output with "
            "Model.output.plot_states()."
        )

    @staticmethod
    def _validate_dt(delta_t: float):
        if delta_t <= 0:
            raise ValueError(
                f"Delta_t must be greater than zero! Delta_t was {delta_t}."
            )
