import jax.numpy as jnp

from jax.typing import ArrayLike


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

        # Calculate the total number of steps
        total_steps = int((t_final - t0) / dt) + 1

        # Initialize the output storage
        self.output.states = jnp.zeros((total_steps, *x0.shape))

        # Initialize state and time
        state = x0
        current_time = t0
        step_index = 0

        # Store the initial state
        self.output.store(state, step_index)

        # Run the model
        for _ in range(1, total_steps):
            state = self.solver.step(state=state, dt=dt, t=current_time)
            current_time += dt
            step_index += 1
            self.output.store(state, step_index)

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
