from jax import numpy as jnp
from matplotlib import pyplot as plt

from sphere.output.abstract import Output
from jax import Array

import numpy as np


class SIROutput(Output):
    def plot_states(
        self, save: bool = False, filename: str = "sir_model_plot.png"
    ) -> None:
        """Displays a time series plot for the system state.

        Args:
           save: If `True`, saves plot to png.
           filename: The filename to save the plot to.
        """
        if not isinstance(self.states, Array):
            raise TypeError("plot_states() expects the output states to be of type Jax.Array")

        states = np.array(self.states)
        fig, ax = plt.subplots()
        ax.plot(states[:, 0], label="S")
        ax.plot(states[:, 1], label="I")
        ax.plot(states[:, 2], label="R")
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.legend()
        plt.title("Time Evolution of SIR Model Compartments")

        if save:
            plt.savefig(filename)
        else:
            plt.show()
