from matplotlib import pyplot as plt

from sphere.output.abstract.output import Output
import jax.numpy as jnp


class LorenzOutput(Output):
    def plot_states(
        self, save: bool = False, filename: str = "lorenz_model_plot.png"
    ) -> None:
        """Displays the output of a Lorenz model on a 3D plot.

        Args:
            save: If `True`, saves plot to png.
            filename: The filename to save the plot to.
        """
        self.states = jnp.transpose(self.states)
        x, y, z = self.states

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x, y, z)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("Lorenz System Solution")

        if save:
            plt.savefig(filename)
        else:
            plt.show()
