from jax import numpy as jnp
from jax.typing import ArrayLike
from matplotlib import pyplot as plt

from sphere.output.abstract import Output


class LorenzOutput(Output):
    """Displays the output of a Lorenz model on a 3D plot."""
    def plot_states(self) -> None:
        x, y, z = self.states

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Lorenz System Solution')
        plt.show()


class SIROutput(Output):
    def plot_states(self):
        states = jnp.array(self.states)
        fig, ax = plt.subplots()
        ax.plot(states[:, 0], label='S')
        ax.plot(states[:, 1], label='I')
        ax.plot(states[:, 2], label='R')
        ax.set_xlabel('Time')
        ax.set_ylabel('Population')
        ax.legend()
        plt.show()
