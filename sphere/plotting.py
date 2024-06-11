import matplotlib.pyplot as plt
from jax.typing import ArrayLike


def plot_lorenz_output(states: ArrayLike) -> None:
    """Displays the output of a Lorenz model on a 3D plot."""
    x, y, z = states

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Lorenz System Solution')
    plt.show()