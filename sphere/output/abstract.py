from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax.typing import ArrayLike


class Output(ABC):
    """
    Defines an abstract Output class for storing model output
    and implementing methods for plotting or saving the output.
    """

    def __init__(self) -> None:
        """
        Store the model's state at each time point.
        The state at time t is stored at self.states[t].
        The initial state is stored at self.states[0].
        """
        self.states = jnp.zeros(0)

    def store(self, state: ArrayLike, t: int) -> None:
        """Appends a new state to self.states."""
        self.states = self.states.at[t].set(state)

    @abstractmethod
    def plot_states(self) -> None:
        """Plots the model output. Plot is specific to each model."""
        pass
