from abc import ABC, abstractmethod


class Output(ABC):
    """
    Defines an abstract Output class for storing model output
    and implementing methods for plotting or saving the output.
    """
    def __init__(self):
        """
        Store the model's state at each time point.
        The state at time t is stored at self.states[t].
        The initial state is stored at self.states[0].
        """
        self.states = []

    def store(self, state):
        """Appends a new state to self.states."""
        self.states.append(state)

    @abstractmethod
    def plot_states(self):
        pass


