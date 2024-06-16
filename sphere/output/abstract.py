from abc import ABC, abstractmethod


class Output(ABC):
    def __init__(self):
        self.states = []

    def store(self, state):
        self.states.append(state)

    @abstractmethod
    def plot_states(self):
        pass


