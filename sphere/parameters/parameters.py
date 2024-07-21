"""
The model parameters. An abstract base class is defined.
Each concrete subclass defines that model's unique parameters.
"""

from abc import ABC, abstractmethod

from sphere.parameters.functional_params import FunctionalParam, ConstantParam


class Parameters(ABC):
    """Abstract model parameters class."""

    @abstractmethod
    def update_all(self, t: int):
        raise NotImplementedError("Subclasses must implement this method.")


class SIRParameters(Parameters):
    def __init__(
        self,
        beta: FunctionalParam | float,
        gamma: FunctionalParam | float,
        population: int,
    ) -> None:
        self.beta_param = self._wrap_param(beta)
        self.gamma_param = self._wrap_param(gamma)
        self.population = population
        self.beta = self.beta_param.get_current_value(0)  # Initialize current values
        self.gamma = self.gamma_param.get_current_value(0)

    def _wrap_param(self, param) -> FunctionalParam:
        """Wrap a parameter in a FunctionalParam if it's not already."""
        if isinstance(param, FunctionalParam):
            return param
        elif isinstance(param, float):
            return ConstantParam(param)
        else:
            raise TypeError("Parameter must be either a FunctionalParam or a float.")

    def update_all(self, t: int) -> None:
        """Update all parameters based on the current time step."""
        self.beta = self.beta_param.get_current_value(t)
        self.gamma = self.gamma_param.get_current_value(t)


class Lorenz63Parameters(Parameters):
    def __init__(
        self, sigma: FunctionalParam, rho: FunctionalParam, beta: FunctionalParam
    ) -> None:
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def update_all(self, t: int) -> None:
        pass
