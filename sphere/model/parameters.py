"""
The model parameters. An abstract base class is defined.
Each concrete subclass defines that model's unique parameters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sphere.model.functional_params import FunctionalParam


class Parameters(ABC):
    """Abstract model parameters class."""
    pass


class SIRParameters(Parameters):
    def __init__(self, beta: FunctionalParam, gamma: FunctionalParam, population: int) -> None:
        self._beta = beta
        self._gamma = gamma
        self._population = population


@dataclass
class SIRHParameters(Parameters):
    beta: FunctionalParam
    gamma: FunctionalParam
    delta: FunctionalParam
    mu: FunctionalParam
    population: int


@dataclass
class LorenzParameters(Parameters):
    sigma: float
    rho: float
    beta: float
