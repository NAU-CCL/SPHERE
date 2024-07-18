"""
The model parameters. An abstract base class is defined.
Each concrete subclass defines that model's unique parameters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Any
import jax.numpy as jnp


@dataclass
class Parameters(ABC):
    pass


@dataclass
class SIRParameters(Parameters):
    beta: FunctionalParam
    gamma: FunctionalParam
    population: int


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



