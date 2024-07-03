"""
The model parameters. An abstract base class is defined.
Each concrete subclass defines that model's unique parameters.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple
import jax.numpy as jnp


@dataclass
class Parameters(ABC):
    pass


@dataclass
class SIRParameters(Parameters):
    beta: float
    gamma: float


@dataclass
class SIRHParameters(Parameters):
    beta: float
    gamma: float
    delta: float
    mu: float


@dataclass
class LorenzParameters(Parameters):
    sigma: float
    rho: float
    beta: float
