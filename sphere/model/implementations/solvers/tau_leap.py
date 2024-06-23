"""A simple implementation of a solver using fixed tau leaping. """

from typing import Callable,Dict,List

import jax.numpy as jnp
import numpy as np

from jax.typing import ArrayLike
from jax import Array

from jax import random,jit
from jax.tree_util import Partial 

from sphere.model.abstract.solver import Solver


class TauLeapSolver(Solver):
    """
        An implementation of solver, uses the approximate stochastic method of 
        tau-leaping described by Gillespie et. al. 

        Takes two additional arguments given by tau, analogous to delta_t in the one
        step Euler method, and PRNG_key, the key to pass to JAX random for the poisson draws. 
    """

    tau: float

    prng_key: Array

    req_keys = ['rates','transitions']

    def __init__(self, tau: float, prng_key: Array, args: Dict[str,Callable]) -> None:

        if(tau <= 0):
            raise ValueError(f"Tau must be greater than zero! Tau was {tau}. ")

        super().__init__(args = args)

        self.tau = tau
        self.prng_key = prng_key

    def solve(
        self,
        x_t: ArrayLike,
        t: int
    ) -> Array:
        """Solves the system described by func for a single discrete time step
        using tau leaping. 

        Args:
            x_t: The state of the system at time t, a JAX or NumPy Array, used in func. 
            t: The current discrete time step of the system, discretization schemes are left 
            up to the user, used in func. 
        Returns:
            The state of the system at time t+1, a JAX Array. Note, regardless of whether x_t was a
            JAX or NumPy Array, the return will always be a JAX Array.

        """

        events = np.random.poisson(lam = self.tau * self.args['rates'](x_t,t))

        result = np.copy(x_t)

        for index,desc in enumerate(self.args['transitions']()):
            for event,sign in desc:
                if sign == '+':
                    result[index] = result[index] + events[event]
                elif sign == '-': 
                    result[index] = result[index] - events[event]
                else: 
                    raise ValueError(f"Invalid symbol in transitions! Symbol was {sign}")

        return result
    

# @jit   
# def solve_helper(x_t:ArrayLike,t:int,lam:Array,transitions:Callable,key:Array) ->Array:
        
#         events = random.poisson(key= key,lam = lam)

#         result = x_t.copy()

#         for index,desc in enumerate(transitions()):
#             for event,sign in desc:
#                 if sign == '+':
#                     result = result.at[index].set(result[index] + events[event])
#                 elif sign == '-': 
#                     result = result.at[index].set(result[index] - events[event])

#         return jnp.array(result)
                                    
