"""Tests for the one-step Euler solver."""

import unittest

import jax.numpy as jnp

from jax.typing import DTypeLike
from numpy.testing import assert_allclose

from sphere.model.implementations.solvers.euler import EulerSolver

class TestEuler(unittest.TestCase):
    """Tests for the one step Euler solver."""
    def setUp(self): 
        self.euler = EulerSolver(delta_t = 1)

    def test_tol(self)->None: 
        """Test whether the results from the 
         Euler solver are within the specified 
         toleranace of the reference solution. """
        dts = [1,0.1,0.01,0.001]
        x_t = jnp.ones(len(dts))
        t = 0
        func = lambda x_t, t: x_t

        true_vals = jnp.array([[   1.,           2.,          4.,          8.,         16.,
                        32.,          64.,        128.,        256.,        512.,        ],
                     [   1.,           1.1,         1.21,        1.331,       1.4641,
                         1.61051,      1.771561,    1.9487171,   2.14358881,  2.35794769,],
                     [   1.,           1.01,        1.0201,      1.030301,    1.04060401,
                         1.05101005,   1.06152015,  1.07213535,  1.08285671,  1.09368527,],
                     [   1.,           1.001,       1.002001,   1.003003,     1.004006,
                         1.00501001,   1.00601502,  1.00702104, 1.00802806,   1.00903608,]])



        #for row,dt in enumerate(dts):
        self.euler.delta_t = 1
        for t in range(1,10):
            self.euler.solve(func,x_t.at[t-1],t)

        #assert_allclose(x_t,true_vals[0,:])



if __name__ == "__main__":
    unittest.main()

