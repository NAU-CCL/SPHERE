import unittest

from sphere.model.abstract.parameters import (
    LorenzParameters,
    SIRHParameters,
    SIRParameters,
)
from sphere.model.implementations import SIR, SIRH, Lorenz


class TestLorenz(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
