from typing import TypeVar, Type

from sphere.model.abstract.model import Model
from sphere.model.implementations.Lorenz import LorenzModel
from sphere.model.implementations.SIR import SIRModel
from sphere.output.implementations.SIR import SIROutput
from sphere.model.abstract.solvers import ODESolver, JAXSolver
from sphere.model.abstract.parameters import Parameters, SIRParameters, LorenzParameters
from sphere.output.implementations.lorenz import LorenzOutput
from sphere.output.abstract.output import Output

some_rume = TypeVar("some_rume", bound="Rume")


class Rume:
    """
    Base class for the RUME (Runnable Modeling Experiment).

    Class methods create instances of a specific Rume.
    """
    def __init__(self, model: Model, output: Output) -> None:
        self.model = model
        self.output = output

    @classmethod
    def create_lorenz_rume(
        cls, parameters: LorenzParameters, solver: ODESolver = JAXSolver()
    ) -> some_rume:
        model = LorenzModel(parameters, solver)
        output = LorenzOutput()
        return cls(model, output)

    @classmethod
    def create_sir_rume(cls, parameters: SIRParameters, solver: ODESolver =
    JAXSolver()) \
            -> some_rume:
        model = SIRModel(parameters, solver)
        output = SIROutput()
        return cls(model, output)

    def run(self, time_steps: int):
        self.output.states = self.model.run(time_steps)
