from typing import TypeVar, Type

from jax import Array
from jax.typing import ArrayLike

from sphere.model.abstract.model import Model
from sphere.model.implementations.Lorenz import LorenzModel
from sphere.model.implementations.SIR import SIRModel
from sphere.output.implementations.SIR import SIROutput
from sphere.model.abstract.solver import Solver
from sphere.model.implementations.solvers.euler import EulerSolver
from sphere.model.abstract.parameters import Parameters, SIRParameters, LorenzParameters
from sphere.output.implementations.lorenz import LorenzOutput
from sphere.output.abstract.output import Output
from sphere.model.abstract.transition import SIRTransition, Lorenz63Transition

T = TypeVar("T", bound="Rume")


class Rume:
    """
    Base class for the RUME (Runnable Modeling Experiment).

    This class serves as a blueprint for creating and running modeling
    experiments with different models, parameters, solvers,
    and outputs.

    Specific types of RUME can be instantiated using the provided class
    methods.

    Attributes:
        model (Model): The model used in the experiment.
        output (Output): The output of the experiment.
    """

    def __init__(self, model: Model, output: Output) -> None:
        self.model = model
        self.output = output

    @classmethod
    def create_lorenz_rume(
        cls: Type[T], parameters: LorenzParameters, solver: Solver = EulerSolver) -> T:
        """
        Create an instance of a Lorenz RUME.

        Args:
            parameters: Parameters for the Lorenz model.
            solver: Solver to use for the experiment. Defaults to EulerSolver.

        Returns:
            An instance of the Rume class configured with a Lorenz model.
        """
        Lorenz_solver = solver(delta_t=.001, transition=Lorenz63Transition(parameters))
        model = LorenzModel(parameters, Lorenz_solver)
        output = LorenzOutput()
        return cls(model, output)

    @classmethod
    def create_sir_rume(
        cls, parameters: SIRParameters, solver: Solver = EulerSolver) -> T:
        """
        Create an instance of an SIR RUME.

        Args:
            parameters: Parameters for the SIR model.
            solver: Solver to use for the experiment. Defaults to EulerSolver.

        Returns:
            An instance of the Rume class configured with an SIR model.
        """
        SIR_solver = solver(delta_t=1, transition=SIRTransition(parameters))
        model = SIRModel(parameters, SIR_solver)
        output = SIROutput()
        return cls(model, output)

    def run(self, x0: ArrayLike, t0: int, t_final: int) -> None:
        """
        Run an experiment using the configured model.

        The result is stored in the `output` attribute. This is essentially a wrapper
            for Model.run() to simplify the user experience.
        """
        self.output.states = self.model.run(x0, t0, t_final)
