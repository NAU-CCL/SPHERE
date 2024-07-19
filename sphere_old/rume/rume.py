from typing import Type, TypeVar

from jax.typing import ArrayLike

from sphere_old.model.abstract.model import Model
from sphere.model.transition import Lorenz63Transition, SIRTransition
from sphere_old.model.implementations.lorenz63 import LorenzModel
from sphere_old.model.implementations.SIR import SIRModel
from sphere.model.parameters import LorenzParameters, SIRParameters
from sphere.model.solver import Solver
from sphere.output.abstract import Output
from sphere.output.lorenz63 import LorenzOutput
from sphere.output.SIR import SIROutput

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
    def create_lorenz63_rume(
        cls: Type[T], parameters: LorenzParameters, solver: Solver
    ) -> T:
        """
        Create an instance of a Lorenz RUME.

        Args:
            parameters: Parameters for the Lorenz model.
            solver: Solver to use for the experiment. Defaults to EulerSolver.

        Returns:
            An instance of the Rume class configured with a Lorenz model.
        """
        Lorenz_solver = solver(delta_t=0.001, transition=Lorenz63Transition(parameters))
        model = LorenzModel(parameters, Lorenz_solver)
        output = LorenzOutput()
        return cls(model, output)

    @classmethod
    def create_sir_rume(cls, parameters: SIRParameters, solver: Solver) -> T:
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


class RumeFactory:
    @staticmethod
    def create_rume(
        transition: Transition, parameters: Parameters, solver: Solver
    ) -> Rume:
        if model_type == "SIR":
            if stochastic == True:
                transition = StochasticSIR()
        elif model_type == "Lorenz63":
            transition = Lorenz63Transition()
        else:
            raise ValueError(f"Unknown model type: {transition}")

        solver.transition = transition
        return Model(params, solver)
