from typing import Dict, Type

from sphere.model.model import Model
from sphere.parameters.parameters import SIRParameters, Lorenz63Parameters, Parameters
from sphere.output.Lorenz63 import Lorenz63Output
from sphere.output.SIR import SIROutput
from sphere.solvers.abstract import Solver, DeterministicSolver, StochasticSolver
from sphere.transition.implementations import DeterministicSIR, StochasticSIR, Lorenz63Transition


class ModelFactory:
    """
    The ModelFactory class is used to create the model instances.

    The ModelFactory infers the type of process (stochastic vs.
        deterministic) depending on the type of solver passed in.
    """

    # Dictionary to hold model types and their respective parameter and transition classes
    MODEL_TYPES: Dict[str, Type] = {
        "SIR": (SIRParameters, DeterministicSIR, StochasticSIR, SIROutput),
        "Lorenz63": (Lorenz63Parameters, Lorenz63Transition, None, Lorenz63Output),
    }

    @staticmethod
    def list_model_types() -> None:
        """Display available model types and their descriptions."""
        print("Available model types:")
        for model_type, (
            param_cls,
            _,
            stochastic,
            _,
        ) in ModelFactory.MODEL_TYPES.items():
            print(f"- {model_type}: Requires {param_cls.__name__}. ", end="")
            if stochastic:
                print("Can be used with deterministic or stochastic solvers.")
            else:
                print("Only supports deterministic solvers.")

    @staticmethod
    def create_model(
        model_type: str, params: Parameters, solver_cls: Type[Solver]
    ) -> Model:
        """Create an instance of a model.

        Args:
            model_type: The type of model to create.
            params: The parameters for the model.
            solver_cls: The solver class to use for the model.

        Returns:
            An instance of the requested model.

        Raises:
            ValueError: If the model_type is unknown.
            TypeError: If the parameters or solver are not compatible with the model type.
        """
        if model_type not in ModelFactory.MODEL_TYPES:
            raise ValueError(
                f"Unknown model type: {model_type}. Use ModelFactory.list_model_types() to see available types."
            )

        param_cls, det_sir_cls, stoch_sir_cls, output_cls = ModelFactory.MODEL_TYPES[
            model_type
        ]

        if not isinstance(params, param_cls):
            raise TypeError(f"Expected {param_cls.__name__} for {model_type} model.")

        if model_type == "SIR":
            if issubclass(solver_cls, DeterministicSolver):
                transition = det_sir_cls(params)
            elif issubclass(solver_cls, StochasticSolver):
                transition = stoch_sir_cls(params)
            else:
                raise ValueError(
                    f"{solver_cls} is not compatible with the SIR system."
                )
        elif model_type == "Lorenz63":
            if not issubclass(solver_cls, DeterministicSolver):
                raise ValueError(
                    f"{solver_cls} is not compatible with the Lorenz 63 system."
                )
            transition = Lorenz63Transition(params)

        solver = solver_cls(transition)
        output = output_cls()
        solver_cls.transition = transition
        return Model(solver=solver, output=output)
