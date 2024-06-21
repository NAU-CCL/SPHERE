from sphere.model.abstract.model import Model
from sphere.model.implementations.SIR import SIRModel
from sphere.output.implementations.SIR import SIROutput
from sphere.model.abstract.solvers import ODESolver
from sphere.model.abstract.parameters import Parameters

output_classes = {
    SIRModel: SIROutput()
}


class Rume:
    """
    Runnable Modeling Experiment.
    """
    def __init__(self, model: Model, parameters: Parameters, solver:
    ODESolver) -> None:
        self.model = model
        self.parameters = parameters
        self.solver = solver
        self.output = None
        self.__post_init__()

    def __post_init__(self):
        self.set_output()
        self.validate_input()

    def set_output(self):
        """Sets the Output class, based on the chosen Model."""

        # Get the corresponding output class for the model
        output_class = output_classes.get(type(self.model))
        if output_class is None:
            raise ValueError(f"No output class defined for model type: {type(self.model)}")
        # Instantiate the output class
        self.output = output_class

    def validate_parameters(self):
        pass
