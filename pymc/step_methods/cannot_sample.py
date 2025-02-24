from pymc.step_methods.arraystep import ArrayStep

class CannotSampleRV(ArrayStep):
    """
    A step method that raises an error when sampling a latent Multinomial variable.
    """
    name = "cannot_sample_rv"
    def __init__(self, vars, **kwargs):
        # Remove keys that ArrayStep.__init__ does not accept.
        kwargs.pop("model", None)
        kwargs.pop("initial_point", None)
        kwargs.pop("compile_kwargs", None)
        self.vars = vars
        super().__init__(vars=vars,fs=[], **kwargs)

    def astep(self, q0):
        # This method is required by the abstract base class.
        raise ValueError(
            "Latent Multinomial variables are not supported"
        )

