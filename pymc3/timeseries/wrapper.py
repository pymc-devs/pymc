import aesara.tensor as aet
import numpy as np

from aesara.graph.op import Op


class Loglike(Op):
    """
    Aesara wrapper for a statsmodels model instance to enable pymc3
    parameter estimation.
    """

    itypes = [aet.dvector]  # input type, expects a vector of parameter values when called
    otypes = [aet.dscalar]  # output type, outputs a single scalar value (the log likelihood)

    def __init__(self, model):
        self.model = model  # statsmodels model instance
        self.score = Score(self.model)

    def perform(self, node, inputs, outputs):
        (theta,) = inputs  # contains the vector of parameters
        llf = self.model.loglike(theta)  # log-likelihood function (llf)
        outputs[0][0] = np.array(llf)  # output the log-likelihood

    def grad(self, inputs, output_gradients):
        """
        Returns the Jacobian Matrix for the calculation of gradients to be used in pymc3 sampling.
        """
        (theta,) = inputs
        out = [output_gradients[0] * self.score(theta)]
        return out


class Score(Op):
    """
    Aesara wrapper for a statsmodels model instance to calculate the
    derivative of the log-likelihood function.
    """

    itypes = [aet.dvector]  # input type, expects a vector of parameter values when called
    otypes = [aet.dvector]  # output type, outputs a vector of derivatives

    def __init__(self, model):
        self.model = model

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        outputs[0][0] = self.model.score(theta)  # derivative of the log-likelihood function
