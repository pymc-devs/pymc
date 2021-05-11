import aesara.tensor as aet
from aesara.graph.op import Op
import numpy as np


class Loglike(Op):
    itypes = [aet.dvector]  # input type, expects a vector of parameter values when called
    otypes = [aet.dscalar]  # output type, outputs a single scalar value (the log likelihood)

    def __init__(self, model):
        self.model = model  # statsmodels model object
        self.score = Score(self.model)

    def perform(self, node, inputs, outputs):
        theta, = inputs  # contains the vector of parameters
        llf = self.model.loglike(theta)  # log-likelihood function (llf)
        outputs[0][0] = np.array(llf)  # output the log-likelihood

    def grad(self, inputs, output_gradients):
        # method to calculate the gradients
        # returns the Jacobian
        theta, = inputs
        out = [output_gradients[0] * self.score(theta)]
        return out


class Score(Op):
    itypes = [aet.dvector]
    otypes = [aet.dvector]

    def __init__(self, model):
        self.model = model

    def perform(self, node, inputs, outputs):
        theta, = inputs
        outputs[0][0] = self.model.score(theta)  # derivative of the log-likelihood function


