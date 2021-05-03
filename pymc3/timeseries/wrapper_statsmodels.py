import theano.tensor as tt
from theano.tensor import Op
import aesara
import aesara.tensor as aet
from aesara.graph.op import Op

class Loglike(Op):

    itypes = [aet.dvector]
    otypes = [aet.dscalar]

    def __init__(self, model):
        self.model = model
        self.score = Score(self.model)

    def perform(self, node, inputs, outputs):
        theta, = inputs
        llf = self.model.loglike(theta)
        outputs[0][0] = np.array(llf)

    def grad(self, inputs, g):
        theta, = inputs
        out  [g[0] * score(theta)]


class Score(Op):
    itypes = [aet.dvector]
    otypes = [aet.dvector]

    def __init__(self, model):
        self.model = model

    def perform(self, node, inputs, outputs):
        theta, = inputs
        outputs[0][0] = self.model.score(theta)



