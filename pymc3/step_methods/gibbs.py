'''
Created on May 12, 2012

@author: john
'''
from ..core import *
from .arraystep import *
from ..distributions.discrete import Categorical
from numpy import array, max, exp, cumsum, nested_iters, empty, searchsorted, ones
from numpy.random import uniform

from theano.gof.graph import inputs
from theano.tensor import add 

__all__ = ['ElemwiseCategoricalStep']


class ElemwiseCategoricalStep(ArrayStep):
    """
    Gibbs sampling for categorical variables that only have only have elementwise effects
    the variable can't be indexed into or transposed or anything otherwise that will mess things up

    """
    # TODO: It would be great to come up with a way to make
    # ElemwiseCategoricalStep  more general (handling more complex elementwise
    # variables)
    def __init__(self, vars, values=None, model=None):
        model = modelcontext(model)
        self.sh = ones(vars.dshape, vars.dtype)
        self.values = values
        self.vars = vars

        super(ElemwiseCategoricalStep, self).__init__([vars], [elemwise_logp(model, vars)])

    def astep(self, q, logp):
        p = array([logp(v * self.sh) for v in self.values])
        return categorical(p, self.var.dshape)

    @staticmethod
    def competence(var):
        if isinstance(var.distribution, Categorical):
            return Competence.ideal
        return Competence.incompatible

def elemwise_logp(model, var):
    terms = [v.logp_elemwiset for v in model.basic_RVs if var in inputs([v.logpt])]
    return model.fn(add(*terms))


def categorical(prob, shape):
    out = empty([1] + list(shape))

    n = len(shape)
    it0, it1 = nested_iters([prob, out], [list(range(1, n + 1)), [0]],
                            op_flags=[['readonly'], ['readwrite']],
                            flags=['reduce_ok'])

    for i in it0:
        p, o = it1.itviews
        p = cumsum(exp(p - max(p, axis=0)))
        r = uniform() * p[-1]

        o[0] = searchsorted(p, r)

    return out[0, ...]
