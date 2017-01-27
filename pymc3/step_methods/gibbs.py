'''
Created on May 12, 2012

@author: john
'''
from .arraystep import ArrayStep, Competence
from ..distributions.discrete import Categorical
import numpy as np
from numpy.random import uniform
from warnings import warn

from theano.gof.graph import inputs
from theano.tensor import add
from ..model import modelcontext
__all__ = ['ElemwiseCategorical']


class ElemwiseCategorical(ArrayStep):
    """
    Gibbs sampling for categorical variables that only have ElemwiseCategoricalise effects
    the variable can't be indexed into or transposed or anything otherwise that will mess things up

    """
    # TODO: It would be great to come up with a way to make
    # ElemwiseCategorical  more general (handling more complex elementwise
    # variables)

    def __init__(self, vars, values=None, model=None):
        warn('ElemwiseCategorical is deprecated, switch to CategoricalGibbsMetropolis.',
             DeprecationWarning, stacklevel=2)
        model = modelcontext(model)
        self.var = vars[0]
        self.sh = np.ones(self.var.dshape, self.var.dtype)
        if values is None:
            self.values = np.arange(self.var.distribution.k)
        else:
            self.values = values

        super(ElemwiseCategorical, self).__init__(
            vars, [elemwise_logp(model, self.var)])

    def astep(self, q, logp):
        p = np.array([logp(v * self.sh) for v in self.values])
        return categorical(p, self.var.dshape)

    @staticmethod
    def competence(var):
        if isinstance(var.distribution, Categorical):
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE


def elemwise_logp(model, var):
    terms = [v.logp_elemwiset for v in model.basic_RVs if var in inputs([
                                                                        v.logpt])]
    return model.fn(add(*terms))


def categorical(prob, shape):
    out = np.empty([1] + list(shape))

    n = len(shape)
    it0, it1 = np.nested_iters([prob, out], [list(range(1, n + 1)), [0]],
                               op_flags=[['readonly'], ['readwrite']],
                               flags=['reduce_ok'])

    for i in it0:
        p, o = it1.itviews
        p = np.cumsum(np.exp(p - np.max(p, axis=0)))
        r = uniform() * p[-1]
        o[0] = np.searchsorted(p, r)

    return out[0, ...]
