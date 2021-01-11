#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Created on May 12, 2012

@author: john
"""
from warnings import warn

from numpy import (
    arange,
    array,
    cumsum,
    empty,
    exp,
    max,
    nested_iters,
    ones,
    searchsorted,
)
from numpy.random import uniform
from theano.graph.basic import graph_inputs
from theano.tensor import add

from pymc3.distributions.discrete import Categorical
from pymc3.model import modelcontext
from pymc3.step_methods.arraystep import ArrayStep, Competence

__all__ = ["ElemwiseCategorical"]


class ElemwiseCategorical(ArrayStep):
    """
    Gibbs sampling for categorical variables that only have ElemwiseCategoricalise effects
    the variable can't be indexed into or transposed or anything otherwise that will mess things up

    """

    # TODO: It would be great to come up with a way to make
    # ElemwiseCategorical  more general (handling more complex elementwise
    # variables)

    def __init__(self, vars, values=None, model=None):
        warn(
            "ElemwiseCategorical is deprecated, switch to CategoricalGibbsMetropolis.",
            DeprecationWarning,
            stacklevel=2,
        )
        model = modelcontext(model)
        self.var = vars[0]
        self.sh = ones(self.var.dshape, self.var.dtype)
        if values is None:
            self.values = arange(self.var.distribution.k)
        else:
            self.values = values

        super().__init__(vars, [elemwise_logp(model, self.var)])

    def astep(self, q, logp):
        p = array([logp(v * self.sh) for v in self.values])
        return categorical(p, self.var.dshape)

    @staticmethod
    def competence(var, has_grad):
        if isinstance(var.distribution, Categorical):
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE


def elemwise_logp(model, var):
    terms = [v.logp_elemwiset for v in model.basic_RVs if var in graph_inputs([v.logpt])]
    return model.fn(add(*terms))


def categorical(prob, shape):
    out = empty([1] + list(shape))

    n = len(shape)
    it0, it1 = nested_iters(
        [prob, out],
        [list(range(1, n + 1)), [0]],
        op_flags=[["readonly"], ["readwrite"]],
        flags=["reduce_ok"],
    )

    for _ in it0:
        p, o = it1.itviews
        p = cumsum(exp(p - max(p, axis=0)))
        r = uniform() * p[-1]

        o[0] = searchsorted(p, r)

    return out[0, ...]
