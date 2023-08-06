#   Copyright 2023 The PyMC Developers
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
#
#   MIT License
#
#   Copyright (c) 2021-2022 aesara-devs
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

from typing import List, Optional

import pytensor.tensor as pt

from pytensor.graph.basic import Node
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.math import Max
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.var import TensorVariable

from pymc.logprob.abstract import (
    MeasurableVariable,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
)
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.pytensorf import constant_fold


class MeasurableMax(Max):
    """A placeholder used to specify a log-likelihood for a max sub-graph."""


MeasurableVariable.register(MeasurableMax)


@node_rewriter([Max])
def find_measurable_max(fgraph: FunctionGraph, node: Node) -> Optional[List[TensorVariable]]:
    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)
    if rv_map_feature is None:
        return None  # pragma: no cover

    if isinstance(node.op, MeasurableMax):
        return None  # pragma: no cover

    base_var = node.inputs[0]

    if base_var.owner is None:
        return None

    if not rv_map_feature.request_measurable(node.inputs):
        return None

    # Non-univariate distributions and non-RVs must be rejected
    if not (isinstance(base_var.owner.op, RandomVariable) and base_var.owner.op.ndim_supp == 0):
        return None

    # TODO: We are currently only supporting continuous rvs
    if isinstance(base_var.owner.op, RandomVariable) and base_var.owner.op.dtype.startswith("int"):
        return None

    # univariate i.i.d. test which also rules out other distributions
    for params in base_var.owner.inputs[3:]:
        if params.type.ndim != 0:
            return None

    # Check whether axis covers all dimensions
    axis = set(node.op.axis)
    base_var_dims = set(range(base_var.ndim))
    if axis != base_var_dims:
        return None

    measurable_max = MeasurableMax(list(axis))
    max_rv_node = measurable_max.make_node(base_var)
    max_rv = max_rv_node.outputs

    return max_rv


measurable_ir_rewrites_db.register(
    "find_measurable_max",
    find_measurable_max,
    "basic",
    "max",
)


@_logprob.register(MeasurableMax)
def max_logprob(op, values, base_rv, **kwargs):
    r"""Compute the log-likelihood graph for the `Max` operation.

    Parameters
    ----------
    op : Max-Op
    values : tensor_like
    rv : TensorVariable

    Returns
    -------
    logprob : TensorVariable

    Examples
    --------
    It is often desirable to find the log-probability of the maximum of i.i.d. random variables.

    The "max of i.i.d. random variables" refers to finding the maximum value among a collection of random variables that are independent and identically distributed.
    The example below illustrates how to find the Maximum from the distribution of random variables.

    .. code-block:: python

        import pytensor.tensor as pt

        x = pt.random.normal(0, 1, size=(3,))
        x.name = "x"
        print(x.eval())
        #[0.61748772 1.08723759 0.98970957]

        x_max = pt.max(x, axis=None)
        print(x_max.eval())
        # 1.087237592696084

    We can also create a Custom Distribution to find the max as

    .. code-block:: python

        def test_doc():
        data = [-1, -1, 0]

        def max_dist(mu, sigma, size):
            return pt.max(pm.Normal.dist(mu, sigma, size=size))

        with pm.Model() as m:
            mu = pm.Normal("mu")
            sigma = pm.HalfNormal("sigma")
            obs = pm.CustomDist("obs", mu, sigma, dist=max_dist, observed=data,)

    The log-probability of the maximum of i.i.d. random variables is a measure of the likelihood of observing a specific maximum value in a set of independent and identically distributed random variables.

    The formula that we use here is :
        \ln(f_{(n)}(x)) = \ln(n) + (n-1) \ln(F(x)) + \ln(f(x))
    where f(x) represents the p.d.f and F(x) represents the c.d.f of the distribution respectively.

    An example corresponding to this is illustrated below:

    .. code-block:: python

        import pytensor.tensor as pt
        from pymc import logp

        x = pt.random.uniform(0, 1, size=(3,))
        x.name = "x"
        # [0.09081509 0.84761712 0.59030273]

        x_max = pt.max(x, axis=-1)
        # 0.8476171198716373

        x_max_value = pt.scalar("x_max_value")
        x_max_logprob = logp(x_max, x_max_value)
        test_value = x_max.eval()

        x_max_logprob.eval({x_max_value: test_value})
        # 0.7679597791946853

    Currently our implementation has certain limitations which are mandated through some constraints.

    We only consider a distribution of RandomVariables and the logp function fails for NonRVs.

    .. code-block:: python

        import pytensor.tensor as pt
        from pymc import logp

        x = pt.exp(pt.random.beta(0, 1, size=(3,)))
        x.name = "x"
        x_max = pt.max(x, axis=-1)
        x_max_value = pt.vector("x_max_value")
        x_max_logprob = logp(x_max, x_max_value)

    The above code gives a Runtime error stating logprob method was not implemented as x in this case is not a pure random variable.
    A pure random variable in PyMC represents an unknown quantity in a Bayesian model and is associated with a prior distribution that is combined with the likelihood of observed data to obtain the posterior distribution through Bayesian inference

    We assume only univariate distributions as for multivariate variables, the concept of ordering is ambiguous since a "depth function" is required .

    We only consider independent and identically distributed random variables, for now.
    In probability theory and statistics, a collection of random variables is independent and identically distributed if each random variable has the same probability distribution as the others and all are mutually independent.

    .. code-block:: python

        import pytensor.tensor as pt
        from pymc import logp

        x = pm.Normal.dist([0, 1, 2, 3, 4], 1, shape=(5,))
        x.name = "x"
        x_max = pt.max(x, axis=-1)
        x_max_value = pt.vector("x_max_value")
        x_max_logprob = logp(x_max, x_max_value)

    The above code gives a Runtime error stating logprob method was not implemented as x in this case is a Non-iid distribution.

    Note: We assume a very fluid definition of i.i.d. here. We say that an RV belongs to an i.i.d. if that RV does not have different stochastic ancestors.


    """
    (value,) = values

    logprob = _logprob_helper(base_rv, value)
    logcdf = _logcdf_helper(base_rv, value)

    [n] = constant_fold([base_rv.size])

    logprob = (n - 1) * logcdf + logprob + pt.math.log(n)

    return logprob
