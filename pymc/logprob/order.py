#   Copyright 2024 - present The PyMC Developers
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
from typing import cast

import pytensor.tensor as pt

from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.math import Max
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import (
    MeasurableElemwise,
    MeasurableOp,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
)
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import filter_measurable_variables
from pymc.math import logdiffexp
from pymc.pytensorf import constant_fold


class MeasurableMax(MeasurableOp, Max):
    """A placeholder used to specify a log-likelihood for a max sub-graph."""


class MeasurableMaxDiscrete(MeasurableOp, Max):
    """A placeholder used to specify a log-likelihood for sub-graphs of maxima of discrete variables."""


@node_rewriter([Max])
def find_measurable_max(fgraph: FunctionGraph, node: Apply) -> list[TensorVariable] | None:
    if isinstance(node.op, MeasurableMax | MeasurableMaxDiscrete):
        return None

    [base_var] = node.inputs

    if base_var.owner is None:
        return None

    if not filter_measurable_variables(node.inputs):
        return None

    # We allow Max of RandomVariables or Elemwise of univariate RandomVariables
    if isinstance(base_var.owner.op, MeasurableElemwise):
        latent_base_vars = [
            var
            for var in base_var.owner.inputs
            if (var.owner and isinstance(var.owner.op, MeasurableOp))
        ]
        if len(latent_base_vars) != 1:
            return None
        [latent_base_var] = latent_base_vars
    else:
        latent_base_var = base_var

    latent_op = latent_base_var.owner.op
    if not (hasattr(latent_op, "dist_params") and getattr(latent_op, "ndim_supp") == 0):
        return None

    # univariate i.i.d. test which also rules out other distributions
    if not all(
        all(params.type.broadcastable) for params in latent_op.dist_params(latent_base_var.owner)
    ):
        return None

    base_var = cast(TensorVariable, base_var)

    if node.op.axis is None:
        axis = tuple(range(base_var.ndim))
    else:
        # Check whether axis covers all dimensions
        axis = tuple(sorted(node.op.axis))
        if axis != tuple(range(base_var.ndim)):
            return None

    # distinguish measurable discrete and continuous (because logprob is different)
    measurable_max_class = (
        MeasurableMaxDiscrete if latent_base_var.type.dtype.startswith("int") else MeasurableMax
    )
    max_rv = cast(TensorVariable, measurable_max_class(axis)(base_var))
    return [max_rv]


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
    op : MeasurableMax
    values : tensor_like
    rv : TensorVariable

    Returns
    -------
    logprob : TensorVariable

    Examples
    --------
    It is often desirable to find the log-probability corresponding to the maximum of i.i.d. random variables.

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

    The above code gives a Runtime error stating that logprob method was not implemented as x in this case is not a pure random variable.
    A pure random variable in PyMC represents an unknown quantity in a Bayesian model and is associated with a prior distribution that is combined with the likelihood of observed data to obtain the posterior distribution through Bayesian inference.

    We assume only univariate distributions as for multivariate variables, the concept of ordering is ambiguous since a "depth function" is required.

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

    base_rv_shape = constant_fold(tuple(base_rv.shape), raise_not_constant=False)
    bcast_value = pt.broadcast_to(value, base_rv_shape)
    logprob = _logprob_helper(base_rv, bcast_value)[0]
    logcdf = _logcdf_helper(base_rv, bcast_value)[0]

    n = pt.prod(base_rv_shape)
    return (n - 1) * logcdf + logprob + pt.math.log(n)


@_logprob.register(MeasurableMaxDiscrete)
def max_logprob_discrete(op, values, base_rv, **kwargs):
    r"""Compute the log-likelihood graph for the `Max` operation.

    The formula that we use here is :
    .. math::
        \ln(P_{(n)}(x)) = \ln(F(x)^n - F(x-1)^n)
    where $P_{(n)}(x)$ represents the p.m.f of the maximum statistic and $F(x)$ represents the c.d.f of the i.i.d. variables.
    """
    (value,) = values

    base_rv_shape = constant_fold(tuple(base_rv.shape), raise_not_constant=False)
    bcast_value = pt.broadcast_to(value, base_rv_shape)
    logcdf = _logcdf_helper(base_rv, bcast_value)[0]
    logcdf_prev = _logcdf_helper(base_rv, bcast_value - 1)[0]

    n = pt.prod(base_rv_shape)
    return logdiffexp(n * logcdf, n * logcdf_prev)
