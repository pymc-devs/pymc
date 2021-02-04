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
from functools import singledispatch
from typing import Generator, List, Optional, Tuple, Union

import aesara.tensor as aet
import numpy as np

from aesara import config
from aesara.graph.basic import Variable, ancestors, clone_replace
from aesara.graph.op import compute_test_value
from aesara.tensor.random.op import Observed, RandomVariable
from aesara.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1, Subtensor
from aesara.tensor.var import TensorVariable

from pymc3.aesaraf import floatX

PotentialShapeType = Union[
    int, np.ndarray, Tuple[Union[int, Variable], ...], List[Union[int, Variable]], Variable
]


def _get_scaling(total_size, shape, ndim):
    """
    Gets scaling constant for logp

    Parameters
    ----------
    total_size: int or list[int]
    shape: shape
        shape to scale
    ndim: int
        ndim hint

    Returns
    -------
    scalar
    """
    if total_size is None:
        coef = floatX(1)
    elif isinstance(total_size, int):
        if ndim >= 1:
            denom = shape[0]
        else:
            denom = 1
        coef = floatX(total_size) / floatX(denom)
    elif isinstance(total_size, (list, tuple)):
        if not all(isinstance(i, int) for i in total_size if (i is not Ellipsis and i is not None)):
            raise TypeError(
                "Unrecognized `total_size` type, expected "
                "int or list of ints, got %r" % total_size
            )
        if Ellipsis in total_size:
            sep = total_size.index(Ellipsis)
            begin = total_size[:sep]
            end = total_size[sep + 1 :]
            if Ellipsis in end:
                raise ValueError(
                    "Double Ellipsis in `total_size` is restricted, got %r" % total_size
                )
        else:
            begin = total_size
            end = []
        if (len(begin) + len(end)) > ndim:
            raise ValueError(
                "Length of `total_size` is too big, "
                "number of scalings is bigger that ndim, got %r" % total_size
            )
        elif (len(begin) + len(end)) == 0:
            return floatX(1)
        if len(end) > 0:
            shp_end = shape[-len(end) :]
        else:
            shp_end = np.asarray([])
        shp_begin = shape[: len(begin)]
        begin_coef = [floatX(t) / shp_begin[i] for i, t in enumerate(begin) if t is not None]
        end_coef = [floatX(t) / shp_end[i] for i, t in enumerate(end) if t is not None]
        coefs = begin_coef + end_coef
        coef = aet.prod(coefs)
    else:
        raise TypeError(
            "Unrecognized `total_size` type, expected int or list of ints, got %r" % total_size
        )
    return aet.as_tensor(floatX(coef))


def change_rv_size(
    rv_var: TensorVariable,
    new_size: PotentialShapeType,
    expand: Optional[bool] = False,
) -> TensorVariable:
    """Change or expand the size of a `RandomVariable`.

    Parameters
    ==========
    rv_var
        The `RandomVariable` output.
    new_size
        The new size.
    expand:
        Whether or not to completely replace the `size` parameter in `rv_var`
        with `new_size` or simply prepend it to the existing `size`.

    """
    rv_node = rv_var.owner
    rng, size, dtype, *dist_params = rv_node.inputs
    name = rv_var.name
    tag = rv_var.tag

    if expand:
        new_size = tuple(np.atleast_1d(new_size)) + tuple(size)

    new_rv_node = rv_node.op.make_node(rng, new_size, dtype, *dist_params)
    rv_var = new_rv_node.outputs[-1]
    rv_var.name = name
    for k, v in tag.__dict__.items():
        rv_var.tag.__dict__.setdefault(k, v)

    if config.compute_test_value != "off":
        compute_test_value(new_rv_node)

    return rv_var


def rv_log_likelihood_args(
    rv_var: TensorVariable,
    rv_value: Optional[TensorVariable] = None,
    transformed: Optional[bool] = True,
) -> Tuple[TensorVariable, TensorVariable]:
    """Get a `RandomVariable` and its corresponding log-likelihood `TensorVariable` value.

    Parameters
    ==========
    rv_var
        A variable corresponding to a `RandomVariable`, whether directly or
        indirectly (e.g. an observed variable that's the output of an
        `Observed` `Op`).
    rv_value
        The measure-space input `TensorVariable` (i.e. "input" to a
        log-likelihood).
    transformed
        When ``True``, return the transformed value var.

    Returns
    =======
    The first value in the tuple is the `RandomVariable`, and the second is the
    measure-space variable that corresponds with the latter.  The first is used
    to determine the log likelihood graph and the second is the "input"
    parameter to that graph.  In the case of an observed `RandomVariable`, the
    "input" is actual data; in all other cases, it's just another
    `TensorVariable`.

    """

    if rv_value is None:
        if rv_var.owner and isinstance(rv_var.owner.op, Observed):
            rv_var, rv_value = rv_var.owner.inputs
        elif hasattr(rv_var.tag, "value_var"):
            rv_value = rv_var.tag.value_var
        else:
            return rv_var, None

    rv_value = aet.as_tensor_variable(rv_value)

    transform = getattr(rv_value.tag, "transform", None)
    if transformed and transform:
        rv_value = transform.forward(rv_value)

    return rv_var, rv_value


def rv_ancestors(graphs: List[TensorVariable]) -> Generator[TensorVariable, None, None]:
    """Yield the ancestors that are `RandomVariable` outputs for the given `graphs`."""
    for anc in ancestors(graphs):
        if anc in graphs:
            continue
        if anc.owner and isinstance(anc.owner.op, RandomVariable):
            yield anc


def strip_observed(x: TensorVariable) -> TensorVariable:
    """Return the `RandomVariable` term for an `Observed` node input; otherwise, return the input."""
    if x.owner and isinstance(x.owner.op, Observed):
        return x.owner.inputs[0]
    else:
        return x


def sample_to_measure_vars(graphs: List[TensorVariable]) -> List[TensorVariable]:
    """Replace `RandomVariable` terms in graphs with their measure-space counterparts."""
    replace = {}
    for anc in rv_ancestors(graphs):
        measure_var = getattr(anc.tag, "value_var", None)
        if measure_var is not None:
            replace[anc] = measure_var

    dist_params = clone_replace(graphs, replace=replace)
    return dist_params


def logpt(
    rv_var: TensorVariable,
    rv_value: Optional[TensorVariable] = None,
    jacobian: bool = True,
    scaling: Optional[bool] = True,
    **kwargs,
) -> TensorVariable:
    """Create a measure-space (i.e. log-likelihood) graph for a random variable at a given point.

    The input `rv_var` determines which log-likelihood graph is used and
    `rv_value` is that graph's input parameter.  For example, if `rv_var` is
    the output of a `NormalRV` `Op`, then the output is
    ``normal_log_pdf(rv_value)``.

    Parameters
    ==========
    rv_var
        The `RandomVariable` output that determines the log-likelihood graph.
    rv_value
        The input variable for the log-likelihood graph.
    jacobian
        Whether or not to include the Jacobian term.
    scaling
        A scaling term to apply to the generated log-likelihood graph.

    """

    rv_var, rv_value = rv_log_likelihood_args(rv_var, rv_value)
    rv_node = rv_var.owner

    if not rv_node:
        raise TypeError("rv_var must be the output of a RandomVariable Op")

    if not isinstance(rv_node.op, RandomVariable):

        if isinstance(rv_node.op, (Subtensor, AdvancedSubtensor, AdvancedSubtensor1)):

            raise NotImplementedError("Missing value support is incomplete")

            # "Flatten" and sum an array of indexed RVs' log-likelihoods
            rv_var, missing_values = rv_node.inputs
            rv_value = rv_var.tag.value_var

            missing_values = missing_values.data
            logp_var = aet.sum(
                [
                    logpt(
                        rv_var,
                    )
                    for idx, missing in zip(
                        np.ndindex(missing_values.shape), missing_values.flatten()
                    )
                    if missing
                ]
            )
            return logp_var

        return aet.zeros_like(rv_var)

    rng, size, dtype, *dist_params = rv_node.inputs

    dist_params = sample_to_measure_vars(dist_params)

    if jacobian:
        logp_var = _logp(rv_node.op, rv_value, *dist_params, **kwargs)
    else:
        logp_var = _logp_nojac(rv_node.op, rv_value, *dist_params, **kwargs)

    # Replace `RandomVariable` ancestors with their corresponding
    # log-likelihood input variables
    lik_replacements = [
        (v, v.tag.value_var)
        for v in ancestors([logp_var])
        if v.owner and isinstance(v.owner.op, RandomVariable) and getattr(v.tag, "value_var", None)
    ]

    (logp_var,) = clone_replace([logp_var], replace=lik_replacements)

    if scaling:
        logp_var *= _get_scaling(
            getattr(rv_var.tag, "total_size", None), rv_value.shape, rv_value.ndim
        )

    if rv_var.name is not None:
        logp_var.name = "__logp_%s" % rv_var.name

    return logp_var


@singledispatch
def _logp(op, value, *dist_params, **kwargs):
    """Create a log-likelihood graph.

    This function dispatches on the type of `op`, which should be a subclass
    of `RandomVariable`.  If you want to implement new log-likelihood graphs
    for a `RandomVariable`, register a new function on this dispatcher.

    """
    return aet.zeros_like(value)


def logcdf(rv_var, rv_value, **kwargs):
    """Create a log-CDF graph."""

    rv_var, rv_value = rv_log_likelihood_args(rv_var, rv_value)
    rv_node = rv_var.owner

    if not rv_node:
        raise TypeError()

    rng, size, dtype, *dist_params = rv_node.inputs

    dist_params = sample_to_measure_vars(dist_params)

    return _logcdf(rv_node.op, rv_value, *dist_params, **kwargs)


@singledispatch
def _logcdf(op, value, *args, **kwargs):
    """Create a log-CDF graph.

    This function dispatches on the type of `op`, which should be a subclass
    of `RandomVariable`.  If you want to implement new log-CDF graphs
    for a `RandomVariable`, register a new function on this dispatcher.

    """
    raise NotImplementedError()


def logp_nojac(rv_var, rv_value=None, **kwargs):
    """Create a graph of the log-likelihood that doesn't include the Jacobian."""

    rv_var, rv_value = rv_log_likelihood_args(rv_var, rv_value)
    rv_node = rv_var.owner

    if not rv_node:
        raise TypeError()

    rng, size, dtype, *dist_params = rv_node.inputs

    dist_params = sample_to_measure_vars(dist_params)

    return _logp_nojac(rv_node.op, rv_value, **kwargs)


@singledispatch
def _logp_nojac(op, value, *args, **kwargs):
    """Return the logp, but do not include a jacobian term for transforms.

    If we use different parametrizations for the same distribution, we
    need to add the determinant of the jacobian of the transformation
    to make sure the densities still describe the same distribution.
    However, MAP estimates are not invariant with respect to the
    parameterization, we need to exclude the jacobian terms in this case.

    This function should be overwritten in base classes for transformed
    distributions.
    """
    return logpt(op, value, *args, **kwargs)


def logpt_sum(rv_var: TensorVariable, rv_value: Optional[TensorVariable] = None, **kwargs):
    """Return the sum of the logp values for the given observations.

    Subclasses can use this to improve the speed of logp evaluations
    if only the sum of the logp values is needed.
    """
    return aet.sum(logpt(rv_var, rv_value, **kwargs))


from pymc3.distributions import shape_utils, timeseries, transforms
from pymc3.distributions.bart import BART
from pymc3.distributions.bound import Bound
from pymc3.distributions.continuous import (
    AsymmetricLaplace,
    Beta,
    Cauchy,
    ChiSquared,
    ExGaussian,
    Exponential,
    Flat,
    Gamma,
    Gumbel,
    HalfCauchy,
    HalfFlat,
    HalfNormal,
    HalfStudentT,
    Interpolated,
    InverseGamma,
    Kumaraswamy,
    Laplace,
    Logistic,
    LogitNormal,
    Lognormal,
    Moyal,
    Normal,
    Pareto,
    Rice,
    SkewNormal,
    StudentT,
    Triangular,
    TruncatedNormal,
    Uniform,
    VonMises,
    Wald,
    Weibull,
)
from pymc3.distributions.discrete import (
    Bernoulli,
    BetaBinomial,
    Binomial,
    Categorical,
    Constant,
    ConstantDist,
    DiscreteUniform,
    DiscreteWeibull,
    Geometric,
    HyperGeometric,
    NegativeBinomial,
    OrderedLogistic,
    OrderedProbit,
    Poisson,
    ZeroInflatedBinomial,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)
from pymc3.distributions.distribution import (
    Continuous,
    DensityDist,
    Discrete,
    Distribution,
    NoDistribution,
)
from pymc3.distributions.mixture import Mixture, MixtureSameFamily, NormalMixture
from pymc3.distributions.multivariate import (
    Dirichlet,
    DirichletMultinomial,
    KroneckerNormal,
    LKJCholeskyCov,
    LKJCorr,
    MatrixNormal,
    Multinomial,
    MvNormal,
    MvStudentT,
    Wishart,
    WishartBartlett,
)
from pymc3.distributions.simulator import Simulator
from pymc3.distributions.timeseries import (
    AR,
    AR1,
    GARCH11,
    GaussianRandomWalk,
    MvGaussianRandomWalk,
    MvStudentTRandomWalk,
)

__all__ = [
    "Uniform",
    "Flat",
    "HalfFlat",
    "TruncatedNormal",
    "Normal",
    "Beta",
    "Kumaraswamy",
    "Exponential",
    "Laplace",
    "StudentT",
    "Cauchy",
    "HalfCauchy",
    "Gamma",
    "Weibull",
    "Bound",
    "Lognormal",
    "HalfStudentT",
    "ChiSquared",
    "HalfNormal",
    "Wald",
    "Pareto",
    "InverseGamma",
    "ExGaussian",
    "VonMises",
    "Binomial",
    "BetaBinomial",
    "Bernoulli",
    "Poisson",
    "NegativeBinomial",
    "ConstantDist",
    "Constant",
    "ZeroInflatedPoisson",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedBinomial",
    "DiscreteUniform",
    "Geometric",
    "HyperGeometric",
    "Categorical",
    "OrderedLogistic",
    "OrderedProbit",
    "DensityDist",
    "Distribution",
    "Continuous",
    "Discrete",
    "NoDistribution",
    "MvNormal",
    "MatrixNormal",
    "KroneckerNormal",
    "MvStudentT",
    "Dirichlet",
    "Multinomial",
    "DirichletMultinomial",
    "Wishart",
    "WishartBartlett",
    "LKJCholeskyCov",
    "LKJCorr",
    "AR1",
    "AR",
    "AsymmetricLaplace",
    "GaussianRandomWalk",
    "MvGaussianRandomWalk",
    "MvStudentTRandomWalk",
    "GARCH11",
    "SkewNormal",
    "Mixture",
    "NormalMixture",
    "MixtureSameFamily",
    "Triangular",
    "DiscreteWeibull",
    "Gumbel",
    "Logistic",
    "LogitNormal",
    "Interpolated",
    "Bound",
    "Rice",
    "Moyal",
    "Simulator",
    "BART",
]
