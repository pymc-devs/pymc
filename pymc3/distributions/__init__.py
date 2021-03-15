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
from itertools import chain
from typing import Generator, List, Optional, Tuple, Union

import aesara.tensor as at
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


@singledispatch
def logp_transform(op, inputs):
    return None


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
        coef = at.prod(coefs)
    else:
        raise TypeError(
            "Unrecognized `total_size` type, expected int or list of ints, got %r" % total_size
        )
    return at.as_tensor(floatX(coef))


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
    transformed: Optional[bool] = True,
) -> Tuple[TensorVariable, TensorVariable]:
    """Get a `RandomVariable` and its corresponding log-likelihood `TensorVariable` value.

    Parameters
    ==========
    rv_var
        A variable corresponding to a `RandomVariable`, whether directly or
        indirectly (e.g. an observed variable that's the output of an
        `Observed` `Op`).
    transformed
        When ``True``, return the transformed value var.

    Returns
    =======
    The first value in the tuple is the `RandomVariable`, and the second is the
    measure-space variable that corresponds with the latter (i.e. the "value"
    variable).

    """

    if rv_var.owner and isinstance(rv_var.owner.op, Observed):
        return tuple(rv_var.owner.inputs)
    elif hasattr(rv_var.tag, "value_var"):
        rv_value = rv_var.tag.value_var
        return rv_var, rv_value
    else:
        return rv_var, None


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


def sample_to_measure_vars(
    graphs: List[TensorVariable],
) -> Tuple[List[TensorVariable], List[TensorVariable]]:
    """Replace sample-space variables in graphs with their measure-space counterparts.

    Sample-space variables are `TensorVariable` outputs of `RandomVariable`
    `Op`s.  Measure-space variables are `TensorVariable`s that correspond to
    the value of a sample-space variable in a likelihood function (e.g. ``x``
    in ``p(X = x)``, where ``X`` is the corresponding sample-space variable).
    (``x`` is also the variable found in ``rv_var.tag.value_var``, so this
    function could also be called ``sample_to_value_vars``.)

    Parameters
    ==========
    graphs
        The graphs in which random variables are to be replaced by their
        measure variables.

    Returns
    =======
    Tuple containing the transformed graphs and a ``dict`` of the replacements
    that were made.
    """
    replace = {}
    for anc in chain(rv_ancestors(graphs), graphs):

        if not (anc.owner and isinstance(anc.owner.op, RandomVariable)):
            continue

        _, value_var = rv_log_likelihood_args(anc)

        if value_var is not None:
            replace[anc] = value_var

    if replace:
        measure_graphs = clone_replace(graphs, replace=replace)
    else:
        measure_graphs = graphs

    return measure_graphs, replace


def logpt(
    rv_var: TensorVariable,
    rv_value: Optional[TensorVariable] = None,
    jacobian: Optional[bool] = True,
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
        The input variable for the log-likelihood graph.  If `rv_value` is
        a transformed variable, its transformations will be applied.
        If no value is provided, `rv_var.tag.value_var` will be checked and,
        when available, used.
    jacobian
        Whether or not to include the Jacobian term.
    scaling
        A scaling term to apply to the generated log-likelihood graph.

    """

    rv_var, rv_value_var = rv_log_likelihood_args(rv_var)

    if rv_value is None:
        rv_value = rv_value_var
    else:
        rv_value = at.as_tensor(rv_value)

    if rv_value_var is None:
        rv_value_var = rv_value

    rv_node = rv_var.owner

    if not rv_node:
        raise TypeError("rv_var must be the output of a RandomVariable Op")

    if not isinstance(rv_node.op, RandomVariable):

        # This will probably need another generic function...
        if isinstance(rv_node.op, (Subtensor, AdvancedSubtensor, AdvancedSubtensor1)):

            raise NotImplementedError("Missing value support is incomplete")

            # "Flatten" and sum an array of indexed RVs' log-likelihoods
            rv_var, missing_values = rv_node.inputs

            missing_values = missing_values.data
            logp_var = at.sum(
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

        return at.zeros_like(rv_var)

    # This case should be reached when `rv_var` is either the result of an
    # `Observed` or a `RandomVariable` `Op`
    rng, size, dtype, *dist_params = rv_node.inputs

    dist_params, replacements = sample_to_measure_vars(dist_params)

    transform = getattr(rv_value_var.tag, "transform", None)

    # If any of the measure vars are transformed measure-space variables
    # (signified by having a `transform` value in their tags), then we apply
    # the their transforms and add their Jacobians (when enabled)
    if transform:
        logp_var = _logp(rv_node.op, transform.backward(rv_value), *dist_params, **kwargs)
        logp_var = transform_logp(
            logp_var,
            tuple(replacements.values()),
        )

        if jacobian:
            transformed_jacobian = transform.jacobian_det(rv_value)
            if transformed_jacobian:
                if logp_var.ndim > transformed_jacobian.ndim:
                    logp_var = logp_var.sum(axis=-1)
                logp_var += transformed_jacobian
    else:
        logp_var = _logp(rv_node.op, rv_value, *dist_params, **kwargs)

    if scaling:
        logp_var *= _get_scaling(
            getattr(rv_var.tag, "total_size", None), rv_value_var.shape, rv_value_var.ndim
        )

    if rv_var.name is not None:
        logp_var.name = "__logp_%s" % rv_var.name

    return logp_var


def transform_logp(logp_var: TensorVariable, inputs: List[TensorVariable]) -> TensorVariable:
    """Transform the inputs of a log-likelihood graph."""
    trans_replacements = {}
    for measure_var in inputs:

        transform = getattr(measure_var.tag, "transform", None)

        if transform is None:
            continue

        trans_rv_value = transform.backward(measure_var)
        trans_replacements[measure_var] = trans_rv_value

    if trans_replacements:
        (logp_var,) = clone_replace([logp_var], trans_replacements)

    return logp_var


@singledispatch
def _logp(op, value, *dist_params, **kwargs):
    """Create a log-likelihood graph.

    This function dispatches on the type of `op`, which should be a subclass
    of `RandomVariable`.  If you want to implement new log-likelihood graphs
    for a `RandomVariable`, register a new function on this dispatcher.

    """
    return at.zeros_like(value)


def logcdf(rv_var, rv_value, jacobian=True, **kwargs):
    """Create a log-CDF graph."""

    rv_var, _ = rv_log_likelihood_args(rv_var)
    rv_node = rv_var.owner

    if not rv_node:
        raise TypeError()

    rv_value = at.as_tensor(rv_value)

    rng, size, dtype, *dist_params = rv_node.inputs

    dist_params, replacements = sample_to_measure_vars(dist_params)

    logp_var = _logcdf(rv_node.op, rv_value, *dist_params, **kwargs)

    return logp_var


@singledispatch
def _logcdf(op, value, *args, **kwargs):
    """Create a log-CDF graph.

    This function dispatches on the type of `op`, which should be a subclass
    of `RandomVariable`.  If you want to implement new log-CDF graphs
    for a `RandomVariable`, register a new function on this dispatcher.

    """
    raise NotImplementedError()


def logpt_sum(rv_var: TensorVariable, rv_value: Optional[TensorVariable] = None, **kwargs):
    """Return the sum of the logp values for the given observations.

    Subclasses can use this to improve the speed of logp evaluations
    if only the sum of the logp values is needed.
    """
    return at.sum(logpt(rv_var, rv_value, **kwargs))


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
    CAR,
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
    "CAR",
]
