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
from typing import Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union

import aesara.tensor as aet
import numpy as np

from aesara import config
from aesara.graph.basic import Variable, clone_replace, graph_inputs, io_toposort, walk
from aesara.graph.op import Op, compute_test_value
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1, Subtensor
from aesara.tensor.var import TensorVariable

from pymc3.aesaraf import floatX

PotentialShapeType = Union[
    int, np.ndarray, Tuple[Union[int, Variable], ...], List[Union[int, Variable]], Variable
]

no_transform_object = object()


@singledispatch
def logp_transform(op: Op):
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


def extract_rv_and_value_vars(
    var: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable]:
    """Extract a random variable and its corresponding value variable from a generic
    `TensorVariable`.

    Parameters
    ==========
    var
        A variable corresponding to a `RandomVariable`.

    Returns
    =======
    The first value in the tuple is the `RandomVariable`, and the second is the
    measure-space variable that corresponds with the latter (i.e. the "value"
    variable).

    """
    if not var.owner:
        return None, None

    if isinstance(var.owner.op, RandomVariable):
        rv_value = getattr(var.tag, "observations", getattr(var.tag, "value_var", None))
        return var, rv_value

    return None, None


def rv_ancestors(
    graphs: Iterable[TensorVariable], walk_past_rvs: bool = False
) -> Generator[TensorVariable, None, None]:
    """Yield everything except the inputs of ``RandomVariable``s.

    Parameters
    ==========
    graphs
        The graphs to walk.
    walk_past_rvs
        If ``True``, do descend into ``RandomVariable``s.
    """

    def expand(var):
        if var.owner and (walk_past_rvs or not isinstance(var.owner.op, RandomVariable)):
            return reversed(var.owner.inputs)

    yield from walk(graphs, expand, False)


def replace_rvs_in_graphs(
    graphs: Iterable[TensorVariable],
    replacement_fn: Callable[[TensorVariable], Dict[TensorVariable, TensorVariable]],
    initial_replacements: Optional[Dict[TensorVariable, TensorVariable]] = None,
) -> Tuple[TensorVariable, Dict[TensorVariable, TensorVariable]]:
    """Replace random variables in graphs

    This will *not* recompute test values.

    Parameters
    ==========
    graphs
        The graphs in which random variables are to be replaced.

    Returns
    =======
    Tuple containing the transformed graphs and a ``dict`` of the replacements
    that were made.
    """
    replacements = {}
    if initial_replacements:
        replacements.update(initial_replacements)

    for var in rv_ancestors(graphs):
        if var.owner and isinstance(var.owner.op, RandomVariable):
            replacement_fn(var, replacements)

    if replacements:
        graphs = clone_replace(graphs, replacements)

    return graphs, replacements


def rvs_to_value_vars(
    graphs: Iterable[TensorVariable], initial_replacements: Dict[TensorVariable, TensorVariable]
) -> Tuple[Iterable[TensorVariable], Dict[TensorVariable, TensorVariable]]:
    """Replace random variables in graphs with their value variables.

    This will *not* recompute test values.
    """

    def value_var_replacements(var, replacements):
        rv_var, rv_value_var = extract_rv_and_value_vars(var)

        if rv_value_var is not None:
            replacements[var] = rv_value_var

    return replace_rvs_in_graphs(graphs, value_var_replacements, initial_replacements)


def apply_transforms(
    graphs: Iterable[TensorVariable],
) -> Tuple[TensorVariable, Dict[TensorVariable, TensorVariable]]:
    """Apply the transforms associated with each random variable in `graphs`.

    This will *not* recompute test values.
    """

    def transform_replacements(var, replacements):
        rv_var, rv_value_var = extract_rv_and_value_vars(var)

        if rv_value_var is None:
            return

        transform = getattr(rv_value_var.tag, "transform", None)

        if transform is None:
            return

        trans_rv_value = transform.backward(rv_var, rv_value_var)
        replacements[var] = trans_rv_value

    return replace_rvs_in_graphs(graphs, transform_replacements)


def logpt(
    rv_var: TensorVariable,
    rv_value: Optional[TensorVariable] = None,
    *,
    jacobian: bool = True,
    scaling: bool = True,
    transformed: bool = True,
    cdf: bool = False,
    sum: bool = False,
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
        The variable that represents the value of `rv_var` in its
        log-likelihood.  If no value is provided, `rv_var.tag.value_var` will
        be checked and, when available, used.
    jacobian
        Whether or not to include the Jacobian term.
    scaling
        A scaling term to apply to the generated log-likelihood graph.
    transformed
        Apply transforms.
    cdf
        Return the log cumulative distribution.
    sum
        Sum the log-likelihood.

    """

    rv_var, rv_value_var = extract_rv_and_value_vars(rv_var)

    if rv_value is None:

        if rv_value_var is None:
            raise ValueError(f"No value variable specified or associated with {rv_var}")

        rv_value = rv_value_var
    else:
        rv_value = aet.as_tensor(rv_value)

        # Make sure that the value is compatible with the random variable
        rv_value = rv_var.type.filter_variable(rv_value.astype(rv_var.dtype))

        if rv_value_var is None:
            rv_value_var = rv_value

    rv_node = rv_var.owner

    if not rv_node:
        return aet.zeros_like(rv_var)

    if not isinstance(rv_node.op, RandomVariable):
        return _logp(rv_node.op, rv_value, rv_node.inputs)

    rng, size, dtype, *dist_params = rv_node.inputs

    # Here, we plug the actual random variable into the log-likelihood graph,
    # because we want a log-likelihood graph that only contains
    # random variables.  This is important, because a random variable's
    # parameters can contain random variables themselves.
    # Ultimately, with a graph containing only random variables and
    # "deterministics", we can simply replace all the random variables with
    # their value variables and be done.
    if not cdf:
        logp_var = _logp(rv_node.op, rv_var, *dist_params, **kwargs)
    else:
        logp_var = _logcdf(rv_node.op, rv_var, *dist_params, **kwargs)

    if transformed and not cdf:
        (logp_var,), _ = apply_transforms((logp_var,))

    transform = getattr(rv_value_var.tag, "transform", None) if rv_value_var else None

    if transform and transformed and not cdf and jacobian:
        transformed_jacobian = transform.jacobian_det(rv_var, rv_value)
        if transformed_jacobian:
            if logp_var.ndim > transformed_jacobian.ndim:
                logp_var = logp_var.sum(axis=-1)
            logp_var += transformed_jacobian

    # Replace random variables with their value variables
    (logp_var,), replaced = rvs_to_value_vars((logp_var,), {rv_var: rv_value})

    if rv_value_var != rv_value:
        (logp_var,) = clone_replace((logp_var,), replace={rv_value_var: rv_value})

    if sum:
        logp_var = aet.sum(logp_var)

    if scaling:
        logp_var *= _get_scaling(
            getattr(rv_var.tag, "total_size", None), rv_value.shape, rv_value.ndim
        )

    # Recompute test values for the changes introduced by the replacements
    # above.
    if config.compute_test_value != "off":
        for node in io_toposort(graph_inputs((logp_var,)), (logp_var,)):
            compute_test_value(node)

    if rv_var.name is not None:
        logp_var.name = "__logp_%s" % rv_var.name

    return logp_var


@singledispatch
def _logp(op: Op, value: TensorVariable, *dist_params, **kwargs):
    """Create a log-likelihood graph.

    This function dispatches on the type of `op`, which should be a subclass
    of `RandomVariable`.  If you want to implement new log-likelihood graphs
    for a `RandomVariable`, register a new function on this dispatcher.

    """
    return aet.zeros_like(value)


@_logp.register(Subtensor)
@_logp.register(AdvancedSubtensor)
@_logp.register(AdvancedSubtensor1)
def subtensor_logp(op, value, *inputs, **kwargs):

    # TODO: Compute the log-likelihood for a subtensor/index operation.
    raise NotImplementedError()

    # "Flatten" and sum an array of indexed RVs' log-likelihoods
    # rv_var, missing_values =
    #
    # missing_values = missing_values.data
    # logp_var = aet.sum(
    #     [
    #         logpt(
    #             rv_var,
    #         )
    #         for idx, missing in zip(
    #             np.ndindex(missing_values.shape), missing_values.flatten()
    #         )
    #         if missing
    #     ]
    # )
    # return logp_var


def logcdf(*args, **kwargs):
    """Create a log-CDF graph."""
    return logpt(*args, cdf=True, **kwargs)


@singledispatch
def _logcdf(op, value, *args, **kwargs):
    """Create a log-CDF graph.

    This function dispatches on the type of `op`, which should be a subclass
    of `RandomVariable`.  If you want to implement new log-CDF graphs
    for a `RandomVariable`, register a new function on this dispatcher.

    """
    raise NotImplementedError()


def logpt_sum(*args, **kwargs):
    """Return the sum of the logp values for the given observations.

    Subclasses can use this to improve the speed of logp evaluations
    if only the sum of the logp values is needed.
    """
    return logpt(*args, sum=True, **kwargs)


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
