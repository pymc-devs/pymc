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
import functools as ft
import itertools as it

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytensor
import pytensor.tensor as pt
import xarray as xr

from arviz import InferenceData
from numpy import random as nr
from numpy import testing as npt
from numpy.typing import NDArray
from pytensor.compile import SharedVariable
from pytensor.compile.mode import Mode, get_default_mode
from pytensor.graph.basic import Constant, Variable, equal_computations
from pytensor.graph.rewriting.basic import in2out
from pytensor.graph.traversal import graph_inputs
from pytensor.link.numba import NumbaLinker
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import RandomType
from scipy import special as sp
from scipy import stats as st

import pymc as pm

from pymc.distributions.distribution import Distribution
from pymc.distributions.shape_utils import change_dist_size
from pymc.initial_point import make_initial_point_fn
from pymc.logprob.basic import icdf, logcdf, logp, transformed_conditional_logp
from pymc.logprob.utils import (
    ParameterValueError,
    local_check_parameter_to_ninf_switch,
)
from pymc.pytensorf import compile, floatX, inputvars, rvs_in_graph

# This mode can be used for tests where model compilations takes the bulk of the runtime
# AND where we don't care about posterior numerical or sampling stability (e.g., when
# all that matters are the shape of the draws or deterministic values of observed data).
# DO NOT USE UNLESS YOU HAVE A GOOD REASON TO!
fast_unstable_sampling_mode = (
    pytensor.compile.mode.FAST_COMPILE
    # Remove slow rewrite phases
    .excluding("canonicalize", "specialize")
    # Include necessary rewrites for proper logp handling
    .including("remove_TransformedVariables")
    .register((in2out(local_check_parameter_to_ninf_switch), -1))
)


def product(domains, n_samples=-1):
    """Get an iterator over a product of domains.

    Args:
    ----
        domains: a dictionary of (name, object) pairs, where the objects
                 must be "domain-like", as in, have a `.vals` property
        n_samples: int, maximum samples to return.  -1 to return whole product

    Returns
    -------
        list of the cartesian product of the domains
    """
    try:
        names, domains = zip(*domains.items())
    except ValueError:  # domains.items() is empty
        return [{}]
    all_vals = [zip(names, val) for val in it.product(*(d.vals for d in domains))]
    if n_samples > 0 and len(all_vals) > n_samples:
        return (all_vals[j] for j in nr.choice(len(all_vals), n_samples, replace=False))
    return all_vals


class Domain:
    def __init__(self, vals, dtype=pytensor.config.floatX, edges=None, shape=None):
        # Infinity values must be kept as floats
        vals = [np.array(v, dtype=dtype) if np.all(np.isfinite(v)) else floatX(v) for v in vals]

        if edges is None:
            edges = np.array(vals[0]), np.array(vals[-1])
            vals = vals[1:-1]
        else:
            edges = list(edges)
            if edges[0] is None:
                edges[0] = np.full_like(vals[0], -np.inf)
            if edges[1] is None:
                edges[1] = np.full_like(vals[0], np.inf)
            edges = tuple(edges)

        if not vals:
            raise ValueError(
                f"Domain has no values left after removing edges: {edges}.\n"
                "You can duplicate the edge values or explicitly specify the edges with the edge keyword.\n"
                f"For example: `Domain([{edges[0]}, {edges[0]}, {edges[1]}, {edges[1]}])`"
            )

        if shape is None:
            shape = vals[0].shape

        self.vals = vals
        self.shape = shape
        self.lower, self.upper = edges
        self.dtype = dtype

    def __add__(self, other):
        """Add two domains."""
        return Domain(
            [v + other for v in self.vals],
            self.dtype,
            (self.lower + other, self.upper + other),
            self.shape,
        )

    def __mul__(self, other):
        """Multiply two domains."""
        try:
            return Domain(
                [v * other for v in self.vals],
                self.dtype,
                (self.lower * other, self.upper * other),
                self.shape,
            )
        except TypeError:
            return Domain(
                [v * other for v in self.vals],
                self.dtype,
                (self.lower, self.upper),
                self.shape,
            )

    def __neg__(self):
        """Negate one domain."""
        return Domain([-v for v in self.vals], self.dtype, (-self.lower, -self.upper), self.shape)


class ProductDomain:
    def __init__(self, domains):
        self.vals = list(it.product(*(d.vals for d in domains)))
        self.shape = (len(domains), *domains[0].shape)
        self.lower = [d.lower for d in domains]
        self.upper = [d.upper for d in domains]
        self.dtype = domains[0].dtype


def Vector(D, n):
    return ProductDomain([D] * n)


def SortedVector(n):
    vals = []
    np.random.seed(42)
    for _ in range(10):
        vals.append(np.sort(np.random.randn(n)))
    return Domain(vals, edges=(None, None))


def UnitSortedVector(n):
    vals = []
    np.random.seed(42)
    for _ in range(10):
        vals.append(np.sort(np.random.rand(n)))
    return Domain(vals, edges=(None, None))


def RealMatrix(n, m):
    vals = []
    np.random.seed(42)
    for _ in range(10):
        vals.append(np.random.randn(n, m))
    return Domain(vals, edges=(None, None))


def simplex_values(n):
    if n == 1:
        yield np.array([1.0])
    else:
        for v in Unit.vals:
            for vals in simplex_values(n - 1):
                yield np.concatenate([[v], (1 - v) * vals])


def Simplex(n):
    return Domain(simplex_values(n), shape=(n,), dtype=Unit.dtype, edges=(None, None))


def MultiSimplex(n_dependent, n_independent):
    vals = []
    for simplex_value in it.product(simplex_values(n_dependent), repeat=n_independent):
        vals.append(np.vstack(simplex_value))

    return Domain(vals, dtype=Unit.dtype, shape=(n_independent, n_dependent))


def RandomPdMatrix(n):
    A = np.random.rand(n, n)
    return np.dot(A, A.T) + n * np.identity(n)


R = Domain([-np.inf, -2.1, -1, -0.01, 0.0, 0.01, 1, 2.1, np.inf])
Rplus = Domain([0, 0.01, 0.1, 0.9, 0.99, 1, 1.5, 2, 100, np.inf])
Rplusbig = Domain([0, 0.5, 0.9, 0.99, 1, 1.5, 2, 20, np.inf])
Rminusbig = Domain([-np.inf, -2, -1.5, -1, -0.99, -0.9, -0.5, -0.01, 0])
Unit = Domain([0, 0.001, 0.1, 0.5, 0.75, 0.99, 1])
Circ = Domain([-np.pi, -2.1, -1, -0.01, 0.0, 0.01, 1, 2.1, np.pi])
Runif = Domain([-np.inf, -0.4, 0, 0.4, np.inf])
Rdunif = Domain([-np.inf, -1, 0, 1, np.inf], "int64")
Rplusunif = Domain([0, 0.5, np.inf])
Rplusdunif = Domain([0, 10, np.inf], "int64")
I = Domain([-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf], "int64")  # noqa: E741
NatSmall = Domain([0, 3, 4, 5, np.inf], "int64")
Nat = Domain([0, 1, 2, 3, np.inf], "int64")
NatBig = Domain([0, 1, 2, 3, 5000, np.inf], "int64")
PosNat = Domain([1, 2, 3, np.inf], "int64")
Bool = Domain([0, 0, 1, 1], "int64")


def select_by_precision(float64, float32):
    """Choose reasonable decimal cutoffs for different floatX modes."""
    decimal = float64 if pytensor.config.floatX == "float64" else float32
    return decimal


def build_model(distfam, valuedomain, vardomains, extra_args=None):
    if extra_args is None:
        extra_args = {}

    with pm.Model() as m:
        param_vars = {}
        for v, dom in vardomains.items():
            v_pt = pytensor.shared(np.asarray(dom.vals[0]))
            v_pt.name = v
            param_vars[v] = v_pt
        param_vars.update(extra_args)
        distfam(
            "value",
            **param_vars,
            default_transform=None,
        )
    return m, param_vars


def create_dist_from_paramdomains(
    pymc_dist: Distribution,
    paramdomains: dict[str, Domain],
    extra_args: dict[str, Any] | None = None,
) -> TensorVariable:
    """Create a PyMC distribution from a dictionary of parameter domains.

    Returns
    -------
        PyMC distribution variable: TensorVariable
        Value variable: TensorVariable
    """
    if extra_args is None:
        extra_args = {}

    param_vars = {}
    for param, domain in paramdomains.items():
        param_type = pt.constant(np.asarray(domain.vals[0])).type()
        param_type.name = param
        param_vars[param] = param_type

    return pymc_dist.dist(**param_vars, **extra_args)


def find_invalid_scalar_params(
    paramdomains: dict["str", Domain],
) -> dict["str", tuple[None | float, None | float]]:
    """Find invalid parameter values from bounded scalar parameter domains.

    For use in `check_logp`-like testing helpers.

    Returns
    -------
    Invalid paramemeter values:
        Dictionary mapping each parameter, to a lower and upper invalid values (out of domain).
        If no lower or upper invalid values exist, None is returned for that entry.
    """
    invalid_params = {}
    for param, paramdomain in paramdomains.items():
        lower_edge, upper_edge = None, None

        if np.ndim(paramdomain.lower) == 0:
            if np.isfinite(paramdomain.lower):
                lower_edge = paramdomain.lower - 1

            if np.isfinite(paramdomain.upper):
                upper_edge = paramdomain.upper + 1

        invalid_params[param] = (lower_edge, upper_edge)
    return invalid_params


def check_logp(
    pymc_dist: Distribution,
    domain: Domain,
    paramdomains: dict[str, Domain],
    scipy_logp: Callable,
    decimal: int | None = None,
    n_samples: int = 100,
    extra_args: dict[str, Any] | None = None,
    scipy_args: dict[str, Any] | None = None,
    skip_paramdomain_outside_edge_test: bool = False,
) -> None:
    """
    Test PyMC logp and equivalent scipy logpmf/logpdf methods give similar results for valid values and parameters inside the supported edges.

    Edges are excluded by default, but can be artificially included by
    creating a domain with repeated values (e.g., `Domain([0, 0, .5, 1, 1]`)

    Parameters
    ----------
    pymc_dist: PyMC distribution
    domain : Domain
        Supported domain of distribution values
    paramdomains : Dictionary of Parameter : Domain pairs
        Supported domains of distribution parameters
    scipy_logp : Scipy logpmf/logpdf method
        Scipy logp method of equivalent pymc_dist distribution
    decimal : Int
        Level of precision with which pymc_dist and scipy logp are compared.
        Defaults to 6 for float64 and 3 for float32
    n_samples : Int
        Upper limit on the number of valid domain and value combinations that
        are compared between pymc and scipy methods. If n_samples is below the
        total number of combinations, a random subset is evaluated. Setting
        n_samples = -1, will return all possible combinations. Defaults to 100
    extra_args : Dictionary with extra arguments needed to build pymc model
        Dictionary is passed to helper function `build_model` from which
        the pymc distribution logp is calculated
    scipy_args : Dictionary with extra arguments needed to call scipy logp method
        Usually the same as extra_args
    """
    import pytest

    if decimal is None:
        decimal = select_by_precision(float64=6, float32=3)

    if scipy_args is None:
        scipy_args = {}

    def scipy_logp_with_scipy_args(**args):
        args.update(scipy_args)
        return scipy_logp(**args)

    dist = create_dist_from_paramdomains(pymc_dist, paramdomains, extra_args)
    value = dist.type()
    value.name = "value"
    pymc_dist_logp = logp(dist, value).sum()
    pymc_logp = pytensor.function(list(inputvars(pymc_dist_logp)), pymc_dist_logp)

    # Test supported value and parameters domain matches Scipy
    domains = paramdomains.copy()
    domains["value"] = domain
    for point in product(domains, n_samples=n_samples):
        point = dict(point)
        npt.assert_almost_equal(
            pymc_logp(**point),
            scipy_logp_with_scipy_args(**point),
            decimal=decimal,
            err_msg=str(point),
        )

    valid_value = domain.vals[0]
    valid_params = {param: paramdomain.vals[0] for param, paramdomain in paramdomains.items()}
    valid_params["value"] = valid_value

    # Test pymc distribution raises ParameterValueError for scalar parameters outside
    # the supported domain edges (excluding edges)
    if not skip_paramdomain_outside_edge_test:
        invalid_params = find_invalid_scalar_params(paramdomains)

        for invalid_param, invalid_edges in invalid_params.items():
            for invalid_edge in invalid_edges:
                if invalid_edge is None:
                    continue

                point = valid_params.copy()  # Shallow copy should be okay
                point[invalid_param] = np.asarray(
                    invalid_edge, dtype=paramdomains[invalid_param].dtype
                )

                with pytest.raises(ParameterValueError):
                    pymc_logp(**point)
                    pytest.fail(f"test_params={point}")

    # Test that values outside of scalar domain support evaluate to -np.inf
    invalid_values = find_invalid_scalar_params({"value": domain})["value"]

    for invalid_value in invalid_values:
        if invalid_value is None:
            continue

        point = valid_params.copy()
        point["value"] = invalid_value
        npt.assert_equal(
            pymc_logp(**point),
            -np.inf,
            err_msg=str(point),
        )


def check_logcdf(
    pymc_dist: Distribution,
    domain: Domain,
    paramdomains: dict[str, Domain],
    scipy_logcdf: Callable,
    decimal: int | None = None,
    n_samples: int = 100,
    skip_paramdomain_inside_edge_test: bool = False,
    skip_paramdomain_outside_edge_test: bool = False,
) -> None:
    """
    Test PyMC logcdf and equivalent scipy logcdf methods give similar results for valid values and parameters inside the supported edges.

    The following tests are performed by default:
        1. Test PyMC logcdf and equivalent scipy logcdf methods give similar
        results for valid values and parameters inside the supported edges.
        Edges are excluded by default, but can be artificially included by
        creating a domain with repeated values (e.g., `Domain([0, 0, .5, 1, 1]`)
        Can be skipped via skip_paramdomain_inside_edge_test
        2. Test PyMC logcdf method returns -inf for invalid parameter values
        outside the supported edges. Can be skipped via skip_paramdomain_outside_edge_test
        3. Test PyMC logcdf method returns -inf and 0 for values below and
        above the supported edge, respectively, when using valid parameters.
        4. Test PyMC logcdf methods works with multiple value or returns
        default informative TypeError

    Parameters
    ----------
    pymc_dist: PyMC distribution
    domain : Domain
        Supported domain of distribution values
    paramdomains : Dictionary of Parameter : Domain pairs
        Supported domains of distribution parameters
    scipy_logcdf : Scipy logcdf method
        Scipy logcdf method of equivalent pymc_dist distribution
    decimal : Int
        Level of precision with which pymc_dist and scipy_logcdf are compared.
        Defaults to 6 for float64 and 3 for float32
    n_samples : Int
        Upper limit on the number of valid domain and value combinations that
        are compared between pymc and scipy methods. If n_samples is below the
        total number of combinations, a random subset is evaluated. Setting
        n_samples = -1, will return all possible combinations. Defaults to 100
    skip_paramdomain_inside_edge_test : Bool
        Whether to run test 1., which checks that pymc and scipy distributions
        match for valid values and parameters inside the respective domain edges
    skip_paramdomain_outside_edge_test : Bool
        Whether to run test 2., which checks that pymc distribution logcdf
        returns -inf for invalid parameter values outside the supported domain edge

    """
    import pytest

    if decimal is None:
        decimal = select_by_precision(float64=6, float32=3)

    dist = create_dist_from_paramdomains(pymc_dist, paramdomains)
    value = dist.type()
    value.name = "value"
    dist_logcdf = logcdf(dist, value)
    pymc_logcdf = pytensor.function(list(inputvars(dist_logcdf)), dist_logcdf)

    # Test pymc and scipy distributions match for values and parameters
    # within the supported domain edges (excluding edges)
    if not skip_paramdomain_inside_edge_test:
        domains = paramdomains.copy()
        domains["value"] = domain
        for point in product(domains, n_samples=n_samples):
            point = dict(point)
            npt.assert_almost_equal(
                pymc_logcdf(**point),
                scipy_logcdf(**point),
                decimal=decimal,
                err_msg=str(point),
            )

    valid_value = domain.vals[0]
    valid_params = {param: paramdomain.vals[0] for param, paramdomain in paramdomains.items()}
    valid_params["value"] = valid_value

    # Test pymc distribution raises ParameterValueError for parameters outside the
    # supported domain edges (excluding edges)
    if not skip_paramdomain_outside_edge_test:
        invalid_params = find_invalid_scalar_params(paramdomains)

        for invalid_param, invalid_edges in invalid_params.items():
            for invalid_edge in invalid_edges:
                if invalid_edge is None:
                    continue

                point = valid_params.copy()
                point[invalid_param] = invalid_edge

                with pytest.raises(ParameterValueError):
                    pymc_logcdf(**point)
                    pytest.fail(f"test_params={point}")

    # Test that values below domain edge evaluate to -np.inf, and above evaluates to 0
    invalid_lower, invalid_upper = find_invalid_scalar_params({"value": domain})["value"]
    if invalid_lower is not None:
        point = valid_params.copy()
        point["value"] = invalid_lower
        npt.assert_equal(
            pymc_logcdf(**point),
            -np.inf,
            err_msg=str(point),
        )
    if invalid_upper is not None:
        point = valid_params.copy()
        point["value"] = invalid_upper
        npt.assert_equal(
            pymc_logcdf(**point),
            0,
            err_msg=str(point),
        )


def check_icdf(
    pymc_dist: Distribution,
    paramdomains: dict[str, Domain],
    scipy_icdf: Callable,
    skip_paramdomain_outside_edge_test=False,
    decimal: int | None = None,
    n_samples: int = 100,
) -> None:
    """
    Test PyMC icdf and equivalent scipy icdf methods give similar results for valid values and parameters inside the supported edges.

    The following tests are performed by default:
        1. Test PyMC icdf and equivalent scipy icdf (ppf) methods give similar
        results for parameters inside the supported edges.
        Edges are excluded by default, but can be artificially included by
        creating a domain with repeated values (e.g., `Domain([0, 0, .5, 1, 1]`)
        2. Test PyMC icdf method raises for invalid parameter values
        outside the supported edges.
        3. Test PyMC icdf method returns np.nan for values below 0 or above 1,
         when using valid parameters.

    Parameters
    ----------
    pymc_dist: PyMC distribution
    paramdomains : Dictionary of Parameter : Domain pairs
        Supported domains of distribution parameters
    scipy_icdf : Scipy icdf method
        Scipy icdf (ppf) method of equivalent pymc_dist distribution
    decimal : int, optional
        Level of precision with which pymc_dist and scipy_icdf are compared.
        Defaults to 6 for float64 and 3 for float32
    n_samples : int
        Upper limit on the number of valid domain and value combinations that
        are compared between pymc and scipy methods. If n_samples is below the
        total number of combinations, a random subset is evaluated. Setting
        n_samples = -1, will return all possible combinations. Defaults to 100
    skip_paradomain_outside_edge_test : Bool
        Whether to run test 2., which checks that pymc distribution icdf
        returns nan for invalid parameter values outside the supported domain edge

    """
    import pytest

    if decimal is None:
        decimal = select_by_precision(float64=6, float32=3)

    dist = create_dist_from_paramdomains(pymc_dist, paramdomains)
    q = pt.scalar(dtype="float64", name="q")
    dist_icdf = icdf(dist, q)
    pymc_icdf = pytensor.function(list(inputvars(dist_icdf)), dist_icdf)

    # Test pymc and scipy distributions match for values and parameters
    # within the supported domain edges (excluding edges)
    domains = paramdomains.copy()
    domain = Domain([0, 0.1, 0.5, 0.75, 0.95, 0.99, 1])  # Values we test the icdf at
    domains["q"] = domain

    for point in product(domains, n_samples=n_samples):
        point = dict(point)
        npt.assert_almost_equal(
            pymc_icdf(**point),
            scipy_icdf(**point),
            decimal=decimal,
            err_msg=str(point),
        )

    valid_value = domain.vals[0]
    valid_params = {param: paramdomain.vals[0] for param, paramdomain in paramdomains.items()}
    valid_params["q"] = valid_value

    if not skip_paramdomain_outside_edge_test:
        # Test pymc distribution raises ParameterValueError for parameters outside the
        # supported domain edges (excluding edges)
        invalid_params = find_invalid_scalar_params(paramdomains)
        for invalid_param, invalid_edges in invalid_params.items():
            for invalid_edge in invalid_edges:
                if invalid_edge is None:
                    continue

                point = valid_params.copy()
                point[invalid_param] = invalid_edge

                with pytest.raises(ParameterValueError):
                    pymc_icdf(**point)
                    pytest.fail(f"test_params={point}")

    # Test that values below 0 or above 1 evaluate to nan
    invalid_values = find_invalid_scalar_params({"q": domain})["q"]
    for invalid_value in invalid_values:
        if invalid_value is not None:
            point = valid_params.copy()
            point["q"] = invalid_value
            npt.assert_equal(
                pymc_icdf(**point),
                np.nan,
                err_msg=str(point),
            )


def check_selfconsistency_discrete_logcdf(
    distribution: Distribution,
    domain: Domain,
    paramdomains: dict[str, Domain],
    decimal: int | None = None,
    n_samples: int = 100,
) -> None:
    """Check that logcdf of discrete distributions matches sum of logps up to value."""
    if decimal is None:
        decimal = select_by_precision(float64=6, float32=3)

    dist = create_dist_from_paramdomains(distribution, paramdomains)
    value = dist.type()
    value.name = "value"
    dist_logp = logp(dist, value)
    dist_logp_fn = pytensor.function(list(inputvars(dist_logp)), dist_logp)

    dist_logcdf = logcdf(dist, value)
    dist_logcdf_fn = compile(list(inputvars(dist_logcdf)), dist_logcdf)

    domains = paramdomains.copy()
    domains["value"] = domain

    for point in product(domains, n_samples=n_samples):
        point = dict(point)
        value = point.pop("value")
        values = np.arange(domain.lower, value + 1)

        with pytensor.config.change_flags(mode=Mode("py")):
            npt.assert_almost_equal(
                dist_logcdf_fn(**point, value=value),
                sp.logsumexp([dist_logp_fn(value=value, **point) for value in values]),
                decimal=decimal,
                err_msg=str(point),
            )


def check_selfconsistency_icdf(
    distribution: Distribution,
    paramdomains: dict[str, Domain],
    *,
    decimal: int | None = None,
    n_samples: int = 100,
) -> None:
    """Check that the icdf and logcdf functions of the distribution are consistent.

    Only works with continuous distributions.
    """
    if decimal is None:
        decimal = select_by_precision(float64=6, float32=3)

    dist = create_dist_from_paramdomains(distribution, paramdomains)
    if dist.type.dtype.startswith("int"):
        raise NotImplementedError(
            "check_selfconsistency_icdf is not robust against discrete distributions."
        )
    value = dist.astype("float64").type("value")
    dist_icdf = icdf(dist, value)
    dist_cdf = pt.exp(logcdf(dist, value))

    py_mode = Mode("py")
    dist_icdf_fn = pytensor.function(list(inputvars(dist_icdf)), dist_icdf, mode=py_mode)
    dist_cdf_fn = compile(list(inputvars(dist_cdf)), dist_cdf, mode=py_mode)

    domains = paramdomains.copy()
    domains["value"] = Domain(np.linspace(0, 1, 10))

    for point in product(domains, n_samples=n_samples):
        point = dict(point)
        value = point.pop("value")
        icdf_value = dist_icdf_fn(**point, value=value)
        recovered_value = dist_cdf_fn(
            **point,
            value=icdf_value,
        )
        np.testing.assert_almost_equal(
            value,
            recovered_value,
            decimal=decimal,
            err_msg=f"point: {point}",
        )


def assert_support_point_is_expected(model, expected, check_finite_logp=True):
    fn = make_initial_point_fn(
        model=model,
        return_transformed=False,
        default_strategy="support_point",
    )
    support_point = fn(0)["x"]
    expected = np.asarray(expected)
    try:
        random_draw = model["x"].eval()
    except NotImplementedError:
        random_draw = support_point

    assert support_point.shape == expected.shape
    assert expected.shape == random_draw.shape
    assert np.allclose(support_point, expected)

    if check_finite_logp:
        logp_support_point = (
            transformed_conditional_logp(
                (model["x"],),
                rvs_to_values={model["x"]: pt.constant(support_point)},
                rvs_to_transforms={},
            )[0]
            .sum()
            .eval()
        )
        assert np.isfinite(logp_support_point)


def continuous_random_tester(
    dist,
    paramdomains,
    ref_rand,
    valuedomain=None,
    size=10000,
    alpha=0.05,
    fails=10,
    extra_args=None,
    model_args=None,
):
    if valuedomain is None:
        valuedomain = Domain([0], edges=(None, None))

    if model_args is None:
        model_args = {}

    model, param_vars = build_model(dist, valuedomain, paramdomains, extra_args)
    model_dist = change_dist_size(model.named_vars["value"], size, expand=True)
    pymc_rand = compile([], model_dist)

    domains = paramdomains.copy()
    for point in product(domains, n_samples=100):
        point = pm.Point(point, model=model)
        point.update(model_args)

        # Update the shared parameter variables in `param_vars`
        for k, v in point.items():
            nv = param_vars.get(k, model.named_vars.get(k))
            if nv.name in param_vars:
                param_vars[nv.name].set_value(v)

        p = alpha
        # Allow KS test to fail (i.e., the samples be different)
        # a certain number of times. Crude, but necessary.
        f = fails
        while p <= alpha and f > 0:
            s0 = pymc_rand()
            s1 = floatX(ref_rand(size=size, **point))
            _, p = st.ks_2samp(np.atleast_1d(s0).flatten(), np.atleast_1d(s1).flatten())
            f -= 1
        assert p > alpha, str(point)


def discrete_random_tester(
    dist,
    paramdomains,
    valuedomain=None,
    ref_rand=None,
    size=100000,
    alpha=0.05,
    fails=20,
):
    if valuedomain is None:
        valuedomain = Domain([0], edges=(None, None))

    model, param_vars = build_model(dist, valuedomain, paramdomains)
    model_dist = change_dist_size(model.named_vars["value"], size, expand=True)
    pymc_rand = compile([], model_dist)

    domains = paramdomains.copy()
    for point in product(domains, n_samples=100):
        point = pm.Point(point, model=model)
        p = alpha

        # Update the shared parameter variables in `param_vars`
        for k, v in point.items():
            nv = param_vars.get(k, model.named_vars.get(k))
            if nv.name in param_vars:
                param_vars[nv.name].set_value(v)

        # Allow Chisq test to fail (i.e., the samples be different)
        # a certain number of times.
        f = fails
        while p <= alpha and f > 0:
            o = pymc_rand()
            e = ref_rand(size=size, **point).astype(int)
            o = np.atleast_1d(o).flatten()
            e = np.atleast_1d(e).flatten()
            bins = min(20, max(len(set(e)), len(set(o))))
            range = (min(min(e), min(o)), max(max(e), max(o)))
            observed, _ = np.histogram(o, bins=bins, range=range)
            expected, _ = np.histogram(e, bins=bins, range=range)
            if np.all(observed == expected):
                p = 1.0
            else:
                _, p = st.chisquare(observed + 1, expected + 1)
            f -= 1
        assert p > alpha, str(point)


class BaseTestDistributionRandom:
    """
    Base class for tests that new RandomVariables are correctly implemented.

    Also checks that the mapping of parameters between the PyMC
    Distribution and the respective RandomVariable is correct.

    Three default tests are provided which check:
    1. Expected inputs are passed to the `rv_op` by the `dist` `classmethod`,
    via `check_pymc_params_match_rv_op`
    2. Expected (exact) draws are being returned, via
    `check_pymc_draws_match_reference`
    3. Shape variable inference is correct, via `check_rv_size`

    Each desired test must be referenced by name in `checks_to_run`, when
    subclassing this distribution. Custom tests can be added to each class as
    well. See `TestFlat` for an example.

    Additional tests should be added for each optional parametrization of the
    distribution. In this case it's enough to include the test
    `check_pymc_params_match_rv_op` since only this differs.

    Note on `check_rv_size` test:
        Custom input sizes (and expected output shapes) can be defined for the
        `check_rv_size` test, by adding the optional class attributes
        `sizes_to_check` and `sizes_expected`:

        ```python
        sizes_to_check = [None, (1), (2, 3)]
        sizes_expected = [(3,), (1, 3), (2, 3, 3)]
        checks_to_run = ["check_rv_size"]
        ```

        This is usually needed for Multivariate distributions. You can see an
        example in `TestDirichlet`


    Notes on `check_pymcs_draws_match_reference` test:

        The `check_pymcs_draws_match_reference` is a very simple test for the
        equality of draws from the `RandomVariable` and the exact same python
        function, given the  same inputs and random seed. A small number
        (`size=15`) is checked. This is not supposed to be a test for the
        correctness of the random generator. The latter kind of test
        (if warranted) can be performed with the aid of `pymc_random` and
        `pymc_random_discrete` methods in this file, which will perform an
        expensive statistical comparison between the RandomVariable `rng_fn`
        and a reference Python function. This kind of test only makes sense if
        there is a good independent generator reference (i.e., not just the same
        composition of numpy / scipy python calls that is done inside `rng_fn`).

        Finally, when your `rng_fn` is doing something more than just calling a
        `numpy` or `scipy` method, you will need to setup an equivalent seeded
        function with which to compare for the exact draws (instead of relying on
        `seeded_[scipy|numpy]_distribution_builder`). You can find an example
        in the `TestWeibull`, whose `rng_fn` returns
        `beta * np.random.weibull(alpha, size=size)`.

    """

    pymc_dist: Callable | None = None
    pymc_dist_params: dict | None = None
    reference_dist: Callable | None = None
    reference_dist_params: dict | None = None
    expected_rv_op_params: dict | None = None
    checks_to_run: list[str] = []
    size = 15
    decimal = select_by_precision(float64=6, float32=3)

    sizes_to_check: list | None = None
    sizes_expected: list | None = None
    repeated_params_shape = 5
    random_state = None

    def test_distribution(self):
        import pytest

        self.validate_tests_list()
        if self.pymc_dist == pm.Wishart:
            with pytest.warns(UserWarning, match="can currently not be used for MCMC sampling"):
                self._instantiate_pymc_rv()
        else:
            self._instantiate_pymc_rv()
        if self.reference_dist is not None:
            self.reference_dist_draws = self.reference_dist()(
                size=self.size, **self.reference_dist_params
            )
        for check_name in self.checks_to_run:
            if check_name.startswith("test_"):
                raise ValueError(
                    "Custom check cannot start with `test_` or else it will be executed twice."
                )
            if self.pymc_dist == pm.Wishart and check_name.startswith("check_rv_size"):
                with pytest.warns(UserWarning, match="can currently not be used for MCMC sampling"):
                    getattr(self, check_name)()
            else:
                getattr(self, check_name)()

    def get_random_state(self, reset=False):
        if self.random_state is None or reset:
            self.random_state = nr.default_rng(20160911)
        return self.random_state

    def _instantiate_pymc_rv(self, dist_params=None):
        params = dist_params if dist_params else self.pymc_dist_params
        self.pymc_rv = self.pymc_dist.dist(
            **params, size=self.size, rng=pytensor.shared(self.get_random_state(reset=True))
        )

    def check_pymc_draws_match_reference(self):
        # need to re-instantiate it to make sure that the order of drawings match the reference distribution one
        # self._instantiate_pymc_rv()
        npt.assert_array_almost_equal(
            self.pymc_rv.eval(), self.reference_dist_draws, decimal=self.decimal
        )

    def check_pymc_draws_match_reference_not_numba(self):
        # This calls `check_pymc_draws_match_reference` but only if the default linker is NOT numba.
        # It's used when the draws aren't expected to match in that backend.
        if isinstance(get_default_mode().linker, NumbaLinker):
            return
        npt.assert_array_almost_equal(
            self.pymc_rv.eval(), self.reference_dist_draws, decimal=self.decimal
        )

    def check_pymc_params_match_rv_op(self):
        op = self.pymc_rv.owner.op
        if isinstance(op, RandomVariable):
            pytensor_dist_inputs = op.dist_params(self.pymc_rv.owner)
        else:
            extended_signature = op.extended_signature
            if extended_signature is None:
                raise NotImplementedError("Op requires extended signature to be tested")
            [_, _, dist_params_idxs], _ = op.get_input_output_type_idxs(extended_signature)
            dist_inputs = self.pymc_rv.owner.inputs
            pytensor_dist_inputs = [dist_inputs[i] for i in dist_params_idxs]

        assert len(self.expected_rv_op_params) == len(pytensor_dist_inputs)
        for (expected_name, expected_value), actual_variable in zip(
            self.expected_rv_op_params.items(), pytensor_dist_inputs
        ):
            # Add additional line to evaluate symbolic inputs to distributions
            if isinstance(expected_value, pytensor.tensor.Variable):
                expected_value = expected_value.eval()

            # RVs introduce expand_dims on the parameters, but the tests do not expect this
            implicit_expand_dims = actual_variable.type.ndim - np.ndim(expected_value)
            actual_variable = actual_variable.squeeze(tuple(range(implicit_expand_dims)))
            npt.assert_almost_equal(expected_value, actual_variable.eval(), decimal=self.decimal)

    def check_rv_size(self):
        # test sizes
        sizes_to_check = self.sizes_to_check or [None, (), 1, (1,), 5, (4, 5), (2, 4, 2)]
        sizes_expected = self.sizes_expected or [(), (), (1,), (1,), (5,), (4, 5), (2, 4, 2)]
        for size, expected in zip(sizes_to_check, sizes_expected):
            rv = self.pymc_dist.dist(**self.pymc_dist_params, size=size)
            expected_symbolic = tuple(rv.shape.eval())
            actual = rv.eval().shape
            assert actual == expected_symbolic
            assert expected_symbolic == expected, (size, expected_symbolic, expected)

        # test multi-parameters sampling for univariate distributions (with univariate inputs)
        rv_op = rv.owner.op
        if rv_op.ndim_supp == 0 and rv_op.ndims_params == 0:
            params = {
                k: p * np.ones(self.repeated_params_shape) for k, p in self.pymc_dist_params.items()
            }
            sizes_to_check = [None, self.repeated_params_shape, (5, self.repeated_params_shape)]
            sizes_expected = [
                (self.repeated_params_shape,),
                (self.repeated_params_shape,),
                (5, self.repeated_params_shape),
            ]
            for size, expected in zip(sizes_to_check, sizes_expected):
                rv = self.pymc_dist.dist(**params, size=size)
                expected_symbolic = tuple(rv.shape.eval())
                actual = rv.eval().shape
                assert actual == expected_symbolic == expected

    def validate_tests_list(self):
        assert len(self.checks_to_run) == len(set(self.checks_to_run)), (
            "There are duplicates in the list of checks_to_run"
        )


def seeded_scipy_distribution_builder(dist_name: str) -> Callable:
    return lambda self: ft.partial(getattr(st, dist_name).rvs, random_state=self.get_random_state())


def seeded_numpy_distribution_builder(dist_name: str) -> Callable:
    return lambda self: getattr(self.get_random_state(), dist_name)


def assert_no_rvs(vars: Sequence[Variable]) -> None:
    """Assert that there are no `MeasurableOp` nodes in a graph."""
    if rvs := rvs_in_graph(vars):
        raise AssertionError(f"RV found in graph: {rvs}")


SampleStatsCreator = Callable[[tuple[int, int]], NDArray]


def mock_sample(
    draws: int = 10,
    sample_stats: dict[str, SampleStatsCreator] | None = None,
    **kwargs,
) -> InferenceData:
    """Mock :func:`pymc.sample` with :func:`pymc.sample_prior_predictive`.

    Useful for testing models that use pm.sample without running MCMC sampling.

    Examples
    --------
    Using mock_sample with pytest

    .. note::

        Use :func:`pymc.testing.mock_sample_setup_and_teardown` directly for pytest fixtures.

    .. code-block:: python

        import pytest

        import pymc as pm
        from pymc.testing import mock_sample


        @pytest.fixture(scope="module")
        def mock_pymc_sample():
            original_sample = pm.sample
            pm.sample = mock_sample

            yield

            pm.sample = original_sample

    By default, the sample_stats group is not created. Pass a dictionary of functions
    that create sample statistics, where the keys are the names of the statistics
    and the values are functions that take a size tuple and return an array of that size.

    .. code-block:: python

        from functools import partial

        import numpy as np
        from numpy.typing import NDArray

        from pymc.testing import mock_sample


        def mock_diverging(size: tuple[int, int]) -> NDArray:
            return np.zeros(size)


        def mock_tree_depth(size: tuple[int, int]) -> NDArray:
            return np.random.choice(range(2, 10), size=size)


        mock_sample_with_stats = partial(
            mock_sample,
            sample_stats={
                "diverging": mock_diverging,
                "tree_depth": mock_tree_depth,
            },
        )

    """
    random_seed = kwargs.get("random_seed", None)
    model = kwargs.get("model", None)
    draws = kwargs.get("draws", draws)
    n_chains = kwargs.get("chains", 1)
    var_names = kwargs.get("var_names", None)
    idata: InferenceData = pm.sample_prior_predictive(
        model=model,
        random_seed=random_seed,
        draws=draws,
        var_names=var_names,
    )

    idata.add_groups(
        posterior=(
            idata["prior"]
            .isel(chain=0)
            .expand_dims({"chain": range(n_chains)})
            .transpose("chain", "draw", ...)
        )
    )
    del idata["prior"]
    if "prior_predictive" in idata:
        del idata["prior_predictive"]

    if sample_stats is not None:
        sizes = idata["posterior"].sizes
        size = (sizes["chain"], sizes["draw"])
        sample_stats_ds = xr.Dataset(
            {name: (("chain", "draw"), creator(size)) for name, creator in sample_stats.items()},
            coords=idata["posterior"].coords,
        )
        idata.add_groups(sample_stats=sample_stats_ds)

    return idata


def mock_sample_setup_and_teardown():
    """Set up and tear down mocking of PyMC sampling functions for testing.

    This function is designed to be used with pytest fixtures to temporarily replace
    PyMC's sampling functionality with faster alternatives for testing purposes.

    Effects during the fixture's active period:

    * Replaces :func:`pymc.sample` with :func:`pymc.testing.mock_sample`, which uses
      prior predictive sampling instead of MCMC
    * Replaces distributions:
        * :class:`pymc.Flat` with :class:`pymc.Normal`
        * :class:`pymc.HalfFlat` with :class:`pymc.HalfNormal`
    * Automatically restores all original functions and distributions after the test completes

    Examples
    --------
    Use with `pytest` to mock actual PyMC sampling in test suite.

    .. code-block:: python

        # tests/conftest.py
        import pytest
        import pymc as pm
        from pymc.testing import mock_sample_setup_and_teardown

        # Register as a pytest fixture
        mock_pymc_sample = pytest.fixture(scope="function")(mock_sample_setup_and_teardown)


        # tests/test_model.py
        # Use in a test function
        def test_model_inference(mock_pymc_sample):
            with pm.Model() as model:
                x = pm.Normal("x", 0, 1)
                # This will use mock_sample instead of actual MCMC
                idata = pm.sample()
                # Test with the inference data...

    """
    import pymc as pm

    original_flat = pm.Flat
    original_half_flat = pm.HalfFlat
    original_sample = pm.sample

    pm.sample = mock_sample
    pm.Flat = pm.Normal
    pm.HalfFlat = pm.HalfNormal

    yield

    pm.sample = original_sample
    pm.Flat = original_flat
    pm.HalfFlat = original_half_flat


def equal_computations_up_to_root(
    xs: Sequence[Variable], ys: Sequence[Variable], ignore_rng_values=True
) -> bool:
    # Check if graphs are equivalent even if root variables have distinct identities

    x_graph_inputs = [var for var in graph_inputs(xs) if not isinstance(var, Constant)]
    y_graph_inputs = [var for var in graph_inputs(ys) if not isinstance(var, Constant)]
    if len(x_graph_inputs) != len(y_graph_inputs):
        return False
    for x, y in zip(x_graph_inputs, y_graph_inputs):
        if x.type != y.type:
            return False
        if x.name != y.name:
            return False
        if isinstance(x, SharedVariable):
            if not isinstance(y, SharedVariable):
                return False
            if isinstance(x.type, RandomType) and ignore_rng_values:
                continue
            if not x.type.values_eq(x.get_value(), y.get_value()):
                return False

    return equal_computations(xs, ys, in_xs=x_graph_inputs, in_ys=y_graph_inputs)  # type: ignore[arg-type]
