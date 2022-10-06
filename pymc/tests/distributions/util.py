import functools as ft
import itertools as it

from contextlib import ExitStack as does_not_raise
from typing import Callable, List, Optional

import aesara
import aesara.tensor as at
import numpy as np
import numpy.random as nr
import numpy.testing as npt
import pytest
import scipy.special as sp
import scipy.stats as st

from aeppl.logprob import ParameterValueError
from aesara.compile.mode import Mode

import pymc as pm

from pymc.aesaraf import compile_pymc, floatX, intX
from pymc.distributions import logcdf, logp
from pymc.distributions.logprob import joint_logp
from pymc.distributions.shape_utils import change_dist_size
from pymc.initial_point import make_initial_point_fn
from pymc.tests.helpers import SeededTest, select_by_precision


def product(domains, n_samples=-1):
    """Get an iterator over a product of domains.

    Args:
        domains: a dictionary of (name, object) pairs, where the objects
                 must be "domain-like", as in, have a `.vals` property
        n_samples: int, maximum samples to return.  -1 to return whole product

    Returns:
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
    def __init__(self, vals, dtype=aesara.config.floatX, edges=None, shape=None):
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
        return Domain(
            [v + other for v in self.vals],
            self.dtype,
            (self.lower + other, self.upper + other),
            self.shape,
        )

    def __mul__(self, other):
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
        return Domain([-v for v in self.vals], self.dtype, (-self.lower, -self.upper), self.shape)


@pytest.mark.parametrize(
    "values, edges, expectation",
    [
        ([], None, pytest.raises(IndexError)),
        ([], (0, 0), pytest.raises(ValueError)),
        ([0], None, pytest.raises(ValueError)),
        ([0], (0, 0), does_not_raise()),
        ([-1, 1], None, pytest.raises(ValueError)),
        ([-1, 0, 1], None, does_not_raise()),
    ],
)
def test_domain(values, edges, expectation):
    with expectation:
        Domain(values, edges=edges)


class ProductDomain:
    def __init__(self, domains):
        self.vals = list(it.product(*(d.vals for d in domains)))
        self.shape = (len(domains),) + domains[0].shape
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

I = Domain([-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf], "int64")

NatSmall = Domain([0, 3, 4, 5, np.inf], "int64")
Nat = Domain([0, 1, 2, 3, np.inf], "int64")
NatBig = Domain([0, 1, 2, 3, 5000, np.inf], "int64")
PosNat = Domain([1, 2, 3, np.inf], "int64")

Bool = Domain([0, 0, 1, 1], "int64")


def build_model(distfam, valuedomain, vardomains, extra_args=None):
    if extra_args is None:
        extra_args = {}

    with pm.Model() as m:
        param_vars = {}
        for v, dom in vardomains.items():
            v_at = aesara.shared(np.asarray(dom.vals[0]))
            v_at.name = v
            param_vars[v] = v_at
        param_vars.update(extra_args)
        distfam(
            "value",
            **param_vars,
            transform=None,
        )
    return m, param_vars


def check_logp(
    pymc_dist,
    domain,
    paramdomains,
    scipy_logp,
    decimal=None,
    n_samples=100,
    extra_args=None,
    scipy_args=None,
    skip_paramdomain_outside_edge_test=False,
):
    """
    Generic test for PyMC logp methods

    Test PyMC logp and equivalent scipy logpmf/logpdf methods give similar
    results for valid values and parameters inside the supported edges.
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
    if decimal is None:
        decimal = select_by_precision(float64=6, float32=3)

    if extra_args is None:
        extra_args = {}

    if scipy_args is None:
        scipy_args = {}

    def logp_reference(args):
        args.update(scipy_args)
        return scipy_logp(**args)

    def _model_input_dict(model, param_vars, pt):
        """Create a dict with only the necessary, transformed logp inputs."""
        pt_d = {}
        for k, v in pt.items():
            rv_var = model.named_vars.get(k)
            nv = param_vars.get(k, rv_var)
            nv = getattr(nv.tag, "value_var", nv)

            transform = getattr(nv.tag, "transform", None)
            if transform:
                # todo: the compiled graph behind this should be cached and
                # reused (if it isn't already).
                v = transform.forward(rv_var, v).eval()

            if nv.name in param_vars:
                # update the shared parameter variables in `param_vars`
                param_vars[nv.name].set_value(v)
            else:
                # create an argument entry for the (potentially
                # transformed) "value" variable
                pt_d[nv.name] = v

        return pt_d

    model, param_vars = build_model(pymc_dist, domain, paramdomains, extra_args)
    logp_pymc = model.compile_logp(jacobian=False)

    # Test supported value and parameters domain matches scipy
    domains = paramdomains.copy()
    domains["value"] = domain
    for pt in product(domains, n_samples=n_samples):
        pt = dict(pt)
        pt_d = _model_input_dict(model, param_vars, pt)
        pt_logp = pm.Point(pt_d, model=model)
        pt_ref = pm.Point(pt, filter_model_vars=False, model=model)
        npt.assert_almost_equal(
            logp_pymc(pt_logp),
            logp_reference(pt_ref),
            decimal=decimal,
            err_msg=str(pt),
        )

    valid_value = domain.vals[0]
    valid_params = {param: paramdomain.vals[0] for param, paramdomain in paramdomains.items()}
    valid_dist = pymc_dist.dist(**valid_params, **extra_args)

    # Test pymc distribution raises ParameterValueError for scalar parameters outside
    # the supported domain edges (excluding edges)
    if not skip_paramdomain_outside_edge_test:
        # Step1: collect potential invalid parameters
        invalid_params = {param: [None, None] for param in paramdomains}
        for param, paramdomain in paramdomains.items():
            if np.ndim(paramdomain.lower) != 0:
                continue
            if np.isfinite(paramdomain.lower):
                invalid_params[param][0] = paramdomain.lower - 1
            if np.isfinite(paramdomain.upper):
                invalid_params[param][1] = paramdomain.upper + 1

        # Step2: test invalid parameters, one a time
        for invalid_param, invalid_edges in invalid_params.items():
            for invalid_edge in invalid_edges:
                if invalid_edge is None:
                    continue
                test_params = valid_params.copy()  # Shallow copy should be okay
                test_params[invalid_param] = at.as_tensor_variable(invalid_edge)
                # We need to remove `Assert`s introduced by checks like
                # `assert_negative_support` and disable test values;
                # otherwise, we won't be able to create the `RandomVariable`
                with aesara.config.change_flags(compute_test_value="off"):
                    invalid_dist = pymc_dist.dist(**test_params, **extra_args)
                with aesara.config.change_flags(mode=Mode("py")):
                    with pytest.raises(ParameterValueError):
                        logp(invalid_dist, valid_value).eval()
                        pytest.fail(f"test_params={test_params}, valid_value={valid_value}")

    # Test that values outside of scalar domain support evaluate to -np.inf
    if np.ndim(domain.lower) != 0:
        return
    invalid_values = [None, None]
    if np.isfinite(domain.lower):
        invalid_values[0] = domain.lower - 1
    if np.isfinite(domain.upper):
        invalid_values[1] = domain.upper + 1

    for invalid_value in invalid_values:
        if invalid_value is None:
            continue
        with aesara.config.change_flags(mode=Mode("py")):
            npt.assert_equal(
                logp(valid_dist, invalid_value).eval(),
                -np.inf,
                err_msg=str(invalid_value),
            )


def check_logcdf(
    pymc_dist,
    domain,
    paramdomains,
    scipy_logcdf,
    decimal=None,
    n_samples=100,
    skip_paramdomain_inside_edge_test=False,
    skip_paramdomain_outside_edge_test=False,
):
    """
    Generic test for PyMC logcdf methods

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

    Returns
    -------

    """
    # Test pymc and scipy distributions match for values and parameters
    # within the supported domain edges (excluding edges)
    if not skip_paramdomain_inside_edge_test:
        domains = paramdomains.copy()
        domains["value"] = domain

        model, param_vars = build_model(pymc_dist, domain, paramdomains)
        rv = model["value"]
        value = model.rvs_to_values[rv]
        pymc_logcdf = model.compile_fn(logcdf(rv, value))

        if decimal is None:
            decimal = select_by_precision(float64=6, float32=3)

        for pt in product(domains, n_samples=n_samples):
            params = dict(pt)
            scipy_eval = scipy_logcdf(**params)

            value = params.pop("value")
            # Update shared parameter variables in pymc_logcdf function
            for param_name, param_value in params.items():
                param_vars[param_name].set_value(param_value)
            pymc_eval = pymc_logcdf({"value": value})

            params["value"] = value  # for displaying in err_msg
            npt.assert_almost_equal(
                pymc_eval,
                scipy_eval,
                decimal=decimal,
                err_msg=str(params),
            )

    valid_value = domain.vals[0]
    valid_params = {param: paramdomain.vals[0] for param, paramdomain in paramdomains.items()}
    valid_dist = pymc_dist.dist(**valid_params)

    # Test pymc distribution raises ParameterValueError for parameters outside the
    # supported domain edges (excluding edges)
    if not skip_paramdomain_outside_edge_test:
        # Step1: collect potential invalid parameters
        invalid_params = {param: [None, None] for param in paramdomains}
        for param, paramdomain in paramdomains.items():
            if np.isfinite(paramdomain.lower):
                invalid_params[param][0] = paramdomain.lower - 1
            if np.isfinite(paramdomain.upper):
                invalid_params[param][1] = paramdomain.upper + 1
        # Step2: test invalid parameters, one a time
        for invalid_param, invalid_edges in invalid_params.items():
            for invalid_edge in invalid_edges:
                if invalid_edge is not None:
                    test_params = valid_params.copy()  # Shallow copy should be okay
                    test_params[invalid_param] = at.as_tensor_variable(invalid_edge)
                    # We need to remove `Assert`s introduced by checks like
                    # `assert_negative_support` and disable test values;
                    # otherwise, we won't be able to create the
                    # `RandomVariable`
                    with aesara.config.change_flags(compute_test_value="off"):
                        invalid_dist = pymc_dist.dist(**test_params)
                    with aesara.config.change_flags(mode=Mode("py")):
                        with pytest.raises(ParameterValueError):
                            logcdf(invalid_dist, valid_value).eval()

    # Test that values below domain edge evaluate to -np.inf
    if np.isfinite(domain.lower):
        below_domain = domain.lower - 1
        with aesara.config.change_flags(mode=Mode("py")):
            npt.assert_equal(
                logcdf(valid_dist, below_domain).eval(),
                -np.inf,
                err_msg=str(below_domain),
            )

    # Test that values above domain edge evaluate to 0
    if np.isfinite(domain.upper):
        above_domain = domain.upper + 1
        with aesara.config.change_flags(mode=Mode("py")):
            npt.assert_equal(
                logcdf(valid_dist, above_domain).eval(),
                0,
                err_msg=str(above_domain),
            )

    # Test that method works with multiple values or raises informative TypeError
    valid_dist = pymc_dist.dist(**valid_params, size=2)
    with aesara.config.change_flags(mode=Mode("py")):
        try:
            logcdf(valid_dist, np.array([valid_value, valid_value])).eval()
        except TypeError as err:
            assert str(err).endswith(
                "logcdf expects a scalar value but received a 1-dimensional object."
            )


def check_selfconsistency_discrete_logcdf(
    distribution,
    domain,
    paramdomains,
    decimal=None,
    n_samples=100,
):
    """
    Check that logcdf of discrete distributions matches sum of logps up to value
    """
    domains = paramdomains.copy()
    domains["value"] = domain
    if decimal is None:
        decimal = select_by_precision(float64=6, float32=3)

    model, param_vars = build_model(distribution, domain, paramdomains)
    rv = model["value"]
    value = model.rvs_to_values[rv]
    dist_logcdf = model.compile_fn(logcdf(rv, value))
    dist_logp = model.compile_fn(logp(rv, value))

    for pt in product(domains, n_samples=n_samples):
        params = dict(pt)
        value = params.pop("value")
        values = np.arange(domain.lower, value + 1)

        # Update shared parameter variables in logp/logcdf function
        for param_name, param_value in params.items():
            param_vars[param_name].set_value(param_value)

        with aesara.config.change_flags(mode=Mode("py")):
            npt.assert_almost_equal(
                dist_logcdf({"value": value}),
                sp.logsumexp([dist_logp({"value": value}) for value in values]),
                decimal=decimal,
                err_msg=str(pt),
            )


def assert_moment_is_expected(model, expected, check_finite_logp=True):
    fn = make_initial_point_fn(
        model=model,
        return_transformed=False,
        default_strategy="moment",
    )
    moment = fn(0)["x"]
    expected = np.asarray(expected)
    try:
        random_draw = model["x"].eval()
    except NotImplementedError:
        random_draw = moment

    assert moment.shape == expected.shape
    assert expected.shape == random_draw.shape
    assert np.allclose(moment, expected)

    if check_finite_logp:
        logp_moment = joint_logp(model["x"], at.constant(moment), transformed=False).eval()
        assert np.isfinite(logp_moment)


def pymc_random(
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
    pymc_rand = compile_pymc([], model_dist)

    domains = paramdomains.copy()
    for pt in product(domains, n_samples=100):
        pt = pm.Point(pt, model=model)
        pt.update(model_args)

        # Update the shared parameter variables in `param_vars`
        for k, v in pt.items():
            nv = param_vars.get(k, model.named_vars.get(k))
            if nv.name in param_vars:
                param_vars[nv.name].set_value(v)

        p = alpha
        # Allow KS test to fail (i.e., the samples be different)
        # a certain number of times. Crude, but necessary.
        f = fails
        while p <= alpha and f > 0:
            s0 = pymc_rand()
            s1 = floatX(ref_rand(size=size, **pt))
            _, p = st.ks_2samp(np.atleast_1d(s0).flatten(), np.atleast_1d(s1).flatten())
            f -= 1
        assert p > alpha, str(pt)


def pymc_random_discrete(
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
    pymc_rand = compile_pymc([], model_dist)

    domains = paramdomains.copy()
    for pt in product(domains, n_samples=100):
        pt = pm.Point(pt, model=model)
        p = alpha

        # Update the shared parameter variables in `param_vars`
        for k, v in pt.items():
            nv = param_vars.get(k, model.named_vars.get(k))
            if nv.name in param_vars:
                param_vars[nv.name].set_value(v)

        # Allow Chisq test to fail (i.e., the samples be different)
        # a certain number of times.
        f = fails
        while p <= alpha and f > 0:
            o = pymc_rand()
            e = intX(ref_rand(size=size, **pt))
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
        assert p > alpha, str(pt)


class BaseTestDistributionRandom(SeededTest):
    """
    This class provides a base for tests that new RandomVariables are correctly
    implemented, and that the mapping of parameters between the PyMC
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

    pymc_dist: Optional[Callable] = None
    pymc_dist_params: Optional[dict] = None
    reference_dist: Optional[Callable] = None
    reference_dist_params: Optional[dict] = None
    expected_rv_op_params: Optional[dict] = None
    checks_to_run = []
    size = 15
    decimal = select_by_precision(float64=6, float32=3)

    sizes_to_check: Optional[List] = None
    sizes_expected: Optional[List] = None
    repeated_params_shape = 5

    def test_distribution(self):
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

    def _instantiate_pymc_rv(self, dist_params=None):
        params = dist_params if dist_params else self.pymc_dist_params
        self.pymc_rv = self.pymc_dist.dist(
            **params, size=self.size, rng=aesara.shared(self.get_random_state(reset=True))
        )

    def check_pymc_draws_match_reference(self):
        # need to re-instantiate it to make sure that the order of drawings match the reference distribution one
        # self._instantiate_pymc_rv()
        npt.assert_array_almost_equal(
            self.pymc_rv.eval(), self.reference_dist_draws, decimal=self.decimal
        )

    def check_pymc_params_match_rv_op(self):
        aesara_dist_inputs = self.pymc_rv.get_parents()[0].inputs[3:]
        assert len(self.expected_rv_op_params) == len(aesara_dist_inputs)
        for (expected_name, expected_value), actual_variable in zip(
            self.expected_rv_op_params.items(), aesara_dist_inputs
        ):

            # Add additional line to evaluate symbolic inputs to distributions
            if isinstance(expected_value, aesara.tensor.Variable):
                expected_value = expected_value.eval()

            npt.assert_almost_equal(expected_value, actual_variable.eval(), decimal=self.decimal)

    def check_rv_size(self):
        # test sizes
        sizes_to_check = self.sizes_to_check or [None, (), 1, (1,), 5, (4, 5), (2, 4, 2)]
        sizes_expected = self.sizes_expected or [(), (), (1,), (1,), (5,), (4, 5), (2, 4, 2)]
        for size, expected in zip(sizes_to_check, sizes_expected):
            pymc_rv = self.pymc_dist.dist(**self.pymc_dist_params, size=size)
            expected_symbolic = tuple(pymc_rv.shape.eval())
            actual = pymc_rv.eval().shape
            assert actual == expected_symbolic
            assert expected_symbolic == expected

        # test multi-parameters sampling for univariate distributions (with univariate inputs)
        if (
            self.pymc_dist.rv_op.ndim_supp == 0
            and self.pymc_dist.rv_op.ndims_params
            and sum(self.pymc_dist.rv_op.ndims_params) == 0
        ):
            params = {
                k: p * np.ones(self.repeated_params_shape) for k, p in self.pymc_dist_params.items()
            }
            self._instantiate_pymc_rv(params)
            sizes_to_check = [None, self.repeated_params_shape, (5, self.repeated_params_shape)]
            sizes_expected = [
                (self.repeated_params_shape,),
                (self.repeated_params_shape,),
                (5, self.repeated_params_shape),
            ]
            for size, expected in zip(sizes_to_check, sizes_expected):
                pymc_rv = self.pymc_dist.dist(**params, size=size)
                expected_symbolic = tuple(pymc_rv.shape.eval())
                actual = pymc_rv.eval().shape
                assert actual == expected_symbolic == expected

    def validate_tests_list(self):
        assert len(self.checks_to_run) == len(
            set(self.checks_to_run)
        ), "There are duplicates in the list of checks_to_run"


def seeded_scipy_distribution_builder(dist_name: str) -> Callable:
    return lambda self: ft.partial(getattr(st, dist_name).rvs, random_state=self.get_random_state())


def seeded_numpy_distribution_builder(dist_name: str) -> Callable:
    return lambda self: ft.partial(
        getattr(np.random.RandomState, dist_name), self.get_random_state()
    )
