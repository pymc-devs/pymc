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
import functools
import itertools
import re
import sys

from typing import Callable, List, Optional

import aesara
import numpy as np
import numpy.testing as npt
import pytest
import scipy.stats as st

from numpy.testing import assert_almost_equal, assert_array_almost_equal

try:
    from polyagamma import random_polyagamma

    _polyagamma_not_installed = False
except ImportError:  # pragma: no cover

    _polyagamma_not_installed = True

    def random_polyagamma(*args, **kwargs):
        raise RuntimeError("polyagamma package is not installed!")


from scipy.special import expit, softmax

import pymc as pm

from pymc.aesaraf import change_rv_size, floatX, intX
from pymc.distributions.continuous import get_tau_sigma, interpolated
from pymc.distributions.discrete import _OrderedLogistic, _OrderedProbit
from pymc.distributions.dist_math import clipped_beta_rvs
from pymc.distributions.logprob import logp
from pymc.distributions.multivariate import (
    _LKJCholeskyCov,
    _OrderedMultinomial,
    quaddist_matrix,
)
from pymc.distributions.shape_utils import to_tuple
from pymc.tests.helpers import SeededTest, select_by_precision
from pymc.tests.test_distributions import (
    Domain,
    R,
    RandomPdMatrix,
    Rplus,
    build_model,
    product,
)


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
    change_rv_size_fn=change_rv_size,
):
    if valuedomain is None:
        valuedomain = Domain([0], edges=(None, None))

    if model_args is None:
        model_args = {}

    model, param_vars = build_model(dist, valuedomain, paramdomains, extra_args)
    model_dist = change_rv_size_fn(model.named_vars["value"], size, expand=True)
    pymc_rand = aesara.function([], model_dist)

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
    model_dist = change_rv_size(model.named_vars["value"], size, expand=True)
    pymc_rand = aesara.function([], model_dist)

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
            observed = dict(zip(*np.unique(o, return_counts=True)))
            expected = dict(zip(*np.unique(e, return_counts=True)))
            for e in expected.keys():
                expected[e] = (observed.get(e, 0), expected[e])
            k = np.array([v for v in expected.values()])
            if np.all(k[:, 0] == k[:, 1]):
                p = 1.0
            else:
                _, p = st.chisquare(k[:, 0], k[:, 1])
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

    Each desired test must be referenced by name in `tests_to_run`, when
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
        tests_to_run = ["check_rv_size"]
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
            getattr(self, check_name)()

    def _instantiate_pymc_rv(self, dist_params=None):
        params = dist_params if dist_params else self.pymc_dist_params
        self.pymc_rv = self.pymc_dist.dist(
            **params, size=self.size, rng=aesara.shared(self.get_random_state(reset=True))
        )

    def check_pymc_draws_match_reference(self):
        # need to re-instantiate it to make sure that the order of drawings match the reference distribution one
        # self._instantiate_pymc_rv()
        assert_array_almost_equal(
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

            assert_almost_equal(expected_value, actual_variable.eval(), decimal=self.decimal)

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
        ), "There are duplicates in the list of tests_to_run"


def seeded_scipy_distribution_builder(dist_name: str) -> Callable:
    return lambda self: functools.partial(
        getattr(st, dist_name).rvs, random_state=self.get_random_state()
    )


def seeded_numpy_distribution_builder(dist_name: str) -> Callable:
    return lambda self: functools.partial(
        getattr(np.random.RandomState, dist_name), self.get_random_state()
    )


class TestFlat(BaseTestDistributionRandom):
    pymc_dist = pm.Flat
    pymc_dist_params = {}
    expected_rv_op_params = {}
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_inferred_size",
        "check_not_implemented",
    ]

    def check_rv_inferred_size(self):
        sizes_to_check = [None, (), 1, (1,), 5, (4, 5), (2, 4, 2)]
        sizes_expected = [(), (), (1,), (1,), (5,), (4, 5), (2, 4, 2)]
        for size, expected in zip(sizes_to_check, sizes_expected):
            pymc_rv = self.pymc_dist.dist(**self.pymc_dist_params, size=size)
            expected_symbolic = tuple(pymc_rv.shape.eval())
            assert expected_symbolic == expected

    def check_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.pymc_rv.eval()


class TestHalfFlat(BaseTestDistributionRandom):
    pymc_dist = pm.HalfFlat
    pymc_dist_params = {}
    expected_rv_op_params = {}
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_inferred_size",
        "check_not_implemented",
    ]

    def check_rv_inferred_size(self):
        sizes_to_check = [None, (), 1, (1,), 5, (4, 5), (2, 4, 2)]
        sizes_expected = [(), (), (1,), (1,), (5,), (4, 5), (2, 4, 2)]
        for size, expected in zip(sizes_to_check, sizes_expected):
            pymc_rv = self.pymc_dist.dist(**self.pymc_dist_params, size=size)
            expected_symbolic = tuple(pymc_rv.shape.eval())
            assert expected_symbolic == expected

    def check_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.pymc_rv.eval()


class TestDiscreteWeibull(BaseTestDistributionRandom):
    def discrete_weibul_rng_fn(self, size, q, beta, uniform_rng_fct):
        return np.ceil(np.power(np.log(1 - uniform_rng_fct(size=size)) / np.log(q), 1.0 / beta)) - 1

    def seeded_discrete_weibul_rng_fn(self):
        uniform_rng_fct = functools.partial(
            getattr(np.random.RandomState, "uniform"), self.get_random_state()
        )
        return functools.partial(self.discrete_weibul_rng_fn, uniform_rng_fct=uniform_rng_fct)

    pymc_dist = pm.DiscreteWeibull
    pymc_dist_params = {"q": 0.25, "beta": 2.0}
    expected_rv_op_params = {"q": 0.25, "beta": 2.0}
    reference_dist_params = {"q": 0.25, "beta": 2.0}
    reference_dist = seeded_discrete_weibul_rng_fn
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestPareto(BaseTestDistributionRandom):
    pymc_dist = pm.Pareto
    pymc_dist_params = {"alpha": 3.0, "m": 2.0}
    expected_rv_op_params = {"alpha": 3.0, "m": 2.0}
    reference_dist_params = {"b": 3.0, "scale": 2.0}
    reference_dist = seeded_scipy_distribution_builder("pareto")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestLaplace(BaseTestDistributionRandom):
    pymc_dist = pm.Laplace
    pymc_dist_params = {"mu": 0.0, "b": 1.0}
    expected_rv_op_params = {"mu": 0.0, "b": 1.0}
    reference_dist_params = {"loc": 0.0, "scale": 1.0}
    reference_dist = seeded_scipy_distribution_builder("laplace")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestAsymmetricLaplace(BaseTestDistributionRandom):
    def asymmetriclaplace_rng_fn(self, b, kappa, mu, size, uniform_rng_fct):
        u = uniform_rng_fct(size=size)
        switch = kappa**2 / (1 + kappa**2)
        non_positive_x = mu + kappa * np.log(u * (1 / switch)) / b
        positive_x = mu - np.log((1 - u) * (1 + kappa**2)) / (kappa * b)
        draws = non_positive_x * (u <= switch) + positive_x * (u > switch)
        return draws

    def seeded_asymmetriclaplace_rng_fn(self):
        uniform_rng_fct = functools.partial(
            getattr(np.random.RandomState, "uniform"), self.get_random_state()
        )
        return functools.partial(self.asymmetriclaplace_rng_fn, uniform_rng_fct=uniform_rng_fct)

    pymc_dist = pm.AsymmetricLaplace

    pymc_dist_params = {"b": 1.0, "kappa": 1.0, "mu": 0.0}
    expected_rv_op_params = {"b": 1.0, "kappa": 1.0, "mu": 0.0}
    reference_dist_params = {"b": 1.0, "kappa": 1.0, "mu": 0.0}
    reference_dist = seeded_asymmetriclaplace_rng_fn
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestExGaussian(BaseTestDistributionRandom):
    def exgaussian_rng_fn(self, mu, sigma, nu, size, normal_rng_fct, exponential_rng_fct):
        return normal_rng_fct(mu, sigma, size=size) + exponential_rng_fct(scale=nu, size=size)

    def seeded_exgaussian_rng_fn(self):
        normal_rng_fct = functools.partial(
            getattr(np.random.RandomState, "normal"), self.get_random_state()
        )
        exponential_rng_fct = functools.partial(
            getattr(np.random.RandomState, "exponential"), self.get_random_state()
        )
        return functools.partial(
            self.exgaussian_rng_fn,
            normal_rng_fct=normal_rng_fct,
            exponential_rng_fct=exponential_rng_fct,
        )

    pymc_dist = pm.ExGaussian

    pymc_dist_params = {"mu": 1.0, "sigma": 1.0, "nu": 1.0}
    expected_rv_op_params = {"mu": 1.0, "sigma": 1.0, "nu": 1.0}
    reference_dist_params = {"mu": 1.0, "sigma": 1.0, "nu": 1.0}
    reference_dist = seeded_exgaussian_rng_fn
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestGumbel(BaseTestDistributionRandom):
    pymc_dist = pm.Gumbel
    pymc_dist_params = {"mu": 1.5, "beta": 3.0}
    expected_rv_op_params = {"mu": 1.5, "beta": 3.0}
    reference_dist_params = {"loc": 1.5, "scale": 3.0}
    reference_dist = seeded_scipy_distribution_builder("gumbel_r")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
    ]


class TestStudentT(BaseTestDistributionRandom):
    pymc_dist = pm.StudentT
    pymc_dist_params = {"nu": 5.0, "mu": -1.0, "sigma": 2.0}
    expected_rv_op_params = {"nu": 5.0, "mu": -1.0, "sigma": 2.0}
    reference_dist_params = {"df": 5.0, "loc": -1.0, "scale": 2.0}
    reference_dist = seeded_scipy_distribution_builder("t")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestMoyal(BaseTestDistributionRandom):
    pymc_dist = pm.Moyal
    pymc_dist_params = {"mu": 0.0, "sigma": 1.0}
    expected_rv_op_params = {"mu": 0.0, "sigma": 1.0}
    reference_dist_params = {"loc": 0.0, "scale": 1.0}
    reference_dist = seeded_scipy_distribution_builder("moyal")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestKumaraswamy(BaseTestDistributionRandom):
    def kumaraswamy_rng_fn(self, a, b, size, uniform_rng_fct):
        return (1 - (1 - uniform_rng_fct(size=size)) ** (1 / b)) ** (1 / a)

    def seeded_kumaraswamy_rng_fn(self):
        uniform_rng_fct = functools.partial(
            getattr(np.random.RandomState, "uniform"), self.get_random_state()
        )
        return functools.partial(self.kumaraswamy_rng_fn, uniform_rng_fct=uniform_rng_fct)

    pymc_dist = pm.Kumaraswamy
    pymc_dist_params = {"a": 1.0, "b": 1.0}
    expected_rv_op_params = {"a": 1.0, "b": 1.0}
    reference_dist_params = {"a": 1.0, "b": 1.0}
    reference_dist = seeded_kumaraswamy_rng_fn
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestTruncatedNormal(BaseTestDistributionRandom):
    pymc_dist = pm.TruncatedNormal
    lower, upper, mu, sigma = -2.0, 2.0, 0, 1.0
    pymc_dist_params = {"mu": mu, "sigma": sigma, "lower": lower, "upper": upper}
    expected_rv_op_params = {"mu": mu, "sigma": sigma, "lower": lower, "upper": upper}
    reference_dist_params = {
        "loc": mu,
        "scale": sigma,
        "a": (lower - mu) / sigma,
        "b": (upper - mu) / sigma,
    }
    reference_dist = seeded_scipy_distribution_builder("truncnorm")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestTruncatedNormalTau(BaseTestDistributionRandom):
    pymc_dist = pm.TruncatedNormal
    lower, upper, mu, tau = -2.0, 2.0, 0, 1.0
    tau, sigma = get_tau_sigma(tau=tau, sigma=None)
    pymc_dist_params = {"mu": mu, "tau": tau, "lower": lower, "upper": upper}
    expected_rv_op_params = {"mu": mu, "sigma": sigma, "lower": lower, "upper": upper}
    checks_to_run = [
        "check_pymc_params_match_rv_op",
    ]


class TestTruncatedNormalLowerTau(BaseTestDistributionRandom):
    pymc_dist = pm.TruncatedNormal
    lower, upper, mu, tau = -2.0, np.inf, 0, 1.0
    tau, sigma = get_tau_sigma(tau=tau, sigma=None)
    pymc_dist_params = {"mu": mu, "tau": tau, "lower": lower}
    expected_rv_op_params = {"mu": mu, "sigma": sigma, "lower": lower, "upper": upper}
    checks_to_run = [
        "check_pymc_params_match_rv_op",
    ]


class TestTruncatedNormalUpperTau(BaseTestDistributionRandom):
    pymc_dist = pm.TruncatedNormal
    lower, upper, mu, tau = -np.inf, 2.0, 0, 1.0
    tau, sigma = get_tau_sigma(tau=tau, sigma=None)
    pymc_dist_params = {"mu": mu, "tau": tau, "upper": upper}
    expected_rv_op_params = {"mu": mu, "sigma": sigma, "lower": lower, "upper": upper}
    checks_to_run = [
        "check_pymc_params_match_rv_op",
    ]


class TestTruncatedNormalUpperArray(BaseTestDistributionRandom):
    pymc_dist = pm.TruncatedNormal
    lower, upper, mu, tau = (
        np.array([-np.inf, -np.inf]),
        np.array([3, 2]),
        np.array([0, 0]),
        np.array(
            [
                1,
                1,
            ]
        ),
    )
    size = (15, 2)
    tau, sigma = get_tau_sigma(tau=tau, sigma=None)
    pymc_dist_params = {"mu": mu, "tau": tau, "upper": upper}
    expected_rv_op_params = {"mu": mu, "sigma": sigma, "lower": lower, "upper": upper}
    checks_to_run = [
        "check_pymc_params_match_rv_op",
    ]


class TestWald(BaseTestDistributionRandom):
    pymc_dist = pm.Wald
    mu, lam, alpha = 1.0, 1.0, 0.0
    mu_rv, lam_rv, phi_rv = pm.Wald.get_mu_lam_phi(mu=mu, lam=lam, phi=None)
    pymc_dist_params = {"mu": mu, "lam": lam, "alpha": alpha}
    expected_rv_op_params = {"mu": mu_rv, "lam": lam_rv, "alpha": alpha}
    reference_dist_params = {"mean": mu, "scale": lam_rv}
    reference_dist = seeded_numpy_distribution_builder("wald")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]

    def check_pymc_draws_match_reference(self):
        assert_array_almost_equal(
            self.pymc_rv.eval(), self.reference_dist_draws + self.alpha, decimal=self.decimal
        )


class TestWaldMuPhi(BaseTestDistributionRandom):
    pymc_dist = pm.Wald
    mu, phi, alpha = 1.0, 3.0, 0.0
    mu_rv, lam_rv, phi_rv = pm.Wald.get_mu_lam_phi(mu=mu, lam=None, phi=phi)
    pymc_dist_params = {"mu": mu, "phi": phi, "alpha": alpha}
    expected_rv_op_params = {"mu": mu_rv, "lam": lam_rv, "alpha": alpha}
    checks_to_run = [
        "check_pymc_params_match_rv_op",
    ]


class TestSkewNormal(BaseTestDistributionRandom):
    pymc_dist = pm.SkewNormal
    pymc_dist_params = {"mu": 0.0, "sigma": 1.0, "alpha": 5.0}
    expected_rv_op_params = {"mu": 0.0, "sigma": 1.0, "alpha": 5.0}
    reference_dist_params = {"loc": 0.0, "scale": 1.0, "a": 5.0}
    reference_dist = seeded_scipy_distribution_builder("skewnorm")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestSkewNormalTau(BaseTestDistributionRandom):
    pymc_dist = pm.SkewNormal
    tau, sigma = get_tau_sigma(tau=2.0)
    pymc_dist_params = {"mu": 0.0, "tau": tau, "alpha": 5.0}
    expected_rv_op_params = {"mu": 0.0, "sigma": sigma, "alpha": 5.0}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestRice(BaseTestDistributionRandom):
    pymc_dist = pm.Rice
    b, sigma = 1, 2
    pymc_dist_params = {"b": b, "sigma": sigma}
    expected_rv_op_params = {"b": b, "sigma": sigma}
    reference_dist_params = {"b": b, "scale": sigma}
    reference_dist = seeded_scipy_distribution_builder("rice")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestRiceNu(BaseTestDistributionRandom):
    pymc_dist = pm.Rice
    nu = sigma = 2
    pymc_dist_params = {"nu": nu, "sigma": sigma}
    expected_rv_op_params = {"b": nu / sigma, "sigma": sigma}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestStudentTLam(BaseTestDistributionRandom):
    pymc_dist = pm.StudentT
    lam, sigma = get_tau_sigma(tau=2.0)
    pymc_dist_params = {"nu": 5.0, "mu": -1.0, "lam": lam}
    expected_rv_op_params = {"nu": 5.0, "mu": -1.0, "lam": sigma}
    reference_dist_params = {"df": 5.0, "loc": -1.0, "scale": sigma}
    reference_dist = seeded_scipy_distribution_builder("t")
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestNormal(BaseTestDistributionRandom):
    pymc_dist = pm.Normal
    pymc_dist_params = {"mu": 5.0, "sigma": 10.0}
    expected_rv_op_params = {"mu": 5.0, "sigma": 10.0}
    reference_dist_params = {"loc": 5.0, "scale": 10.0}
    size = 15
    reference_dist = seeded_numpy_distribution_builder("normal")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestLogitNormal(BaseTestDistributionRandom):
    def logit_normal_rng_fn(self, rng, size, loc, scale):
        return expit(st.norm.rvs(loc=loc, scale=scale, size=size, random_state=rng))

    pymc_dist = pm.LogitNormal
    pymc_dist_params = {"mu": 5.0, "sigma": 10.0}
    expected_rv_op_params = {"mu": 5.0, "sigma": 10.0}
    reference_dist_params = {"loc": 5.0, "scale": 10.0}
    reference_dist = lambda self: functools.partial(
        self.logit_normal_rng_fn, rng=self.get_random_state()
    )
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestLogitNormalTau(BaseTestDistributionRandom):
    pymc_dist = pm.LogitNormal
    tau, sigma = get_tau_sigma(tau=25.0)
    pymc_dist_params = {"mu": 1.0, "tau": tau}
    expected_rv_op_params = {"mu": 1.0, "sigma": sigma}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestNormalTau(BaseTestDistributionRandom):
    pymc_dist = pm.Normal
    tau, sigma = get_tau_sigma(tau=25.0)
    pymc_dist_params = {"mu": 1.0, "tau": tau}
    expected_rv_op_params = {"mu": 1.0, "sigma": sigma}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestNormalSd(BaseTestDistributionRandom):
    pymc_dist = pm.Normal
    pymc_dist_params = {"mu": 1.0, "sigma": 5.0}
    expected_rv_op_params = {"mu": 1.0, "sigma": 5.0}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestUniform(BaseTestDistributionRandom):
    pymc_dist = pm.Uniform
    pymc_dist_params = {"lower": 0.5, "upper": 1.5}
    expected_rv_op_params = {"lower": 0.5, "upper": 1.5}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestHalfNormal(BaseTestDistributionRandom):
    pymc_dist = pm.HalfNormal
    pymc_dist_params = {"sigma": 10.0}
    expected_rv_op_params = {"mean": 0, "sigma": 10.0}
    reference_dist_params = {"loc": 0, "scale": 10.0}
    reference_dist = seeded_scipy_distribution_builder("halfnorm")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
    ]


class TestHalfNormalTau(BaseTestDistributionRandom):
    pymc_dist = pm.Normal
    tau, sigma = get_tau_sigma(tau=25.0)
    pymc_dist_params = {"tau": tau}
    expected_rv_op_params = {"mu": 0.0, "sigma": sigma}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestHalfNormalSd(BaseTestDistributionRandom):
    pymc_dist = pm.Normal
    pymc_dist_params = {"sigma": 5.0}
    expected_rv_op_params = {"mu": 0.0, "sigma": 5.0}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestBeta(BaseTestDistributionRandom):
    pymc_dist = pm.Beta
    pymc_dist_params = {"alpha": 2.0, "beta": 5.0}
    expected_rv_op_params = {"alpha": 2.0, "beta": 5.0}
    reference_dist_params = {"a": 2.0, "b": 5.0}
    size = 15
    reference_dist = lambda self: functools.partial(
        clipped_beta_rvs, random_state=self.get_random_state()
    )
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestBetaMuSigma(BaseTestDistributionRandom):
    pymc_dist = pm.Beta
    pymc_dist_params = {"mu": 0.5, "sigma": 0.25}
    expected_alpha, expected_beta = pm.Beta.get_alpha_beta(
        mu=pymc_dist_params["mu"], sigma=pymc_dist_params["sigma"]
    )
    expected_rv_op_params = {"alpha": expected_alpha, "beta": expected_beta}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestExponential(BaseTestDistributionRandom):
    pymc_dist = pm.Exponential
    pymc_dist_params = {"lam": 10.0}
    expected_rv_op_params = {"mu": 1.0 / pymc_dist_params["lam"]}
    reference_dist_params = {"scale": 1.0 / pymc_dist_params["lam"]}
    reference_dist = seeded_numpy_distribution_builder("exponential")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
    ]


class TestCauchy(BaseTestDistributionRandom):
    pymc_dist = pm.Cauchy
    pymc_dist_params = {"alpha": 2.0, "beta": 5.0}
    expected_rv_op_params = {"alpha": 2.0, "beta": 5.0}
    reference_dist_params = {"loc": 2.0, "scale": 5.0}
    reference_dist = seeded_scipy_distribution_builder("cauchy")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
    ]


class TestHalfCauchy(BaseTestDistributionRandom):
    pymc_dist = pm.HalfCauchy
    pymc_dist_params = {"beta": 5.0}
    expected_rv_op_params = {"alpha": 0.0, "beta": 5.0}
    reference_dist_params = {"loc": 0.0, "scale": 5.0}
    reference_dist = seeded_scipy_distribution_builder("halfcauchy")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
    ]


class TestGamma(BaseTestDistributionRandom):
    pymc_dist = pm.Gamma
    pymc_dist_params = {"alpha": 2.0, "beta": 5.0}
    expected_rv_op_params = {"alpha": 2.0, "beta": 1 / 5.0}
    reference_dist_params = {"shape": 2.0, "scale": 1 / 5.0}
    reference_dist = seeded_numpy_distribution_builder("gamma")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
    ]


class TestGammaMuSigma(BaseTestDistributionRandom):
    pymc_dist = pm.Gamma
    pymc_dist_params = {"mu": 0.5, "sigma": 0.25}
    expected_alpha, expected_beta = pm.Gamma.get_alpha_beta(
        mu=pymc_dist_params["mu"], sigma=pymc_dist_params["sigma"]
    )
    expected_rv_op_params = {"alpha": expected_alpha, "beta": 1 / expected_beta}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestInverseGamma(BaseTestDistributionRandom):
    pymc_dist = pm.InverseGamma
    pymc_dist_params = {"alpha": 2.0, "beta": 5.0}
    expected_rv_op_params = {"alpha": 2.0, "beta": 5.0}
    reference_dist_params = {"a": 2.0, "scale": 5.0}
    reference_dist = seeded_scipy_distribution_builder("invgamma")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
    ]


class TestInverseGammaMuSigma(BaseTestDistributionRandom):
    pymc_dist = pm.InverseGamma
    pymc_dist_params = {"mu": 0.5, "sigma": 0.25}
    expected_alpha, expected_beta = pm.InverseGamma._get_alpha_beta(
        alpha=None,
        beta=None,
        mu=pymc_dist_params["mu"],
        sigma=pymc_dist_params["sigma"],
    )
    expected_rv_op_params = {"alpha": expected_alpha, "beta": expected_beta}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestChiSquared(BaseTestDistributionRandom):
    pymc_dist = pm.ChiSquared
    pymc_dist_params = {"nu": 2.0}
    expected_rv_op_params = {"nu": 2.0}
    reference_dist_params = {"df": 2.0}
    reference_dist = seeded_numpy_distribution_builder("chisquare")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestBinomial(BaseTestDistributionRandom):
    pymc_dist = pm.Binomial
    pymc_dist_params = {"n": 100, "p": 0.33}
    expected_rv_op_params = {"n": 100, "p": 0.33}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestLogitBinomial(BaseTestDistributionRandom):
    pymc_dist = pm.Binomial
    pymc_dist_params = {"n": 100, "logit_p": 0.5}
    expected_rv_op_params = {"n": 100, "p": expit(0.5)}
    tests_to_run = ["check_pymc_params_match_rv_op"]

    @pytest.mark.parametrize(
        "n, p, logit_p, expected",
        [
            (5, None, None, "Must specify either p or logit_p."),
            (5, 0.5, 0.5, "Can't specify both p and logit_p."),
        ],
    )
    def test_binomial_init_fail(self, n, p, logit_p, expected):
        with pm.Model() as model:
            with pytest.raises(ValueError, match=f"Incompatible parametrization. {expected}"):
                pm.Binomial("x", n=n, p=p, logit_p=logit_p)


class TestNegativeBinomial(BaseTestDistributionRandom):
    pymc_dist = pm.NegativeBinomial
    pymc_dist_params = {"n": 100, "p": 0.33}
    expected_rv_op_params = {"n": 100, "p": 0.33}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestNegativeBinomialMuSigma(BaseTestDistributionRandom):
    pymc_dist = pm.NegativeBinomial
    pymc_dist_params = {"mu": 5.0, "alpha": 8.0}
    expected_n, expected_p = pm.NegativeBinomial.get_n_p(
        mu=pymc_dist_params["mu"],
        alpha=pymc_dist_params["alpha"],
        n=None,
        p=None,
    )
    expected_rv_op_params = {"n": expected_n, "p": expected_p}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestBernoulli(BaseTestDistributionRandom):
    pymc_dist = pm.Bernoulli
    pymc_dist_params = {"p": 0.33}
    expected_rv_op_params = {"p": 0.33}
    reference_dist_params = {"p": 0.33}
    reference_dist = seeded_scipy_distribution_builder("bernoulli")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
    ]


class TestBernoulliLogitP(BaseTestDistributionRandom):
    pymc_dist = pm.Bernoulli
    pymc_dist_params = {"logit_p": 1.0}
    expected_rv_op_params = {"p": expit(1.0)}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestPoisson(BaseTestDistributionRandom):
    pymc_dist = pm.Poisson
    pymc_dist_params = {"mu": 4.0}
    expected_rv_op_params = {"mu": 4.0}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestMvNormalCov(BaseTestDistributionRandom):
    pymc_dist = pm.MvNormal
    pymc_dist_params = {
        "mu": np.array([1.0, 2.0]),
        "cov": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    expected_rv_op_params = {
        "mu": np.array([1.0, 2.0]),
        "cov": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    sizes_to_check = [None, (1), (2, 3)]
    sizes_expected = [(2,), (1, 2), (2, 3, 2)]
    reference_dist_params = {
        "mean": np.array([1.0, 2.0]),
        "cov": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    reference_dist = seeded_numpy_distribution_builder("multivariate_normal")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
        "check_mu_broadcast_helper",
    ]

    def check_mu_broadcast_helper(self):
        """Test that mu is broadcasted to the shape of cov"""
        x = pm.MvNormal.dist(mu=1, cov=np.eye(3))
        mu = x.owner.inputs[3]
        assert mu.eval().shape == (3,)

        x = pm.MvNormal.dist(mu=np.ones(1), cov=np.eye(3))
        mu = x.owner.inputs[3]
        assert mu.eval().shape == (3,)

        x = pm.MvNormal.dist(mu=np.ones((1, 1)), cov=np.eye(3))
        mu = x.owner.inputs[3]
        assert mu.eval().shape == (1, 3)

        x = pm.MvNormal.dist(mu=np.ones((10, 1)), cov=np.eye(3))
        mu = x.owner.inputs[3]
        assert mu.eval().shape == (10, 3)

        # Cov is artificually limited to being 2D
        # x = pm.MvNormal.dist(mu=np.ones((10, 1)), cov=np.full((2, 3, 3), np.eye(3)))
        # mu = x.owner.inputs[3]
        # assert mu.eval().shape == (10, 2, 3)


class TestMvNormalChol(BaseTestDistributionRandom):
    pymc_dist = pm.MvNormal
    pymc_dist_params = {
        "mu": np.array([1.0, 2.0]),
        "chol": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    expected_rv_op_params = {
        "mu": np.array([1.0, 2.0]),
        "cov": quaddist_matrix(chol=pymc_dist_params["chol"]).eval(),
    }
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestMvNormalTau(BaseTestDistributionRandom):
    pymc_dist = pm.MvNormal
    pymc_dist_params = {
        "mu": np.array([1.0, 2.0]),
        "tau": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    expected_rv_op_params = {
        "mu": np.array([1.0, 2.0]),
        "cov": quaddist_matrix(tau=pymc_dist_params["tau"]).eval(),
    }
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestMvNormalMisc:
    def test_with_chol_rv(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0, size=3)
            sd_dist = pm.Exponential.dist(1.0, size=3)
            # pylint: disable=unpacking-non-sequence
            chol, _, _ = pm.LKJCholeskyCov(
                "chol_cov", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            # pylint: enable=unpacking-non-sequence
            mv = pm.MvNormal("mv", mu, chol=chol, size=4)
            prior = pm.sample_prior_predictive(samples=10, return_inferencedata=False)

        assert prior["mv"].shape == (10, 4, 3)

    def test_with_cov_rv(
        self,
    ):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0, shape=3)
            sd_dist = pm.Exponential.dist(1.0, shape=3)
            # pylint: disable=unpacking-non-sequence
            chol, corr, stds = pm.LKJCholeskyCov(
                "chol_cov", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            # pylint: enable=unpacking-non-sequence
            mv = pm.MvNormal("mv", mu, cov=pm.math.dot(chol, chol.T), size=4)
            prior = pm.sample_prior_predictive(samples=10, return_inferencedata=False)

        assert prior["mv"].shape == (10, 4, 3)

    def test_issue_3758(self):
        np.random.seed(42)
        ndim = 50
        with pm.Model() as model:
            a = pm.Normal("a", sigma=100, shape=ndim)
            b = pm.Normal("b", mu=a, sigma=1, shape=ndim)
            c = pm.MvNormal("c", mu=a, chol=np.linalg.cholesky(np.eye(ndim)), shape=ndim)
            d = pm.MvNormal("d", mu=a, cov=np.eye(ndim), shape=ndim)
            samples = pm.sample_prior_predictive(1000, return_inferencedata=False)

        for var in "abcd":
            assert not np.isnan(np.std(samples[var]))

        for var in "bcd":
            std = np.std(samples[var] - samples["a"])
            npt.assert_allclose(std, 1, rtol=2e-2)

    def test_issue_3829(self):
        with pm.Model() as model:
            x = pm.MvNormal("x", mu=np.zeros(5), cov=np.eye(5), shape=(2, 5))
            trace_pp = pm.sample_prior_predictive(50, return_inferencedata=False)

        assert np.shape(trace_pp["x"][0]) == (2, 5)

    def test_issue_3706(self):
        N = 10
        Sigma = np.eye(2)

        with pm.Model() as model:
            X = pm.MvNormal("X", mu=np.zeros(2), cov=Sigma, shape=(N, 2))
            betas = pm.Normal("betas", 0, 1, shape=2)
            y = pm.Deterministic("y", pm.math.dot(X, betas))

            prior_pred = pm.sample_prior_predictive(1, return_inferencedata=False)

        assert prior_pred["X"].shape == (1, N, 2)


class TestMvStudentTCov(BaseTestDistributionRandom):
    def mvstudentt_rng_fn(self, size, nu, mu, cov, rng):
        mv_samples = rng.multivariate_normal(np.zeros_like(mu), cov, size=size)
        chi2_samples = rng.chisquare(nu, size=size)
        return (mv_samples / np.sqrt(chi2_samples[:, None] / nu)) + mu

    pymc_dist = pm.MvStudentT
    pymc_dist_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "cov": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    expected_rv_op_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "cov": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    sizes_to_check = [None, (1), (2, 3)]
    sizes_expected = [(2,), (1, 2), (2, 3, 2)]
    reference_dist_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "cov": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    reference_dist = lambda self: functools.partial(
        self.mvstudentt_rng_fn, rng=self.get_random_state()
    )
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
        "check_errors",
        "check_mu_broadcast_helper",
    ]

    def check_errors(self):
        msg = "nu must be a scalar (ndim=0)."
        with pm.Model():
            with pytest.raises(ValueError, match=re.escape(msg)):
                mvstudentt = pm.MvStudentT(
                    "mvstudentt",
                    nu=np.array([1, 2]),
                    mu=np.ones(2),
                    cov=np.full((2, 2), np.ones(2)),
                )

    def check_mu_broadcast_helper(self):
        """Test that mu is broadcasted to the shape of cov"""
        x = pm.MvStudentT.dist(nu=4, mu=1, cov=np.eye(3))
        mu = x.owner.inputs[4]
        assert mu.eval().shape == (3,)

        x = pm.MvStudentT.dist(nu=4, mu=np.ones(1), cov=np.eye(3))
        mu = x.owner.inputs[4]
        assert mu.eval().shape == (3,)

        x = pm.MvStudentT.dist(nu=4, mu=np.ones((1, 1)), cov=np.eye(3))
        mu = x.owner.inputs[4]
        assert mu.eval().shape == (1, 3)

        x = pm.MvStudentT.dist(nu=4, mu=np.ones((10, 1)), cov=np.eye(3))
        mu = x.owner.inputs[4]
        assert mu.eval().shape == (10, 3)

        # Cov is artificually limited to being 2D
        # x = pm.MvStudentT.dist(nu=4, mu=np.ones((10, 1)), cov=np.full((2, 3, 3), np.eye(3)))
        # mu = x.owner.inputs[4]
        # assert mu.eval().shape == (10, 2, 3)


class TestMvStudentTChol(BaseTestDistributionRandom):
    pymc_dist = pm.MvStudentT
    pymc_dist_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "chol": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    expected_rv_op_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "cov": quaddist_matrix(chol=pymc_dist_params["chol"]).eval(),
    }
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestMvStudentTTau(BaseTestDistributionRandom):
    pymc_dist = pm.MvStudentT
    pymc_dist_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "tau": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    expected_rv_op_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "cov": quaddist_matrix(tau=pymc_dist_params["tau"]).eval(),
    }
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestDirichlet(BaseTestDistributionRandom):
    pymc_dist = pm.Dirichlet
    pymc_dist_params = {"a": np.array([1.0, 2.0])}
    expected_rv_op_params = {"a": np.array([1.0, 2.0])}
    sizes_to_check = [None, (1), (4,), (3, 4)]
    sizes_expected = [(2,), (1, 2), (4, 2), (3, 4, 2)]
    reference_dist_params = {"alpha": np.array([1.0, 2.0])}
    reference_dist = seeded_numpy_distribution_builder("dirichlet")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestMultinomial(BaseTestDistributionRandom):
    pymc_dist = pm.Multinomial
    pymc_dist_params = {"n": 85, "p": np.array([0.28, 0.62, 0.10])}
    expected_rv_op_params = {"n": 85, "p": np.array([0.28, 0.62, 0.10])}
    sizes_to_check = [None, (1), (4,), (3, 2)]
    sizes_expected = [(3,), (1, 3), (4, 3), (3, 2, 3)]
    reference_dist_params = {"n": 85, "pvals": np.array([0.28, 0.62, 0.10])}
    reference_dist = seeded_numpy_distribution_builder("multinomial")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestDirichletMultinomial(BaseTestDistributionRandom):
    pymc_dist = pm.DirichletMultinomial

    pymc_dist_params = {"n": 85, "a": np.array([1.0, 2.0, 1.5, 1.5])}
    expected_rv_op_params = {"n": 85, "a": np.array([1.0, 2.0, 1.5, 1.5])}

    sizes_to_check = [None, 1, (4,), (3, 4)]
    sizes_expected = [(4,), (1, 4), (4, 4), (3, 4, 4)]

    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
        "check_random_draws",
    ]

    def check_random_draws(self):
        default_rng = aesara.shared(np.random.default_rng(1234))
        draws = pm.DirichletMultinomial.dist(
            n=np.array([5, 100]),
            a=np.array([[0.001, 0.001, 0.001, 1000], [1000, 1000, 0.001, 0.001]]),
            size=(2, 3, 2),
            rng=default_rng,
        ).eval()
        assert np.all(draws.sum(-1) == np.array([5, 100]))
        assert np.all((draws.sum(-2)[:, :, 0] > 30) & (draws.sum(-2)[:, :, 0] <= 70))
        assert np.all((draws.sum(-2)[:, :, 1] > 30) & (draws.sum(-2)[:, :, 1] <= 70))
        assert np.all((draws.sum(-2)[:, :, 2] >= 0) & (draws.sum(-2)[:, :, 2] <= 2))
        assert np.all((draws.sum(-2)[:, :, 3] > 3) & (draws.sum(-2)[:, :, 3] <= 5))


class TestDirichletMultinomial_1D_n_2D_a(BaseTestDistributionRandom):
    pymc_dist = pm.DirichletMultinomial
    pymc_dist_params = {
        "n": np.array([23, 29]),
        "a": np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]),
    }
    sizes_to_check = [None, (1, 2), (4, 2), (3, 4, 2)]
    sizes_expected = [(2, 4), (1, 2, 4), (4, 2, 4), (3, 4, 2, 4)]
    checks_to_run = ["check_rv_size"]


class TestStickBreakingWeights(BaseTestDistributionRandom):
    pymc_dist = pm.StickBreakingWeights
    pymc_dist_params = {"alpha": 2.0, "K": 19}
    expected_rv_op_params = {"alpha": 2.0, "K": 19}
    sizes_to_check = [None, 17, (5,), (11, 5), (3, 13, 5)]
    sizes_expected = [
        (20,),
        (17, 20),
        (
            5,
            20,
        ),
        (11, 5, 20),
        (3, 13, 5, 20),
    ]
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
        "check_basic_properties",
    ]

    def check_basic_properties(self):
        default_rng = aesara.shared(np.random.default_rng(1234))
        draws = pm.StickBreakingWeights.dist(
            alpha=3.5,
            K=19,
            size=(2, 3, 5),
            rng=default_rng,
        ).eval()

        assert np.allclose(draws.sum(-1), 1)
        assert np.all(draws >= 0)
        assert np.all(draws <= 1)


class TestCategorical(BaseTestDistributionRandom):
    pymc_dist = pm.Categorical
    pymc_dist_params = {"p": np.array([0.28, 0.62, 0.10])}
    expected_rv_op_params = {"p": np.array([0.28, 0.62, 0.10])}
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
    ]


class TestLogitCategorical(BaseTestDistributionRandom):
    pymc_dist = pm.Categorical
    pymc_dist_params = {"logit_p": np.array([[0.28, 0.62, 0.10], [0.28, 0.62, 0.10]])}
    expected_rv_op_params = {
        "p": softmax(np.array([[0.28, 0.62, 0.10], [0.28, 0.62, 0.10]]), axis=-1)
    }
    tests_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
    ]

    @pytest.mark.parametrize(
        "p, logit_p, expected",
        [
            (None, None, "Must specify either p or logit_p."),
            (0.5, 0.5, "Can't specify both p and logit_p."),
        ],
    )
    def test_categorical_init_fail(self, p, logit_p, expected):
        with pm.Model() as model:
            with pytest.raises(ValueError, match=f"Incompatible parametrization. {expected}"):
                pm.Categorical("x", p=p, logit_p=logit_p)


class TestGeometric(BaseTestDistributionRandom):
    pymc_dist = pm.Geometric
    pymc_dist_params = {"p": 0.9}
    expected_rv_op_params = {"p": 0.9}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestHyperGeometric(BaseTestDistributionRandom):
    pymc_dist = pm.HyperGeometric
    pymc_dist_params = {"N": 20, "k": 12, "n": 5}
    expected_rv_op_params = {
        "ngood": pymc_dist_params["k"],
        "nbad": pymc_dist_params["N"] - pymc_dist_params["k"],
        "nsample": pymc_dist_params["n"],
    }
    reference_dist_params = expected_rv_op_params
    reference_dist = seeded_numpy_distribution_builder("hypergeometric")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
    ]


class TestLogistic(BaseTestDistributionRandom):
    pymc_dist = pm.Logistic
    pymc_dist_params = {"mu": 1.0, "s": 2.0}
    expected_rv_op_params = {"mu": 1.0, "s": 2.0}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestLogNormal(BaseTestDistributionRandom):
    pymc_dist = pm.LogNormal
    pymc_dist_params = {"mu": 1.0, "sigma": 5.0}
    expected_rv_op_params = {"mu": 1.0, "sigma": 5.0}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestLognormalTau(BaseTestDistributionRandom):
    pymc_dist = pm.Lognormal
    tau, sigma = get_tau_sigma(tau=25.0)
    pymc_dist_params = {"mu": 1.0, "tau": 25.0}
    expected_rv_op_params = {"mu": 1.0, "sigma": sigma}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestLognormalSd(BaseTestDistributionRandom):
    pymc_dist = pm.Lognormal
    pymc_dist_params = {"mu": 1.0, "sigma": 5.0}
    expected_rv_op_params = {"mu": 1.0, "sigma": 5.0}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestTriangular(BaseTestDistributionRandom):
    pymc_dist = pm.Triangular
    pymc_dist_params = {"lower": 0, "upper": 1, "c": 0.5}
    expected_rv_op_params = {"lower": 0, "c": 0.5, "upper": 1}
    reference_dist_params = {"left": 0, "mode": 0.5, "right": 1}
    reference_dist = seeded_numpy_distribution_builder("triangular")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
    ]


class TestVonMises(BaseTestDistributionRandom):
    pymc_dist = pm.VonMises
    pymc_dist_params = {"mu": -2.1, "kappa": 5}
    expected_rv_op_params = {"mu": -2.1, "kappa": 5}
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestWeibull(BaseTestDistributionRandom):
    def weibull_rng_fn(self, size, alpha, beta, std_weibull_rng_fct):
        return beta * std_weibull_rng_fct(alpha, size=size)

    def seeded_weibul_rng_fn(self):
        std_weibull_rng_fct = functools.partial(
            getattr(np.random.RandomState, "weibull"), self.get_random_state()
        )
        return functools.partial(self.weibull_rng_fn, std_weibull_rng_fct=std_weibull_rng_fct)

    pymc_dist = pm.Weibull
    pymc_dist_params = {"alpha": 1.0, "beta": 2.0}
    expected_rv_op_params = {"alpha": 1.0, "beta": 2.0}
    reference_dist_params = {"alpha": 1.0, "beta": 2.0}
    reference_dist = seeded_weibul_rng_fn
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestBetaBinomial(BaseTestDistributionRandom):
    pymc_dist = pm.BetaBinomial
    pymc_dist_params = {"alpha": 2.0, "beta": 1.0, "n": 5}
    expected_rv_op_params = {"n": 5, "alpha": 2.0, "beta": 1.0}
    reference_dist_params = {"n": 5, "a": 2.0, "b": 1.0}
    reference_dist = seeded_scipy_distribution_builder("betabinom")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


@pytest.mark.skipif(
    condition=_polyagamma_not_installed,
    reason="`polyagamma package is not available/installed.",
)
class TestPolyaGamma(BaseTestDistributionRandom):
    def polyagamma_rng_fn(self, size, h, z, rng):
        return random_polyagamma(h, z, size=size, random_state=rng._bit_generator)

    pymc_dist = pm.PolyaGamma
    pymc_dist_params = {"h": 1.0, "z": 0.0}
    expected_rv_op_params = {"h": 1.0, "z": 0.0}
    reference_dist_params = {"h": 1.0, "z": 0.0}
    reference_dist = lambda self: functools.partial(
        self.polyagamma_rng_fn, rng=self.get_random_state()
    )
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestDiscreteUniform(BaseTestDistributionRandom):
    def discrete_uniform_rng_fn(self, size, lower, upper, rng):
        return st.randint.rvs(lower, upper + 1, size=size, random_state=rng)

    pymc_dist = pm.DiscreteUniform
    pymc_dist_params = {"lower": -1, "upper": 9}
    expected_rv_op_params = {"lower": -1, "upper": 9}
    reference_dist_params = {"lower": -1, "upper": 9}
    reference_dist = lambda self: functools.partial(
        self.discrete_uniform_rng_fn, rng=self.get_random_state()
    )
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestConstant(BaseTestDistributionRandom):
    def constant_rng_fn(self, size, c):
        if size is None:
            return c
        return np.full(size, c)

    pymc_dist = pm.Constant
    pymc_dist_params = {"c": 3}
    expected_rv_op_params = {"c": 3}
    reference_dist_params = {"c": 3}
    reference_dist = lambda self: self.constant_rng_fn
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]

    @pytest.mark.parametrize("floatX", ["float32", "float64"])
    @pytest.mark.xfail(
        sys.platform == "win32", reason="https://github.com/aesara-devs/aesara/issues/871"
    )
    def test_dtype(self, floatX):
        with aesara.config.change_flags(floatX=floatX):
            assert pm.Constant.dist(2**4).dtype == "int8"
            assert pm.Constant.dist(2**16).dtype == "int32"
            assert pm.Constant.dist(2**32).dtype == "int64"
            assert pm.Constant.dist(2.0).dtype == floatX


class TestOrderedLogistic(BaseTestDistributionRandom):
    pymc_dist = _OrderedLogistic
    pymc_dist_params = {"eta": 0, "cutpoints": np.array([-2, 0, 2])}
    expected_rv_op_params = {"p": np.array([0.11920292, 0.38079708, 0.38079708, 0.11920292])}
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
    ]

    @pytest.mark.parametrize(
        "eta, cutpoints, expected",
        [
            (0, [-2.0, 0, 2.0], (4,)),
            ([-1], [-2.0, 0, 2.0], (1, 4)),
            ([1.0, -2.0], [-1.0, 0, 1.0], (2, 4)),
            (np.zeros((3, 2)), [-2.0, 0, 1.0], (3, 2, 4)),
            (np.ones((5, 2)), [[-2.0, 0, 1.0], [-1.0, 0, 1.0]], (5, 2, 4)),
            (np.ones((3, 5, 2)), [[-2.0, 0, 1.0], [-1.0, 0, 1.0]], (3, 5, 2, 4)),
        ],
    )
    def test_shape_inputs(self, eta, cutpoints, expected):
        """
        This test checks when providing different shapes for `eta` parameters.
        """
        categorical = _OrderedLogistic.dist(
            eta=eta,
            cutpoints=cutpoints,
        )
        p = categorical.owner.inputs[3].eval()
        assert p.shape == expected


class TestOrderedProbit(BaseTestDistributionRandom):
    pymc_dist = _OrderedProbit
    pymc_dist_params = {"eta": 0, "cutpoints": np.array([-2, 0, 2])}
    expected_rv_op_params = {"p": np.array([0.02275013, 0.47724987, 0.47724987, 0.02275013])}
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
    ]

    @pytest.mark.parametrize(
        "eta, cutpoints, sigma, expected",
        [
            (0, [-2.0, 0, 2.0], 1.0, (4,)),
            ([-1], [-1.0, 0, 2.0], [2.0], (1, 4)),
            ([1.0, -2.0], [-1.0, 0, 1.0], 1.0, (2, 4)),
            ([1.0, -2.0, 3.0], [-1.0, 0, 2.0], np.ones((1, 3)), (1, 3, 4)),
            (np.zeros((2, 3)), [-2.0, 0, 1.0], [1.0, 2.0, 5.0], (2, 3, 4)),
            (np.ones((2, 3)), [-1.0, 0, 1.0], np.ones((2, 3)), (2, 3, 4)),
            (np.zeros((5, 2)), [[-2, 0, 1], [-1, 0, 1]], np.ones((2, 5, 2)), (2, 5, 2, 4)),
        ],
    )
    def test_shape_inputs(self, eta, cutpoints, sigma, expected):
        """
        This test checks when providing different shapes for `eta` and `sigma` parameters.
        """
        categorical = _OrderedProbit.dist(
            eta=eta,
            cutpoints=cutpoints,
            sigma=sigma,
        )
        p = categorical.owner.inputs[3].eval()
        assert p.shape == expected


class TestOrderedMultinomial(BaseTestDistributionRandom):
    pymc_dist = _OrderedMultinomial
    pymc_dist_params = {"eta": 0, "cutpoints": np.array([-2, 0, 2]), "n": 1000}
    sizes_to_check = [None, (1), (4,), (3, 2)]
    sizes_expected = [(4,), (1, 4), (4, 4), (3, 2, 4)]
    expected_rv_op_params = {
        "n": 1000,
        "p": np.array([0.11920292, 0.38079708, 0.38079708, 0.11920292]),
    }
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
    ]


class TestWishart(BaseTestDistributionRandom):
    def wishart_rng_fn(self, size, nu, V, rng):
        return st.wishart.rvs(np.int(nu), V, size=size, random_state=rng)

    pymc_dist = pm.Wishart

    V = np.eye(3)
    pymc_dist_params = {"nu": 4, "V": V}
    reference_dist_params = {"nu": 4, "V": V}
    expected_rv_op_params = {"nu": 4, "V": V}
    sizes_to_check = [None, 1, (4, 5)]
    sizes_expected = [
        (3, 3),
        (1, 3, 3),
        (4, 5, 3, 3),
    ]
    reference_dist = lambda self: functools.partial(
        self.wishart_rng_fn, rng=self.get_random_state()
    )
    checks_to_run = [
        "check_rv_size",
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size_batched_params",
    ]

    def check_rv_size_batched_params(self):
        for size in (None, (2,), (1, 2), (4, 3, 2)):
            x = pm.Wishart.dist(nu=4, V=np.stack([np.eye(3), np.eye(3)]), size=size)

            if size is None:
                expected_shape = (2, 3, 3)
            else:
                expected_shape = size + (3, 3)

            assert tuple(x.shape.eval()) == expected_shape

            # RNG does not currently support batched parameters, whet it does this test
            # should be updated to check that draws also have the expected shape
            with pytest.raises(ValueError):
                x.eval()


class TestMatrixNormal(BaseTestDistributionRandom):

    pymc_dist = pm.MatrixNormal

    mu = np.random.random((3, 3))
    row_cov = np.eye(3)
    col_cov = np.eye(3)
    pymc_dist_params = {"mu": mu, "rowcov": row_cov, "colcov": col_cov}
    expected_rv_op_params = {"mu": mu, "rowcov": row_cov, "colcov": col_cov}

    sizes_to_check = (None, (1,), (2, 4))
    sizes_expected = [(3, 3), (1, 3, 3), (2, 4, 3, 3)]

    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
        "check_draws",
        "check_errors",
        "check_random_variable_prior",
    ]

    def check_draws(self):
        delta = 0.05  # limit for KS p-value
        n_fails = 10  # Allows the KS fails a certain number of times

        def ref_rand(mu, rowcov, colcov):
            return st.matrix_normal.rvs(mean=mu, rowcov=rowcov, colcov=colcov)

        with pm.Model(rng_seeder=1):
            matrixnormal = pm.MatrixNormal(
                "matnormal",
                mu=np.random.random((3, 3)),
                rowcov=np.eye(3),
                colcov=np.eye(3),
            )
            check = pm.sample_prior_predictive(n_fails, return_inferencedata=False)

        ref_smp = ref_rand(mu=np.random.random((3, 3)), rowcov=np.eye(3), colcov=np.eye(3))

        p, f = delta, n_fails
        while p <= delta and f > 0:
            matrixnormal_smp = check["matnormal"]

            p = np.min(
                [
                    st.ks_2samp(
                        np.atleast_1d(matrixnormal_smp).flatten(),
                        np.atleast_1d(ref_smp).flatten(),
                    )
                ]
            )
            f -= 1

        assert p > delta

    def check_errors(self):
        with pm.Model():
            matrixnormal = pm.MatrixNormal(
                "matnormal",
                mu=np.random.random((3, 3)),
                rowcov=np.eye(3),
                colcov=np.eye(3),
            )
            with pytest.raises(ValueError):
                logp(matrixnormal, aesara.tensor.ones((3, 3, 3)))

    def check_random_variable_prior(self):
        """
        This test checks for shape correctness when using MatrixNormal distribution
        with parameters as random variables.
        Originally reported - https://github.com/pymc-devs/pymc/issues/3585
        """
        K = 3
        D = 15
        mu_0 = np.zeros((D, K))
        lambd = 1.0
        with pm.Model() as model:
            sd_dist = pm.HalfCauchy.dist(beta=2.5, size=D)
            packedL = pm.LKJCholeskyCov("packedL", eta=2, n=D, sd_dist=sd_dist, compute_corr=False)
            L = pm.expand_packed_triangular(D, packedL, lower=True)
            Sigma = pm.Deterministic("Sigma", L.dot(L.T))  # D x D covariance
            mu = pm.MatrixNormal(
                "mu", mu=mu_0, rowcov=(1 / lambd) * Sigma, colcov=np.eye(K), shape=(D, K)
            )
            prior = pm.sample_prior_predictive(2, return_inferencedata=False)

        assert prior["mu"].shape == (2, D, K)


class TestInterpolated(BaseTestDistributionRandom):
    def interpolated_rng_fn(self, size, mu, sigma, rng):
        return st.norm.rvs(loc=mu, scale=sigma, size=size)

    pymc_dist = pm.Interpolated

    # Dummy values for RV size testing
    mu = sigma = 1
    x_points = pdf_points = np.linspace(1, 100, 100)

    pymc_dist_params = {"x_points": x_points, "pdf_points": pdf_points}
    reference_dist_params = {"mu": mu, "sigma": sigma}

    reference_dist = lambda self: functools.partial(
        self.interpolated_rng_fn, rng=self.get_random_state()
    )
    checks_to_run = [
        "check_rv_size",
        "check_draws",
    ]

    def check_draws(self):
        for mu in R.vals:
            for sigma in Rplus.vals:
                # pylint: disable=cell-var-from-loop
                rng = self.get_random_state()

                def ref_rand(size):
                    return st.norm.rvs(loc=mu, scale=sigma, size=size, random_state=rng)

                class TestedInterpolated(pm.Interpolated):
                    rv_op = interpolated

                    @classmethod
                    def dist(cls, **kwargs):
                        x_points = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)
                        pdf_points = st.norm.pdf(x_points, loc=mu, scale=sigma)
                        return super().dist(x_points=x_points, pdf_points=pdf_points, **kwargs)

                pymc_random(
                    TestedInterpolated,
                    {},
                    extra_args={"rng": aesara.shared(rng)},
                    ref_rand=ref_rand,
                )


class TestKroneckerNormal(BaseTestDistributionRandom):
    def kronecker_rng_fn(self, size, mu, covs=None, sigma=None, rng=None):
        cov = pm.math.kronecker(covs[0], covs[1]).eval()
        cov += sigma**2 * np.identity(cov.shape[0])
        return st.multivariate_normal.rvs(mean=mu, cov=cov, size=size)

    pymc_dist = pm.KroneckerNormal

    n = 3
    N = n**2
    covs = [RandomPdMatrix(n), RandomPdMatrix(n)]
    mu = np.random.random(N) * 0.1
    sigma = 1

    pymc_dist_params = {"mu": mu, "covs": covs, "sigma": sigma}
    expected_rv_op_params = {"mu": mu, "covs": covs, "sigma": sigma}
    reference_dist_params = {"mu": mu, "covs": covs, "sigma": sigma}
    sizes_to_check = [None, (), 1, (1,), 5, (4, 5), (2, 4, 2)]
    sizes_expected = [(N,), (N,), (1, N), (1, N), (5, N), (4, 5, N), (2, 4, 2, N)]

    reference_dist = lambda self: functools.partial(
        self.kronecker_rng_fn, rng=self.get_random_state()
    )
    checks_to_run = [
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestLKJCorr(BaseTestDistributionRandom):
    pymc_dist = pm.LKJCorr
    pymc_dist_params = {"n": 3, "eta": 1.0}
    expected_rv_op_params = {"n": 3, "eta": 1.0}

    sizes_to_check = [None, (), 1, (1,), 5, (4, 5), (2, 4, 2)]
    sizes_expected = [
        (3,),
        (3,),
        (1, 3),
        (1, 3),
        (5, 3),
        (4, 5, 3),
        (2, 4, 2, 3),
    ]

    tests_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
        "check_draws_match_expected",
    ]

    def check_draws_match_expected(self):
        def ref_rand(size, n, eta):
            shape = int(n * (n - 1) // 2)
            beta = eta - 1 + n / 2
            return (st.beta.rvs(size=(size, shape), a=beta, b=beta) - 0.5) * 2

        pymc_random(
            pm.LKJCorr,
            {
                "n": Domain([2, 10, 50], edges=(None, None)),
                "eta": Domain([1.0, 10.0, 100.0], edges=(None, None)),
            },
            ref_rand=ref_rand,
            size=1000,
        )


class TestLKJCholeskyCov(BaseTestDistributionRandom):
    pymc_dist = _LKJCholeskyCov
    pymc_dist_params = {"eta": 1.0, "n": 3, "sd_dist": pm.Constant.dist([0.5, 1.0, 2.0])}
    expected_rv_op_params = {"n": 3, "eta": 1.0, "sd_dist": pm.Constant.dist([0.5, 1.0, 2.0])}
    size = None

    sizes_to_check = [None, (), 1, (1,), 5, (4, 5), (2, 4, 2)]
    sizes_expected = [
        (6,),
        (6,),
        (1, 6),
        (1, 6),
        (5, 6),
        (4, 5, 6),
        (2, 4, 2, 6),
    ]

    tests_to_run = [
        "check_rv_size",
        "check_draws_match_expected",
    ]

    def check_rv_size(self):
        for size, expected in zip(self.sizes_to_check, self.sizes_expected):
            sd_dist = pm.Exponential.dist(1, size=(*to_tuple(size), 3))
            pymc_rv = self.pymc_dist.dist(n=3, eta=1, sd_dist=sd_dist, size=size)
            expected_symbolic = tuple(pymc_rv.shape.eval())
            actual = pymc_rv.eval().shape
            assert actual == expected_symbolic == expected

    def check_draws_match_expected(self):
        # TODO: Find better comparison:
        rng = aesara.shared(self.get_random_state(reset=True))
        x = _LKJCholeskyCov.dist(n=2, eta=10_000, sd_dist=pm.Constant.dist([0.5, 2.0]), rng=rng)
        assert np.all(np.abs(x.eval() - np.array([0.5, 0, 2.0])) < 0.01)


class TestDensityDist:
    @pytest.mark.parametrize("size", [(), (3,), (3, 2)], ids=str)
    def test_density_dist_with_random(self, size):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1)
            obs = pm.DensityDist(
                "density_dist",
                mu,
                random=lambda mu, rng=None, size=None: rng.normal(loc=mu, scale=1, size=size),
                observed=np.random.randn(100, *size),
                size=size,
            )

        assert obs.eval().shape == (100,) + size

    def test_density_dist_without_random(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1)
            pm.DensityDist(
                "density_dist",
                mu,
                logp=lambda value, mu: logp(pm.Normal.dist(mu, 1, size=100), value),
                observed=np.random.randn(100),
                initval=0,
            )
            idata = pm.sample(tune=50, draws=100, cores=1, step=pm.Metropolis())

        with pytest.raises(NotImplementedError):
            pm.sample_posterior_predictive(idata, model=model)

    @pytest.mark.parametrize("size", [(), (3,), (3, 2)], ids=str)
    def test_density_dist_with_random_multivariate(self, size):
        supp_shape = 5
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1, size=supp_shape)
            obs = pm.DensityDist(
                "density_dist",
                mu,
                random=lambda mu, rng=None, size=None: rng.multivariate_normal(
                    mean=mu, cov=np.eye(len(mu)), size=size
                ),
                observed=np.random.randn(100, *size, supp_shape),
                size=size,
                ndims_params=[1],
                ndim_supp=1,
            )

        assert obs.eval().shape == (100,) + size + (supp_shape,)


class TestNestedRandom(SeededTest):
    def build_model(self, distribution, shape, nested_rvs_info):
        with pm.Model() as model:
            nested_rvs = {}
            for rv_name, info in nested_rvs_info.items():
                try:
                    value, nested_shape = info
                    loc = 0.0
                except ValueError:
                    value, nested_shape, loc = info
                if value is None:
                    nested_rvs[rv_name] = pm.Uniform(
                        rv_name,
                        0 + loc,
                        1 + loc,
                        shape=nested_shape,
                    )
                else:
                    nested_rvs[rv_name] = value * np.ones(nested_shape)
            rv = distribution(
                "target",
                shape=shape,
                **nested_rvs,
            )
        return model, rv, nested_rvs

    def sample_prior(self, distribution, shape, nested_rvs_info, prior_samples):
        model, rv, nested_rvs = self.build_model(
            distribution,
            shape,
            nested_rvs_info,
        )
        with model:
            return pm.sample_prior_predictive(prior_samples, return_inferencedata=False)

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "mu", "alpha"],
        [
            [10, (3,), (None, tuple()), (None, (3,))],
            [10, (3,), (None, (3,)), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, (4, 3)),
            ],
        ],
        ids=str,
    )
    def test_NegativeBinomial(
        self,
        prior_samples,
        shape,
        mu,
        alpha,
    ):
        prior = self.sample_prior(
            distribution=pm.NegativeBinomial,
            shape=shape,
            nested_rvs_info=dict(mu=mu, alpha=alpha),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "psi", "mu", "alpha"],
        [
            [10, (3,), (0.5, tuple()), (None, tuple()), (None, (3,))],
            [10, (3,), (0.5, (3,)), (None, tuple()), (None, (3,))],
            [10, (3,), (0.5, tuple()), (None, (3,)), (None, tuple())],
            [10, (3,), (0.5, (3,)), (None, (3,)), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (0.5, (3,)),
                (None, (3,)),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (0.5, (3,)),
                (None, (3,)),
                (None, (4, 3)),
            ],
        ],
        ids=str,
    )
    def test_ZeroInflatedNegativeBinomial(
        self,
        prior_samples,
        shape,
        psi,
        mu,
        alpha,
    ):
        prior = self.sample_prior(
            distribution=pm.ZeroInflatedNegativeBinomial,
            shape=shape,
            nested_rvs_info=dict(psi=psi, mu=mu, alpha=alpha),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "nu", "sigma"],
        [
            [10, (3,), (None, tuple()), (None, (3,))],
            [10, (3,), (None, tuple()), (None, (3,))],
            [10, (3,), (None, (3,)), (None, tuple())],
            [10, (3,), (None, (3,)), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, (4, 3)),
            ],
        ],
        ids=str,
    )
    def test_Rice(
        self,
        prior_samples,
        shape,
        nu,
        sigma,
    ):
        prior = self.sample_prior(
            distribution=pm.Rice,
            shape=shape,
            nested_rvs_info=dict(nu=nu, sigma=sigma),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "mu", "sigma", "lower", "upper"],
        [
            [10, (3,), (None, tuple()), (1.0, tuple()), (None, tuple(), -1), (None, (3,))],
            [10, (3,), (None, tuple()), (1.0, tuple()), (None, tuple(), -1), (None, (3,))],
            [10, (3,), (None, tuple()), (1.0, tuple()), (None, (3,), -1), (None, tuple())],
            [10, (3,), (None, tuple()), (1.0, tuple()), (None, (3,), -1), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (1.0, tuple()),
                (None, (3,), -1),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (1.0, tuple()),
                (None, (3,), -1),
                (None, (4, 3)),
            ],
            [10, (3,), (0.0, tuple()), (None, tuple()), (None, tuple(), -1), (None, (3,))],
            [10, (3,), (0.0, tuple()), (None, tuple()), (None, tuple(), -1), (None, (3,))],
            [10, (3,), (0.0, tuple()), (None, tuple()), (None, (3,), -1), (None, tuple())],
            [10, (3,), (0.0, tuple()), (None, tuple()), (None, (3,), -1), (None, tuple())],
            [
                10,
                (
                    4,
                    3,
                ),
                (0.0, tuple()),
                (None, (3,)),
                (None, (3,), -1),
                (None, (3,)),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (0.0, tuple()),
                (None, (3,)),
                (None, (3,), -1),
                (None, (4, 3)),
            ],
        ],
        ids=str,
    )
    def test_TruncatedNormal(
        self,
        prior_samples,
        shape,
        mu,
        sigma,
        lower,
        upper,
    ):
        prior = self.sample_prior(
            distribution=pm.TruncatedNormal,
            shape=shape,
            nested_rvs_info=dict(mu=mu, sigma=sigma, lower=lower, upper=upper),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape

    @pytest.mark.parametrize(
        ["prior_samples", "shape", "c", "lower", "upper"],
        [
            [10, (3,), (None, tuple()), (-1.0, (3,)), (2, tuple())],
            [10, (3,), (None, tuple()), (-1.0, tuple()), (None, tuple(), 1)],
            [10, (3,), (None, (3,)), (-1.0, tuple()), (None, tuple(), 1)],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (-1.0, tuple()),
                (None, (3,), 1),
            ],
            [
                10,
                (
                    4,
                    3,
                ),
                (None, (3,)),
                (None, tuple(), -1),
                (None, (3,), 1),
            ],
        ],
        ids=str,
    )
    def test_Triangular(
        self,
        prior_samples,
        shape,
        c,
        lower,
        upper,
    ):
        prior = self.sample_prior(
            distribution=pm.Triangular,
            shape=shape,
            nested_rvs_info=dict(c=c, lower=lower, upper=upper),
            prior_samples=prior_samples,
        )
        assert prior["target"].shape == (prior_samples,) + shape


def generate_shapes(include_params=False):
    # fmt: off
    mudim_as_event = [
        [None, 1, 3, 10, (10, 3), 100],
        [(3,)],
        [(1,), (3,)],
        ["cov", "chol", "tau"]
    ]
    # fmt: on
    mudim_as_dist = [
        [None, 1, 3, 10, (10, 3), 100],
        [(10, 3)],
        [(1,), (3,), (1, 1), (1, 3), (10, 1), (10, 3)],
        ["cov", "chol", "tau"],
    ]
    if not include_params:
        del mudim_as_event[-1]
        del mudim_as_dist[-1]
    data = itertools.chain(itertools.product(*mudim_as_event), itertools.product(*mudim_as_dist))
    return data


@pytest.mark.xfail(reason="This distribution has not been refactored for v4")
class TestMvGaussianRandomWalk(SeededTest):
    @pytest.mark.parametrize(
        ["sample_shape", "dist_shape", "mu_shape", "param"],
        generate_shapes(include_params=True),
        ids=str,
    )
    def test_with_np_arrays(self, sample_shape, dist_shape, mu_shape, param):
        dist = pm.MvGaussianRandomWalk.dist(
            mu=np.ones(mu_shape), **{param: np.eye(3)}, shape=dist_shape
        )
        output_shape = to_tuple(sample_shape) + dist_shape
        assert dist.random(size=sample_shape).shape == output_shape

    @pytest.mark.parametrize(
        ["sample_shape", "dist_shape", "mu_shape"],
        generate_shapes(include_params=False),
        ids=str,
    )
    def test_with_chol_rv(self, sample_shape, dist_shape, mu_shape):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0, shape=mu_shape)
            sd_dist = pm.Exponential.dist(1.0, shape=3)
            # pylint: disable=unpacking-non-sequence
            chol, corr, stds = pm.LKJCholeskyCov(
                "chol_cov", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            # pylint: enable=unpacking-non-sequence
            mv = pm.MvGaussianRandomWalk("mv", mu, chol=chol, shape=dist_shape)
            prior = pm.sample_prior_predictive(samples=sample_shape)

        assert prior["mv"].shape == to_tuple(sample_shape) + dist_shape

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        ["sample_shape", "dist_shape", "mu_shape"],
        generate_shapes(include_params=False),
        ids=str,
    )
    def test_with_cov_rv(self, sample_shape, dist_shape, mu_shape):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0, shape=mu_shape)
            sd_dist = pm.Exponential.dist(1.0, shape=3)
            # pylint: disable=unpacking-non-sequence
            chol, corr, stds = pm.LKJCholeskyCov(
                "chol_cov", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            # pylint: enable=unpacking-non-sequence
            mv = pm.MvGaussianRandomWalk("mv", mu, cov=pm.math.dot(chol, chol.T), shape=dist_shape)
            prior = pm.sample_prior_predictive(samples=sample_shape)

        assert prior["mv"].shape == to_tuple(sample_shape) + dist_shape


@pytest.mark.parametrize("sparse", [True, False])
def test_car_rng_fn(sparse):
    delta = 0.05  # limit for KS p-value
    n_fails = 20  # Allows the KS fails a certain number of times
    size = (100,)

    W = np.array(
        [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
    )

    tau = 2
    alpha = 0.5
    mu = np.array([1, 1, 1, 1])

    D = W.sum(axis=0)
    prec = tau * (np.diag(D) - alpha * W)
    cov = np.linalg.inv(prec)
    W = aesara.tensor.as_tensor_variable(W)
    if sparse:
        W = aesara.sparse.csr_from_dense(W)

    with pm.Model(rng_seeder=1):
        car = pm.CAR("car", mu, W, alpha, tau, size=size)
        mn = pm.MvNormal("mn", mu, cov, size=size)
        check = pm.sample_prior_predictive(n_fails, return_inferencedata=False)

    p, f = delta, n_fails
    while p <= delta and f > 0:
        car_smp, mn_smp = check["car"][f - 1, :, :], check["mn"][f - 1, :, :]
        p = min(
            st.ks_2samp(
                np.atleast_1d(car_smp[..., idx]).flatten(),
                np.atleast_1d(mn_smp[..., idx]).flatten(),
            )[1]
            for idx in range(car_smp.shape[-1])
        )
        f -= 1
    assert p > delta
