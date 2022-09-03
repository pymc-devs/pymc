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

import sys
import warnings

import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as st

from aeppl.logprob import ParameterValueError
from aesara.compile.mode import Mode
from aesara.tensor.var import TensorVariable

import pymc as pm

from pymc.aesaraf import floatX
from pymc.distributions import joint_logp, logcdf, logp
from pymc.tests.distributions.util import (
    Bool,
    Domain,
    I,
    Nat,
    NatSmall,
    R,
    Rdunif,
    Rplus,
    Rplusbig,
    Rplusdunif,
    Runif,
    Simplex,
    Unit,
    UnitSortedVector,
    Vector,
    assert_moment_is_expected,
    check_logcdf,
    check_logp,
    check_selfconsistency_discrete_logcdf,
)
from pymc.vartypes import discrete_types


def discrete_weibull_logpmf(value, q, beta):
    return floatX(
        np.log(np.power(q, np.power(value, beta)) - np.power(q, np.power(value + 1, beta)))
    )


def categorical_logpdf(value, p):
    if value >= 0 and value <= len(p):
        return floatX(np.log(np.moveaxis(p, -1, 0)[value]))
    else:
        return -inf


def invlogit(x, eps=sys.float_info.epsilon):
    return (1.0 - 2.0 * eps) / (1.0 + np.exp(-x)) + eps


def orderedlogistic_logpdf(value, eta, cutpoints):
    c = np.concatenate(([-np.inf], cutpoints, [np.inf]))
    ps = np.array([invlogit(eta - cc) - invlogit(eta - cc1) for cc, cc1 in zip(c[:-1], c[1:])])
    p = ps[value]
    return np.where(np.all(ps >= 0), np.log(p), -np.inf)


def invprobit(x):
    return (sp.erf(x / np.sqrt(2)) + 1) / 2


def orderedprobit_logpdf(value, eta, cutpoints):
    c = np.concatenate(([-np.inf], cutpoints, [np.inf]))
    ps = np.array([invprobit(eta - cc) - invprobit(eta - cc1) for cc, cc1 in zip(c[:-1], c[1:])])
    p = ps[value]
    return np.where(np.all(ps >= 0), np.log(p), -np.inf)


class TestMatchesScipy:
    def test_discrete_unif(self):
        check_logp(
            pm.DiscreteUniform,
            Rdunif,
            {"lower": -Rplusdunif, "upper": Rplusdunif},
            lambda value, lower, upper: st.randint.logpmf(value, lower, upper + 1),
            skip_paramdomain_outside_edge_test=True,
        )
        check_logcdf(
            pm.DiscreteUniform,
            Rdunif,
            {"lower": -Rplusdunif, "upper": Rplusdunif},
            lambda value, lower, upper: st.randint.logcdf(value, lower, upper + 1),
            skip_paramdomain_outside_edge_test=True,
        )
        check_selfconsistency_discrete_logcdf(
            pm.DiscreteUniform,
            Domain([-10, 0, 10], "int64"),
            {"lower": -Rplusdunif, "upper": Rplusdunif},
        )
        # Custom logp / logcdf check for invalid parameters
        invalid_dist = pm.DiscreteUniform.dist(lower=1, upper=0)
        with aesara.config.change_flags(mode=Mode("py")):
            with pytest.raises(ParameterValueError):
                logp(invalid_dist, 0.5).eval()
            with pytest.raises(ParameterValueError):
                logcdf(invalid_dist, 2).eval()

    def test_geometric(self):
        check_logp(
            pm.Geometric,
            Nat,
            {"p": Unit},
            lambda value, p: np.log(st.geom.pmf(value, p)),
        )
        check_logcdf(
            pm.Geometric,
            Nat,
            {"p": Unit},
            lambda value, p: st.geom.logcdf(value, p),
        )
        check_selfconsistency_discrete_logcdf(
            pm.Geometric,
            Nat,
            {"p": Unit},
        )

    def test_hypergeometric(self):
        def modified_scipy_hypergeom_logpmf(value, N, k, n):
            # Convert nan to -np.inf
            original_res = st.hypergeom.logpmf(value, N, k, n)
            return original_res if not np.isnan(original_res) else -np.inf

        def modified_scipy_hypergeom_logcdf(value, N, k, n):
            # Convert nan to -np.inf
            original_res = st.hypergeom.logcdf(value, N, k, n)

            # Correct for scipy bug in logcdf method (see https://github.com/scipy/scipy/issues/13280)
            if not np.isnan(original_res):
                pmfs = st.hypergeom.logpmf(np.arange(value + 1), N, k, n)
                if np.all(np.isnan(pmfs)):
                    original_res = np.nan

            return original_res if not np.isnan(original_res) else -np.inf

        check_logp(
            pm.HyperGeometric,
            Nat,
            {"N": NatSmall, "k": NatSmall, "n": NatSmall},
            modified_scipy_hypergeom_logpmf,
        )
        check_logcdf(
            pm.HyperGeometric,
            Nat,
            {"N": NatSmall, "k": NatSmall, "n": NatSmall},
            modified_scipy_hypergeom_logcdf,
        )
        check_selfconsistency_discrete_logcdf(
            pm.HyperGeometric,
            Nat,
            {"N": NatSmall, "k": NatSmall, "n": NatSmall},
        )

    @pytest.mark.xfail(
        condition=(aesara.config.floatX == "float32"),
        reason="SciPy log CDF stopped working after un-pinning NumPy version.",
    )
    def test_negative_binomial(self):
        def scipy_mu_alpha_logpmf(value, mu, alpha):
            return st.nbinom.logpmf(value, alpha, 1 - mu / (mu + alpha))

        def scipy_mu_alpha_logcdf(value, mu, alpha):
            return st.nbinom.logcdf(value, alpha, 1 - mu / (mu + alpha))

        check_logp(
            pm.NegativeBinomial,
            Nat,
            {"mu": Rplus, "alpha": Rplus},
            scipy_mu_alpha_logpmf,
        )
        check_logp(
            pm.NegativeBinomial,
            Nat,
            {"p": Unit, "n": Rplus},
            lambda value, p, n: st.nbinom.logpmf(value, n, p),
        )
        check_logcdf(
            pm.NegativeBinomial,
            Nat,
            {"mu": Rplus, "alpha": Rplus},
            scipy_mu_alpha_logcdf,
        )
        check_logcdf(
            pm.NegativeBinomial,
            Nat,
            {"p": Unit, "n": Rplus},
            lambda value, p, n: st.nbinom.logcdf(value, n, p),
        )
        check_selfconsistency_discrete_logcdf(
            pm.NegativeBinomial,
            Nat,
            {"mu": Rplus, "alpha": Rplus},
        )

    @pytest.mark.parametrize(
        "mu, p, alpha, n, expected",
        [
            (5, None, None, None, "Must specify either alpha or n."),
            (None, 0.5, None, None, "Must specify either alpha or n."),
            (None, None, None, None, "Must specify either alpha or n."),
            (5, None, 2, 2, "Can't specify both alpha and n."),
            (None, 0.5, 2, 2, "Can't specify both alpha and n."),
            (None, None, 2, 2, "Can't specify both alpha and n."),
            (None, None, 2, None, "Must specify either mu or p."),
            (None, None, None, 2, "Must specify either mu or p."),
            (5, 0.5, 2, None, "Can't specify both mu and p."),
            (5, 0.5, None, 2, "Can't specify both mu and p."),
        ],
    )
    def test_negative_binomial_init_fail(self, mu, p, alpha, n, expected):
        with pm.Model():
            with pytest.raises(ValueError, match=f"Incompatible parametrization. {expected}"):
                pm.NegativeBinomial("x", mu=mu, p=p, alpha=alpha, n=n)

    def test_binomial(self):
        check_logp(
            pm.Binomial,
            Nat,
            {"n": NatSmall, "p": Unit},
            lambda value, n, p: st.binom.logpmf(value, n, p),
        )
        check_logcdf(
            pm.Binomial,
            Nat,
            {"n": NatSmall, "p": Unit},
            lambda value, n, p: st.binom.logcdf(value, n, p),
        )
        check_selfconsistency_discrete_logcdf(
            pm.Binomial,
            Nat,
            {"n": NatSmall, "p": Unit},
        )

    def test_beta_binomial(self):
        check_logp(
            pm.BetaBinomial,
            Nat,
            {"alpha": Rplus, "beta": Rplus, "n": NatSmall},
            lambda value, alpha, beta, n: st.betabinom.logpmf(value, a=alpha, b=beta, n=n),
        )
        check_logcdf(
            pm.BetaBinomial,
            Nat,
            {"alpha": Rplus, "beta": Rplus, "n": NatSmall},
            lambda value, alpha, beta, n: st.betabinom.logcdf(value, a=alpha, b=beta, n=n),
        )
        check_selfconsistency_discrete_logcdf(
            pm.BetaBinomial,
            Nat,
            {"alpha": Rplus, "beta": Rplus, "n": NatSmall},
        )

    def test_bernoulli(self):
        check_logp(
            pm.Bernoulli,
            Bool,
            {"p": Unit},
            lambda value, p: st.bernoulli.logpmf(value, p),
        )
        check_logp(
            pm.Bernoulli,
            Bool,
            {"logit_p": R},
            lambda value, logit_p: st.bernoulli.logpmf(value, sp.expit(logit_p)),
        )
        check_logcdf(
            pm.Bernoulli,
            Bool,
            {"p": Unit},
            lambda value, p: st.bernoulli.logcdf(value, p),
        )
        check_logcdf(
            pm.Bernoulli,
            Bool,
            {"logit_p": R},
            lambda value, logit_p: st.bernoulli.logcdf(value, sp.expit(logit_p)),
        )
        check_selfconsistency_discrete_logcdf(
            pm.Bernoulli,
            Bool,
            {"p": Unit},
        )

    def test_bernoulli_wrong_arguments(self):
        m = pm.Model()

        msg = "Incompatible parametrization. Can't specify both p and logit_p"
        with m:
            with pytest.raises(ValueError, match=msg):
                pm.Bernoulli("x", p=0.5, logit_p=0)

        msg = "Incompatible parametrization. Must specify either p or logit_p"
        with m:
            with pytest.raises(ValueError, match=msg):
                pm.Bernoulli("x")

    def test_discrete_weibull(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "divide by zero encountered in log", RuntimeWarning)
            check_logp(
                pm.DiscreteWeibull,
                Nat,
                {"q": Unit, "beta": NatSmall},
                discrete_weibull_logpmf,
            )
        check_selfconsistency_discrete_logcdf(
            pm.DiscreteWeibull,
            Nat,
            {"q": Unit, "beta": NatSmall},
        )

    def test_poisson(self):
        check_logp(
            pm.Poisson,
            Nat,
            {"mu": Rplus},
            lambda value, mu: st.poisson.logpmf(value, mu),
        )
        check_logcdf(
            pm.Poisson,
            Nat,
            {"mu": Rplus},
            lambda value, mu: st.poisson.logcdf(value, mu),
        )
        check_selfconsistency_discrete_logcdf(
            pm.Poisson,
            Nat,
            {"mu": Rplus},
        )

    def test_diracdeltadist(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "divide by zero encountered in log", RuntimeWarning)
            check_logp(pm.DiracDelta, I, {"c": I}, lambda value, c: np.log(c == value))
            check_logcdf(pm.DiracDelta, I, {"c": I}, lambda value, c: np.log(value >= c))

    def test_zeroinflatedpoisson(self):
        def logp_fn(value, psi, mu):
            if value == 0:
                return np.log((1 - psi) * st.poisson.pmf(0, mu))
            else:
                return np.log(psi * st.poisson.pmf(value, mu))

        def logcdf_fn(value, psi, mu):
            return np.log((1 - psi) + psi * st.poisson.cdf(value, mu))

        check_logp(
            pm.ZeroInflatedPoisson,
            Nat,
            {"psi": Unit, "mu": Rplus},
            logp_fn,
        )

        check_logcdf(
            pm.ZeroInflatedPoisson,
            Nat,
            {"psi": Unit, "mu": Rplus},
            logcdf_fn,
        )

        check_selfconsistency_discrete_logcdf(
            pm.ZeroInflatedPoisson,
            Nat,
            {"mu": Rplus, "psi": Unit},
        )

    def test_zeroinflatednegativebinomial(self):
        def logp_fn(value, psi, mu, alpha):
            n, p = pm.NegativeBinomial.get_n_p(mu=mu, alpha=alpha)
            if value == 0:
                return np.log((1 - psi) * st.nbinom.pmf(0, n, p))
            else:
                return np.log(psi * st.nbinom.pmf(value, n, p))

        def logcdf_fn(value, psi, mu, alpha):
            n, p = pm.NegativeBinomial.get_n_p(mu=mu, alpha=alpha)
            return np.log((1 - psi) + psi * st.nbinom.cdf(value, n, p))

        check_logp(
            pm.ZeroInflatedNegativeBinomial,
            Nat,
            {"psi": Unit, "mu": Rplusbig, "alpha": Rplusbig},
            logp_fn,
        )

        check_logp(
            pm.ZeroInflatedNegativeBinomial,
            Nat,
            {"psi": Unit, "p": Unit, "n": NatSmall},
            lambda value, psi, p, n: np.log((1 - psi) * st.nbinom.pmf(0, n, p))
            if value == 0
            else np.log(psi * st.nbinom.pmf(value, n, p)),
        )

        check_logcdf(
            pm.ZeroInflatedNegativeBinomial,
            Nat,
            {"psi": Unit, "mu": Rplusbig, "alpha": Rplusbig},
            logcdf_fn,
        )

        check_logcdf(
            pm.ZeroInflatedNegativeBinomial,
            Nat,
            {"psi": Unit, "p": Unit, "n": NatSmall},
            lambda value, psi, p, n: np.log((1 - psi) + psi * st.nbinom.cdf(value, n, p)),
        )

        check_selfconsistency_discrete_logcdf(
            pm.ZeroInflatedNegativeBinomial,
            Nat,
            {"psi": Unit, "mu": Rplusbig, "alpha": Rplusbig},
        )

    def test_zeroinflatedbinomial(self):
        def logp_fn(value, psi, n, p):
            if value == 0:
                return np.log((1 - psi) * st.binom.pmf(0, n, p))
            else:
                return np.log(psi * st.binom.pmf(value, n, p))

        def logcdf_fn(value, psi, n, p):
            return np.log((1 - psi) + psi * st.binom.cdf(value, n, p))

        check_logp(
            pm.ZeroInflatedBinomial,
            Nat,
            {"psi": Unit, "n": NatSmall, "p": Unit},
            logp_fn,
        )

        check_logcdf(
            pm.ZeroInflatedBinomial,
            Nat,
            {"psi": Unit, "n": NatSmall, "p": Unit},
            logcdf_fn,
        )

        check_selfconsistency_discrete_logcdf(
            pm.ZeroInflatedBinomial,
            Nat,
            {"n": NatSmall, "p": Unit, "psi": Unit},
        )

    @aesara.config.change_flags(compute_test_value="raise")
    def test_categorical_bounds(self):
        with pm.Model():
            x = pm.Categorical("x", p=np.array([0.2, 0.3, 0.5]))
            assert np.isinf(logp(x, -1).eval())
            assert np.isinf(logp(x, 3).eval())

    @aesara.config.change_flags(compute_test_value="raise")
    @pytest.mark.parametrize(
        "p",
        [
            np.array([-0.2, 0.3, 0.5]),
            # A model where p sums to 1 but contains negative values
            np.array([-0.2, 0.7, 0.5]),
            # Hard edge case from #2082
            # Early automatic normalization of p's sum would hide the negative
            # entries if there is a single or pair number of negative values
            # and the rest are zero
            np.array([-1, -1, 0, 0]),
        ],
    )
    def test_categorical_negative_p(self, p):
        with pytest.raises(ValueError, match=f"{p}"):
            with pm.Model():
                x = pm.Categorical("x", p=p)

    def test_categorical_negative_p_symbolic(self):
        with pytest.raises(ParameterValueError):
            value = np.array([[1, 1, 1]])
            invalid_dist = pm.Categorical.dist(p=at.as_tensor_variable([-1, 0.5, 0.5]))
            pm.logp(invalid_dist, value).eval()

    def test_categorical_p_not_normalized_symbolic(self):
        with pytest.raises(ParameterValueError):
            value = np.array([[1, 1, 1]])
            invalid_dist = pm.Categorical.dist(p=at.as_tensor_variable([2, 2, 2]))
            pm.logp(invalid_dist, value).eval()

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_categorical(self, n):
        check_logp(
            pm.Categorical,
            Domain(range(n), dtype="int64", edges=(0, n)),
            {"p": Simplex(n)},
            lambda value, p: categorical_logpdf(value, p),
        )

    def test_categorical_p_not_normalized(self):
        # test UserWarning is raised for p vals that sum to more than 1
        # and normaliation is triggered
        with pytest.warns(UserWarning, match="[5]"):
            with pm.Model() as m:
                x = pm.Categorical("x", p=[1, 1, 1, 1, 1])
        assert np.isclose(m.x.owner.inputs[3].sum().eval(), 1.0)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_orderedlogistic(self, n):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in log", RuntimeWarning)
            warnings.filterwarnings("ignore", "divide by zero encountered in log", RuntimeWarning)
            check_logp(
                pm.OrderedLogistic,
                Domain(range(n), dtype="int64", edges=(None, None)),
                {"eta": R, "cutpoints": Vector(R, n - 1)},
                lambda value, eta, cutpoints: orderedlogistic_logpdf(value, eta, cutpoints),
            )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_orderedprobit(self, n):
        check_logp(
            pm.OrderedProbit,
            Domain(range(n), dtype="int64", edges=(None, None)),
            {"eta": Runif, "cutpoints": UnitSortedVector(n - 1)},
            lambda value, eta, cutpoints: orderedprobit_logpdf(value, eta, cutpoints),
        )


# TODO: Is this test working as expected / still relevant?
@pytest.mark.parametrize("shape", [tuple(), (1,), (3, 1), (3, 2)], ids=str)
def test_orderedlogistic_dimensions(shape):
    # Test for issue #3535
    loge = np.log10(np.exp(1))
    size = 7
    p = np.ones(shape + (10,)) / 10
    cutpoints = np.tile(sp.logit(np.linspace(0, 1, 11)[1:-1]), shape + (1,))
    obs = np.random.randint(0, 2, size=(size,) + shape)
    with pm.Model():
        ol = pm.OrderedLogistic(
            "ol",
            eta=np.zeros(shape),
            cutpoints=cutpoints,
            observed=obs,
        )
        c = pm.Categorical(
            "c",
            p=p,
            observed=obs,
        )
    ologp = joint_logp(ol, np.ones_like(obs), sum=True).eval() * loge
    clogp = joint_logp(c, np.ones_like(obs), sum=True).eval() * loge
    expected = -np.prod((size,) + shape)

    assert c.owner.inputs[3].ndim == (len(shape) + 1)
    assert np.allclose(clogp, expected)
    assert ol.owner.inputs[3].ndim == (len(shape) + 1)
    assert np.allclose(ologp, expected)


def test_ordered_multinomial_probs():
    with pm.Model() as m:
        pm.OrderedMultinomial("om_p", n=1000, cutpoints=np.array([-2, 0, 2]), eta=0)
        pm.OrderedMultinomial(
            "om_no_p", n=1000, cutpoints=np.array([-2, 0, 2]), eta=0, compute_p=False
        )
    assert len(m.deterministics) == 1

    x = pm.OrderedMultinomial.dist(n=1000, cutpoints=np.array([-2, 0, 2]), eta=0)
    assert isinstance(x, TensorVariable)


def test_ordered_logistic_probs():
    with pm.Model() as m:
        pm.OrderedLogistic("ol_p", cutpoints=np.array([-2, 0, 2]), eta=0)
        pm.OrderedLogistic("ol_no_p", cutpoints=np.array([-2, 0, 2]), eta=0, compute_p=False)
    assert len(m.deterministics) == 1

    x = pm.OrderedLogistic.dist(cutpoints=np.array([-2, 0, 2]), eta=0)
    assert isinstance(x, TensorVariable)


def test_ordered_probit_probs():
    with pm.Model() as m:
        pm.OrderedProbit("op_p", cutpoints=np.array([-2, 0, 2]), eta=0, sigma=1)
        pm.OrderedProbit("op_no_p", cutpoints=np.array([-2, 0, 2]), eta=0, sigma=1, compute_p=False)
    assert len(m.deterministics) == 1

    x = pm.OrderedProbit.dist(cutpoints=np.array([-2, 0, 2]), eta=0, sigma=1)
    assert isinstance(x, TensorVariable)


@pytest.mark.parametrize(
    "dist, non_psi_args",
    [
        (pm.ZeroInflatedPoisson.dist, (2,)),
        (pm.ZeroInflatedBinomial.dist, (2, 0.5)),
        (pm.ZeroInflatedNegativeBinomial.dist, (2, 2)),
    ],
)
def test_zero_inflated_dists_dtype_and_broadcast(dist, non_psi_args):
    x = dist([0.5, 0.5, 0.5], *non_psi_args)
    assert x.dtype in discrete_types
    assert x.eval().shape == (3,)


def test_constantdist_deprecated():
    with pytest.warns(FutureWarning, match="DiracDelta"):
        with pm.Model() as m:
            x = pm.Constant("x", c=1)
            assert isinstance(x.owner.op, pm.distributions.discrete.DiracDeltaRV)


class TestMoments:
    @pytest.mark.parametrize(
        "p, size, expected",
        [
            (0.3, None, 0),
            (0.9, 5, np.ones(5)),
            (np.linspace(0, 1, 4), None, [0, 0, 1, 1]),
            (np.linspace(0, 1, 4), (2, 4), np.full((2, 4), [0, 0, 1, 1])),
        ],
    )
    def test_bernoulli_moment(self, p, size, expected):
        with pm.Model() as model:
            pm.Bernoulli("x", p=p, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "n, alpha, beta, size, expected",
        [
            (10, 1, 1, None, 5),
            (10, 1, 1, 5, np.full(5, 5)),
            (10, 1, np.arange(1, 6), None, np.round(10 / np.arange(2, 7))),
            (10, 1, np.arange(1, 6), (2, 5), np.full((2, 5), np.round(10 / np.arange(2, 7)))),
        ],
    )
    def test_beta_binomial_moment(self, alpha, beta, n, size, expected):
        with pm.Model() as model:
            pm.BetaBinomial("x", alpha=alpha, beta=beta, n=n, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "n, p, size, expected",
        [
            (7, 0.7, None, 5),
            (7, 0.3, 5, np.full(5, 2)),
            (10, np.arange(1, 6) / 10, None, np.arange(1, 6)),
            (10, np.arange(1, 6) / 10, (2, 5), np.full((2, 5), np.arange(1, 6))),
        ],
    )
    def test_binomial_moment(self, n, p, size, expected):
        with pm.Model() as model:
            pm.Binomial("x", n=n, p=p, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, size, expected",
        [
            (2.7, None, 2),
            (2.3, 5, np.full(5, 2)),
            (np.arange(1, 5), None, np.arange(1, 5)),
            (np.arange(1, 5), (2, 4), np.full((2, 4), np.arange(1, 5))),
        ],
    )
    def test_poisson_moment(self, mu, size, expected):
        with pm.Model() as model:
            pm.Poisson("x", mu=mu, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "n, p, size, expected",
        [
            (10, 0.7, None, 4),
            (10, 0.7, 5, np.full(5, 4)),
            (np.full(3, 10), np.arange(1, 4) / 10, None, np.array([90, 40, 23])),
            (
                10,
                np.arange(1, 4) / 10,
                (2, 3),
                np.full((2, 3), np.array([90, 40, 23])),
            ),
        ],
    )
    def test_negative_binomial_moment(self, n, p, size, expected):
        with pm.Model() as model:
            pm.NegativeBinomial("x", n=n, p=p, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "c, size, expected",
        [
            (1, None, 1),
            (1, 5, np.full(5, 1)),
            (np.arange(1, 6), None, np.arange(1, 6)),
        ],
    )
    def test_diracdelta_moment(self, c, size, expected):
        with pm.Model() as model:
            pm.DiracDelta("x", c=c, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "psi, mu, size, expected",
        [
            (0.9, 3.0, None, 3),
            (0.8, 2.9, 5, np.full(5, 2)),
            (0.2, np.arange(1, 5) * 5, None, np.arange(1, 5)),
            (0.2, np.arange(1, 5) * 5, (2, 4), np.full((2, 4), np.arange(1, 5))),
        ],
    )
    def test_zero_inflated_poisson_moment(self, psi, mu, size, expected):
        with pm.Model() as model:
            pm.ZeroInflatedPoisson("x", psi=psi, mu=mu, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "psi, n, p, size, expected",
        [
            (0.8, 7, 0.7, None, 4),
            (0.8, 7, 0.3, 5, np.full(5, 2)),
            (0.4, 25, np.arange(1, 6) / 10, None, np.arange(1, 6)),
            (
                0.4,
                25,
                np.arange(1, 6) / 10,
                (2, 5),
                np.full((2, 5), np.arange(1, 6)),
            ),
        ],
    )
    def test_zero_inflated_binomial_moment(self, psi, n, p, size, expected):
        with pm.Model() as model:
            pm.ZeroInflatedBinomial("x", psi=psi, n=n, p=p, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "p, size, expected",
        [
            (0.5, None, 2),
            (0.2, 5, 5 * np.ones(5)),
            (np.linspace(0.25, 1, 4), None, [4, 2, 1, 1]),
            (np.linspace(0.25, 1, 4), (2, 4), np.full((2, 4), [4, 2, 1, 1])),
        ],
    )
    def test_geometric_moment(self, p, size, expected):
        with pm.Model() as model:
            pm.Geometric("x", p=p, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "N, k, n, size, expected",
        [
            (50, 10, 20, None, 4),
            (50, 10, 23, 5, np.full(5, 5)),
            (50, 10, np.arange(23, 28), None, np.full(5, 5)),
            (
                50,
                10,
                np.arange(18, 23),
                (2, 5),
                np.full((2, 5), 4),
            ),
        ],
    )
    def test_hyper_geometric_moment(self, N, k, n, size, expected):
        with pm.Model() as model:
            pm.HyperGeometric("x", N=N, k=k, n=n, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "lower, upper, size, expected",
        [
            (1, 5, None, 3),
            (1, 5, 5, np.full(5, 3)),
            (1, np.arange(5, 22, 4), None, np.arange(3, 13, 2)),
            (
                1,
                np.arange(5, 22, 4),
                (2, 5),
                np.full((2, 5), np.arange(3, 13, 2)),
            ),
        ],
    )
    def test_discrete_uniform_moment(self, lower, upper, size, expected):
        with pm.Model() as model:
            pm.DiscreteUniform("x", lower=lower, upper=upper, size=size)
            assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "q, beta, size, expected",
        [
            (0.5, 0.5, None, 0),
            (0.6, 0.1, 5, (20,) * 5),
            (np.linspace(0.25, 0.99, 4), 0.42, None, [0, 0, 6, 23862]),
            (
                np.linspace(0.5, 0.99, 3),
                [[1, 1.25, 1.75], [1.25, 0.75, 0.5]],
                None,
                [[0, 0, 10], [0, 2, 4755]],
            ),
        ],
    )
    def test_discrete_weibull_moment(self, q, beta, size, expected):
        with pm.Model() as model:
            pm.DiscreteWeibull("x", q=q, beta=beta, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "p, size, expected",
        [
            (np.array([0.1, 0.3, 0.6]), None, 2),
            (np.array([0.6, 0.1, 0.3]), 5, np.full(5, 0)),
            (np.full((2, 3), np.array([0.6, 0.1, 0.3])), None, [0, 0]),
            (
                np.full((2, 3), np.array([0.1, 0.3, 0.6])),
                (3, 2),
                np.full((3, 2), [2, 2]),
            ),
        ],
    )
    def test_categorical_moment(self, p, size, expected):
        with pm.Model() as model:
            pm.Categorical("x", p=p, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "psi, mu, alpha, size, expected",
        [
            (0.2, 10, 3, None, 2),
            (0.2, 10, 4, 5, np.full(5, 2)),
            (
                0.4,
                np.arange(1, 5),
                np.arange(2, 6),
                None,
                np.array([0, 1, 1, 2] if aesara.config.floatX == "float64" else [0, 0, 1, 1]),
            ),
            (
                np.linspace(0.2, 0.6, 3),
                np.arange(1, 10, 4),
                np.arange(1, 4),
                (2, 3),
                np.full((2, 3), np.array([0, 2, 5])),
            ),
        ],
    )
    def test_zero_inflated_negative_binomial_moment(self, psi, mu, alpha, size, expected):
        with pm.Model() as model:
            pm.ZeroInflatedNegativeBinomial("x", psi=psi, mu=mu, alpha=alpha, size=size)
        assert_moment_is_expected(model, expected)
