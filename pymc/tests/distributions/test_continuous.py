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

import functools as ft

import aesara
import aesara.tensor as at
import numpy as np
import numpy.testing as npt
import pytest
import scipy.special as sp
import scipy.stats as st

from aeppl.logprob import ParameterValueError
from aesara.compile.mode import Mode

import pymc as pm

from pymc.aesaraf import floatX
from pymc.distributions import logcdf, logp
from pymc.distributions.continuous import get_tau_sigma
from pymc.tests.distributions.util import (
    Circ,
    Domain,
    R,
    Rplus,
    Rplusbig,
    Rplusunif,
    Runif,
    Unit,
    assert_moment_is_expected,
    check_logcdf,
    check_logp,
)
from pymc.tests.helpers import select_by_precision

try:
    from polyagamma import polyagamma_cdf, polyagamma_pdf

    _polyagamma_not_installed = False
except ImportError:  # pragma: no cover

    _polyagamma_not_installed = True

    def polyagamma_pdf(*args, **kwargs):
        raise RuntimeError("polyagamma package is not installed!")

    def polyagamma_cdf(*args, **kwargs):
        raise RuntimeError("polyagamma package is not installed!")


class TestBoundedContinuous:
    def get_dist_params_and_interval_bounds(self, model, rv_name):
        interval_rv = model.named_vars[f"{rv_name}_interval__"]
        rv = model.named_vars[rv_name]
        dist_params = rv.owner.inputs
        lower_interval, upper_interval = interval_rv.tag.transform.args_fn(*rv.owner.inputs)
        return (
            dist_params,
            lower_interval,
            upper_interval,
        )

    def test_upper_bounded(self):
        bounded_rv_name = "lower_bounded"
        with pm.Model() as model:
            pm.TruncatedNormal(bounded_rv_name, mu=1, sigma=2, lower=None, upper=3)
        (
            (_, _, _, _, _, lower, upper),
            lower_interval,
            upper_interval,
        ) = self.get_dist_params_and_interval_bounds(model, bounded_rv_name)
        assert lower.value == -np.inf
        assert upper.value == 3
        assert lower_interval is None
        assert upper_interval.value == 3

    def test_lower_bounded(self):
        bounded_rv_name = "upper_bounded"
        with pm.Model() as model:
            pm.TruncatedNormal(bounded_rv_name, mu=1, sigma=2, lower=-2, upper=None)
        (
            (_, _, _, _, _, lower, upper),
            lower_interval,
            upper_interval,
        ) = self.get_dist_params_and_interval_bounds(model, bounded_rv_name)
        assert lower.value == -2
        assert upper.value == np.inf
        assert lower_interval.value == -2
        assert upper_interval is None

    def test_lower_bounded_vector(self):
        bounded_rv_name = "upper_bounded"
        with pm.Model() as model:
            pm.TruncatedNormal(
                bounded_rv_name,
                mu=np.array([1, 1]),
                sigma=np.array([2, 3]),
                lower=np.array([-1.0, 0]),
                upper=None,
            )
        (
            (_, _, _, _, _, lower, upper),
            lower_interval,
            upper_interval,
        ) = self.get_dist_params_and_interval_bounds(model, bounded_rv_name)

        assert np.array_equal(lower.value, [-1, 0])
        assert upper.value == np.inf
        assert np.array_equal(lower_interval.value, [-1, 0])
        assert upper_interval is None

    def test_lower_bounded_broadcasted(self):
        bounded_rv_name = "upper_bounded"
        with pm.Model() as model:
            pm.TruncatedNormal(
                bounded_rv_name,
                mu=np.array([1, 1]),
                sigma=np.array([2, 3]),
                lower=-1,
                upper=np.array([np.inf, np.inf]),
            )
        (
            (_, _, _, _, _, lower, upper),
            lower_interval,
            upper_interval,
        ) = self.get_dist_params_and_interval_bounds(model, bounded_rv_name)

        assert lower.value == -1
        assert np.array_equal(upper.value, [np.inf, np.inf])
        assert lower_interval.value == -1
        assert upper_interval is None


def laplace_asymmetric_logpdf(value, kappa, b, mu):
    kapinv = 1 / kappa
    value = value - mu
    lPx = value * b * np.where(value >= 0, -kappa, kapinv)
    lPx += np.log(b / (kappa + kapinv))
    return lPx


def beta_mu_sigma(value, mu, sigma):
    kappa = mu * (1 - mu) / sigma**2 - 1
    if kappa > 0:
        return st.beta.logpdf(value, mu * kappa, (1 - mu) * kappa)
    else:
        return -np.inf


class TestMatchesScipy:
    def test_uniform(self):
        check_logp(
            pm.Uniform,
            Runif,
            {"lower": -Rplusunif, "upper": Rplusunif},
            lambda value, lower, upper: st.uniform.logpdf(value, lower, upper - lower),
            skip_paramdomain_outside_edge_test=True,
        )
        check_logcdf(
            pm.Uniform,
            Runif,
            {"lower": -Rplusunif, "upper": Rplusunif},
            lambda value, lower, upper: st.uniform.logcdf(value, lower, upper - lower),
            skip_paramdomain_outside_edge_test=True,
        )
        # Custom logp / logcdf check for invalid parameters
        invalid_dist = pm.Uniform.dist(lower=1, upper=0)
        with aesara.config.change_flags(mode=Mode("py")):
            assert logp(invalid_dist, np.array(0.5)).eval() == -np.inf
            assert logcdf(invalid_dist, np.array(2.0)).eval() == -np.inf

    def test_triangular(self):
        check_logp(
            pm.Triangular,
            Runif,
            {"lower": -Rplusunif, "c": Runif, "upper": Rplusunif},
            lambda value, c, lower, upper: st.triang.logpdf(value, c - lower, lower, upper - lower),
            skip_paramdomain_outside_edge_test=True,
        )
        check_logcdf(
            pm.Triangular,
            Runif,
            {"lower": -Rplusunif, "c": Runif, "upper": Rplusunif},
            lambda value, c, lower, upper: st.triang.logcdf(value, c - lower, lower, upper - lower),
            skip_paramdomain_outside_edge_test=True,
        )

        # Custom logp/logcdf check for values outside of domain
        valid_dist = pm.Triangular.dist(lower=0, upper=1, c=0.9, size=2)
        with aesara.config.change_flags(mode=Mode("py")):
            assert np.all(logp(valid_dist, np.array([-1, 2])).eval() == -np.inf)
            assert np.all(logcdf(valid_dist, np.array([-1, 2])).eval() == [-np.inf, 0])

        # Custom logcdf check for invalid parameters.
        # Invalid logp checks for triangular are being done in aeppl
        invalid_dist = pm.Triangular.dist(lower=1, upper=0, c=0.1)
        with aesara.config.change_flags(mode=Mode("py")):
            with pytest.raises(ParameterValueError):
                logcdf(invalid_dist, 2).eval()

        invalid_dist = pm.Triangular.dist(lower=0, upper=1, c=2.0)
        with aesara.config.change_flags(mode=Mode("py")):
            with pytest.raises(ParameterValueError):
                logcdf(invalid_dist, 2).eval()

    @pytest.mark.skipif(
        condition=_polyagamma_not_installed,
        reason="`polyagamma package is not available/installed.",
    )
    def test_polyagamma(self):
        check_logp(
            pm.PolyaGamma,
            Rplus,
            {"h": Rplus, "z": R},
            lambda value, h, z: polyagamma_pdf(value, h, z, return_log=True),
            decimal=select_by_precision(float64=6, float32=-1),
        )
        check_logcdf(
            pm.PolyaGamma,
            Rplus,
            {"h": Rplus, "z": R},
            lambda value, h, z: polyagamma_cdf(value, h, z, return_log=True),
            decimal=select_by_precision(float64=6, float32=-1),
        )

    def test_flat(self):
        check_logp(pm.Flat, R, {}, lambda value: 0)
        with pm.Model():
            x = pm.Flat("a")
        check_logcdf(pm.Flat, R, {}, lambda value: np.log(0.5))
        # Check infinite cases individually.
        assert 0.0 == logcdf(pm.Flat.dist(), np.inf).eval()
        assert -np.inf == logcdf(pm.Flat.dist(), -np.inf).eval()

    def test_half_flat(self):
        check_logp(pm.HalfFlat, Rplus, {}, lambda value: 0)
        with pm.Model():
            x = pm.HalfFlat("a", size=2)
        check_logcdf(pm.HalfFlat, Rplus, {}, lambda value: -np.inf)
        # Check infinite cases individually.
        assert 0.0 == logcdf(pm.HalfFlat.dist(), np.inf).eval()
        assert -np.inf == logcdf(pm.HalfFlat.dist(), -np.inf).eval()

    def test_normal(self):
        check_logp(
            pm.Normal,
            R,
            {"mu": R, "sigma": Rplus},
            lambda value, mu, sigma: st.norm.logpdf(value, mu, sigma),
            decimal=select_by_precision(float64=6, float32=1),
        )
        check_logcdf(
            pm.Normal,
            R,
            {"mu": R, "sigma": Rplus},
            lambda value, mu, sigma: st.norm.logcdf(value, mu, sigma),
            decimal=select_by_precision(float64=6, float32=1),
        )

    def test_half_normal(self):
        check_logp(
            pm.HalfNormal,
            Rplus,
            {"sigma": Rplus},
            lambda value, sigma: st.halfnorm.logpdf(value, scale=sigma),
            decimal=select_by_precision(float64=6, float32=-1),
        )
        check_logcdf(
            pm.HalfNormal,
            Rplus,
            {"sigma": Rplus},
            lambda value, sigma: st.halfnorm.logcdf(value, scale=sigma),
        )

    def test_chisquared_logp(self):
        check_logp(
            pm.ChiSquared,
            Rplus,
            {"nu": Rplus},
            lambda value, nu: st.chi2.logpdf(value, df=nu),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_chisquared_logcdf(self):
        check_logcdf(
            pm.ChiSquared,
            Rplus,
            {"nu": Rplus},
            lambda value, nu: st.chi2.logcdf(value, df=nu),
        )

    def test_wald_logp(self):
        check_logp(
            pm.Wald,
            Rplus,
            {"mu": Rplus, "alpha": Rplus},
            lambda value, mu, alpha: st.invgauss.logpdf(value, mu=mu, loc=alpha),
            decimal=select_by_precision(float64=6, float32=1),
        )

    def test_wald_logcdf(self):
        check_logcdf(
            pm.Wald,
            Rplus,
            {"mu": Rplus, "alpha": Rplus},
            lambda value, mu, alpha: st.invgauss.logcdf(value, mu=mu, loc=alpha),
        )

    @pytest.mark.parametrize(
        "value,mu,lam,phi,alpha,logp",
        [
            (0.5, 0.001, 0.5, None, 0.0, -124500.7257914),
            (1.0, 0.5, 0.001, None, 0.0, -4.3733162),
            (2.0, 1.0, None, None, 0.0, -2.2086593),
            (5.0, 2.0, 2.5, None, 0.0, -3.4374500),
            (7.5, 5.0, None, 1.0, 0.0, -3.2199074),
            (15.0, 10.0, None, 0.75, 0.0, -4.0360623),
            (50.0, 15.0, None, 0.66666, 0.0, -6.1801249),
            (0.5, 0.001, 0.5, None, 0.0, -124500.7257914),
            (1.0, 0.5, 0.001, None, 0.5, -3.3330954),
            (2.0, 1.0, None, None, 1.0, -0.9189385),
            (5.0, 2.0, 2.5, None, 2.0, -2.2128783),
            (7.5, 5.0, None, 1.0, 2.5, -2.5283764),
            (15.0, 10.0, None, 0.75, 5.0, -3.3653647),
            (50.0, 15.0, None, 0.666666, 10.0, -5.6481874),
        ],
    )
    def test_wald_logp_custom_points(self, value, mu, lam, phi, alpha, logp):
        # Log probabilities calculated using the dIG function from the R package gamlss.
        # See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or
        # http://www.gamlss.org/.
        with pm.Model() as model:
            pm.Wald("wald", mu=mu, lam=lam, phi=phi, alpha=alpha, transform=None)
        pt = {"wald": value}
        decimals = select_by_precision(float64=6, float32=1)
        npt.assert_almost_equal(model.compile_logp()(pt), logp, decimal=decimals, err_msg=str(pt))

    def test_beta_logp(self):
        check_logp(
            pm.Beta,
            Unit,
            {"alpha": Rplus, "beta": Rplus},
            lambda value, alpha, beta: st.beta.logpdf(value, alpha, beta),
        )
        check_logp(
            pm.Beta,
            Unit,
            {"mu": Unit, "sigma": Rplus},
            beta_mu_sigma,
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_beta_logcdf(self):
        check_logcdf(
            pm.Beta,
            Unit,
            {"alpha": Rplus, "beta": Rplus},
            lambda value, alpha, beta: st.beta.logcdf(value, alpha, beta),
        )

    def test_kumaraswamy(self):
        # Scipy does not have a built-in Kumaraswamy
        def scipy_log_pdf(value, a, b):
            return (
                np.log(a) + np.log(b) + (a - 1) * np.log(value) + (b - 1) * np.log(1 - value**a)
            )

        def scipy_log_cdf(value, a, b):
            return pm.math.log1mexp_numpy(b * np.log1p(-(value**a)), negative_input=True)

        check_logp(
            pm.Kumaraswamy,
            Unit,
            {"a": Rplus, "b": Rplus},
            scipy_log_pdf,
        )
        check_logcdf(
            pm.Kumaraswamy,
            Unit,
            {"a": Rplus, "b": Rplus},
            scipy_log_cdf,
        )

    def test_exponential(self):
        check_logp(
            pm.Exponential,
            Rplus,
            {"lam": Rplus},
            lambda value, lam: st.expon.logpdf(value, 0, 1 / lam),
        )
        check_logcdf(
            pm.Exponential,
            Rplus,
            {"lam": Rplus},
            lambda value, lam: st.expon.logcdf(value, 0, 1 / lam),
        )

    def test_laplace(self):
        check_logp(
            pm.Laplace,
            R,
            {"mu": R, "b": Rplus},
            lambda value, mu, b: st.laplace.logpdf(value, mu, b),
        )
        check_logcdf(
            pm.Laplace,
            R,
            {"mu": R, "b": Rplus},
            lambda value, mu, b: st.laplace.logcdf(value, mu, b),
        )

    def test_laplace_asymmetric(self):
        check_logp(
            pm.AsymmetricLaplace,
            R,
            {"b": Rplus, "kappa": Rplus, "mu": R},
            laplace_asymmetric_logpdf,
            decimal=select_by_precision(float64=6, float32=2),
        )

    def test_lognormal(self):
        check_logp(
            pm.LogNormal,
            Rplus,
            {"mu": R, "tau": Rplusbig},
            lambda value, mu, tau: floatX(st.lognorm.logpdf(value, tau**-0.5, 0, np.exp(mu))),
        )
        check_logp(
            pm.LogNormal,
            Rplus,
            {"mu": R, "sigma": Rplusbig},
            lambda value, mu, sigma: floatX(st.lognorm.logpdf(value, sigma, 0, np.exp(mu))),
        )
        check_logcdf(
            pm.LogNormal,
            Rplus,
            {"mu": R, "tau": Rplusbig},
            lambda value, mu, tau: st.lognorm.logcdf(value, tau**-0.5, 0, np.exp(mu)),
        )
        check_logcdf(
            pm.LogNormal,
            Rplus,
            {"mu": R, "sigma": Rplusbig},
            lambda value, mu, sigma: st.lognorm.logcdf(value, sigma, 0, np.exp(mu)),
        )

    def test_studentt_logp(self):
        check_logp(
            pm.StudentT,
            R,
            {"nu": Rplus, "mu": R, "lam": Rplus},
            lambda value, nu, mu, lam: st.t.logpdf(value, nu, mu, lam**-0.5),
        )
        check_logp(
            pm.StudentT,
            R,
            {"nu": Rplus, "mu": R, "sigma": Rplus},
            lambda value, nu, mu, sigma: st.t.logpdf(value, nu, mu, sigma),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_studentt_logcdf(self):
        check_logcdf(
            pm.StudentT,
            R,
            {"nu": Rplus, "mu": R, "lam": Rplus},
            lambda value, nu, mu, lam: st.t.logcdf(value, nu, mu, lam**-0.5),
        )
        check_logcdf(
            pm.StudentT,
            R,
            {"nu": Rplus, "mu": R, "sigma": Rplus},
            lambda value, nu, mu, sigma: st.t.logcdf(value, nu, mu, sigma),
        )

    def test_cauchy(self):
        check_logp(
            pm.Cauchy,
            R,
            {"alpha": R, "beta": Rplusbig},
            lambda value, alpha, beta: st.cauchy.logpdf(value, alpha, beta),
        )
        check_logcdf(
            pm.Cauchy,
            R,
            {"alpha": R, "beta": Rplusbig},
            lambda value, alpha, beta: st.cauchy.logcdf(value, alpha, beta),
        )

    def test_half_cauchy(self):
        check_logp(
            pm.HalfCauchy,
            Rplus,
            {"beta": Rplusbig},
            lambda value, beta: st.halfcauchy.logpdf(value, scale=beta),
        )
        check_logcdf(
            pm.HalfCauchy,
            Rplus,
            {"beta": Rplusbig},
            lambda value, beta: st.halfcauchy.logcdf(value, scale=beta),
        )

    def test_gamma_logp(self):
        check_logp(
            pm.Gamma,
            Rplus,
            {"alpha": Rplusbig, "beta": Rplusbig},
            lambda value, alpha, beta: st.gamma.logpdf(value, alpha, scale=1.0 / beta),
        )

        def test_fun(value, mu, sigma):
            return st.gamma.logpdf(value, mu**2 / sigma**2, scale=1.0 / (mu / sigma**2))

        check_logp(
            pm.Gamma,
            Rplus,
            {"mu": Rplusbig, "sigma": Rplusbig},
            test_fun,
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_gamma_logcdf(self):
        check_logcdf(
            pm.Gamma,
            Rplus,
            {"alpha": Rplusbig, "beta": Rplusbig},
            lambda value, alpha, beta: st.gamma.logcdf(value, alpha, scale=1.0 / beta),
        )

    def test_inverse_gamma_logp(self):
        check_logp(
            pm.InverseGamma,
            Rplus,
            {"alpha": Rplus, "beta": Rplus},
            lambda value, alpha, beta: st.invgamma.logpdf(value, alpha, scale=beta),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_inverse_gamma_logcdf(self):
        check_logcdf(
            pm.InverseGamma,
            Rplus,
            {"alpha": Rplus, "beta": Rplus},
            lambda value, alpha, beta: st.invgamma.logcdf(value, alpha, scale=beta),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to scaling issues",
    )
    def test_inverse_gamma_alt_params(self):
        def test_fun(value, mu, sigma):
            alpha, beta = pm.InverseGamma._get_alpha_beta(None, None, mu, sigma)
            return st.invgamma.logpdf(value, alpha, scale=beta)

        check_logp(
            pm.InverseGamma,
            Rplus,
            {"mu": Rplus, "sigma": Rplus},
            test_fun,
            decimal=select_by_precision(float64=4, float32=3),
        )

    def test_pareto(self):
        check_logp(
            pm.Pareto,
            Rplus,
            {"alpha": Rplusbig, "m": Rplusbig},
            lambda value, alpha, m: st.pareto.logpdf(value, alpha, scale=m),
        )
        check_logcdf(
            pm.Pareto,
            Rplus,
            {"alpha": Rplusbig, "m": Rplusbig},
            lambda value, alpha, m: st.pareto.logcdf(value, alpha, scale=m),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_weibull_logp(self):
        check_logp(
            pm.Weibull,
            Rplus,
            {"alpha": Rplusbig, "beta": Rplusbig},
            lambda value, alpha, beta: st.exponweib.logpdf(value, 1, alpha, scale=beta),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to inf issues",
    )
    def test_weibull_logcdf(self):
        check_logcdf(
            pm.Weibull,
            Rplus,
            {"alpha": Rplusbig, "beta": Rplusbig},
            lambda value, alpha, beta: st.exponweib.logcdf(value, 1, alpha, scale=beta),
        )

    def test_half_studentt(self):
        # this is only testing for nu=1 (halfcauchy)
        check_logp(
            pm.HalfStudentT,
            Rplus,
            {"sigma": Rplus},
            lambda value, sigma: st.halfcauchy.logpdf(value, 0, sigma),
            extra_args={"nu": 1},
        )

    def test_skew_normal(self):
        check_logp(
            pm.SkewNormal,
            R,
            {"mu": R, "sigma": Rplusbig, "alpha": R},
            lambda value, alpha, mu, sigma: st.skewnorm.logpdf(value, alpha, mu, sigma),
            decimal=select_by_precision(float64=5, float32=3),
        )

    @pytest.mark.parametrize(
        "value,mu,sigma,nu,logcdf_val",
        [
            (0.5, -50.000, 0.500, 0.500, 0.0000000),
            (1.0, -1.000, 0.001, 0.001, 0.0000000),
            (2.0, 0.001, 1.000, 1.000, -0.2365674),
            (5.0, 0.500, 2.500, 2.500, -0.2886489),
            (7.5, 2.000, 5.000, 5.000, -0.5655104),
            (15.0, 5.000, 7.500, 7.500, -0.4545255),
            (50.0, 50.000, 10.000, 10.000, -1.433714),
            (1000.0, 500.000, 10.000, 20.000, -1.573708e-11),
            (0.01, 0.01, 100.0, 0.01, -0.69314718),  # Fails in scipy version
            (-0.43402407, 0.0, 0.1, 0.1, -13.59615423),  # Previous 32-bit version failed here
            (-0.72402009, 0.0, 0.1, 0.1, -31.26571842),  # Previous 64-bit version failed here
        ],
    )
    def test_ex_gaussian_cdf(self, value, mu, sigma, nu, logcdf_val):
        """Log probabilities calculated using the pexGAUS function from the R package gamlss.
        See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or http://www.gamlss.org/."""
        npt.assert_almost_equal(
            logcdf(pm.ExGaussian.dist(mu=mu, sigma=sigma, nu=nu), value).eval(),
            logcdf_val,
            decimal=select_by_precision(float64=6, float32=2),
            err_msg=str((value, mu, sigma, nu, logcdf_val)),
        )

    def test_ex_gaussian_cdf_outside_edges(self):
        check_logcdf(
            pm.ExGaussian,
            R,
            {"mu": R, "sigma": Rplus, "nu": Rplus},
            None,
            skip_paramdomain_inside_edge_test=True,  # Valid values are tested above
        )

    @pytest.mark.skipif(condition=(aesara.config.floatX == "float32"), reason="Fails on float32")
    def test_vonmises(self):
        check_logp(
            pm.VonMises,
            Circ,
            {"mu": R, "kappa": Rplus},
            lambda value, mu, kappa: floatX(st.vonmises.logpdf(value, kappa, loc=mu)),
        )

    def test_gumbel(self):
        check_logp(
            pm.Gumbel,
            R,
            {"mu": R, "beta": Rplusbig},
            lambda value, mu, beta: st.gumbel_r.logpdf(value, loc=mu, scale=beta),
        )
        check_logcdf(
            pm.Gumbel,
            R,
            {"mu": R, "beta": Rplusbig},
            lambda value, mu, beta: st.gumbel_r.logcdf(value, loc=mu, scale=beta),
        )

    def test_logistic(self):
        check_logp(
            pm.Logistic,
            R,
            {"mu": R, "s": Rplus},
            lambda value, mu, s: st.logistic.logpdf(value, mu, s),
            decimal=select_by_precision(float64=6, float32=1),
        )
        check_logcdf(
            pm.Logistic,
            R,
            {"mu": R, "s": Rplus},
            lambda value, mu, s: st.logistic.logcdf(value, mu, s),
            decimal=select_by_precision(float64=6, float32=1),
        )

    def test_logitnormal(self):
        check_logp(
            pm.LogitNormal,
            Unit,
            {"mu": R, "sigma": Rplus},
            lambda value, mu, sigma: (
                st.norm.logpdf(sp.logit(value), mu, sigma) - (np.log(value) + np.log1p(-value))
            ),
            decimal=select_by_precision(float64=6, float32=1),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Some combinations underflow to -inf in float32 in pymc version",
    )
    def test_rice(self):
        check_logp(
            pm.Rice,
            Rplus,
            {"b": Rplus, "sigma": Rplusbig},
            lambda value, b, sigma: st.rice.logpdf(value, b=b, loc=0, scale=sigma),
        )
        if aesara.config.floatX == "float32":
            raise Exception("Flaky test: It passed this time, but XPASS is not allowed.")

    def test_rice_nu(self):
        check_logp(
            pm.Rice,
            Rplus,
            {"nu": Rplus, "sigma": Rplusbig},
            lambda value, nu, sigma: st.rice.logpdf(value, b=nu / sigma, loc=0, scale=sigma),
        )

    def test_moyal_logp(self):
        # Using a custom domain, because the standard `R` domain undeflows with scipy in float64
        value_domain = Domain([-np.inf, -1.5, -1, -0.01, 0.0, 0.01, 1, 1.5, np.inf])
        check_logp(
            pm.Moyal,
            value_domain,
            {"mu": R, "sigma": Rplusbig},
            lambda value, mu, sigma: floatX(st.moyal.logpdf(value, mu, sigma)),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Pymc3 underflows earlier than scipy on float32",
    )
    def test_moyal_logcdf(self):
        check_logcdf(
            pm.Moyal,
            R,
            {"mu": R, "sigma": Rplusbig},
            lambda value, mu, sigma: floatX(st.moyal.logcdf(value, mu, sigma)),
        )
        if aesara.config.floatX == "float32":
            raise Exception("Flaky test: It passed this time, but XPASS is not allowed.")

    def test_interpolated(self):
        for mu in R.vals:
            for sigma in Rplus.vals:
                # pylint: disable=cell-var-from-loop
                xmin = mu - 5 * sigma
                xmax = mu + 5 * sigma

                from pymc.distributions.continuous import interpolated

                class TestedInterpolated(pm.Interpolated):
                    rv_op = interpolated

                    @classmethod
                    def dist(cls, **kwargs):
                        x_points = np.linspace(xmin, xmax, 100000)
                        pdf_points = st.norm.pdf(x_points, loc=mu, scale=sigma)
                        return super().dist(x_points=x_points, pdf_points=pdf_points, **kwargs)

                def ref_pdf(value):
                    return np.where(
                        np.logical_and(value >= xmin, value <= xmax),
                        st.norm.logpdf(value, mu, sigma),
                        -np.inf * np.ones(value.shape),
                    )

                check_logp(TestedInterpolated, R, {}, ref_pdf)

    @pytest.mark.parametrize("transform", [pm.util.UNSET, None])
    def test_interpolated_transform(self, transform):
        # Issue: https://github.com/pymc-devs/pymc/issues/5048
        x_points = np.linspace(0, 10, 10)
        pdf_points = st.norm.pdf(x_points, loc=1, scale=1)
        with pm.Model() as m:
            x = pm.Interpolated("x", x_points, pdf_points, transform=transform)

        if transform is pm.util.UNSET:
            assert np.isfinite(m.compile_logp()({"x_interval__": -1.0}))
            assert np.isfinite(m.compile_logp()({"x_interval__": 11.0}))
        else:
            assert not np.isfinite(m.compile_logp()({"x": -1.0}))
            assert not np.isfinite(m.compile_logp()({"x": 11.0}))

    def test_truncated_normal(self):
        def scipy_logp(value, mu, sigma, lower, upper):
            return st.truncnorm.logpdf(
                value, (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
            )

        check_logp(
            pm.TruncatedNormal,
            R,
            {"mu": R, "sigma": Rplusbig, "lower": -Rplusbig, "upper": Rplusbig},
            scipy_logp,
            decimal=select_by_precision(float64=6, float32=1),
            skip_paramdomain_outside_edge_test=True,
        )

        check_logp(
            pm.TruncatedNormal,
            R,
            {"mu": R, "sigma": Rplusbig, "upper": Rplusbig},
            ft.partial(scipy_logp, lower=-np.inf),
            decimal=select_by_precision(float64=6, float32=1),
            skip_paramdomain_outside_edge_test=True,
        )

        check_logp(
            pm.TruncatedNormal,
            R,
            {"mu": R, "sigma": Rplusbig, "lower": -Rplusbig},
            ft.partial(scipy_logp, upper=np.inf),
            decimal=select_by_precision(float64=6, float32=1),
            skip_paramdomain_outside_edge_test=True,
        )

    def test_get_tau_sigma(self):
        sigma = np.array(2)
        npt.assert_almost_equal(get_tau_sigma(sigma=sigma), [1.0 / sigma**2, sigma])

        tau = np.array(2)
        npt.assert_almost_equal(get_tau_sigma(tau=tau), [tau, tau**-0.5])

        tau, _ = get_tau_sigma(sigma=at.constant(-2))
        with pytest.raises(ParameterValueError):
            tau.eval()

        _, sigma = get_tau_sigma(tau=at.constant(-2))
        with pytest.raises(ParameterValueError):
            sigma.eval()

        sigma = [1, 2]
        npt.assert_almost_equal(
            get_tau_sigma(sigma=sigma), [1.0 / np.array(sigma) ** 2, np.array(sigma)]
        )

    @pytest.mark.parametrize(
        "value,mu,sigma,nu,logp",
        [
            (0.5, -50.000, 0.500, 0.500, -99.8068528),
            (1.0, -1.000, 0.001, 0.001, -1992.5922447),
            (2.0, 0.001, 1.000, 1.000, -1.6720416),
            (5.0, 0.500, 2.500, 2.500, -2.4543644),
            (7.5, 2.000, 5.000, 5.000, -2.8259429),
            (15.0, 5.000, 7.500, 7.500, -3.3093854),
            (50.0, 50.000, 10.000, 10.000, -3.6436067),
            (1000.0, 500.000, 10.000, 20.000, -27.8707323),
            (-1.0, 1.0, 20.0, 0.9, -3.91967108),  # Fails in scipy version
            (0.01, 0.01, 100.0, 0.01, -5.5241087),  # Fails in scipy version
            (-1.0, 0.0, 0.1, 0.1, -51.022349),  # Fails in previous pymc version
        ],
    )
    def test_ex_gaussian(self, value, mu, sigma, nu, logp):
        """Log probabilities calculated using the dexGAUS function from the R package gamlss.
        See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or http://www.gamlss.org/."""
        with pm.Model() as model:
            pm.ExGaussian("eg", mu=mu, sigma=sigma, nu=nu)
        pt = {"eg": value}
        npt.assert_almost_equal(
            model.compile_logp()(pt),
            logp,
            decimal=select_by_precision(float64=6, float32=2),
            err_msg=str(pt),
        )


class TestMoments:
    @pytest.mark.parametrize(
        "size, expected",
        [
            (None, 0),
            (5, np.zeros(5)),
            ((2, 5), np.zeros((2, 5))),
        ],
    )
    def test_flat_moment(self, size, expected):
        with pm.Model() as model:
            pm.Flat("x", size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "size, expected",
        [
            (None, 1),
            (5, np.ones(5)),
            ((2, 5), np.ones((2, 5))),
        ],
    )
    def test_halfflat_moment(self, size, expected):
        with pm.Model() as model:
            pm.HalfFlat("x", size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "lower, upper, size, expected",
        [
            (-1, 1, None, 0),
            (-1, 1, 5, np.zeros(5)),
            (0, np.arange(1, 6), None, np.arange(1, 6) / 2),
            (0, np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(1, 6) / 2)),
        ],
    )
    def test_uniform_moment(self, lower, upper, size, expected):
        with pm.Model() as model:
            pm.Uniform("x", lower=lower, upper=upper, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, sigma, size, expected",
        [
            (0, 1, None, 0),
            (0, np.ones(5), None, np.zeros(5)),
            (np.arange(5), 1, None, np.arange(5)),
            (np.arange(5), np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(5))),
        ],
    )
    def test_normal_moment(self, mu, sigma, size, expected):
        with pm.Model() as model:
            pm.Normal("x", mu=mu, sigma=sigma, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "sigma, size, expected",
        [
            (1, None, 1),
            (1, 5, np.ones(5)),
            (np.arange(1, 6), None, np.arange(1, 6)),
            (np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(1, 6))),
        ],
    )
    def test_halfnormal_moment(self, sigma, size, expected):
        with pm.Model() as model:
            pm.HalfNormal("x", sigma=sigma, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "nu, sigma, size, expected",
        [
            (1, 1, None, 1),
            (1, 1, 5, np.ones(5)),
            (1, np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(1, 6))),
            (np.arange(1, 6), 1, None, np.full(5, 1)),
        ],
    )
    def test_halfstudentt_moment(self, nu, sigma, size, expected):
        with pm.Model() as model:
            pm.HalfStudentT("x", nu=nu, sigma=sigma, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, sigma, lower, upper, size, expected",
        [
            (0.9, 1, -5, 5, None, 0),
            (1, np.ones(5), -10, np.inf, None, np.full(5, -9)),
            (np.arange(5), 1, None, 10, (2, 5), np.full((2, 5), 9)),
            (1, 1, [-np.inf, -np.inf, -np.inf], 10, None, np.full(3, 9)),
        ],
    )
    def test_truncatednormal_moment(self, mu, sigma, lower, upper, size, expected):
        with pm.Model() as model:
            pm.TruncatedNormal("x", mu=mu, sigma=sigma, lower=lower, upper=upper, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "alpha, beta, size, expected",
        [
            (1, 1, None, 0.5),
            (1, 1, 5, np.full(5, 0.5)),
            (1, np.arange(1, 6), None, 1 / np.arange(2, 7)),
            (1, np.arange(1, 6), (2, 5), np.full((2, 5), 1 / np.arange(2, 7))),
        ],
    )
    def test_beta_moment(self, alpha, beta, size, expected):
        with pm.Model() as model:
            pm.Beta("x", alpha=alpha, beta=beta, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "nu, size, expected",
        [
            (1, None, 1),
            (1, 5, np.full(5, 1)),
            (np.arange(1, 6), None, np.arange(1, 6)),
        ],
    )
    def test_chisquared_moment(self, nu, size, expected):
        with pm.Model() as model:
            pm.ChiSquared("x", nu=nu, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "lam, size, expected",
        [
            (2, None, 0.5),
            (2, 5, np.full(5, 0.5)),
            (np.arange(1, 5), None, 1 / np.arange(1, 5)),
            (np.arange(1, 5), (2, 4), np.full((2, 4), 1 / np.arange(1, 5))),
        ],
    )
    def test_exponential_moment(self, lam, size, expected):
        with pm.Model() as model:
            pm.Exponential("x", lam=lam, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, b, size, expected",
        [
            (0, 1, None, 0),
            (0, np.ones(5), None, np.zeros(5)),
            (np.arange(5), 1, None, np.arange(5)),
            (np.arange(5), np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(5))),
        ],
    )
    def test_laplace_moment(self, mu, b, size, expected):
        with pm.Model() as model:
            pm.Laplace("x", mu=mu, b=b, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, nu, sigma, size, expected",
        [
            (0, 1, 1, None, 0),
            (0, np.ones(5), 1, None, np.zeros(5)),
            (np.arange(5), 10, np.arange(1, 6), None, np.arange(5)),
            (
                np.arange(5),
                10,
                np.arange(1, 6),
                (2, 5),
                np.full((2, 5), np.arange(5)),
            ),
        ],
    )
    def test_studentt_moment(self, mu, nu, sigma, size, expected):
        with pm.Model() as model:
            pm.StudentT("x", mu=mu, nu=nu, sigma=sigma, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "alpha, beta, size, expected",
        [
            (0, 1, None, 0),
            (0, np.ones(5), None, np.zeros(5)),
            (np.arange(5), 1, None, np.arange(5)),
            (np.arange(5), np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(5))),
        ],
    )
    def test_cauchy_moment(self, alpha, beta, size, expected):
        with pm.Model() as model:
            pm.Cauchy("x", alpha=alpha, beta=beta, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "a, b, size, expected",
        [
            (1, 1, None, 0.5),
            (1, 1, 5, np.full(5, 0.5)),
            (1, np.arange(1, 6), None, 1 / np.arange(2, 7)),
            (np.arange(1, 6), 1, None, np.arange(1, 6) / np.arange(2, 7)),
            (1, np.arange(1, 6), (2, 5), np.full((2, 5), 1 / np.arange(2, 7))),
        ],
    )
    def test_kumaraswamy_moment(self, a, b, size, expected):
        with pm.Model() as model:
            pm.Kumaraswamy("x", a=a, b=b, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, sigma, size, expected",
        [
            (0, 1, None, np.exp(0.5)),
            (0, 1, 5, np.full(5, np.exp(0.5))),
            (np.arange(5), 1, None, np.exp(np.arange(5) + 0.5)),
            (
                np.arange(5),
                np.arange(1, 6),
                (2, 5),
                np.full((2, 5), np.exp(np.arange(5) + 0.5 * np.arange(1, 6) ** 2)),
            ),
        ],
    )
    def test_lognormal_moment(self, mu, sigma, size, expected):
        with pm.Model() as model:
            pm.LogNormal("x", mu=mu, sigma=sigma, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "beta, size, expected",
        [
            (1, None, 1),
            (1, 5, np.ones(5)),
            (np.arange(1, 5), None, np.arange(1, 5)),
            (
                np.arange(1, 5),
                (2, 4),
                np.full((2, 4), np.arange(1, 5)),
            ),
        ],
    )
    def test_halfcauchy_moment(self, beta, size, expected):
        with pm.Model() as model:
            pm.HalfCauchy("x", beta=beta, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "alpha, beta, size, expected",
        [
            (1, 1, None, 1),
            (1, 1, 5, np.full(5, 1)),
            (np.arange(1, 6), 1, None, np.arange(1, 6)),
            (
                np.arange(1, 6),
                2 * np.arange(1, 6),
                (2, 5),
                np.full((2, 5), 0.5),
            ),
        ],
    )
    def test_gamma_moment(self, alpha, beta, size, expected):
        with pm.Model() as model:
            pm.Gamma("x", alpha=alpha, beta=beta, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "alpha, beta, size, expected",
        [
            (5, 1, None, 1 / 4),
            (0.5, 1, None, 1 / 1.5),
            (5, 1, 5, np.full(5, 1 / (5 - 1))),
            (np.arange(1, 6), 1, None, np.array([0.5, 1, 1 / 2, 1 / 3, 1 / 4])),
        ],
    )
    def test_inverse_gamma_moment(self, alpha, beta, size, expected):
        with pm.Model() as model:
            pm.InverseGamma("x", alpha=alpha, beta=beta, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "alpha, m, size, expected",
        [
            (2, 1, None, 1 * 2 ** (1 / 2)),
            (2, 1, 5, np.full(5, 1 * 2 ** (1 / 2))),
            (np.arange(2, 7), np.arange(1, 6), None, np.arange(1, 6) * 2 ** (1 / np.arange(2, 7))),
            (
                np.arange(2, 7),
                np.arange(1, 6),
                (2, 5),
                np.full((2, 5), np.arange(1, 6) * 2 ** (1 / np.arange(2, 7))),
            ),
        ],
    )
    def test_pareto_moment(self, alpha, m, size, expected):
        with pm.Model() as model:
            pm.Pareto("x", alpha=alpha, m=m, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, kappa, size, expected",
        [
            (0, 1, None, 0),
            (0, np.ones(4), None, np.zeros(4)),
            (np.arange(4), 0.5, None, np.arange(4)),
            (np.arange(4), np.arange(1, 5), (2, 4), np.full((2, 4), np.arange(4))),
        ],
    )
    def test_vonmises_moment(self, mu, kappa, size, expected):
        with pm.Model() as model:
            pm.VonMises("x", mu=mu, kappa=kappa, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, lam, phi, size, expected",
        [
            (2, None, None, None, 2),
            (None, 1, 1, 5, np.full(5, 1)),
            (1, None, np.ones(5), None, np.full(5, 1)),
            (3, np.full(5, 2), None, None, np.full(5, 3)),
            (np.arange(1, 6), None, np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(1, 6))),
        ],
    )
    def test_wald_moment(self, mu, lam, phi, size, expected):
        with pm.Model() as model:
            pm.Wald("x", mu=mu, lam=lam, phi=phi, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "alpha, beta, size, expected",
        [
            (1, 1, None, 1),
            (1, 1, 5, np.full(5, 1)),
            (np.arange(1, 6), 1, None, sp.gamma(1 + 1 / np.arange(1, 6))),
            (
                np.arange(1, 6),
                np.arange(2, 7),
                (2, 5),
                np.full(
                    (2, 5),
                    np.arange(2, 7) * sp.gamma(1 + 1 / np.arange(1, 6)),
                ),
            ),
        ],
    )
    def test_weibull_moment(self, alpha, beta, size, expected):
        with pm.Model() as model:
            pm.Weibull("x", alpha=alpha, beta=beta, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, s, size, expected",
        [
            (1, 1, None, 1),
            (1, 1, 5, np.full(5, 1)),
            (2, np.arange(1, 6), None, np.full(5, 2)),
            (
                np.arange(1, 6),
                np.arange(1, 6),
                (2, 5),
                np.full((2, 5), np.arange(1, 6)),
            ),
        ],
    )
    def test_logistic_moment(self, mu, s, size, expected):
        with pm.Model() as model:
            pm.Logistic("x", mu=mu, s=s, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, nu, sigma, size, expected",
        [
            (1, 1, 1, None, 2),
            (1, 1, np.ones((2, 5)), None, np.full([2, 5], 2)),
            (1, 1, 3, 5, np.full(5, 2)),
            (1, np.arange(1, 6), 5, None, np.arange(2, 7)),
            (1, np.arange(1, 6), 1, (2, 5), np.full((2, 5), np.arange(2, 7))),
        ],
    )
    def test_exgaussian_moment(self, mu, nu, sigma, size, expected):
        with pm.Model() as model:
            pm.ExGaussian("x", mu=mu, sigma=sigma, nu=nu, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, beta, size, expected",
        [
            (0, 2, None, 2 * np.euler_gamma),
            (1, np.arange(1, 4), None, 1 + np.arange(1, 4) * np.euler_gamma),
            (np.arange(5), 2, None, np.arange(5) + 2 * np.euler_gamma),
            (1, 2, 5, np.full(5, 1 + 2 * np.euler_gamma)),
            (
                np.arange(5),
                np.arange(1, 6),
                (2, 5),
                np.full((2, 5), np.arange(5) + np.arange(1, 6) * np.euler_gamma),
            ),
        ],
    )
    def test_gumbel_moment(self, mu, beta, size, expected):
        with pm.Model() as model:
            pm.Gumbel("x", mu=mu, beta=beta, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "c, lower, upper, size, expected",
        [
            (1, 0, 5, None, 2),
            (3, np.arange(-3, 6, 3), np.arange(3, 12, 3), None, np.array([1, 3, 5])),
            (np.arange(-3, 6, 3), -3, 3, None, np.array([-1, 0, 1])),
            (3, -3, 6, 5, np.full(5, 2)),
            (
                np.arange(-3, 6, 3),
                np.arange(-9, -2, 3),
                np.arange(3, 10, 3),
                (2, 3),
                np.full((2, 3), np.array([-3, 0, 3])),
            ),
        ],
    )
    def test_triangular_moment(self, c, lower, upper, size, expected):
        with pm.Model() as model:
            pm.Triangular("x", c=c, lower=lower, upper=upper, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, sigma, size, expected",
        [
            (1, 2, None, sp.expit(1)),
            (0, np.arange(1, 5), None, sp.expit(np.zeros(4))),
            (np.arange(4), 1, None, sp.expit(np.arange(4))),
            (1, 5, 4, sp.expit(np.ones(4))),
            (np.arange(4), np.arange(1, 5), (2, 4), np.full((2, 4), sp.expit(np.arange(4)))),
        ],
    )
    def test_logitnormal_moment(self, mu, sigma, size, expected):
        with pm.Model() as model:
            pm.LogitNormal("x", mu=mu, sigma=sigma, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "x_points, pdf_points, size, expected",
        [
            (np.array([-1, 1]), np.array([0.4, 0.6]), None, 0.2),
            (
                np.array([-4, -1, 3, 9, 19]),
                np.array([0.1, 0.15, 0.2, 0.25, 0.3]),
                None,
                9.34782609,
            ),
            (
                np.array([-22, -4, 0, 8, 13]),
                np.tile(1 / 5, 5),
                (5, 3),
                np.full((5, 3), -4.5),
            ),
            (
                np.arange(-100, 10),
                np.arange(1, 111) / 6105,
                (2, 5, 3),
                np.full((2, 5, 3), -27.65765766),
            ),
            (
                # from https://github.com/pymc-devs/pymc/issues/5959
                np.linspace(0, 10, 10),
                st.norm.pdf(np.linspace(0, 10, 10), loc=2.5, scale=1),
                None,
                2.5270134,
            ),
            (
                np.linspace(0, 10, 100),
                st.norm.pdf(np.linspace(0, 10, 100), loc=2.5, scale=1),
                None,
                2.51771721,
            ),
        ],
    )
    def test_interpolated_moment(self, x_points, pdf_points, size, expected):
        with pm.Model() as model:
            pm.Interpolated("x", x_points=x_points, pdf_points=pdf_points, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, sigma, size, expected",
        [
            (4.0, 3.0, None, 7.8110885363844345),
            (4.0, np.full(5, 3), None, np.full(5, 7.8110885363844345)),
            (np.arange(5), 1, None, np.arange(5) + 1.2703628454614782),
            (np.arange(5), np.ones(5), (2, 5), np.full((2, 5), np.arange(5) + 1.2703628454614782)),
        ],
    )
    def test_moyal_moment(self, mu, sigma, size, expected):
        with pm.Model() as model:
            pm.Moyal("x", mu=mu, sigma=sigma, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "alpha, mu, sigma, size, expected",
        [
            (1.0, 1.0, 1.0, None, 1.56418958),
            (1.0, np.ones(5), 1.0, None, np.full(5, 1.56418958)),
            (np.ones(5), 1, np.ones(5), None, np.full(5, 1.56418958)),
            (
                np.arange(5),
                np.arange(1, 6),
                np.arange(1, 6),
                None,
                (1.0, 3.12837917, 5.14094894, 7.02775903, 8.87030861),
            ),
            (
                np.arange(5),
                np.arange(1, 6),
                np.arange(1, 6),
                (2, 5),
                np.full((2, 5), (1.0, 3.12837917, 5.14094894, 7.02775903, 8.87030861)),
            ),
        ],
    )
    def test_skewnormal_moment(self, alpha, mu, sigma, size, expected):
        with pm.Model() as model:
            pm.SkewNormal("x", alpha=alpha, mu=mu, sigma=sigma, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "b, kappa, mu, size, expected",
        [
            (1.0, 1.0, 1.0, None, 1.0),
            (1.0, np.ones(5), 1.0, None, np.full(5, 1.0)),
            (np.arange(1, 6), 1.0, np.ones(5), None, np.full(5, 1.0)),
            (
                np.arange(1, 6),
                np.arange(1, 6),
                np.arange(1, 6),
                None,
                (1.0, 1.25, 2.111111111111111, 3.0625, 4.04),
            ),
            (
                np.arange(1, 6),
                np.arange(1, 6),
                np.arange(1, 6),
                (2, 5),
                np.full((2, 5), (1.0, 1.25, 2.111111111111111, 3.0625, 4.04)),
            ),
        ],
    )
    def test_asymmetriclaplace_moment(self, b, kappa, mu, size, expected):
        with pm.Model() as model:
            pm.AsymmetricLaplace("x", b=b, kappa=kappa, mu=mu, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "nu, sigma, size, expected",
        [
            (1.0, 1.0, None, 1.5485724605511453),
            (1.0, np.ones(5), None, np.full(5, 1.5485724605511453)),
            (
                np.arange(1, 6),
                1.0,
                None,
                (
                    1.5485724605511453,
                    2.2723834280687427,
                    3.1725772879007166,
                    4.127193542536757,
                    5.101069639492123,
                ),
            ),
            (
                np.arange(1, 6),
                np.ones(5),
                (2, 5),
                np.full(
                    (2, 5),
                    (
                        1.5485724605511453,
                        2.2723834280687427,
                        3.1725772879007166,
                        4.127193542536757,
                        5.101069639492123,
                    ),
                ),
            ),
        ],
    )
    def test_rice_moment(self, nu, sigma, size, expected):
        with pm.Model() as model:
            pm.Rice("x", nu=nu, sigma=sigma, size=size)

    @pytest.mark.parametrize(
        "h, z, size, expected",
        [
            (1.0, 0.0, None, 0.25),
            (
                1.0,
                np.arange(5),
                None,
                (
                    0.25,
                    0.23105857863000487,
                    0.1903985389889412,
                    0.1508580422741444,
                    0.12050344750947711,
                ),
            ),
            (
                np.arange(1, 6),
                np.arange(5),
                None,
                (
                    0.25,
                    0.46211715726000974,
                    0.5711956169668236,
                    0.6034321690965776,
                    0.6025172375473855,
                ),
            ),
            (
                np.arange(1, 6),
                np.arange(5),
                (2, 5),
                np.full(
                    (2, 5),
                    (
                        0.25,
                        0.46211715726000974,
                        0.5711956169668236,
                        0.6034321690965776,
                        0.6025172375473855,
                    ),
                ),
            ),
        ],
    )
    def test_polyagamma_moment(self, h, z, size, expected):
        with pm.Model() as model:
            pm.PolyaGamma("x", h=h, z=z, size=size)
        assert_moment_is_expected(model, expected)
