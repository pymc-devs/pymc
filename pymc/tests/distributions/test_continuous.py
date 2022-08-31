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
