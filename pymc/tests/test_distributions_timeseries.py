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

import numpy as np
import pytest

from scipy import stats

import pymc as pm

from pymc.aesaraf import floatX
from pymc.distributions.continuous import Flat, Normal
from pymc.distributions.timeseries import (
    AR,
    AR1,
    GARCH11,
    EulerMaruyama,
    GaussianRandomWalk,
    gaussianrandomwalk,
)
from pymc.model import Model
from pymc.sampling import sample, sample_posterior_predictive
from pymc.tests.helpers import select_by_precision


class TestGaussianRandomWalk:
    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({"steps": 5}, (6,)),
            ({"size": 1}, (5,)),
            ({"size": 2}, (5,)),
            # implied dims are not working
            pytest.param({"mu": [0, 0]}, (2, 5), marks=pytest.mark.xfail),
        ],
    )
    def test_grw_rv_op_shape(self, kwargs, expected):
        """Basic test for GRW RV op"""
        default_kwargs = dict(init=1, mu=3, sd=0.0000001, steps=4, size=None)

        combined_kwargs = {**default_kwargs, **kwargs}
        grw = gaussianrandomwalk(
            combined_kwargs["mu"],
            combined_kwargs["sd"],
            combined_kwargs["init"],
            combined_kwargs["steps"],
        ).eval()

        assert grw.shape == expected

    def test_grw_logp(self):
        vals = [0, 1, 2]
        mu = 1
        sigma = 1
        init = 0

        with pm.Model():
            grw = GaussianRandomWalk("grw", mu, sigma, init, steps=2)

        logp = pm.logp(grw, vals)
        logp_vals = logp.eval()

        # Calculate logp in explicit loop to make testing sequence obvious
        init_val = vals[0]
        init_logp = stats.norm(init, sigma).logpdf(init_val)
        logp_reference = [init_logp]

        for x_minus_one_val, x_val in zip(vals, vals[1:]):
            logp_point = stats.norm(x_minus_one_val + mu + init, sigma).logpdf(x_val)
            logp_reference.append(logp_point)

        np.testing.assert_almost_equal(logp_vals, logp_reference)

    def test_grw_inference(self):
        mu, sigma, steps = 2, 1, 10000
        obs = np.concatenate([[0], np.random.normal(mu, sigma, size=steps)]).cumsum()

        with pm.Model():
            _mu = pm.Uniform("mu", -10, 10)
            _sigma = pm.Uniform("sigma", 0, 10)
            grw = GaussianRandomWalk("grw", _mu, _sigma, init=0, steps=steps, observed=obs)
            trace = pm.sample()

        recovered_mu = trace.posterior["mu"].mean()
        recovered_sigma = trace.posterior["sigma"].mean()
        np.testing.assert_allclose([mu, sigma], [recovered_mu, recovered_sigma], atol=0.2)

    @pytest.mark.parametrize(
        "steps,size,expected",
        (
            (1, None, (2,)),
            (2, 1, (1, 3)),
            (2, 5, (5, 3)),
            (10, 5, (5, 11)),
        ),
    )
    def test_grw_shape(self, steps, size, expected):
        grw_dist = pm.GaussianRandomWalk.dist(mu=0, sigma=1, steps=steps, size=size)
        expected_symbolic = tuple(grw_dist.shape.eval())
        assert expected_symbolic == expected


@pytest.mark.xfail(reason="Timeseries not refactored")
def test_AR():
    # AR1
    data = np.array([0.3, 1, 2, 3, 4])
    phi = np.array([0.99])
    with Model() as t:
        y = AR("y", phi, sigma=1, shape=len(data))
        z = Normal("z", mu=phi * data[:-1], sigma=1, shape=len(data) - 1)
    ar_like = t["y"].logp({"z": data[1:], "y": data})
    reg_like = t["z"].logp({"z": data[1:], "y": data})
    np.testing.assert_allclose(ar_like, reg_like)

    # AR1 and AR(1)
    with Model() as t:
        rho = Normal("rho", 0.0, 1.0)
        y1 = AR1("y1", rho, 1.0, observed=data)
        y2 = AR("y2", rho, 1.0, init=Normal.dist(0, 1), observed=data)
    initial_point = t.compute_initial_point()
    np.testing.assert_allclose(y1.logp(initial_point), y2.logp(initial_point))

    # AR1 + constant
    with Model() as t:
        y = AR("y", np.hstack((0.3, phi)), sigma=1, shape=len(data), constant=True)
        z = Normal("z", mu=0.3 + phi * data[:-1], sigma=1, shape=len(data) - 1)
    ar_like = t["y"].logp({"z": data[1:], "y": data})
    reg_like = t["z"].logp({"z": data[1:], "y": data})
    np.testing.assert_allclose(ar_like, reg_like)

    # AR2
    phi = np.array([0.84, 0.10])
    with Model() as t:
        y = AR("y", phi, sigma=1, shape=len(data))
        z = Normal("z", mu=phi[0] * data[1:-1] + phi[1] * data[:-2], sigma=1, shape=len(data) - 2)
    ar_like = t["y"].logp({"z": data[2:], "y": data})
    reg_like = t["z"].logp({"z": data[2:], "y": data})
    np.testing.assert_allclose(ar_like, reg_like)


@pytest.mark.xfail(reason="Timeseries not refactored")
def test_AR_nd():
    # AR2 multidimensional
    p, T, n = 3, 100, 5
    beta_tp = np.random.randn(p, n)
    y_tp = np.random.randn(T, n)
    with Model() as t0:
        beta = Normal("beta", 0.0, 1.0, shape=(p, n), initval=beta_tp)
        AR("y", beta, sigma=1.0, shape=(T, n), initval=y_tp)

    with Model() as t1:
        beta = Normal("beta", 0.0, 1.0, shape=(p, n), initval=beta_tp)
        for i in range(n):
            AR("y_%d" % i, beta[:, i], sigma=1.0, shape=T, initval=y_tp[:, i])

    np.testing.assert_allclose(
        t0.logp(t0.compute_initial_point()), t1.logp(t1.compute_initial_point())
    )


@pytest.mark.xfail(reason="Timeseries not refactored")
def test_GARCH11():
    # test data ~ N(0, 1)
    data = np.array(
        [
            -1.35078362,
            -0.81254164,
            0.28918551,
            -2.87043544,
            -0.94353337,
            0.83660719,
            -0.23336562,
            -0.58586298,
            -1.36856736,
            -1.60832975,
            -1.31403141,
            0.05446936,
            -0.97213128,
            -0.18928725,
            1.62011258,
            -0.95978616,
            -2.06536047,
            0.6556103,
            -0.27816645,
            -1.26413397,
        ]
    )
    omega = 0.6
    alpha_1 = 0.4
    beta_1 = 0.5
    initial_vol = np.float64(0.9)
    vol = np.empty_like(data)
    vol[0] = initial_vol
    for i in range(len(data) - 1):
        vol[i + 1] = np.sqrt(omega + beta_1 * vol[i] ** 2 + alpha_1 * data[i] ** 2)

    with Model() as t:
        y = GARCH11(
            "y",
            omega=omega,
            alpha_1=alpha_1,
            beta_1=beta_1,
            initial_vol=initial_vol,
            shape=data.shape,
        )
        z = Normal("z", mu=0, sigma=vol, shape=data.shape)
    garch_like = t["y"].logp({"z": data, "y": data})
    reg_like = t["z"].logp({"z": data, "y": data})
    decimal = select_by_precision(float64=7, float32=4)
    np.testing.assert_allclose(garch_like, reg_like, 10 ** (-decimal))


def _gen_sde_path(sde, pars, dt, n, x0):
    xs = [x0]
    wt = np.random.normal(size=(n,) if isinstance(x0, float) else (n, x0.size))
    for i in range(n):
        f, g = sde(xs[-1], *pars)
        xs.append(xs[-1] + f * dt + np.sqrt(dt) * g * wt[i])
    return np.array(xs)


@pytest.mark.xfail(reason="Timeseries not refactored")
def test_linear():
    lam = -0.78
    sig2 = 5e-3
    N = 300
    dt = 1e-1
    sde = lambda x, lam: (lam * x, sig2)
    x = floatX(_gen_sde_path(sde, (lam,), dt, N, 5.0))
    z = x + np.random.randn(x.size) * sig2
    # build model
    with Model() as model:
        lamh = Flat("lamh")
        xh = EulerMaruyama("xh", dt, sde, (lamh,), shape=N + 1, initval=x)
        Normal("zh", mu=xh, sigma=sig2, observed=z)
    # invert
    with model:
        trace = sample(init="advi+adapt_diag", chains=1)

    ppc = sample_posterior_predictive(trace, model=model)

    p95 = [2.5, 97.5]
    lo, hi = np.percentile(trace[lamh], p95, axis=0)
    assert (lo < lam) and (lam < hi)
    lo, hi = np.percentile(ppc["zh"], p95, axis=0)
    assert ((lo < z) * (z < hi)).mean() > 0.95
