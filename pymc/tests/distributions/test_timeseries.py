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
import itertools as it
import warnings

import aesara
import numpy as np
import pytest
import scipy.stats as st

from aesara.tensor import TensorVariable

import pymc as pm

from pymc.aesaraf import floatX
from pymc.distributions.continuous import Flat, HalfNormal, Normal
from pymc.distributions.discrete import DiracDelta
from pymc.distributions.logprob import logp
from pymc.distributions.multivariate import Dirichlet
from pymc.distributions.shape_utils import change_dist_size
from pymc.distributions.timeseries import (
    AR,
    GARCH11,
    EulerMaruyama,
    GaussianRandomWalk,
    get_steps,
)
from pymc.model import Model
from pymc.sampling import draw, sample, sample_posterior_predictive
from pymc.tests.distributions.util import (
    R,
    Rplus,
    Vector,
    assert_moment_is_expected,
    check_logp,
)
from pymc.tests.helpers import SeededTest, select_by_precision


@pytest.mark.parametrize(
    "steps, shape, step_shape_offset, expected_steps, consistent",
    [
        (10, None, 0, 10, True),
        (10, None, 1, 10, True),
        (None, (10,), 0, 10, True),
        (None, (10,), 1, 9, True),
        (None, (10, 5), 0, 5, True),
        (None, None, 0, None, True),
        (10, (10,), 0, 10, True),
        (10, (11,), 1, 10, True),
        (10, (5, 5), 0, 5, False),
        (10, (5, 10), 1, 9, False),
    ],
)
@pytest.mark.parametrize("info_source", ("shape", "dims", "observed"))
def test_get_steps(info_source, steps, shape, step_shape_offset, expected_steps, consistent):
    if info_source == "shape":
        inferred_steps = get_steps(steps=steps, shape=shape, step_shape_offset=step_shape_offset)

    elif info_source == "dims":
        if shape is None:
            dims = None
            coords = {}
        else:
            dims = tuple(str(i) for i, shape in enumerate(shape))
            coords = {str(i): range(shape) for i, shape in enumerate(shape)}
        with Model(coords=coords):
            inferred_steps = get_steps(steps=steps, dims=dims, step_shape_offset=step_shape_offset)

    elif info_source == "observed":
        if shape is None:
            observed = None
        else:
            observed = np.zeros(shape)
        inferred_steps = get_steps(
            steps=steps, observed=observed, step_shape_offset=step_shape_offset
        )

    if not isinstance(inferred_steps, TensorVariable):
        assert inferred_steps == expected_steps
    else:
        if consistent:
            assert inferred_steps.eval() == expected_steps
        else:
            assert inferred_steps.owner.inputs[0].eval() == expected_steps
            with pytest.raises(AssertionError, match="Steps do not match"):
                inferred_steps.eval()


class TestGaussianRandomWalk:
    def test_logp(self):
        def ref_logp(value, mu, sigma):
            return (
                st.norm.logpdf(value[0], 0, 100)  # default init_dist has a scale 100
                + st.norm.logpdf(np.diff(value), mu, sigma).sum()
            )

        check_logp(
            pm.GaussianRandomWalk,
            Vector(R, 4),
            {"mu": R, "sigma": Rplus},
            ref_logp,
            decimal=select_by_precision(float64=6, float32=1),
            extra_args={"steps": 3, "init_dist": Normal.dist(0, 100)},
        )

    def test_gaussianrandomwalk_inference(self):
        mu, sigma, steps = 2, 1, 1000
        obs = np.concatenate([[0], np.random.normal(mu, sigma, size=steps)]).cumsum()

        with pm.Model():
            _mu = pm.Uniform("mu", -10, 10)
            _sigma = pm.Uniform("sigma", 0, 10)

            obs_data = pm.MutableData("obs_data", obs)
            grw = GaussianRandomWalk(
                "grw", _mu, _sigma, steps=steps, observed=obs_data, init_dist=Normal.dist(0, 100)
            )

            trace = pm.sample(chains=1)

        recovered_mu = trace.posterior["mu"].mean()
        recovered_sigma = trace.posterior["sigma"].mean()
        np.testing.assert_allclose([mu, sigma], [recovered_mu, recovered_sigma], atol=0.2)

    @pytest.mark.parametrize("init", [None, pm.Normal.dist()])
    def test_gaussian_random_walk_init_dist_shape(self, init):
        """Test that init_dist is properly resized"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Initial distribution not specified.*", UserWarning)
            grw = pm.GaussianRandomWalk.dist(mu=0, sigma=1, steps=1, init_dist=init)
            assert tuple(grw.owner.inputs[0].shape.eval()) == ()

            grw = pm.GaussianRandomWalk.dist(mu=0, sigma=1, steps=1, init_dist=init, size=(5,))
            assert tuple(grw.owner.inputs[0].shape.eval()) == (5,)

            grw = pm.GaussianRandomWalk.dist(mu=0, sigma=1, steps=1, init_dist=init, shape=2)
            assert tuple(grw.owner.inputs[0].shape.eval()) == ()

            grw = pm.GaussianRandomWalk.dist(mu=0, sigma=1, steps=1, init_dist=init, shape=(5, 2))
            assert tuple(grw.owner.inputs[0].shape.eval()) == (5,)

            grw = pm.GaussianRandomWalk.dist(mu=[0, 0], sigma=1, steps=1, init_dist=init)
            assert tuple(grw.owner.inputs[0].shape.eval()) == (2,)

            grw = pm.GaussianRandomWalk.dist(mu=0, sigma=[1, 1], steps=1, init_dist=init)
            assert tuple(grw.owner.inputs[0].shape.eval()) == (2,)

            grw = pm.GaussianRandomWalk.dist(
                mu=np.zeros((3, 1)), sigma=[1, 1], steps=1, init_dist=init
            )
            assert tuple(grw.owner.inputs[0].shape.eval()) == (3, 2)

    def test_gaussianrandomwalk_broadcasted_by_init_dist(self):
        grw = pm.GaussianRandomWalk.dist(
            mu=0, sigma=1, steps=4, init_dist=pm.Normal.dist(size=(2, 3))
        )
        assert tuple(grw.shape.eval()) == (2, 3, 5)
        assert grw.eval().shape == (2, 3, 5)

    @pytest.mark.parametrize("shape", ((6,), (3, 6)))
    def test_inferred_steps_from_shape(self, shape):
        x = GaussianRandomWalk.dist(shape=shape, init_dist=Normal.dist(0, 100))
        steps = x.owner.inputs[-1]
        assert steps.eval() == 5

    def test_missing_steps(self):
        with pytest.raises(ValueError, match="Must specify steps or shape parameter"):
            GaussianRandomWalk.dist(shape=None, init_dist=Normal.dist(0, 100))

    def test_inconsistent_steps_and_shape(self):
        with pytest.raises(AssertionError, match="Steps do not match last shape dimension"):
            x = GaussianRandomWalk.dist(steps=12, shape=45, init_dist=Normal.dist(0, 100))

    def test_inferred_steps_from_dims(self):
        with pm.Model(coords={"batch": range(5), "steps": range(20)}):
            x = GaussianRandomWalk("x", dims=("batch", "steps"), init_dist=Normal.dist(0, 100))
        steps = x.owner.inputs[-1]
        assert steps.eval() == 19

    def test_inferred_steps_from_observed(self):
        with pm.Model():
            x = GaussianRandomWalk("x", observed=np.zeros(10), init_dist=Normal.dist(0, 100))
        steps = x.owner.inputs[-1]
        assert steps.eval() == 9

    @pytest.mark.parametrize(
        "init",
        [
            pm.HalfNormal.dist(sigma=2),
            pm.StudentT.dist(nu=4, mu=1, sigma=0.5),
        ],
    )
    def test_gaussian_random_walk_init_dist_logp(self, init):
        grw = pm.GaussianRandomWalk.dist(init_dist=init, steps=1)
        assert np.isclose(
            pm.logp(grw, [0, 0]).eval(),
            pm.logp(init, 0).eval() + st.norm.logpdf(0),
        )

    @pytest.mark.parametrize(
        "mu, sigma, init_dist, steps, size, expected",
        [
            (0, 1, Normal.dist(1), 10, None, np.ones((11,))),
            (1, 1, Normal.dist(0), 10, (2,), np.full((2, 11), np.arange(11))),
            (1, 1, Normal.dist([0, 1]), 10, None, np.vstack((np.arange(11), np.arange(11) + 1))),
            (0, [1, 1], Normal.dist(0), 10, None, np.zeros((2, 11))),
            (
                [1, -1],
                1,
                Normal.dist(0),
                10,
                (4, 2),
                np.full((4, 2, 11), np.vstack((np.arange(11), -np.arange(11)))),
            ),
        ],
    )
    def test_moment(self, mu, sigma, init_dist, steps, size, expected):
        with Model() as model:
            GaussianRandomWalk("x", mu=mu, sigma=sigma, init_dist=init_dist, steps=steps, size=size)
        assert_moment_is_expected(model, expected)

    def test_init_deprecated_arg(self):
        with pytest.warns(FutureWarning, match="init parameter is now called init_dist"):
            pm.GaussianRandomWalk.dist(init=Normal.dist(), shape=(10,))


class TestAR:
    def test_order1_logp(self):
        data = np.array([0.3, 1, 2, 3, 4])
        phi = np.array([0.99])
        with Model() as t:
            y = AR("y", phi, sigma=1, init_dist=Flat.dist(), shape=len(data))
            z = Normal("z", mu=phi * data[:-1], sigma=1, shape=len(data) - 1)
        ar_like = t.compile_logp(y)({"y": data})
        reg_like = t.compile_logp(z)({"z": data[1:]})
        np.testing.assert_allclose(ar_like, reg_like)

        with Model() as t_constant:
            y = AR(
                "y",
                np.hstack((0.3, phi)),
                sigma=1,
                init_dist=Flat.dist(),
                shape=len(data),
                constant=True,
            )
            z = Normal("z", mu=0.3 + phi * data[:-1], sigma=1, shape=len(data) - 1)
        ar_like = t_constant.compile_logp(y)({"y": data})
        reg_like = t_constant.compile_logp(z)({"z": data[1:]})
        np.testing.assert_allclose(ar_like, reg_like)

    def test_order2_logp(self):
        data = np.array([0.3, 1, 2, 3, 4])
        phi = np.array([0.84, 0.10])
        with Model() as t:
            y = AR("y", phi, sigma=1, init_dist=Flat.dist(), shape=len(data))
            z = Normal(
                "z", mu=phi[0] * data[1:-1] + phi[1] * data[:-2], sigma=1, shape=len(data) - 2
            )
        ar_like = t.compile_logp(y)({"y": data})
        reg_like = t.compile_logp(z)({"z": data[2:]})
        np.testing.assert_allclose(ar_like, reg_like)

    @pytest.mark.parametrize("constant", (False, True))
    def test_batched_size(self, constant):
        ar_order, steps, batch_size = 3, 100, 5
        beta_tp = np.random.randn(batch_size, ar_order + int(constant))
        y_tp = np.random.randn(batch_size, steps)
        with Model() as t0:
            y = AR(
                "y",
                beta_tp,
                shape=(batch_size, steps),
                initval=y_tp,
                constant=constant,
                init_dist=Normal.dist(0, 100, shape=(batch_size, steps)),
            )
        with Model() as t1:
            for i in range(batch_size):
                AR(
                    f"y_{i}",
                    beta_tp[i],
                    sigma=1.0,
                    shape=steps,
                    initval=y_tp[i],
                    constant=constant,
                    init_dist=Normal.dist(0, 100, shape=steps),
                )

        assert y.owner.op.ar_order == ar_order

        np.testing.assert_allclose(
            t0.compile_logp()(t0.initial_point()),
            t1.compile_logp()(t1.initial_point()),
        )

        y_eval = draw(y, draws=2)
        assert y_eval[0].shape == (batch_size, steps)
        assert not np.any(np.isclose(y_eval[0], y_eval[1]))

    def test_batched_rhos(self):
        ar_order, steps, batch_size = 3, 100, 5
        beta_tp = np.random.randn(batch_size, ar_order)
        y_tp = np.random.randn(batch_size, steps)
        with Model() as t0:
            beta = Normal("beta", 0.0, 1.0, shape=(batch_size, ar_order), initval=beta_tp)
            AR(
                "y",
                beta,
                sigma=1.0,
                init_dist=Normal.dist(0, 1),
                shape=(batch_size, steps),
                initval=y_tp,
            )
        with Model() as t1:
            beta = Normal("beta", 0.0, 1.0, shape=(batch_size, ar_order), initval=beta_tp)
            for i in range(batch_size):
                AR(
                    f"y_{i}",
                    beta[i],
                    init_dist=Normal.dist(0, 1),
                    sigma=1.0,
                    shape=steps,
                    initval=y_tp[i],
                )

        np.testing.assert_allclose(
            t0.compile_logp()(t0.initial_point()),
            t1.compile_logp()(t1.initial_point()),
        )

        beta_tp[1] = 0  # Should always be close to zero
        y_eval = t0["y"].eval({t0["beta"]: beta_tp})
        assert y_eval.shape == (batch_size, steps)
        assert np.all(abs(y_eval[1]) < 5)

    def test_batched_sigma(self):
        ar_order, steps, batch_size = 4, 100, (7, 5)
        # AR order cannot be inferred from beta_tp because it is not fixed.
        # We specify it manually below
        beta_tp = aesara.shared(np.random.randn(ar_order))
        sigma_tp = np.abs(np.random.randn(*batch_size))
        y_tp = np.random.randn(*batch_size, steps)
        with Model() as t0:
            sigma = HalfNormal("sigma", 1.0, shape=batch_size, initval=sigma_tp)
            AR(
                "y",
                beta_tp,
                sigma=sigma,
                init_dist=Normal.dist(0, sigma[..., None]),
                size=batch_size,
                steps=steps,
                initval=y_tp,
                ar_order=ar_order,
            )
        with Model() as t1:
            sigma = HalfNormal("beta", 1.0, shape=batch_size, initval=sigma_tp)
            for i in range(batch_size[0]):
                for j in range(batch_size[1]):
                    AR(
                        f"y_{i}{j}",
                        beta_tp,
                        sigma=sigma[i][j],
                        init_dist=Normal.dist(0, sigma[i][j]),
                        shape=steps,
                        initval=y_tp[i, j],
                        ar_order=ar_order,
                    )

        # Check logp shape
        sigma_logp, y_logp = t0.compile_logp(sum=False)(t0.initial_point())
        assert tuple(y_logp.shape) == batch_size

        np.testing.assert_allclose(
            sigma_logp.sum() + y_logp.sum(),
            t1.compile_logp()(t1.initial_point()),
        )

        beta_tp.set_value(np.zeros((ar_order,)))  # Should always be close to zero
        sigma_tp = np.full(batch_size, [0.01, 0.1, 1, 10, 100])
        y_eval = t0["y"].eval({t0["sigma"]: sigma_tp})
        assert y_eval.shape == (*batch_size, steps + ar_order)
        assert np.allclose(y_eval.std(axis=(0, 2)), [0.01, 0.1, 1, 10, 100], rtol=0.1)

    def test_batched_init_dist(self):
        ar_order, steps, batch_size = 3, 100, 5
        beta_tp = aesara.shared(np.random.randn(ar_order), shape=(3,))
        y_tp = np.random.randn(batch_size, steps)
        with Model() as t0:
            init_dist = Normal.dist(0.0, 100.0, size=(batch_size, ar_order))
            AR("y", beta_tp, sigma=0.01, init_dist=init_dist, steps=steps, initval=y_tp)
        with Model() as t1:
            for i in range(batch_size):
                AR(
                    f"y_{i}",
                    beta_tp,
                    sigma=0.01,
                    shape=steps,
                    initval=y_tp[i],
                    init_dist=Normal.dist(0, 100, shape=steps),
                )

        np.testing.assert_allclose(
            t0.compile_logp()(t0.initial_point()),
            t1.compile_logp()(t1.initial_point()),
        )

        # Next values should keep close to previous ones
        beta_tp.set_value(np.full((ar_order,), 1 / ar_order))
        # Init dist is cloned when creating the AR, so the original variable is not
        # part of the AR graph. We retrieve the one actually used manually
        init_dist = t0["y"].owner.inputs[2]
        init_dist_tp = np.full((batch_size, ar_order), (np.arange(batch_size) * 100)[:, None])
        y_eval = t0["y"].eval({init_dist: init_dist_tp})
        assert y_eval.shape == (batch_size, steps + ar_order)
        assert np.allclose(
            y_eval[:, -10:].mean(-1), np.arange(batch_size) * 100, rtol=0.1, atol=0.5
        )

    def test_constant_random(self):
        x = AR.dist(
            rho=[100, 0, 0],
            sigma=0.1,
            init_dist=Normal.dist(-100.0, sigma=0.1),
            constant=True,
            shape=(6,),
        )
        x_eval = x.eval()
        assert np.allclose(x_eval[:2], -100, rtol=0.1)
        assert np.allclose(x_eval[2:], 100, rtol=0.1)

    def test_multivariate_init_dist(self):
        init_dist = Dirichlet.dist(a=np.full((5, 2), [1, 10]))
        x = AR.dist(rho=[0, 0], init_dist=init_dist, steps=0)

        x_eval = x.eval()
        assert x_eval.shape == (5, 2)

        init_dist_eval = init_dist.eval()
        init_dist_logp_eval = logp(init_dist, init_dist_eval).eval()
        x_logp_eval = logp(x, init_dist_eval).eval()
        assert x_logp_eval.shape == (5,)
        assert np.allclose(x_logp_eval, init_dist_logp_eval)

    @pytest.mark.parametrize(
        "size, expected",
        [
            (None, np.full((2, 7), [[2.0], [4.0]])),
            ((5, 2), np.full((5, 2, 7), [[2.0], [4.0]])),
        ],
    )
    def test_moment(self, size, expected):
        with Model() as model:
            init_dist = DiracDelta.dist([[1.0, 2.0], [3.0, 4.0]])
            AR("x", rho=[0, 0], init_dist=init_dist, steps=5, size=size)
        assert_moment_is_expected(model, expected, check_finite_logp=False)

    def test_init_deprecated_arg(self):
        with pytest.warns(FutureWarning, match="init parameter is now called init_dist"):
            pm.AR.dist(rho=[1, 2, 3], init=Normal.dist(), shape=(10,))

    def test_change_dist_size(self):
        base_dist = pm.AR.dist(rho=[0.5, 0.5], init_dist=pm.Normal.dist(size=(2,)), shape=(3, 10))

        new_dist = change_dist_size(base_dist, (4,))
        assert new_dist.eval().shape == (4, 10)

        new_dist = change_dist_size(base_dist, (4,), expand=True)
        assert new_dist.eval().shape == (4, 3, 10)


class TestGARCH11:
    def test_logp(self):
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
        garch_like = t.compile_logp(y)({"y": data})
        reg_like = t.compile_logp(z)({"z": data})
        decimal = select_by_precision(float64=7, float32=4)
        np.testing.assert_allclose(garch_like, reg_like, 10 ** (-decimal))

    @pytest.mark.parametrize(
        "batched_param",
        ["omega", "alpha_1", "beta_1", "initial_vol"],
    )
    @pytest.mark.parametrize("explicit_shape", (True, False))
    def test_batched_size(self, explicit_shape, batched_param):
        steps, batch_size = 100, 5
        param_val = np.square(np.random.randn(batch_size))
        init_kwargs = dict(
            omega=1.25,
            alpha_1=0.5,
            beta_1=0.45,
            initial_vol=2.5,
        )
        kwargs0 = init_kwargs.copy()
        kwargs0[batched_param] = init_kwargs[batched_param] * param_val
        if explicit_shape:
            kwargs0["shape"] = (batch_size, steps)
        else:
            kwargs0["steps"] = steps - 1
        with Model() as t0:
            y = GARCH11("y", **kwargs0)

        y_eval = draw(y, draws=2)
        assert y_eval[0].shape == (batch_size, steps)
        assert not np.any(np.isclose(y_eval[0], y_eval[1]))

        kwargs1 = init_kwargs.copy()
        if explicit_shape:
            kwargs1["shape"] = steps
        else:
            kwargs1["steps"] = steps - 1
        with Model() as t1:
            for i in range(batch_size):
                kwargs1[batched_param] = init_kwargs[batched_param] * param_val[i]
                GARCH11(f"y_{i}", **kwargs1)

        np.testing.assert_allclose(
            t0.compile_logp()(t0.initial_point()),
            t1.compile_logp()(t1.initial_point()),
        )

    @pytest.mark.parametrize(
        "size, expected",
        [
            (None, np.zeros((2, 8))),
            ((5, 2), np.zeros((5, 2, 8))),
        ],
    )
    def test_moment(self, size, expected):
        with Model() as model:
            GARCH11(
                "x",
                omega=1.25,
                alpha_1=0.5,
                beta_1=0.45,
                initial_vol=np.ones(2),
                steps=7,
                size=size,
            )
        assert_moment_is_expected(model, expected, check_finite_logp=True)

    def test_change_dist_size(self):
        base_dist = pm.GARCH11.dist(
            omega=1.25, alpha_1=0.5, beta_1=0.45, initial_vol=1.0, shape=(3, 10)
        )

        new_dist = change_dist_size(base_dist, (4,))
        assert new_dist.eval().shape == (4, 10)

        new_dist = change_dist_size(base_dist, (4,), expand=True)
        assert new_dist.eval().shape == (4, 3, 10)


def _gen_sde_path(sde, pars, dt, n, x0):
    xs = [x0]
    wt = np.random.normal(size=(n,) if isinstance(x0, float) else (n, x0.size))
    for i in range(n):
        f, g = sde(xs[-1], *pars)
        xs.append(xs[-1] + f * dt + np.sqrt(dt) * g * wt[i])
    return np.array(xs)


@pytest.mark.xfail(reason="Timeseries not refactored", raises=NotImplementedError)
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
    data = it.chain(it.product(*mudim_as_event), it.product(*mudim_as_dist))
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
