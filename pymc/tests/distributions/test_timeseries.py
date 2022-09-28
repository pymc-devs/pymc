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
import pytensor
import pytest
import scipy.stats as st

from pytensor.tensor.random.op import RandomVariable

import pymc as pm

from pymc import MutableData
from pymc.distributions.continuous import Exponential, Flat, HalfNormal, Normal, Uniform
from pymc.distributions.discrete import DiracDelta
from pymc.distributions.logprob import logp
from pymc.distributions.multivariate import (
    Dirichlet,
    LKJCholeskyCov,
    MvNormal,
    MvStudentT,
)
from pymc.distributions.shape_utils import change_dist_size
from pymc.distributions.timeseries import (
    AR,
    GARCH11,
    EulerMaruyama,
    GaussianRandomWalk,
    MvGaussianRandomWalk,
    MvStudentTRandomWalk,
    RandomWalk,
)
from pymc.model import Model
from pymc.pytensorf import floatX
from pymc.sampling.forward import draw, sample_posterior_predictive
from pymc.sampling.mcmc import sample
from pymc.tests.distributions.util import assert_moment_is_expected
from pymc.tests.helpers import select_by_precision


class TestRandomWalk:
    def test_dists_types(self):
        init_dist = Normal.dist()
        innovation_dist = Normal.dist()

        with pytest.raises(
            TypeError,
            match="init_dist must be a distribution variable",
        ):
            RandomWalk.dist(init_dist=5, innovation_dist=innovation_dist, steps=5)

        with pytest.raises(
            TypeError,
            match="innovation_dist must be a distribution variable",
        ):
            RandomWalk.dist(init_dist=init_dist, innovation_dist=5, steps=5)

        init_dist = MvNormal.dist([0], [[1]])
        innovation_dist = Normal.dist(size=(1,))
        with pytest.raises(
            TypeError,
            match="init_dist and innovation_dist must have the same support dimensionality",
        ):
            RandomWalk.dist(init_dist=init_dist, innovation_dist=innovation_dist, steps=5)

    def test_dists_not_registered_check(self):
        with Model():
            init = Normal("init")
            innovation = Normal("innovation")

            init_dist = Normal.dist()
            innovation_dist = Normal.dist()
            with pytest.raises(
                ValueError,
                match="The dist init was already registered in the current model",
            ):
                RandomWalk("rw", init_dist=init, innovation_dist=innovation_dist, steps=5)

            with pytest.raises(
                ValueError,
                match="The dist innovation was already registered in the current model",
            ):
                RandomWalk("rw", init_dist=init_dist, innovation_dist=innovation, steps=5)

    @pytest.mark.parametrize(
        "init_dist, innovation_dist, steps, size, shape, "
        "init_dist_size, innovation_dist_size, rw_shape",
        [
            (Normal.dist(), Normal.dist(), 1, None, None, 1, 1, (2,)),
            (Normal.dist(), Normal.dist(), 1, None, (2,), 1, 1, (2,)),
            (Normal.dist(), Normal.dist(), 3, (5,), None, 5, 5 * 3, (5, 4)),
            (Normal.dist(), Normal.dist(), 3, None, (5, 4), 5, 5 * 3, (5, 4)),
            (Normal.dist(), Normal.dist(mu=np.zeros(3)), 1, None, None, 3, 3, (3, 2)),
            (
                Normal.dist(),
                Normal.dist(mu=np.zeros((3, 1)), sigma=np.ones(2)),
                4,
                None,
                None,
                3 * 2,
                3 * 2 * 4,
                (3, 2, 5),
            ),
            (Normal.dist(size=(2, 3)), Normal.dist(), 4, None, None, 2 * 3, 2 * 3 * 4, (2, 3, 5)),
            (Dirichlet.dist(np.ones(3)), Dirichlet.dist(np.ones(3)), 1, None, None, 3, 3, (2, 3)),
            (Dirichlet.dist(np.ones(3)), Dirichlet.dist(np.ones(3)), 1, None, (2, 3), 3, 3, (2, 3)),
            (
                Dirichlet.dist(np.ones(3)),
                Dirichlet.dist(np.ones(3)),
                4,
                (6,),
                None,
                6 * 3,
                6 * 4 * 3,
                (6, 5, 3),
            ),
            (
                Dirichlet.dist(np.ones(3)),
                Dirichlet.dist(np.ones(3)),
                4,
                None,
                (6, 5, 3),
                6 * 3,
                6 * 4 * 3,
                (6, 5, 3),
            ),
            (
                Dirichlet.dist(np.ones(3)),
                Dirichlet.dist(np.ones((4, 3))),
                1,
                None,
                None,
                4 * 3,
                4 * 3,
                (4, 2, 3),
            ),
            (
                Dirichlet.dist(np.ones((2, 3))),
                Dirichlet.dist(np.ones(3)),
                4,
                None,
                None,
                2 * 3,
                2 * 4 * 3,
                (2, 5, 3),
            ),
        ],
    )
    def test_dist_sizes(
        self,
        init_dist,
        innovation_dist,
        steps,
        size,
        shape,
        init_dist_size,
        innovation_dist_size,
        rw_shape,
    ):
        """Test that init and innovation dists are properly resized"""
        rw = RandomWalk.dist(
            steps=steps,
            init_dist=init_dist,
            innovation_dist=innovation_dist,
            size=size,
            shape=shape,
        )
        init_dist, innovation_dist = rw.owner.inputs[:2]
        # Check the inputs are pure RandomVariable's (and not e.g., Broadcasts)
        assert isinstance(init_dist.owner.op, RandomVariable)
        assert isinstance(innovation_dist.owner.op, RandomVariable)
        assert init_dist.size.eval() == init_dist_size
        assert innovation_dist.size.eval() == innovation_dist_size
        assert tuple(rw.shape.eval()) == rw_shape

    def test_change_size_univariate(self):
        init_dist = Normal.dist()
        innovation_dist = Normal.dist()

        # size = 5
        rw = RandomWalk.dist(init_dist=init_dist, innovation_dist=innovation_dist, shape=(5, 100))

        new_rw = change_dist_size(rw, new_size=(7,))
        assert tuple(new_rw.shape.eval()) == (7, 100)

        new_rw = change_dist_size(rw, new_size=(4, 3), expand=True)
        assert tuple(new_rw.shape.eval()) == (4, 3, 5, 100)

    def test_change_size_multivariate(self):
        init_dist = Dirichlet.dist([1, 1, 1])
        innovation_dist = Dirichlet.dist([1, 1, 1])

        # size = 5
        rw = RandomWalk.dist(
            init_dist=init_dist, innovation_dist=innovation_dist, shape=(5, 100, 3)
        )

        new_rw = change_dist_size(rw, new_size=(7,))
        assert tuple(new_rw.shape.eval()) == (7, 100, 3)

        new_rw = change_dist_size(rw, new_size=(4, 3), expand=True)
        assert tuple(new_rw.shape.eval()) == (4, 3, 5, 100, 3)

    @pytest.mark.parametrize(
        "init_dist, innovation_dist, shape, steps",
        [
            (Normal.dist(), Normal.dist(), (5, 3), 2),
            (Dirichlet.dist([1, 1, 1]), Dirichlet.dist([1, 1, 1]), (5, 3), 4),
        ],
    )
    @pytest.mark.parametrize("steps_source", ("shape", "dims", "observed"))
    def test_infer_steps(self, init_dist, innovation_dist, shape, steps, steps_source):
        shape_source_kwargs = dict(shape=None, dims=None, observed=None)
        if steps_source == "shape":
            shape_source_kwargs["shape"] = shape
        elif steps_source == "dims":
            shape_source_kwargs["dims"] = [f"dim{i}" for i in range(len(shape))]
        elif steps_source == "observed":
            shape_source_kwargs["observed"] = np.zeros(shape)
        else:
            raise ValueError

        coords = {f"dim{i}": range(s) for i, s in enumerate(shape)}
        with Model(coords=coords):
            x = RandomWalk(
                "x", init_dist=init_dist, innovation_dist=innovation_dist, **shape_source_kwargs
            )
        inferred_steps = x.owner.inputs[-1]
        assert inferred_steps.eval().item() == steps

    def test_infer_steps_error(self):
        with pytest.raises(ValueError, match="Must specify steps or shape parameter"):
            RandomWalk.dist(init_dist=Normal.dist(), innovation_dist=Normal.dist())

    @pytest.mark.parametrize(
        "init_dist, innovation_dist, steps, shape",
        [
            (Normal.dist(), Normal.dist(), 12, (13, 42)),
            (Dirichlet.dist([1, 1, 1]), Dirichlet.dist([1, 1, 1]), 12, (13, 42, 3)),
        ],
    )
    def test_inconsistent_steps_and_shape(self, init_dist, innovation_dist, steps, shape):
        with pytest.raises(
            AssertionError, match="support_shape does not match respective shape dimension"
        ):
            RandomWalk.dist(
                init_dist=init_dist,
                innovation_dist=innovation_dist,
                steps=steps,
                shape=shape,
            ).eval()

    @pytest.mark.parametrize(
        "init_dist",
        [
            pm.HalfNormal.dist(sigma=2),
            pm.StudentT.dist(nu=4, mu=1, sigma=0.5),
        ],
    )
    def test_init_logp_univariate(self, init_dist):
        rw = RandomWalk.dist(init_dist=init_dist, innovation_dist=Normal.dist(), steps=1)
        assert np.isclose(
            pm.logp(rw, [0, 0]).eval(),
            pm.logp(init_dist, 0).eval() + st.norm.logpdf(0),
        )

    @pytest.mark.parametrize(
        "init_dist",
        [
            MvNormal.dist(mu=[1, 2, 3], cov=np.eye(3) * 1.5),
            MvStudentT.dist(nu=4, mu=[1, 2, 3], scale=np.eye(3) * 0.5),
        ],
    )
    def test_init_logp_multivariate(self, init_dist):
        rw = RandomWalk.dist(
            init_dist=init_dist,
            innovation_dist=MvNormal.dist(np.zeros(3), np.eye(3)),
            steps=1,
        )
        assert np.isclose(
            pm.logp(rw, np.zeros((2, 3))).eval(),
            pm.logp(init_dist, np.zeros(3)).eval()
            + st.multivariate_normal.logpdf(np.zeros(3)).sum(),
        )

    def test_innovation_logp_univariate(self):
        steps = 5
        dist = RandomWalk.dist(
            init_dist=Normal.dist(0, 1),
            innovation_dist=Normal.dist(1, 1),
            shape=(steps,),
        )
        assert np.isclose(
            logp(dist, np.arange(5)).eval(),
            logp(Normal.dist(0, 1), 0).eval() * steps,
        )

    def test_innovation_logp_multivariate(self):
        steps = 5
        dist = RandomWalk.dist(
            init_dist=MvNormal.dist(np.zeros(3), cov=np.eye(3)),
            innovation_dist=MvNormal.dist(mu=np.ones(3), cov=np.eye(3)),
            shape=(steps, 3),
        )
        assert np.isclose(
            logp(dist, np.full((3, 5), np.arange(5)).T).eval(),
            logp(MvNormal.dist(np.zeros(3), cov=np.eye(3)), np.zeros(3)).eval() * steps,
        )

    @pytest.mark.parametrize(
        "init_dist, innovation_dist, steps, size, expected, check_finite_logp",
        [
            (Normal.dist(-10), Normal.dist(1), 9, None, np.arange(-10, 0), True),
            (
                Normal.dist(-10),
                Normal.dist(1),
                9,
                (5, 3),
                np.full((5, 3, 10), np.arange(-10, 0)),
                True,
            ),
            (
                Normal.dist([-10, 0]),
                Normal.dist(1),
                9,
                None,
                np.concatenate(
                    [[np.arange(-10, 0)], [np.arange(0, 10)]],
                    axis=0,
                ),
                True,
            ),
            (
                MvNormal.dist([-10, 0, 10], np.eye(3)),
                MvNormal.dist([1, 2, 3], np.eye(3)),
                9,
                None,
                np.concatenate(
                    [[np.arange(-10, 0, 1)], [np.arange(0, 20, 2)], [np.arange(10, 40, 3)]],
                    axis=0,
                ).T,
                True,
            ),
            (
                MvNormal.dist([-10, 0, 10], np.eye(3)),
                MvNormal.dist([1, 2, 3], np.eye(3)),
                9,
                (5, 4),
                np.full(
                    (5, 4, 10, 3),
                    np.concatenate(
                        [[np.arange(-10, 0, 1)], [np.arange(0, 20, 2)], [np.arange(10, 40, 3)]],
                        axis=0,
                    ).T,
                ),
                False,  # MvNormal logp only supports 2D values
            ),
        ],
    )
    def test_moment(self, init_dist, innovation_dist, steps, size, expected, check_finite_logp):
        with Model() as model:
            RandomWalk(
                "x", init_dist=init_dist, innovation_dist=innovation_dist, steps=steps, size=size
            )
        assert_moment_is_expected(model, expected, check_finite_logp=check_finite_logp)


class TestPredefinedRandomWalk:
    def test_gaussian(self):
        x = GaussianRandomWalk.dist(mu=0, sigma=1, init_dist=Flat.dist(), steps=5)
        init_dist, innovation_dist = x.owner.inputs[:2]
        assert isinstance(init_dist.owner.op, Flat)
        assert isinstance(innovation_dist.owner.op, Normal)

    def test_gaussian_inference(self):
        mu, sigma, steps = 2, 1, 1000
        obs = np.concatenate([[0], np.random.normal(mu, sigma, size=steps)]).cumsum()

        with Model():
            _mu = Uniform("mu", -10, 10)
            _sigma = Uniform("sigma", 0, 10)

            obs_data = MutableData("obs_data", obs)
            grw = GaussianRandomWalk(
                "grw", _mu, _sigma, steps=steps, observed=obs_data, init_dist=Normal.dist(0, 100)
            )

            trace = sample(chains=1)

        recovered_mu = trace.posterior["mu"].mean()
        recovered_sigma = trace.posterior["sigma"].mean()
        np.testing.assert_allclose([mu, sigma], [recovered_mu, recovered_sigma], atol=0.2)

    @pytest.mark.parametrize("param", ["cov", "chol", "tau"])
    def test_mvgaussian(self, param):
        x = MvGaussianRandomWalk.dist(
            mu=np.ones(3),
            **{param: np.eye(3)},
            init_dist=Dirichlet.dist(np.ones(3)),
            steps=5,
        )
        init_dist, innovation_dist = x.owner.inputs[:2]
        assert isinstance(init_dist.owner.op, Dirichlet)
        assert isinstance(innovation_dist.owner.op, MvNormal)

    @pytest.mark.parametrize("param", ("chol", "cov"))
    def test_mvgaussian_with_chol_cov_rv(self, param):
        with pm.Model() as model:
            mu = Normal("mu", 0.0, 1.0, shape=3)
            sd_dist = Exponential.dist(1.0, shape=3)
            # pylint: disable=unpacking-non-sequence
            chol, corr, stds = LKJCholeskyCov(
                "chol_cov", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            # pylint: enable=unpacking-non-sequence
            if param == "chol":
                mv = MvGaussianRandomWalk("mv", mu, chol=chol, shape=(10, 7, 3))
            elif param == "cov":
                mv = MvGaussianRandomWalk("mv", mu, cov=pm.math.dot(chol, chol.T), shape=(10, 7, 3))
            else:
                raise ValueError
        assert draw(mv, draws=5).shape == (5, 10, 7, 3)

    @pytest.mark.parametrize("param", ["cov", "chol", "tau"])
    def test_mvstudentt(self, param):
        x = MvStudentTRandomWalk.dist(
            nu=4,
            mu=np.ones(3),
            **{param: np.eye(3)},
            init_dist=Dirichlet.dist(np.ones(3)),
            steps=5,
        )
        init_dist, innovation_dist = x.owner.inputs[:2]
        assert isinstance(init_dist.owner.op, Dirichlet)
        assert isinstance(innovation_dist.owner.op, MvStudentT)

    @pytest.mark.parametrize(
        "distribution, init_dist, build_kwargs",
        [
            (GaussianRandomWalk, Normal.dist(), dict()),
            (MvGaussianRandomWalk, Dirichlet.dist(np.ones(3)), dict(mu=np.zeros(3), tau=np.eye(3))),
            (
                MvStudentTRandomWalk,
                Dirichlet.dist(np.ones(3)),
                dict(nu=4, mu=np.zeros(3), tau=np.eye(3)),
            ),
        ],
    )
    def test_init_deprecated_arg(self, distribution, init_dist, build_kwargs):
        with pytest.warns(FutureWarning, match="init parameter is now called init_dist"):
            distribution.dist(init=init_dist, steps=10, **build_kwargs)


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
        beta_tp = pytensor.shared(np.random.randn(ar_order))
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
        beta_tp = pytensor.shared(np.random.randn(ar_order), shape=(3,))
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
            AR.dist(rho=[1, 2, 3], init=Normal.dist(), shape=(10,))

    def test_change_dist_size(self):
        base_dist = AR.dist(rho=[0.5, 0.5], init_dist=pm.Normal.dist(size=(2,)), shape=(3, 10))

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
        rtol = select_by_precision(float64=1e-9, float32=1e-6)
        np.testing.assert_allclose(garch_like, reg_like, rtol)

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
        base_dist = GARCH11.dist(
            omega=1.25, alpha_1=0.5, beta_1=0.45, initial_vol=1.0, shape=(3, 10)
        )

        new_dist = change_dist_size(base_dist, (4,))
        assert new_dist.eval().shape == (4, 10)

        new_dist = change_dist_size(base_dist, (4,), expand=True)
        assert new_dist.eval().shape == (4, 3, 10)


class TestEulerMaruyama:
    @pytest.mark.parametrize("batched_param", [1, 2])
    @pytest.mark.parametrize("explicit_shape", (True, False))
    def test_batched_size(self, explicit_shape, batched_param):
        RANDOM_SEED = 42
        numpy_rng = np.random.default_rng(RANDOM_SEED)

        steps, batch_size = 100, 5
        param_val = np.square(numpy_rng.standard_normal(batch_size))
        if explicit_shape:
            kwargs = {"shape": (batch_size, steps)}
        else:
            kwargs = {"steps": steps - 1}

        def sde_fn(x, k, d, s):
            return (k - d * x, s)

        sde_pars = [1.0, 2.0, 0.1]
        sde_pars[batched_param] = sde_pars[batched_param] * param_val
        with Model() as t0:
            init_dist = pm.Normal.dist(0, 10, shape=(batch_size,))
            y = EulerMaruyama(
                "y", dt=0.02, sde_fn=sde_fn, sde_pars=sde_pars, init_dist=init_dist, **kwargs
            )

        y_eval = draw(y, draws=2, random_seed=numpy_rng)
        assert y_eval[0].shape == (batch_size, steps)
        assert not np.any(np.isclose(y_eval[0], y_eval[1]))

        if explicit_shape:
            kwargs["shape"] = steps
        with Model() as t1:
            for i in range(batch_size):
                sde_pars_slice = sde_pars.copy()
                sde_pars_slice[batched_param] = sde_pars[batched_param][i]
                init_dist = pm.Normal.dist(0, 10)
                EulerMaruyama(
                    f"y_{i}",
                    dt=0.02,
                    sde_fn=sde_fn,
                    sde_pars=sde_pars_slice,
                    init_dist=init_dist,
                    **kwargs,
                )

        t0_init = t0.initial_point(random_seed=RANDOM_SEED)
        t1_init = {f"y_{i}": t0_init["y"][i] for i in range(batch_size)}
        np.testing.assert_allclose(
            t0.compile_logp()(t0_init),
            t1.compile_logp()(t1_init),
        )

    def test_change_dist_size1(self):
        def sde1(x, k, d, s):
            return (k - d * x, s)

        base_dist = EulerMaruyama.dist(
            dt=0.01,
            sde_fn=sde1,
            sde_pars=(1, 2, 0.1),
            init_dist=pm.Normal.dist(0, 10),
            shape=(5, 10),
        )

        new_dist = change_dist_size(base_dist, (4,))
        assert new_dist.eval().shape == (4, 10)

        new_dist = change_dist_size(base_dist, (4,), expand=True)
        assert new_dist.eval().shape == (4, 5, 10)

    def test_change_dist_size2(self):
        def sde2(p, s):
            N = 500.0
            return s * p * (1 - p) / (1 + s * p), pm.math.sqrt(p * (1 - p) / N)

        base_dist = EulerMaruyama.dist(
            dt=0.01, sde_fn=sde2, sde_pars=(0.1,), init_dist=pm.Normal.dist(0, 10), shape=(3, 10)
        )

        new_dist = change_dist_size(base_dist, (4,))
        assert new_dist.eval().shape == (4, 10)

        new_dist = change_dist_size(base_dist, (4,), expand=True)
        assert new_dist.eval().shape == (4, 3, 10)

    def test_linear_model(self):
        lam = -0.78
        sig2 = 5e-3
        N = 300
        dt = 1e-1

        RANDOM_SEED = 42
        numpy_rng = np.random.default_rng(RANDOM_SEED)

        def _gen_sde_path(sde, pars, dt, n, x0):
            xs = [x0]
            wt = numpy_rng.normal(size=(n,) if isinstance(x0, float) else (n, x0.size))
            for i in range(n):
                f, g = sde(xs[-1], *pars)
                xs.append(xs[-1] + f * dt + np.sqrt(dt) * g * wt[i])
            return np.array(xs)

        sde = lambda x, lam: (lam * x, sig2)
        x = floatX(_gen_sde_path(sde, (lam,), dt, N, 5.0))
        z = x + numpy_rng.standard_normal(size=x.size) * sig2
        # build model
        with Model() as model:
            lamh = Flat("lamh")
            xh = EulerMaruyama(
                "xh", dt, sde, (lamh,), steps=N, initval=x, init_dist=pm.Normal.dist(0, 10)
            )
            Normal("zh", mu=xh, sigma=sig2, observed=z)
        # invert
        with model:
            trace = sample(chains=1, random_seed=numpy_rng)

        ppc = sample_posterior_predictive(trace, model=model, random_seed=numpy_rng)

        p95 = [2.5, 97.5]
        lo, hi = np.percentile(trace.posterior["lamh"], p95, axis=[0, 1])
        assert (lo < lam) and (lam < hi)
        lo, hi = np.percentile(ppc.posterior_predictive["zh"], p95, axis=[0, 1])
        assert ((lo < z) * (z < hi)).mean() > 0.95
