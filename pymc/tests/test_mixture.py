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

from contextlib import ExitStack as does_not_raise

import aesara
import numpy as np
import pytest
import scipy.stats as st

from aeppl.transforms import IntervalTransform, LogTransform
from aeppl.transforms import Simplex as SimplexTransform
from aesara import tensor as at
from aesara.tensor import TensorVariable
from aesara.tensor.random.op import RandomVariable
from numpy.testing import assert_allclose
from scipy.special import logsumexp

from pymc.aesaraf import floatX
from pymc.distributions import (
    Categorical,
    Dirichlet,
    DirichletMultinomial,
    Exponential,
    Gamma,
    HalfNormal,
    HalfStudentT,
    LKJCholeskyCov,
    LogNormal,
    Mixture,
    Multinomial,
    MvNormal,
    Normal,
    NormalMixture,
    Poisson,
    StickBreakingWeights,
    Triangular,
    Uniform,
)
from pymc.distributions.logprob import logp
from pymc.distributions.mixture import MixtureTransformWarning
from pymc.distributions.shape_utils import to_tuple
from pymc.distributions.transforms import _default_transform
from pymc.math import expand_packed_triangular
from pymc.model import Model
from pymc.sampling import (
    draw,
    sample,
    sample_posterior_predictive,
    sample_prior_predictive,
)
from pymc.step_methods import Metropolis
from pymc.tests.helpers import SeededTest
from pymc.tests.test_distributions import Domain, Simplex
from pymc.tests.test_distributions_moments import assert_moment_is_expected
from pymc.tests.test_distributions_random import pymc_random


def generate_normal_mixture_data(w, mu, sigma, size=1000):
    component = np.random.choice(w.size, size=size, p=w)
    mu, sigma = np.broadcast_arrays(mu, sigma)
    out_size = to_tuple(size) + mu.shape[:-1]
    mu_ = np.array([mu[..., comp] for comp in component.ravel()])
    sigma_ = np.array([sigma[..., comp] for comp in component.ravel()])
    mu_ = np.reshape(mu_, out_size)
    sigma_ = np.reshape(sigma_, out_size)
    x = np.random.normal(mu_, sigma_, size=out_size)

    return x


def generate_poisson_mixture_data(w, mu, size=1000):
    component = np.random.choice(w.size, size=size, p=w)
    mu = np.atleast_1d(mu)
    out_size = to_tuple(size) + mu.shape[:-1]
    mu_ = np.array([mu[..., comp] for comp in component.ravel()])
    mu_ = np.reshape(mu_, out_size)
    x = np.random.poisson(mu_, size=out_size)

    return x


class TestMixture(SeededTest):
    def get_inital_point(self, model):
        """Get initial point with untransformed variables for posterior predictive sampling"""
        return {
            var.name: initial_point
            for var, initial_point in zip(
                model.unobserved_value_vars,
                model.compile_fn(model.unobserved_value_vars)(model.compute_initial_point()),
            )
        }

    @pytest.mark.parametrize(
        "weights",
        [
            np.array([1, 0]),
            np.array([[1, 0], [0, 1], [1, 0]]),
        ],
    )
    @pytest.mark.parametrize(
        "component",
        [
            Normal.dist([-10, 10]),
            Normal.dist([-10, 10], size=(3, 2)),
            Normal.dist([[-15, 15], [-10, 10], [-5, 5]], 1e-3),
            Normal.dist([-10, 10], size=(4, 3, 2)),
        ],
    )
    @pytest.mark.parametrize("size", [None, (3,), (4, 3)])
    def test_single_univariate_component_deterministic_weights(self, weights, component, size):
        # Size can't be smaller than what is implied by replication dimensions
        if size is not None and len(size) < max(component.ndim - 1, weights.ndim - 1):
            return

        mix = Mixture.dist(weights, component, size=size)
        mix_eval = mix.eval()

        # Test shape
        # component shape is either (4, 3, 2), (3, 2) or (2,)
        # weights shape is either (3, 2) or (2,)
        if size is not None:
            expected_shape = size
        elif component.ndim == 3:
            expected_shape = (4, 3)
        elif component.ndim == 2 or weights.ndim == 2:
            expected_shape = (3,)
        else:
            expected_shape = ()
        assert mix_eval.shape == expected_shape

        # Test draws
        expected_positive = np.zeros_like(mix_eval)
        if expected_positive.ndim > 0:
            expected_positive[..., :] = (weights == 1)[..., 1]
        assert np.all((mix_eval > 0) == expected_positive)
        repetitions = np.unique(mix_eval).size < mix_eval.size
        assert not repetitions

        # Test logp
        mix_logp_eval = logp(mix, mix_eval).eval()
        assert mix_logp_eval.shape == expected_shape
        bcast_weights = np.broadcast_to(weights, (*expected_shape, 2))
        expected_logp = logp(component, mix_eval[..., None]).eval()[bcast_weights == 1]
        expected_logp = expected_logp.reshape(expected_shape)
        assert np.allclose(mix_logp_eval, expected_logp)

    @pytest.mark.parametrize(
        "weights",
        [
            np.array([1, 0]),
            np.array([[1, 0], [0, 1], [1, 0]]),
        ],
    )
    @pytest.mark.parametrize(
        "components",
        [
            (Normal.dist(-10, 1e-3), Normal.dist(10, 1e-3)),
            (Normal.dist(-10, 1e-3, size=(3,)), Normal.dist(10, 1e-3, size=(3,))),
            (Normal.dist([-15, -10, -5], 1e-3), Normal.dist([15, 10, 5], 1e-3)),
            (Normal.dist(-10, 1e-3, size=(4, 3)), Normal.dist(10, 1e-3, size=(4, 3))),
        ],
    )
    @pytest.mark.parametrize("size", [None, (3,), (4, 3)])
    def test_list_univariate_components_deterministic_weights(self, weights, components, size):
        # Size can't be smaller than what is implied by replication dimensions
        if size is not None and len(size) < max(components[0].ndim, weights.ndim - 1):
            return

        mix = Mixture.dist(weights, components, size=size)
        mix_eval = mix.eval()

        # Test shape
        # components[0] shape is either (4, 3), (3,) or ()
        # weights shape is either (3, 2) or (2,)
        if size is not None:
            expected_shape = size
        elif components[0].ndim == 2:
            expected_shape = (4, 3)
        elif components[0].ndim == 1 or weights.ndim == 2:
            expected_shape = (3,)
        else:
            expected_shape = ()
        assert mix_eval.shape == expected_shape

        # Test draws
        expected_positive = np.zeros_like(mix_eval)
        if expected_positive.ndim > 0:
            expected_positive[..., :] = (weights == 1)[..., 1]
        assert np.all((mix_eval > 0) == expected_positive)
        repetitions = np.unique(mix_eval).size < mix_eval.size
        assert not repetitions

        # Test logp
        mix_logp_eval = logp(mix, mix_eval).eval()
        assert mix_logp_eval.shape == expected_shape
        bcast_weights = np.broadcast_to(weights, (*expected_shape, 2))
        expected_logp = np.stack(
            (
                logp(components[0], mix_eval).eval(),
                logp(components[1], mix_eval).eval(),
            ),
            axis=-1,
        )[bcast_weights == 1]
        expected_logp = expected_logp.reshape(expected_shape)
        assert np.allclose(mix_logp_eval, expected_logp)

    @pytest.mark.parametrize(
        "weights",
        [
            np.array([1, 0]),
            np.array([[1, 0], [0, 1], [1, 0], [0, 1]]),
        ],
    )
    @pytest.mark.parametrize(
        "component",
        [
            DirichletMultinomial.dist(n=[5_000, 10_000], a=np.ones((3,))),
            DirichletMultinomial.dist(n=[5_000, 10_000], a=np.ones((3,)), size=(4, 2)),
        ],
    )
    @pytest.mark.parametrize("size", [None, (4,), (5, 4)])
    def test_single_multivariate_component_deterministic_weights(self, weights, component, size):
        # This test needs seeding to avoid repetitions
        rngs = [
            aesara.shared(np.random.default_rng(seed))
            for seed in self.get_random_state().randint(2**30, size=2)
        ]
        mix = Mixture.dist(weights, component, size=size, rngs=rngs)
        mix_eval = mix.eval()

        # Test shape
        # component shape is either (4, 2, 3), (2, 3)
        # weights shape is either (4, 2) or (2,)
        if size is not None:
            expected_shape = size + (3,)
        elif component.ndim == 3 or weights.ndim == 2:
            expected_shape = (4, 3)
        else:
            expected_shape = (3,)
        assert mix_eval.shape == expected_shape

        # Test draws
        totals = mix_eval.sum(-1)
        expected_large_count = (weights == 1)[..., 1]
        assert np.all((totals == 10_000) == expected_large_count)
        repetitions = np.unique(mix_eval[..., 0]).size < totals.size
        assert not repetitions

        # Test logp
        mix_logp_eval = logp(mix, mix_eval).eval()
        expected_logp_shape = expected_shape[:-1]
        assert mix_logp_eval.shape == expected_logp_shape
        bcast_weights = np.broadcast_to(weights, (*expected_logp_shape, 2))
        expected_logp = logp(component, mix_eval[..., None, :]).eval()[bcast_weights == 1]
        expected_logp = expected_logp.reshape(expected_logp_shape)
        assert np.allclose(mix_logp_eval, expected_logp)

    @pytest.mark.parametrize(
        "weights",
        [
            np.array([1, 0]),
            np.array([[1, 0], [0, 1], [1, 0], [0, 1]]),
        ],
    )
    @pytest.mark.parametrize(
        "components",
        [
            (
                MvNormal.dist([-15, -10, -5], np.eye(3) * 1e-3),
                MvNormal.dist([15, 10, 5], np.eye(3) * 1e-3),
            ),
            (
                MvNormal.dist([-15, -10, -5], np.eye(3) * 1e-3, size=(4,)),
                MvNormal.dist([15, 10, 5], np.eye(3) * 1e-3, size=(4,)),
            ),
        ],
    )
    @pytest.mark.parametrize("size", [None, (4,), (5, 4)])
    def test_list_multivariate_components_deterministic_weights(self, weights, components, size):
        mix = Mixture.dist(weights, components, size=size)
        mix_eval = mix.eval()

        # Test shape
        # components[0] shape is either (4, 3) or (3,)
        # weights shape is either (4, 2) or (2,)
        if size is not None:
            expected_shape = size + (3,)
        elif components[0].ndim == 2 or weights.ndim == 2:
            expected_shape = (4, 3)
        else:
            expected_shape = (3,)
        assert mix_eval.shape == expected_shape

        # Test draws
        expected_positive = np.zeros_like(mix_eval)
        expected_positive[..., :] = (weights == 1)[..., 1, None]
        assert np.all((mix_eval > 0) == expected_positive)
        repetitions = np.unique(mix_eval).size < mix_eval.size
        assert not repetitions

        # Test logp
        # MvNormal logp is currently limited to 2d values
        expectation = pytest.raises(ValueError) if mix_eval.ndim > 2 else does_not_raise()
        with expectation:
            mix_logp_eval = logp(mix, mix_eval).eval()
            assert mix_logp_eval.shape == expected_shape[:-1]
            bcast_weights = np.broadcast_to(weights, (*expected_shape[:-1], 2))
            expected_logp = np.stack(
                (
                    logp(components[0], mix_eval).eval(),
                    logp(components[1], mix_eval).eval(),
                ),
                axis=-1,
            )[bcast_weights == 1]
            expected_logp = expected_logp.reshape(expected_shape[:-1])
            assert np.allclose(mix_logp_eval, expected_logp)

    def test_component_choice_random(self):
        """Test that mixture choices change over evaluations"""
        with Model() as m:
            weights = [0.5, 0.5]
            components = [Normal.dist(-10, 0.01), Normal.dist(10, 0.01)]
            mix = Mixture.dist(weights, components)
        draws = draw(mix, draws=10)
        # Probability of coming from same component 10 times is 0.5**10
        assert np.unique(draws > 0).size == 2

    @pytest.mark.parametrize(
        "comp_dists",
        (
            [Normal.dist(size=(2,))],
            [Normal.dist(), Normal.dist()],
            [MvNormal.dist(np.ones(3), np.eye(3), size=(2,))],
            [
                MvNormal.dist(np.ones(3), np.eye(3)),
                MvNormal.dist(np.ones(3), np.eye(3)),
            ],
        ),
    )
    def test_components_expanded_by_weights(self, comp_dists):
        """Test that components are expanded when size or weights are larger than components"""
        univariate = comp_dists[0].owner.op.ndim_supp == 0

        mix = Mixture.dist(
            w=Dirichlet.dist([1, 1], shape=(3, 2)),
            comp_dists=comp_dists,
            size=(3,),
        )
        draws = mix.eval()
        assert draws.shape == (3,) if univariate else (3, 3)
        assert np.unique(draws).size == draws.size

        mix = Mixture.dist(
            w=Dirichlet.dist([1, 1], shape=(4, 3, 2)),
            comp_dists=comp_dists,
            size=(3,),
        )
        draws = mix.eval()
        assert draws.shape == (4, 3) if univariate else (4, 3, 3)
        assert np.unique(draws).size == draws.size

    @pytest.mark.parametrize(
        "comp_dists",
        (
            [Normal.dist(size=(2,))],
            [Normal.dist(), Normal.dist()],
            [MvNormal.dist(np.ones(3), np.eye(3), size=(2,))],
            [
                MvNormal.dist(np.ones(3), np.eye(3)),
                MvNormal.dist(np.ones(3), np.eye(3)),
            ],
        ),
    )
    @pytest.mark.parametrize("expand", (False, True))
    def test_change_size(self, comp_dists, expand):
        univariate = comp_dists[0].owner.op.ndim_supp == 0

        mix = Mixture.dist(w=Dirichlet.dist([1, 1]), comp_dists=comp_dists)
        mix = Mixture.change_size(mix, new_size=(4,), expand=expand)
        draws = mix.eval()
        expected_shape = (4,) if univariate else (4, 3)
        assert draws.shape == expected_shape
        assert np.unique(draws).size == draws.size

        mix = Mixture.dist(w=Dirichlet.dist([1, 1]), comp_dists=comp_dists, size=(3,))
        mix = Mixture.change_size(mix, new_size=(5, 4), expand=expand)
        draws = mix.eval()
        expected_shape = (5, 4) if univariate else (5, 4, 3)
        if expand:
            expected_shape = expected_shape + (3,)
        assert draws.shape == expected_shape
        assert np.unique(draws).size == draws.size

    def test_list_mvnormals_logp(self):
        mu1 = np.asarray([0.0, 1.0])
        cov1 = np.diag([1.5, 2.5])
        mu2 = np.asarray([1.0, 0.0])
        cov2 = np.diag([2.5, 3.5])
        obs = np.asarray([[0.5, 0.5], mu1, mu2])
        with Model() as model:
            w = Dirichlet("w", floatX(np.ones(2)), transform=None, shape=(2,))
            mvncomp1 = MvNormal.dist(mu=mu1, cov=cov1)
            mvncomp2 = MvNormal.dist(mu=mu2, cov=cov2)
            y = Mixture("x_obs", w, [mvncomp1, mvncomp2], observed=obs)

        # check logp of each component
        complogp_st = np.vstack(
            (
                st.multivariate_normal.logpdf(obs, mu1, cov1),
                st.multivariate_normal.logpdf(obs, mu2, cov2),
            )
        ).T

        # check logp of mixture
        testpoint = model.compute_initial_point()
        mixlogp_st = logsumexp(np.log(testpoint["w"]) + complogp_st, axis=-1, keepdims=False)
        assert_allclose(model.compile_logp(y, sum=False)(testpoint)[0], mixlogp_st)

        # check logp of model
        priorlogp = st.dirichlet.logpdf(
            x=testpoint["w"],
            alpha=np.ones(2),
        )
        assert_allclose(model.compile_logp()(testpoint), mixlogp_st.sum() + priorlogp)

    def test_single_poisson_sampling(self):
        pois_w = np.array([0.4, 0.6])
        pois_mu = np.array([5.0, 20.0])
        pois_x = generate_poisson_mixture_data(pois_w, pois_mu, size=1000)

        with Model() as model:
            w = Dirichlet("w", floatX(np.ones_like(pois_w)), shape=pois_w.shape)
            mu = Gamma("mu", 1.0, 1.0, shape=pois_w.size)
            Mixture("x_obs", w, Poisson.dist(mu), observed=pois_x)
            step = Metropolis()
            trace = sample(
                5000,
                step,
                random_seed=self.random_seed,
                progressbar=False,
                chains=1,
                return_inferencedata=False,
            )

        assert_allclose(np.sort(trace["w"].mean(axis=0)), np.sort(pois_w), rtol=0.1, atol=0.1)
        assert_allclose(np.sort(trace["mu"].mean(axis=0)), np.sort(pois_mu), rtol=0.1, atol=0.1)

    def test_list_poissons_sampling(self):
        pois_w = np.array([0.4, 0.6])
        pois_mu = np.array([5.0, 20.0])
        pois_x = generate_poisson_mixture_data(pois_w, pois_mu, size=1000)

        with Model() as model:
            w = Dirichlet("w", floatX(np.ones_like(pois_w)), shape=pois_w.shape)
            mu = Gamma("mu", 1.0, 1.0, shape=pois_w.size)
            Mixture("x_obs", w, [Poisson.dist(mu[0]), Poisson.dist(mu[1])], observed=pois_x)
            trace = sample(
                5000,
                chains=1,
                step=Metropolis(),
                random_seed=self.random_seed,
                progressbar=False,
                return_inferencedata=False,
            )

        assert_allclose(np.sort(trace["w"].mean(axis=0)), np.sort(pois_w), rtol=0.1, atol=0.1)
        assert_allclose(np.sort(trace["mu"].mean(axis=0)), np.sort(pois_mu), rtol=0.1, atol=0.1)

    def test_list_normals_sampling(self):
        norm_w = np.array([0.75, 0.25])
        norm_mu = np.array([0.0, 5.0])
        norm_sigma = np.ones_like(norm_mu)
        norm_x = generate_normal_mixture_data(norm_w, norm_mu, norm_sigma, size=1000)

        with Model() as model:
            w = Dirichlet("w", floatX(np.ones_like(norm_w)), shape=norm_w.size)
            mu = Normal("mu", 0.0, 10.0, shape=norm_w.size)
            tau = Gamma("tau", 1.0, 1.0, shape=norm_w.size)
            Mixture(
                "x_obs",
                w,
                [Normal.dist(mu[0], tau=tau[0]), Normal.dist(mu[1], tau=tau[1])],
                observed=norm_x,
            )
            trace = sample(
                5000,
                chains=1,
                step=Metropolis(),
                random_seed=self.random_seed,
                progressbar=False,
                return_inferencedata=False,
            )

        assert_allclose(np.sort(trace["w"].mean(axis=0)), np.sort(norm_w), rtol=0.1, atol=0.1)
        assert_allclose(np.sort(trace["mu"].mean(axis=0)), np.sort(norm_mu), rtol=0.1, atol=0.1)

    def test_single_poisson_predictive_sampling_shape(self):
        # test the shape broadcasting in mixture random
        rng = self.get_random_state()
        y = np.concatenate([rng.poisson(5, size=10), rng.poisson(9, size=10)])
        with Model() as model:
            comp0 = Poisson.dist(mu=np.ones(2))
            w0 = Dirichlet("w0", a=np.ones(2), shape=(2,))
            like0 = Mixture("like0", w=w0, comp_dists=comp0, observed=y)

            comp1 = Poisson.dist(mu=np.ones((20, 2)), shape=(20, 2))
            w1 = Dirichlet("w1", a=np.ones(2), shape=(2,))
            like1 = Mixture("like1", w=w1, comp_dists=comp1, observed=y)

            comp2 = Poisson.dist(mu=np.ones(2))
            w2 = Dirichlet("w2", a=np.ones(2), shape=(20, 2))
            like2 = Mixture("like2", w=w2, comp_dists=comp2, observed=y)

            comp3 = Poisson.dist(mu=np.ones(2), shape=(20, 2))
            w3 = Dirichlet("w3", a=np.ones(2), shape=(20, 2))
            like3 = Mixture("like3", w=w3, comp_dists=comp3, observed=y)

        n_samples = 30
        with model:
            prior = sample_prior_predictive(samples=n_samples, return_inferencedata=False)
            ppc = sample_posterior_predictive(
                [self.get_inital_point(model)], samples=n_samples, return_inferencedata=False
            )

        assert prior["like0"].shape == (n_samples, 20)
        assert prior["like1"].shape == (n_samples, 20)
        assert prior["like2"].shape == (n_samples, 20)
        assert prior["like3"].shape == (n_samples, 20)

        assert ppc["like0"].shape == (n_samples, 20)
        assert ppc["like1"].shape == (n_samples, 20)
        assert ppc["like2"].shape == (n_samples, 20)
        assert ppc["like3"].shape == (n_samples, 20)

    def test_list_mvnormals_predictive_sampling_shape(self):
        N = 100  # number of data points
        K = 3  # number of mixture components
        D = 3  # dimensionality of the data
        X = MvNormal.dist(np.zeros(D), np.eye(D), size=N).eval()

        with Model() as model:
            pi = Dirichlet("pi", np.ones(K), shape=(K,))

            comp_dist = []
            mu = []
            packed_chol = []
            chol = []
            for i in range(K):
                mu.append(Normal(f"mu{i}", 0, 10, shape=D))
                packed_chol.append(
                    LKJCholeskyCov(
                        f"chol_cov_{i}",
                        eta=2,
                        n=D,
                        sd_dist=HalfNormal.dist(2.5, size=D),
                        compute_corr=False,
                    )
                )
                chol.append(expand_packed_triangular(D, packed_chol[i], lower=True))
                comp_dist.append(MvNormal.dist(mu=mu[i], chol=chol[i], shape=D))

            Mixture("x_obs", pi, comp_dist, observed=X)

        n_samples = 20
        with model:
            prior = sample_prior_predictive(samples=n_samples, return_inferencedata=False)
            ppc = sample_posterior_predictive(
                [self.get_inital_point(model)], samples=n_samples, return_inferencedata=False
            )
        assert ppc["x_obs"].shape == (n_samples,) + X.shape
        assert prior["x_obs"].shape == (n_samples,) + X.shape
        assert prior["mu0"].shape == (n_samples, D)
        assert prior["chol_cov_0"].shape == (n_samples, D * (D + 1) // 2)

    @pytest.mark.xfail(reason="Mixture from single component not refactored yet")
    def test_nested_mixture(self):
        if aesara.config.floatX == "float32":
            rtol = 1e-4
        else:
            rtol = 1e-7
        nbr = 4
        with Model() as model:
            # mixtures components
            g_comp = Normal.dist(
                mu=Exponential("mu_g", lam=1.0, shape=nbr, transform=None), sigma=1, shape=nbr
            )
            l_comp = LogNormal.dist(
                mu=Exponential("mu_l", lam=1.0, shape=nbr, transform=None), sigma=1, shape=nbr
            )
            # weight vector for the mixtures
            g_w = Dirichlet("g_w", a=floatX(np.ones(nbr) * 0.0000001), transform=None, shape=(nbr,))
            l_w = Dirichlet("l_w", a=floatX(np.ones(nbr) * 0.0000001), transform=None, shape=(nbr,))
            # mixture components
            g_mix = Mixture.dist(w=g_w, comp_dists=g_comp)
            l_mix = Mixture.dist(w=l_w, comp_dists=l_comp)
            # mixture of mixtures
            mix_w = Dirichlet("mix_w", a=floatX(np.ones(2)), transform=None, shape=(2,))
            mix = Mixture("mix", w=mix_w, comp_dists=[g_mix, l_mix], observed=np.exp(self.norm_x))

        test_point = model.compute_initial_point()

        def mixmixlogp(value, point):
            floatX = aesara.config.floatX
            priorlogp = (
                st.dirichlet.logpdf(
                    x=point["g_w"],
                    alpha=np.ones(nbr) * 0.0000001,
                ).astype(floatX)
                + st.expon.logpdf(x=point["mu_g"]).sum(dtype=floatX)
                + st.dirichlet.logpdf(
                    x=point["l_w"],
                    alpha=np.ones(nbr) * 0.0000001,
                ).astype(floatX)
                + st.expon.logpdf(x=point["mu_l"]).sum(dtype=floatX)
                + st.dirichlet.logpdf(
                    x=point["mix_w"],
                    alpha=np.ones(2),
                ).astype(floatX)
            )
            complogp1 = st.norm.logpdf(x=value, loc=point["mu_g"]).astype(floatX)
            mixlogp1 = logsumexp(
                np.log(point["g_w"]).astype(floatX) + complogp1, axis=-1, keepdims=True
            )
            complogp2 = st.lognorm.logpdf(value, 1.0, 0.0, np.exp(point["mu_l"])).astype(floatX)
            mixlogp2 = logsumexp(
                np.log(point["l_w"]).astype(floatX) + complogp2, axis=-1, keepdims=True
            )
            complogp_mix = np.concatenate((mixlogp1, mixlogp2), axis=1)
            mixmixlogpg = logsumexp(
                np.log(point["mix_w"]).astype(floatX) + complogp_mix, axis=-1, keepdims=False
            )
            return priorlogp, mixmixlogpg

        value = np.exp(self.norm_x)[:, None]
        priorlogp, mixmixlogpg = mixmixlogp(value, test_point)

        # check logp of mixture
        assert_allclose(mixmixlogpg, mix.logp_elemwise(test_point), rtol=rtol)

        # check model logp
        assert_allclose(priorlogp + mixmixlogpg.sum(), model.logp(test_point), rtol=rtol)

        # check input and check logp again
        test_point["g_w"] = np.asarray([0.1, 0.1, 0.2, 0.6])
        test_point["mu_g"] = np.exp(np.random.randn(nbr))
        priorlogp, mixmixlogpg = mixmixlogp(value, test_point)
        assert_allclose(mixmixlogpg, mix.logp_elemwise(test_point), rtol=rtol)
        assert_allclose(priorlogp + mixmixlogpg.sum(), model.logp(test_point), rtol=rtol)

    def test_iterable_single_component_warning(self):
        with pytest.warns(None) as record:
            Mixture.dist(w=[0.5, 0.5], comp_dists=Normal.dist(size=2))
            Mixture.dist(w=[0.5, 0.5], comp_dists=[Normal.dist(size=2), Normal.dist(size=2)])
        assert not record

        with pytest.warns(UserWarning, match="Single component will be treated as a mixture"):
            Mixture.dist(w=[0.5, 0.5], comp_dists=[Normal.dist(size=2)])

    @pytest.mark.parametrize("floatX", ["float32", "float64"])
    def test_mixture_dtype(self, floatX):
        with aesara.config.change_flags(floatX=floatX):
            mix_dtype = Mixture.dist(
                w=[0.5, 0.5],
                comp_dists=[
                    Multinomial.dist(n=5, p=[0.5, 0.5]),
                    Multinomial.dist(n=5, p=[0.5, 0.5]),
                ],
            ).dtype
            assert mix_dtype == "int64"

            mix_dtype = Mixture.dist(
                w=[0.5, 0.5],
                comp_dists=[
                    Dirichlet.dist(a=[0.5, 0.5]),
                    Dirichlet.dist(a=[0.5, 0.5]),
                ],
            ).dtype
            assert mix_dtype == floatX

    @pytest.mark.parametrize(
        "comp_dists, expected_shape",
        [
            (
                [
                    Normal.dist([[0, 0, 0], [0, 0, 0]]),
                    Normal.dist([0, 0, 0]),
                    Normal.dist([0]),
                ],
                (2, 3),
            ),
            (
                [
                    Dirichlet.dist([[1, 1, 1], [1, 1, 1]]),
                    Dirichlet.dist([1, 1, 1]),
                ],
                (2, 3),
            ),
        ],
    )
    def test_broadcast_components(self, comp_dists, expected_shape):
        n_dists = len(comp_dists)
        mix = Mixture.dist(w=np.ones(n_dists) / n_dists, comp_dists=comp_dists)
        mix_eval = mix.eval()
        assert tuple(mix_eval.shape) == expected_shape
        assert np.unique(mix_eval).size == mix.eval().size
        for comp_dist in mix.owner.inputs[2:]:
            # We check that the input is a "pure" RandomVariable and not a broadcast
            # operation. This confirms that all draws will be unique
            assert isinstance(comp_dist.owner.op, RandomVariable)
            assert tuple(comp_dist.shape.eval()) == expected_shape

    def test_preventing_mixing_cont_and_discrete(self):
        with pytest.raises(
            ValueError,
            match="All distributions in comp_dists must be either discrete or continuous.",
        ):
            with Model() as model:
                mix = Mixture(
                    "x",
                    w=[0.5, 0.3, 0.2],
                    comp_dists=[
                        Categorical.dist(np.tile(1 / 3, 3)),
                        Normal.dist(np.ones(3), 3),
                    ],
                )


class TestNormalMixture(SeededTest):
    def test_normal_mixture_sampling(self):
        norm_w = np.array([0.75, 0.25])
        norm_mu = np.array([0.0, 5.0])
        norm_sigma = np.ones_like(norm_mu)
        norm_x = generate_normal_mixture_data(norm_w, norm_mu, norm_sigma, size=1000)

        with Model() as model:
            w = Dirichlet("w", floatX(np.ones_like(norm_w)), shape=norm_w.size)
            mu = Normal("mu", 0.0, 10.0, shape=norm_w.size)
            tau = Gamma("tau", 1.0, 1.0, shape=norm_w.size)
            NormalMixture("x_obs", w, mu, tau=tau, observed=norm_x)
            step = Metropolis()
            trace = sample(
                5000,
                step,
                random_seed=self.random_seed,
                progressbar=False,
                chains=1,
                return_inferencedata=False,
            )

        assert_allclose(np.sort(trace["w"].mean(axis=0)), np.sort(norm_w), rtol=0.1, atol=0.1)
        assert_allclose(np.sort(trace["mu"].mean(axis=0)), np.sort(norm_mu), rtol=0.1, atol=0.1)

    @pytest.mark.parametrize(
        "nd, ncomp", [(tuple(), 5), (1, 5), (3, 5), ((3, 3), 5), (3, 3), ((3, 3), 3)], ids=str
    )
    def test_normal_mixture_nd(self, nd, ncomp):
        nd = to_tuple(nd)
        ncomp = int(ncomp)
        comp_shape = nd + (ncomp,)
        test_mus = np.random.randn(*comp_shape)
        test_taus = np.random.gamma(1, 1, size=comp_shape)
        observed = generate_normal_mixture_data(
            w=np.ones(ncomp) / ncomp, mu=test_mus, sigma=1 / np.sqrt(test_taus), size=10
        )

        with Model() as model0:
            mus = Normal("mus", shape=comp_shape)
            taus = Gamma("taus", alpha=1, beta=1, shape=comp_shape)
            ws = Dirichlet("ws", np.ones(ncomp), shape=(ncomp,))
            mixture0 = NormalMixture("m", w=ws, mu=mus, tau=taus, shape=nd, comp_shape=comp_shape)
            obs0 = NormalMixture(
                "obs", w=ws, mu=mus, tau=taus, comp_shape=comp_shape, observed=observed
            )

        with Model() as model1:
            mus = Normal("mus", shape=comp_shape)
            taus = Gamma("taus", alpha=1, beta=1, shape=comp_shape)
            ws = Dirichlet("ws", np.ones(ncomp), shape=(ncomp,))
            comp_dist = [
                Normal.dist(mu=mus[..., i], tau=taus[..., i], shape=nd) for i in range(ncomp)
            ]
            mixture1 = Mixture("m", w=ws, comp_dists=comp_dist, shape=nd)
            obs1 = Mixture("obs", w=ws, comp_dists=comp_dist, observed=observed)

        with Model() as model2:
            # Test that results are correct without comp_shape being passed to the Mixture.
            # This used to fail in V3
            mus = Normal("mus", shape=comp_shape)
            taus = Gamma("taus", alpha=1, beta=1, shape=comp_shape)
            ws = Dirichlet("ws", np.ones(ncomp), shape=(ncomp,))
            mixture2 = NormalMixture("m", w=ws, mu=mus, tau=taus, shape=nd)
            obs2 = NormalMixture("obs", w=ws, mu=mus, tau=taus, observed=observed)

        testpoint = model0.compute_initial_point()
        testpoint["mus"] = test_mus
        testpoint["taus_log__"] = np.log(test_taus)
        for logp0, logp1, logp2 in zip(
            model0.compile_logp(vars=[mixture0, obs0], sum=False)(testpoint),
            model1.compile_logp(vars=[mixture1, obs1], sum=False)(testpoint),
            model2.compile_logp(vars=[mixture2, obs2], sum=False)(testpoint),
        ):
            assert_allclose(logp0, logp1)
            assert_allclose(logp0, logp2)

    def test_random(self):
        def ref_rand(size, w, mu, sigma):
            component = np.random.choice(w.size, size=size, p=w)
            return np.random.normal(mu[component], sigma[component], size=size)

        pymc_random(
            NormalMixture,
            {
                "w": Simplex(2),
                "mu": Domain([[0.05, 2.5], [-5.0, 1.0]], edges=(None, None)),
                "sigma": Domain([[1, 1], [1.5, 2.0]], edges=(None, None)),
            },
            extra_args={"comp_shape": 2},
            size=1000,
            ref_rand=ref_rand,
            change_rv_size_fn=Mixture.change_size,
        )
        pymc_random(
            NormalMixture,
            {
                "w": Simplex(3),
                "mu": Domain([[-5.0, 1.0, 2.5]], edges=(None, None)),
                "sigma": Domain([[1.5, 2.0, 3.0]], edges=(None, None)),
            },
            extra_args={"comp_shape": 3},
            size=1000,
            ref_rand=ref_rand,
            change_rv_size_fn=Mixture.change_size,
        )


class TestMixtureVsLatent(SeededTest):
    """This class contains tests that compare a marginal Mixture with a latent indexed Mixture"""

    def test_scalar_components(self):
        nd = 3
        npop = 4
        # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        mus = at.constant(np.full((nd, npop), np.arange(npop)))

        with Model(rng_seeder=self.get_random_state()) as model:
            m = NormalMixture(
                "m",
                w=np.ones(npop) / npop,
                mu=mus,
                sigma=1e-5,
                comp_shape=(nd, npop),
                shape=nd,
            )
            z = Categorical("z", p=np.ones(npop) / npop, shape=nd)
            mu = at.as_tensor_variable([mus[i, z[i]] for i in range(nd)])
            latent_m = Normal("latent_m", mu=mu, sigma=1e-5, shape=nd)

        size = 100
        m_val = draw(m, draws=size)
        latent_m_val = draw(latent_m, draws=size)

        assert m_val.shape == latent_m_val.shape
        # Test that each element in axis = -1 can come from independent
        # components
        assert not all(np.all(np.diff(m_val) < 1e-3, axis=-1))
        assert not all(np.all(np.diff(latent_m_val) < 1e-3, axis=-1))
        self.samples_from_same_distribution(m_val, latent_m_val)

        # Check that logp is the same whether elements of the last axis are mixed or not
        logp_fn = model.compile_logp(vars=[m])
        assert np.isclose(logp_fn({"m": [0, 0, 0]}), logp_fn({"m": [0, 1, 2]}))
        self.logp_matches(m, latent_m, z, npop, model=model)

    def test_vector_components(self):
        nd = 3
        npop = 4
        # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        mus = at.constant(np.full((nd, npop), np.arange(npop)))

        with Model(rng_seeder=self.get_random_state()) as model:
            m = Mixture(
                "m",
                w=np.ones(npop) / npop,
                # MvNormal distribution with squared sigma diagonal covariance should
                # be equal to vector of Normals from latent_m
                comp_dists=[MvNormal.dist(mus[:, i], np.eye(nd) * 1e-5**2) for i in range(npop)],
            )
            z = Categorical("z", p=np.ones(npop) / npop)
            latent_m = Normal("latent_m", mu=mus[..., z], sigma=1e-5, shape=nd)

        size = 100
        m_val = draw(m, draws=size)
        latent_m_val = draw(latent_m, draws=size)
        assert m_val.shape == latent_m_val.shape
        # Test that each element in axis = -1 comes from the same mixture
        # component
        assert np.all(np.diff(m_val) < 1e-3)
        assert np.all(np.diff(latent_m_val) < 1e-3)
        # TODO: The following statistical test appears to be more flaky than expected
        #  even though the  distributions should be the same. Seeding should make it
        #  stable but might be worth investigating further
        self.samples_from_same_distribution(m_val, latent_m_val)

        # Check that mixing of values in the last axis leads to smaller logp
        logp_fn = model.compile_logp(vars=[m])
        assert logp_fn({"m": [0, 0, 0]}) > logp_fn({"m": [0, 1, 0]}) > logp_fn({"m": [0, 1, 2]})
        self.logp_matches(m, latent_m, z, npop, model=model)

    def samples_from_same_distribution(self, *args):
        # Test if flattened samples distributions match (marginals match)
        _, p_marginal = st.ks_2samp(*(s.flatten() for s in args))
        # Test if correlations within non independent draws match
        _, p_correlation = st.ks_2samp(
            *(np.array([np.corrcoef(ss) for ss in s]).flatten() for s in args)
        )
        # This has a success rate of 10% (0.95**2), even if the distributions are the same
        assert p_marginal >= 0.05 and p_correlation >= 0.05

    def logp_matches(self, mixture, latent_mix, z, npop, model):
        def loose_logp(model, vars):
            """Return logp function that accepts dictionary with unused variables as input"""
            return model.compile_fn(
                model.logpt(vars=vars, sum=False),
                inputs=model.value_vars,
                on_unused_input="ignore",
            )

        if aesara.config.floatX == "float32":
            rtol = 1e-4
        else:
            rtol = 1e-7
        test_point = model.compute_initial_point()
        test_point["m"] = test_point["latent_m"]

        mix_logp = loose_logp(model, mixture)(test_point)[0]

        z_shape = z.shape.eval()
        latent_mix_components_logps = []
        for component in range(npop):
            test_point["z"] = np.full(z_shape, component)
            z_logp = loose_logp(model, z)(test_point)[0]
            latent_mix_component_logp = loose_logp(model, latent_mix)(test_point)[0]
            # If the mixture ndim_supp is a vector, the logp should be summed within
            # components, as its items are not independent
            if mix_logp.ndim == 0:
                latent_mix_component_logp = latent_mix_component_logp.sum()
            latent_mix_components_logps.append(z_logp + latent_mix_component_logp)
        latent_mix_logp = logsumexp(np.array(latent_mix_components_logps), axis=0)
        if mix_logp.ndim == 0:
            latent_mix_logp = latent_mix_logp.sum()

        assert_allclose(mix_logp, latent_mix_logp, rtol=rtol)


class TestMixtureSameFamily(SeededTest):
    """Tests that used to belong to deprecated `TestMixtureSameFamily`.

    The functionality is now expected to be provided by `Mixture`
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.size = 50
        cls.n_samples = 1000
        cls.mixture_comps = 10

    @pytest.mark.parametrize("batch_shape", [(3, 4), (20,)], ids=str)
    def test_with_multinomial(self, batch_shape):
        p = np.random.uniform(size=(*batch_shape, self.mixture_comps, 3))
        n = 100 * np.ones((*batch_shape, 1))
        w = np.ones(self.mixture_comps) / self.mixture_comps
        mixture_axis = len(batch_shape)
        with Model() as model:
            comp_dists = Multinomial.dist(p=p, n=n, shape=(*batch_shape, self.mixture_comps, 3))
            mixture = Mixture(
                "mixture",
                w=w,
                comp_dists=comp_dists,
                shape=(*batch_shape, 3),
            )
            prior = sample_prior_predictive(samples=self.n_samples, return_inferencedata=False)

        assert prior["mixture"].shape == (self.n_samples, *batch_shape, 3)
        assert draw(mixture, draws=self.size).shape == (self.size, *batch_shape, 3)

        if aesara.config.floatX == "float32":
            rtol = 1e-4
        else:
            rtol = 1e-7

        initial_point = model.compute_initial_point()
        comp_logp = logp(comp_dists, initial_point["mixture"].reshape(*batch_shape, 1, 3))
        log_sum_exp = logsumexp(
            comp_logp.eval() + np.log(w), axis=mixture_axis, keepdims=True
        ).sum()
        assert_allclose(
            model.compile_logp()(initial_point),
            log_sum_exp,
            rtol,
        )

    def test_with_mvnormal(self):
        # 10 batch, 3-variate Gaussian
        mu = np.random.randn(self.mixture_comps, 3)
        mat = np.random.randn(3, 3)
        cov = mat @ mat.T
        chol = np.linalg.cholesky(cov)
        w = np.ones(self.mixture_comps) / self.mixture_comps

        with Model() as model:
            comp_dists = MvNormal.dist(mu=mu, chol=chol, shape=(self.mixture_comps, 3))
            mixture = Mixture("mixture", w=w, comp_dists=comp_dists, shape=(3,))
            prior = sample_prior_predictive(samples=self.n_samples, return_inferencedata=False)

        assert prior["mixture"].shape == (self.n_samples, 3)
        assert draw(mixture, draws=self.size).shape == (self.size, 3)

        if aesara.config.floatX == "float32":
            rtol = 1e-4
        else:
            rtol = 1e-7

        initial_point = model.compute_initial_point()
        comp_logp = logp(comp_dists, initial_point["mixture"].reshape(1, 3))
        log_sum_exp = logsumexp(comp_logp.eval() + np.log(w), axis=0, keepdims=True).sum()
        assert_allclose(
            model.compile_logp()(initial_point),
            log_sum_exp,
            rtol,
        )

    def test_broadcasting_in_shape(self):
        with Model() as model:
            mu = Gamma("mu", 1.0, 1.0, shape=2)
            comp_dists = Poisson.dist(mu, shape=2)
            mix = Mixture("mix", w=np.ones(2) / 2, comp_dists=comp_dists, shape=(1000,))
            prior = sample_prior_predictive(samples=self.n_samples, return_inferencedata=False)

        assert prior["mix"].shape == (self.n_samples, 1000)


class TestMixtureMoments:
    @pytest.mark.parametrize(
        "weights, comp_dists, size, expected",
        [
            (
                np.array([0.4, 0.6]),
                Normal.dist(mu=np.array([-2, 6]), sigma=np.array([5, 3])),
                None,
                2.8,
            ),
            (
                np.tile(1 / 13, 13),
                Normal.dist(-2, 1, size=(13,)),
                (3,),
                np.full((3,), -2),
            ),
            (
                np.array([0.4, 0.6]),
                Normal.dist([-2, 6], 3),
                (5, 3),
                np.full((5, 3), 2.8),
            ),
            (
                np.broadcast_to(np.array([0.4, 0.6]), (5, 3, 2)),
                Normal.dist(np.array([-2, 6]), np.array([5, 3])),
                None,
                np.full(shape=(5, 3), fill_value=2.8),
            ),
            (
                np.array([0.4, 0.6]),
                Normal.dist(np.array([-2, 6]), np.array([5, 3]), size=(5, 3, 2)),
                None,
                np.full(shape=(5, 3), fill_value=2.8),
            ),
            (
                np.array([[0.8, 0.2], [0.2, 0.8]]),
                Normal.dist(np.array([-2, 6])),
                None,
                np.array([-0.4, 4.4]),
            ),
            # implied size = (11, 7) will be overwritten by (5, 3)
            (
                np.array([0.4, 0.6]),
                Normal.dist(np.array([-2, 6]), np.array([5, 3]), size=(11, 7, 2)),
                (5, 3),
                np.full(shape=(5, 3), fill_value=2.8),
            ),
        ],
    )
    def test_single_univariate_component(self, weights, comp_dists, size, expected):
        with Model() as model:
            Mixture("x", weights, comp_dists, size=size)
        assert_moment_is_expected(model, expected, check_finite_logp=False)

    @pytest.mark.parametrize(
        "weights, comp_dists, size, expected",
        [
            (
                np.array([1, 0]),
                [Normal.dist(-2, 5), Normal.dist(6, 3)],
                None,
                -2,
            ),
            (
                np.array([0.4, 0.6]),
                [Normal.dist(-2, 5, size=(2,)), Normal.dist(6, 3, size=(2,))],
                None,
                np.full((2,), 2.8),
            ),
            (
                np.array([0.5, 0.5]),
                [Normal.dist(-2, 5), Exponential.dist(lam=1 / 3)],
                (3, 5),
                np.full((3, 5), 0.5),
            ),
            (
                np.broadcast_to(np.array([0.4, 0.6]), (5, 3, 2)),
                [Normal.dist(-2, 5), Normal.dist(6, 3)],
                None,
                np.full(shape=(5, 3), fill_value=2.8),
            ),
            (
                np.array([[0.8, 0.2], [0.2, 0.8]]),
                [Normal.dist(-2, 5), Normal.dist(6, 3)],
                None,
                np.array([-0.4, 4.4]),
            ),
            (
                np.array([[0.8, 0.2], [0.2, 0.8]]),
                [Normal.dist(-2, 5), Normal.dist(6, 3)],
                (3, 2),
                np.full((3, 2), np.array([-0.4, 4.4])),
            ),
            (
                # implied size = (11, 7) will be overwritten by (5, 3)
                np.array([0.4, 0.6]),
                [Normal.dist(-2, 5, size=(11, 7)), Normal.dist(6, 3, size=(11, 7))],
                (5, 3),
                np.full(shape=(5, 3), fill_value=2.8),
            ),
        ],
    )
    def test_list_univariate_components(self, weights, comp_dists, size, expected):
        with Model() as model:
            Mixture("x", weights, comp_dists, size=size)
        assert_moment_is_expected(model, expected, check_finite_logp=False)

    @pytest.mark.parametrize(
        "weights, comp_dists, size, expected",
        [
            (
                np.array([0.4, 0.6]),
                MvNormal.dist(mu=np.array([[-1, -2], [3, 5]]), cov=np.eye(2) * 0.3),
                None,
                np.array([1.4, 2.2]),
            ),
            (
                np.array([0.5, 0.5]),
                Dirichlet.dist(a=np.array([[0.0001, 0.0001, 1000], [2, 4, 6]])),
                (4,),
                np.array(np.full((4, 3), [1 / 12, 1 / 6, 3 / 4])),
            ),
            (
                np.array([0.4, 0.6]),
                MvNormal.dist(mu=np.array([-10, 0, 10]), cov=np.eye(3) * 3, size=(4, 2)),
                None,
                np.full((4, 3), [-10, 0, 10]),
            ),
            (
                np.array([[1.0, 0], [0.0, 1.0]]),
                MvNormal.dist(
                    mu=np.array([[-5, -10, -15], [5, 10, 15]]), cov=np.eye(3) * 3, size=(2,)
                ),
                (3, 2),
                np.full((3, 2, 3), [[-5, -10, -15], [5, 10, 15]]),
            ),
        ],
    )
    def test_single_multivariate_component(self, weights, comp_dists, size, expected):
        with Model() as model:
            Mixture("x", weights, comp_dists, size=size)
        assert_moment_is_expected(model, expected, check_finite_logp=False)

    @pytest.mark.parametrize(
        "weights, comp_dists, size, expected",
        [
            (
                np.array([0.4, 0.6]),
                [
                    MvNormal.dist(mu=np.array([-1, -2]), cov=np.eye(2) * 0.3),
                    MvNormal.dist(mu=np.array([3, 5]), cov=np.eye(2) * 0.8),
                ],
                None,
                np.array([1.4, 2.2]),
            ),
            (
                np.array([0.4, 0.6]),
                [
                    Dirichlet.dist(a=np.array([2, 3, 5])),
                    MvNormal.dist(mu=np.array([-10, 0, 10]), cov=np.eye(3) * 3),
                ],
                (4,),
                np.array(np.full((4, 3), [-5.92, 0.12, 6.2])),
            ),
            (
                np.array([0.4, 0.6]),
                [
                    Dirichlet.dist(a=np.array([2, 3, 5]), size=(2,)),
                    MvNormal.dist(mu=np.array([-10, 0, 10]), cov=np.eye(3) * 3, size=(2,)),
                ],
                None,
                np.full((2, 3), [-5.92, 0.12, 6.2]),
            ),
            (
                np.array([[1.0, 0], [0.0, 1.0]]),
                [
                    MvNormal.dist(mu=np.array([-5, -10, -15]), cov=np.eye(3) * 3, size=(2,)),
                    MvNormal.dist(mu=np.array([5, 10, 15]), cov=np.eye(3) * 3, size=(2,)),
                ],
                (3, 2),
                np.full((3, 2, 3), [[-5, -10, -15], [5, 10, 15]]),
            ),
        ],
    )
    def test_list_multivariate_components(self, weights, comp_dists, size, expected):
        with Model() as model:
            Mixture("x", weights, comp_dists, size=size)
        assert_moment_is_expected(model, expected, check_finite_logp=False)


class TestMixtureDefaultTransforms:
    @pytest.mark.parametrize(
        "comp_dists, expected_transform_type",
        [
            (Poisson.dist(1, size=2), type(None)),
            (Normal.dist(size=2), type(None)),
            (Uniform.dist(size=2), IntervalTransform),
            (HalfNormal.dist(size=2), LogTransform),
            ([HalfNormal.dist(), Normal.dist()], type(None)),
            ([HalfNormal.dist(1), Exponential.dist(1), HalfStudentT.dist(4, 1)], LogTransform),
            ([Dirichlet.dist([1, 2, 3, 4]), StickBreakingWeights.dist(1, K=3)], SimplexTransform),
            ([Uniform.dist(0, 1), Uniform.dist(0, 1), Triangular.dist(0, 1)], IntervalTransform),
            ([Uniform.dist(0, 1), Uniform.dist(0, 2)], type(None)),
        ],
    )
    def test_expected(self, comp_dists, expected_transform_type):
        if isinstance(comp_dists, TensorVariable):
            weights = np.ones(2) / 2
        else:
            weights = np.ones(len(comp_dists)) / len(comp_dists)
        mix = Mixture.dist(weights, comp_dists)
        assert isinstance(_default_transform(mix.owner.op, mix), expected_transform_type)

    def test_hierarchical_interval_transform(self):
        with Model() as model:
            lower = Normal("lower", 0.5)
            upper = Uniform("upper", 0, 1)
            uniform = Uniform("uniform", -at.abs(lower), at.abs(upper), transform=None)
            triangular = Triangular(
                "triangular", -at.abs(lower), at.abs(upper), c=0.25, transform=None
            )
            comp_dists = [
                Uniform.dist(-at.abs(lower), at.abs(upper)),
                Triangular.dist(-at.abs(lower), at.abs(upper), c=0.25),
            ]
            mix1 = Mixture("mix1", [0.3, 0.7], comp_dists)
            mix2 = Mixture("mix2", [0.3, 0.7][::-1], comp_dists[::-1])

        ip = model.compute_initial_point()
        # We want an informative moment, other than zero
        assert ip["mix1_interval__"] != 0

        expected_mix_ip = (
            IntervalTransform(args_fn=lambda *args: (-0.5, 0.5))
            .forward(0.3 * ip["uniform"] + 0.7 * ip["triangular"])
            .eval()
        )
        assert np.isclose(ip["mix1_interval__"], ip["mix2_interval__"])
        assert np.isclose(ip["mix1_interval__"], expected_mix_ip)

    def test_logp(self):
        with Model() as m:
            halfnorm = HalfNormal("halfnorm")
            comp_dists = [HalfNormal.dist(), HalfNormal.dist()]
            mix_transf = Mixture("mix_transf", w=[0.5, 0.5], comp_dists=comp_dists)
            mix = Mixture("mix", w=[0.5, 0.5], comp_dists=comp_dists, transform=None)

        logp_fn = m.compile_logp(vars=[halfnorm, mix_transf, mix], sum=False)
        test_point = {"halfnorm_log__": 1, "mix_transf_log__": 1, "mix": np.exp(1)}
        logp_halfnorm, logp_mix_transf, logp_mix = logp_fn(test_point)
        assert np.isclose(logp_halfnorm, logp_mix_transf)
        assert np.isclose(logp_halfnorm, logp_mix + 1)

    def test_warning(self):
        with Model() as m:
            comp_dists = [HalfNormal.dist(), Exponential.dist(1)]
            with pytest.warns(None) as rec:
                Mixture("mix1", w=[0.5, 0.5], comp_dists=comp_dists)
            assert not rec

            comp_dists = [Uniform.dist(0, 1), Uniform.dist(0, 2)]
            with pytest.warns(MixtureTransformWarning):
                Mixture("mix2", w=[0.5, 0.5], comp_dists=comp_dists)

            comp_dists = [Normal.dist(), HalfNormal.dist()]
            with pytest.warns(MixtureTransformWarning):
                Mixture("mix3", w=[0.5, 0.5], comp_dists=comp_dists)

            with pytest.warns(None) as rec:
                Mixture("mix4", w=[0.5, 0.5], comp_dists=comp_dists, transform=None)
            assert not rec

            with pytest.warns(None) as rec:
                Mixture("mix5", w=[0.5, 0.5], comp_dists=comp_dists, observed=1)
            assert not rec
