#   Copyright 2024 The PyMC Developers
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
import logging
import warnings

import numpy as np
import numpy.random as npr
import numpy.testing as npt
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from arviz import InferenceData
from arviz import from_dict as az_from_dict
from arviz.tests.helpers import check_multiple_attrs
from pytensor import Mode, shared
from pytensor.compile import SharedVariable
from scipy import stats

import pymc as pm

from pymc.backends.base import MultiTrace
from pymc.pytensorf import compile_pymc
from pymc.sampling.forward import (
    compile_forward_sampling_function,
    get_vars_in_point_list,
    observed_dependent_deterministics,
)
from pymc.testing import fast_unstable_sampling_mode


class TestDraw:
    def test_univariate(self):
        with pm.Model():
            x = pm.Normal("x")

        x_draws = pm.draw(x)
        assert x_draws.shape == ()

        (x_draws,) = pm.draw([x])
        assert x_draws.shape == ()

        x_draws = pm.draw(x, draws=10)
        assert x_draws.shape == (10,)

        (x_draws,) = pm.draw([x], draws=10)
        assert x_draws.shape == (10,)

    def test_multivariate(self):
        with pm.Model():
            mln = pm.Multinomial("mln", n=5, p=np.array([0.25, 0.25, 0.25, 0.25]))

        mln_draws = pm.draw(mln, draws=1)
        assert mln_draws.shape == (4,)

        (mln_draws,) = pm.draw([mln], draws=1)
        assert mln_draws.shape == (4,)

        mln_draws = pm.draw(mln, draws=10)
        assert mln_draws.shape == (10, 4)

        (mln_draws,) = pm.draw([mln], draws=10)
        assert mln_draws.shape == (10, 4)

    def test_multiple_variables(self):
        with pm.Model():
            x = pm.Normal("x")
            y = pm.Normal("y", shape=10)
            z = pm.Uniform("z", shape=5)
            w = pm.Dirichlet("w", a=[1, 1, 1])

        num_draws = 100
        draws = pm.draw((x, y, z, w), draws=num_draws)
        assert draws[0].shape == (num_draws,)
        assert draws[1].shape == (num_draws, 10)
        assert draws[2].shape == (num_draws, 5)
        assert draws[3].shape == (num_draws, 3)

    def test_draw_different_samples(self):
        with pm.Model():
            x = pm.Normal("x")

        x_draws_1 = pm.draw(x, 100)
        x_draws_2 = pm.draw(x, 100)
        assert not np.all(np.isclose(x_draws_1, x_draws_2))

    def test_draw_pytensor_function_kwargs(self):
        sharedvar = pytensor.shared(0)
        x = pm.DiracDelta.dist(0.0)
        y = x + sharedvar
        draws = pm.draw(
            y,
            draws=5,
            mode=Mode("py"),
            updates={sharedvar: sharedvar + 1},
        )
        assert np.all(draws == np.arange(5))


class TestCompileForwardSampler:
    @staticmethod
    def get_function_roots(function):
        return [
            var
            for var in pytensor.graph.basic.graph_inputs(function.maker.fgraph.outputs)
            if var.name
        ]

    @staticmethod
    def get_function_inputs(function):
        return {i for i in function.maker.fgraph.inputs if not isinstance(i, SharedVariable)}

    def test_linear_model(self):
        with pm.Model() as model:
            x = pm.Data("x", np.linspace(0, 1, 10))
            y = pm.Data("y", np.ones(10))

            alpha = pm.Normal("alpha", 0, 0.1)
            beta = pm.Normal("beta", 0, 0.1)
            mu = pm.Deterministic("mu", alpha + beta * x)
            sigma = pm.HalfNormal("sigma", 0.1)
            obs = pm.Normal("obs", mu, sigma, observed=y, shape=x.shape)

        f, volatile_rvs = compile_forward_sampling_function(
            [obs],
            vars_in_trace=[alpha, beta, sigma, mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"alpha", "beta", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"x", "alpha", "beta", "sigma"}

    def test_nested_observed_model(self):
        with pm.Model() as model:
            p = pm.Data("p", np.array([0.25, 0.5, 0.25]))
            x = pm.Data("x", np.zeros(10))
            y = pm.Data("y", np.ones(10))

            category = pm.Categorical("category", p, observed=x)
            beta = pm.Normal("beta", 0, 0.1, size=p.shape)
            mu = pm.Deterministic("mu", beta[category])
            sigma = pm.HalfNormal("sigma", 0.1)
            obs = pm.Normal("obs", mu, sigma, observed=y, shape=mu.shape)

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=model.observed_RVs,
            vars_in_trace=[beta, mu, sigma],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {category, beta, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"x", "p", "sigma"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=model.observed_RVs,
            vars_in_trace=[beta, mu, sigma],
            constant_data={"p": p.get_value()},
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {category, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"beta", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"x", "p", "beta", "sigma"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=model.observed_RVs,
            vars_in_trace=[beta, mu, sigma],
            constant_data={"p": p.get_value()},
            basic_rvs=model.basic_RVs,
            givens_dict={category: np.zeros(10, dtype=category.dtype)},
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"beta", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {
            "x",
            "p",
            "category",
            "beta",
            "sigma",
        }

    def test_volatile_parameters(self):
        with pm.Model() as model:
            y = pm.Data("y", np.ones(10))
            mu = pm.Normal("mu", 0, 1)
            nested_mu = pm.Normal("nested_mu", mu, 1, size=10)
            sigma = pm.HalfNormal("sigma", 1)
            obs = pm.Normal("obs", nested_mu, sigma, observed=y, shape=nested_mu.shape)

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=model.observed_RVs,
            vars_in_trace=[nested_mu, sigma],  # mu isn't in the trace and will be deemed volatile
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {mu, nested_mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"sigma"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=model.observed_RVs,
            vars_in_trace=[mu, nested_mu, sigma],
            basic_rvs=model.basic_RVs,
            givens_dict={
                mu: np.array(1.0)
            },  # mu will be considered volatile because it's in givens
        )
        assert volatile_rvs == {nested_mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu", "sigma"}

    def test_mixture(self):
        with pm.Model() as model:
            w = pm.Dirichlet("w", a=np.ones(3), size=(5, 3))

            mu = pm.Normal("mu", mu=np.arange(3), sigma=1)

            components = pm.Normal.dist(mu=mu, sigma=1, size=w.shape)
            mix_mu = pm.Mixture("mix_mu", w=w, comp_dists=components)
            obs = pm.Normal("obs", mix_mu, 1, observed=np.ones((5, 3)))

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[mix_mu, mu, w],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"w", "mu", "mix_mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"mix_mu"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[mu, w],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {mix_mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"w", "mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"w", "mu"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[mix_mu, mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {w, mix_mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu"}

    def test_censored(self):
        with pm.Model() as model:
            latent_mu = pm.Normal("latent_mu", mu=np.arange(3), sigma=1)
            mu = pm.Censored("mu", pm.Normal.dist(mu=latent_mu, sigma=1), lower=-1, upper=1)
            obs = pm.Normal("obs", mu, 1, observed=np.ones((10, 3)))

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[latent_mu, mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"latent_mu", "mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {latent_mu, mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == set()
        assert {i.name for i in self.get_function_roots(f)} == set()

    def test_lkj_cholesky_cov(self):
        with pm.Model() as model:
            mu = np.zeros(3)
            sd_dist = pm.Exponential.dist(1.0, size=3)
            chol, corr, stds = pm.LKJCholeskyCov(
                "chol_packed", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            chol_packed = model["chol_packed"]
            chol = pm.Deterministic("chol", chol)
            obs = pm.MvNormal("obs", mu=mu, chol=chol, observed=np.zeros(3))

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[chol_packed, chol],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"chol_packed", "chol"}
        assert {i.name for i in self.get_function_roots(f)} == {"chol"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[chol_packed],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"chol_packed"}
        assert {i.name for i in self.get_function_roots(f)} == {"chol_packed"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[chol],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {chol_packed, obs}
        assert {i.name for i in self.get_function_inputs(f)} == set()
        assert {i.name for i in self.get_function_roots(f)} == set()

    def test_non_random_model_variable(self):
        with pm.Model() as model:
            # A user may register non-pure RandomVariables that can nevertheless be
            # sampled, as long as a custom logprob is dispatched or we can infer
            # its logprob (which is the case for `clip`)
            y = pt.clip(pm.Normal.dist(), -1, 1)
            y = model.register_rv(y, name="y")
            y_abs = pm.Deterministic("y_abs", pt.abs(y))
            obs = pm.Normal("obs", y_abs, observed=np.zeros(10))

        # y_abs should be resampled even if in the trace, because the source y is missing
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[y_abs],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {y, obs}
        assert {i.name for i in self.get_function_inputs(f)} == set()
        assert {i.name for i in self.get_function_roots(f)} == set()

    def test_mutable_coords_volatile(self):
        rng = np.random.default_rng(seed=42)
        data = rng.normal(loc=1, scale=0.2, size=(10, 3))
        with pm.Model() as model:
            model.add_coord("name", ["A", "B", "C"])
            model.add_coord("obs", list(range(10, 20)))
            offsets = pm.Data("offsets", rng.normal(0, 1, size=(10,)))
            a = pm.Normal("a", mu=0, sigma=1, dims=["name"])
            b = pm.Normal("b", mu=offsets, sigma=1)
            mu = pm.Deterministic("mu", a + b[..., None], dims=["obs", "name"])
            sigma = pm.HalfNormal("sigma", sigma=1, dims=["name"])

            data = pm.Data(
                "y_obs",
                data,
                dims=["obs", "name"],
            )
            y = pm.Normal("y", mu=mu, sigma=sigma, observed=data, dims=["obs", "name"])

        # When no constant_data and constant_coords, all the dependent nodes will be volatile and
        # resampled
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {y, a, b, sigma}
        assert {i.name for i in self.get_function_inputs(f)} == set()
        assert {i.name for i in self.get_function_roots(f)} == {"name", "obs", "offsets"}

        # When the constant data has the same values as the shared data, offsets wont be volatile
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
            constant_data={"offsets": offsets.get_value()},
        )
        assert volatile_rvs == {y, a, sigma}
        assert {i.name for i in self.get_function_inputs(f)} == {"b"}
        assert {i.name for i in self.get_function_roots(f)} == {"b", "name", "obs"}

        # When we declare constant_coords, the shared variables with matching names wont be volatile
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
            constant_coords={"name", "obs"},
        )
        assert volatile_rvs == {y, b}
        assert {i.name for i in self.get_function_inputs(f)} == {"a", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {
            "a",
            "sigma",
            "name",
            "obs",
            "offsets",
        }

        # When we have both constant_data and constant_coords, only y will be volatile
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
            constant_data={"offsets": offsets.get_value()},
            constant_coords={"name", "obs"},
        )
        assert volatile_rvs == {y}
        assert {i.name for i in self.get_function_inputs(f)} == {"a", "b", "mu", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu", "sigma", "name", "obs"}

        # When constant_data has different values than the shared variable, then
        # offsets will be volatile
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
            constant_data={"offsets": offsets.get_value() + 1},
            constant_coords={"name", "obs"},
        )
        assert volatile_rvs == {y, b}
        assert {i.name for i in self.get_function_inputs(f)} == {"a", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {
            "a",
            "sigma",
            "name",
            "obs",
            "offsets",
        }


class TestSamplePPC:
    def test_normal_scalar(self):
        nchains = 2
        ndraws = 500
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=0.0)
            trace = pm.sample(
                draws=ndraws,
                chains=nchains,
            )

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive(
                10 * [model.initial_point()], return_inferencedata=False
            )
            assert "a" in ppc0
            assert len(ppc0["a"][0]) == 10
            # test empty ppc
            ppc = pm.sample_posterior_predictive(trace, var_names=[], return_inferencedata=False)
            assert len(ppc) == 0

            # test keep_size parameter
            ppc = pm.sample_posterior_predictive(trace, return_inferencedata=False)
            assert ppc["a"].shape == (nchains, ndraws)

            # test default case
            random_state = np.random.RandomState(20160911)
            idata_ppc = pm.sample_posterior_predictive(
                trace, var_names=["a"], random_seed=random_state
            )
            ppc = idata_ppc.posterior_predictive
            assert "a" in ppc
            assert ppc["a"].shape == (nchains, ndraws)
            # mu's standard deviation may have changed thanks to a's observed
            _, pval = stats.kstest(
                (ppc["a"] - trace.posterior["mu"]).values.flatten(), stats.norm(loc=0, scale=1).cdf
            )
            assert pval > 0.001

    def test_normal_scalar_idata(self):
        nchains = 2
        ndraws = 500
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=0.0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Tuning samples will be included.*", UserWarning)
                trace = pm.sample(
                    draws=ndraws,
                    chains=nchains,
                    return_inferencedata=False,
                    discard_tuned_samples=False,
                )

        assert not isinstance(trace, InferenceData)

        with model:
            # test keep_size parameter and idata input
            idata = pm.to_inference_data(trace)
            assert isinstance(idata, InferenceData)

            ppc = pm.sample_posterior_predictive(idata, return_inferencedata=False)
            assert ppc["a"].shape == (nchains, ndraws)

    def test_normal_vector(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
            trace = pm.sample(return_inferencedata=False, draws=12, chains=1)

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive(
                10 * [model.initial_point()],
                return_inferencedata=False,
            )
            ppc = pm.sample_posterior_predictive(trace, return_inferencedata=False, var_names=[])
            assert len(ppc) == 0

            ppc = pm.sample_posterior_predictive(trace, return_inferencedata=False)
            assert ppc["a"].shape == (trace.nchains, len(trace), 2)
            assert ppc0["a"].shape == (1, 10, 2)

    def test_normal_vector_idata(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
            trace = pm.sample(return_inferencedata=False)

        assert not isinstance(trace, InferenceData)

        with model:
            # test keep_size parameter with inference data as input...
            idata = pm.to_inference_data(trace)
            assert isinstance(idata, InferenceData)

            ppc = pm.sample_posterior_predictive(idata, return_inferencedata=False)
            assert ppc["a"].shape == (trace.nchains, len(trace), 2)

    def test_exceptions(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
            idata = pm.sample(idata_kwargs={"log_likelihood": False})

        with model:
            # test wrong type argument
            bad_trace = {"mu": stats.norm.rvs(size=1000)}
            with pytest.raises(TypeError, match="type for `trace`"):
                ppc = pm.sample_posterior_predictive(bad_trace)

    def test_sum_normal(self):
        with pm.Model() as model:
            a = pm.Normal("a", sigma=0.2)
            b = pm.Normal("b", mu=a)
            idata = pm.sample(draws=1000, chains=1)

        with model:
            # test list input
            ppc0 = pm.sample_posterior_predictive(
                10 * [model.initial_point()], return_inferencedata=False
            )
            assert ppc0 == {}
            ppc = pm.sample_posterior_predictive(idata, return_inferencedata=False, var_names=["b"])
            assert len(ppc) == 1
            assert ppc["b"].shape == (
                1,
                1000,
            )
            scale = np.sqrt(1 + 0.2**2)
            _, pval = stats.kstest(ppc["b"].flatten(), stats.norm(scale=scale).cdf)
            assert pval > 0.001

    def test_model_not_drawable_prior(self, seeded_test):
        data = np.random.poisson(lam=10, size=200)
        model = pm.Model()
        with model:
            mu = pm.HalfFlat("sigma")
            pm.Poisson("foo", mu=mu, observed=data)
            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                    idata = pm.sample(tune=10, draws=40, chains=1)

        with model:
            with pytest.raises(NotImplementedError) as excinfo:
                pm.sample_prior_predictive(50)
            assert "Cannot sample" in str(excinfo.value)
            samples = pm.sample_posterior_predictive(idata, return_inferencedata=False)
            assert samples["foo"].shape == (1, 40, 200)

    def test_model_shared_variable(self):
        rng = np.random.RandomState(9832)

        x = rng.randn(100)
        y = x > 0
        x_shared = pytensor.shared(x)
        y_shared = pytensor.shared(y)
        samples = 100
        with pm.Model() as model:
            coeff = pm.Normal("x", mu=0, sigma=1)
            logistic = pm.Deterministic("p", pm.math.sigmoid(coeff * x_shared))

            obs = pm.Bernoulli("obs", p=logistic, observed=y_shared)
            trace = pm.sample(
                samples,
                chains=1,
                return_inferencedata=False,
                compute_convergence_checks=False,
                random_seed=rng,
            )

        x_shared.set_value([-1, 0, 1.0])
        y_shared.set_value([0, 0, 0])

        with model:
            post_pred = pm.sample_posterior_predictive(
                trace, return_inferencedata=False, var_names=["p", "obs"]
            )

        expected_p = np.array([[logistic.eval({coeff: val}) for val in trace["x"][:samples]]])
        assert post_pred["obs"].shape == (1, samples, 3)
        npt.assert_allclose(post_pred["p"], expected_p)

    def test_deterministic_of_observed(self):
        rng = np.random.RandomState(8442)

        meas_in_1 = pm.pytensorf.floatX(2 + 4 * rng.randn(10))
        meas_in_2 = pm.pytensorf.floatX(5 + 4 * rng.randn(10))
        nchains = 2
        with pm.Model() as model:
            mu_in_1 = pm.Normal("mu_in_1", 0, 2)
            sigma_in_1 = pm.HalfNormal("sd_in_1", 1)
            mu_in_2 = pm.Normal("mu_in_2", 0, 2)
            sigma_in_2 = pm.HalfNormal("sd__in_2", 1)

            in_1 = pm.Normal("in_1", mu_in_1, sigma_in_1, observed=meas_in_1)
            in_2 = pm.Normal("in_2", mu_in_2, sigma_in_2, observed=meas_in_2)
            out_diff = in_1 + in_2
            pm.Deterministic("out", out_diff)

            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                trace = pm.sample(
                    tune=100,
                    draws=100,
                    chains=nchains,
                    step=pm.Metropolis(),
                    return_inferencedata=False,
                    compute_convergence_checks=False,
                    random_seed=rng,
                )

            rtol = 1e-5 if pytensor.config.floatX == "float64" else 1e-4

            ppc = pm.sample_posterior_predictive(
                return_inferencedata=False,
                model=model,
                trace=trace,
                random_seed=0,
                var_names=[var.name for var in (model.deterministics + model.basic_RVs)],
            )

            npt.assert_allclose(ppc["in_1"] + ppc["in_2"], ppc["out"], rtol=rtol)

    def test_deterministic_of_observed_modified_interface(self):
        rng = np.random.RandomState(4982)

        meas_in_1 = pm.pytensorf.floatX(2 + 4 * rng.randn(100))
        meas_in_2 = pm.pytensorf.floatX(5 + 4 * rng.randn(100))
        with pm.Model() as model:
            mu_in_1 = pm.Normal("mu_in_1", 0, 1, initval=0)
            sigma_in_1 = pm.HalfNormal("sd_in_1", 1, initval=1)
            mu_in_2 = pm.Normal("mu_in_2", 0, 1, initval=0)
            sigma_in_2 = pm.HalfNormal("sd__in_2", 1, initval=1)

            in_1 = pm.Normal("in_1", mu_in_1, sigma_in_1, observed=meas_in_1)
            in_2 = pm.Normal("in_2", mu_in_2, sigma_in_2, observed=meas_in_2)
            out_diff = in_1 + in_2
            pm.Deterministic("out", out_diff)

            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                trace = pm.sample(
                    tune=100,
                    draws=100,
                    step=pm.Metropolis(),
                    return_inferencedata=False,
                    compute_convergence_checks=False,
                    random_seed=rng,
                )
            varnames = [v for v in trace.varnames if v != "out"]
            ppc_trace = [
                dict(zip(varnames, row)) for row in zip(*(trace.get_values(v) for v in varnames))
            ]
            ppc = pm.sample_posterior_predictive(
                return_inferencedata=False,
                model=model,
                trace=ppc_trace,
                var_names=[x.name for x in (model.deterministics + model.basic_RVs)],
            )

            rtol = 1e-5 if pytensor.config.floatX == "float64" else 1e-3
            npt.assert_allclose(ppc["in_1"] + ppc["in_2"], ppc["out"], rtol=rtol)

    def test_variable_type(self):
        with pm.Model() as model:
            mu = pm.HalfNormal("mu", 1)
            a = pm.Normal("a", mu=mu, sigma=2, observed=np.array([1, 2]))
            b = pm.Poisson("b", mu, observed=np.array([1, 2]))
            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                trace = pm.sample(
                    tune=10, draws=10, compute_convergence_checks=False, return_inferencedata=False
                )

        with model:
            ppc = pm.sample_posterior_predictive(trace, return_inferencedata=False)
            assert ppc["a"].dtype.kind == "f"
            assert ppc["b"].dtype.kind == "i"

    def test_potentials_warning(self):
        warning_msg = "The effect of Potentials on other parameters is ignored during"
        with pm.Model() as m:
            a = pm.Normal("a", 0, 1)
            p = pm.Potential("p", a + 1)
            obs = pm.Normal("obs", a, 1, observed=5)

        trace = az_from_dict({"a": np.random.rand(5)})
        with m:
            with pytest.warns(UserWarning, match=warning_msg):
                pm.sample_posterior_predictive(trace)

    def test_idata_extension(self):
        """Testing if sample_posterior_predictive() extends inferenceData"""

        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            a = pm.Normal("a", mu=mu, sigma=1, observed=[0.0, 1.0])
            idata = pm.sample(tune=10, draws=10, compute_convergence_checks=False)

        base_test_dict = {
            "posterior": ["mu", "~a"],
            "sample_stats": ["diverging", "lp"],
            "observed_data": ["a"],
        }
        test_dict = {"~posterior_predictive": [], "~predictions": [], **base_test_dict}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

        # extending idata with in-sample ppc
        with model:
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        # test addition
        test_dict = {"posterior_predictive": ["a"], "~predictions": [], **base_test_dict}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

        # extending idata with out-of-sample ppc
        with model:
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, predictions=True)
        # test addition
        test_dict = {"posterior_predictive": ["a"], "predictions": ["a"], **base_test_dict}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

    @pytest.mark.parametrize("multitrace", [False, True])
    def test_deterministics_out_of_idata(self, multitrace):
        draws = 10
        chains = 2
        coords = {"draw": range(draws), "chain": range(chains)}
        ds = xr.Dataset(
            {
                "a": xr.DataArray(
                    [[0] * draws] * chains,
                    coords=coords,
                    dims=["chain", "draw"],
                )
            },
            coords=coords,
        )
        with pm.Model() as m:
            a = pm.Normal("a")
            if multitrace:
                straces = []
                for chain in ds.chain:
                    strace = pm.backends.NDArray(model=m, vars=[a])
                    strace.setup(len(ds.draw), int(chain))
                    strace.values = {"a": ds.a.sel(chain=chain).data}
                    strace.draw_idx = len(ds.draw)
                    straces.append(strace)
                trace = MultiTrace(straces)
            else:
                trace = ds

            d = pm.Deterministic("d", a - 4)
            pm.Normal("c", d, sigma=0.01)
            ppc = pm.sample_posterior_predictive(trace, var_names="c", return_inferencedata=True)
        assert np.all(np.abs(ppc.posterior_predictive.c + 4) <= 0.1)

    def test_logging_sampled_basic_rvs_prior(self, caplog):
        with pm.Model() as m:
            x = pm.Normal("x")
            y = pm.Deterministic("y", x + 1)
            z = pm.Normal("z", y, observed=0)

        with m:
            pm.sample_prior_predictive(draws=1)
        assert caplog.record_tuples == [("pymc.sampling.forward", logging.INFO, "Sampling: [x, z]")]
        caplog.clear()

        with m:
            pm.sample_prior_predictive(draws=1, var_names=["x"])
        assert caplog.record_tuples == [("pymc.sampling.forward", logging.INFO, "Sampling: [x]")]
        caplog.clear()

    def test_logging_sampled_basic_rvs_posterior(self, caplog):
        with pm.Model() as m:
            x = pm.Normal("x")
            x_det = pm.Deterministic("x_det", x + 1)
            y = pm.Normal("y", x_det)
            z = pm.Normal("z", y, observed=0)

        idata = az_from_dict(posterior={"x": np.zeros(5), "x_det": np.ones(5), "y": np.ones(5)})
        with m:
            pm.sample_posterior_predictive(idata)
        assert caplog.record_tuples == [("pymc.sampling.forward", logging.INFO, "Sampling: [z]")]
        caplog.clear()

        with m:
            pm.sample_posterior_predictive(idata, var_names=["y", "z"])
        assert caplog.record_tuples == [("pymc.sampling.forward", logging.INFO, "Sampling: [y, z]")]
        caplog.clear()

        # Resampling `x` will force resampling of `y`, even if it is in trace
        with m:
            pm.sample_posterior_predictive(idata, var_names=["x", "z"])
        assert caplog.record_tuples == [
            ("pymc.sampling.forward", logging.INFO, "Sampling: [x, y, z]")
        ]
        caplog.clear()

        # Missing deterministic `x_det` does not show in the log, even if it is being
        # recomputed, only `y` RV shows
        idata = az_from_dict(posterior={"x": np.zeros(5)})
        with m:
            pm.sample_posterior_predictive(idata)
        assert caplog.record_tuples == [("pymc.sampling.forward", logging.INFO, "Sampling: [y, z]")]
        caplog.clear()

        # Missing deterministic `x_det` does not cause recomputation of downstream `y` RV
        idata = az_from_dict(posterior={"x": np.zeros(5), "y": np.ones(5)})
        with m:
            pm.sample_posterior_predictive(idata)
        assert caplog.record_tuples == [("pymc.sampling.forward", logging.INFO, "Sampling: [z]")]
        caplog.clear()

        # Missing `x` causes sampling of downstream `y` RV, even if it is present in trace
        idata = az_from_dict(posterior={"y": np.ones(5)})
        with m:
            pm.sample_posterior_predictive(idata)
        assert caplog.record_tuples == [
            ("pymc.sampling.forward", logging.INFO, "Sampling: [x, y, z]")
        ]
        caplog.clear()

    def test_logging_sampled_basic_rvs_posterior_deterministic(self, caplog):
        with pm.Model() as m:
            x = pm.Normal("x")
            x_det = pm.Deterministic("x_det", x + 1)
            y = pm.Normal("y", x_det)
            z = pm.Normal("z", y, observed=0)

        # Explicit resampling a deterministic will lead to resampling of downstream RV `y`
        # This behavior could change in the future as the posterior of `y` is still valid
        idata = az_from_dict(posterior={"x": np.zeros(5), "x_det": np.ones(5), "y": np.ones(5)})
        with m:
            pm.sample_posterior_predictive(idata, var_names=["x_det", "z"])
        assert caplog.record_tuples == [("pymc.sampling.forward", logging.INFO, "Sampling: [y, z]")]
        caplog.clear()

    @staticmethod
    def make_mock_model():
        rng = np.random.default_rng(seed=42)
        data = rng.normal(loc=1, scale=0.2, size=(10, 3))
        with pm.Model() as model:
            model.add_coord("name", ["A", "B", "C"])
            model.add_coord("obs", list(range(10, 20)))
            offsets = pm.Data("offsets", rng.normal(0, 1, size=(10,)))
            a = pm.Normal("a", mu=0, sigma=1, dims=["name"])
            b = pm.Normal("b", mu=offsets, sigma=1)
            mu = pm.Deterministic("mu", a + b[..., None], dims=["obs", "name"])
            sigma = pm.HalfNormal("sigma", sigma=1, dims=["name"])

            data = pm.Data(
                "y_obs",
                data,
                dims=["obs", "name"],
            )
            pm.Normal("y", mu=mu, sigma=sigma, observed=data, dims=["obs", "name"])
        return model

    @pytest.fixture(scope="class")
    def mock_multitrace(self):
        with self.make_mock_model():
            trace = pm.sample(
                draws=10,
                tune=10,
                chains=2,
                progressbar=False,
                compute_convergence_checks=False,
                return_inferencedata=False,
                random_seed=42,
            )
        return trace

    @pytest.fixture(scope="class", params=["MultiTrace", "InferenceData", "Dataset"])
    def mock_sample_results(self, request, mock_multitrace):
        kind = request.param
        trace = mock_multitrace
        # We rebuild the class to ensure that all dimensions, data and coords start out
        # the same across params values
        model = self.make_mock_model()
        if kind == "MultiTrace":
            return kind, trace, model
        else:
            idata = pm.to_inference_data(
                trace,
                save_warmup=False,
                model=model,
                log_likelihood=False,
            )
            if kind == "Dataset":
                return kind, idata.posterior, model
            else:
                return kind, idata, model

    def test_logging_sampled_basic_rvs_posterior_mutable(self, mock_sample_results, caplog):
        kind, samples, model = mock_sample_results
        with model:
            pm.sample_posterior_predictive(samples)
        if kind == "MultiTrace":
            # MultiTrace will only have the actual MCMC posterior samples but no information on
            # the Data and coordinate values, so it will always assume they are volatile
            # and resample their descendants
            assert caplog.record_tuples == [
                ("pymc.sampling.forward", logging.INFO, "Sampling: [a, b, sigma, y]")
            ]
            caplog.clear()
        elif kind == "InferenceData":
            # InferenceData has all MCMC posterior samples and the values for both coordinates and
            # data containers. This enables it to see that no data has changed and it should only
            # resample the observed variable
            assert caplog.record_tuples == [
                ("pymc.sampling.forward", logging.INFO, "Sampling: [y]")
            ]
            caplog.clear()
        elif kind == "Dataset":
            # Dataset has all MCMC posterior samples and the values of the coordinates. This
            # enables it to see that the coordinates have not changed, but the MutableData is
            # assumed volatile by default
            assert caplog.record_tuples == [
                ("pymc.sampling.forward", logging.INFO, "Sampling: [b, y]")
            ]
            caplog.clear()

        original_offsets = model["offsets"].get_value()
        with model:
            # Changing the MutableData values. This will only be picked up by InferenceData
            pm.set_data({"offsets": original_offsets + 1})
            pm.sample_posterior_predictive(samples)
        if kind == "MultiTrace":
            assert caplog.record_tuples == [
                ("pymc.sampling.forward", logging.INFO, "Sampling: [a, b, sigma, y]")
            ]
            caplog.clear()
        elif kind == "InferenceData":
            assert caplog.record_tuples == [
                ("pymc.sampling.forward", logging.INFO, "Sampling: [b, y]")
            ]
            caplog.clear()
        elif kind == "Dataset":
            assert caplog.record_tuples == [
                ("pymc.sampling.forward", logging.INFO, "Sampling: [b, y]")
            ]
            caplog.clear()

        with model:
            # Changing the mutable coordinates. This will be picked up by InferenceData and Dataset
            model.set_dim("name", new_length=4, coord_values=["D", "E", "F", "G"])
            pm.set_data({"offsets": original_offsets, "y_obs": np.zeros((10, 4))})
            pm.sample_posterior_predictive(samples)
        if kind == "MultiTrace":
            assert caplog.record_tuples == [
                ("pymc.sampling.forward", logging.INFO, "Sampling: [a, b, sigma, y]")
            ]
            caplog.clear()
        elif kind == "InferenceData":
            assert caplog.record_tuples == [
                ("pymc.sampling.forward", logging.INFO, "Sampling: [a, sigma, y]")
            ]
            caplog.clear()
        elif kind == "Dataset":
            assert caplog.record_tuples == [
                ("pymc.sampling.forward", logging.INFO, "Sampling: [a, b, sigma, y]")
            ]
            caplog.clear()

        with model:
            # Changing the mutable coordinate values, but not shape, and also changing MutableData.
            # This will trigger resampling of all variables
            model.set_dim("name", new_length=3, coord_values=["A", "B", "D"])
            pm.set_data({"offsets": original_offsets + 1, "y_obs": np.zeros((10, 3))})
            pm.sample_posterior_predictive(samples)
        if kind == "MultiTrace":
            assert caplog.record_tuples == [
                ("pymc.sampling.forward", logging.INFO, "Sampling: [a, b, sigma, y]")
            ]
            caplog.clear()
        elif kind == "InferenceData":
            assert caplog.record_tuples == [
                ("pymc.sampling.forward", logging.INFO, "Sampling: [a, b, sigma, y]")
            ]
            caplog.clear()
        elif kind == "Dataset":
            assert caplog.record_tuples == [
                ("pymc.sampling.forward", logging.INFO, "Sampling: [a, b, sigma, y]")
            ]
            caplog.clear()

    def test_observed_data_needed_in_pp(self):
        # Model where y_data is not part of the generative graph.
        # It shouldn't be needed to set a dummy value for posterior predictive sampling

        with pm.Model(coords={"trial": range(5), "feature": range(3)}) as m:
            x_data = pm.Data("x_data", np.random.normal(size=(5, 3)), dims=("trial", "feat"))
            y_data = pm.Data("y_data", np.random.normal(size=(5,)), dims=("trial",))
            sigma = pm.HalfNormal("sigma")
            mu = x_data.sum(-1)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y_data, shape=mu.shape, dims=("trial",))

            prior = pm.sample_prior_predictive(draws=25).prior

        fake_idata = InferenceData(posterior=prior)

        new_coords = {"trial": range(2), "feature": range(3)}
        new_x_data = np.random.normal(size=(2, 3))
        with m:
            pm.set_data(
                {
                    "x_data": new_x_data,
                },
                coords=new_coords,
            )
            pp = pm.sample_posterior_predictive(fake_idata, predictions=True, progressbar=False)
        assert pp.predictions["y"].shape == (1, 25, 2)

        # In this case y_data is part of the generative graph, so we must set it to a compatible value
        with pm.Model(coords={"trial": range(5), "feature": range(3)}) as m:
            x_data = pm.Data("x_data", np.random.normal(size=(5, 3)), dims=("trial", "feat"))
            y_data = pm.Data("y_data", np.random.normal(size=(5,)), dims=("trial",))
            sigma = pm.HalfNormal("sigma")
            mu = (y_data.sum() * x_data).sum(-1)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y_data, shape=mu.shape, dims=("trial",))

            prior = pm.sample_prior_predictive(draws=25).prior

        fake_idata = InferenceData(posterior=prior)

        with m:
            pm.set_data({"x_data": new_x_data}, coords=new_coords)
            with pytest.raises(ValueError, match="conflicting sizes for dimension 'trial'"):
                pm.sample_posterior_predictive(fake_idata, predictions=True, progressbar=False)

        new_y_data = np.random.normal(size=(2,))
        with m:
            pm.set_data({"y_data": new_y_data})
        assert pp.predictions["y"].shape == (1, 25, 2)


@pytest.fixture(scope="class")
def point_list_arg_bug_fixture() -> tuple[pm.Model, pm.backends.base.MultiTrace]:
    with pm.Model() as pmodel:
        n = pm.Normal("n")
        trace = pm.sample(return_inferencedata=False)

    with pmodel:
        d = pm.Deterministic("d", n * 4)
    return pmodel, trace


class TestSamplePriorPredictive:
    def test_ignores_observed(self, seeded_test):
        observed = np.random.normal(10, 1, size=200)
        with pm.Model():
            # Use a prior that's way off to show we're ignoring the observed variables
            observed_data = pm.Data("observed_data", observed)
            mu = pm.Normal("mu", mu=-100, sigma=1)
            positive_mu = pm.Deterministic("positive_mu", np.abs(mu))
            z = -1 - positive_mu
            pm.Normal("x_obs", mu=z, sigma=1, observed=observed_data)
            prior = pm.sample_prior_predictive(return_inferencedata=False)

        assert "observed_data" not in prior
        assert (prior["mu"] < -90).all()
        assert (prior["positive_mu"] > 90).all()
        assert (prior["x_obs"] < -90).all()
        assert prior["x_obs"].shape == (500, 200)
        npt.assert_array_almost_equal(prior["positive_mu"], np.abs(prior["mu"]), decimal=4)

    def test_respects_shape(self):
        for shape in (2, (2,), (10, 2), (10, 10)):
            with pm.Model():
                mu = pm.Gamma("mu", 3, 1, size=1)
                goals = pm.Poisson("goals", mu, size=shape)
                trace1 = pm.sample_prior_predictive(
                    10, return_inferencedata=False, var_names=["mu", "mu", "goals"]
                )
                trace2 = pm.sample_prior_predictive(
                    10, return_inferencedata=False, var_names=["mu", "goals"]
                )
            if shape == 2:  # want to test shape as an int
                shape = (2,)
            assert trace1["goals"].shape == (10, *shape)
            assert trace2["goals"].shape == (10, *shape)

    def test_multivariate(self):
        with pm.Model():
            m = pm.Multinomial("m", n=5, p=np.array([0.25, 0.25, 0.25, 0.25]))
            trace = pm.sample_prior_predictive(10)

        assert trace.prior["m"].shape == (1, 10, 4)

    def test_multivariate2(self, seeded_test):
        # Added test for issue #3271
        mn_data = np.random.multinomial(n=100, pvals=[1 / 6.0] * 6, size=10)
        with pm.Model() as dm_model:
            probs = pm.Dirichlet("probs", a=np.ones(6))
            obs = pm.Multinomial("obs", n=100, p=probs, observed=mn_data)
            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                burned_trace = pm.sample(
                    tune=10,
                    draws=20,
                    chains=1,
                    return_inferencedata=False,
                    compute_convergence_checks=False,
                )
        sim_priors = pm.sample_prior_predictive(
            return_inferencedata=False, draws=20, model=dm_model
        )
        sim_ppc = pm.sample_posterior_predictive(
            burned_trace, return_inferencedata=False, model=dm_model
        )
        assert sim_priors["probs"].shape == (20, 6)
        assert sim_priors["obs"].shape == (20, *mn_data.shape)
        assert sim_ppc["obs"].shape == (1, 20, *mn_data.shape)

    def test_layers(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0, upper=1, size=10)
            b = pm.Binomial("b", n=1, p=a, size=10)

        b_sampler = compile_pymc([], b, mode="FAST_RUN", random_seed=232093)
        avg = np.stack([b_sampler() for i in range(10000)]).mean(0)
        npt.assert_array_almost_equal(avg, 0.5 * np.ones((10,)), decimal=2)

    def test_transformed(self, seeded_test):
        n = 18
        at_bats = 45 * np.ones(n, dtype=int)
        hits = np.random.randint(1, 40, size=n, dtype=int)
        draws = 50

        with pm.Model() as model:
            phi = pm.Beta("phi", alpha=1.0, beta=1.0)

            kappa_log = pm.Exponential("logkappa", lam=5.0)
            kappa = pm.Deterministic("kappa", pt.exp(kappa_log))

            thetas = pm.Beta("thetas", alpha=phi * kappa, beta=(1.0 - phi) * kappa, size=n)

            y = pm.Binomial("y", n=at_bats, p=thetas, observed=hits)
            gen = pm.sample_prior_predictive(draws)

        assert gen.prior["phi"].shape == (1, draws)
        assert gen.prior_predictive["y"].shape == (1, draws, n)
        assert "thetas" in gen.prior.data_vars

    def test_shared(self, seeded_test):
        n1 = 10
        obs = shared(np.random.rand(n1) < 0.5)
        draws = 50

        with pm.Model() as m:
            p = pm.Beta("p", 1.0, 1.0)
            y = pm.Bernoulli("y", p, observed=obs)
            o = pm.Deterministic("o", obs)
            gen1 = pm.sample_prior_predictive(draws)

        assert gen1.prior_predictive["y"].shape == (1, draws, n1)
        assert gen1.prior["o"].shape == (1, draws, n1)

        n2 = 20
        obs.set_value(np.random.rand(n2) < 0.5)
        with m:
            gen2 = pm.sample_prior_predictive(draws)

        assert gen2.prior_predictive["y"].shape == (1, draws, n2)
        assert gen2.prior["o"].shape == (1, draws, n2)

    def test_density_dist(self, seeded_test):
        obs = np.random.normal(-1, 0.1, size=10)
        with pm.Model():
            mu = pm.Normal("mu", 0, 1)
            sigma = pm.HalfNormal("sigma", 1e-6)
            a = pm.CustomDist(
                "a",
                mu,
                sigma,
                random=lambda mu, sigma, rng=None, size=None: rng.normal(
                    loc=mu, scale=sigma, size=size
                ),
                observed=obs,
            )
            prior = pm.sample_prior_predictive(return_inferencedata=False)

        npt.assert_almost_equal((prior["a"] - prior["mu"][..., None]).mean(), 0, decimal=3)

    def test_shape_edgecase(self):
        with pm.Model():
            mu = pm.Normal("mu", size=5)
            sigma = pm.Uniform("sigma", lower=2, upper=3)
            x = pm.Normal("x", mu=mu, sigma=sigma, size=5)
            prior = pm.sample_prior_predictive(10)
        assert prior.prior["mu"].shape == (1, 10, 5)

    def test_zeroinflatedpoisson(self):
        with pm.Model():
            mu = pm.Beta("mu", alpha=1, beta=1)
            psi = pm.HalfNormal("psi", sigma=1)
            pm.ZeroInflatedPoisson("suppliers", psi=psi, mu=mu, size=20)
            gen_data = pm.sample_prior_predictive(draws=5000)
            assert gen_data.prior["mu"].shape == (1, 5000)
            assert gen_data.prior["psi"].shape == (1, 5000)
            assert gen_data.prior["suppliers"].shape == (1, 5000, 20)

    def test_potentials_warning(self):
        warning_msg = "The effect of Potentials on other parameters is ignored during"
        with pm.Model() as m:
            a = pm.Normal("a", 0, 1)
            p = pm.Potential("p", a + 1)

        with m:
            with pytest.warns(UserWarning, match=warning_msg):
                pm.sample_prior_predictive(draws=5)

    def test_transformed_vars_not_supported(self):
        with pm.Model() as model:
            ub = pm.HalfNormal("ub", 10)
            x = pm.Uniform("x", 0, ub)

            with pytest.raises(ValueError, match="Unrecognized var_names"):
                pm.sample_prior_predictive(var_names=["ub", "ub_log__", "x", "x_interval__"])

    def test_issue_4490(self):
        # Test that samples do not depend on var_name order or, more fundamentally,
        # that they do not depend on the set order used inside `sample_prior_predictive`
        seed = 4490
        with pm.Model() as m1:
            a = pm.Normal("a")
            b = pm.Normal("b")
            c = pm.Normal("c")
            d = pm.Normal("d")
            prior1 = pm.sample_prior_predictive(
                draws=1, var_names=["a", "b", "c", "d"], random_seed=seed
            )

        with pm.Model() as m2:
            a = pm.Normal("a")
            b = pm.Normal("b")
            c = pm.Normal("c")
            d = pm.Normal("d")
            prior2 = pm.sample_prior_predictive(
                draws=1, var_names=["b", "a", "d", "c"], random_seed=seed
            )

        assert prior1.prior["a"] == prior2.prior["a"]
        assert prior1.prior["b"] == prior2.prior["b"]
        assert prior1.prior["c"] == prior2.prior["c"]
        assert prior1.prior["d"] == prior2.prior["d"]

    def test_pytensor_function_kwargs(self):
        sharedvar = pytensor.shared(0)
        with pm.Model() as m:
            x = pm.DiracDelta("x", 0)
            y = pm.Deterministic("y", x + sharedvar)

            prior = pm.sample_prior_predictive(
                draws=5,
                return_inferencedata=False,
                compile_kwargs=dict(
                    mode=Mode("py"),
                    updates={sharedvar: sharedvar + 1},
                ),
            )

        assert np.all(prior["y"] == np.arange(5))


class TestSamplePosteriorPredictive:
    def test_point_list_arg_bug_spp(self, point_list_arg_bug_fixture):
        pmodel, trace = point_list_arg_bug_fixture
        with pmodel:
            pp = pm.sample_posterior_predictive(
                [trace[15]], return_inferencedata=False, var_names=["d"]
            )

    def test_sample_from_xarray_prior(self, point_list_arg_bug_fixture):
        pmodel, trace = point_list_arg_bug_fixture

        with pmodel:
            prior = pm.sample_prior_predictive(
                draws=20,
                return_inferencedata=False,
            )
            idat = pm.to_inference_data(trace, prior=prior)

        with pmodel:
            pp = pm.sample_posterior_predictive(
                idat.prior, return_inferencedata=False, var_names=["d"]
            )

    def test_sample_from_xarray_posterior(self, point_list_arg_bug_fixture):
        pmodel, trace = point_list_arg_bug_fixture
        with pmodel:
            idat = pm.to_inference_data(trace)
            pp = pm.sample_posterior_predictive(idat.posterior, var_names=["d"])

    def test_pytensor_function_kwargs(self):
        sharedvar = pytensor.shared(0)
        with pm.Model() as m:
            x = pm.DiracDelta("x", 0.0)
            y = pm.Deterministic("y", x + sharedvar)

            pp = pm.sample_posterior_predictive(
                trace=az_from_dict({"x": np.arange(5)}),
                var_names=["y"],
                return_inferencedata=False,
                compile_kwargs=dict(
                    mode=Mode("py"),
                    updates={sharedvar: sharedvar + 1},
                ),
            )

        assert np.all(pp["y"] == np.arange(5) * 2)

    def test_sample_dims(self, point_list_arg_bug_fixture):
        pmodel, trace = point_list_arg_bug_fixture
        with pmodel:
            post = pm.to_inference_data(trace).posterior.stack(sample=["chain", "draw"])
            pp = pm.sample_posterior_predictive(post, var_names=["d"], sample_dims=["sample"])
            assert "sample" in pp.posterior_predictive
            assert len(pp.posterior_predictive["sample"]) == len(post["sample"])
            post = post.expand_dims(pred_id=5)
            pp = pm.sample_posterior_predictive(
                post, var_names=["d"], sample_dims=["sample", "pred_id"]
            )
            assert "sample" in pp.posterior_predictive
            assert "pred_id" in pp.posterior_predictive
            assert len(pp.posterior_predictive["sample"]) == len(post["sample"])
            assert len(pp.posterior_predictive["pred_id"]) == 5


def test_distinct_rvs():
    """Make sure `RandomVariable`s generated using a `Model`'s default RNG state all have distinct states."""

    with pm.Model() as model:
        X_rv = pm.Normal("x")
        Y_rv = pm.Normal("y")

        pp_samples = pm.sample_prior_predictive(
            draws=2, return_inferencedata=False, random_seed=npr.RandomState(2023532)
        )

    assert X_rv.owner.inputs[0] != Y_rv.owner.inputs[0]

    with pm.Model():
        X_rv = pm.Normal("x")
        Y_rv = pm.Normal("y")

        pp_samples_2 = pm.sample_prior_predictive(
            draws=2, return_inferencedata=False, random_seed=npr.RandomState(2023532)
        )

    assert np.array_equal(pp_samples["y"], pp_samples_2["y"])


class TestNestedRandom:
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
        assert prior["target"].shape == (prior_samples, *shape)

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
        assert prior["target"].shape == (prior_samples, *shape)

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
        assert prior["target"].shape == (prior_samples, *shape)

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
        assert prior["target"].shape == (prior_samples, *shape)

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
        assert prior["target"].shape == (prior_samples, *shape)


def test_get_vars_in_point_list():
    with pm.Model() as modelA:
        pm.Normal("a", 0, 1)
        pm.Normal("b", 0, 1)
        pm.Normal("d", 0, 1)
    with pm.Model() as modelB:
        a = pm.Normal("a", 0, 1)
        pm.Normal("c", 0, 1)
        pm.Data("d", 0)

    point_list = [{"a": 0, "b": 0, "d": 0}]
    vars_in_trace = get_vars_in_point_list(point_list, modelB)
    assert set(vars_in_trace) == {a}

    strace = pm.backends.NDArray(model=modelB, vars=modelA.free_RVs)
    strace.setup(1, 1)
    strace.values = point_list[0]
    strace.draw_idx = 1
    trace = MultiTrace([strace])
    vars_in_trace = get_vars_in_point_list(trace, modelB)
    assert set(vars_in_trace) == {a}


def test_observed_dependent_deterministics():
    with pm.Model() as m:
        free = pm.Normal("free")
        obs = pm.Normal("obs", observed=1)

        det_free = pm.Deterministic("det_free", free + 1)
        det_free2 = pm.Deterministic("det_free2", det_free + 1)

        det_obs = pm.Deterministic("det_obs", obs + 1)
        det_obs2 = pm.Deterministic("det_obs2", det_obs + 1)

        det_mixed = pm.Deterministic("det_mixed", free + obs)

    assert set(observed_dependent_deterministics(m)) == {det_obs, det_obs2, det_mixed}


def test_sample_prior_predictive_samples_deprecated_warns() -> None:
    with pm.Model() as m:
        pm.Normal("a")

    match = "The samples argument has been deprecated"
    with pytest.warns(DeprecationWarning, match=match):
        pm.sample_prior_predictive(model=m, samples=10)
