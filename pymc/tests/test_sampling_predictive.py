#   Copyright 2022 The PyMC Developers
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

from typing import Tuple

import aesara
import aesara.tensor as at
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import pytest
import xarray as xr

from aesara import Mode, shared
from arviz import InferenceData
from arviz import from_dict as az_from_dict
from arviz.tests.helpers import check_multiple_attrs
from scipy import stats

import pymc as pm

from pymc.aesaraf import compile_pymc
from pymc.backends.base import MultiTrace
from pymc.tests.helpers import SeededTest, fast_unstable_sampling_mode


class TestSamplePPC(SeededTest):
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
            random_state = self.get_random_state()
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

    def test_model_not_drawable_prior(self):
        data = np.random.poisson(lam=10, size=200)
        model = pm.Model()
        with model:
            mu = pm.HalfFlat("sigma")
            pm.Poisson("foo", mu=mu, observed=data)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
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
        x_shared = aesara.shared(x)
        y_shared = aesara.shared(y)
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

        meas_in_1 = pm.aesaraf.floatX(2 + 4 * rng.randn(10))
        meas_in_2 = pm.aesaraf.floatX(5 + 4 * rng.randn(10))
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

            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                trace = pm.sample(
                    tune=100,
                    draws=100,
                    chains=nchains,
                    step=pm.Metropolis(),
                    return_inferencedata=False,
                    compute_convergence_checks=False,
                    random_seed=rng,
                )

            rtol = 1e-5 if aesara.config.floatX == "float64" else 1e-4

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

        meas_in_1 = pm.aesaraf.floatX(2 + 4 * rng.randn(100))
        meas_in_2 = pm.aesaraf.floatX(5 + 4 * rng.randn(100))
        with pm.Model() as model:
            mu_in_1 = pm.Normal("mu_in_1", 0, 1, initval=0)
            sigma_in_1 = pm.HalfNormal("sd_in_1", 1, initval=1)
            mu_in_2 = pm.Normal("mu_in_2", 0, 1, initval=0)
            sigma_in_2 = pm.HalfNormal("sd__in_2", 1, initval=1)

            in_1 = pm.Normal("in_1", mu_in_1, sigma_in_1, observed=meas_in_1)
            in_2 = pm.Normal("in_2", mu_in_2, sigma_in_2, observed=meas_in_2)
            out_diff = in_1 + in_2
            pm.Deterministic("out", out_diff)

            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
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

            rtol = 1e-5 if aesara.config.floatX == "float64" else 1e-3
            npt.assert_allclose(ppc["in_1"] + ppc["in_2"], ppc["out"], rtol=rtol)

    def test_variable_type(self):
        with pm.Model() as model:
            mu = pm.HalfNormal("mu", 1)
            a = pm.Normal("a", mu=mu, sigma=2, observed=np.array([1, 2]))
            b = pm.Poisson("b", mu, observed=np.array([1, 2]))
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
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
            "log_likelihood": ["a"],
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
            pm.sample_prior_predictive(samples=1)
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [x, z]")]
        caplog.clear()

        with m:
            pm.sample_prior_predictive(samples=1, var_names=["x"])
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [x]")]
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
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [z]")]
        caplog.clear()

        with m:
            pm.sample_posterior_predictive(idata, var_names=["y", "z"])
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [y, z]")]
        caplog.clear()

        # Resampling `x` will force resampling of `y`, even if it is in trace
        with m:
            pm.sample_posterior_predictive(idata, var_names=["x", "z"])
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [x, y, z]")]
        caplog.clear()

        # Missing deterministic `x_det` does not show in the log, even if it is being
        # recomputed, only `y` RV shows
        idata = az_from_dict(posterior={"x": np.zeros(5)})
        with m:
            pm.sample_posterior_predictive(idata)
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [y, z]")]
        caplog.clear()

        # Missing deterministic `x_det` does not cause recomputation of downstream `y` RV
        idata = az_from_dict(posterior={"x": np.zeros(5), "y": np.ones(5)})
        with m:
            pm.sample_posterior_predictive(idata)
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [z]")]
        caplog.clear()

        # Missing `x` causes sampling of downstream `y` RV, even if it is present in trace
        idata = az_from_dict(posterior={"y": np.ones(5)})
        with m:
            pm.sample_posterior_predictive(idata)
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [x, y, z]")]
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
        assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [y, z]")]
        caplog.clear()

    @staticmethod
    def make_mock_model():
        rng = np.random.default_rng(seed=42)
        data = rng.normal(loc=1, scale=0.2, size=(10, 3))
        with pm.Model() as model:
            model.add_coord("name", ["A", "B", "C"], mutable=True)
            model.add_coord("obs", list(range(10, 20)), mutable=True)
            offsets = pm.MutableData("offsets", rng.normal(0, 1, size=(10,)))
            a = pm.Normal("a", mu=0, sigma=1, dims=["name"])
            b = pm.Normal("b", mu=offsets, sigma=1)
            mu = pm.Deterministic("mu", a + b[..., None], dims=["obs", "name"])
            sigma = pm.HalfNormal("sigma", sigma=1, dims=["name"])

            data = pm.MutableData(
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
            # the MutableData and mutable coordinate values, so it will always assume they are volatile
            # and resample their descendants
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()
        elif kind == "InferenceData":
            # InferenceData has all MCMC posterior samples and the values for both coordinates and
            # data containers. This enables it to see that no data has changed and it should only
            # resample the observed variable
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [y]")]
            caplog.clear()
        elif kind == "Dataset":
            # Dataset has all MCMC posterior samples and the values of the coordinates. This
            # enables it to see that the coordinates have not changed, but the MutableData is
            # assumed volatile by default
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [b, y]")]
            caplog.clear()

        original_offsets = model["offsets"].get_value()
        with model:
            # Changing the MutableData values. This will only be picked up by InferenceData
            pm.set_data({"offsets": original_offsets + 1})
            pm.sample_posterior_predictive(samples)
        if kind == "MultiTrace":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()
        elif kind == "InferenceData":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [b, y]")]
            caplog.clear()
        elif kind == "Dataset":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [b, y]")]
            caplog.clear()

        with model:
            # Changing the mutable coordinates. This will be picked up by InferenceData and Dataset
            model.set_dim("name", new_length=4, coord_values=["D", "E", "F", "G"])
            pm.set_data({"offsets": original_offsets, "y_obs": np.zeros((10, 4))})
            pm.sample_posterior_predictive(samples)
        if kind == "MultiTrace":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()
        elif kind == "InferenceData":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, sigma, y]")]
            caplog.clear()
        elif kind == "Dataset":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()

        with model:
            # Changing the mutable coordinate values, but not shape, and also changing MutableData.
            # This will trigger resampling of all variables
            model.set_dim("name", new_length=3, coord_values=["A", "B", "D"])
            pm.set_data({"offsets": original_offsets + 1, "y_obs": np.zeros((10, 3))})
            pm.sample_posterior_predictive(samples)
        if kind == "MultiTrace":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()
        elif kind == "InferenceData":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()
        elif kind == "Dataset":
            assert caplog.record_tuples == [("pymc", logging.INFO, "Sampling: [a, b, sigma, y]")]
            caplog.clear()


@pytest.fixture(scope="class")
def point_list_arg_bug_fixture() -> Tuple[pm.Model, pm.backends.base.MultiTrace]:
    with pm.Model() as pmodel:
        n = pm.Normal("n")
        trace = pm.sample(return_inferencedata=False)

    with pmodel:
        d = pm.Deterministic("d", n * 4)
    return pmodel, trace


class TestSamplePriorPredictive(SeededTest):
    def test_ignores_observed(self):
        observed = np.random.normal(10, 1, size=200)
        with pm.Model():
            # Use a prior that's way off to show we're ignoring the observed variables
            observed_data = pm.MutableData("observed_data", observed)
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
            assert trace1["goals"].shape == (10,) + shape
            assert trace2["goals"].shape == (10,) + shape

    def test_multivariate(self):
        with pm.Model():
            m = pm.Multinomial("m", n=5, p=np.array([0.25, 0.25, 0.25, 0.25]))
            trace = pm.sample_prior_predictive(10)

        assert trace.prior["m"].shape == (1, 10, 4)

    def test_multivariate2(self):
        # Added test for issue #3271
        mn_data = np.random.multinomial(n=100, pvals=[1 / 6.0] * 6, size=10)
        with pm.Model() as dm_model:
            probs = pm.Dirichlet("probs", a=np.ones(6))
            obs = pm.Multinomial("obs", n=100, p=probs, observed=mn_data)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                burned_trace = pm.sample(
                    tune=10,
                    draws=20,
                    chains=1,
                    return_inferencedata=False,
                    compute_convergence_checks=False,
                )
        sim_priors = pm.sample_prior_predictive(
            return_inferencedata=False, samples=20, model=dm_model
        )
        sim_ppc = pm.sample_posterior_predictive(
            burned_trace, return_inferencedata=False, model=dm_model
        )
        assert sim_priors["probs"].shape == (20, 6)
        assert sim_priors["obs"].shape == (20,) + mn_data.shape
        assert sim_ppc["obs"].shape == (1, 20) + mn_data.shape

    def test_layers(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0, upper=1, size=10)
            b = pm.Binomial("b", n=1, p=a, size=10)

        b_sampler = compile_pymc([], b, mode="FAST_RUN", random_seed=232093)
        avg = np.stack([b_sampler() for i in range(10000)]).mean(0)
        npt.assert_array_almost_equal(avg, 0.5 * np.ones((10,)), decimal=2)

    def test_transformed(self):
        n = 18
        at_bats = 45 * np.ones(n, dtype=int)
        hits = np.random.randint(1, 40, size=n, dtype=int)
        draws = 50

        with pm.Model() as model:
            phi = pm.Beta("phi", alpha=1.0, beta=1.0)

            kappa_log = pm.Exponential("logkappa", lam=5.0)
            kappa = pm.Deterministic("kappa", at.exp(kappa_log))

            thetas = pm.Beta("thetas", alpha=phi * kappa, beta=(1.0 - phi) * kappa, size=n)

            y = pm.Binomial("y", n=at_bats, p=thetas, observed=hits)
            gen = pm.sample_prior_predictive(draws)

        assert gen.prior["phi"].shape == (1, draws)
        assert gen.prior_predictive["y"].shape == (1, draws, n)
        assert "thetas" in gen.prior.data_vars

    def test_shared(self):
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

    def test_density_dist(self):
        obs = np.random.normal(-1, 0.1, size=10)
        with pm.Model():
            mu = pm.Normal("mu", 0, 1)
            sigma = pm.HalfNormal("sigma", 1e-6)
            a = pm.DensityDist(
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
            gen_data = pm.sample_prior_predictive(samples=5000)
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
                pm.sample_prior_predictive(samples=5)

    def test_transformed_vars(self):
        # Test that prior predictive returns transformation of RVs when these are
        # passed explicitly in `var_names`

        def ub_interval_forward(x, ub):
            # Interval transform assuming lower bound is zero
            return np.log(x - 0) - np.log(ub - x)

        with pm.Model() as model:
            ub = pm.HalfNormal("ub", 10)
            x = pm.Uniform("x", 0, ub)

            prior = pm.sample_prior_predictive(
                var_names=["ub", "ub_log__", "x", "x_interval__"],
                samples=10,
                random_seed=123,
            )

        # Check values are correct
        assert np.allclose(prior.prior["ub_log__"].data, np.log(prior.prior["ub"].data))
        assert np.allclose(
            prior.prior["x_interval__"].data,
            ub_interval_forward(prior.prior["x"].data, prior.prior["ub"].data),
        )

        # Check that it works when the original RVs are not mentioned in var_names
        with pm.Model() as model_transformed_only:
            ub = pm.HalfNormal("ub", 10)
            x = pm.Uniform("x", 0, ub)

            prior_transformed_only = pm.sample_prior_predictive(
                var_names=["ub_log__", "x_interval__"],
                samples=10,
                random_seed=123,
            )
        assert (
            "ub" not in prior_transformed_only.prior.data_vars
            and "x" not in prior_transformed_only.prior.data_vars
        )
        assert np.allclose(
            prior.prior["ub_log__"].data, prior_transformed_only.prior["ub_log__"].data
        )
        assert np.allclose(
            prior.prior["x_interval__"], prior_transformed_only.prior["x_interval__"].data
        )

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
                samples=1, var_names=["a", "b", "c", "d"], random_seed=seed
            )

        with pm.Model() as m2:
            a = pm.Normal("a")
            b = pm.Normal("b")
            c = pm.Normal("c")
            d = pm.Normal("d")
            prior2 = pm.sample_prior_predictive(
                samples=1, var_names=["b", "a", "d", "c"], random_seed=seed
            )

        assert prior1.prior["a"] == prior2.prior["a"]
        assert prior1.prior["b"] == prior2.prior["b"]
        assert prior1.prior["c"] == prior2.prior["c"]
        assert prior1.prior["d"] == prior2.prior["d"]

    def test_aesara_function_kwargs(self):
        sharedvar = aesara.shared(0)
        with pm.Model() as m:
            x = pm.DiracDelta("x", 0)
            y = pm.Deterministic("y", x + sharedvar)

            prior = pm.sample_prior_predictive(
                samples=5,
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
                samples=20,
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

    def test_aesara_function_kwargs(self):
        sharedvar = aesara.shared(0)
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
            samples=2, return_inferencedata=False, random_seed=npr.RandomState(2023532)
        )

    assert X_rv.owner.inputs[0] != Y_rv.owner.inputs[0]

    with pm.Model():
        X_rv = pm.Normal("x")
        Y_rv = pm.Normal("y")

        pp_samples_2 = pm.sample_prior_predictive(
            samples=2, return_inferencedata=False, random_seed=npr.RandomState(2023532)
        )

    assert np.array_equal(pp_samples["y"], pp_samples_2["y"])


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
