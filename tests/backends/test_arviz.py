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
import warnings

import numpy as np
import pytensor.tensor as pt
import pytest

from arviz import InferenceData
from arviz.tests.helpers import check_multiple_attrs
from numpy import ma
from pytensor.tensor.subtensor import AdvancedIncSubtensor, AdvancedIncSubtensor1

import pymc as pm

from pymc.backends.arviz import (
    InferenceDataConverter,
    predictions_to_inference_data,
    to_inference_data,
)
from pymc.exceptions import ImputationWarning

# Turn all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


@pytest.fixture(scope="module")
def eight_schools_params():
    """Share setup for eight schools."""
    return {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }


@pytest.fixture(scope="module")
def draws():
    """Share default draw count."""
    return 500


@pytest.fixture(scope="module")
def chains():
    """Share default chain count."""
    return 2


class TestDataPyMC:
    class Data:
        def __init__(self, model, trace):
            self.model = model
            self.obj = trace

    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=5)
            tau = pm.HalfCauchy("tau", beta=5)
            eta = pm.Normal("eta", mu=0, sigma=1, size=eight_schools_params["J"])
            theta = pm.Deterministic("theta", mu + tau * eta)
            pm.Normal(
                "obs",
                mu=theta,
                sigma=eight_schools_params["sigma"],
                observed=eight_schools_params["y"],
            )
            trace = pm.sample(
                draws,
                chains=chains,
                return_inferencedata=False,
            )

        return self.Data(model, trace)

    def get_inference_data(self, data, eight_schools_params):
        with data.model:
            prior = pm.sample_prior_predictive(return_inferencedata=False)
            posterior_predictive = pm.sample_posterior_predictive(
                data.obj, return_inferencedata=False
            )

        return to_inference_data(
            trace=data.obj,
            prior=prior,
            posterior_predictive=posterior_predictive,
            log_likelihood=True,
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "eta": ["school"]},
            model=data.model,
        )

    def get_predictions_inference_data(
        self, data, eight_schools_params, inplace
    ) -> tuple[InferenceData, dict[str, np.ndarray]]:
        with data.model:
            prior = pm.sample_prior_predictive(return_inferencedata=False)
            posterior_predictive = pm.sample_posterior_predictive(
                data.obj, return_inferencedata=False
            )

            idata = to_inference_data(
                trace=data.obj,
                prior=prior,
                coords={"school": np.arange(eight_schools_params["J"])},
                dims={"theta": ["school"], "eta": ["school"]},
            )
            assert isinstance(idata, InferenceData)
            extended = predictions_to_inference_data(
                posterior_predictive, idata_orig=idata, inplace=inplace
            )
            assert isinstance(extended, InferenceData)
            assert (id(idata) == id(extended)) == inplace
        return (extended, posterior_predictive)

    def make_predictions_inference_data(
        self, data, eight_schools_params
    ) -> tuple[InferenceData, dict[str, np.ndarray]]:
        with data.model:
            posterior_predictive = pm.sample_posterior_predictive(
                data.obj, return_inferencedata=False
            )
            idata = predictions_to_inference_data(
                posterior_predictive,
                posterior_trace=data.obj,
                coords={"school": np.arange(eight_schools_params["J"])},
                dims={"theta": ["school"], "eta": ["school"]},
            )
            assert isinstance(idata, InferenceData)
        return idata, posterior_predictive

    def test_to_idata(self, data, eight_schools_params, chains, draws):
        inference_data = self.get_inference_data(data, eight_schools_params)
        test_dict = {
            "posterior": ["mu", "tau", "eta", "theta"],
            "sample_stats": ["diverging", "lp", "~log_likelihood"],
            "log_likelihood": ["obs"],
            "posterior_predictive": ["obs"],
            "prior": ["mu", "tau", "eta", "theta"],
            "prior_predictive": ["obs"],
            "observed_data": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        chains = inference_data.posterior.sizes["chain"]
        draws = inference_data.posterior.sizes["draw"]
        obs = inference_data.observed_data["obs"]
        assert inference_data.log_likelihood["obs"].shape == (chains, draws) + obs.shape

    def test_predictions_to_idata(self, data, eight_schools_params):
        "Test that we can add predictions to a previously-existing InferenceData."
        test_dict = {
            "posterior": ["mu", "tau", "eta", "theta"],
            "sample_stats": ["diverging", "lp"],
            "predictions": ["obs"],
            "prior": ["mu", "tau", "eta", "theta"],
            "observed_data": ["obs"],
        }

        # check adding non-destructively
        inference_data, _ = self.get_predictions_inference_data(data, eight_schools_params, False)
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        for key, ivalues in inference_data.predictions.items():
            assert (
                len(ivalues["chain"]) == inference_data.posterior.sizes["chain"]
            )  # same chains as in posterior

        # check adding in place
        inference_data, posterior_predictive = self.get_predictions_inference_data(
            data, eight_schools_params, True
        )
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        for key, ivalues in inference_data.predictions.items():
            assert (
                len(ivalues["chain"]) == inference_data.posterior.sizes["chain"]
            )  # same chains as in posterior

    def test_predictions_to_idata_new(self, data, eight_schools_params):
        # check creating new
        inference_data, posterior_predictive = self.make_predictions_inference_data(
            data, eight_schools_params
        )
        test_dict = {
            "posterior": ["mu", "tau", "eta", "theta"],
            "predictions": ["obs"],
            "~observed_data": "",
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        for key, values in posterior_predictive.items():
            ivalues = inference_data.predictions[key]
            assert (len(ivalues["chain"]) == 2) and (len(ivalues["draw"]) == 500)

    def test_posterior_predictive_keep_size(self, data, chains, draws, eight_schools_params):
        with data.model:
            posterior_predictive = pm.sample_posterior_predictive(
                data.obj, return_inferencedata=False
            )
            inference_data = to_inference_data(
                trace=data.obj,
                posterior_predictive=posterior_predictive,
                coords={"school": np.arange(eight_schools_params["J"])},
                dims={"theta": ["school"], "eta": ["school"]},
            )

        shape = inference_data.posterior_predictive.obs.shape
        assert np.all(
            [obs_s == s for obs_s, s in zip(shape, (chains, draws, eight_schools_params["J"]))]
        )

    def test_posterior_predictive_thinned(self, data):
        with data.model:
            draws = 20
            thin_by = 4
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                idata = pm.sample(tune=5, draws=draws, chains=2, return_inferencedata=True)
            thinned_idata = idata.sel(draw=slice(None, None, thin_by))
            idata.extend(pm.sample_posterior_predictive(thinned_idata))
        test_dict = {
            "posterior": ["mu", "tau", "eta", "theta"],
            "sample_stats": ["diverging", "lp", "~log_likelihood"],
            "posterior_predictive": ["obs"],
            "observed_data": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails
        assert idata.posterior.sizes["chain"] == 2
        assert idata.posterior.sizes["draw"] == draws
        assert idata.posterior_predictive.sizes["chain"] == 2
        assert idata.posterior_predictive.sizes["draw"] == draws / thin_by
        assert np.allclose(idata.posterior["draw"], np.arange(draws))
        assert np.allclose(idata.posterior_predictive["draw"], np.arange(draws, step=thin_by))

    @pytest.mark.parametrize("use_context", [True, False])
    def test_autodetect_coords_from_model(self, use_context):
        pd = pytest.importorskip("pandas")
        df_data = pd.DataFrame(columns=["date"]).set_index("date")
        dates = pd.date_range(start="2020-05-01", end="2020-05-20")
        for city, mu in {"Berlin": 15, "San Marino": 18, "Paris": 16}.items():
            df_data[city] = np.random.normal(loc=mu, size=len(dates))
        df_data.index = dates
        df_data.index.name = "date"

        coords = {"date": df_data.index, "city": df_data.columns}
        with pm.Model(coords=coords) as model:
            europe_mean = pm.Normal("europe_mean_temp", mu=15.0, sigma=3.0)
            city_offset = pm.Normal("city_offset", mu=0.0, sigma=3.0, dims="city")
            city_temperature = pm.Deterministic(
                "city_temperature", europe_mean + city_offset, dims="city"
            )

            data_dims = ("date", "city")
            data = pm.ConstantData("data", df_data, dims=data_dims)
            _ = pm.Normal(
                "likelihood", mu=city_temperature, sigma=0.5, observed=data, dims=data_dims
            )

            trace = pm.sample(
                return_inferencedata=False,
                compute_convergence_checks=False,
                cores=1,
                chains=1,
                tune=20,
                draws=30,
                step=pm.Metropolis(),
            )
            if use_context:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "More chains .* than draws.*", UserWarning)
                    idata = to_inference_data(trace=trace)
        if not use_context:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "More chains .* than draws.*", UserWarning)
                idata = to_inference_data(trace=trace, model=model)

        assert "city" in list(idata.posterior.dims)
        assert "city" in list(idata.observed_data.dims)
        assert "date" in list(idata.observed_data.dims)

        np.testing.assert_array_equal(idata.posterior.coords["city"], coords["city"])
        np.testing.assert_array_equal(idata.observed_data.coords["date"], coords["date"])
        np.testing.assert_array_equal(idata.observed_data.coords["city"], coords["city"])

    def test_overwrite_model_coords_dims(self):
        """Check coords and dims from model object can be partially overwritten."""
        dim1 = ["a", "b"]
        new_dim1 = ["c", "d"]
        coords = {"dim1": dim1, "dim2": ["c1", "c2"]}
        x_data = np.arange(4).reshape((2, 2))
        y = x_data + np.random.normal(size=(2, 2))
        with pm.Model(coords=coords):
            x = pm.ConstantData("x", x_data, dims=("dim1", "dim2"))
            beta = pm.Normal("beta", 0, 1, dims="dim1")
            _ = pm.Normal("obs", x * beta, 1, observed=y, dims=("dim1", "dim2"))
            trace = pm.sample(100, tune=100, return_inferencedata=False)
            idata1 = to_inference_data(trace)
            idata2 = to_inference_data(trace, coords={"dim1": new_dim1}, dims={"beta": ["dim2"]})

        test_dict = {"posterior": ["beta"], "observed_data": ["obs"], "constant_data": ["x"]}
        fails1 = check_multiple_attrs(test_dict, idata1)
        assert not fails1
        fails2 = check_multiple_attrs(test_dict, idata2)
        assert not fails2
        assert "dim1" in list(idata1.posterior.beta.dims)
        assert "dim2" in list(idata2.posterior.beta.dims)
        assert np.all(idata1.constant_data.x.dim1.values == np.array(dim1))
        assert np.all(idata1.constant_data.x.dim2.values == np.array(["c1", "c2"]))
        assert np.all(idata2.constant_data.x.dim1.values == np.array(new_dim1))
        assert np.all(idata2.constant_data.x.dim2.values == np.array(["c1", "c2"]))

    def test_missing_data_model(self):
        # source tests/test_missing.py
        data = ma.masked_values([1, 2, -1, 4, -1], value=-1)
        model = pm.Model()
        with model:
            x = pm.Normal("x", 1, 1)
            with pytest.warns(ImputationWarning):
                y = pm.Normal("y", x, 1, observed=data)
            inference_data = pm.sample(
                100, chains=2, return_inferencedata=True, idata_kwargs=dict(log_likelihood=True)
            )

        # make sure that data is really missing
        assert "y_unobserved" in model.named_vars

        test_dict = {
            "posterior": ["x", "y_unobserved"],
            "observed_data": ["y_observed"],
            "log_likelihood": ["y_observed"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

        # The missing part of partial observed RVs is not included in log_likelihood
        # See https://github.com/pymc-devs/pymc/issues/5255
        assert inference_data.log_likelihood["y_observed"].shape == (2, 100, 3)

    def test_mv_missing_data_model(self):
        data = ma.masked_values([[1, 2], [2, 2], [-1, 4], [2, -1], [-1, -1]], value=-1)

        model = pm.Model()
        with model:
            mu = pm.Normal("mu", 0, 1, size=2)
            sd_dist = pm.HalfNormal.dist(1.0, size=2)
            chol, *_ = pm.LKJCholeskyCov("chol_cov", n=2, eta=1, sd_dist=sd_dist)
            with pytest.warns(ImputationWarning):
                y = pm.MvNormal("y", mu=mu, chol=chol, observed=data)
            inference_data = pm.sample(
                tune=10,
                draws=10,
                chains=2,
                step=pm.Metropolis(),
                idata_kwargs=dict(log_likelihood=True),
            )

        # make sure that data is really missing
        assert isinstance(y.owner.inputs[0].owner.op, (AdvancedIncSubtensor, AdvancedIncSubtensor1))

        test_dict = {
            "posterior": ["mu", "chol_cov"],
            "observed_data": ["y_observed"],
            "log_likelihood": ["y_observed"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    @pytest.mark.parametrize("log_likelihood", [True, False, ["y1"]])
    def test_multiple_observed_rv(self, log_likelihood):
        y1_data = np.random.randn(10)
        y2_data = np.random.randn(100)
        with pm.Model():
            x = pm.Normal("x", 1, 1)
            pm.Normal("y1", x, 1, observed=y1_data)
            pm.Normal("y2", x, 1, observed=y2_data)
            inference_data = pm.sample(
                100,
                chains=2,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": log_likelihood},
            )
        test_dict = {
            "posterior": ["x"],
            "observed_data": ["y1", "y2"],
            "log_likelihood": ["y1", "y2"],
            "sample_stats": ["diverging", "lp", "~log_likelihood"],
        }
        if not log_likelihood:
            test_dict.pop("log_likelihood")
            test_dict["~log_likelihood"] = []
        elif isinstance(log_likelihood, list):
            test_dict["log_likelihood"] = ["y1", "~y2"]
            assert inference_data.log_likelihood["y1"].shape == (2, 100, 10)
        else:
            assert inference_data.log_likelihood["y1"].shape == (2, 100, 10)
            assert inference_data.log_likelihood["y2"].shape == (2, 100, 100)

        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_single_observation(self):
        with pm.Model():
            p = pm.Uniform("p", 0, 1)
            pm.Binomial("w", p=p, n=2, observed=[1])
            inference_data = pm.sample(
                500, chains=2, return_inferencedata=True, idata_kwargs=dict(log_likelihood=True)
            )

        assert inference_data
        assert inference_data.log_likelihood["w"].shape == (2, 500, 1)

    def test_potential(self):
        with pm.Model():
            x = pm.Normal("x", 0.0, 1.0)
            pm.Potential("z", pm.logp(pm.Normal.dist(x, 1.0), np.random.randn(10)))
            inference_data = pm.sample(100, chains=2, return_inferencedata=True)

        assert inference_data

    @pytest.mark.parametrize("use_context", [True, False])
    def test_constant_data(self, use_context):
        """Test constant_data group behaviour."""
        with pm.Model() as model:
            x = pm.ConstantData("x", [1.0, 2.0, 3.0])
            y = pm.MutableData("y", [1.0, 2.0, 3.0])
            beta_sigma = pm.MutableData("beta_sigma", 1)
            beta = pm.Normal("beta", 0, beta_sigma)
            obs = pm.Normal("obs", x * beta, 1, observed=y)
            trace = pm.sample(100, chains=2, tune=100, return_inferencedata=False)
            if use_context:
                inference_data = to_inference_data(trace=trace, log_likelihood=True)

        if not use_context:
            inference_data = to_inference_data(trace=trace, model=model, log_likelihood=True)
        test_dict = {
            "posterior": ["beta"],
            "observed_data": ["obs"],
            "constant_data": ["x", "y", "beta_sigma"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        assert inference_data.log_likelihood["obs"].shape == (2, 100, 3)
        # test that scalars are dimensionless in constant_data (issue #6755)
        assert inference_data.constant_data["beta_sigma"].ndim == 0

    def test_predictions_constant_data(self):
        with pm.Model():
            x = pm.ConstantData("x", [1.0, 2.0, 3.0])
            y = pm.MutableData("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 1)
            obs = pm.Normal("obs", x * beta, 1, observed=y)
            trace = pm.sample(100, tune=100, return_inferencedata=False)
            inference_data = to_inference_data(trace)

        test_dict = {"posterior": ["beta"], "observed_data": ["obs"], "constant_data": ["x"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

        with pm.Model():
            x = pm.MutableData("x", [1.0, 2.0])
            y = pm.ConstantData("y", [1.0, 2.0])
            beta = pm.Normal("beta", 0, 1)
            obs = pm.Normal("obs", x * beta, 1, observed=y)
            predictive_trace = pm.sample_posterior_predictive(
                inference_data, return_inferencedata=False
            )
            assert set(predictive_trace.keys()) == {"obs"}
            # this should be four chains of 100 samples
            # assert predictive_trace["obs"].shape == (400, 2)
            # but the shape seems to vary between pymc versions
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "More chains .* than draws.*", UserWarning)
                inference_data = predictions_to_inference_data(
                    predictive_trace, posterior_trace=trace
                )
        test_dict = {"posterior": ["beta"], "~observed_data": ""}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, "Posterior data not copied over as expected."
        test_dict = {"predictions": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, "Predictions not instantiated as expected."
        test_dict = {"predictions_constant_data": ["x"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, "Predictions constant data not instantiated as expected."

    def test_no_trace(self):
        with pm.Model() as model:
            x = pm.ConstantData("x", [1.0, 2.0, 3.0])
            y = pm.MutableData("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 1)
            obs = pm.Normal("obs", x * beta, 1, observed=y)
            idata = pm.sample(100, tune=100)
            prior = pm.sample_prior_predictive(return_inferencedata=False)
            posterior_predictive = pm.sample_posterior_predictive(idata, return_inferencedata=False)

        # Only prior
        inference_data = to_inference_data(prior=prior, model=model)
        test_dict = {"prior": ["beta"], "prior_predictive": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        # Only posterior_predictive
        inference_data = to_inference_data(posterior_predictive=posterior_predictive, model=model)
        test_dict = {"posterior_predictive": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        # Prior and posterior_predictive but no trace
        inference_data = to_inference_data(
            prior=prior, posterior_predictive=posterior_predictive, model=model
        )
        test_dict = {
            "prior": ["beta"],
            "prior_predictive": ["obs"],
            "posterior_predictive": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    @pytest.mark.parametrize("use_context", [True, False])
    def test_priors_separation(self, use_context):
        """Test model is enough to get prior, prior predictive, constant_data and observed_data."""
        with pm.Model() as model:
            x = pm.MutableData("x", [1.0, 2.0, 3.0])
            y = pm.ConstantData("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 1)
            obs = pm.Normal("obs", x * beta, 1, observed=y)
            prior = pm.sample_prior_predictive(return_inferencedata=False)

        test_dict = {
            "prior": ["beta", "~obs"],
            "observed_data": ["obs"],
            "prior_predictive": ["obs"],
            "constant_data": ["x", "y"],
        }
        if use_context:
            with model:
                inference_data = to_inference_data(prior=prior)
        else:
            inference_data = to_inference_data(prior=prior, model=model)
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_conversion_from_variables_subset(self):
        """This is a regression test for issue #5337."""
        with pm.Model() as model:
            x = pm.Normal("x")
            pm.Normal("y", x, observed=5)
            idata = pm.sample(
                tune=10, draws=20, chains=1, step=pm.Metropolis(), compute_convergence_checks=False
            )
            pm.sample_posterior_predictive(idata, var_names=["x"])
            pm.sample_prior_predictive(var_names=["x"])

    def test_multivariate_observations(self):
        coords = {"direction": ["x", "y", "z"], "experiment": np.arange(20)}
        data = np.random.multinomial(20, [0.2, 0.3, 0.5], size=20)
        with pm.Model(coords=coords):
            p = pm.Beta("p", 1, 1, size=(3,))
            p = p / p.sum()
            pm.Multinomial("y", 20, p, dims=("experiment", "direction"), observed=data)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                idata = pm.sample(
                    draws=50,
                    chains=2,
                    tune=100,
                    return_inferencedata=True,
                    idata_kwargs=dict(log_likelihood=True),
                )
        test_dict = {
            "posterior": ["p"],
            "sample_stats": ["lp"],
            "log_likelihood": ["y"],
            "observed_data": ["y"],
        }
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails
        assert "direction" not in idata.log_likelihood.dims
        assert "direction" in idata.observed_data.dims
        assert idata.log_likelihood["y"].shape == (2, 50, 20)

    def test_constant_data_coords_issue_5046(self):
        """This is a regression test against a bug where a local coords variable was overwritten."""
        dims = {"alpha": ["backwards"], "bravo": ["letters", "yesno"]}
        coords = {
            "backwards": np.arange(17)[::-1],
            "letters": list("ABCDEFGHIJK"),
            "yesno": ["yes", "no"],
        }
        data = {
            name: np.random.uniform(size=[len(coords[dn]) for dn in dnames])
            for name, dnames in dims.items()
        }

        for k in data:
            assert len(data[k].shape) == len(dims[k])

        ds = pm.backends.arviz.dict_to_dataset(
            data=data, library=pm, coords=coords, dims=dims, default_dims=[], index_origin=0
        )
        for dname, cvals in coords.items():
            np.testing.assert_array_equal(ds[dname].values, cvals)

    def test_issue_5043_autoconvert_coord_values(self):
        pd = pytest.importorskip("pandas")
        coords = {"city": pd.Series(["Bonn", "Berlin"])}
        with pm.Model(coords=coords) as pmodel:
            # The model tracks coord values as (immutable) tuples
            assert isinstance(pmodel.coords["city"], tuple)
            pm.Normal("x", dims="city")
            mtrace = pm.sample(
                return_inferencedata=False,
                compute_convergence_checks=False,
                step=pm.Metropolis(),
                cores=1,
                tune=7,
                draws=15,
            )
            # The converter must convert coord values them to numpy arrays
            # because tuples as coordinate values causes problems with xarray.
            converter = InferenceDataConverter(trace=mtrace)
            assert isinstance(converter.coords["city"], np.ndarray)
            converter.to_inference_data()

            # We're not automatically converting things other than tuple,
            # so advanced use cases remain supported at the InferenceData level.
            # They just can't be used in the model construction already.
            converter = InferenceDataConverter(
                trace=mtrace,
                coords={
                    "city": pd.MultiIndex.from_tuples(
                        [("Bonn", 53111), ("Berlin", 10178)], names=["name", "zipcode"]
                    )
                },
            )
            assert isinstance(converter.coords["city"], pd.MultiIndex)

    def test_variable_dimension_name_collision(self):
        with pytest.raises(ValueError, match="same name as its dimension"):
            with pm.Model() as pmodel:
                var = pt.as_tensor([1, 2, 3])
                pmodel.register_rv(var, name="time", dims=("time",))

    def test_include_transformed(self):
        with pm.Model():
            pm.Uniform("p", 0, 1)

            # First check that the default is to exclude the transformed variables
            sample_kwargs = dict(
                tune=5, draws=7, chains=2, cores=1, compute_convergence_checks=False
            )
            inference_data = pm.sample(**sample_kwargs, step=pm.Metropolis())
            assert "p_interval__" not in inference_data.posterior

            # Now check that they are included when requested
            inference_data = pm.sample(
                **sample_kwargs,
                step=pm.Metropolis(),
                idata_kwargs={"include_transformed": True},
            )
            assert "p_interval__" in inference_data.posterior

    @pytest.mark.parametrize("chains", (1, 2))
    def test_single_chain(self, chains):
        # Test that no UserWarning is raised when sampling with NUTS defaults

        # When this test was added, a `UserWarning: More chains (500) than draws (1)` used to be issued
        # when sampling with a single chain
        warnings.simplefilter("error")
        with pm.Model():
            pm.Normal("x")
            pm.sample(chains=chains, return_inferencedata=True)


class TestPyMCWarmupHandling:
    @pytest.mark.parametrize("save_warmup", [False, True])
    @pytest.mark.parametrize("chains", [1, 2])
    @pytest.mark.parametrize("tune,draws", [(0, 50), (10, 40), (30, 0)])
    def test_save_warmup(self, save_warmup, chains, tune, draws):
        with pm.Model():
            pm.Uniform("u1")
            pm.Normal("n1")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                warnings.filterwarnings("ignore", "More chains .* than draws.*", UserWarning)
                idata = pm.sample(
                    tune=tune,
                    draws=draws,
                    chains=chains,
                    cores=1,
                    step=pm.Metropolis(),
                    discard_tuned_samples=False,
                    return_inferencedata=True,
                    idata_kwargs={"save_warmup": save_warmup},
                )
        warmup_prefix = "" if save_warmup and (tune > 0) else "~"
        post_prefix = "" if draws > 0 else "~"
        test_dict = {
            f"{post_prefix}posterior": ["u1", "n1"],
            f"{post_prefix}sample_stats": ["~tune", "accept"],
            f"{warmup_prefix}warmup_posterior": ["u1", "n1"],
            f"{warmup_prefix}warmup_sample_stats": ["~tune"],
            "~warmup_log_likelihood": [],
            "~log_likelihood": [],
        }
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails
        if hasattr(idata, "posterior"):
            assert idata.posterior.sizes["chain"] == chains
            assert idata.posterior.sizes["draw"] == draws
        if hasattr(idata, "warmup_posterior"):
            assert idata.warmup_posterior.sizes["chain"] == chains
            assert idata.warmup_posterior.sizes["draw"] == tune

    def test_save_warmup_issue_1208_after_3_9(self):
        with pm.Model():
            pm.Uniform("u1")
            pm.Normal("n1")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Tuning samples will be included.*", UserWarning)
                trace = pm.sample(
                    tune=100,
                    draws=200,
                    chains=2,
                    cores=1,
                    step=pm.Metropolis(),
                    discard_tuned_samples=False,
                    return_inferencedata=False,
                )
            assert isinstance(trace, pm.backends.base.MultiTrace)
            assert len(trace) == 300

            # from original trace, warmup draws should be separated out
            idata = to_inference_data(trace, save_warmup=True)
            test_dict = {
                "posterior": ["u1", "n1"],
                "sample_stats": ["~tune", "accept"],
                "warmup_posterior": ["u1", "n1"],
                "warmup_sample_stats": ["~tune", "accept"],
            }
            fails = check_multiple_attrs(test_dict, idata)
            assert not fails
            assert idata.posterior.sizes["chain"] == 2
            assert idata.posterior.sizes["draw"] == 200

            # manually sliced trace triggers the same warning as <=3.8
            with pytest.warns(UserWarning, match="Warmup samples"):
                idata = to_inference_data(trace[-30:], save_warmup=True)
            test_dict = {
                "posterior": ["u1", "n1"],
                "sample_stats": ["~tune", "accept"],
                "~warmup_posterior": [],
                "~warmup_sample_stats": [],
            }
            fails = check_multiple_attrs(test_dict, idata)
            assert not fails
            assert idata.posterior.sizes["chain"] == 2
            assert idata.posterior.sizes["draw"] == 30
