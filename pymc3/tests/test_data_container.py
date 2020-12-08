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
import pandas as pd
import pytest

import pymc3 as pm

from pymc3.tests.helpers import SeededTest
from pymc3.theanof import floatX


class TestData(SeededTest):
    def test_deterministic(self):
        data_values = np.array([0.5, 0.4, 5, 2])
        with pm.Model() as model:
            X = pm.Data("X", data_values)
            pm.Normal("y", 0, 1, observed=X)
            model.logp(model.test_point)

    def test_sample(self):
        x = np.random.normal(size=100)
        y = x + np.random.normal(scale=1e-2, size=100)

        x_pred = np.linspace(-3, 3, 200, dtype="float32")

        with pm.Model():
            x_shared = pm.Data("x_shared", x)
            b = pm.Normal("b", 0.0, 10.0)
            pm.Normal("obs", b * x_shared, np.sqrt(1e-2), observed=y)

            prior_trace0 = pm.sample_prior_predictive(1000)
            trace = pm.sample(1000, init=None, tune=1000, chains=1)
            pp_trace0 = pm.sample_posterior_predictive(trace, 1000)
            pp_trace01 = pm.fast_sample_posterior_predictive(trace, 1000)

            x_shared.set_value(x_pred)
            prior_trace1 = pm.sample_prior_predictive(1000)
            pp_trace1 = pm.sample_posterior_predictive(trace, samples=1000)
            pp_trace11 = pm.fast_sample_posterior_predictive(trace, samples=1000)

        assert prior_trace0["b"].shape == (1000,)
        assert prior_trace0["obs"].shape == (1000, 100)
        assert prior_trace1["obs"].shape == (1000, 200)

        assert pp_trace0["obs"].shape == (1000, 100)
        assert pp_trace01["obs"].shape == (1000, 100)

        np.testing.assert_allclose(x, pp_trace0["obs"].mean(axis=0), atol=1e-1)
        np.testing.assert_allclose(x, pp_trace01["obs"].mean(axis=0), atol=1e-1)

        assert pp_trace1["obs"].shape == (1000, 200)
        assert pp_trace11["obs"].shape == (1000, 200)

        np.testing.assert_allclose(x_pred, pp_trace1["obs"].mean(axis=0), atol=1e-1)
        np.testing.assert_allclose(x_pred, pp_trace11["obs"].mean(axis=0), atol=1e-1)

    def test_sample_posterior_predictive_after_set_data(self):
        with pm.Model() as model:
            x = pm.Data("x", [1.0, 2.0, 3.0])
            y = pm.Data("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 10.0)
            pm.Normal("obs", beta * x, np.sqrt(1e-2), observed=y)
            trace = pm.sample(1000, tune=1000, chains=1)
        # Predict on new data.
        with model:
            x_test = [5, 6, 9]
            pm.set_data(new_data={"x": x_test})
            y_test = pm.sample_posterior_predictive(trace)
            y_test1 = pm.fast_sample_posterior_predictive(trace)

        assert y_test["obs"].shape == (1000, 3)
        assert y_test1["obs"].shape == (1000, 3)
        np.testing.assert_allclose(x_test, y_test["obs"].mean(axis=0), atol=1e-1)
        np.testing.assert_allclose(x_test, y_test1["obs"].mean(axis=0), atol=1e-1)

    def test_sample_after_set_data(self):
        with pm.Model() as model:
            x = pm.Data("x", [1.0, 2.0, 3.0])
            y = pm.Data("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 10.0)
            pm.Normal("obs", beta * x, np.sqrt(1e-2), observed=y)
            pm.sample(1000, init=None, tune=1000, chains=1)
        # Predict on new data.
        new_x = [5.0, 6.0, 9.0]
        new_y = [5.0, 6.0, 9.0]
        with model:
            pm.set_data(new_data={"x": new_x, "y": new_y})
            new_trace = pm.sample(1000, init=None, tune=1000, chains=1)
            pp_trace = pm.sample_posterior_predictive(new_trace, 1000)
            pp_tracef = pm.fast_sample_posterior_predictive(new_trace, 1000)

        assert pp_trace["obs"].shape == (1000, 3)
        assert pp_tracef["obs"].shape == (1000, 3)
        np.testing.assert_allclose(new_y, pp_trace["obs"].mean(axis=0), atol=1e-1)
        np.testing.assert_allclose(new_y, pp_tracef["obs"].mean(axis=0), atol=1e-1)

    def test_shared_data_as_index(self):
        """
        Allow pm.Data to be used for index variables, i.e with integers as well as floats.
        See https://github.com/pymc-devs/pymc3/issues/3813
        """
        with pm.Model() as model:
            index = pm.Data("index", [2, 0, 1, 0, 2])
            y = pm.Data("y", [1.0, 2.0, 3.0, 2.0, 1.0])
            alpha = pm.Normal("alpha", 0, 1.5, shape=3)
            pm.Normal("obs", alpha[index], np.sqrt(1e-2), observed=y)

            prior_trace = pm.sample_prior_predictive(1000, var_names=["alpha"])
            trace = pm.sample(1000, init=None, tune=1000, chains=1)

        # Predict on new data
        new_index = np.array([0, 1, 2])
        new_y = [5.0, 6.0, 9.0]
        with model:
            pm.set_data(new_data={"index": new_index, "y": new_y})
            pp_trace = pm.sample_posterior_predictive(trace, 1000, var_names=["alpha", "obs"])
            pp_tracef = pm.fast_sample_posterior_predictive(trace, 1000, var_names=["alpha", "obs"])

        assert prior_trace["alpha"].shape == (1000, 3)
        assert trace["alpha"].shape == (1000, 3)
        assert pp_trace["alpha"].shape == (1000, 3)
        assert pp_trace["obs"].shape == (1000, 3)
        assert pp_tracef["alpha"].shape == (1000, 3)
        assert pp_tracef["obs"].shape == (1000, 3)

    def test_shared_data_as_rv_input(self):
        """
        Allow pm.Data to be used as input for other RVs.
        See https://github.com/pymc-devs/pymc3/issues/3842
        """
        with pm.Model() as m:
            x = pm.Data("x", [1.0, 2.0, 3.0])
            _ = pm.Normal("y", mu=x, shape=3)
            trace = pm.sample(chains=1)

        np.testing.assert_allclose(np.array([1.0, 2.0, 3.0]), x.get_value(), atol=1e-1)
        np.testing.assert_allclose(np.array([1.0, 2.0, 3.0]), trace["y"].mean(0), atol=1e-1)

        with m:
            pm.set_data({"x": np.array([2.0, 4.0, 6.0])})
            trace = pm.sample(chains=1)

        np.testing.assert_allclose(np.array([2.0, 4.0, 6.0]), x.get_value(), atol=1e-1)
        np.testing.assert_allclose(np.array([2.0, 4.0, 6.0]), trace["y"].mean(0), atol=1e-1)

    def test_creation_of_data_outside_model_context(self):
        with pytest.raises((IndexError, TypeError)) as error:
            pm.Data("data", [1.1, 2.2, 3.3])
        error.match("No model on context stack")

    def test_set_data_to_non_data_container_variables(self):
        with pm.Model() as model:
            x = np.array([1.0, 2.0, 3.0])
            y = np.array([1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 10.0)
            pm.Normal("obs", beta * x, np.sqrt(1e-2), observed=y)
            pm.sample(1000, init=None, tune=1000, chains=1)
        with pytest.raises(TypeError) as error:
            pm.set_data({"beta": [1.1, 2.2, 3.3]}, model=model)
        error.match("defined as `pymc3.Data` inside the model")

    def test_model_to_graphviz_for_model_with_data_container(self):
        with pm.Model() as model:
            x = pm.Data("x", [1.0, 2.0, 3.0])
            y = pm.Data("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 10.0)
            obs_sigma = floatX(np.sqrt(1e-2))
            pm.Normal("obs", beta * x, obs_sigma, observed=y)
            pm.sample(1000, init=None, tune=1000, chains=1)

        for formatting in {"latex", "latex_with_params"}:
            with pytest.raises(ValueError, match="Unsupported formatting"):
                pm.model_to_graphviz(model, formatting=formatting)

        exp_without = [
            'x [label="x\n~\nData" shape=box style="rounded, filled"]',
            'beta [label="beta\n~\nNormal"]',
            'obs [label="obs\n~\nNormal" style=filled]',
        ]
        exp_with = [
            'x [label="x\n~\nData" shape=box style="rounded, filled"]',
            'beta [label="beta\n~\nNormal(mu=0.0, sigma=10.0)"]',
            f'obs [label="obs\n~\nNormal(mu=f(f(beta), x), sigma={obs_sigma})" style=filled]',
        ]
        for formatting, expected_substrings in [
            ("plain", exp_without),
            ("plain_with_params", exp_with),
        ]:
            g = pm.model_to_graphviz(model, formatting=formatting)
            # check formatting of RV nodes
            for expected in expected_substrings:
                assert expected in g.source

    def test_explicit_coords(self):
        N_rows = 5
        N_cols = 7
        data = np.random.uniform(size=(N_rows, N_cols))
        coords = {
            "rows": [f"R{r+1}" for r in range(N_rows)],
            "columns": [f"C{c+1}" for c in range(N_cols)],
        }
        # pass coordinates explicitly, use numpy array in Data container
        with pm.Model(coords=coords) as pmodel:
            pm.Data("observations", data, dims=("rows", "columns"))

        assert "rows" in pmodel.coords
        assert pmodel.coords["rows"] == ["R1", "R2", "R3", "R4", "R5"]
        assert "columns" in pmodel.coords
        assert pmodel.coords["columns"] == ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        assert pmodel.RV_dims == {"observations": ("rows", "columns")}

    def test_implicit_coords_series(self):
        ser_sales = pd.Series(
            data=np.random.randint(low=0, high=30, size=22),
            index=pd.date_range(start="2020-05-01", periods=22, freq="24H", name="date"),
            name="sales",
        )
        with pm.Model() as pmodel:
            pm.Data("sales", ser_sales, dims="date", export_index_as_coords=True)

        assert "date" in pmodel.coords
        assert len(pmodel.coords["date"]) == 22
        assert pmodel.RV_dims == {"sales": ("date",)}

    def test_implicit_coords_dataframe(self):
        N_rows = 5
        N_cols = 7
        df_data = pd.DataFrame()
        for c in range(N_cols):
            df_data[f"Column {c+1}"] = np.random.normal(size=(N_rows,))
        df_data.index.name = "rows"
        df_data.columns.name = "columns"

        # infer coordinates from index and columns of the DataFrame
        with pm.Model() as pmodel:
            pm.Data("observations", df_data, dims=("rows", "columns"), export_index_as_coords=True)

        assert "rows" in pmodel.coords
        assert "columns" in pmodel.coords
        assert pmodel.RV_dims == {"observations": ("rows", "columns")}


def test_data_naming():
    """
    This is a test for issue #3793 -- `Data` objects in named models are
    not given model-relative names.
    """
    with pm.Model("named_model") as model:
        x = pm.Data("x", [1.0, 2.0, 3.0])
        y = pm.Normal("y")
    assert y.name == "named_model_y"
    assert x.name == "named_model_x"
