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

import io

import numpy as np
import pytest

from aesara import shared
from aesara.compile.sharedvalue import SharedVariable
from aesara.tensor.sharedvar import ScalarSharedVariable
from aesara.tensor.var import TensorVariable

import pymc as pm

from pymc.aesaraf import floatX
from pymc.exceptions import ShapeError
from pymc.tests.helpers import SeededTest


class TestData(SeededTest):
    def test_deterministic(self):
        data_values = np.array([0.5, 0.4, 5, 2])
        with pm.Model() as model:
            X = pm.MutableData("X", data_values)
            pm.Normal("y", 0, 1, observed=X)
            model.compile_logp()(model.compute_initial_point())

    def test_sample(self):
        x = np.random.normal(size=100)
        y = x + np.random.normal(scale=1e-2, size=100)

        x_pred = np.linspace(-3, 3, 200, dtype="float32")

        with pm.Model():
            x_shared = pm.MutableData("x_shared", x)
            b = pm.Normal("b", 0.0, 10.0)
            pm.Normal("obs", b * x_shared, np.sqrt(1e-2), observed=y)

            prior_trace0 = pm.sample_prior_predictive(1000)
            idata = pm.sample(1000, tune=1000, chains=1)
            pp_trace0 = pm.sample_posterior_predictive(idata)

            x_shared.set_value(x_pred)
            prior_trace1 = pm.sample_prior_predictive(1000)
            pp_trace1 = pm.sample_posterior_predictive(idata)

        assert prior_trace0.prior["b"].shape == (1, 1000)
        assert prior_trace0.prior_predictive["obs"].shape == (1, 1000, 100)
        assert prior_trace1.prior_predictive["obs"].shape == (1, 1000, 200)

        assert pp_trace0.posterior_predictive["obs"].shape == (1, 1000, 100)
        np.testing.assert_allclose(
            x, pp_trace0.posterior_predictive["obs"].mean(("chain", "draw")), atol=1e-1
        )

        assert pp_trace1.posterior_predictive["obs"].shape == (1, 1000, 200)
        np.testing.assert_allclose(
            x_pred, pp_trace1.posterior_predictive["obs"].mean(("chain", "draw")), atol=1e-1
        )

    def test_sample_posterior_predictive_after_set_data(self):
        with pm.Model() as model:
            x = pm.MutableData("x", [1.0, 2.0, 3.0])
            y = pm.ConstantData("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 10.0)
            pm.Normal("obs", beta * x, np.sqrt(1e-2), observed=y)
            trace = pm.sample(
                1000,
                tune=1000,
                chains=1,
                return_inferencedata=False,
                compute_convergence_checks=False,
            )
        # Predict on new data.
        with model:
            x_test = [5, 6, 9]
            pm.set_data(new_data={"x": x_test})
            y_test = pm.sample_posterior_predictive(trace)

        assert y_test.posterior_predictive["obs"].shape == (1, 1000, 3)
        np.testing.assert_allclose(
            x_test, y_test.posterior_predictive["obs"].mean(("chain", "draw")), atol=1e-1
        )

    def test_sample_after_set_data(self):
        with pm.Model() as model:
            x = pm.MutableData("x", [1.0, 2.0, 3.0])
            y = pm.MutableData("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 10.0)
            pm.Normal("obs", beta * x, np.sqrt(1e-2), observed=y)
            pm.sample(
                1000,
                tune=1000,
                chains=1,
                compute_convergence_checks=False,
            )
        # Predict on new data.
        new_x = [5.0, 6.0, 9.0]
        new_y = [5.0, 6.0, 9.0]
        with model:
            pm.set_data(new_data={"x": new_x, "y": new_y})
            new_idata = pm.sample(
                1000,
                tune=1000,
                chains=1,
                compute_convergence_checks=False,
            )
            pp_trace = pm.sample_posterior_predictive(new_idata)

        assert pp_trace.posterior_predictive["obs"].shape == (1, 1000, 3)
        np.testing.assert_allclose(
            new_y, pp_trace.posterior_predictive["obs"].mean(("chain", "draw")), atol=0.015
        )

    def test_shared_data_as_index(self):
        """
        Allow pm.Data to be used for index variables, i.e with integers as well as floats.
        See https://github.com/pymc-devs/pymc/issues/3813
        """
        with pm.Model() as model:
            index = pm.MutableData("index", [2, 0, 1, 0, 2])
            y = pm.MutableData("y", [1.0, 2.0, 3.0, 2.0, 1.0])
            alpha = pm.Normal("alpha", 0, 1.5, size=3)
            pm.Normal("obs", alpha[index], np.sqrt(1e-2), observed=y)

            prior_trace = pm.sample_prior_predictive(1000)
            idata = pm.sample(
                1000,
                tune=1000,
                chains=1,
                compute_convergence_checks=False,
            )

        # Predict on new data
        new_index = np.array([0, 1, 2])
        new_y = [5.0, 6.0, 9.0]
        with model:
            pm.set_data(new_data={"index": new_index, "y": new_y})
            pp_trace = pm.sample_posterior_predictive(idata, var_names=["alpha", "obs"])

        assert prior_trace.prior["alpha"].shape == (1, 1000, 3)
        assert idata.posterior["alpha"].shape == (1, 1000, 3)
        assert pp_trace.posterior_predictive["alpha"].shape == (1, 1000, 3)
        assert pp_trace.posterior_predictive["obs"].shape == (1, 1000, 3)

    def test_shared_data_as_rv_input(self):
        """
        Allow pm.Data to be used as input for other RVs.
        See https://github.com/pymc-devs/pymc/issues/3842
        """
        with pm.Model() as m:
            x = pm.MutableData("x", [1.0, 2.0, 3.0])
            y = pm.Normal("y", mu=x, size=(2, 3))
            assert y.eval().shape == (2, 3)
            idata = pm.sample(
                chains=1,
                tune=500,
                draws=550,
                return_inferencedata=True,
                compute_convergence_checks=False,
            )
        samples = idata.posterior["y"]
        assert samples.shape == (1, 550, 2, 3)

        np.testing.assert_allclose(np.array([1.0, 2.0, 3.0]), x.get_value(), atol=1e-1)
        np.testing.assert_allclose(
            np.array([1.0, 2.0, 3.0]), samples.mean(("chain", "draw", "y_dim_0")), atol=1e-1
        )

        with m:
            pm.set_data({"x": np.array([2.0, 4.0, 6.0])})
            assert y.eval().shape == (2, 3)
            idata = pm.sample(
                chains=1,
                tune=500,
                draws=620,
                return_inferencedata=True,
                compute_convergence_checks=False,
            )
        samples = idata.posterior["y"]
        assert samples.shape == (1, 620, 2, 3)

        np.testing.assert_allclose(np.array([2.0, 4.0, 6.0]), x.get_value(), atol=1e-1)
        np.testing.assert_allclose(
            np.array([2.0, 4.0, 6.0]), samples.mean(("chain", "draw", "y_dim_0")), atol=1e-1
        )

    def test_shared_scalar_as_rv_input(self):
        # See https://github.com/pymc-devs/pymc/issues/3139
        with pm.Model() as m:
            shared_var = shared(5.0)
            v = pm.Normal("v", mu=shared_var, size=1)

        m_logp_fn = m.compile_logp()

        np.testing.assert_allclose(
            m_logp_fn({"v": np.r_[5.0]}),
            -0.91893853,
            rtol=1e-5,
        )

        shared_var.set_value(10.0)

        np.testing.assert_allclose(
            m_logp_fn({"v": np.r_[10.0]}),
            -0.91893853,
            rtol=1e-5,
        )

    def test_creation_of_data_outside_model_context(self):
        with pytest.raises((IndexError, TypeError)) as error:
            pm.ConstantData("data", [1.1, 2.2, 3.3])
        error.match("No model on context stack")

    def test_set_data_to_non_data_container_variables(self):
        with pm.Model() as model:
            x = np.array([1.0, 2.0, 3.0])
            y = np.array([1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 10.0)
            pm.Normal("obs", beta * x, np.sqrt(1e-2), observed=y)
            pm.sample(
                1000,
                tune=1000,
                chains=1,
                compute_convergence_checks=False,
            )
        with pytest.raises(TypeError) as error:
            pm.set_data({"beta": [1.1, 2.2, 3.3]}, model=model)
        error.match("The variable `beta` must be a `SharedVariable`")

    @pytest.mark.xfail(reason="Depends on ModelGraph")
    def test_model_to_graphviz_for_model_with_data_container(self):
        with pm.Model() as model:
            x = pm.ConstantData("x", [1.0, 2.0, 3.0])
            y = pm.MutableData("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 10.0)
            obs_sigma = floatX(np.sqrt(1e-2))
            pm.Normal("obs", beta * x, obs_sigma, observed=y)
            pm.sample(
                1000,
                tune=1000,
                chains=1,
                compute_convergence_checks=False,
            )

        for formatting in {"latex", "latex_with_params"}:
            with pytest.raises(ValueError, match="Unsupported formatting"):
                pm.model_to_graphviz(model, formatting=formatting)

        exp_without = [
            'x [label="x\n~\nConstantData" shape=box style="rounded, filled"]',
            'y [label="x\n~\nMutableData" shape=box style="rounded, filled"]',
            'beta [label="beta\n~\nNormal"]',
            'obs [label="obs\n~\nNormal" style=filled]',
        ]
        exp_with = [
            'x [label="x\n~\nConstantData" shape=box style="rounded, filled"]',
            'y [label="x\n~\nMutableData" shape=box style="rounded, filled"]',
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
            pm.MutableData("observations", data, dims=("rows", "columns"))

        assert "rows" in pmodel.coords
        assert pmodel.coords["rows"] == ("R1", "R2", "R3", "R4", "R5")
        assert "rows" in pmodel.dim_lengths
        assert isinstance(pmodel.dim_lengths["rows"], ScalarSharedVariable)
        assert pmodel.dim_lengths["rows"].eval() == 5
        assert "columns" in pmodel.coords
        assert pmodel.coords["columns"] == ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
        assert pmodel.RV_dims == {"observations": ("rows", "columns")}
        assert "columns" in pmodel.dim_lengths
        assert isinstance(pmodel.dim_lengths["columns"], ScalarSharedVariable)
        assert pmodel.dim_lengths["columns"].eval() == 7

    def test_symbolic_coords(self):
        """
        In v4 dimensions can be created without passing coordinate values.
        Their lengths are then automatically linked to the corresponding Tensor dimension.
        """
        with pm.Model() as pmodel:
            intensity = pm.MutableData("intensity", np.ones((2, 3)), dims=("row", "column"))
            assert "row" in pmodel.dim_lengths
            assert "column" in pmodel.dim_lengths
            assert isinstance(pmodel.dim_lengths["row"], TensorVariable)
            assert isinstance(pmodel.dim_lengths["column"], TensorVariable)
            assert pmodel.dim_lengths["row"].eval() == 2
            assert pmodel.dim_lengths["column"].eval() == 3

            intensity.set_value(floatX(np.ones((4, 5))))
            assert pmodel.dim_lengths["row"].eval() == 4
            assert pmodel.dim_lengths["column"].eval() == 5

    def test_no_resize_of_implied_dimensions(self):
        with pm.Model() as pmodel:
            # Imply a dimension through RV params
            pm.Normal("n", mu=[1, 2, 3], dims="city")
            # _Use_ the dimension for a data variable
            inhabitants = pm.MutableData("inhabitants", [100, 200, 300], dims="city")

            # Attempting to re-size the dimension through the data variable would
            # cause shape problems in InferenceData conversion, because the RV remains (3,).
            with pytest.raises(
                ShapeError, match="was initialized from 'n' which is not a shared variable"
            ):
                pmodel.set_data("inhabitants", [1, 2, 3, 4])

    def test_implicit_coords_series(self):
        pd = pytest.importorskip("pandas")
        ser_sales = pd.Series(
            data=np.random.randint(low=0, high=30, size=22),
            index=pd.date_range(start="2020-05-01", periods=22, freq="24H", name="date"),
            name="sales",
        )
        with pm.Model() as pmodel:
            pm.ConstantData("sales", ser_sales, dims="date", export_index_as_coords=True)

        assert "date" in pmodel.coords
        assert len(pmodel.coords["date"]) == 22
        assert pmodel.RV_dims == {"sales": ("date",)}

    def test_implicit_coords_dataframe(self):
        pd = pytest.importorskip("pandas")
        N_rows = 5
        N_cols = 7
        df_data = pd.DataFrame()
        for c in range(N_cols):
            df_data[f"Column {c+1}"] = np.random.normal(size=(N_rows,))
        df_data.index.name = "rows"
        df_data.columns.name = "columns"

        # infer coordinates from index and columns of the DataFrame
        with pm.Model() as pmodel:
            pm.ConstantData(
                "observations", df_data, dims=("rows", "columns"), export_index_as_coords=True
            )

        assert "rows" in pmodel.coords
        assert "columns" in pmodel.coords
        assert pmodel.RV_dims == {"observations": ("rows", "columns")}

    def test_data_kwargs(self):
        strict_value = True
        allow_downcast_value = False
        with pm.Model():
            data = pm.MutableData(
                "mdata",
                value=[[1.0], [2.0], [3.0]],
                strict=strict_value,
                allow_downcast=allow_downcast_value,
            )
        assert data.container.strict is strict_value
        assert data.container.allow_downcast is allow_downcast_value

    def test_data_mutable_default_warning(self):
        with pm.Model():
            with pytest.warns(FutureWarning, match="`mutable` kwarg was not specified"):
                data = pm.Data("x", [1, 2, 3])
            assert isinstance(data, SharedVariable)
        pass


def test_data_naming():
    """
    This is a test for issue #3793 -- `Data` objects in named models are
    not given model-relative names.
    """
    with pm.Model("named_model") as model:
        x = pm.ConstantData("x", [1.0, 2.0, 3.0])
        y = pm.Normal("y")
    assert y.name == "named_model::y"
    assert x.name == "named_model::x"


def test_get_data():
    data = pm.get_data("radon.csv")
    assert type(data) == io.BytesIO
