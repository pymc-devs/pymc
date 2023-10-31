#   Copyright 2023 The PyMC Developers
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
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.tensor.variable import TensorConstant

import pymc as pm

from pymc.exceptions import ImputationWarning
from pymc.model_graph import ModelGraph, model_to_graphviz, model_to_networkx


def school_model():
    """
    Schools model to use in testing model_to_networkx function
    """
    J = 8
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
    with pm.Model() as schools:
        eta = pm.Normal("eta", 0, 1, shape=J)
        mu = pm.Normal("mu", 0, sigma=1e6)
        tau = pm.HalfCauchy("tau", 25)
        theta = mu + tau * eta
        obs = pm.Normal("obs", theta, sigma=sigma, observed=y)
    return schools


class BaseModelNXTest:
    network_model = {
        "graph_attr_dict_factory": dict,
        "node_dict_factory": dict,
        "node_attr_dict_factory": dict,
        "adjlist_outer_dict_factory": dict,
        "adjlist_inner_dict_factory": dict,
        "edge_attr_dict_factory": dict,
        "graph": {"name": "", "label": "8"},
        "_node": {
            "eta": {
                "shape": "ellipse",
                "style": "rounded",
                "label": "eta\n~\nNormal",
                "cluster": "cluster8",
                "labeljust": "r",
                "labelloc": "b",
            },
            "obs": {
                "shape": "ellipse",
                "style": "rounded",
                "label": "obs\n~\nNormal",
                "cluster": "cluster8",
                "labeljust": "r",
                "labelloc": "b",
            },
            "tau": {"shape": "ellipse", "style": None, "label": "tau\n~\nHalfCauchy"},
            "mu": {"shape": "ellipse", "style": None, "label": "mu\n~\nNormal"},
        },
        "_adj": {"eta": {"obs": {}}, "obs": {}, "tau": {"obs": {}}, "mu": {"obs": {}}},
        "_pred": {"eta": {}, "obs": {"tau": {}, "eta": {}, "mu": {}}, "tau": {}, "mu": {}},
        "_succ": {"eta": {"obs": {}}, "obs": {}, "tau": {"obs": {}}, "mu": {"obs": {}}},
    }

    def test_networkx(self):
        assert self.network_model == model_to_networkx(school_model()).__dict__


def radon_model():
    """Similar in shape to the Radon model"""
    n_homes = 919
    counties = 85
    uranium = np.random.normal(-0.1, 0.4, size=n_homes)
    xbar = np.random.normal(1, 0.1, size=n_homes)
    floor_measure = np.random.randint(0, 2, size=n_homes)

    d, r = divmod(919, 85)
    county = np.hstack((np.tile(np.arange(counties, dtype=int), d), np.arange(r)))
    with pm.Model() as model:
        sigma_a = pm.HalfCauchy("sigma_a", 5)
        gamma = pm.Normal("gamma", mu=0.0, sigma=1e5, shape=3)
        mu_a = pm.Deterministic("mu_a", gamma[0] + gamma[1] * uranium + gamma[2] * xbar)
        eps_a = pm.Normal("eps_a", mu=0, sigma=sigma_a, shape=counties)
        a = pm.Deterministic("a", mu_a + eps_a[county])
        b = pm.Normal("b", mu=0.0, sigma=1e15)
        sigma_y = pm.Uniform("sigma_y", lower=0, upper=100)

        # Anonymous SharedVariables don't show up
        floor_measure = pytensor.shared(floor_measure)
        floor_measure_offset = pm.MutableData("floor_measure_offset", 1)
        y_hat = a + b * floor_measure + floor_measure_offset
        log_radon = pm.MutableData("log_radon", np.random.normal(1, 1, size=n_homes))
        y_like = pm.Normal("y_like", mu=y_hat, sigma=sigma_y, observed=log_radon)

    compute_graph = {
        # variable_name : set of named parents in the graph
        "sigma_a": set(),
        "gamma": set(),
        "mu_a": {"gamma"},
        "eps_a": {"sigma_a"},
        "a": {"mu_a", "eps_a"},
        "b": set(),
        "sigma_y": set(),
        "y_like": {"a", "b", "sigma_y", "floor_measure_offset"},
        "floor_measure_offset": set(),
        # observed data don't have parents in the model graph, but are shown as descendants
        # of the model variables that the observations belong to:
        "log_radon": {"y_like"},
    }
    plates = {
        "": {"b", "sigma_a", "sigma_y", "floor_measure_offset"},
        "3": {"gamma"},
        "85": {"eps_a"},
        "919": {"a", "mu_a", "y_like", "log_radon"},
    }
    return model, compute_graph, plates


def model_with_imputations():
    """The example from https://github.com/pymc-devs/pymc/issues/4043"""
    x = np.random.randn(10) + 10.0
    x = np.concatenate([x, [np.nan], [np.nan]])
    x = np.ma.masked_array(x, np.isnan(x))

    with pm.Model() as model:
        a = pm.Normal("a")
        with pytest.warns(ImputationWarning):
            pm.Normal("L", a, 1.0, observed=x)

    compute_graph = {
        "a": set(),
        "L_unobserved": {"a"},
        "L_observed": {"a"},
        "L": {"L_unobserved", "L_observed"},
    }
    plates = {
        "": {"a"},
        "2": {"L_unobserved"},
        "10": {"L_observed"},
        "12": {"L"},
    }
    return model, compute_graph, plates


def model_with_dims():
    with pm.Model(coords={"city": ["Aachen", "Maastricht", "London", "Bergheim"]}) as pmodel:
        economics = pm.Uniform("economics", lower=-1, upper=1, shape=(1,))

        population = pm.HalfNormal("population", sigma=5, dims=("city"))

        time = pm.ConstantData("time", [2014, 2015, 2016], dims="year")

        n = pm.Deterministic(
            "tax revenue", economics * population[None, :] * time[:, None], dims=("year", "city")
        )

        yobs = pm.MutableData("observed", np.ones((3, 4)))
        L = pm.Normal("L", n, observed=yobs)

    compute_graph = {
        "economics": set(),
        "population": set(),
        "time": set(),
        "tax revenue": {"economics", "population", "time"},
        "L": {"tax revenue"},
        "observed": {"L"},
    }
    plates = {
        "1": {"economics"},
        "city (4)": {"population"},
        "year (3)": {"time"},
        "year (3) x city (4)": {"tax revenue"},
        "3 x 4": {"L", "observed"},
    }

    return pmodel, compute_graph, plates


def model_unnamed_observed_node():
    """
    Model at the source of the following issue: https://github.com/pymc-devs/pymc/issues/5892
    """
    data = [-1, 0, 0.5, 1]

    with pm.Model() as model:
        mu = pm.Normal(name="mu", mu=0.0, sigma=5.0)
        y = pm.Normal(name="y", mu=mu, sigma=3.0, observed=data)

    compute_graph = {
        "mu": set(),
        "y": {"mu"},
    }
    plates = {
        "": {"mu"},
        "4": {"y"},
    }

    return model, compute_graph, plates


def model_observation_dtype_casting():
    """
    Model at the source of the following issue: https://github.com/pymc-devs/pymc/issues/5795
    """
    with pm.Model() as model:
        data = pm.ConstantData("data", [0, 0, 1, 1], dtype=int)
        p = pm.Beta("p", 1, 1)
        bern = pm.Bernoulli("response", p, observed=data)

    compute_graph = {
        "p": set(),
        "response": {"p"},
        "data": {"response"},
    }
    plates = {"": {"p"}, "4": {"data", "response"}}

    return model, compute_graph, plates


def model_non_random_variable_rvs():
    """Test that node types are not inferred based on the variable Op type, but
    model properties

    See https://github.com/pymc-devs/pymc/issues/5766
    """
    with pm.Model() as model:
        mu = pm.Normal(name="mu", mu=0.0, sigma=5.0)

        y_raw = pm.Normal.dist(mu)
        y = pm.math.clip(y_raw, -3, 3)
        model.register_rv(y, name="y")

        z_raw = pm.Normal.dist(y, shape=(5,))
        z = pm.math.clip(z_raw, -1, 1)
        model.register_rv(z, name="z", observed=[0] * 5)

    compute_graph = {
        "mu": set(),
        "y": {"mu"},
        "z": {"y"},
    }
    plates = {
        "": {"mu", "y"},
        "5": {"z"},
    }

    return model, compute_graph, plates


class BaseModelGraphTest:
    model_func = None

    @classmethod
    def setup_class(cls):
        cls.model, cls.compute_graph, cls.plates = cls.model_func()
        cls.model_graph = ModelGraph(cls.model)

    def test_inputs(self):
        for child, parents_in_plot in self.compute_graph.items():
            var = self.model[child]
            parents_in_graph = self.model_graph.get_parent_names(var)
            if isinstance(var, (SharedVariable, TensorConstant)):
                # observed data also doesn't have parents in the compute graph!
                # But for the visualization we like them to become descendants of the
                # RVs that these observations belong to.
                assert not parents_in_graph
            else:
                assert parents_in_plot == parents_in_graph

    def test_compute_graph(self):
        expected = self.compute_graph
        actual = self.model_graph.make_compute_graph()
        assert actual == expected

    def test_plates(self):
        assert self.plates == self.model_graph.get_plates()

    def test_graphviz(self):
        # just make sure everything runs without error

        g = self.model_graph.make_graph()
        for key in self.compute_graph:
            assert key in g.source
        g = model_to_graphviz(self.model)
        for key in self.compute_graph:
            assert key in g.source


class TestRadonModel(BaseModelGraphTest):
    model_func = radon_model

    def test_checks_formatting(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model_to_graphviz(self.model, formatting="plain")
        with pytest.raises(ValueError, match="Unsupported formatting"):
            model_to_graphviz(self.model, formatting="latex")
        with pytest.warns(UserWarning, match="currently not supported"):
            model_to_graphviz(self.model, formatting="plain_with_params")


def model_with_different_descendants():
    """
    Model proposed by Michael to test variable selection functionality
    From here: https://github.com/pymc-devs/pymc/pull/5634#pullrequestreview-916297509
    """
    with pm.Model() as pmodel2:
        a = pm.Normal("a")
        b = pm.Normal("b")
        pm.Normal("c", a * b)
        intermediate = pm.Deterministic("intermediate", a + b)
        pred = pm.Deterministic("pred", intermediate * 3)

        obs = pm.ConstantData("obs", 1.75)

        L = pm.Normal("L", mu=1 + 0.5 * pred, observed=obs)

    return pmodel2


class TestImputationModel(BaseModelGraphTest):
    model_func = model_with_imputations


class TestModelWithDims(BaseModelGraphTest):
    model_func = model_with_dims

    def test_issue_6335_dims_containing_none(self):
        with pm.Model(coords=dict(time=np.arange(5))) as pmodel:
            data = pt.as_tensor(np.ones((3, 5)))
            pm.Deterministic("n", data, dims=(None, "time"))

        mg = ModelGraph(pmodel)
        plates_actual = mg.get_plates()
        plates_expected = {
            "n_dim0 (3) x time (5)": {"n"},
        }
        assert plates_actual == plates_expected


class TestUnnamedObservedNodes(BaseModelGraphTest):
    model_func = model_unnamed_observed_node


class TestObservationDtypeCasting(BaseModelGraphTest):
    model_func = model_observation_dtype_casting


class TestVariableSelection:
    @pytest.mark.parametrize(
        "var_names, vars_to_plot, compute_graph",
        [
            (["c"], ["a", "b", "c"], {"c": {"a", "b"}, "a": set(), "b": set()}),
            (
                ["L"],
                ["pred", "obs", "L", "intermediate", "a", "b"],
                {
                    "pred": {"intermediate"},
                    "obs": {"L"},
                    "L": {"pred"},
                    "intermediate": {"a", "b"},
                    "a": set(),
                    "b": set(),
                },
            ),
            (
                ["obs"],
                ["pred", "obs", "L", "intermediate", "a", "b"],
                {
                    "pred": {"intermediate"},
                    "obs": {"L"},
                    "L": {"pred"},
                    "intermediate": {"a", "b"},
                    "a": set(),
                    "b": set(),
                },
            ),
            # selecting ["c", "L"] is akin to selecting the entire graph
            (
                ["c", "L"],
                ModelGraph(model_with_different_descendants()).vars_to_plot(),
                ModelGraph(model_with_different_descendants()).make_compute_graph(),
            ),
        ],
    )
    def test_subgraph(self, var_names, vars_to_plot, compute_graph):
        mg = ModelGraph(model_with_different_descendants())
        assert set(mg.vars_to_plot(var_names=var_names)) == set(vars_to_plot)
        assert mg.make_compute_graph(var_names=var_names) == compute_graph


class TestModelNonRandomVariableRVs(BaseModelGraphTest):
    model_func = model_non_random_variable_rvs


def test_model_graph_with_intermediate_named_variables():
    # Issue 6421
    with pm.Model() as m1:
        a = pm.Normal("a", 0, 1, shape=3)
        pm.Normal("b", a.mean(axis=-1), 1)
    assert dict(ModelGraph(m1).make_compute_graph()) == {"a": set(), "b": {"a"}}

    with pm.Model() as m2:
        a = pm.Normal("a", 0, 1)
        b = a + 1
        b.name = "b"
        pm.Normal("c", b, 1)
    assert dict(ModelGraph(m2).make_compute_graph()) == {"a": set(), "c": {"a"}}
