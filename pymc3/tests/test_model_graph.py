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
import theano as th

import pymc3 as pm

from pymc3.model_graph import ModelGraph, model_to_graphviz
from pymc3.tests.helpers import SeededTest


def radon_model():
    """Similar in shape to the Radon model"""
    n_homes = 919
    counties = 85
    uranium = np.random.normal(-0.1, 0.4, size=n_homes)
    xbar = np.random.normal(1, 0.1, size=n_homes)
    floor_measure = np.random.randint(0, 2, size=n_homes)
    log_radon = np.random.normal(1, 1, size=n_homes)

    floor_measure = th.shared(floor_measure)

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
        y_hat = a + b * floor_measure
        y_like = pm.Normal("y_like", mu=y_hat, sigma=sigma_y, observed=log_radon)

    compute_graph = {
        "sigma_a": set(),
        "gamma": set(),
        "mu_a": {"gamma"},
        "eps_a": {"sigma_a"},
        "a": {"mu_a", "eps_a"},
        "b": set(),
        "sigma_y": set(),
        "y_like": {"a", "b", "sigma_y"},
    }
    plates = {
        (): {"b", "sigma_a", "sigma_y"},
        (3,): {"gamma"},
        (85,): {"eps_a"},
        (919,): {"a", "mu_a", "y_like"},
    }
    return model, compute_graph, plates


class TestSimpleModel(SeededTest):
    @classmethod
    def setup_class(cls):
        cls.model, cls.compute_graph, cls.plates = radon_model()
        cls.model_graph = ModelGraph(cls.model)

    def test_inputs(self):
        for child, parents in self.compute_graph.items():
            var = self.model[child]
            found_parents = self.model_graph.get_parents(var)
            assert found_parents == parents

    def test_compute_graph(self):
        assert self.compute_graph == self.model_graph.make_compute_graph()

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
