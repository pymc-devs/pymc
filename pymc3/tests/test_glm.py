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

from numpy.testing import assert_equal

import pymc3

from pymc3 import (
    GLM,
    LinearComponent,
    Model,
    Normal,
    Slice,
    Uniform,
    families,
    find_MAP,
    sample,
)
from pymc3.tests.helpers import SeededTest


# Generate data
def generate_data(intercept, slope, size=700):
    x = np.linspace(-1, 1, size)
    y = intercept + x * slope
    return x, y


class TestGLM(SeededTest):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.intercept = 1
        cls.slope = 3
        cls.sigma = 0.05
        x_linear, cls.y_linear = generate_data(cls.intercept, cls.slope, size=1000)
        cls.y_linear += np.random.normal(size=1000, scale=cls.sigma)
        cls.data_linear = pd.DataFrame(dict(x=x_linear, y=cls.y_linear))

        x_logistic, y_logistic = generate_data(cls.intercept, cls.slope, size=3000)
        y_logistic = 1 / (1 + np.exp(-y_logistic))
        bern_trials = np.random.binomial(1, y_logistic)
        cls.data_logistic = dict(x=x_logistic, y=bern_trials)

        n_trials = np.random.randint(1, 20, size=y_logistic.shape)
        binom_trials = np.random.binomial(n_trials, y_logistic)
        cls.data_logistic2 = dict(x=x_logistic, y=binom_trials, n=n_trials)

    def test_linear_component(self):
        with Model() as model:
            lm = LinearComponent.from_formula("y ~ x", self.data_linear)
            sigma = Uniform("sigma", 0, 20)
            Normal("y_obs", mu=lm.y_est, sigma=sigma, observed=self.y_linear)
            start = find_MAP(vars=[sigma])
            step = Slice(model.vars)
            trace = sample(
                500, tune=0, step=step, start=start, progressbar=False, random_seed=self.random_seed
            )

            assert round(abs(np.mean(trace["Intercept"]) - self.intercept), 1) == 0
            assert round(abs(np.mean(trace["x"]) - self.slope), 1) == 0
            assert round(abs(np.mean(trace["sigma"]) - self.sigma), 1) == 0

    def test_glm(self):
        with Model() as model:
            GLM.from_formula("y ~ x", self.data_linear)
            step = Slice(model.vars)
            trace = sample(500, step=step, tune=0, progressbar=False, random_seed=self.random_seed)

            assert round(abs(np.mean(trace["Intercept"]) - self.intercept), 1) == 0
            assert round(abs(np.mean(trace["x"]) - self.slope), 1) == 0
            assert round(abs(np.mean(trace["sd"]) - self.sigma), 1) == 0

    def test_glm_offset(self):
        offset = 1.0
        with Model() as model:
            GLM.from_formula("y ~ x", self.data_linear, offset=offset)
            step = Slice(model.vars)
            trace = sample(500, step=step, tune=0, progressbar=False, random_seed=self.random_seed)

            assert round(abs(np.mean(trace["Intercept"]) - self.intercept + offset), 1) == 0

    def test_glm_link_func(self):
        with Model() as model:
            GLM.from_formula(
                "y ~ x", self.data_logistic, family=families.Binomial(link=families.logit)
            )
            step = Slice(model.vars)
            trace = sample(1000, step=step, tune=0, progressbar=False, random_seed=self.random_seed)

            assert round(abs(np.mean(trace["Intercept"]) - self.intercept), 1) == 0
            assert round(abs(np.mean(trace["x"]) - self.slope), 1) == 0

    def test_glm_link_func2(self):
        with Model() as model:
            GLM.from_formula(
                "y ~ x",
                self.data_logistic2,
                family=families.Binomial(priors={"n": self.data_logistic2["n"]}),
            )
            trace = sample(1000, progressbar=False, init="adapt_diag", random_seed=self.random_seed)

            assert round(abs(np.mean(trace["Intercept"]) - self.intercept), 1) == 0
            assert round(abs(np.mean(trace["x"]) - self.slope), 1) == 0

    def test_more_than_one_glm_is_ok(self):
        with Model():
            GLM.from_formula(
                "y ~ x",
                self.data_logistic,
                family=families.Binomial(link=families.logit),
                name="glm1",
            )
            GLM.from_formula(
                "y ~ x",
                self.data_logistic,
                family=families.Binomial(link=families.logit),
                name="glm2",
            )

    def test_from_xy(self):
        with Model():
            GLM(
                self.data_logistic["x"],
                self.data_logistic["y"],
                family=families.Binomial(link=families.logit),
                name="glm1",
            )

    def test_boolean_y(self):
        model = GLM.from_formula(
            "y ~ x", pd.DataFrame({"x": self.data_logistic["x"], "y": self.data_logistic["y"]})
        )
        model_bool = GLM.from_formula(
            "y ~ x",
            pd.DataFrame(
                {"x": self.data_logistic["x"], "y": [bool(i) for i in self.data_logistic["y"]]}
            ),
        )
        assert_equal(model.y.observations, model_bool.y.observations)

    def test_glm_formula_from_calling_scope(self):
        """Formula can extract variables from the calling scope."""
        z = pd.Series([10, 20, 30])
        df = pd.DataFrame({"y": [0, 1, 0], "x": [1.0, 2.0, 3.0]})
        GLM.from_formula("y ~ x + z", df, family=pymc3.glm.families.Binomial())

    def test_linear_component_formula_from_calling_scope(self):
        """Formula can extract variables from the calling scope."""
        z = pd.Series([10, 20, 30])
        df = pd.DataFrame({"y": [0, 1, 0], "x": [1.0, 2.0, 3.0]})
        LinearComponent.from_formula("y ~ x + z", df)
