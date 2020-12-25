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
import pytest

from pymc3 import Model, Normal, Slice, Uniform, find_MAP, sample
from pymc3.glm import GLM, LinearComponent
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
        cls.data_linear = dict(x=x_linear, y=cls.y_linear)

        x_logistic, y_logistic = generate_data(cls.intercept, cls.slope, size=3000)
        y_logistic = 1 / (1 + np.exp(-y_logistic))
        bern_trials = [np.random.binomial(1, i) for i in y_logistic]
        cls.data_logistic = dict(x=x_logistic, y=bern_trials)

    def test_linear_component(self):
        vars_to_create = {"sigma", "sigma_interval__", "y_obs", "lm_x0", "lm_Intercept"}
        with Model() as model:
            lm = LinearComponent(
                self.data_linear["x"], self.data_linear["y"], name="lm"
            )  # yields lm_x0, lm_Intercept
            sigma = Uniform("sigma", 0, 20)  # yields sigma_interval__
            Normal("y_obs", mu=lm.y_est, sigma=sigma, observed=self.y_linear)  # yields y_obs
            start = find_MAP(vars=[sigma])
            step = Slice(model.vars)
            trace = sample(
                500, tune=0, step=step, start=start, progressbar=False, random_seed=self.random_seed
            )

            assert round(abs(np.mean(trace["lm_Intercept"]) - self.intercept), 1) == 0
            assert round(abs(np.mean(trace["lm_x0"]) - self.slope), 1) == 0
            assert round(abs(np.mean(trace["sigma"]) - self.sigma), 1) == 0
        assert vars_to_create == set(model.named_vars.keys())

    def test_linear_component_from_formula(self):
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
            vars_to_create = {"glm_sd", "glm_sd_log__", "glm_y", "glm_x0", "glm_Intercept"}
            GLM(self.data_linear["x"], self.data_linear["y"], name="glm")
            start = find_MAP()
            step = Slice(model.vars)
            trace = sample(
                500, tune=0, step=step, start=start, progressbar=False, random_seed=self.random_seed
            )
            assert round(abs(np.mean(trace["glm_Intercept"]) - self.intercept), 1) == 0
            assert round(abs(np.mean(trace["glm_x0"]) - self.slope), 1) == 0
            assert round(abs(np.mean(trace["glm_sd"]) - self.sigma), 1) == 0
            assert vars_to_create == set(model.named_vars.keys())

    def test_glm_from_formula(self):
        with Model() as model:
            NAME = "glm"
            GLM.from_formula("y ~ x", self.data_linear, name=NAME)
            start = find_MAP()
            step = Slice(model.vars)
            trace = sample(
                500, tune=0, step=step, start=start, progressbar=False, random_seed=self.random_seed
            )

            assert round(abs(np.mean(trace["%s_Intercept" % NAME]) - self.intercept), 1) == 0
            assert round(abs(np.mean(trace["%s_x" % NAME]) - self.slope), 1) == 0
            assert round(abs(np.mean(trace["%s_sd" % NAME]) - self.sigma), 1) == 0

    def test_strange_types(self):
        with Model():
            with pytest.raises(ValueError):
                GLM(1, self.data_linear["y"], name="lm")
