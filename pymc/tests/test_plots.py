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

import matplotlib.pyplot as plt
import numpy as np
import pytest

from arviz import from_pymc3

import pymc3 as pm

from pymc3.backends.ndarray import point_list_to_multitrace
from pymc3.plots import plot_posterior_predictive_glm


@pytest.mark.parametrize("inferencedata", [True, False])
def test_plot_posterior_predictive_glm_defaults(inferencedata):
    with pm.Model() as model:
        pm.Normal("x")
        pm.Normal("Intercept")
    trace = point_list_to_multitrace([{"x": np.array([1]), "Intercept": np.array([1])}], model)
    if inferencedata:
        trace = from_pymc3(trace, model=model)
    _, ax = plt.subplots()
    plot_posterior_predictive_glm(trace, samples=1)
    lines = ax.get_lines()
    expected_xvalues = np.linspace(0, 1, 100)
    expected_yvalues = np.linspace(1, 2, 100)
    for line in lines:
        x_axis, y_axis = line.get_data()
        np.testing.assert_array_equal(x_axis, expected_xvalues)
        np.testing.assert_array_equal(y_axis, expected_yvalues)
        assert line.get_lw() == 0.2
        assert line.get_c() == "k"


@pytest.mark.parametrize("inferencedata", [True, False])
def test_plot_posterior_predictive_glm_non_defaults(inferencedata):
    with pm.Model() as model:
        pm.Normal("x")
        pm.Normal("Intercept")
    trace = point_list_to_multitrace([{"x": np.array([1]), "Intercept": np.array([1])}], model)
    if inferencedata:
        trace = from_pymc3(trace, model=model)
    _, ax = plt.subplots()
    plot_posterior_predictive_glm(
        trace, samples=1, lm=lambda x, _: x, eval=np.linspace(0, 1, 10), lw=0.3, c="b"
    )
    lines = ax.get_lines()
    expected_xvalues = np.linspace(0, 1, 10)
    expected_yvalues = np.linspace(0, 1, 10)
    for line in lines:
        x_axis, y_axis = line.get_data()
        np.testing.assert_array_equal(x_axis, expected_xvalues)
        np.testing.assert_array_equal(y_axis, expected_yvalues)
        assert line.get_lw() == 0.3
        assert line.get_c() == "b"
