from pymc3.plots import plot_posterior_predictive_glm
from pymc3.backends.ndarray import point_list_to_multitrace
import pytest
import numpy as np

import pymc3 as pm
import matplotlib.pyplot as plt


def test_plot_posterior_predictive_glm_multitrace():
    with pm.Model() as model:
        pm.Normal("x")
        pm.Normal("Intercept")
    trace = point_list_to_multitrace([{"x": np.array([1]), "Intercept": np.array([1])}], model)
    _, ax = plt.subplots()
    plot_posterior_predictive_glm(trace, samples=1)
    lines = ax.get_lines()
    expected_xvalues = np.linspace(0, 1, 100)
    expected_yvalues = np.linspace(1, 2, 100)
    for line in lines:
        x_axis, y_axis = line.get_data()
        # check x-axis
        np.testing.assert_array_equal(x_axis, expected_xvalues)
        np.testing.assert_array_equal(y_axis, expected_yvalues)
