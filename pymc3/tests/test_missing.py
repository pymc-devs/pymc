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

import numpy
import pandas as pd
import pytest

from numpy import array, ma

from pymc3 import ImputationWarning, Model, Normal, sample, sample_prior_predictive


def test_missing():
    data = ma.masked_values([1, 2, -1, 4, -1], value=-1)
    with Model() as model:
        x = Normal("x", 1, 1)
        with pytest.warns(ImputationWarning):
            Normal("y", x, 1, observed=data)

    (y_missing,) = model.missing_values
    assert y_missing.tag.test_value.shape == (2,)

    model.logp(model.test_point)

    with model:
        prior_trace = sample_prior_predictive()
    assert {"x", "y"} <= set(prior_trace.keys())


def test_missing_pandas():
    data = pd.DataFrame([1, 2, numpy.nan, 4, numpy.nan])
    with Model() as model:
        x = Normal("x", 1, 1)
        with pytest.warns(ImputationWarning):
            Normal("y", x, 1, observed=data)

    (y_missing,) = model.missing_values
    assert y_missing.tag.test_value.shape == (2,)

    model.logp(model.test_point)

    with model:
        prior_trace = sample_prior_predictive()
    assert {"x", "y"} <= set(prior_trace.keys())


def test_missing_with_predictors():
    predictors = array([0.5, 1, 0.5, 2, 0.3])
    data = ma.masked_values([1, 2, -1, 4, -1], value=-1)
    with Model() as model:
        x = Normal("x", 1, 1)
        with pytest.warns(ImputationWarning):
            Normal("y", x * predictors, 1, observed=data)

    (y_missing,) = model.missing_values
    assert y_missing.tag.test_value.shape == (2,)

    model.logp(model.test_point)

    with model:
        prior_trace = sample_prior_predictive()
    assert {"x", "y"} <= set(prior_trace.keys())


def test_missing_dual_observations():
    with Model() as model:
        obs1 = ma.masked_values([1, 2, -1, 4, -1], value=-1)
        obs2 = ma.masked_values([-1, -1, 6, -1, 8], value=-1)
        beta1 = Normal("beta1", 1, 1)
        beta2 = Normal("beta2", 2, 1)
        latent = Normal("theta", shape=5)
        with pytest.warns(ImputationWarning):
            ovar1 = Normal("o1", mu=beta1 * latent, observed=obs1)
        with pytest.warns(ImputationWarning):
            ovar2 = Normal("o2", mu=beta2 * latent, observed=obs2)

        prior_trace = sample_prior_predictive()
        assert {"beta1", "beta2", "theta", "o1", "o2"} <= set(prior_trace.keys())
        sample()


def test_internal_missing_observations():
    with Model() as model:
        obs1 = ma.masked_values([1, 2, -1, 4, -1], value=-1)
        obs2 = ma.masked_values([-1, -1, 6, -1, 8], value=-1)
        with pytest.warns(ImputationWarning):
            theta1 = Normal("theta1", mu=2, observed=obs1)
        with pytest.warns(ImputationWarning):
            theta2 = Normal("theta2", mu=theta1, observed=obs2)

        prior_trace = sample_prior_predictive()
        assert {"theta1", "theta2"} <= set(prior_trace.keys())
        sample()
