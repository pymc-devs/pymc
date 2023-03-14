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
import re

import numpy as np
import pytest

import pymc as pm

from pymc.exceptions import ImputationWarning
from pymc.step_methods.metropolis import tune
from pymc.testing import select_by_precision
from pymc.tuning import find_MAP
from tests import models
from tests.checks import close_to
from tests.models import non_normal, simple_arbitrary_det, simple_model


@pytest.mark.parametrize("bounded", [False, True])
def test_mle_jacobian(bounded):
    """Test MAP / MLE estimation for distributions with flat priors."""
    truth = 10.0  # Simple normal model should give mu=10.0
    rtol = 1e-4  # this rtol should work on both floatX precisions

    start, model, _ = models.simple_normal(bounded_prior=bounded)
    with model:
        map_estimate = find_MAP(method="BFGS", model=model)
    np.testing.assert_allclose(map_estimate["mu_i"], truth, rtol=rtol)


def test_tune_not_inplace():
    orig_scaling = np.array([0.001, 0.1])
    returned_scaling = tune(orig_scaling, acc_rate=0.6)
    assert returned_scaling is not orig_scaling
    assert np.all(orig_scaling == np.array([0.001, 0.1]))


def test_accuracy_normal():
    _, model, (mu, _) = simple_model()
    with model:
        newstart = find_MAP(pm.Point(x=[-10.5, 100.5]))
        close_to(newstart["x"], [mu, mu], select_by_precision(float64=1e-5, float32=1e-4))


def test_accuracy_non_normal():
    _, model, (mu, _) = non_normal(4)
    with model:
        newstart = find_MAP(pm.Point(x=[0.5, 0.01, 0.95, 0.99]))
        close_to(newstart["x"], mu, select_by_precision(float64=1e-5, float32=1e-4))


def test_find_MAP_discrete():
    tol1 = 2.0**-11
    tol2 = 2.0**-6
    alpha = 4
    beta = 4
    n = 20
    yes = 15

    with pm.Model() as model:
        p = pm.Beta("p", alpha, beta)
        pm.Binomial("ss", n=n, p=p)
        pm.Binomial("s", n=n, p=p, observed=yes)

        map_est1 = find_MAP()
        map_est2 = find_MAP(vars=model.value_vars)

    close_to(map_est1["p"], 0.6086956533498806, tol1)

    close_to(map_est2["p"], 0.695642178810167, tol2)
    assert map_est2["ss"] == 14


def test_find_MAP_no_gradient():
    _, model = simple_arbitrary_det()
    with model:
        find_MAP()


def test_find_MAP():
    tol = 2.0**-11  # 16 bit machine epsilon, a low bar
    data = np.random.randn(100)
    # data should be roughly mean 0, std 1, but let's
    # normalize anyway to get it really close
    data = (data - np.mean(data)) / np.std(data)

    with pm.Model():
        mu = pm.Uniform("mu", -1, 1)
        sigma = pm.Uniform("sigma", 0.5, 1.5)
        pm.Normal("y", mu=mu, tau=sigma**-2, observed=data)

        # Test gradient minimization
        map_est1 = find_MAP(progressbar=False)
        # Test non-gradient minimization
        map_est2 = find_MAP(progressbar=False, method="Powell")

    close_to(map_est1["mu"], 0, tol)
    close_to(map_est1["sigma"], 1, tol)

    close_to(map_est2["mu"], 0, tol)
    close_to(map_est2["sigma"], 1, tol)


def test_find_MAP_issue_5923():
    # Test that gradient-based minimization works well regardless of the order
    # of variables in `vars`, and even when starting a reasonable distance from
    # the MAP.
    tol = 2.0**-11  # 16 bit machine epsilon, a low bar
    data = np.random.randn(100)
    # data should be roughly mean 0, std 1, but let's
    # normalize anyway to get it really close
    data = (data - np.mean(data)) / np.std(data)

    with pm.Model():
        mu = pm.Uniform("mu", -1, 1)
        sigma = pm.Uniform("sigma", 0.5, 1.5)
        pm.Normal("y", mu=mu, tau=sigma**-2, observed=data)

        start = {"mu": -0.5, "sigma": 1.25}
        map_est1 = find_MAP(progressbar=False, vars=[mu, sigma], start=start)
        map_est2 = find_MAP(progressbar=False, vars=[sigma, mu], start=start)

    close_to(map_est1["mu"], 0, tol)
    close_to(map_est1["sigma"], 1, tol)

    close_to(map_est2["mu"], 0, tol)
    close_to(map_est2["sigma"], 1, tol)


def test_find_MAP_issue_4488():
    # Test for https://github.com/pymc-devs/pymc/issues/4488
    with pm.Model() as m:
        with pytest.warns(ImputationWarning):
            x = pm.Gamma("x", alpha=3, beta=10, observed=np.array([1, np.nan]))
        y = pm.Deterministic("y", x + 1)
        map_estimate = find_MAP()

    assert not set.difference({"x_missing", "x_missing_log__", "y"}, set(map_estimate.keys()))
    np.testing.assert_allclose(map_estimate["x_missing"], 0.2, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(map_estimate["y"], [2.0, map_estimate["x_missing"][0] + 1])


def test_find_MAP_warning_non_free_RVs():
    with pm.Model() as m:
        x = pm.Normal("x")
        y = pm.Normal("y")
        det = pm.Deterministic("det", x + y)
        pm.Normal("z", det, 1e-5, observed=100)

        msg = "Intermediate variables (such as Deterministic or Potential) were passed"
        with pytest.warns(UserWarning, match=re.escape(msg)):
            r = pm.find_MAP(vars=[det])
        np.testing.assert_allclose([r["x"], r["y"], r["det"]], [50, 50, 100])
