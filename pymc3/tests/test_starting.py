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

from pytest import raises

from pymc3 import Beta, Binomial, Model, Normal, Point, Uniform, find_MAP
from pymc3.tests.checks import close_to
from pymc3.tests.helpers import select_by_precision
from pymc3.tests.models import non_normal, simple_arbitrary_det, simple_model
from pymc3.tuning import starting


def test_accuracy_normal():
    _, model, (mu, _) = simple_model()
    with model:
        newstart = find_MAP(Point(x=[-10.5, 100.5]))
        close_to(newstart["x"], [mu, mu], select_by_precision(float64=1e-5, float32=1e-4))


def test_accuracy_non_normal():
    _, model, (mu, _) = non_normal(4)
    with model:
        newstart = find_MAP(Point(x=[0.5, 0.01, 0.95, 0.99]))
        close_to(newstart["x"], mu, select_by_precision(float64=1e-5, float32=1e-4))


def test_find_MAP_discrete():
    tol = 2.0 ** -11
    alpha = 4
    beta = 4
    n = 20
    yes = 15

    with Model() as model:
        p = Beta("p", alpha, beta)
        Binomial("ss", n=n, p=p)
        Binomial("s", n=n, p=p, observed=yes)

        map_est1 = starting.find_MAP()
        map_est2 = starting.find_MAP(vars=model.vars)

    close_to(map_est1["p"], 0.6086956533498806, tol)

    close_to(map_est2["p"], 0.695642178810167, tol)
    assert map_est2["ss"] == 14


def test_find_MAP_no_gradient():
    _, model = simple_arbitrary_det()
    with model:
        find_MAP()


def test_find_MAP():
    tol = 2.0 ** -11  # 16 bit machine epsilon, a low bar
    data = np.random.randn(100)
    # data should be roughly mean 0, std 1, but let's
    # normalize anyway to get it really close
    data = (data - np.mean(data)) / np.std(data)

    with Model():
        mu = Uniform("mu", -1, 1)
        sigma = Uniform("sigma", 0.5, 1.5)
        Normal("y", mu=mu, tau=sigma ** -2, observed=data)

        # Test gradient minimization
        map_est1 = starting.find_MAP(progressbar=False)
        # Test non-gradient minimization
        map_est2 = starting.find_MAP(progressbar=False, method="Powell")

    close_to(map_est1["mu"], 0, tol)
    close_to(map_est1["sigma"], 1, tol)

    close_to(map_est2["mu"], 0, tol)
    close_to(map_est2["sigma"], 1, tol)


def test_allinmodel():
    model1 = Model()
    model2 = Model()
    with model1:
        x1 = Normal("x1", mu=0, sigma=1)
        y1 = Normal("y1", mu=0, sigma=1)
    with model2:
        x2 = Normal("x2", mu=0, sigma=1)
        y2 = Normal("y2", mu=0, sigma=1)

    starting.allinmodel([x1, y1], model1)
    starting.allinmodel([x1], model1)
    with raises(ValueError, match=r"Some variables not in the model: \['x2', 'y2'\]"):
        starting.allinmodel([x2, y2], model1)
    with raises(ValueError, match=r"Some variables not in the model: \['x2'\]"):
        starting.allinmodel([x2, y1], model1)
    with raises(ValueError, match=r"Some variables not in the model: \['x2'\]"):
        starting.allinmodel([x2], model1)
