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
import scipy.stats as sp

import pymc3 as pm

from pymc3.tests.checks import close_to
from pymc3.tests.models import mv_simple, simple_model

tol = 2.0 ** -11


def test_logp():
    start, model, (mu, sig) = simple_model()
    lp = model.fastlogp
    lp(start)
    close_to(lp(start), sp.norm.logpdf(start["x"], mu, sig).sum(), tol)


def test_dlogp():
    start, model, (mu, sig) = simple_model()
    dlogp = model.fastdlogp()
    close_to(dlogp(start), -(start["x"] - mu) / sig ** 2, 1.0 / sig ** 2 / 100.0)


def test_dlogp2():
    start, model, (_, sig) = mv_simple()
    H = np.linalg.inv(sig)
    d2logp = model.fastd2logp()
    close_to(d2logp(start), H, np.abs(H / 100.0))


def test_deterministic():
    with pm.Model() as model:
        x = pm.Normal("x", 0, 1)
        y = pm.Deterministic("y", x ** 2)

    assert model.y == y
    assert model["y"] == y


def test_mapping():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        sd = pm.Gamma("sd", 1, 1)
        y = pm.Normal("y", mu, sd, observed=np.array([0.1, 0.5]))
    lp = model.fastlogp
    lparray = model.logp_array
    point = model.test_point
    parray = model.bijection.map(point)
    assert lp(point) == lparray(parray)

    randarray = np.random.randn(*parray.shape)
    randpoint = model.bijection.rmap(randarray)
    assert lp(randpoint) == lparray(randarray)
