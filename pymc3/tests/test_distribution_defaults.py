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

from pymc3.distributions import Categorical, Continuous, DiscreteUniform
from pymc3.model import Model


class DistTest(Continuous):
    def __init__(self, a, b, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a
        self.b = b

    def logp(self, v):
        return 0


def test_default_nan_fail():
    with Model(), pytest.raises(AttributeError):
        DistTest("x", np.nan, 2, defaults=["a"])


def test_default_empty_fail():
    with Model(), pytest.raises(AttributeError):
        DistTest("x", 1, 2, defaults=[])


def test_default_testval():
    with Model():
        x = DistTest("x", 1, 2, testval=5, defaults=[])
        assert x.tag.test_value == 5


def test_default_testval_nan():
    with Model():
        x = DistTest("x", 1, 2, testval=np.nan, defaults=["a"])
        np.testing.assert_almost_equal(x.tag.test_value, np.nan)


def test_default_a():
    with Model():
        x = DistTest("x", 1, 2, defaults=["a"])
        assert x.tag.test_value == 1


def test_default_b():
    with Model():
        x = DistTest("x", np.nan, 2, defaults=["a", "b"])
        assert x.tag.test_value == 2


def test_default_c():
    with Model():
        y = DistTest("y", 7, 8, testval=94)
        x = DistTest("x", y, 2, defaults=["a", "b"])
        assert x.tag.test_value == 94


def test_default_discrete_uniform():
    with Model():
        x = DiscreteUniform("x", lower=1, upper=2)
        assert x.init_value == 1


def test_discrete_uniform_negative():
    model = Model()
    with model:
        x = DiscreteUniform("x", lower=-10, upper=0)
    assert model.test_point["x"] == -5


def test_categorical_mode():
    model = Model()
    with model:
        x = Categorical("x", p=np.eye(4), shape=4)
    assert np.allclose(model.test_point["x"], np.arange(4))
