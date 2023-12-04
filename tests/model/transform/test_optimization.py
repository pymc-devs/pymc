#   Copyright 2024 The PyMC Developers
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
from pytensor.graph import Constant

from pymc.data import Data
from pymc.distributions import HalfNormal, Normal
from pymc.model import Model
from pymc.model.transform.optimization import freeze_dims_and_data


def test_freeze_existing_rv_dims_and_data():
    with Model(coords={"test_dim": range(5)}) as m:
        std = Data("std", [1])
        x = HalfNormal("x", std, dims=("test_dim",))
        y = Normal("y", shape=x.shape[0] + 1)

    x_logp, y_logp = m.logp(sum=False)

    assert not isinstance(std, Constant)
    assert x.type.shape == (None,)
    assert y.type.shape == (None,)
    assert x_logp.type.shape == (None,)
    assert y_logp.type.shape == (None,)

    frozen_m = freeze_dims_and_data(m)
    std, x, y = frozen_m["std"], frozen_m["x"], frozen_m["y"]
    x_logp, y_logp = frozen_m.logp(sum=False)
    assert isinstance(std, Constant)
    assert x.type.shape == (5,)
    assert y.type.shape == (6,)
    assert x_logp.type.shape == (5,)
    assert y_logp.type.shape == (6,)


def test_freeze_rv_dims_nothing_to_change():
    with Model(coords={"test_dim": range(5)}) as m:
        x = HalfNormal("x", shape=(5,))
        y = Normal("y", shape=x.shape[0] + 1)

    assert m.point_logps() == freeze_dims_and_data(m).point_logps()
