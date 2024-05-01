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
import numpy as np
import pytest

from pytensor.compile import SharedVariable
from pytensor.graph import Constant

from pymc import Deterministic, do
from pymc.data import Data
from pymc.distributions import HalfNormal, Normal
from pymc.model import Model
from pymc.model.transform.optimization import freeze_dims_and_data


def test_freeze_dims_and_data():
    with Model(coords={"test_dim": range(5)}) as m:
        std = Data("test_data", [1])
        x = HalfNormal("x", std, dims=("test_dim",))
        y = Normal("y", shape=x.shape[0] + 1)

    x_logp, y_logp = m.logp(sum=False)

    assert not isinstance(std, Constant)
    assert x.type.shape == (None,)
    assert y.type.shape == (None,)
    assert x_logp.type.shape == (None,)
    assert y_logp.type.shape == (None,)

    frozen_m = freeze_dims_and_data(m)
    data, x, y = frozen_m["test_data"], frozen_m["x"], frozen_m["y"]
    x_logp, y_logp = frozen_m.logp(sum=False)
    assert isinstance(data, Constant)
    assert x.type.shape == (5,)
    assert y.type.shape == (6,)
    assert x_logp.type.shape == (5,)
    assert y_logp.type.shape == (6,)

    # Test trying to update a frozen data or dim raises an informative error
    with frozen_m:
        with pytest.raises(TypeError, match="The variable `test_data` must be a `SharedVariable`"):
            frozen_m.set_data("test_data", values=[2])
        with pytest.raises(
            TypeError, match="The dim_length of `test_dim` must be a `SharedVariable`"
        ):
            frozen_m.set_dim("test_dim", new_length=6, coord_values=range(6))

    # Test we can still update original model
    with m:
        m.set_data("test_data", values=[2])
        m.set_dim("test_dim", new_length=6, coord_values=range(6))
    assert m["test_data"].get_value() == [2]
    assert m.dim_lengths["test_dim"].get_value() == 6


def test_freeze_dims_nothing_to_change():
    with Model(coords={"test_dim": range(5)}) as m:
        x = HalfNormal("x", shape=(5,))
        y = Normal("y", shape=x.shape[0] + 1)

    assert m.point_logps() == freeze_dims_and_data(m).point_logps()


def test_freeze_dims_and_data_subset():
    with Model(coords={"dim1": range(3), "dim2": range(5)}) as m:
        data1 = Data("data1", [1, 2, 3], dims="dim1")
        data2 = Data("data2", [1, 2, 3, 4, 5], dims="dim2")
        var1 = Normal("var1", dims="dim1")
        var2 = Normal("var2", dims="dim2")
        x = data1 * var1
        y = data2 * var2
        det = Deterministic("det", x[:, None] + y[None, :])

    assert det.type.shape == (None, None)

    new_m = freeze_dims_and_data(m, dims=["dim1"], data=[])
    assert new_m["det"].type.shape == (3, None)
    assert isinstance(new_m.dim_lengths["dim1"], Constant) and new_m.dim_lengths["dim1"].data == 3
    assert isinstance(new_m.dim_lengths["dim2"], SharedVariable)
    assert isinstance(new_m["data1"], SharedVariable)
    assert isinstance(new_m["data2"], SharedVariable)

    new_m = freeze_dims_and_data(m, dims=["dim2"], data=[])
    assert new_m["det"].type.shape == (None, 5)
    assert isinstance(new_m.dim_lengths["dim1"], SharedVariable)
    assert isinstance(new_m.dim_lengths["dim2"], Constant) and new_m.dim_lengths["dim2"].data == 5
    assert isinstance(new_m["data1"], SharedVariable)
    assert isinstance(new_m["data2"], SharedVariable)

    new_m = freeze_dims_and_data(m, dims=["dim1", "dim2"], data=[])
    assert new_m["det"].type.shape == (3, 5)
    assert isinstance(new_m.dim_lengths["dim1"], Constant) and new_m.dim_lengths["dim1"].data == 3
    assert isinstance(new_m.dim_lengths["dim2"], Constant) and new_m.dim_lengths["dim2"].data == 5
    assert isinstance(new_m["data1"], SharedVariable)
    assert isinstance(new_m["data2"], SharedVariable)

    new_m = freeze_dims_and_data(m, dims=[], data=["data1"])
    assert new_m["det"].type.shape == (3, None)
    assert isinstance(new_m.dim_lengths["dim1"], SharedVariable)
    assert isinstance(new_m.dim_lengths["dim2"], SharedVariable)
    assert isinstance(new_m["data1"], Constant) and np.all(new_m["data1"].data == [1, 2, 3])
    assert isinstance(new_m["data2"], SharedVariable)

    new_m = freeze_dims_and_data(m, dims=[], data=["data2"])
    assert new_m["det"].type.shape == (None, 5)
    assert isinstance(new_m.dim_lengths["dim1"], SharedVariable)
    assert isinstance(new_m.dim_lengths["dim2"], SharedVariable)
    assert isinstance(new_m["data1"], SharedVariable)
    assert isinstance(new_m["data2"], Constant) and np.all(new_m["data2"].data == [1, 2, 3, 4, 5])

    new_m = freeze_dims_and_data(m, dims=[], data=["data1", "data2"])
    assert new_m["det"].type.shape == (3, 5)
    assert isinstance(new_m.dim_lengths["dim1"], SharedVariable)
    assert isinstance(new_m.dim_lengths["dim2"], SharedVariable)
    assert isinstance(new_m["data1"], Constant) and np.all(new_m["data1"].data == [1, 2, 3])
    assert isinstance(new_m["data2"], Constant) and np.all(new_m["data2"].data == [1, 2, 3, 4, 5])

    new_m = freeze_dims_and_data(m, dims=["dim1"], data=["data2"])
    assert new_m["det"].type.shape == (3, 5)
    assert isinstance(new_m.dim_lengths["dim1"], Constant) and new_m.dim_lengths["dim1"].data == 3
    assert isinstance(new_m.dim_lengths["dim2"], SharedVariable)
    assert isinstance(new_m["data1"], SharedVariable)
    assert isinstance(new_m["data2"], Constant) and np.all(new_m["data2"].data == [1, 2, 3, 4, 5])


def test_freeze_dim_after_do_intervention():
    with Model(coords={"test_dim": range(5)}) as m:
        mu = Data("mu", [0, 1, 2, 3, 4], dims="test_dim")
        x = Normal("x", mu=mu, dims="test_dim")

    do_m = do(m, {mu: mu * 100})
    assert do_m["x"].type.shape == (None,)

    frozen_do_m = freeze_dims_and_data(do_m)
    assert frozen_do_m["x"].type.shape == (5,)
