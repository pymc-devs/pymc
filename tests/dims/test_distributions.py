#   Copyright 2025 - present The PyMC Developers
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

from pymc import Model, draw
from pymc.dims import ZeroSumNormal
from pymc.distributions import ZeroSumNormal as RegularZeroSumNormal


def test_zerosumnormal():
    coords = {"time": range(5), "item": range(3)}

    with Model(coords=coords) as model:
        zsn_item = ZeroSumNormal("zsn_item", core_dims="item", dims=("time", "item"))
        zsn_time = ZeroSumNormal("zsn_time", core_dims="time", dims=("time", "item"))
        zsn_item_time = ZeroSumNormal("zsn_item_time", core_dims=("item", "time"))
    assert zsn_item.type.dims == ("time", "item")
    assert zsn_time.type.dims == ("time", "item")
    assert zsn_item_time.type.dims == ("item", "time")

    zsn_item_draw, zsn_time_draw, zsn_item_time_draw = draw(
        [zsn_item, zsn_time, zsn_item_time], random_seed=1
    )
    assert zsn_item_draw.shape == (5, 3)
    np.testing.assert_allclose(zsn_item_draw.mean(-1), 0, atol=1e-13)
    assert not np.allclose(zsn_item_draw.mean(0), 0, atol=1e-13)

    assert zsn_time_draw.shape == (5, 3)
    np.testing.assert_allclose(zsn_time_draw.mean(0), 0, atol=1e-13)
    assert not np.allclose(zsn_time_draw.mean(-1), 0, atol=1e-13)

    assert zsn_item_time_draw.shape == (3, 5)
    np.testing.assert_allclose(zsn_item_time_draw.mean(), 0, atol=1e-13)

    with Model(coords=coords) as ref_model:
        # Check that the ZeroSumNormal can be used in a model
        RegularZeroSumNormal("zsn_item", dims=("time", "item"))
        RegularZeroSumNormal("zsn_time", dims=("item", "time"))
        RegularZeroSumNormal("zsn_item_time", n_zerosum_axes=2, dims=("item", "time"))

    # Check initial_point and logp
    ip = model.initial_point()
    ref_ip = ref_model.initial_point()
    assert ip.keys() == ref_ip.keys()
    for i, (ip_value, ref_ip_value) in enumerate(zip(ip.values(), ref_ip.values())):
        if i == 1:
            # zsn_time is actually transposed in the original model
            ip_value = ip_value.T
        np.testing.assert_allclose(ip_value, ref_ip_value)

    logp_fn = model.compile_logp()
    ref_logp_fn = ref_model.compile_logp()
    logp_fn(ip)
    # np.testing.assert_allclose(logp_fn(ip), ref_logp_fn(ref_ip))
    # Test a new
