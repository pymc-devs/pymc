#   Copyright 2022 The PyMC Developers
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

from pymc.sampling_utils import _get_seeds_per_chain


def test_get_seeds_per_chain():
    ret = _get_seeds_per_chain(None, chains=1)
    assert len(ret) == 1 and isinstance(ret[0], int)

    ret = _get_seeds_per_chain(None, chains=2)
    assert len(ret) == 2 and isinstance(ret[0], int)

    ret = _get_seeds_per_chain(5, chains=1)
    assert ret == (5,)

    ret = _get_seeds_per_chain(5, chains=3)
    assert len(ret) == 3 and isinstance(ret[0], int) and not any(r == 5 for r in ret)

    rng = np.random.default_rng(123)
    expected_ret = rng.integers(2**30, dtype=np.int64, size=1)
    rng = np.random.default_rng(123)
    ret = _get_seeds_per_chain(rng, chains=1)
    assert ret == expected_ret

    rng = np.random.RandomState(456)
    expected_ret = rng.randint(2**30, dtype=np.int64, size=2)
    rng = np.random.RandomState(456)
    ret = _get_seeds_per_chain(rng, chains=2)
    assert np.all(ret == expected_ret)

    for expected_ret in ([0, 1, 2], (0, 1, 2, 3), np.arange(5)):
        ret = _get_seeds_per_chain(expected_ret, chains=len(expected_ret))
        assert ret is expected_ret

        with pytest.raises(ValueError, match="does not match the number of chains"):
            _get_seeds_per_chain(expected_ret, chains=len(expected_ret) + 1)

    with pytest.raises(ValueError, match=re.escape("The `seeds` must be array-like")):
        _get_seeds_per_chain({1: 1, 2: 2}, 2)
