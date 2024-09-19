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

from pytensor import function

import pymc as pm

from pymc.dispatch import dispatch_jax  # noqa: F401

jax = pytest.importorskip("jax")


def test_jax_TruncatedNormal():
    with pm.Model() as m:
        f_jax = function(
            [],
            [pm.TruncatedNormal("a", 0, 1, lower=-1, upper=2, rng=np.random.default_rng(seed=123))],
            mode="JAX",
        )
        f_py = function(
            [],
            [pm.TruncatedNormal("b", 0, 1, lower=-1, upper=2, rng=np.random.default_rng(seed=123))],
        )

    assert jax.numpy.array_equal(a1=f_py(), a2=f_jax())
