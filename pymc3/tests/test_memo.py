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

import pymc3 as pm

from pymc3 import memoize


def test_memo():
    def fun(inputs, suffix="_a"):
        return str(inputs) + str(suffix)

    inputs = ["i1", "i2"]
    assert fun(inputs) == "['i1', 'i2']_a"
    assert fun(inputs, "_b") == "['i1', 'i2']_b"

    funmem = memoize.memoize(fun)
    assert hasattr(fun, "cache")
    assert isinstance(fun.cache, dict)
    assert len(fun.cache) == 0

    # call the memoized function with a list input
    # and check the size of the cache!
    assert funmem(inputs) == "['i1', 'i2']_a"
    assert funmem(inputs) == "['i1', 'i2']_a"
    assert len(fun.cache) == 1
    assert funmem(inputs, "_b") == "['i1', 'i2']_b"
    assert funmem(inputs, "_b") == "['i1', 'i2']_b"
    assert len(fun.cache) == 2

    # add items to the inputs list (the list instance remains identical !!)
    inputs.append("i3")
    assert funmem(inputs) == "['i1', 'i2', 'i3']_a"
    assert funmem(inputs) == "['i1', 'i2', 'i3']_a"
    assert len(fun.cache) == 3


def test_hashing_of_rv_tuples():
    obs = np.random.normal(-1, 0.1, size=10)
    with pm.Model() as pmodel:
        mu = pm.Normal("mu", 0, 1)
        sd = pm.Gamma("sd", 1, 2)
        dd = pm.DensityDist(
            "dd",
            pm.Normal.dist(mu, sd).logp,
            random=pm.Normal.dist(mu, sd).random,
            observed=obs,
        )
        for freerv in [mu, sd, dd] + pmodel.free_RVs:
            for structure in [
                freerv,
                {"alpha": freerv, "omega": None},
                [freerv, []],
                (freerv, []),
            ]:
                assert isinstance(memoize.hashable(structure), int)
