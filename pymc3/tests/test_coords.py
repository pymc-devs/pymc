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

import pymc3 as pm
import numpy as np


def test_coords():
    chains = 2
    n_features = 3
    n_samples = 10

    coords = {"features": np.arange(n_features)}

    with pm.Model(coords=coords):
        a = pm.Uniform("a", -100, 100, dims="features")
        b = pm.Uniform("b", -100, 100, dims="features")
        tr = pm.sample(n_samples, chains=chains, return_inferencedata=True)

    assert "features" in tr.posterior.a.coords.dims
    assert "features" in tr.posterior.b.coords.dims
