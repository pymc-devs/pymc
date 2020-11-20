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
from pymc3.distributions.posterior_predictive import _TraceDict
import numpy as np

from pymc3.backends.ndarray import point_list_to_multitrace


def test_translate_point_list():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        a = pm.Normal("a", mu=mu, sigma=1, observed=0.0)
        mt = point_list_to_multitrace([model.test_point], model)
        assert isinstance(mt, pm.backends.base.MultiTrace)
        assert {"mu"} == set(mt.varnames)
        assert len(mt) == 1


def test_build_TraceDict():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
        trace = pm.sample(chains=2, draws=500)
        dict = _TraceDict(multi_trace=trace)
        assert isinstance(dict, _TraceDict)
        assert len(dict) == 1000
        np.testing.assert_array_equal(trace["mu"], dict["mu"])
        assert set(trace.varnames) == set(dict.varnames) == {"mu"}


def test_build_TraceDict_point_list():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
        dict = _TraceDict(point_list=[model.test_point])
        assert set(dict.varnames) == {"mu"}
        assert len(dict) == 1
        assert len(dict["mu"]) == 1
        assert dict["mu"][0] == 0.0
