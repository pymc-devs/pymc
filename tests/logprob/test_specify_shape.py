#   Copyright 2023 The PyMC Developers
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
import pytensor
import pytensor.tensor as pt
import pytest

from scipy import stats

from pymc.distributions import Dirichlet
from pymc.logprob.joint_logprob import factorized_joint_logprob
from pymc.tests.distributions.test_multivariate import dirichlet_logpdf


def test_specify_shape_logprob():
    # 1. Create graph using SpecifyShape
    # Use symbolic last dimension, so that SpecifyShape is not useless
    last_dim = pt.scalar(name="last_dim", dtype="int64")
    x_base = Dirichlet.dist(pt.ones((last_dim,)), shape=(5, last_dim))
    x_base.name = "x"
    x_rv = pt.specify_shape(x_base, shape=(5, 3))
    x_rv.name = "x"

    # 2. Request logp
    x_vv = x_rv.clone()
    [x_logp] = factorized_joint_logprob({x_rv: x_vv}).values()

    # 3. Test logp
    x_logp_fn = pytensor.function([last_dim, x_vv], x_logp)

    # 3.1 Test valid logp
    x_vv_test = stats.dirichlet(np.ones((3,))).rvs(size=(5,))
    np.testing.assert_array_almost_equal(
        x_logp_fn(last_dim=3, x=x_vv_test),
        dirichlet_logpdf(x_vv_test, np.ones((3,))),
    )

    # 3.2 Test shape error
    x_vv_test_invalid = stats.dirichlet(np.ones((1,))).rvs(size=(5,))
    with pytest.raises(TypeError, match=re.escape("not compatible with the data's ((5, 1))")):
        x_logp_fn(last_dim=1, x=x_vv_test_invalid)
