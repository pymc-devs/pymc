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

from numpy import inf

from pymc.step_methods.metropolis import tune
from pymc.tests import models
from pymc.tuning import find_MAP, scaling


def test_adjust_precision():
    a = np.array([-10, -0.01, 0, 10, 1e300, -inf, inf])
    a1 = scaling.adjust_precision(a)
    assert all((a1 > 0) & (a1 < 1e200))


def test_guess_scaling():
    start, model, _ = models.non_normal(n=5)
    a1 = scaling.guess_scaling(start, model=model)
    assert all((a1 > 0) & (a1 < 1e200))


@pytest.mark.parametrize("bounded", [False, True])
def test_mle_jacobian(bounded):
    """Test MAP / MLE estimation for distributions with flat priors."""
    truth = 10.0  # Simple normal model should give mu=10.0
    rtol = 1e-4  # this rtol should work on both floatX precisions

    start, model, _ = models.simple_normal(bounded_prior=bounded)
    with model:
        map_estimate = find_MAP(method="BFGS", model=model)
    np.testing.assert_allclose(map_estimate["mu_i"], truth, rtol=rtol)


def test_tune_not_inplace():
    orig_scaling = np.array([0.001, 0.1])
    returned_scaling = tune(orig_scaling, acc_rate=0.6)
    assert not returned_scaling is orig_scaling
    assert np.all(orig_scaling == np.array([0.001, 0.1]))
    pass
