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

import aesara.tensor as at
import numpy as np
import scipy.special as ss

from aesara import config, function

import pymc3.distributions.special as ps

from pymc3.tests.checks import close_to


def check_vals(fn1, fn2, *args):
    v = fn1(*args)
    close_to(v, fn2(*args), 1e-6 if v.dtype == np.float64 else 1e-4)


def test_multigamma():
    x = at.vector("x")
    p = at.scalar("p")

    xvals = [np.array([v], dtype=config.floatX) for v in [0.1, 2, 5, 10, 50, 100]]

    multigammaln = function([x, p], ps.multigammaln(x, p), mode="FAST_COMPILE")

    def ssmultigammaln(a, b):
        return np.array(ss.multigammaln(a[0], b), config.floatX)

    for p in [0, 1, 2, 3, 4, 100]:
        for x in xvals:
            if np.all(x > 0.5 * (p - 1)):
                check_vals(multigammaln, ssmultigammaln, x, p)
