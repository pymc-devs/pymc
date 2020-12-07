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
import theano

from pymc3.distributions import multivariate as mv


def test_posdef_symmetric1():
    data = np.array([[1.0, 0], [0, 1]], dtype=theano.config.floatX)
    assert mv.posdef(data) == 1


def test_posdef_symmetric2():
    data = np.array([[1.0, 2], [2, 1]], dtype=theano.config.floatX)
    assert mv.posdef(data) == 0


def test_posdef_symmetric3():
    """The test return 0 if the matrix has 0 eigenvalue.

    Is this correct?
    """
    data = np.array([[1.0, 1], [1, 1]], dtype=theano.config.floatX)
    assert mv.posdef(data) == 0


def test_posdef_symmetric4():
    d = np.array([[1, 0.99, 1], [0.99, 1, 0.999], [1, 0.999, 1]], theano.config.floatX)

    assert mv.posdef(d) == 0
