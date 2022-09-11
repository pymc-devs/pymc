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
import warnings

import numpy as np

from pymc.tests import models
from pymc.tuning import scaling


def test_adjust_precision():
    a = np.array([-10, -0.01, 0, 10, 1e300, -np.inf, np.inf])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero encountered in log", RuntimeWarning)
        a1 = scaling.adjust_precision(a)
    assert all((a1 > 0) & (a1 < 1e200))


def test_guess_scaling():
    start, model, _ = models.non_normal(n=5)
    a1 = scaling.guess_scaling(start, model=model)
    assert all((a1 > 0) & (a1 < 1e200))
