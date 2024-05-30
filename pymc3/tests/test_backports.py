#   Copyright 2021 The PyMC Developers
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


class TestLogpSyntax:
    def test_equivalence(self):
        with pm.Model():
            rv = pm.Normal("n")
        input = {"n": 2}
        np.testing.assert_array_equal(rv.logp(input), pm.logp(rv, input))

    def test_equivalence_dist(self):
        rv = pm.Normal.dist()
        assert rv.logp(2).eval() == pm.logp(rv, 2).eval()
        np.testing.assert_array_equal(
            rv.logp(np.arange(3)).eval(), pm.logp(rv, np.arange(3)).eval()
        )
