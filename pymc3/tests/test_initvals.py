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
import pytest

import pymc3 as pm


def transform_fwd(rv, expected_untransformed):
    return rv.tag.value_var.tag.transform.forward(rv, expected_untransformed).eval()


class TestInitvalAssignment:
    def test_dist_warnings_and_errors(self):
        with pytest.warns(DeprecationWarning, match="argument is deprecated and has no effect"):
            rv = pm.Exponential.dist(lam=1, testval=0.5)
        assert not hasattr(rv.tag, "test_value")

        with pytest.raises(TypeError, match="Unexpected keyword argument `initval`."):
            pm.Normal.dist(1, 2, initval=None)
        pass

    def test_new_warnings(self):
        with pm.Model() as pmodel:
            with pytest.warns(DeprecationWarning, match="`testval` argument is deprecated"):
                rv = pm.Uniform("u", 0, 1, testval=0.75)
                assert pmodel.initial_values[rv.tag.value_var] == transform_fwd(rv, 0.75)
                assert not hasattr(rv.tag, "test_value")
        pass


class TestSpecialDistributions:
    def test_automatically_assigned_test_values(self):
        # ...because they don't have random number generators.
        rv = pm.Flat.dist()
        assert hasattr(rv.tag, "test_value")
        rv = pm.HalfFlat.dist()
        assert hasattr(rv.tag, "test_value")
        pass
