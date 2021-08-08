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
import aesara
import aesara.tensor as at
import aesara.tensor.random.basic as atr
import numpy as np
import pandas as pd
import pytest

from aesara.graph.basic import Variable

import pymc3 as pm

from pymc3.util import UNSET, select_initval


def test_util_select_initval():
    # The candidate is preferred, particularly if the default is invalid
    assert select_initval(4, default=None) == 4
    assert select_initval([1, 2], default=None) == [1, 2]
    assert select_initval(None, default=3) is None
    assert select_initval(UNSET, default=pd.Series([1, 2, 3])) is UNSET

    # The default is preferred if the candidate is UNSET or invalid
    assert select_initval(UNSET, default=3) == 3
    assert select_initval(UNSET, default=None) is None
    assert select_initval(pd.Series([1, 2, 3]), default=None) is None
    assert isinstance(select_initval(UNSET, default=at.scalar()), Variable)
    assert isinstance(select_initval(at.scalar(), default=3), Variable)

    # None is the fallback if both are invalid
    assert select_initval(pd.Series([1, 2, 3]), default="not good") is None
    pass


class NormalWithoutInitval(pm.Distribution):
    """A distribution that does not specify a default initial value."""

    rv_op = atr.normal

    @classmethod
    def dist(cls, mu=0, sigma=None, **kwargs):
        mu = at.as_tensor(pm.floatX(mu))
        sigma = at.as_tensor(pm.floatX(sigma))
        return super().dist([mu, sigma], **kwargs)


class UniformWithInitval(pm.distributions.continuous.BoundedContinuous):
    """
    A distribution that defaults the initial value.
    """

    rv_op = atr.uniform
    bound_args_indices = (0, 1)  # Lower, Upper

    @classmethod
    def dist(cls, lower=0, upper=1, initval=UNSET, **kwargs):
        lower = at.as_tensor_variable(pm.floatX(lower))
        upper = at.as_tensor_variable(pm.floatX(upper))
        return super().dist([lower, upper], **kwargs)

    @classmethod
    def pick_initval(cls, lower, upper, **kwargs):
        return (lower + upper) / 2


def transform_fwd(rv, expected_untransformed):
    return rv.tag.value_var.tag.transform.forward(rv, expected_untransformed).eval()


class TestInitvalAssignment:
    def test_dist_warnings_and_errors(self):
        rv = UniformWithInitval.dist(1, 2)
        assert not hasattr(rv.tag, "test_value")

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

    def test_new_initval_behaviors(self):
        """
        No test values are set on the RV unless specified by either the user or the RV Op.
        But initial values are always determined and managed by the Model object.
        """
        with pm.Model() as pmodel:
            rv1 = NormalWithoutInitval("default to random draw", 1, 2)
            rv2 = NormalWithoutInitval("default to random draw the second", 1, 2)
            assert pmodel.initial_values[rv1.tag.value_var] != 1
            assert pmodel.initial_values[rv2.tag.value_var] != 1
            assert (
                pmodel.initial_values[rv1.tag.value_var] != pmodel.initial_values[rv2.tag.value_var]
            )
            # Randomly drawn initvals are not attached to the rv:
            assert not hasattr(rv1.tag, "test_value")
            assert not hasattr(rv2.tag, "test_value")

            rv = NormalWithoutInitval("user provided", 1, 2, initval=-0.2)
            assert pmodel.initial_values[rv.tag.value_var] == np.array(
                -0.2, dtype=aesara.config.floatX
            )
            assert not hasattr(rv.tag, "test_value")

            rv = UniformWithInitval("RVOp default", 1.5, 2)
            assert pmodel.initial_values[rv.tag.value_var] == transform_fwd(rv, 1.75)
            assert not hasattr(rv.tag, "test_value")

            rv = UniformWithInitval("user can override RVOp default", 1.5, 2, initval=1.8)
            assert pmodel.initial_values[rv.tag.value_var] == transform_fwd(rv, 1.8)
            assert not hasattr(rv.tag, "test_value")

            rv = UniformWithInitval("user can revert to random draw", 1.5, 2, initval=None)
            assert pmodel.initial_values[rv.tag.value_var] != transform_fwd(rv, 1.75)
            assert not hasattr(rv.tag, "test_value")
        pass

    def test_symbolic_initval(self):
        """A regression tests for https://github.com/pymc-devs/pymc3/issues/4911"""
        with pm.Model() as pmodel:
            a = pm.Normal("a")
            b = pm.Normal("b", a, initval=a)
            # Initval assignment should evaluate symbolics:
            assert isinstance(pmodel.initial_point["b"], np.ndarray)


class TestSpecialDistributions:
    def test_flat(self):
        pm.Flat.pick_initval(initval=4) == 4
        pm.Flat.pick_initval(size=(2,), initval=UNSET) == np.array([0, 0])
        with pytest.raises(NotImplementedError, match="does not support random initval"):
            pm.Flat.pick_initval(initval=None)
        pass

    def test_halfflat(self):
        pm.HalfFlat.pick_initval(initval=4) == 4
        pm.HalfFlat.pick_initval(size=(2,), initval=UNSET) == np.array([1, 1])
        with pytest.raises(NotImplementedError, match="does not support random initval"):
            pm.HalfFlat.pick_initval(initval=None)
        pass

    def test_automatically_assigned_test_values(self):
        # ...because they don't have random number generators.
        rv = pm.Flat.dist()
        assert hasattr(rv.tag, "test_value")
        rv = pm.HalfFlat.dist()
        assert hasattr(rv.tag, "test_value")
        pass
