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

from numpy.testing import assert_almost_equal

import pymc3 as pm

from pymc3.distributions.transforms import Transform
from pymc3.tests.helpers import SeededTest


class TestTransformName:
    cases = [("var", "var_test__"), ("var_test_", "var_test__test__")]
    transform_name = "test"

    def test_get_transformed_name(self):
        test_transform = Transform()
        test_transform.name = self.transform_name
        for name, transformed in self.cases:
            assert pm.util.get_transformed_name(name, test_transform) == transformed

    def test_is_transformed_name(self):
        for name, transformed in self.cases:
            assert pm.util.is_transformed_name(transformed)
            assert not pm.util.is_transformed_name(name)

    def test_get_untransformed_name(self):
        for name, transformed in self.cases:
            assert pm.util.get_untransformed_name(transformed) == name
            with pytest.raises(ValueError):
                pm.util.get_untransformed_name(name)


class TestUpdateStartVals(SeededTest):
    def setup_method(self):
        super().setup_method()

    def test_soft_update_all_present(self):
        start = {"a": 1, "b": 2}
        test_point = {"a": 3, "b": 4}
        pm.util.update_start_vals(start, test_point, model=None)
        assert start == {"a": 1, "b": 2}

    def test_soft_update_one_missing(self):
        start = {
            "a": 1,
        }
        test_point = {"a": 3, "b": 4}
        pm.util.update_start_vals(start, test_point, model=None)
        assert start == {"a": 1, "b": 4}

    def test_soft_update_empty(self):
        start = {}
        test_point = {"a": 3, "b": 4}
        pm.util.update_start_vals(start, test_point, model=None)
        assert start == test_point

    def test_soft_update_transformed(self):
        with pm.Model() as model:
            pm.Exponential("a", 1)
        start = {"a": 2.0}
        test_point = {"a_log__": 0}
        pm.util.update_start_vals(start, test_point, model)
        assert_almost_equal(np.exp(start["a_log__"]), start["a"])

    def test_soft_update_parent(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0.0, upper=1.0)
            b = pm.Uniform("b", lower=2.0, upper=3.0)
            pm.Uniform("lower", lower=a, upper=3.0)
            pm.Uniform("upper", lower=0.0, upper=b)
            pm.Uniform("interv", lower=a, upper=b)

        start = {"a": 0.3, "b": 2.1, "lower": 1.4, "upper": 1.4, "interv": 1.4}
        test_point = {
            "lower_interval__": -0.3746934494414109,
            "upper_interval__": 0.693147180559945,
            "interv_interval__": 0.4519851237430569,
        }
        pm.util.update_start_vals(start, model.test_point, model)
        assert_almost_equal(start["lower_interval__"], test_point["lower_interval__"])
        assert_almost_equal(start["upper_interval__"], test_point["upper_interval__"])
        assert_almost_equal(start["interv_interval__"], test_point["interv_interval__"])


class TestCheckStartVals(SeededTest):
    def setup_method(self):
        super().setup_method()

    def test_valid_start_point(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0.0, upper=1.0)
            b = pm.Uniform("b", lower=2.0, upper=3.0)

        start = {"a": 0.3, "b": 2.1}
        pm.util.update_start_vals(start, model.test_point, model)
        pm.util.check_start_vals(start, model)

    def test_invalid_start_point(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0.0, upper=1.0)
            b = pm.Uniform("b", lower=2.0, upper=3.0)

        start = {"a": np.nan, "b": np.nan}
        pm.util.update_start_vals(start, model.test_point, model)
        with pytest.raises(pm.exceptions.SamplingError):
            pm.util.check_start_vals(start, model)

    def test_invalid_variable_name(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0.0, upper=1.0)
            b = pm.Uniform("b", lower=2.0, upper=3.0)

        start = {"a": 0.3, "b": 2.1, "c": 1.0}
        pm.util.update_start_vals(start, model.test_point, model)
        with pytest.raises(KeyError):
            pm.util.check_start_vals(start, model)


class TestExceptions:
    def test_shape_error(self):
        with pytest.raises(pm.exceptions.ShapeError) as exinfo:
            raise pm.exceptions.ShapeError("Just the message.")
        assert "Just" in exinfo.value.args[0]

        with pytest.raises(pm.exceptions.ShapeError) as exinfo:
            raise pm.exceptions.ShapeError("With shapes.", actual=(2, 3))
        assert "(2, 3)" in exinfo.value.args[0]

        with pytest.raises(pm.exceptions.ShapeError) as exinfo:
            raise pm.exceptions.ShapeError("With shapes.", expected="(2,3) or (5,6)")
        assert "(5,6)" in exinfo.value.args[0]

        with pytest.raises(pm.exceptions.ShapeError) as exinfo:
            raise pm.exceptions.ShapeError("With shapes.", actual=(), expected="(5,4) or (?,?,6)")
        assert "(?,?,6)" in exinfo.value.args[0]
        pass

    def test_dtype_error(self):
        with pytest.raises(pm.exceptions.DtypeError) as exinfo:
            raise pm.exceptions.DtypeError("Just the message.")
        assert "Just" in exinfo.value.args[0]

        with pytest.raises(pm.exceptions.DtypeError) as exinfo:
            raise pm.exceptions.DtypeError("With types.", actual=str)
        assert "str" in exinfo.value.args[0]

        with pytest.raises(pm.exceptions.DtypeError) as exinfo:
            raise pm.exceptions.DtypeError("With types.", expected=float)
        assert "float" in exinfo.value.args[0]

        with pytest.raises(pm.exceptions.DtypeError) as exinfo:
            raise pm.exceptions.DtypeError("With types.", actual=int, expected=str)
        assert "int" in exinfo.value.args[0] and "str" in exinfo.value.args[0]
        pass
