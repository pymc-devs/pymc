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

from cachetools import cached

import pymc as pm

from pymc.distributions.transforms import RVTransform
from pymc.util import UNSET, hash_key, hashable, locally_cachedmethod


class TestTransformName:
    cases = [("var", "var_test__"), ("var_test_", "var_test__test__")]
    transform_name = "test"

    def test_get_transformed_name(self):
        class NewTransform(RVTransform):
            name = self.transform_name

            def forward(self, value):
                return 0

            def backward(self, value):
                return 0

        test_transform = NewTransform()

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


def test_hashing_of_rv_tuples():
    obs = np.random.normal(-1, 0.1, size=10)
    with pm.Model() as pmodel:
        mu = pm.Normal("mu", 0, 1)
        sd = pm.Gamma("sd", 1, 2)
        dd = pm.Normal("dd", observed=obs)
        for freerv in [mu, sd, dd] + pmodel.free_RVs:
            for structure in [
                freerv,
                {"alpha": freerv, "omega": None},
                [freerv, []],
                (freerv, []),
            ]:
                assert isinstance(hashable(structure), int)


def test_hash_key():
    class Bad1:
        def __hash__(self):
            return 329

    class Bad2:
        def __hash__(self):
            return 329

    b1 = Bad1()
    b2 = Bad2()

    assert b1 != b2

    @cached({}, key=hash_key)
    def some_func(x):
        return x

    assert some_func(b1) != some_func(b2)

    class TestClass:
        @locally_cachedmethod
        def some_method(self, x):
            return x

    tc = TestClass()
    assert tc.some_method(b1) != tc.some_method(b2)


def test_unset_repr(capsys):
    def fn(a=UNSET):
        return

    help(fn)
    captured = capsys.readouterr()
    assert "a=UNSET" in captured.out


def test_find_optim_prior():
    MASS = 0.95

    # normal case
    opt_params = pm.find_optim_prior(
        pm.Gamma, lower=0.1, upper=0.4, mass=MASS, init_guess={"alpha": 1, "beta": 10}
    )
    np.testing.assert_allclose(np.asarray(opt_params.values()), np.array([8.47481597, 37.65435601]))

    # normal case, other distribution
    opt_params = pm.find_optim_prior(
        pm.Normal, lower=155, upper=180, mass=MASS, init_guess={"mu": 170, "sigma": 3}
    )
    np.testing.assert_allclose(np.asarray(opt_params.values()), np.array([167.5000001, 6.37766828]))

    # 1-param case
    opt_params = pm.find_optim_prior(
        pm.Exponential, lower=0.1, upper=0.4, mass=MASS, init_guess={"lam": 10}
    )
    np.testing.assert_allclose(np.asarray(opt_params.values()), np.array([0.79929324]))

    # 3-param case
    opt_params = pm.find_optim_prior(
        pm.StudentT,
        lower=0.1,
        upper=0.4,
        mass=MASS,
        init_guess={"mu": 170, "sigma": 3},
        fixed_params={"nu": 7},
    )
    np.testing.assert_allclose(np.asarray(opt_params.values()), np.array([0.25, 0.06343503]))

    with pytest.raises(ValueError, match="parameters, but you provided only"):
        pm.find_optim_prior(
            pm.Gamma,
            lower=0.1,
            upper=0.4,
            mass=MASS,
            init_guess={"alpha": 1},
            fixed_params={"beta": 10},
        )

    with pytest.raises(TypeError, match="required positional argument"):
        pm.find_optim_prior(
            pm.StudentT, lower=0.1, upper=0.4, mass=MASS, init_guess={"mu": 170, "sigma": 3}
        )

    with pytest.raises(NotImplementedError, match="This function can only optimize two parameters"):
        pm.find_optim_prior(
            pm.StudentT,
            lower=0.1,
            upper=0.4,
            mass=MASS,
            init_guess={"mu": 170, "sigma": 3, "nu": 7},
        )
