#   Copyright 2024 The PyMC Developers
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

import arviz
import numpy as np
import pytest
import xarray

from cachetools import cached

import pymc as pm

from pymc.distributions.transforms import Transform
from pymc.util import (
    UNSET,
    _get_seeds_per_chain,
    drop_warning_stat,
    get_value_vars_from_user_vars,
    hash_key,
    hashable,
    locally_cachedmethod,
)


class TestTransformName:
    cases = [("var", "var_test__"), ("var_test_", "var_test__test__")]
    transform_name = "test"

    def test_get_transformed_name(self):
        class NewTransform(Transform):
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
        sigma = pm.Gamma("sigma", 1, 2)
        dd = pm.Normal("dd", observed=obs)
        for freerv in [mu, sigma, dd, *pmodel.free_RVs]:
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


def test_drop_warning_stat():
    idata = arviz.from_dict(
        sample_stats={
            "a": np.ones((2, 5, 4)),
            "warning": np.ones((2, 5, 3), dtype=object),
        },
        warmup_sample_stats={
            "a": np.ones((2, 5, 4)),
            "warning": np.ones((2, 5, 3), dtype=object),
        },
        attrs={"version": "0.1.2"},
        coords={
            "adim": [0, 1, None, 3],
            "warning_dim_0": list("ABC"),
        },
        dims={"a": ["adim"], "warning": ["warning_dim_0"]},
        save_warmup=True,
    )

    new = drop_warning_stat(idata)

    assert new is not idata
    assert new.attrs.get("version") == "0.1.2"

    for gname in ["sample_stats", "warmup_sample_stats"]:
        ss = new.get(gname)
        assert isinstance(ss, xarray.Dataset), gname
        assert "a" in ss
        assert "warning" not in ss
        assert "warning_dim_0" not in ss


def test_get_seeds_per_chain():
    ret = _get_seeds_per_chain(None, chains=1)
    assert len(ret) == 1 and isinstance(ret[0], int)

    ret = _get_seeds_per_chain(None, chains=2)
    assert len(ret) == 2 and isinstance(ret[0], int)

    ret = _get_seeds_per_chain(5, chains=1)
    assert ret == (5,)

    ret = _get_seeds_per_chain(5, chains=3)
    assert len(ret) == 3 and isinstance(ret[0], int) and not any(r == 5 for r in ret)

    rng = np.random.default_rng(123)
    expected_ret = rng.integers(2**30, dtype=np.int64, size=1)
    rng = np.random.default_rng(123)
    ret = _get_seeds_per_chain(rng, chains=1)
    assert ret == expected_ret

    rng = np.random.RandomState(456)
    expected_ret = rng.randint(2**30, dtype=np.int64, size=2)
    rng = np.random.RandomState(456)
    ret = _get_seeds_per_chain(rng, chains=2)
    assert np.all(ret == expected_ret)

    for expected_ret in ([0, 1, 2], (0, 1, 2, 3), np.arange(5)):
        ret = _get_seeds_per_chain(expected_ret, chains=len(expected_ret))
        assert ret is expected_ret

        with pytest.raises(ValueError, match="does not match the number of chains"):
            _get_seeds_per_chain(expected_ret, chains=len(expected_ret) + 1)

    with pytest.raises(ValueError, match=re.escape("The `seeds` must be array-like")):
        _get_seeds_per_chain({1: 1, 2: 2}, 2)


def test_get_value_vars_from_user_vars():
    with pm.Model() as model1:
        x1 = pm.Normal("x1", mu=0, sigma=1)
        y1 = pm.Normal("y1", mu=0, sigma=1)

    x1_value = model1.rvs_to_values[x1]
    y1_value = model1.rvs_to_values[y1]
    assert get_value_vars_from_user_vars([x1, y1], model1) == [x1_value, y1_value]
    assert get_value_vars_from_user_vars([x1], model1) == [x1_value]
    # The next line does not wrap the variable in a list on purpose, to test the
    # utility function can handle those as promised
    assert get_value_vars_from_user_vars(x1_value, model1) == [x1_value]

    with pm.Model() as model2:
        x2 = pm.Normal("x2", mu=0, sigma=1)
        y2 = pm.Normal("y2", mu=0, sigma=1)
        det2 = pm.Deterministic("det2", x2 + y2)

    prefix = "The following variables are not random variables in the model:"
    with pytest.raises(ValueError, match=rf"{prefix} \['x2', 'y2'\]"):
        get_value_vars_from_user_vars([x2, y2], model1)
    with pytest.raises(ValueError, match=rf"{prefix} \['x2'\]"):
        get_value_vars_from_user_vars([x2, y1], model1)
    with pytest.raises(ValueError, match=rf"{prefix} \['x2'\]"):
        get_value_vars_from_user_vars([x2], model1)
    with pytest.raises(ValueError, match=rf"{prefix} \['det2'\]"):
        get_value_vars_from_user_vars([det2], model2)
