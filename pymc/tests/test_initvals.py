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
import numpy as np
import pytest

import pymc as pm

from pymc.distributions.distribution import get_moment


def transform_fwd(rv, expected_untransformed):
    return rv.tag.value_var.tag.transform.forward(rv, expected_untransformed).eval()


def transform_back(rv, transformed):
    return rv.tag.value_var.tag.transform.backward(rv, transformed).eval()


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
                initial_point = pmodel.recompute_initial_point()
                assert initial_point["u_interval__"] == transform_fwd(rv, 0.75)
                assert not hasattr(rv.tag, "test_value")
        pass


class TestInitvalEvaluation:
    def test_random_draws(self):
        pmodel = pm.Model()
        rv = pm.Uniform.dist(lower=1, upper=2)
        iv = pmodel._eval_initval(
            rv_var=rv,
            initval=None,
            transform=None,
        )
        assert isinstance(iv, np.ndarray)
        assert 1 <= iv <= 2
        pass

    def test_applies_transform(self):
        pmodel = pm.Model()
        rv = pm.Uniform.dist()
        tf = pm.Uniform.default_transform()
        iv = pmodel._eval_initval(
            rv_var=rv,
            initval=0.5,
            transform=tf,
        )
        assert isinstance(iv, np.ndarray)
        assert iv == 0
        pass

    def test_dependent_initvals(self):
        with pm.Model() as pmodel:
            L = pm.Uniform("L", 0, 1, initval=0.5)
            B = pm.Uniform("B", lower=L, upper=2, initval=1.25)
            ip = pmodel.recompute_initial_point()
            assert ip["L_interval__"] == 0
            assert ip["B_interval__"] == 0

            # Modify initval of L and re-evaluate
            pmodel.initial_values[L] = 0.9
            ip = pmodel.recompute_initial_point()
            assert ip["B_interval__"] < 0
        pass

    def test_initval_resizing(self):
        with pm.Model() as pmodel:
            data = aesara.shared(np.arange(4))
            rv = pm.Uniform("u", lower=data, upper=10)

            ip = pmodel.recompute_initial_point()
            assert np.shape(ip["u_interval__"]) == (4,)

            data.set_value(np.arange(5))
            ip = pmodel.recompute_initial_point()
            assert np.shape(ip["u_interval__"]) == (5,)
        pass

    def test_seeding(self):
        with pm.Model() as pmodel:
            pm.Normal("A", initval="prior")
            pm.Uniform("B", initval="moment")
            pm.Normal("C", initval="moment")
            ip1 = pmodel.recompute_initial_point(rng=42)
            ip2 = pmodel.recompute_initial_point(rng=42)
            ip3 = pmodel.recompute_initial_point(rng=15)
            assert ip1 == ip2
            assert ip3 != ip2
        pass

    def test_adds_jitter(self):
        with pm.Model() as pmodel:
            A = pm.Flat("A", initval="moment")
            B = pm.HalfFlat("B", initval="moment")
            C = pm.Normal("C", mu=A + B, initval="moment", sd=0.001)
            fn = pmodel.make_initial_point_fn(jitter_rvs={B})
            iv = fn()
        # Moment of the Flat is 0
        assert iv[pmodel.rvs_to_values[A]] == 0
        # Moment of the HalfFlat is 1, but HalfFlat is log-transformed by default
        # so the transformed initial value with jitter will be
        b_transformed = iv[pmodel.rvs_to_values[B]]
        b_untransformed = transform_back(B, b_transformed)
        assert b_transformed != 0
        assert -1 < b_transformed < 1
        # C is centered on 0 + untransformed initval of B
        assert iv[pmodel.rvs_to_values[C]] == 0 + b_untransformed
        pass


class TestMoment:
    def test_basic(self):
        # Standard distributions
        rv = pm.Normal.dist(mu=2.3)
        np.testing.assert_allclose(get_moment(rv).eval(), 2.3)

        # Special distributions
        rv = pm.Flat.dist()
        assert get_moment(rv).eval() == np.zeros(())
        rv = pm.HalfFlat.dist()
        assert get_moment(rv).eval() == np.ones(())
        rv = pm.Flat.dist(size=(2, 4))
        assert np.all(get_moment(rv).eval() == np.zeros((2, 4)))
        rv = pm.HalfFlat.dist(size=(2, 4))
        assert np.all(get_moment(rv).eval() == np.ones((2, 4)))

    @pytest.mark.parametrize("rv_cls", [pm.Flat, pm.HalfFlat])
    def test_numeric_moment_shape(self, rv_cls):
        rv = rv_cls.dist(shape=(2,))
        assert not hasattr(rv.tag, "test_value")
        assert tuple(get_moment(rv).shape.eval()) == (2,)

    @pytest.mark.parametrize("rv_cls", [pm.Flat, pm.HalfFlat])
    def test_symbolic_moment_shape(self, rv_cls):
        s = at.scalar()
        rv = rv_cls.dist(shape=(s,))
        assert not hasattr(rv.tag, "test_value")
        assert tuple(get_moment(rv).shape.eval({s: 4})) == (4,)
        pass

    @pytest.mark.parametrize("rv_cls", [pm.Flat, pm.HalfFlat])
    def test_moment_from_dims(self, rv_cls):
        with pm.Model(
            coords={
                "year": [2019, 2020, 2021, 2022],
                "city": ["Bonn", "Paris", "Lisbon"],
            }
        ):
            rv = rv_cls("rv", dims=("year", "city"))
            assert not hasattr(rv.tag, "test_value")
            assert tuple(get_moment(rv).shape.eval()) == (4, 3)
        pass
