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
import cloudpickle
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.compile.builders import OpFromGraph
from pytensor.tensor.random.op import RandomVariable

import pymc as pm

from pymc.distributions.distribution import _support_point, support_point
from pymc.initial_point import make_initial_point_fn, make_initial_point_fns_per_chain


def transform_fwd(rv, expected_untransformed, model):
    return model.rvs_to_transforms[rv].forward(expected_untransformed, *rv.owner.inputs).eval()


def transform_back(rv, transformed, model) -> np.ndarray:
    return model.rvs_to_transforms[rv].backward(transformed, *rv.owner.inputs).eval()


class TestInitvalEvaluation:
    def test_make_initial_point_fns_per_chain_checks_kwargs(self):
        with pm.Model() as pmodel:
            A = pm.Uniform("A", 0, 1, initval=0.5)
            B = pm.Uniform("B", lower=A, upper=1.5, default_transform=None, initval="support_point")
        with pytest.raises(ValueError, match="Number of initval dicts"):
            make_initial_point_fns_per_chain(
                model=pmodel,
                overrides=[{}, None],
                jitter_rvs={},
                chains=1,
            )
        pass

    @pytest.mark.parametrize("reverse_rvs", [False, True])
    def test_dependent_initvals(self, reverse_rvs):
        with pm.Model() as pmodel:
            L = pm.Uniform("L", 0, 1, initval=0.5)
            U = pm.Uniform("U", lower=9, upper=10, initval=9.5)
            B1 = pm.Uniform("B1", lower=L, upper=U, initval=5)
            B2 = pm.Uniform("B2", lower=L, upper=U, initval=(L + U) / 2)

            if reverse_rvs:
                pmodel.free_RVs = pmodel.free_RVs[::-1]

            ip = pmodel.initial_point(random_seed=0)
            assert ip["L_interval__"] == 0
            assert ip["U_interval__"] == 0
            assert ip["B1_interval__"] == 0
            assert ip["B2_interval__"] == 0

            # Modify initval of L and re-evaluate
            pmodel.rvs_to_initial_values[U] = 9.9
            ip = pmodel.initial_point(random_seed=0)
            assert ip["B1_interval__"] < 0
            assert ip["B2_interval__"] == 0
        pass

    def test_nested_initvals(self):
        # See issue #5168
        with pm.Model() as pmodel:
            one = pm.LogNormal("one", mu=np.log(1), sigma=1e-5, initval="prior")
            two = pm.Lognormal("two", mu=np.log(one * 2), sigma=1e-5, initval="prior")
            three = pm.LogNormal("three", mu=np.log(two * 2), sigma=1e-5, initval="prior")
            four = pm.LogNormal("four", mu=np.log(three * 2), sigma=1e-5, initval="prior")
            five = pm.LogNormal("five", mu=np.log(four * 2), sigma=1e-5, initval="prior")
            six = pm.LogNormal("six", mu=np.log(five * 2), sigma=1e-5, initval="prior")

        ip_vals = list(make_initial_point_fn(model=pmodel, return_transformed=True)(0).values())
        assert np.allclose(np.exp(ip_vals), [1, 2, 4, 8, 16, 32], rtol=1e-3)

        ip_vals = list(make_initial_point_fn(model=pmodel, return_transformed=False)(0).values())
        assert np.allclose(ip_vals, [1, 2, 4, 8, 16, 32], rtol=1e-3)

        pmodel.rvs_to_initial_values[four] = 1

        ip_vals = list(make_initial_point_fn(model=pmodel, return_transformed=True)(0).values())
        assert np.allclose(np.exp(ip_vals), [1, 2, 4, 1, 2, 4], rtol=1e-3)

        ip_vals = list(make_initial_point_fn(model=pmodel, return_transformed=False)(0).values())
        assert np.allclose(ip_vals, [1, 2, 4, 1, 2, 4], rtol=1e-3)

    def test_initval_resizing(self):
        with pm.Model() as pmodel:
            data = pytensor.shared(np.arange(4))
            rv = pm.Uniform("u", lower=data, upper=10, initval="prior")

            ip = pmodel.initial_point(random_seed=0)
            assert np.shape(ip["u_interval__"]) == (4,)

            data.set_value(np.arange(5))
            ip = pmodel.initial_point(random_seed=0)
            assert np.shape(ip["u_interval__"]) == (5,)
        pass

    def test_seeding(self):
        with pm.Model() as pmodel:
            pm.Normal("A", initval="prior")
            pm.Uniform("B", initval="prior")
            pm.Normal("C", initval="support_point")
            ip1 = pmodel.initial_point(random_seed=42)
            ip2 = pmodel.initial_point(random_seed=42)
            ip3 = pmodel.initial_point(random_seed=15)
            assert ip1 == ip2
            assert ip3 != ip2
        pass

    def test_untransformed_initial_point(self):
        with pm.Model() as pmodel:
            pm.Flat("A", initval="support_point")
            pm.HalfFlat("B", initval="support_point")
        fn = make_initial_point_fn(model=pmodel, jitter_rvs={}, return_transformed=False)
        iv = fn(0)
        assert iv["A"] == 0
        assert iv["B"] == 1
        pass

    def test_adds_jitter(self):
        with pm.Model() as pmodel:
            A = pm.Flat("A", initval="support_point")
            B = pm.HalfFlat("B", initval="support_point")
            C = pm.Normal("C", mu=A + B, initval="support_point")
        fn = make_initial_point_fn(model=pmodel, jitter_rvs={B}, return_transformed=True)
        iv = fn(0)
        # Moment of the Flat is 0
        assert iv["A"] == 0
        # Moment of the HalfFlat is 1, but HalfFlat is log-transformed by default
        # so the transformed initial value with jitter will be zero plus a jitter between [-1, 1].
        b_transformed = iv["B_log__"]
        b_untransformed = transform_back(B, b_transformed, model=pmodel)
        assert b_transformed != 0
        assert -1 < b_transformed < 1
        # C is centered on 0 + untransformed initval of B
        assert np.isclose(iv["C"], np.array(0 + b_untransformed, dtype=pytensor.config.floatX))
        # Test jitter respects seeding.
        assert fn(0) == fn(0)
        assert fn(0) != fn(1)

    def test_respects_overrides(self):
        with pm.Model() as pmodel:
            A = pm.Flat("A", initval="support_point")
            B = pm.HalfFlat("B", initval=4)
            C = pm.Normal("C", mu=A + B, initval="support_point")
        fn = make_initial_point_fn(
            model=pmodel,
            jitter_rvs={},
            return_transformed=True,
            overrides={
                A: pt.as_tensor(2, dtype=int),
                B: 3,
                C: 5,
            },
        )
        iv = fn(0)
        assert iv["A"] == 2
        assert np.isclose(iv["B_log__"], np.log(3))
        assert iv["C"] == 5

    def test_string_overrides_work(self):
        with pm.Model() as pmodel:
            A = pm.Flat("A", initval=10)
            B = pm.HalfFlat("B", initval=10)
            C = pm.HalfFlat("C", initval=10)

        fn = make_initial_point_fn(
            model=pmodel,
            jitter_rvs={},
            return_transformed=True,
            overrides={
                "A": 1,
                "B": 1,
                "C_log__": 0,
            },
        )
        iv = fn(0)
        assert iv["A"] == 1
        assert np.isclose(iv["B_log__"], 0)
        assert iv["C_log__"] == 0

    @pytest.mark.parametrize("reverse_rvs", [False, True])
    def test_dependent_initval_from_OFG(self, reverse_rvs):
        class MyTestOp(OpFromGraph):
            pass

        @_support_point.register(MyTestOp)
        def my_test_op_support_point(op, out):
            out1, out2 = out.owner.outputs
            if out is out1:
                return out1
            else:
                return out1 * 4

        out1 = pt.zeros(())
        out2 = out1 * 2
        rv_op = MyTestOp([], [out1, out2])

        with pm.Model() as model:
            A, B = rv_op()
            if reverse_rvs:
                model.register_rv(B, "B")
                model.register_rv(A, "A")
            else:
                model.register_rv(A, "A")
                model.register_rv(B, "B")

        assert model.initial_point() == {"A": 0, "B": 0}

        model.set_initval(A, 1)
        assert model.initial_point() == {"A": 1, "B": 4}

        model.set_initval(B, 3)
        assert model.initial_point() == {"A": 1, "B": 3}


class TestSupportPoint:
    def test_basic(self):
        # Standard distributions
        rv = pm.Normal.dist(mu=2.3)
        np.testing.assert_allclose(support_point(rv).eval(), 2.3)

        # Special distributions
        rv = pm.Flat.dist()
        assert support_point(rv).eval() == np.zeros(())
        rv = pm.HalfFlat.dist()
        assert support_point(rv).eval() == np.ones(())
        rv = pm.Flat.dist(size=(2, 4))
        assert np.all(support_point(rv).eval() == np.zeros((2, 4)))
        rv = pm.HalfFlat.dist(size=(2, 4))
        assert np.all(support_point(rv).eval() == np.ones((2, 4)))

    @pytest.mark.parametrize("rv_cls", [pm.Flat, pm.HalfFlat])
    def test_numeric_support_point_shape(self, rv_cls):
        rv = rv_cls.dist(shape=(2,))
        assert not hasattr(rv.tag, "test_value")
        assert tuple(support_point(rv).shape.eval()) == (2,)

    @pytest.mark.parametrize("rv_cls", [pm.Flat, pm.HalfFlat])
    def test_symbolic_support_point_shape(self, rv_cls):
        s = pt.scalar(dtype="int64")
        rv = rv_cls.dist(shape=(s,))
        assert not hasattr(rv.tag, "test_value")
        assert tuple(support_point(rv).shape.eval({s: 4})) == (4,)
        pass

    @pytest.mark.parametrize("rv_cls", [pm.Flat, pm.HalfFlat])
    def test_support_point_from_dims(self, rv_cls):
        with pm.Model(
            coords={
                "year": [2019, 2020, 2021, 2022],
                "city": ["Bonn", "Paris", "Lisbon"],
            }
        ):
            rv = rv_cls("rv", dims=("year", "city"))
            assert not hasattr(rv.tag, "test_value")
            assert tuple(support_point(rv).shape.eval()) == (4, 3)
        pass

    def test_support_point_not_implemented_fallback(self):
        class MyNormalRV(RandomVariable):
            name = "my_normal"
            signature = "(),()->()"
            dtype = "floatX"

            @classmethod
            def rng_fn(cls, rng, mu, sigma, size):
                return np.pi

        class MyNormalDistribution(pm.Normal):
            rv_op = MyNormalRV()

        with pm.Model() as m:
            x = MyNormalDistribution("x", 0, 1, initval="support_point")

        with pytest.warns(
            UserWarning, match="Moment not defined for variable x of type MyNormalRV"
        ):
            res = m.initial_point()

        assert np.isclose(res["x"], np.pi)

    def test_future_warning_moment(self):
        with pm.Model() as m:
            pm.Normal("x", initval="moment")
            with pytest.warns(
                FutureWarning,
                match="The 'moment' strategy is deprecated. Use 'support_point' instead.",
            ):
                ip = m.initial_point(random_seed=42)


def test_pickling_issue_5090():
    with pm.Model() as model:
        pm.Normal("x", initval="prior")
    ip_before = model.initial_point(random_seed=5090)
    model = cloudpickle.loads(cloudpickle.dumps(model))
    ip_after = model.initial_point(random_seed=5090)
    assert ip_before["x"] == ip_after["x"]
