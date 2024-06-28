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
import numpy as np
import pytensor
import pytest

from scipy import stats as st

import pymc as pm

from pymc import Normal, draw
from pymc.data import minibatch_index
from pymc.testing import select_by_precision
from pymc.variational.minibatch_rv import create_minibatch_rv
from tests.test_data import gen1, gen2


class TestMinibatchRandomVariable:
    """
    Related to minibatch training
    """

    def test_density_scaling(self):
        with pm.Model() as model1:
            pm.Normal("n", observed=[[1]], total_size=1)
            p1 = pytensor.function([], model1.logp())

        with pm.Model() as model2:
            pm.Normal("n", observed=[[1]], total_size=2)
            p2 = pytensor.function([], model2.logp())
        assert p1() * 2 == p2()

    def test_density_scaling_with_generator(self):
        # We have different size generators

        def true_dens():
            g = gen1()
            for i, point in enumerate(g):
                yield st.norm.logpdf(point).sum() * 10

        t = true_dens()
        # We have same size models
        with pm.Model() as model1:
            pm.Normal("n", observed=gen1(), total_size=100)
            p1 = pytensor.function([], model1.logp())

        with pm.Model() as model2:
            gen_var = pm.generator(gen2())
            pm.Normal("n", observed=gen_var, total_size=100)
            p2 = pytensor.function([], model2.logp())

        for i in range(10):
            _1, _2, _t = p1(), p2(), next(t)
            decimals = select_by_precision(float64=7, float32=1)
            np.testing.assert_almost_equal(_1, _t, decimal=decimals)  # Value O(-50,000)
            np.testing.assert_almost_equal(_1, _2)
        # Done

    def test_gradient_with_scaling(self):
        with pm.Model() as model1:
            genvar = pm.generator(gen1())
            m = pm.Normal("m")
            pm.Normal("n", observed=genvar, total_size=1000)
            grad1 = model1.compile_fn(model1.dlogp(vars=m), point_fn=False)
        with pm.Model() as model2:
            m = pm.Normal("m")
            shavar = pytensor.shared(np.ones((1000, 100)))
            pm.Normal("n", observed=shavar)
            grad2 = model2.compile_fn(model2.dlogp(vars=m), point_fn=False)

        for i in range(10):
            shavar.set_value(np.ones((100, 100)) * i)
            g1 = grad1(1)
            g2 = grad2(1)
            np.testing.assert_almost_equal(g1, g2)

    def test_multidim_scaling(self):
        with pm.Model() as model0:
            pm.Normal("n", observed=[[1, 1], [1, 1]], total_size=[])
            p0 = pytensor.function([], model0.logp())

        with pm.Model() as model1:
            pm.Normal("n", observed=[[1, 1], [1, 1]], total_size=[2, 2])
            p1 = pytensor.function([], model1.logp())

        with pm.Model() as model2:
            pm.Normal("n", observed=[[1], [1]], total_size=[2, 2])
            p2 = pytensor.function([], model2.logp())

        with pm.Model() as model3:
            pm.Normal("n", observed=[[1, 1]], total_size=[2, 2])
            p3 = pytensor.function([], model3.logp())

        with pm.Model() as model4:
            pm.Normal("n", observed=[[1]], total_size=[2, 2])
            p4 = pytensor.function([], model4.logp())

        with pm.Model() as model5:
            pm.Normal("n", observed=[[1]], total_size=[2, Ellipsis, 2])
            p5 = pytensor.function([], model5.logp())
        _p0 = p0()
        assert (
            np.allclose(_p0, p1())
            and np.allclose(_p0, p2())
            and np.allclose(_p0, p3())
            and np.allclose(_p0, p4())
            and np.allclose(_p0, p5())
        )

    def test_common_errors(self):
        with pytest.raises(ValueError, match="Length of"):
            with pm.Model() as m:
                pm.Normal("n", observed=[[1]], total_size=[2, Ellipsis, 2, 2])
                m.logp()
        with pytest.raises(ValueError, match="Length of"):
            with pm.Model() as m:
                pm.Normal("n", observed=[[1]], total_size=[2, 2, 2])
                m.logp()
        with pytest.raises(TypeError, match="Invalid type for total_size"):
            with pm.Model() as m:
                pm.Normal("n", observed=[[1]], total_size="foo")
                m.logp()
        with pytest.raises(NotImplementedError, match="Cannot convert"):
            with pm.Model() as m:
                pm.Normal("n", observed=[[1]], total_size=["foo"])
                m.logp()
        with pytest.raises(ValueError, match="Only one Ellipsis"):
            with pm.Model() as m:
                pm.Normal("n", observed=[[1]], total_size=[Ellipsis, Ellipsis])
                m.logp()

        with pm.Model() as model4:
            with pytest.raises(ValueError, match="only be passed to observed RVs"):
                pm.Normal("n", shape=(1, 1), total_size=[2, 2])

    def test_mixed1(self):
        with pm.Model():
            data = np.random.rand(10, 20)
            mb = pm.Minibatch(data, batch_size=5)
            v = pm.Normal("n", observed=mb, total_size=10)
            assert pm.logp(v, 1) is not None, "Check index is allowed in graph"

    def test_random(self):
        x = Normal.dist(size=(5,))
        mx = create_minibatch_rv(x, total_size=(10,))
        assert mx is not x
        np.testing.assert_array_equal(draw(mx, random_seed=1), draw(x, random_seed=1))

    @pytest.mark.filterwarnings("error")
    def test_minibatch_parameter_and_value(self):
        rng = np.random.default_rng(161)
        total_size = 1000

        with pm.Model(check_bounds=False) as m:
            AD = pm.Data("AD", np.arange(total_size, dtype="float64"))
            TD = pm.Data("TD", np.arange(total_size, dtype="float64"))

            minibatch_idx = minibatch_index(0, 10, size=(9,))
            AD_mt = AD[minibatch_idx]
            TD_mt = TD[minibatch_idx]

            pm.Normal(
                "AD_predicted",
                mu=TD_mt,
                observed=AD_mt,
                total_size=1000,
            )

        logp_fn = m.compile_logp()

        ip = m.initial_point()
        np.testing.assert_allclose(logp_fn(ip), st.norm.logpdf(0) * 1000)

        with m:
            pm.set_data({"AD": np.arange(total_size) + 1})
        np.testing.assert_allclose(logp_fn(ip), st.norm.logpdf(1) * 1000)

        with m:
            pm.set_data({"AD": rng.normal(size=1000)})
        assert logp_fn(ip) != logp_fn(ip)
