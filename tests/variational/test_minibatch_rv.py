#   Copyright 2024 - present The PyMC Developers
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
import pytensor.tensor as pt
import pytest

from scipy import stats as st

import pymc as pm

from pymc import Normal, draw
from pymc.data import Minibatch
from pymc.variational.minibatch_rv import create_minibatch_rv


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
            AD_mt, TD_mt = Minibatch(AD, TD, batch_size=9)

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

    def test_derived_rv(self):
        """Test we can obtain a minibatch logp out of a derived RV."""
        dist = pt.clip(pm.Normal.dist(0, 1, size=(1,)), -1, 1)
        mb_dist = create_minibatch_rv(dist, total_size=(2,))
        np.testing.assert_allclose(
            pm.logp(mb_dist, -1).eval(),
            pm.logp(dist, -1).eval() * 2,
        )
