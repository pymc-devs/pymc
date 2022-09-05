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
import warnings

import aesara
import aesara.tensor as at
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import pytest
import scipy.stats as st

from aeppl.abstract import get_measurable_outputs
from aesara.tensor import TensorVariable

import pymc as pm

from pymc.distributions import DiracDelta, Flat, MvNormal, MvStudentT, joint_logp, logp
from pymc.distributions.distribution import SymbolicRandomVariable, _moment, moment
from pymc.distributions.shape_utils import to_tuple
from pymc.tests.distributions.util import assert_moment_is_expected


class TestBugfixes:
    @pytest.mark.parametrize("dist_cls,kwargs", [(MvNormal, dict()), (MvStudentT, dict(nu=2))])
    @pytest.mark.parametrize("dims", [1, 2, 4])
    def test_issue_3051(self, dims, dist_cls, kwargs):
        mu = np.repeat(0, dims)
        d = dist_cls.dist(mu=mu, cov=np.eye(dims), **kwargs, size=(20))

        X = npr.normal(size=(20, dims))
        actual_t = logp(d, X)
        assert isinstance(actual_t, TensorVariable)
        actual_a = actual_t.eval()
        assert isinstance(actual_a, np.ndarray)
        assert actual_a.shape == (X.shape[0],)

    def test_issue_4499(self):
        # Test for bug in Uniform and DiscreteUniform logp when setting check_bounds = False
        # https://github.com/pymc-devs/pymc/issues/4499
        with pm.Model(check_bounds=False) as m:
            x = pm.Uniform("x", 0, 2, size=10, transform=None)
        npt.assert_almost_equal(m.compile_logp()({"x": np.ones(10)}), -np.log(2) * 10)

        with pm.Model(check_bounds=False) as m:
            x = pm.DiscreteUniform("x", 0, 1, size=10)
        npt.assert_almost_equal(m.compile_logp()({"x": np.ones(10)}), -np.log(2) * 10)

        with pm.Model(check_bounds=False) as m:
            x = pm.DiracDelta("x", 1, size=10)
        npt.assert_almost_equal(m.compile_logp()({"x": np.ones(10)}), 0 * 10)


@pytest.mark.parametrize(
    "method,newcode",
    [
        ("logp", r"pm.logp\(rv, x\)"),
        ("logcdf", r"pm.logcdf\(rv, x\)"),
        ("random", r"pm.draw\(rv\)"),
    ],
)
def test_logp_gives_migration_instructions(method, newcode):
    rv = pm.Normal.dist()
    f = getattr(rv, method)
    with pytest.raises(AttributeError, match=rf"use `{newcode}`"):
        f()

    # A dim-induced resize of the rv created by the `.dist()` API,
    # happening in Distribution.__new__ would make us loose the monkeypatches.
    # So this triggers it to test if the monkeypatch still works.
    with pm.Model(coords={"year": [2019, 2021, 2022]}):
        rv = pm.Normal("n", dims="year")
        f = getattr(rv, method)
        with pytest.raises(AttributeError, match=rf"use `{newcode}`"):
            f()
    pass


def test_all_distributions_have_moments():
    import pymc.distributions as dist_module

    from pymc.distributions.distribution import DistributionMeta

    dists = (getattr(dist_module, dist) for dist in dist_module.__all__)
    dists = (dist for dist in dists if isinstance(dist, DistributionMeta))
    missing_moments = {
        dist for dist in dists if getattr(dist, "rv_type", None) not in _moment.registry
    }

    # Ignore super classes
    missing_moments -= {
        dist_module.Distribution,
        dist_module.Discrete,
        dist_module.Continuous,
        dist_module.DensityDist,
        dist_module.simulator.Simulator,
    }

    # Distributions that have not been refactored for V4 yet
    not_implemented = {
        dist_module.timeseries.GARCH11,
        dist_module.timeseries.MvGaussianRandomWalk,
        dist_module.timeseries.MvStudentTRandomWalk,
        dist_module.timeseries.EulerMaruyama,
    }

    # Distributions that have been refactored but don't yet have moments
    not_implemented |= {
        dist_module.multivariate.Wishart,
    }

    unexpected_implemented = not_implemented - missing_moments
    if unexpected_implemented:
        raise Exception(
            f"Distributions {unexpected_implemented} have a `moment` implemented. "
            "This test must be updated to expect this."
        )

    unexpected_not_implemented = missing_moments - not_implemented
    if unexpected_not_implemented:
        raise NotImplementedError(
            f"Unexpected by this test, distributions {unexpected_not_implemented} do "
            "not have a `moment` implementation. Either add a moment or filter "
            "these distributions in this test."
        )


class TestDensityDist:
    @pytest.mark.parametrize("size", [(), (3,), (3, 2)], ids=str)
    def test_density_dist_with_random(self, size):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1)
            obs = pm.DensityDist(
                "density_dist",
                mu,
                random=lambda mu, rng=None, size=None: rng.normal(loc=mu, scale=1, size=size),
                observed=np.random.randn(100, *size),
            )

        assert obs.eval().shape == (100,) + size

    def test_density_dist_without_random(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1)
            pm.DensityDist(
                "density_dist",
                mu,
                logp=lambda value, mu: logp(pm.Normal.dist(mu, 1, size=100), value),
                observed=np.random.randn(100),
                initval=0,
            )
            idata = pm.sample(tune=50, draws=100, cores=1, step=pm.Metropolis())

        with pytest.raises(NotImplementedError):
            pm.sample_posterior_predictive(idata, model=model)

    @pytest.mark.parametrize("size", [(), (3,), (3, 2)], ids=str)
    def test_density_dist_with_random_multivariate(self, size):
        supp_shape = 5
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1, size=supp_shape)
            obs = pm.DensityDist(
                "density_dist",
                mu,
                random=lambda mu, rng=None, size=None: rng.multivariate_normal(
                    mean=mu, cov=np.eye(len(mu)), size=size
                ),
                observed=np.random.randn(100, *size, supp_shape),
                ndims_params=[1],
                ndim_supp=1,
            )

        assert obs.eval().shape == (100,) + size + (supp_shape,)

    def test_serialize_density_dist(self):
        def func(x):
            return -2 * (x**2).sum()

        def random(rng, size):
            return rng.uniform(-2, 2, size=size)

        with pm.Model():
            pm.Normal("x")
            y = pm.DensityDist("y", logp=func, random=random)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                pm.sample(draws=5, tune=1, mp_ctx="spawn")

        import cloudpickle

        cloudpickle.loads(cloudpickle.dumps(y))

    def test_density_dist_old_api_error(self):
        with pm.Model():
            with pytest.raises(
                TypeError, match="The DensityDist API has changed, you are using the old API"
            ):
                pm.DensityDist("a", lambda x: x)

    @pytest.mark.parametrize("size", [None, (), (2,)], ids=str)
    def test_density_dist_multivariate_logp(self, size):
        supp_shape = 5
        with pm.Model() as model:

            def logp(value, mu):
                return pm.MvNormal.logp(value, mu, at.eye(mu.shape[0]))

            mu = pm.Normal("mu", size=supp_shape)
            a = pm.DensityDist("a", mu, logp=logp, ndims_params=[1], ndim_supp=1, size=size)
        mu_val = npr.normal(loc=0, scale=1, size=supp_shape).astype(aesara.config.floatX)
        a_val = npr.normal(loc=mu_val, scale=1, size=to_tuple(size) + (supp_shape,)).astype(
            aesara.config.floatX
        )
        log_densityt = joint_logp(a, a.tag.value_var, sum=False)[0]
        assert log_densityt.eval(
            {a.tag.value_var: a_val, mu.tag.value_var: mu_val},
        ).shape == to_tuple(size)

    @pytest.mark.parametrize(
        "moment, size, expected",
        [
            (None, None, 0.0),
            (None, 5, np.zeros(5)),
            ("custom_moment", None, 5),
            ("custom_moment", (2, 5), np.full((2, 5), 5)),
        ],
    )
    def test_density_dist_default_moment_univariate(self, moment, size, expected):
        if moment == "custom_moment":
            moment = lambda rv, size, *rv_inputs: 5 * at.ones(size, dtype=rv.dtype)
        with pm.Model() as model:
            pm.DensityDist("x", moment=moment, size=size)
        assert_moment_is_expected(model, expected, check_finite_logp=False)

    @pytest.mark.parametrize("size", [(), (2,), (3, 2)], ids=str)
    def test_density_dist_custom_moment_univariate(self, size):
        def density_moment(rv, size, mu):
            return (at.ones(size) * mu).astype(rv.dtype)

        mu_val = np.array(np.random.normal(loc=2, scale=1)).astype(aesara.config.floatX)
        with pm.Model():
            mu = pm.Normal("mu")
            a = pm.DensityDist("a", mu, moment=density_moment, size=size)
        evaled_moment = moment(a).eval({mu: mu_val})
        assert evaled_moment.shape == to_tuple(size)
        assert np.all(evaled_moment == mu_val)

    @pytest.mark.parametrize("size", [(), (2,), (3, 2)], ids=str)
    def test_density_dist_custom_moment_multivariate(self, size):
        def density_moment(rv, size, mu):
            return (at.ones(size)[..., None] * mu).astype(rv.dtype)

        mu_val = np.random.normal(loc=2, scale=1, size=5).astype(aesara.config.floatX)
        with pm.Model():
            mu = pm.Normal("mu", size=5)
            a = pm.DensityDist(
                "a", mu, moment=density_moment, ndims_params=[1], ndim_supp=1, size=size
            )
        evaled_moment = moment(a).eval({mu: mu_val})
        assert evaled_moment.shape == to_tuple(size) + (5,)
        assert np.all(evaled_moment == mu_val)

    @pytest.mark.parametrize(
        "with_random, size",
        [
            (True, ()),
            (True, (2,)),
            (True, (3, 2)),
            (False, ()),
            (False, (2,)),
        ],
    )
    def test_density_dist_default_moment_multivariate(self, with_random, size):
        def _random(mu, rng=None, size=None):
            return rng.normal(mu, scale=1, size=to_tuple(size) + mu.shape)

        if with_random:
            random = _random
        else:
            random = None

        mu_val = np.random.normal(loc=2, scale=1, size=5).astype(aesara.config.floatX)
        with pm.Model():
            mu = pm.Normal("mu", size=5)
            a = pm.DensityDist("a", mu, random=random, ndims_params=[1], ndim_supp=1, size=size)
        if with_random:
            evaled_moment = moment(a).eval({mu: mu_val})
            assert evaled_moment.shape == to_tuple(size) + (5,)
            assert np.all(evaled_moment == 0)
        else:
            with pytest.raises(
                TypeError,
                match="Cannot safely infer the size of a multivariate random variable's moment.",
            ):
                evaled_moment = moment(a).eval({mu: mu_val})

    def test_dist(self):
        mu = 1
        x = pm.DensityDist.dist(
            mu,
            class_name="test",
            logp=lambda value, mu: pm.logp(pm.Normal.dist(mu), value),
            random=lambda mu, rng=None, size=None: rng.normal(loc=mu, scale=1, size=size),
            shape=(3,),
        )

        test_value = pm.draw(x, random_seed=1)
        assert np.all(test_value == pm.draw(x, random_seed=1))

        x_logp = pm.logp(x, test_value)
        assert np.allclose(x_logp.eval(), st.norm(1).logpdf(test_value))


class TestSymbolicRandomVarible:
    def test_inline(self):
        class TestSymbolicRV(SymbolicRandomVariable):
            pass

        x = TestSymbolicRV([], [Flat.dist()], ndim_supp=0)()

        # By default, the SymbolicRandomVariable will not be inlined. Because we did not
        # dispatch a custom logprob function it will raise next
        with pytest.raises(NotImplementedError):
            logp(x, 0)

        class TestInlinedSymbolicRV(SymbolicRandomVariable):
            inline_aeppl = True

        x_inline = TestInlinedSymbolicRV([], [Flat.dist()], ndim_supp=0)()
        assert np.isclose(logp(x_inline, 0).eval(), 0)

    def test_measurable_outputs(self):
        class TestSymbolicRV(SymbolicRandomVariable):
            pass

        next_rng_, dirac_delta_ = DiracDelta.dist(5).owner.outputs
        next_rng, dirac_delta = TestSymbolicRV([], [next_rng_, dirac_delta_], ndim_supp=0)()
        node = dirac_delta.owner
        assert get_measurable_outputs(node.op, node) == [dirac_delta]
