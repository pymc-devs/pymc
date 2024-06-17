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
import sys
import warnings

import cloudpickle
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.stats as st

from pytensor import scan, shared
from pytensor.tensor import TensorVariable

import pymc as pm

from pymc.distributions import (
    Censored,
    Flat,
    HalfNormal,
    LogNormal,
    MvNormal,
    MvStudentT,
    Normal,
)
from pymc.distributions.distribution import (
    CustomDist,
    CustomDistRV,
    CustomSymbolicDistRV,
    DiracDelta,
    PartialObservedRV,
    SymbolicRandomVariable,
    _support_point,
    create_partial_observed_rv,
    support_point,
)
from pymc.distributions.shape_utils import change_dist_size, to_tuple
from pymc.distributions.transforms import log
from pymc.exceptions import BlockModelAccessError
from pymc.logprob.basic import conditional_logp, logcdf, logp
from pymc.model import Deterministic, Model
from pymc.pytensorf import collect_default_updates, compile_pymc
from pymc.sampling import draw, sample
from pymc.testing import (
    BaseTestDistributionRandom,
    I,
    assert_support_point_is_expected,
    check_logcdf,
    check_logp,
)
from pymc.util import _FutureWarningValidatingScratchpad


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
            x = pm.Uniform("x", 0, 2, size=10, default_transform=None)
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


def test_all_distributions_have_support_points():
    import pymc.distributions as dist_module

    from pymc.distributions.distribution import DistributionMeta

    dists = (getattr(dist_module, dist) for dist in dist_module.__all__)
    dists = (dist for dist in dists if isinstance(dist, DistributionMeta))
    missing_support_points = {
        dist for dist in dists if getattr(dist, "rv_type", None) not in _support_point.registry
    }

    # Ignore super classes
    missing_support_points -= {
        dist_module.Distribution,
        dist_module.Discrete,
        dist_module.Continuous,
        dist_module.CustomDist,
        dist_module.simulator.Simulator,
    }

    # Distributions that have not been refactored for V4 yet
    not_implemented = {
        dist_module.timeseries.EulerMaruyama,
    }

    # Distributions that have been refactored but don't yet have support_points
    not_implemented |= {
        dist_module.multivariate.Wishart,
    }

    unexpected_implemented = not_implemented - missing_support_points
    if unexpected_implemented:
        raise Exception(
            f"Distributions {unexpected_implemented} have a `support_point` implemented. "
            "This test must be updated to expect this."
        )

    unexpected_not_implemented = missing_support_points - not_implemented
    if unexpected_not_implemented:
        raise NotImplementedError(
            f"Unexpected by this test, distributions {unexpected_not_implemented} do "
            "not have a `support_point` implementation. Either add a support_point or filter "
            "these distributions in this test."
        )


class TestCustomDist:
    @pytest.mark.parametrize("size", [(), (3,), (3, 2)], ids=str)
    def test_custom_dist_with_random(self, size):
        with Model() as model:
            mu = Normal("mu", 0, 1)
            obs = CustomDist(
                "custom_dist",
                mu,
                random=lambda mu, rng=None, size=None: rng.normal(loc=mu, scale=1, size=size),
                observed=np.random.randn(100, *size),
            )
        assert isinstance(obs.owner.op, CustomDistRV)
        assert obs.eval().shape == (100, *size)

    def test_custom_dist_with_random_invalid_observed(self):
        with pytest.raises(
            TypeError,
            match=(
                "Since ``v4.0.0`` the ``observed`` parameter should be of type"
                " ``pd.Series``, ``np.array``, or ``pm.Data``."
                " Previous versions allowed passing distribution parameters as"
                " a dictionary in ``observed``, in the current version these "
                "parameters are positional arguments."
            ),
        ):
            size = (3,)
            with Model() as model:
                mu = Normal("mu", 0, 1)
                CustomDist(
                    "custom_dist",
                    mu,
                    random=lambda mu, rng=None, size=None: rng.normal(loc=mu, scale=1, size=size),
                    observed={"values": np.random.randn(100, *size)},
                )

    def test_custom_dist_without_random(self):
        with Model() as model:
            mu = Normal("mu", 0, 1)
            custom_dist = CustomDist(
                "custom_dist",
                mu,
                logp=lambda value, mu: logp(pm.Normal.dist(mu, 1, size=100), value),
                observed=np.random.randn(100),
                initval=0,
            )
            assert isinstance(custom_dist.owner.op, CustomDistRV)
            idata = sample(tune=50, draws=100, cores=1, step=pm.Metropolis())

        with pytest.raises(NotImplementedError):
            pm.sample_posterior_predictive(idata, model=model)

    @pytest.mark.xfail(
        NotImplementedError,
        reason="Support shape of multivariate CustomDist cannot be inferred. See https://github.com/pymc-devs/pytensor/pull/388",
    )
    @pytest.mark.parametrize("size", [(), (3,), (3, 2)], ids=str)
    def test_custom_dist_with_random_multivariate(self, size):
        supp_shape = 5
        with Model() as model:
            mu = Normal("mu", 0, 1, size=supp_shape)
            obs = CustomDist(
                "custom_dist",
                mu,
                random=lambda mu, rng=None, size=None: rng.multivariate_normal(
                    mean=mu, cov=np.eye(len(mu)), size=size
                ),
                observed=np.random.randn(100, *size, supp_shape),
                ndims_params=[1],
                ndim_supp=1,
            )

        assert isinstance(obs.owner.op, CustomDistRV)
        assert obs.eval().shape == (100, *size, supp_shape)

    def test_serialize_custom_dist(self):
        def func(x):
            return -2 * (x**2).sum()

        def random(rng, size):
            return rng.uniform(-2, 2, size=size)

        with Model():
            Normal("x")
            y = CustomDist("y", logp=func, random=random)
            y_dist = CustomDist.dist(logp=func, random=random)
            Deterministic("y_dist", y_dist)
            assert isinstance(y.owner.op, CustomDistRV)
            assert isinstance(y_dist.owner.op, CustomDistRV)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                sample(draws=5, tune=1, mp_ctx="spawn")

        cloudpickle.loads(cloudpickle.dumps(y))
        cloudpickle.loads(cloudpickle.dumps(y_dist))

    def test_custom_dist_old_api_error(self):
        with Model():
            with pytest.raises(
                TypeError, match="The DensityDist API has changed, you are using the old API"
            ):
                CustomDist("a", lambda x: x)

    @pytest.mark.xfail(
        NotImplementedError,
        reason="Support shape of multivariate CustomDist cannot be inferred. See https://github.com/pymc-devs/pytensor/pull/388",
    )
    @pytest.mark.parametrize("size", [None, (), (2,)], ids=str)
    def test_custom_dist_multivariate_logp(self, size):
        supp_shape = 5
        with Model() as model:

            def logp(value, mu):
                return pm.MvNormal.logp(value, mu, pt.eye(mu.shape[0]))

            mu = Normal("mu", size=supp_shape)
            a = CustomDist("a", mu, logp=logp, ndims_params=[1], ndim_supp=1, size=size)

        assert isinstance(a.owner.op, CustomDistRV)
        mu_test_value = npr.normal(loc=0, scale=1, size=supp_shape).astype(pytensor.config.floatX)
        a_test_value = npr.normal(
            loc=mu_test_value, scale=1, size=(*to_tuple(size), supp_shape)
        ).astype(pytensor.config.floatX)
        log_densityf = model.compile_logp(vars=[a], sum=False)
        assert log_densityf({"a": a_test_value, "mu": mu_test_value})[0].shape == to_tuple(size)

    @pytest.mark.parametrize(
        "support_point, size, expected",
        [
            (None, None, 0.0),
            (None, 5, np.zeros(5)),
            ("custom_support_point", None, 5),
            ("custom_support_point", (2, 5), np.full((2, 5), 5)),
        ],
    )
    def test_custom_dist_default_support_point_univariate(self, support_point, size, expected):
        if support_point == "custom_support_point":
            support_point = lambda rv, size, *rv_inputs: 5 * pt.ones(size, dtype=rv.dtype)  # noqa E731
        with pm.Model() as model:
            x = CustomDist("x", support_point=support_point, size=size)
        assert isinstance(x.owner.op, CustomDistRV)
        assert_support_point_is_expected(model, expected, check_finite_logp=False)

    def test_custom_dist_moment_future_warning(self):
        moment = lambda rv, size, *rv_inputs: 5 * pt.ones(size, dtype=rv.dtype)  # noqa E731
        with pm.Model() as model:
            with pytest.warns(
                FutureWarning, match="`moment` argument is deprecated. Use `support_point` instead."
            ):
                x = CustomDist("x", moment=moment)
        assert_support_point_is_expected(model, 5, check_finite_logp=False)

    @pytest.mark.parametrize("size", [(), (2,), (3, 2)], ids=str)
    def test_custom_dist_custom_support_point_univariate(self, size):
        def density_support_point(rv, size, mu):
            return (pt.ones(size) * mu).astype(rv.dtype)

        mu_val = np.array(np.random.normal(loc=2, scale=1)).astype(pytensor.config.floatX)
        with Model():
            mu = Normal("mu")
            a = CustomDist("a", mu, support_point=density_support_point, size=size)
        assert isinstance(a.owner.op, CustomDistRV)
        evaled_support_point = support_point(a).eval({mu: mu_val})
        assert evaled_support_point.shape == to_tuple(size)
        assert np.all(evaled_support_point == mu_val)

    @pytest.mark.xfail(
        NotImplementedError,
        reason="Support shape of multivariate CustomDist cannot be inferred. See https://github.com/pymc-devs/pytensor/pull/388",
    )
    @pytest.mark.parametrize("size", [(), (2,), (3, 2)], ids=str)
    def test_custom_dist_custom_support_point_multivariate(self, size):
        def density_support_point(rv, size, mu):
            return (pt.ones(size)[..., None] * mu).astype(rv.dtype)

        mu_val = np.random.normal(loc=2, scale=1, size=5).astype(pytensor.config.floatX)
        with Model():
            mu = Normal("mu", size=5)
            a = CustomDist(
                "a",
                mu,
                support_point=density_support_point,
                ndims_params=[1],
                ndim_supp=1,
                size=size,
            )
        assert isinstance(a.owner.op, CustomDistRV)
        evaled_support_point = support_point(a).eval({mu: mu_val})
        assert evaled_support_point.shape == (*to_tuple(size), 5)
        assert np.all(evaled_support_point == mu_val)

    @pytest.mark.xfail(
        NotImplementedError,
        reason="Support shape of multivariate CustomDist cannot be inferred. See https://github.com/pymc-devs/pytensor/pull/388",
    )
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
    def test_custom_dist_default_support_point_multivariate(self, with_random, size):
        def _random(mu, rng=None, size=None):
            return rng.normal(mu, scale=1, size=to_tuple(size) + mu.shape)

        if with_random:
            random = _random
        else:
            random = None

        mu_val = np.random.normal(loc=2, scale=1, size=5).astype(pytensor.config.floatX)
        with Model():
            mu = Normal("mu", size=5)
            a = CustomDist("a", mu, random=random, ndims_params=[1], ndim_supp=1, size=size)
        assert isinstance(a.owner.op, CustomDistRV)
        if with_random:
            evaled_support_point = support_point(a).eval({mu: mu_val})
            assert evaled_support_point.shape == (*to_tuple(size), 5)
            assert np.all(evaled_support_point == 0)
        else:
            with pytest.raises(
                TypeError,
                match="Cannot safely infer the size of a multivariate random variable's support_point.",
            ):
                evaled_support_point = support_point(a).eval({mu: mu_val})

    def test_dist(self):
        mu = 1
        x = pm.CustomDist.dist(
            mu,
            logp=lambda value, mu: pm.logp(pm.Normal.dist(mu), value),
            random=lambda mu, rng=None, size=None: rng.normal(loc=mu, scale=1, size=size),
            shape=(3,),
        )

        x = cloudpickle.loads(cloudpickle.dumps(x))

        test_value = pm.draw(x, random_seed=1)
        assert np.all(test_value == pm.draw(x, random_seed=1))

        x_logp = pm.logp(x, test_value)
        assert np.allclose(x_logp.eval(), st.norm(1).logpdf(test_value))


class TestCustomSymbolicDist:
    def test_basic(self):
        def custom_dist(mu, sigma, size):
            return pt.exp(pm.Normal.dist(mu, sigma, size=size))

        with Model() as m:
            mu = Normal("mu")
            sigma = HalfNormal("sigma")
            lognormal = CustomDist(
                "lognormal",
                mu,
                sigma,
                dist=custom_dist,
                size=(10,),
                transform=log,
                initval=np.ones(10),
            )

        assert isinstance(lognormal.owner.op, CustomSymbolicDistRV)

        # Fix mu and sigma, so that all source of randomness comes from the symbolic RV
        draws = pm.draw(lognormal, draws=3, givens={mu: 0.0, sigma: 1.0})
        assert draws.shape == (3, 10)
        assert np.unique(draws).size == 30

        with Model() as ref_m:
            mu = Normal("mu")
            sigma = HalfNormal("sigma")
            LogNormal("lognormal", mu, sigma, size=(10,))

        ip = m.initial_point()
        np.testing.assert_allclose(m.compile_logp()(ip), ref_m.compile_logp()(ip))

    @pytest.mark.parametrize(
        "dist_params, size, expected, dist_fn",
        [
            (
                (5, 1),
                None,
                np.exp(5),
                lambda mu, sigma, size: pt.exp(pm.Normal.dist(mu, sigma, size=size)),
            ),
            (
                (2, np.ones(5)),
                None,
                np.exp(2 + np.ones(5)),
                lambda mu, sigma, size: pt.exp(
                    pm.Normal.dist(mu, sigma, size=size) + pt.ones(size)
                ),
            ),
            (
                (1, 2),
                None,
                np.sqrt(np.exp(1 + 0.5 * 2**2)),
                lambda mu, sigma, size: pt.sqrt(pm.LogNormal.dist(mu, sigma, size=size)),
            ),
            (
                (4,),
                (3,),
                np.log([4, 4, 4]),
                lambda nu, size: pt.log(pm.ChiSquared.dist(nu, size=size)),
            ),
            (
                (12, 1),
                None,
                12,
                lambda mu1, sigma, size: pm.Normal.dist(mu1, sigma, size=size),
            ),
        ],
    )
    def test_custom_dist_default_support_point(self, dist_params, size, expected, dist_fn):
        with Model() as model:
            CustomDist("x", *dist_params, dist=dist_fn, size=size)
        assert_support_point_is_expected(model, expected)

    def test_custom_dist_default_support_point_scan(self):
        def scan_step(left, right):
            x = pm.Uniform.dist(left, right)
            x_update = collect_default_updates([x])
            return x, x_update

        def dist(size):
            xs, updates = scan(
                fn=scan_step,
                sequences=[
                    pt.as_tensor_variable(np.array([-4, -3])),
                    pt.as_tensor_variable(np.array([-2, -1])),
                ],
                name="xs",
            )
            return xs

        with Model() as model:
            CustomDist("x", dist=dist)
        assert_support_point_is_expected(model, np.array([-3, -2]))

    def test_custom_dist_default_support_point_scan_recurring(self):
        def scan_step(xtm1):
            x = pm.Normal.dist(xtm1 + 1)
            x_update = collect_default_updates([x])
            return x, x_update

        def dist(size):
            xs, _ = scan(
                fn=scan_step,
                outputs_info=pt.as_tensor_variable(np.array([0])).astype(float),
                n_steps=3,
                name="xs",
            )
            return xs

        with Model() as model:
            CustomDist("x", dist=dist)
        assert_support_point_is_expected(model, np.array([[1], [2], [3]]))

    @pytest.mark.parametrize(
        "left, right, size, expected",
        [
            (-1, 1, None, 0 + 5),
            (-3, -1, None, -2 + 5),
            (-3, 1, (3,), np.array([-1 + 5, -1 + 5, -1 + 5])),
        ],
    )
    def test_custom_dist_default_support_point_nested(self, left, right, size, expected):
        def dist_fn(left, right, size):
            return pm.Truncated.dist(pm.Normal.dist(0, 1), left, right, size=size) + 5

        with Model() as model:
            CustomDist("x", left, right, size=size, dist=dist_fn)
        assert_support_point_is_expected(model, expected)

    def test_logcdf_inference(self):
        def custom_dist(mu, sigma, size):
            return pt.exp(pm.Normal.dist(mu, sigma, size=size))

        mu = 1
        sigma = 1.25
        test_value = 0.9

        custom_lognormal = CustomDist.dist(mu, sigma, dist=custom_dist)
        ref_lognormal = LogNormal.dist(mu, sigma)

        np.testing.assert_allclose(
            pm.logcdf(custom_lognormal, test_value).eval(),
            pm.logcdf(ref_lognormal, test_value).eval(),
        )

    def test_random_multiple_rngs(self):
        def custom_dist(p, sigma, size):
            idx = pm.Bernoulli.dist(p=p)
            comps = pm.Normal.dist([-sigma, sigma], 1e-1, size=(*size, 2)).T
            return comps[idx]

        customdist = CustomDist.dist(
            0.5,
            10.0,
            dist=custom_dist,
            size=(10,),
        )

        assert isinstance(customdist.owner.op, CustomSymbolicDistRV)

        node = customdist.owner
        assert len(node.inputs) == 5  # Size, 2 inputs and 2 RNGs
        assert len(node.outputs) == 3  # RV and 2 updated RNGs
        assert len(node.op.update(node)) == 2

        draws = pm.draw(customdist, draws=2, random_seed=123)
        assert np.unique(draws).size == 20

    def test_custom_methods(self):
        def custom_dist(mu, size):
            return DiracDelta.dist(mu, size=size)

        def custom_support_point(rv, size, mu):
            return pt.full_like(rv, mu + 1)

        def custom_logp(value, mu):
            return pt.full_like(value, mu + 2)

        def custom_logcdf(value, mu):
            return pt.full_like(value, mu + 3)

        customdist = CustomDist.dist(
            [np.e, np.e],
            dist=custom_dist,
            support_point=custom_support_point,
            logp=custom_logp,
            logcdf=custom_logcdf,
        )

        assert isinstance(customdist.owner.op, CustomSymbolicDistRV)

        np.testing.assert_allclose(draw(customdist), [np.e, np.e])
        np.testing.assert_allclose(support_point(customdist).eval(), [np.e + 1, np.e + 1])
        np.testing.assert_allclose(logp(customdist, [0, 0]).eval(), [np.e + 2, np.e + 2])
        np.testing.assert_allclose(logcdf(customdist, [0, 0]).eval(), [np.e + 3, np.e + 3])

    def test_change_size(self):
        def custom_dist(mu, sigma, size):
            return pt.exp(pm.Normal.dist(mu, sigma, size=size))

        lognormal = CustomDist.dist(
            0,
            1,
            dist=custom_dist,
            size=(10,),
        )
        assert isinstance(lognormal.owner.op, CustomSymbolicDistRV)
        assert tuple(lognormal.shape.eval()) == (10,)

        new_lognormal = change_dist_size(lognormal, new_size=(2, 5))
        assert isinstance(new_lognormal.owner.op, CustomSymbolicDistRV)
        assert tuple(new_lognormal.shape.eval()) == (2, 5)

        new_lognormal = change_dist_size(lognormal, new_size=(2, 5), expand=True)
        assert isinstance(new_lognormal.owner.op, CustomSymbolicDistRV)
        assert tuple(new_lognormal.shape.eval()) == (2, 5, 10)

    def test_error_model_access(self):
        def custom_dist(size):
            return pm.Flat("Flat", size=size)

        with pm.Model() as m:
            with pytest.raises(
                BlockModelAccessError,
                match="Model variables cannot be created in the dist function",
            ):
                CustomDist("custom_dist", dist=custom_dist)

    def test_api_change_error(self):
        def old_random(size):
            return pm.Flat.dist(size=size)

        # Old API raises
        with pytest.raises(TypeError, match="API change: function passed to `random` argument"):
            pm.CustomDist.dist(random=old_random, class_name="custom_dist")

        # New API is fine
        pm.CustomDist.dist(dist=old_random, class_name="custom_dist")

    def test_scan(self):
        def trw(nu, sigma, steps, size):
            def step(xtm1, nu, sigma):
                x = pm.StudentT.dist(nu=nu, mu=xtm1, sigma=sigma, shape=size)
                return x, collect_default_updates([x])

            xs, _ = scan(
                fn=step,
                outputs_info=pt.zeros(size),
                non_sequences=[nu, sigma],
                n_steps=steps,
            )

            # Logprob inference cannot be derived yet  https://github.com/pymc-devs/pymc/issues/6360
            # xs = swapaxes(xs, 0, -1)

            return xs

        nu = 4
        sigma = 0.7
        steps = 99
        batch_size = 3
        x = CustomDist.dist(nu, sigma, steps, dist=trw, size=batch_size)

        x_draw = pm.draw(x, random_seed=1)
        assert x_draw.shape == (steps, batch_size)
        np.testing.assert_allclose(pm.draw(x, random_seed=1), x_draw)
        assert not np.any(pm.draw(x, random_seed=2) == x_draw)

        ref_dist = pm.RandomWalk.dist(
            init_dist=pm.Flat.dist(),
            innovation_dist=pm.StudentT.dist(nu=nu, sigma=sigma),
            steps=steps,
            size=(batch_size,),
        )
        ref_val = pt.concatenate([np.zeros((1, batch_size)), x_draw]).T

        np.testing.assert_allclose(
            pm.logp(x, x_draw).eval().sum(0),
            pm.logp(ref_dist, ref_val).eval(),
        )

    def test_inferred_logp_mixture(self):
        import numpy as np

        import pymc as pm

        def shifted_normal(mu, sigma, size):
            return mu + pm.Normal.dist(0, sigma, shape=size)

        mus = [3.5, -4.3]
        sds = [1.5, 2.3]
        w = [0.3, 0.7]
        with pm.Model() as m:
            comp_dists = [
                pm.DensityDist.dist(mus[0], sds[0], dist=shifted_normal),
                pm.DensityDist.dist(mus[1], sds[1], dist=shifted_normal),
            ]
            pm.Mixture("mix", w=w, comp_dists=comp_dists)

        test_value = 0.1
        np.testing.assert_allclose(
            m.compile_logp()({"mix": test_value}),
            pm.logp(pm.NormalMixture.dist(w=w, mu=mus, sigma=sds), test_value).eval(),
        )

    def test_symbolic_dist(self):
        # Test we can create a SymbolicDist inside a CustomDist
        def dist(size):
            return pm.Truncated.dist(pm.Beta.dist(1, 1, size=size), lower=0.1, upper=0.9)

        assert pm.CustomDist.dist(dist=dist)

    def test_nested_custom_dist(self):
        """Test we can create CustomDist that creates another CustomDist"""

        def dist(size=None):
            def inner_dist(size=None):
                return pm.Normal.dist(size=size)

            inner_dist = pm.CustomDist.dist(dist=inner_dist, size=size)
            return pt.exp(inner_dist)

        rv = pm.CustomDist.dist(dist=dist)
        np.testing.assert_allclose(
            pm.logp(rv, 1.0).eval(),
            pm.logp(pm.LogNormal.dist(), 1.0).eval(),
        )

    def test_signature(self):
        def dist(p, size):
            return -pm.Categorical.dist(p=p, size=size)

        out = CustomDist.dist([0.25, 0.75], dist=dist, signature="(p)->()")
        # Size and updates are added automatically to the signature
        assert out.owner.op.signature == "[size],(p),[rng]->(),[rng]"
        assert out.owner.op.ndim_supp == 0
        assert out.owner.op.ndims_params == [1]

        # When recreated internally, the whole signature may already be known
        out = CustomDist.dist([0.25, 0.75], dist=dist, signature="[size],(p),[rng]->(),[rng]")
        assert out.owner.op.signature == "[size],(p),[rng]->(),[rng]"
        assert out.owner.op.ndim_supp == 0
        assert out.owner.op.ndims_params == [1]

        # A safe signature can be inferred from ndim_supp and ndims_params
        out = CustomDist.dist([0.25, 0.75], dist=dist, ndim_supp=0, ndims_params=[1])
        assert out.owner.op.signature == "[size],(i00),[rng]->(),[rng]"
        assert out.owner.op.ndim_supp == 0
        assert out.owner.op.ndims_params == [1]

        # Otherwise be default we assume everything is scalar, even though it's wrong in this case
        out = CustomDist.dist([0.25, 0.75], dist=dist)
        assert out.owner.op.signature == "[size],(),[rng]->(),[rng]"
        assert out.owner.op.ndim_supp == 0
        assert out.owner.op.ndims_params == [0]


class TestSymbolicRandomVariable:
    def test_inline(self):
        class TestSymbolicRV(SymbolicRandomVariable):
            pass

        rng = pytensor.shared(np.random.default_rng())
        x = TestSymbolicRV([rng], [Flat.dist(rng=rng)], ndim_supp=0)(rng)

        # By default, the SymbolicRandomVariable will not be inlined. Because we did not
        # dispatch a custom logprob function it will raise next
        with pytest.raises(NotImplementedError):
            logp(x, 0)

        class TestInlinedSymbolicRV(SymbolicRandomVariable):
            inline_logprob = True

        x_inline = TestInlinedSymbolicRV([rng], [Flat.dist(rng=rng)], ndim_supp=0)(rng)
        assert np.isclose(logp(x_inline, 0).eval(), 0)

    def test_default_update(self):
        """Test SymbolicRandomVariable Op default to updates from inner graph."""

        class SymbolicRVDefaultUpdates(SymbolicRandomVariable):
            pass

        class SymbolicRVCustomUpdates(SymbolicRandomVariable):
            def update(self, node):
                return {}

        rng = pytensor.shared(np.random.default_rng())
        dummy_rng = rng.type()
        dummy_next_rng, dummy_x = pt.random.normal(rng=dummy_rng).owner.outputs

        # Check that default updates work
        next_rng, x = SymbolicRVDefaultUpdates(
            inputs=[dummy_rng],
            outputs=[dummy_next_rng, dummy_x],
            ndim_supp=0,
        )(rng)
        fn = compile_pymc(inputs=[], outputs=x, random_seed=431)
        assert fn() != fn()

        # Check that custom updates are respected, by using one that's broken
        next_rng, x = SymbolicRVCustomUpdates(
            inputs=[dummy_rng],
            outputs=[dummy_next_rng, dummy_x],
            ndim_supp=0,
        )(rng)
        with pytest.raises(
            ValueError,
            match="No update found for at least one RNG used in SymbolicRandomVariable Op SymbolicRVCustomUpdates",
        ):
            compile_pymc(inputs=[], outputs=x, random_seed=431)

    def test_recreate_with_different_rng_inputs(self):
        """Test that we can recreate a SymbolicRandomVariable with new RNG inputs.

        Related to https://github.com/pymc-devs/pytensor/issues/473
        """
        rng = pytensor.shared(np.random.default_rng())

        dummy_rng = rng.type()
        dummy_next_rng, dummy_x = pt.random.normal(rng=dummy_rng).owner.outputs

        op = SymbolicRandomVariable(
            [dummy_rng],
            [dummy_next_rng, dummy_x],
            ndim_supp=0,
        )

        next_rng, x = op(rng)
        assert op.update(x.owner) == {rng: next_rng}

        new_rng = pytensor.shared(np.random.default_rng())
        inputs = x.owner.inputs.copy()
        inputs[0] = new_rng
        # This would fail with the default OpFromGraph.__call__()
        new_next_rng, new_x = x.owner.op(*inputs)
        assert op.update(new_x.owner) == {new_rng: new_next_rng}


def test_tag_future_warning_dist():
    # Test no unexpected warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        x = pm.Normal.dist()
        assert isinstance(x.tag, _FutureWarningValidatingScratchpad)

        x.tag.banana = "banana"
        assert x.tag.banana == "banana"

        # Check we didn't break test_value filtering
        x.tag.test_value = np.array(1)
        assert x.tag.test_value == 1
        with pytest.raises(TypeError, match="Wrong number of dimensions"):
            x.tag.test_value = np.array([1, 1])
        assert x.tag.test_value == 1

        # No warning if deprecated attribute is not present
        with pytest.raises(AttributeError):
            x.tag.value_var

        # Warning if present
        x.tag.value_var = "1"
        with pytest.warns(FutureWarning, match="Use model.rvs_to_values"):
            value_var = x.tag.value_var
        assert value_var == "1"

        # Check that PyMC method that copies tag contents does not erase special tag
        new_x = change_dist_size(x, new_size=5)
        assert new_x.tag is not x.tag
        assert isinstance(new_x.tag, _FutureWarningValidatingScratchpad)
        with pytest.warns(FutureWarning, match="Use model.rvs_to_values"):
            value_var = new_x.tag.value_var
        assert value_var == "1"


def test_distribution_op_registered():
    """Test that returned Ops are registered as virtual subclasses of the respective PyMC distributions."""
    assert isinstance(Normal.dist().owner.op, Normal)
    assert isinstance(Censored.dist(Normal.dist(), lower=None, upper=None).owner.op, Censored)


class TestDiracDelta:
    def test_logp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "divide by zero encountered in log", RuntimeWarning)
            check_logp(pm.DiracDelta, I, {"c": I}, lambda value, c: np.log(c == value))
            check_logcdf(pm.DiracDelta, I, {"c": I}, lambda value, c: np.log(value >= c))

        @pytest.mark.parametrize(
            "c, size, expected",
            [
                (1, None, 1),
                (1, 5, np.full(5, 1)),
                (np.arange(1, 6), None, np.arange(1, 6)),
            ],
        )
        def test_support_point(self, c, size, expected):
            with pm.Model() as model:
                pm.DiracDelta("x", c=c, size=size)
            assert_support_point_is_expected(model, expected)

    class TestDiracDelta(BaseTestDistributionRandom):
        def diracdelta_rng_fn(self, size, c):
            if size is None:
                return c
            return np.full(size, c)

        pymc_dist = pm.DiracDelta
        pymc_dist_params = {"c": 3}
        expected_rv_op_params = {"c": 3}
        reference_dist_params = {"c": 3}
        reference_dist = lambda self: self.diracdelta_rng_fn  # noqa E731
        checks_to_run = [
            "check_pymc_params_match_rv_op",
            "check_pymc_draws_match_reference",
            "check_rv_size",
        ]

        @pytest.mark.parametrize("floatX", ["float32", "float64"])
        @pytest.mark.xfail(
            sys.platform == "win32", reason="https://github.com/aesara-devs/aesara/issues/871"
        )
        def test_dtype(self, floatX):
            with pytensor.config.change_flags(floatX=floatX):
                assert pm.DiracDelta.dist(2**4).dtype == "int8"
                assert pm.DiracDelta.dist(2**16).dtype == "int32"
                assert pm.DiracDelta.dist(2**32).dtype == "int64"
                assert pm.DiracDelta.dist(2.0).dtype == floatX


class TestPartialObservedRV:
    @pytest.mark.parametrize("symbolic_rv", (False, True))
    def test_univariate(self, symbolic_rv):
        data = np.array([0.25, 0.5, 0.25])
        mask = np.array([False, False, True])

        rv = pm.Normal.dist([1, 2, 3])
        if symbolic_rv:
            # We use a Censored Normal so that PartialObservedRV is needed,
            # but don't use the bounds for testing the logp
            rv = pm.Censored.dist(rv, lower=-100, upper=100)
        (obs_rv, obs_mask), (unobs_rv, unobs_mask), joined_rv = create_partial_observed_rv(rv, mask)

        # Test types
        if symbolic_rv:
            assert isinstance(obs_rv.owner.op, PartialObservedRV)
            assert isinstance(unobs_rv.owner.op, PartialObservedRV)
        else:
            assert isinstance(obs_rv.owner.op, Normal)
            assert isinstance(unobs_rv.owner.op, Normal)

        # Tesh shapes
        assert tuple(obs_rv.shape.eval()) == (2,)
        assert tuple(unobs_rv.shape.eval()) == (1,)
        assert tuple(joined_rv.shape.eval()) == (3,)

        # Test logp
        logp = conditional_logp(
            {obs_rv: pt.as_tensor(data[~mask]), unobs_rv: pt.as_tensor(data[mask])}
        )
        obs_logp, unobs_logp = pytensor.function([], list(logp.values()))()
        np.testing.assert_allclose(obs_logp, st.norm([1, 2]).logpdf([0.25, 0.5]))
        np.testing.assert_allclose(unobs_logp, st.norm([3]).logpdf([0.25]))

    @pytest.mark.parametrize("mutable_shape", (False, True))
    @pytest.mark.parametrize("obs_component_selected", (True, False))
    def test_multivariate_constant_mask_separable(self, obs_component_selected, mutable_shape):
        if obs_component_selected:
            mask = np.zeros((1, 4), dtype=bool)
        else:
            mask = np.ones((1, 4), dtype=bool)
        obs_data = np.array([[0.1, 0.4, 0.1, 0.4]])
        unobs_data = np.array([[0.4, 0.1, 0.4, 0.1]])

        if mutable_shape:
            shape = (1, pytensor.shared(np.array(4, dtype=int)))
        else:
            shape = (1, 4)
        rv = pm.Dirichlet.dist(pt.arange(shape[-1]) + 1, shape=shape)
        (obs_rv, obs_mask), (unobs_rv, unobs_mask), joined_rv = create_partial_observed_rv(rv, mask)

        # Test types
        assert isinstance(obs_rv.owner.op, pm.Dirichlet)
        assert isinstance(unobs_rv.owner.op, pm.Dirichlet)

        # Test shapes
        if obs_component_selected:
            expected_obs_shape = (1, 4)
            expected_unobs_shape = (0, 4)
        else:
            expected_obs_shape = (0, 4)
            expected_unobs_shape = (1, 4)
        assert tuple(obs_rv.shape.eval()) == expected_obs_shape
        assert tuple(unobs_rv.shape.eval()) == expected_unobs_shape
        assert tuple(joined_rv.shape.eval()) == (1, 4)

        # Test logp
        logp = conditional_logp(
            {
                obs_rv: pt.as_tensor(obs_data)[obs_mask],
                unobs_rv: pt.as_tensor(unobs_data)[unobs_mask],
            }
        )
        obs_logp, unobs_logp = pytensor.function([], list(logp.values()))()
        if obs_component_selected:
            expected_obs_logp = pm.logp(rv, obs_data).eval()
            expected_unobs_logp = []
        else:
            expected_obs_logp = []
            expected_unobs_logp = pm.logp(rv, unobs_data).eval()
        np.testing.assert_allclose(obs_logp, expected_obs_logp)
        np.testing.assert_allclose(unobs_logp, expected_unobs_logp)

        if mutable_shape:
            shape[-1].set_value(7)
            assert tuple(joined_rv.shape.eval()) == (1, 7)

    def test_multivariate_constant_mask_unseparable(self):
        mask = pt.constant(np.array([[True, True, False, False]]))
        obs_data = np.array([[0.1, 0.4, 0.1, 0.4]])
        unobs_data = np.array([[0.4, 0.1, 0.4, 0.1]])

        rv = pm.Dirichlet.dist([1, 2, 3, 4], shape=(1, 4))
        (obs_rv, obs_mask), (unobs_rv, unobs_mask), joined_rv = create_partial_observed_rv(rv, mask)

        # Test types
        assert isinstance(obs_rv.owner.op, PartialObservedRV)
        assert isinstance(unobs_rv.owner.op, PartialObservedRV)

        # Test shapes
        assert tuple(obs_rv.shape.eval()) == (2,)
        assert tuple(unobs_rv.shape.eval()) == (2,)
        assert tuple(joined_rv.shape.eval()) == (1, 4)

        # Test logp
        logp = conditional_logp(
            {
                obs_rv: pt.as_tensor(obs_data)[obs_mask],
                unobs_rv: pt.as_tensor(unobs_data)[unobs_mask],
            }
        )
        obs_logp, unobs_logp = pytensor.function([], list(logp.values()))()

        # For non-separable cases the logp always shows up in the observed variable
        expected_logp = pm.logp(rv, [[0.1, 0.4, 0.4, 0.1]]).eval()
        np.testing.assert_almost_equal(obs_logp, expected_logp)
        np.testing.assert_array_equal(unobs_logp, [])

    def test_multivariate_shared_mask_separable(self):
        mask = shared(np.array([True]))
        obs_data = np.array([[0.1, 0.4, 0.1, 0.4]])
        unobs_data = np.array([[0.4, 0.1, 0.4, 0.1]])

        rv = pm.Dirichlet.dist([1, 2, 3, 4], shape=(1, 4))
        (obs_rv, obs_mask), (unobs_rv, unobs_mask), joined_rv = create_partial_observed_rv(rv, mask)

        # Test types
        # Multivariate RVs with shared masks on the last component are always unseparable.
        assert isinstance(obs_rv.owner.op, pm.Dirichlet)
        assert isinstance(unobs_rv.owner.op, pm.Dirichlet)

        # Test shapes
        assert tuple(obs_rv.shape.eval()) == (0, 4)
        assert tuple(unobs_rv.shape.eval()) == (1, 4)
        assert tuple(joined_rv.shape.eval()) == (1, 4)

        # Test logp
        logp = conditional_logp(
            {
                obs_rv: pt.as_tensor(obs_data)[obs_mask],
                unobs_rv: pt.as_tensor(unobs_data)[unobs_mask],
            }
        )
        logp_fn = pytensor.function([], list(logp.values()))
        obs_logp, unobs_logp = logp_fn()
        expected_logp = pm.logp(rv, unobs_data).eval()
        np.testing.assert_almost_equal(obs_logp, [])
        np.testing.assert_array_equal(unobs_logp, expected_logp)

        # Test that we can update a shared mask
        mask.set_value(np.array([False]))

        assert tuple(obs_rv.shape.eval()) == (1, 4)
        assert tuple(unobs_rv.shape.eval()) == (0, 4)

        new_expected_logp = pm.logp(rv, obs_data).eval()
        assert not np.isclose(expected_logp, new_expected_logp)  # Otherwise test is weak
        obs_logp, unobs_logp = logp_fn()
        np.testing.assert_almost_equal(obs_logp, new_expected_logp)
        np.testing.assert_array_equal(unobs_logp, [])

    @pytest.mark.parametrize("mutable_shape", (False, True))
    def test_multivariate_shared_mask_unseparable(self, mutable_shape):
        # Even if the mask is initially not mixing support dims,
        # it could later be changed in a way that does!
        mask = shared(np.array([[True, True, True, True]]))
        obs_data = np.array([[0.1, 0.4, 0.1, 0.4]])
        unobs_data = np.array([[0.4, 0.1, 0.4, 0.1]])

        if mutable_shape:
            shape = mask.shape
        else:
            shape = (1, 4)
        rv = pm.Dirichlet.dist([1, 2, 3, 4], shape=shape)
        (obs_rv, obs_mask), (unobs_rv, unobs_mask), joined_rv = create_partial_observed_rv(rv, mask)

        # Test types
        # Multivariate RVs with shared masks on the last component are always unseparable.
        assert isinstance(obs_rv.owner.op, PartialObservedRV)
        assert isinstance(unobs_rv.owner.op, PartialObservedRV)

        # Test shapes
        assert tuple(obs_rv.shape.eval()) == (0,)
        assert tuple(unobs_rv.shape.eval()) == (4,)
        assert tuple(joined_rv.shape.eval()) == (1, 4)

        # Test logp
        logp = conditional_logp(
            {
                obs_rv: pt.as_tensor(obs_data)[obs_mask],
                unobs_rv: pt.as_tensor(unobs_data)[unobs_mask],
            }
        )
        logp_fn = pytensor.function([], list(logp.values()))
        obs_logp, unobs_logp = logp_fn()
        # For non-separable cases the logp always shows up in the observed variable
        # Even in this case where all entries come from an unobserved component
        expected_logp = pm.logp(rv, unobs_data).eval()
        np.testing.assert_almost_equal(obs_logp, expected_logp)
        np.testing.assert_array_equal(unobs_logp, [])

        # Test that we can update a shared mask
        mask.set_value(np.array([[False, False, True, True]]))
        equivalent_value = np.array([0.1, 0.4, 0.4, 0.1])

        assert tuple(obs_rv.shape.eval()) == (2,)
        assert tuple(unobs_rv.shape.eval()) == (2,)

        new_expected_logp = pm.logp(rv, equivalent_value).eval()
        assert not np.isclose(expected_logp, new_expected_logp)  # Otherwise test is weak
        obs_logp, unobs_logp = logp_fn()
        np.testing.assert_almost_equal(obs_logp, new_expected_logp)
        np.testing.assert_array_equal(unobs_logp, [])

        if mutable_shape:
            mask.set_value(np.array([[False, False, True, False], [False, False, False, True]]))
            assert tuple(obs_rv.shape.eval()) == (6,)
            assert tuple(unobs_rv.shape.eval()) == (2,)

    def test_support_point(self):
        x = pm.GaussianRandomWalk.dist(init_dist=pm.Normal.dist(-5), mu=1, steps=9)
        ref_support_point = support_point(x).eval()
        assert not np.allclose(
            ref_support_point[::2], ref_support_point[1::2]
        )  # Otherwise test is weak

        (obs_x, _), (unobs_x, _), _ = create_partial_observed_rv(
            x, mask=np.array([False, True] * 5)
        )
        np.testing.assert_allclose(support_point(obs_x).eval(), ref_support_point[::2])
        np.testing.assert_allclose(support_point(unobs_x).eval(), ref_support_point[1::2])

    def test_wrong_mask(self):
        rv = pm.Normal.dist(shape=(5,))

        invalid_mask = np.array([0, 2, 4])
        with pytest.raises(ValueError, match="mask must be an array or tensor of boolean dtype"):
            create_partial_observed_rv(rv, invalid_mask)

        invalid_mask = np.zeros((1, 5), dtype=bool)
        with pytest.raises(ValueError, match="mask can't have more dims than rv"):
            create_partial_observed_rv(rv, invalid_mask)

    @pytest.mark.filterwarnings("error")
    def test_default_updates(self):
        mask = np.array([True, True, False])
        rv = pm.Normal.dist(shape=(3,))
        (obs_rv, _), (unobs_rv, _), joined_rv = create_partial_observed_rv(rv, mask)

        draws_obs_rv, draws_unobs_rv, draws_joined_rv = pm.draw(
            [obs_rv, unobs_rv, joined_rv], draws=2
        )

        assert np.all(draws_obs_rv[0] != draws_obs_rv[1])
        assert np.all(draws_unobs_rv[0] != draws_unobs_rv[1])
        assert np.all(draws_joined_rv[0] != draws_joined_rv[1])
