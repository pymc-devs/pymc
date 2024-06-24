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
import warnings

import cloudpickle
import numpy as np
import pytensor
import pytest

from numpy import random as npr
from pytensor import scan
from pytensor import tensor as pt
from scipy import stats as st

import pymc as pm

from pymc import (
    CustomDist,
    Deterministic,
    DiracDelta,
    HalfNormal,
    LogNormal,
    Model,
    Normal,
    draw,
    logcdf,
    logp,
    sample,
)
from pymc.distributions.custom import CustomDistRV, CustomSymbolicDistRV
from pymc.distributions.distribution import support_point
from pymc.distributions.shape_utils import change_dist_size, rv_size_is_none, to_tuple
from pymc.distributions.transforms import log
from pymc.exceptions import BlockModelAccessError
from pymc.pytensorf import collect_default_updates
from pymc.testing import assert_support_point_is_expected


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
            ("custom_support_point", (), 5),
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
                x = CustomDist("x", moment=moment, size=())
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
        x = CustomDist.dist(
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
                lambda mu, sigma, size: pt.exp(pm.Normal.dist(mu, sigma, size=size) + 1.0),
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
            if rv_size_is_none(size):
                size = pt.broadcast_shape(p, sigma)
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
            CustomDist.dist(random=old_random, class_name="custom_dist")

        # New API is fine
        CustomDist.dist(dist=old_random, class_name="custom_dist")

    def test_scan(self):
        def trw(nu, sigma, steps, size):
            if rv_size_is_none(size):
                size = ()

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
                CustomDist.dist(mus[0], sds[0], dist=shifted_normal),
                CustomDist.dist(mus[1], sds[1], dist=shifted_normal),
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

        assert CustomDist.dist(dist=dist)

    def test_nested_custom_dist(self):
        """Test we can create CustomDist that creates another CustomDist"""

        def dist(size=None):
            def inner_dist(size=None):
                return pm.Normal.dist(size=size)

            inner_dist = CustomDist.dist(dist=inner_dist, size=size)
            return pt.exp(inner_dist)

        rv = CustomDist.dist(dist=dist)
        np.testing.assert_allclose(
            pm.logp(rv, 1.0).eval(),
            pm.logp(pm.LogNormal.dist(), 1.0).eval(),
        )

    def test_signature(self):
        def dist(p, size):
            return -pm.Categorical.dist(p=p, size=size)

        out = CustomDist.dist([0.25, 0.75], dist=dist, signature="(p)->()")
        # Size and updates are added automatically to the signature
        assert out.owner.op.extended_signature == "[size],(p),[rng]->(),[rng]"
        assert out.owner.op.ndim_supp == 0
        assert out.owner.op.ndims_params == [1]

        # When recreated internally, the whole signature may already be known
        out = CustomDist.dist([0.25, 0.75], dist=dist, signature="[size],(p),[rng]->(),[rng]")
        assert out.owner.op.extended_signature == "[size],(p),[rng]->(),[rng]"
        assert out.owner.op.ndim_supp == 0
        assert out.owner.op.ndims_params == [1]

        # A safe signature can be inferred from ndim_supp and ndims_params
        out = CustomDist.dist([0.25, 0.75], dist=dist, ndim_supp=0, ndims_params=[1])
        assert out.owner.op.extended_signature == "[size],(i00),[rng]->(),[rng]"
        assert out.owner.op.ndim_supp == 0
        assert out.owner.op.ndims_params == [1]

        # Otherwise be default we assume everything is scalar, even though it's wrong in this case
        out = CustomDist.dist([0.25, 0.75], dist=dist)
        assert out.owner.op.extended_signature == "[size],(),[rng]->(),[rng]"
        assert out.owner.op.ndim_supp == 0
        assert out.owner.op.ndims_params == [0]
