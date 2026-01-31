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
import warnings

import numpy as np
import numpy.random as npr
import numpy.testing as npt
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.stats as st

from pytensor import shared
from pytensor.tensor import NoneConst, TensorVariable
from pytensor.tensor.random.utils import normalize_size_param

import pymc as pm

from pymc.distributions import (
    Censored,
    Flat,
    MvNormal,
    MvStudentT,
    Normal,
)
from pymc.distributions.distribution import (
    PartialObservedRV,
    SymbolicRandomVariable,
    _support_point,
    create_partial_observed_rv,
    support_point,
)
from pymc.distributions.shape_utils import change_dist_size
from pymc.logprob.basic import conditional_logp, logp
from pymc.pytensorf import compile, normalize_rng_param
from pymc.testing import (
    BaseTestDistributionRandom,
    I,
    assert_support_point_is_expected,
    check_icdf,
    check_logcdf,
    check_logp,
)


class TestBugfixes:
    @pytest.mark.parametrize("dist_cls,kwargs", [(MvNormal, {}), (MvStudentT, {"nu": 2})])
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


def test_all_distributions_have_support_points():
    import pymc.distributions as dist_module

    from pymc.distributions.distribution import DistributionMeta

    dists = (getattr(dist_module, dist) for dist in dist_module.__all__)
    dists = (dist for dist in dists if isinstance(dist, DistributionMeta))
    generic_func = _support_point.dispatch(object)
    missing_support_points = {
        dist
        for dist in dists
        if (
            getattr(dist, "rv_type", None) is not None
            and _support_point.dispatch(dist.rv_type) is generic_func
        )
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
        fn = compile(inputs=[], outputs=x, random_seed=431)
        assert fn() != fn()

        # Check that custom updates are respected, by using one that's broken
        next_rng, x = SymbolicRVCustomUpdates(
            inputs=[dummy_rng],
            outputs=[dummy_next_rng, dummy_x],
            ndim_supp=0,
        )(rng)
        with pytest.raises(
            ValueError,
            match="No update found for at least one RNG used in SymbolicRVCustomUpdates",
        ):
            compile(inputs=[], outputs=x, random_seed=431)

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

    def test_change_dist_size_none(self):
        class TestRV(SymbolicRandomVariable):
            extended_signature = "[rng],[size]->[rng],(n)"

            @classmethod
            def rv_op(cls, size=None, rng=None):
                rng = normalize_rng_param(rng)
                size = normalize_size_param(size)
                next_rng, draws = Normal.dist(size=size, rng=rng).owner.outputs
                return cls(inputs=[rng, size], outputs=[next_rng, draws])(rng, size)

        size = NoneConst
        rv = TestRV.rv_op(size=size)
        assert rv.type.shape == ()

        resized_rv = change_dist_size(rv, new_size=5)
        assert resized_rv.type.shape == (5,)

        resized_rv = change_dist_size(rv, new_size=5, expand=True)
        assert resized_rv.type.shape == (5,)

    def test_logccdf_with_extended_signature(self):
        """Test logccdf registration for SymbolicRandomVariable with extended_signature."""
        from pymc.distributions.dist_math import normal_lccdf
        from pymc.distributions.distribution import Distribution

        class TestDistWithLogccdf(Distribution):
            # Create a SymbolicRandomVariable type with extended_signature
            rv_type = type(
                "TestRVWithLogccdf",
                (SymbolicRandomVariable,),
                {"extended_signature": "[rng],[size],(),()->[rng],()"},
            )

            @classmethod
            def dist(cls, mu, sigma, **kwargs):
                mu = pt.as_tensor(mu)
                sigma = pt.as_tensor(sigma)
                return super().dist([mu, sigma], **kwargs)

            @classmethod
            def rv_op(cls, mu, sigma, size=None, rng=None):
                rng = normalize_rng_param(rng)
                size = normalize_size_param(size)
                # Internally uses Normal, but wrapped in SymbolicRandomVariable
                next_rng, draws = Normal.dist(mu, sigma, size=size, rng=rng).owner.outputs
                return cls.rv_type(
                    inputs=[rng, size, mu, sigma],
                    outputs=[next_rng, draws],
                    ndim_supp=0,
                )(rng, size, mu, sigma)

            # This logccdf will be registered via params_idxs path
            def logccdf(value, mu, sigma):
                return normal_lccdf(mu, sigma, value)

        rv = TestDistWithLogccdf.dist(0, 1)
        result = pm.logccdf(rv, 0.5).eval()
        expected = st.norm(0, 1).logsf(0.5)  # ≈ -0.994
        npt.assert_allclose(result, expected)


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
        reference_dist = lambda self: self.diracdelta_rng_fn  # noqa: E731
        checks_to_run = [
            "check_pymc_params_match_rv_op",
            "check_pymc_draws_match_reference",
            "check_rv_size",
        ]

        @pytest.mark.parametrize("floatX", ["float32", "float64"])
        def test_dtype(self, floatX):
            with pytensor.config.change_flags(floatX=floatX):
                assert pm.DiracDelta.dist(2**4).dtype == "int8"
                assert pm.DiracDelta.dist(2**16).dtype == "int32"
                assert pm.DiracDelta.dist(2**32).dtype == "int64"
                assert pm.DiracDelta.dist(2.0).dtype == floatX

        def test_icdf(self):
            check_icdf(
                pm.DiracDelta,
                {"c": I},
                lambda q, c: np.full_like(q, c),
            )


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
        np.testing.assert_almost_equal(unobs_logp, expected_logp)

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
