#   Copyright 2023 The PyMC Developers
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
import pytest
import scipy.stats as st

from pytensor.tensor.random.op import RandomVariable

import pymc as pm


class TestBound:
    """Tests for pm.Bound distribution"""

    def test_continuous(self):
        with pm.Model() as model:
            dist = pm.Normal.dist(mu=0, sigma=1)
            with pytest.warns(FutureWarning, match="Bound has been deprecated"):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "invalid value encountered in add", RuntimeWarning
                    )
                    UnboundedNormal = pm.Bound("unbound", dist, transform=None)
                    InfBoundedNormal = pm.Bound(
                        "infbound", dist, lower=-np.inf, upper=np.inf, transform=None
                    )
                LowerNormal = pm.Bound("lower", dist, lower=0, transform=None)
                UpperNormal = pm.Bound("upper", dist, upper=0, transform=None)
                BoundedNormal = pm.Bound("bounded", dist, lower=1, upper=10, transform=None)
                LowerNormalTransform = pm.Bound("lowertrans", dist, lower=1)
                UpperNormalTransform = pm.Bound("uppertrans", dist, upper=10)
                BoundedNormalTransform = pm.Bound("boundedtrans", dist, lower=1, upper=10)

        assert model.compile_fn(model.logp(LowerNormal), point_fn=False)(-1) == -np.inf
        assert model.compile_fn(model.logp(UpperNormal), point_fn=False)(1) == -np.inf
        assert model.compile_fn(model.logp(BoundedNormal), point_fn=False)(0) == -np.inf
        assert model.compile_fn(model.logp(BoundedNormal), point_fn=False)(11) == -np.inf

        assert model.compile_fn(model.logp(UnboundedNormal), point_fn=False)(0) != -np.inf
        assert model.compile_fn(model.logp(UnboundedNormal), point_fn=False)(11) != -np.inf
        assert model.compile_fn(model.logp(InfBoundedNormal), point_fn=False)(0) != -np.inf
        assert model.compile_fn(model.logp(InfBoundedNormal), point_fn=False)(11) != -np.inf

        assert model.compile_fn(model.logp(LowerNormalTransform), point_fn=False)(-1) != -np.inf
        assert model.compile_fn(model.logp(UpperNormalTransform), point_fn=False)(1) != -np.inf
        assert model.compile_fn(model.logp(BoundedNormalTransform), point_fn=False)(0) != -np.inf
        assert model.compile_fn(model.logp(BoundedNormalTransform), point_fn=False)(11) != -np.inf

        ref_dist = pm.Normal.dist(mu=0, sigma=1)
        assert np.allclose(
            model.compile_fn(model.logp(UnboundedNormal), point_fn=False)(5),
            pm.logp(ref_dist, 5).eval(),
        )
        assert np.allclose(
            model.compile_fn(model.logp(LowerNormal), point_fn=False)(5),
            pm.logp(ref_dist, 5).eval(),
        )
        assert np.allclose(
            model.compile_fn(model.logp(UpperNormal), point_fn=False)(-5),
            pm.logp(ref_dist, 5).eval(),
        )
        assert np.allclose(
            model.compile_fn(model.logp(BoundedNormal), point_fn=False)(5),
            pm.logp(ref_dist, 5).eval(),
        )

    def test_discrete(self):
        with pm.Model() as model:
            dist = pm.Poisson.dist(mu=4)
            with pytest.warns(FutureWarning, match="Bound has been deprecated"):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "invalid value encountered in add", RuntimeWarning
                    )
                    UnboundedPoisson = pm.Bound("unbound", dist)
                LowerPoisson = pm.Bound("lower", dist, lower=1)
                UpperPoisson = pm.Bound("upper", dist, upper=10)
                BoundedPoisson = pm.Bound("bounded", dist, lower=1, upper=10)

        assert model.compile_fn(model.logp(LowerPoisson), point_fn=False)(0) == -np.inf
        assert model.compile_fn(model.logp(UpperPoisson), point_fn=False)(11) == -np.inf
        assert model.compile_fn(model.logp(BoundedPoisson), point_fn=False)(0) == -np.inf
        assert model.compile_fn(model.logp(BoundedPoisson), point_fn=False)(11) == -np.inf

        assert model.compile_fn(model.logp(UnboundedPoisson), point_fn=False)(0) != -np.inf
        assert model.compile_fn(model.logp(UnboundedPoisson), point_fn=False)(11) != -np.inf

        ref_dist = pm.Poisson.dist(mu=4)
        assert np.allclose(
            model.compile_fn(model.logp(UnboundedPoisson), point_fn=False)(5),
            pm.logp(ref_dist, 5).eval(),
        )
        assert np.allclose(
            model.compile_fn(model.logp(LowerPoisson), point_fn=False)(5),
            pm.logp(ref_dist, 5).eval(),
        )
        assert np.allclose(
            model.compile_fn(model.logp(UpperPoisson), point_fn=False)(5),
            pm.logp(ref_dist, 5).eval(),
        )
        assert np.allclose(
            model.compile_fn(model.logp(BoundedPoisson), point_fn=False)(5),
            pm.logp(ref_dist, 5).eval(),
        )

    def create_invalid_distribution(self):
        class MyNormal(RandomVariable):
            name = "my_normal"
            ndim_supp = 0
            ndims_params = [0, 0]
            dtype = "floatX"

        my_normal = MyNormal()

        class InvalidDistribution(pm.Distribution):
            rv_op = my_normal

            @classmethod
            def dist(cls, mu=0, sigma=1, **kwargs):
                return super().dist([mu, sigma], **kwargs)

        return InvalidDistribution

    def test_arguments_checks(self):
        msg = "Observed Bound distributions are not supported"
        with pm.Model() as m:
            x = pm.Normal("x", 0, 1)
            with pytest.warns(FutureWarning, match="Bound has been deprecated"):
                with pytest.raises(ValueError, match=msg):
                    pm.Bound("bound", x, observed=5)

        msg = "Cannot transform discrete variable."
        with pm.Model() as m:
            x = pm.Poisson.dist(0.5)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "invalid value encountered in add", RuntimeWarning
                )
                with pytest.warns(FutureWarning, match="Bound has been deprecated"):
                    with pytest.raises(ValueError, match=msg):
                        pm.Bound("bound", x, transform=pm.distributions.transforms.log)

        msg = "Given dims do not exist in model coordinates."
        with pm.Model() as m:
            x = pm.Poisson.dist(0.5)
            with pytest.warns(FutureWarning, match="Bound has been deprecated"):
                with pytest.raises(ValueError, match=msg):
                    pm.Bound("bound", x, dims="random_dims")

        msg = "The dist x was already registered in the current model"
        with pm.Model() as m:
            x = pm.Normal("x", 0, 1)
            with pytest.warns(FutureWarning, match="Bound has been deprecated"):
                with pytest.raises(ValueError, match=msg):
                    pm.Bound("bound", x)

        msg = "Passing a distribution class to `Bound` is no longer supported"
        with pm.Model() as m:
            with pytest.warns(FutureWarning, match="Bound has been deprecated"):
                with pytest.raises(ValueError, match=msg):
                    pm.Bound("bound", pm.Normal)

        msg = "Bounding of MultiVariate RVs is not yet supported"
        with pm.Model() as m:
            x = pm.MvNormal.dist(np.zeros(3), np.eye(3))
            with pytest.warns(FutureWarning, match="Bound has been deprecated"):
                with pytest.raises(NotImplementedError, match=msg):
                    pm.Bound("bound", x)

        msg = "must be a Discrete or Continuous distribution subclass"
        with pm.Model() as m:
            x = self.create_invalid_distribution().dist()
            with pytest.warns(FutureWarning, match="Bound has been deprecated"):
                with pytest.raises(ValueError, match=msg):
                    pm.Bound("bound", x)

    def test_invalid_sampling(self):
        msg = "Cannot sample from a bounded variable"
        with pm.Model() as m:
            dist = pm.Normal.dist(mu=0, sigma=1)
            with pytest.warns(FutureWarning, match="Bound has been deprecated"):
                BoundedNormal = pm.Bound("bounded", dist, lower=1, upper=10)
            with pytest.raises(NotImplementedError, match=msg):
                pm.sample_prior_predictive()

    def test_bound_shapes(self):
        with pm.Model(coords={"sample": np.ones((2, 5))}) as m:
            dist = pm.Normal.dist(mu=0, sigma=1)
            with pytest.warns(FutureWarning, match="Bound has been deprecated"):
                bound_sized = pm.Bound("boundedsized", dist, lower=1, upper=10, size=(4, 5))
                bound_shaped = pm.Bound("boundedshaped", dist, lower=1, upper=10, shape=(3, 5))
                bound_dims = pm.Bound("boundeddims", dist, lower=1, upper=10, dims="sample")

        initial_point = m.initial_point()
        dist_size = initial_point["boundedsized_interval__"].shape
        dist_shape = initial_point["boundedshaped_interval__"].shape
        dist_dims = initial_point["boundeddims_interval__"].shape

        assert dist_size == (4, 5)
        assert dist_shape == (3, 5)
        assert dist_dims == (2, 5)

    def test_bound_dist(self):
        # Continuous
        bound = pm.Bound.dist(pm.Normal.dist(0, 1), lower=0)
        assert pm.logp(bound, -1).eval() == -np.inf
        assert np.isclose(pm.logp(bound, 1).eval(), st.norm(0, 1).logpdf(1))

        # Discrete
        bound = pm.Bound.dist(pm.Poisson.dist(1), lower=2)
        assert pm.logp(bound, 1).eval() == -np.inf
        assert np.isclose(pm.logp(bound, 2).eval(), st.poisson(1).logpmf(2))

    def test_array_bound(self):
        with pm.Model() as model:
            dist = pm.Normal.dist()
            with pytest.warns(FutureWarning, match="Bound has been deprecated"):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "invalid value encountered in add", RuntimeWarning
                    )
                    LowerPoisson = pm.Bound("lower", dist, lower=[1, None], transform=None)
                    UpperPoisson = pm.Bound("upper", dist, upper=[np.inf, 10], transform=None)
                BoundedPoisson = pm.Bound(
                    "bounded", dist, lower=[1, 2], upper=[9, 10], transform=None
                )

        first, second = model.compile_fn(model.logp(LowerPoisson, sum=False)[0], point_fn=False)(
            [0, 0]
        )
        assert first == -np.inf
        assert second != -np.inf

        first, second = model.compile_fn(model.logp(UpperPoisson, sum=False)[0], point_fn=False)(
            [11, 11]
        )
        assert first != -np.inf
        assert second == -np.inf

        first, second = model.compile_fn(model.logp(BoundedPoisson, sum=False)[0], point_fn=False)(
            [1, 1]
        )
        assert first != -np.inf
        assert second == -np.inf

        first, second = model.compile_fn(model.logp(BoundedPoisson, sum=False)[0], point_fn=False)(
            [10, 10]
        )
        assert first == -np.inf
        assert second != -np.inf
