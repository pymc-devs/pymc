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
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import scipy

from pytensor.tensor.random.basic import GeometricRV, NormalRV

from pymc import Censored, Model, draw, find_MAP
from pymc.distributions.continuous import Exponential, Gamma, TruncatedNormalRV
from pymc.distributions.shape_utils import change_dist_size
from pymc.distributions.transforms import _default_transform
from pymc.distributions.truncated import Truncated, TruncatedRV, _truncated
from pymc.exceptions import TruncationError
from pymc.logprob.abstract import _icdf
from pymc.logprob.basic import logcdf, logp
from pymc.logprob.transforms import IntervalTransform
from pymc.logprob.utils import ParameterValueError
from pymc.testing import assert_moment_is_expected


class IcdfNormalRV(NormalRV):
    """Normal RV that has icdf but not truncated dispatching"""


class RejectionNormalRV(NormalRV):
    """Normal RV that has neither icdf nor truncated dispatching."""


class IcdfGeometricRV(GeometricRV):
    """Geometric RV that has icdf but not truncated dispatching."""


class RejectionGeometricRV(GeometricRV):
    """Geometric RV that has neither icdf nor truncated dispatching."""


icdf_normal = no_moment_normal = IcdfNormalRV()
rejection_normal = RejectionNormalRV()
icdf_geometric = IcdfGeometricRV()
rejection_geometric = RejectionGeometricRV()


@_truncated.register(IcdfNormalRV)
@_truncated.register(RejectionNormalRV)
@_truncated.register(IcdfGeometricRV)
@_truncated.register(RejectionGeometricRV)
def _truncated_not_implemented(*args, **kwargs):
    raise NotImplementedError()


@_icdf.register(RejectionNormalRV)
@_icdf.register(RejectionGeometricRV)
def _icdf_not_implemented(*args, **kwargs):
    raise NotImplementedError()


@pytest.mark.parametrize("shape_info", ("shape", "dims", "observed"))
def test_truncation_specialized_op(shape_info):
    rng = pytensor.shared(np.random.default_rng())
    x = pt.random.normal(0, 10, rng=rng, name="x")

    with Model(coords={"dim": range(100)}) as m:
        if shape_info == "shape":
            xt = Truncated("xt", dist=x, lower=5, upper=15, shape=(100,))
        elif shape_info == "dims":
            xt = Truncated("xt", dist=x, lower=5, upper=15, dims=("dim",))
        elif shape_info == "observed":
            xt = Truncated(
                "xt",
                dist=x,
                lower=5,
                upper=15,
                observed=np.zeros(100),
            )
        else:
            raise ValueError(f"Not a valid shape_info parametrization: {shape_info}")

    assert isinstance(xt.owner.op, TruncatedNormalRV)
    assert xt.shape.eval() == (100,)

    # Test RNG is not reused
    assert xt.owner.inputs[0] is not rng

    lower_upper = pt.stack(xt.owner.inputs[5:])
    assert np.all(lower_upper.eval() == [5, 15])


@pytest.mark.parametrize("lower, upper", [(-1, np.inf), (-1, 1.5), (-np.inf, 1.5)])
@pytest.mark.parametrize("op_type", ["icdf", "rejection"])
def test_truncation_continuous_random(op_type, lower, upper):
    loc = 0.15
    scale = 10
    normal_op = icdf_normal if op_type == "icdf" else rejection_normal
    x = normal_op(loc, scale, name="x", size=100)

    xt = Truncated.dist(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)
    assert xt.type.dtype == x.type.dtype

    xt_draws = draw(xt, draws=5)
    assert np.all(xt_draws >= lower)
    assert np.all(xt_draws <= upper)
    assert np.unique(xt_draws).size == xt_draws.size

    # Compare with reference
    ref_xt = scipy.stats.truncnorm(
        (lower - loc) / scale,
        (upper - loc) / scale,
        loc,
        scale,
    )
    assert scipy.stats.cramervonmises(xt_draws.ravel(), ref_xt.cdf).pvalue > 0.001

    # Test max_n_steps
    xt = Truncated.dist(x, lower=lower, upper=upper, max_n_steps=1)
    if op_type == "icdf":
        xt_draws = draw(xt)
        assert np.all(xt_draws >= lower)
        assert np.all(xt_draws <= upper)
        assert np.unique(xt_draws).size == xt_draws.size
    else:
        with pytest.raises(TruncationError, match="^Truncation did not converge"):
            draw(xt)


@pytest.mark.parametrize("lower, upper", [(-1, np.inf), (-1, 1.5), (-np.inf, 1.5)])
@pytest.mark.parametrize("op_type", ["icdf", "rejection"])
def test_truncation_continuous_logp(op_type, lower, upper):
    loc = 0.15
    scale = 10
    op = icdf_normal if op_type == "icdf" else rejection_normal

    x = op(loc, scale, name="x")
    xt = Truncated.dist(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)

    xt_vv = xt.clone()
    xt_logp_fn = pytensor.function([xt_vv], logp(xt, xt_vv))

    ref_xt = scipy.stats.truncnorm(
        (lower - loc) / scale,
        (upper - loc) / scale,
        loc,
        scale,
    )
    for bound in (lower, upper):
        if np.isinf(bound):
            return
        for offset in (-1, 0, 1):
            test_xt_v = bound + offset
            assert np.isclose(xt_logp_fn(test_xt_v), ref_xt.logpdf(test_xt_v))


@pytest.mark.parametrize("lower, upper", [(-1, np.inf), (-1, 1.5), (-np.inf, 1.5)])
@pytest.mark.parametrize("op_type", ["icdf", "rejection"])
def test_truncation_continuous_logcdf(op_type, lower, upper):
    loc = 0.15
    scale = 10
    op = icdf_normal if op_type == "icdf" else rejection_normal

    x = op(loc, scale, name="x")
    xt = Truncated.dist(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)

    xt_vv = xt.clone()
    xt_logcdf_fn = pytensor.function([xt_vv], logcdf(xt, xt_vv))

    ref_xt = scipy.stats.truncnorm(
        (lower - loc) / scale,
        (upper - loc) / scale,
        loc,
        scale,
    )
    for bound in (lower, upper):
        if np.isinf(bound):
            return
        for offset in (-1, 0, 1):
            test_xt_v = bound + offset
            assert np.isclose(xt_logcdf_fn(test_xt_v), ref_xt.logcdf(test_xt_v))


@pytest.mark.parametrize("lower, upper", [(2, np.inf), (2, 5), (-np.inf, 5)])
@pytest.mark.parametrize("op_type", ["icdf", "rejection"])
def test_truncation_discrete_random(op_type, lower, upper):
    p = 0.2
    geometric_op = icdf_geometric if op_type == "icdf" else rejection_geometric

    x = geometric_op(p, name="x", size=500)
    xt = Truncated.dist(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)

    xt_draws = draw(xt)
    assert np.all(xt_draws >= lower)
    assert np.all(xt_draws <= upper)
    assert np.any(xt_draws == (max(1, lower)))
    if upper != np.inf:
        assert np.any(xt_draws == upper)

    # Test max_n_steps
    xt = Truncated.dist(x, lower=lower, upper=upper, max_n_steps=3)
    if op_type == "icdf":
        xt_draws = draw(xt)
        assert np.all(xt_draws >= lower)
        assert np.all(xt_draws <= upper)
        assert np.any(xt_draws == (max(1, lower)))
        if upper != np.inf:
            assert np.any(xt_draws == upper)
    else:
        with pytest.raises(TruncationError, match="^Truncation did not converge"):
            # Rejections sampling with probability = (1 - p ** max_n_steps) ** sample_size =
            # = (1 - 0.2 ** 3) ** 500 = 0.018
            # Still, this probability can be too high to make this test pass with any seed.
            draw(xt, random_seed=2297228)


@pytest.mark.parametrize("lower, upper", [(2, np.inf), (2, 5), (-np.inf, 5)])
@pytest.mark.parametrize("op_type", ["icdf", "rejection"])
def test_truncation_discrete_logp(op_type, lower, upper):
    p = 0.7
    op = icdf_geometric if op_type == "icdf" else rejection_geometric

    x = op(p, name="x")
    xt = Truncated.dist(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)

    xt_vv = xt.clone()
    xt_logp_fn = pytensor.function([xt_vv], logp(xt, xt_vv))

    ref_xt = scipy.stats.geom(p)
    log_norm = np.log(ref_xt.cdf(upper) - ref_xt.cdf(lower - 1))

    def ref_xt_logpmf(value):
        if value < lower or value > upper:
            return -np.inf
        return ref_xt.logpmf(value) - log_norm

    for bound in (lower, upper):
        if np.isinf(bound):
            continue
        for offset in (-1, 0, 1):
            test_xt_v = bound + offset
            assert np.isclose(xt_logp_fn(test_xt_v), ref_xt_logpmf(test_xt_v))

    # Check that it integrates to 1
    log_integral = scipy.special.logsumexp([xt_logp_fn(v) for v in range(min(upper + 1, 20))])
    assert np.isclose(log_integral, 0.0, atol=1e-5)


@pytest.mark.parametrize("lower, upper", [(2, np.inf), (2, 5), (-np.inf, 5)])
@pytest.mark.parametrize("op_type", ["icdf", "rejection"])
def test_truncation_discrete_logcdf(op_type, lower, upper):
    p = 0.7
    op = icdf_geometric if op_type == "icdf" else rejection_geometric

    x = op(p, name="x")
    xt = Truncated.dist(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)

    xt_vv = xt.clone()
    xt_logcdf_fn = pytensor.function([xt_vv], logcdf(xt, xt_vv))

    ref_xt = scipy.stats.geom(p)
    log_norm = np.log(ref_xt.cdf(upper) - ref_xt.cdf(lower - 1))

    def ref_xt_logcdf(value):
        if value < lower:
            return -np.inf
        elif value > upper:
            return 0.0

        return np.log(ref_xt.cdf(value) - ref_xt.cdf(lower - 1)) - log_norm

    for bound in (lower, upper):
        if np.isinf(bound):
            continue
        for offset in (-1, 0, 1):
            test_xt_v = bound + offset
            assert np.isclose(xt_logcdf_fn(test_xt_v), ref_xt_logcdf(test_xt_v))


def test_truncation_exceptions():
    with pytest.raises(ValueError, match="lower and upper cannot both be None"):
        Truncated.dist(pt.random.normal())

    # Truncation does not work with SymbolicRV inputs
    with pytest.raises(
        NotImplementedError,
        match="Truncation not implemented for SymbolicRandomVariable CensoredRV",
    ):
        Truncated.dist(Censored.dist(pt.random.normal(), lower=-1, upper=1), -1, 1)

    with pytest.raises(
        NotImplementedError,
        match="Truncation not implemented for multivariate distributions",
    ):
        Truncated.dist(pt.random.dirichlet([1, 1, 1]), -1, 1)


def test_truncation_logprob_bound_check():
    x = pt.random.normal(name="x")
    xt = Truncated.dist(x, lower=5, upper=-5)
    with pytest.raises(ParameterValueError):
        logp(xt, 0).eval()


def test_change_truncated_size():
    x = Truncated.dist(icdf_normal(0, [1, 2, 3]), lower=-1, size=(2, 3))
    x.eval().shape == (2, 3)

    new_x = change_dist_size(x, (4, 3))
    assert isinstance(new_x.owner.op, TruncatedRV)
    new_x.eval().shape == (4, 3)

    new_x = change_dist_size(x, (4, 3), expand=True)
    assert isinstance(new_x.owner.op, TruncatedRV)
    new_x.eval().shape == (4, 3, 2, 3)


def test_truncated_default_transform():
    base_dist = rejection_geometric(1)
    x = Truncated.dist(base_dist, lower=None, upper=5)
    assert _default_transform(x.owner.op, x) is None

    base_dist = rejection_normal(0, 1)
    x = Truncated.dist(base_dist, lower=None, upper=5)
    assert isinstance(_default_transform(x.owner.op, x), IntervalTransform)


def test_truncated_transform_logp():
    with Model() as m:
        base_dist = rejection_normal(0, 1)
        x = Truncated("x", base_dist, lower=0, upper=None, transform=None)
        y = Truncated("y", base_dist, lower=0, upper=None)
        logp_eval = m.compile_logp(sum=False)({"x": -1, "y_interval__": -1})
    assert logp_eval[0] == -np.inf
    assert np.isfinite(logp_eval[1])


@pytest.mark.parametrize(
    "truncated_dist, lower, upper, shape, expected",
    [
        # Moment of truncated dist can be used
        (icdf_normal(0, 1), -1, 2, None, 0),
        # Moment of truncated dist cannot be used, both bounds are finite
        (icdf_normal(3, 1), -1, 2, (2,), np.full((2,), 3 / 2)),
        # Moment of truncated dist cannot be used, lower bound is finite
        (icdf_normal(-3, 1), -1, None, (2, 3), np.full((2, 3), 0)),
        # Moment of truncated dist can be used for 1st and 3rd mus, upper bound is finite
        (icdf_normal([0, 3, 3], 1), None, [2, 2, 4], (4, 3), np.full((4, 3), [0, 1, 3])),
    ],
)
def test_truncated_moment(truncated_dist, lower, upper, shape, expected):
    with Model() as model:
        Truncated("x", dist=truncated_dist, lower=lower, upper=upper, shape=shape)
    assert_moment_is_expected(model, expected)


def test_truncated_inference():
    # exercise 3.3, p 47, from David MacKay Information Theory book
    lam_true = 3
    lower = 0
    upper = 5

    rng = np.random.default_rng(260)
    x = rng.exponential(lam_true, size=5000)
    obs = x[np.where(~((x < lower) | (x > upper)))]  # remove values outside range

    with Model() as m:
        lam = Exponential("lam", lam=1 / 5)  # prior exponential with mean of 5
        Truncated(
            "x",
            dist=Exponential.dist(lam=1 / lam),
            lower=lower,
            upper=upper,
            observed=obs,
        )

        map = find_MAP(progressbar=False)

    assert np.isclose(map["lam"], lam_true, atol=0.1)


def test_truncated_gamma():
    # Regression test for https://github.com/pymc-devs/pymc/issues/6931
    alpha = 3.0
    beta = 3.0
    upper = 2.5
    x = np.linspace(0.0, upper + 0.5, 100)

    gamma_scipy = scipy.stats.gamma(a=alpha, scale=1.0 / beta)
    logp_scipy = gamma_scipy.logpdf(x) - gamma_scipy.logcdf(upper)
    logp_scipy[x > upper] = -np.inf

    gamma_trunc_pymc = Truncated.dist(
        Gamma.dist(alpha=alpha, beta=beta),
        upper=upper,
    )
    logp_pymc = logp(gamma_trunc_pymc, x).eval()
    np.testing.assert_allclose(
        logp_pymc,
        logp_scipy,
    )

    # Changing the size used to invert the beta Gamma parameter again
    resized_gamma_trunc_pymc = change_dist_size(gamma_trunc_pymc, new_size=x.shape)
    logp_resized_pymc = logp(resized_gamma_trunc_pymc, x).eval()
    np.testing.assert_allclose(
        logp_resized_pymc,
        logp_scipy,
    )
