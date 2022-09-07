import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy as sp
import scipy.stats as st

from aeppl import factorized_joint_logprob, joint_logprob
from aeppl.transforms import LogTransform, TransformValuesRewrite
from tests.utils import assert_no_rvs


@aesara.config.change_flags(compute_test_value="raise")
def test_continuous_rv_clip():
    x_rv = at.random.normal(0.5, 1)
    cens_x_rv = at.clip(x_rv, -2, 2)

    cens_x_vv = cens_x_rv.clone()
    cens_x_vv.tag.test_value = 0

    logp = joint_logprob({cens_x_rv: cens_x_vv})
    assert_no_rvs(logp)

    logp_fn = aesara.function([cens_x_vv], logp)
    ref_scipy = st.norm(0.5, 1)

    assert logp_fn(-3) == -np.inf
    assert logp_fn(3) == -np.inf

    assert np.isclose(logp_fn(-2), ref_scipy.logcdf(-2))
    assert np.isclose(logp_fn(2), ref_scipy.logsf(2))
    assert np.isclose(logp_fn(0), ref_scipy.logpdf(0))


def test_discrete_rv_clip():
    x_rv = at.random.poisson(2)
    cens_x_rv = at.clip(x_rv, 1, 4)

    cens_x_vv = cens_x_rv.clone()

    logp = joint_logprob({cens_x_rv: cens_x_vv})
    assert_no_rvs(logp)

    logp_fn = aesara.function([cens_x_vv], logp)
    ref_scipy = st.poisson(2)

    assert logp_fn(0) == -np.inf
    assert logp_fn(5) == -np.inf

    assert np.isclose(logp_fn(1), ref_scipy.logcdf(1))
    assert np.isclose(logp_fn(4), np.logaddexp(ref_scipy.logsf(4), ref_scipy.logpmf(4)))
    assert np.isclose(logp_fn(2), ref_scipy.logpmf(2))


def test_one_sided_clip():
    x_rv = at.random.normal(0, 1)
    lb_cens_x_rv = at.clip(x_rv, -1, x_rv)
    ub_cens_x_rv = at.clip(x_rv, x_rv, 1)

    lb_cens_x_vv = lb_cens_x_rv.clone()
    ub_cens_x_vv = ub_cens_x_rv.clone()

    lb_logp = joint_logprob({lb_cens_x_rv: lb_cens_x_vv})
    ub_logp = joint_logprob({ub_cens_x_rv: ub_cens_x_vv})
    assert_no_rvs(lb_logp)
    assert_no_rvs(ub_logp)

    logp_fn = aesara.function([lb_cens_x_vv, ub_cens_x_vv], [lb_logp, ub_logp])
    ref_scipy = st.norm(0, 1)

    assert np.all(np.array(logp_fn(-2, 2)) == -np.inf)
    assert np.all(np.array(logp_fn(2, -2)) != -np.inf)
    np.testing.assert_almost_equal(logp_fn(-1, 1), ref_scipy.logcdf(-1))
    np.testing.assert_almost_equal(logp_fn(1, -1), ref_scipy.logpdf(-1))


def test_useless_clip():
    x_rv = at.random.normal(0.5, 1, size=3)
    cens_x_rv = at.clip(x_rv, x_rv, x_rv)

    cens_x_vv = cens_x_rv.clone()

    logp = joint_logprob({cens_x_rv: cens_x_vv}, sum=False)
    assert_no_rvs(logp)

    logp_fn = aesara.function([cens_x_vv], logp)
    ref_scipy = st.norm(0.5, 1)

    np.testing.assert_allclose(logp_fn([-2, 0, 2]), ref_scipy.logpdf([-2, 0, 2]))


def test_random_clip():
    lb_rv = at.random.normal(0, 1, size=2)
    x_rv = at.random.normal(0, 2)
    cens_x_rv = at.clip(x_rv, lb_rv, [1, 1])

    lb_vv = lb_rv.clone()
    cens_x_vv = cens_x_rv.clone()
    logp = joint_logprob({cens_x_rv: cens_x_vv, lb_rv: lb_vv}, sum=False)
    assert_no_rvs(logp)

    logp_fn = aesara.function([lb_vv, cens_x_vv], logp)
    res = logp_fn([0, -1], [-1, -1])
    assert res[0] == -np.inf
    assert res[1] != -np.inf


def test_broadcasted_clip_constant():
    lb_rv = at.random.uniform(0, 1)
    x_rv = at.random.normal(0, 2)
    cens_x_rv = at.clip(x_rv, lb_rv, [1, 1])

    lb_vv = lb_rv.clone()
    cens_x_vv = cens_x_rv.clone()

    logp = joint_logprob({cens_x_rv: cens_x_vv, lb_rv: lb_vv})
    assert_no_rvs(logp)


def test_broadcasted_clip_random():
    lb_rv = at.random.normal(0, 1)
    x_rv = at.random.normal(0, 2, size=2)
    cens_x_rv = at.clip(x_rv, lb_rv, 1)

    lb_vv = lb_rv.clone()
    cens_x_vv = cens_x_rv.clone()

    logp = joint_logprob({cens_x_rv: cens_x_vv, lb_rv: lb_vv})
    assert_no_rvs(logp)


def test_fail_base_and_clip_have_values():
    """Test failure when both base_rv and clipped_rv are given value vars"""
    x_rv = at.random.normal(0, 1)
    cens_x_rv = at.clip(x_rv, x_rv, 1)
    cens_x_rv.name = "cens_x"

    x_vv = x_rv.clone()
    cens_x_vv = cens_x_rv.clone()
    with pytest.raises(RuntimeError, match="could not be derived: {cens_x}"):
        factorized_joint_logprob({cens_x_rv: cens_x_vv, x_rv: x_vv})


def test_fail_multiple_clip_single_base():
    """Test failure when multiple clipped_rvs share a single base_rv"""
    base_rv = at.random.normal(0, 1)
    cens_rv1 = at.clip(base_rv, -1, 1)
    cens_rv1.name = "cens1"
    cens_rv2 = at.clip(base_rv, -1, 1)
    cens_rv2.name = "cens2"

    cens_vv1 = cens_rv1.clone()
    cens_vv2 = cens_rv2.clone()
    with pytest.raises(RuntimeError, match="could not be derived: {cens2}"):
        factorized_joint_logprob({cens_rv1: cens_vv1, cens_rv2: cens_vv2})


def test_deterministic_clipping():
    x_rv = at.random.normal(0, 1)
    clip = at.clip(x_rv, 0, 0)
    y_rv = at.random.normal(clip, 1)

    x_vv = x_rv.clone()
    y_vv = y_rv.clone()
    logp = joint_logprob({x_rv: x_vv, y_rv: y_vv})
    assert_no_rvs(logp)

    logp_fn = aesara.function([x_vv, y_vv], logp)
    assert np.isclose(
        logp_fn(-1, 1),
        st.norm(0, 1).logpdf(-1) + st.norm(0, 1).logpdf(1),
    )


def test_clip_transform():
    x_rv = at.random.normal(0.5, 1)
    cens_x_rv = at.clip(x_rv, 0, x_rv)

    cens_x_vv = cens_x_rv.clone()

    transform = TransformValuesRewrite({cens_x_vv: LogTransform()})
    logp = joint_logprob({cens_x_rv: cens_x_vv}, extra_rewrites=transform)

    cens_x_vv_testval = -1
    obs_logp = logp.eval({cens_x_vv: cens_x_vv_testval})
    exp_logp = (
        sp.stats.norm(0.5, 1).logpdf(np.exp(cens_x_vv_testval)) + cens_x_vv_testval
    )

    assert np.isclose(obs_logp, exp_logp)
