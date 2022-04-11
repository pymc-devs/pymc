import contextlib

import aesara.tensor as at
import numpy as np
import pytest
import scipy.stats as stats
from aesara import function

from aeppl.dists import dirac_delta
from aeppl.logprob import ParameterValueError, icdf, logcdf, logprob

# @pytest.fixture(scope="module", autouse=True)
# def set_aesara_flags():
#     with aesara.config.change_flags(cxx=""):
#         yield


def scipy_logprob(obs, p):
    if p.ndim > 1:
        if p.ndim > obs.ndim:
            obs = obs[((None,) * (p.ndim - obs.ndim) + (Ellipsis,))]
        elif p.ndim < obs.ndim:
            p = p[((None,) * (obs.ndim - p.ndim) + (Ellipsis,))]

        pattern = (p.ndim - 1,) + tuple(range(p.ndim - 1))
        return np.log(np.take_along_axis(p.transpose(pattern), obs, 0))
    else:
        return np.log(p[obs])


def create_aesara_params(dist_params, obs, size):
    dist_params_at = []
    for p in dist_params:
        p_aet = at.as_tensor(p).type()
        p_aet.tag.test_value = p
        dist_params_at.append(p_aet)

    size_at = []
    for s in size:
        s_aet = at.iscalar()
        s_aet.tag.test_value = s
        size_at.append(s_aet)

    obs_at = at.as_tensor(obs).type()
    obs_at.tag.test_value = obs

    return dist_params_at, obs_at, size_at


def scipy_logprob_tester(
    rv_var, obs, dist_params, test_fn=None, check_broadcastable=True, test="logprob"
):
    """Test for correspondence between `RandomVariable` and NumPy shape and
    broadcast dimensions.
    """
    if test_fn is None:
        name = getattr(rv_var.owner.op, "name", None)

        if name is None:
            name = rv_var.__name__

        test_fn = getattr(stats, name)

    if test == "logprob":
        aesara_res = logprob(rv_var, at.as_tensor(obs))
    elif test == "logcdf":
        aesara_res = logcdf(rv_var, at.as_tensor(obs))
    elif test == "icdf":
        aesara_res = icdf(rv_var, at.as_tensor(obs))
    else:
        raise ValueError(f"test must be one of (logprob, logcdf, icdf), got {test}")

    aesara_res_val = aesara_res.eval(dist_params)

    numpy_res = np.asarray(test_fn(obs, *dist_params.values()))

    assert aesara_res.type.numpy_dtype.kind == numpy_res.dtype.kind

    if check_broadcastable:
        numpy_shape = np.shape(numpy_res)
        numpy_bcast = [s == 1 for s in numpy_shape]
        np.testing.assert_array_equal(aesara_res.type.broadcastable, numpy_bcast)

    np.testing.assert_array_equal(aesara_res_val.shape, numpy_res.shape)

    np.testing.assert_array_almost_equal(aesara_res_val, numpy_res, 4)


@pytest.mark.parametrize(
    "dist_params, obs, size",
    [
        ((0, 1), np.array([0, 0.5, 1, -1], dtype=np.float64), ()),
        ((-2, -1), np.array([0, 0.5, 1, -1, -1.5], dtype=np.float64), ()),
    ],
)
def test_uniform_logprob(dist_params, obs, size):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.uniform(*dist_params_at, size=size_at)

    def scipy_logprob(obs, l, u):
        return stats.uniform.logpdf(obs, loc=l, scale=u - l)

    scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size",
    [
        ((0, 1), np.array([-1, 0, 0.5, 1, 2], dtype=np.float64), ()),
        ((-2, -1), np.array([-3, -2, -0.5, -1, 0], dtype=np.float64), ()),
    ],
)
def test_uniform_logcdf(dist_params, obs, size):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.uniform(*dist_params_at, size=size_at)

    def scipy_logcdf(obs, l, u):
        return stats.uniform.logcdf(obs, loc=l, scale=u - l)

    scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logcdf, test="logcdf")


@pytest.mark.parametrize(
    "dist_params, obs, size",
    [
        ((0, 1), np.array([0, 0.5, 1, -1], dtype=np.float64), ()),
        ((-1, 20), np.array([0, 0.5, 1, -1], dtype=np.float64), ()),
        ((-1, 20), np.array([0, 0.5, 1, -1], dtype=np.float64), (2, 3)),
    ],
)
def test_normal_logprob(dist_params, obs, size):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.normal(*dist_params_at, size=size_at)

    scipy_logprob_tester(x, obs, dist_params, test_fn=stats.norm.logpdf)


@pytest.mark.parametrize(
    "dist_params, obs, size",
    [
        ((0, 1), np.array([0, 0.5, 1, -1], dtype=np.float64), ()),
        ((-1, 20), np.array([0, 0.5, 1, -1], dtype=np.float64), ()),
        ((-1, 20), np.array([0, 0.5, 1, -1], dtype=np.float64), (2, 3)),
    ],
)
def test_normal_logcdf(dist_params, obs, size):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.normal(*dist_params_at, size=size_at)

    scipy_logprob_tester(x, obs, dist_params, test_fn=stats.norm.logcdf, test="logcdf")


@pytest.mark.parametrize(
    "dist_params, obs, size",
    [
        ((0, 1), np.array([-0.5, 0, 0.3, 0.5, 1, 1.5], dtype=np.float64), ()),
        ((-1, 20), np.array([-0.5, 0, 0.3, 0.5, 1, 1.5], dtype=np.float64), ()),
        ((-1, 20), np.array([-0.5, 0, 0.3, 0.5, 1, 1.5], dtype=np.float64), (2, 3)),
    ],
)
def test_normal_icdf(dist_params, obs, size):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.normal(*dist_params_at, size=size_at)

    scipy_logprob_tester(x, obs, dist_params, test_fn=stats.norm.ppf, test="icdf")


@pytest.mark.parametrize(
    "dist_params, obs, size",
    [
        ((0, 1), np.array([0, 0.5, 1, -1], dtype=np.float64), ()),
        ((-1, 20), np.array([0, 0.5, 1, -1], dtype=np.float64), ()),
        ((-1, 20), np.array([0, 0.5, 1, -1], dtype=np.float64), (2, 3)),
    ],
)
def test_halfnormal_logprob(dist_params, obs, size):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.halfnormal(*dist_params_at, size=size_at)

    scipy_logprob_tester(x, obs, dist_params, test_fn=stats.halfnorm.logpdf)


@pytest.mark.parametrize(
    "dist_params, obs, size",
    [
        ((0.5, 0.5), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), ()),
        ((1.5, 1.5), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), ()),
        ((1.5, 1.5), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (2, 3)),
        ((1.5, 0.5), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), ()),
    ],
)
def test_beta_logprob(dist_params, obs, size):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.beta(*dist_params_at, size=size_at)

    scipy_logprob_tester(x, obs, dist_params, test_fn=stats.beta.logpdf)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1,), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), True),
        ((1.5,), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
        ((1.5,), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (2, 3), False),
        ((10,), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
    ],
)
def test_exponential_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.exponential(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, mu):
        return stats.expon.logpdf(obs, scale=mu)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size",
    [
        ((-1, 1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), ()),
        ((1.5, 1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), ()),
        ((1.5, 2.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (2, 3)),
        ((10, 3.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), ()),
    ],
)
def test_laplace_logprob(dist_params, obs, size):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.laplace(*dist_params_at, size=size_at)

    def scipy_logprob(obs, mu, b):
        return stats.laplace.logpdf(obs, loc=mu, scale=b)

    scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1, -1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), True),
        ((-1.5, 10.5), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
        ((1.5, 2.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (2, 3), False),
        ((10, 1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
    ],
)
def test_lognormal_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.lognormal(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, mu, sigma):
        return stats.lognorm.logpdf(obs, s=sigma, scale=np.exp(mu))

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1, -1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), True),
        ((1.5, 10.5), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
        ((1.5, 2.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (2, 3), False),
        ((10, 1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
    ],
)
def test_pareto_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.pareto(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, b, scale):
        return stats.pareto.logpdf(obs, b, scale=scale)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1, -1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), True),
        ((-1.5, 10.5), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
        ((1.5, 2.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (2, 3), False),
        ((10, 1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
    ],
)
def test_halfcauchy_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.halfcauchy(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, alpha, beta):
        return stats.halfcauchy.logpdf(obs, loc=alpha, scale=beta)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1, -1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), True),
        ((1.5, 10.5), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
        ((1.5, 2.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (2, 3), False),
        ((10, 1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
    ],
)
def test_gamma_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.gamma(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, alpha, inv_beta):
        return stats.gamma.logpdf(obs, alpha, scale=1.0 / inv_beta)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1, -1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), True),
        ((1.5, 10.5), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
        ((1.5, 2.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (2, 3), False),
        ((10, 1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
    ],
)
def test_invgamma_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.invgamma(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, alpha, beta):
        return stats.invgamma.logpdf(obs, alpha, scale=beta)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        # XXX: SciPy returns `inf` for `stats.chi2.logpdf(0, 1.5)`; we
        # return `-inf`
        ((-1,), np.array([0.5, 1, 10, -1], dtype=np.float64), (), True),
        ((1.5,), np.array([0.5, 1, 10, -1], dtype=np.float64), (), False),
        ((1.5,), np.array([0.5, 1, 10, -1], dtype=np.float64), (2, 3), False),
        ((10,), np.array([0.5, 1, 10, -1], dtype=np.float64), (), False),
    ],
)
def test_chisquare_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.chisquare(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, df):
        return stats.chi2.logpdf(obs, df)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1, -1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), True),
        ((1.5, 10.5), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
        ((1.5, 2.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (2, 3), False),
        ((10, 1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
    ],
)
def test_wald_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.wald(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, mean, scale):
        return stats.invgauss.logpdf(obs, mean / scale, scale=scale)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1, -1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), True),
        ((1.5, 10.5), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
        ((1.5, 2.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (2, 3), False),
        ((10, 1.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
    ],
)
def test_weibull_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.weibull(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, alpha, beta):
        return stats.weibull_min.logpdf(obs, alpha, scale=beta)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1, -1.0), np.array([-np.pi, -0.5, 0, 1, np.pi], dtype=np.float64), (), True),
        (
            (1.5, 10.5),
            np.array([-np.pi, -0.5, 0, 1, np.pi], dtype=np.float64),
            (),
            False,
        ),
        (
            (1.5, 2.0),
            np.array([-np.pi, -0.5, 0, 1, np.pi], dtype=np.float64),
            (2, 3),
            False,
        ),
        ((10, 1.0), np.array([-np.pi, -0.5, 0, 1, np.pi], dtype=np.float64), (), False),
    ],
)
def test_vonmises_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.vonmises(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, mu, kappa):
        return stats.vonmises.logpdf(obs, kappa, loc=mu)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1, -1.5, 0.0), np.array([0, -0.5, 10, -1], dtype=np.float64), (), True),
        ((1.5, 3.0, 10.5), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
        (
            (1.5, 1.8, 2.0),
            np.array([0, 0.5, 1, 10, -1], dtype=np.float64),
            (2, 3),
            False,
        ),
        ((10, 50, 100.0), np.array([0, 10.1, 80, 103], dtype=np.float64), (), False),
    ],
)
def test_triangular_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.triangular(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, lower, mode, upper):
        return stats.triang.logpdf(
            obs, (mode - lower) / (upper - lower), loc=lower, scale=upper - lower
        )

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1, -1.5), np.array([0, -0.5, 10, -1], dtype=np.float64), (), True),
        ((1.5, 3.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
        ((1.5, 1.8), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (2, 3), False),
        ((10, 50), np.array([0, 10.1, 80, 103], dtype=np.float64), (), False),
    ],
)
def test_gumbel_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.gumbel(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, mu, beta):
        return stats.gumbel_r.logpdf(obs, loc=mu, scale=beta)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1, -1.5), np.array([0, -0.5, 10, -1], dtype=np.float64), (), True),
        ((1.5, 3.0), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (), False),
        ((1.5, 1.8), np.array([0, 0.5, 1, 10, -1], dtype=np.float64), (2, 3), False),
        ((10, 50), np.array([0, 10.1, 80, 103], dtype=np.float64), (), False),
    ],
)
def test_logistic_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.logistic(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, mu, s):
        return stats.logistic.logpdf(obs, loc=mu, scale=s)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((1, 2.0), np.array([0, 1], dtype=np.int64), (), True),
        ((1, 1.0), np.array([0, 1], dtype=np.int64), (), False),
        ((10, 0.0), np.array([0, 1], dtype=np.int64), (), False),
        ((10, 0.5), np.array([0, 1], dtype=np.int64), (3, 2), False),
        (
            (10, np.array([0.0, 0.1, 0.9, 1.0])),
            np.array([0, 1, 4, 10], dtype=np.int64),
            (),
            False,
        ),
    ],
)
def test_binomial_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.binomial(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, n, p):
        return stats.binom.logpmf(obs, n, p)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((1, 1.0, -1.0), np.array([0, 1], dtype=np.int64), (), True),
        ((1, 1.0, 1.0), np.array([0, 1], dtype=np.int64), (), False),
        ((10, 3.0, 2.0), np.array([0, 1], dtype=np.int64), (3, 2), False),
        (
            (10, np.array([0.01, 0.2, 0.9, 1.0]), 2.0),
            np.array([0, 1, 4, 10], dtype=np.int64),
            (),
            False,
        ),
    ],
)
def test_betabinomial_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.betabinom(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, n, alpha, beta):
        return stats.betabinom.logpmf(obs, n, alpha, beta)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1,), np.array([0, 1], dtype=np.int64), (), True),
        ((1.0,), np.array([0, 1], dtype=np.int64), (), False),
        ((0.5,), np.array([0, 1], dtype=np.int64), (3, 2), False),
        ((np.array([0.01, 0.2, 0.9]),), np.array([0, 1, 2], dtype=np.int64), (), False),
    ],
)
def test_bernoulli_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.bernoulli(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, p):
        return stats.bernoulli.logpmf(obs, p)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1,), np.array([0, 1, 100, 10000], dtype=np.int64), (), True),
        ((1.0,), np.array([0, 1, 100, 10000], dtype=np.int64), (), False),
        ((0.5,), np.array([0, 1, 100, 10000], dtype=np.int64), (3, 2), False),
        (
            (np.array([0.01, 0.2, 200]),),
            np.array([-1, 1, 84], dtype=np.int64),
            (),
            False,
        ),
    ],
)
def test_poisson_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.poisson(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, mu):
        return stats.poisson.logpmf(obs, mu)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1,), np.array([-1, 0, 1, 100, 10000], dtype=np.int64), (), True),
        ((1.0,), np.array([-1, 0, 1, 100, 10000], dtype=np.int64), (), False),
        ((0.5,), np.array([-1, 0, 1, 100, 10000], dtype=np.int64), (3, 2), False),
        (
            (np.array([0.01, 0.2, 200]),),
            np.array([-1, 1, 84], dtype=np.int64),
            (),
            False,
        ),
    ],
)
def test_poisson_logcdf(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.poisson(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logcdf(obs, mu):
        return stats.poisson.logcdf(obs, mu)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logcdf, test="logcdf")


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((10, -1), np.array([0, 1, 100, 10000], dtype=np.int64), (), True),
        ((0.1, 0.9), np.array([0, 1, 100, 10000], dtype=np.int64), (), False),
        ((10, 0.5), np.array([0, 1, 100, 10000], dtype=np.int64), (3, 2), False),
        (
            (10, np.array([0.01, 0.2, 0.8])),
            np.array([-1, 1, 84], dtype=np.int64),
            (),
            False,
        ),
        (
            (np.array([2e10, 2, 1], dtype=np.int64), 0.5),
            np.array([-1, 1, 84], dtype=np.int64),
            (),
            False,
        ),
    ],
)
def test_nbinom_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.nbinom(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, n, p):
        return stats.nbinom.logpmf(obs, n, p)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1,), np.array([0, 1, 100, 10000], dtype=np.int64), (), True),
        ((0.1,), np.array([0, 1, 100, 10000], dtype=np.int64), (), False),
        ((1.0,), np.array([0, 1, 100, 10000], dtype=np.int64), (3, 2), False),
        (
            (np.array([0.01, 0.2, 0.8]),),
            np.array([-1, 1, 84], dtype=np.int64),
            (),
            False,
        ),
    ],
)
def test_geometric_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.geometric(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, p):
        return stats.geom.logpmf(obs, p)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1,), np.array([0, 1, 100, 10000], dtype=np.int64), (), True),
        ((0.1,), np.array([0, 1, 100, 10000], dtype=np.int64), (), False),
        ((1.0,), np.array([0, 1, 100, 10000], dtype=np.int64), (3, 2), False),
        (
            (np.array([0.01, 0.2, 0.8]),),
            np.array([-1, 1, 84], dtype=np.int64),
            (),
            False,
        ),
    ],
)
def test_geometric_logcdf(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.geometric(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    with cm:
        scipy_logprob_tester(
            x, obs, dist_params, test_fn=stats.geom.logcdf, test="logcdf"
        )


@pytest.mark.parametrize(
    "dist_params, obs, size",
    [
        ((0.1,), np.array([-0.5, 0, 0.1, 0.5, 0.9, 1.0, 1.5], dtype=np.int64), ()),
        ((0.5,), np.array([-0.5, 0, 0.1, 0.5, 0.9, 1.0, 1.5], dtype=np.int64), (3, 2)),
        (
            (np.array([0.0, 0.2, 0.5, 1.0]),),
            np.array([0.7, 0.7, 0.7, 0.7], dtype=np.int64),
            (),
        ),
    ],
)
def test_geometric_icdf(dist_params, obs, size):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.geometric(*dist_params_at, size=size_at)

    def scipy_geom_icdf(value, p):
        # Scipy ppf returns floats
        return stats.geom.ppf(value, p).astype(value.dtype)

    scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_geom_icdf, test="icdf")


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((-1, 0, 1), np.array([0, 1, 100, 10000], dtype=np.int64), (), True),
        ((1, 0, 1), np.array([0, 1, 100, 10000], dtype=np.int64), (), False),
        ((10, 2, 4), np.array([0, 1, 100, 10000], dtype=np.int64), (3, 2), False),
        (
            (np.array([10, 5, 3], dtype=np.int64), 1, 2),
            np.array([-1, 1, 84], dtype=np.int64),
            (),
            False,
        ),
    ],
)
def test_hypergeometric_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.hypergeometric(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(AssertionError)

    def scipy_logprob(obs, good, bad, n):
        N = n
        M = good + bad
        n = good
        return stats.hypergeom.logpmf(obs, M, n, N)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, exc_type, chk_bcast",
    [
        (
            (np.array([-0.5, 0.5]),),
            np.array([0, 1], dtype=np.int64),
            (),
            ParameterValueError,
            True,
        ),
        (
            (np.array([0.5, 0.5]),),
            np.array([0, 1, 100], dtype=np.int64),
            (),
            IndexError,
            True,
        ),
        (
            (np.array([0.1, 0.9]),),
            np.array([0, 1, 0, 1], dtype=np.int64),
            (),
            False,
            True,
        ),
        (
            (np.array([0.1, 0.9]),),
            np.array([0, 1, 0, 1], dtype=np.int64),
            (1,),
            False,
            True,
        ),
        (
            (np.array([0.1, 0.9]),),
            np.array([0, 1, 0, 1], dtype=np.int64),
            (3, 2),
            False,
            True,
        ),
        (
            (np.array([0.1, 0.9]),),
            np.array([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.int64).T,
            (),
            False,
            True,
        ),
        (
            (np.array([[0.1, 0.9], [0.9, 0.1]]),),
            np.array([[[0], [1], [1], [1]], [[1], [1], [0], [1]]], dtype=np.int64).T,
            (),
            False,
            False,
        ),
        (
            (np.array([[[0.1, 0.9]]]),),
            np.array([0, 1, 1], dtype=np.int64),
            (),
            False,
            False,
        ),
        (
            (np.array([[0.1, 0.9], [0.9, 0.1]]),),
            np.array([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.int64).T,
            (),
            False,
            True,
        ),
        (
            (np.array([[0.1, 0.9], [0.9, 0.1]]),),
            np.array([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.int64).T,
            (3, 2),
            False,
            True,
        ),
    ],
)
def test_categorical_logprob(dist_params, obs, size, exc_type, chk_bcast):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.categorical(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not exc_type else pytest.raises(exc_type)

    with cm:
        scipy_logprob_tester(
            x, obs, dist_params, test_fn=scipy_logprob, check_broadcastable=chk_bcast
        )


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((np.array([-0.5, 0.5]), -1.0 * np.eye(2)), np.array([0, 1]), (), True),
        (
            (np.array([0.5, 0.5]), np.eye(2)),
            np.array([[0.0, 0.0], [1.0, -1.0], [100.0, 200.0]]),
            (),
            False,
        ),
        (
            (np.array([0.5, 0.5]), 10.0 * np.eye(2)),
            np.array([[0.0, 0.0], [1.0, -1.0], [100.0, 200.0]]),
            (3, 2),
            False,
        ),
        pytest.param(
            (np.array([[0.3, 0.7], [0.1, 0.8]]), np.eye(2)[None, ...]),
            np.array([[0.0, 0.0], [1.0, -1.0], [100.0, 200.0]]),
            (),
            False,
            marks=pytest.mark.xfail(
                reason=(
                    "This won't work until the Cholesky is replaced with something "
                    "that can handle more than two dimensions."
                )
            ),
        ),
    ],
)
def test_mvnormal_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.multivariate_normal(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, mu, cov):
        return stats.multivariate_normal.logpdf(obs, mu, cov)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((np.array([-0.5, 0.5]),), np.array([[0.0, 1.0], [1.0, 0.0]]), (), True),
        ((np.array([0.5, 0.5]),), np.array([[0.1, 0.9], [0.5, 0.5]]), (), False),
        ((np.array([0.5, 0.5]),), np.array([[0.1, 0.9], [0.5, 0.5]]), (3, 2), False),
        pytest.param(
            (np.array([[10.0, 5.7], [0.1, 0.8]]),),
            np.array([[0.1, 0.9], [0.5, 0.5]]),
            (),
            False,
            marks=pytest.mark.xfail(
                reason=("SciPy doesn't support parameter broadcasting")
            ),
        ),
    ],
)
def test_dirichlet_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.dirichlet(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, alpha):
        return stats.dirichlet.logpdf(obs.T, alpha)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


@pytest.mark.parametrize(
    "dist_params, obs, size, error",
    [
        ((10, np.array([0.8, 0.9])), np.array([0, 10], dtype=np.int64), (), True),
        ((10, np.array([0.1, 0.9])), np.array([0, 10], dtype=np.int64), (), False),
        ((10, np.array([0.1, 0.9])), np.array([0, 10], dtype=np.int64), (3, 2), False),
        (
            (
                np.array([10, 3], dtype=np.int64),
                np.array([[0.1, 0.9], [0.8, 0.2]]),
            ),
            np.array([[3, 1], [7, 9]], dtype=np.int64),
            (),
            False,
        ),
    ],
)
def test_multinomial_logprob(dist_params, obs, size, error):

    dist_params_at, obs_at, size_at = create_aesara_params(dist_params, obs, size)
    dist_params = dict(zip(dist_params_at, dist_params))

    x = at.random.multinomial(*dist_params_at, size=size_at)

    cm = contextlib.suppress() if not error else pytest.raises(ParameterValueError)

    def scipy_logprob(obs, n, p):
        return stats.multinomial.logpmf(obs, n, p)

    with cm:
        scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)


def test_CheckParameter():
    mu = at.constant(0)
    sigma = at.scalar("sigma")
    x_rv = at.random.normal(mu, sigma, name="x")
    x_vv = at.constant(0)
    x_logp = logprob(x_rv, x_vv)

    x_logp_fn = function([sigma], x_logp)
    with pytest.raises(ParameterValueError, match="sigma > 0"):
        x_logp_fn(-1)


@pytest.mark.parametrize(
    "dist_params, obs",
    [
        ((np.array(0, dtype=np.float64),), np.array([0, 0.5, 1, -1], dtype=np.float64)),
        ((np.array([0, 0], dtype=np.int64),), np.array(0, dtype=np.int64)),
    ],
)
def test_dirac_delta_logprob(dist_params, obs):

    dist_params_at, obs_at, _ = create_aesara_params(dist_params, obs, ())
    dist_params = dict(zip(dist_params_at, dist_params))

    x = dirac_delta(*dist_params_at)

    @np.vectorize
    def scipy_logprob(obs, c):
        return 0.0 if obs == c else -np.inf

    scipy_logprob_tester(x, obs, dist_params, test_fn=scipy_logprob)
