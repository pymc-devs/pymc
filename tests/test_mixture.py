import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy.stats.distributions as sp
from aesara.graph.basic import Variable

from aeppl.joint_logprob import joint_logprob
from tests.utils import assert_no_rvs


def test_mixture_basics():
    srng = at.random.RandomStream(29833)

    def create_mix_model(size, axis):
        X_rv = srng.normal(0, 1, size=size, name="X")
        Y_rv = srng.gamma(0.5, 0.5, size=size, name="Y")

        p_at = at.scalar("p")
        p_at.tag.test_value = 0.5

        I_rv = srng.bernoulli(p_at, size=size, name="I")
        i_vv = I_rv.clone()
        i_vv.name = "i"

        if isinstance(axis, Variable):
            M_rv = at.join(axis, X_rv, Y_rv)[I_rv]
        else:
            M_rv = at.stack([X_rv, Y_rv], axis=axis)[I_rv]

        M_rv.name = "M"
        m_vv = M_rv.clone()
        m_vv.name = "m"

        return locals()

    with pytest.raises(ValueError, match=".*value variable was specified.*"):
        env = create_mix_model(None, 0)
        X_rv = env["X_rv"]
        I_rv = env["I_rv"]
        i_vv = env["i_vv"]
        M_rv = env["M_rv"]
        m_vv = env["m_vv"]

        x_vv = X_rv.clone()
        x_vv.name = "x"

        joint_logprob(M_rv, {M_rv: m_vv, I_rv: i_vv, X_rv: x_vv})

    with pytest.raises(NotImplementedError):
        env = create_mix_model((2,), 1)
        I_rv = env["I_rv"]
        i_vv = env["i_vv"]
        M_rv = env["M_rv"]
        m_vv = env["m_vv"]
        joint_logprob(M_rv, {M_rv: m_vv, I_rv: i_vv})

    with pytest.raises(NotImplementedError):
        axis_at = at.lscalar("axis")
        axis_at.tag.test_value = 0
        env = create_mix_model((2,), axis_at)
        I_rv = env["I_rv"]
        i_vv = env["i_vv"]
        M_rv = env["M_rv"]
        m_vv = env["m_vv"]
        joint_logprob(M_rv, {M_rv: m_vv, I_rv: i_vv})


@pytest.mark.parametrize(
    "p_val, size",
    [
        (np.array(0.0, dtype=aesara.config.floatX), ()),
        (np.array(1.0, dtype=aesara.config.floatX), ()),
        (np.array(0.0, dtype=aesara.config.floatX), (2,)),
        (np.array(1.0, dtype=aesara.config.floatX), (2, 1)),
    ],
)
@aesara.config.change_flags(compute_test_value="raise")
def test_hetero_mixture_scalar(p_val, size):
    srng = at.random.RandomStream(29833)

    X_rv = srng.normal(0, 1, size=size, name="X")
    Y_rv = srng.gamma(0.5, 0.5, size=size, name="Y")

    p_at = at.scalar("p")
    p_at.tag.test_value = p_val

    I_rv = srng.bernoulli(p_at, size=size, name="I")
    i_vv = I_rv.clone()
    i_vv.name = "i"

    M_rv = at.stack([X_rv, Y_rv])[I_rv]
    M_rv.name = "M"
    m_vv = M_rv.clone()
    m_vv.name = "m"

    M_logp = joint_logprob(M_rv, {M_rv: m_vv, I_rv: i_vv})

    M_logp_fn = aesara.function([p_at, m_vv, i_vv], M_logp)

    # The compiled graph should not contain any `RandomVariables`
    assert_no_rvs(M_logp_fn.maker.fgraph.outputs[0])

    decimals = 6 if aesara.config.floatX == "float64" else 4

    test_val_rng = np.random.RandomState(3238)

    bern_sp = sp.bernoulli(p_val)
    norm_sp = sp.norm(loc=0, scale=1)
    gamma_sp = sp.gamma(0.5, scale=1.0 / 0.5)

    for i in range(10):
        i_val = bern_sp.rvs(size=size, random_state=test_val_rng)
        x_val = norm_sp.rvs(size=size, random_state=test_val_rng)
        y_val = gamma_sp.rvs(size=size, random_state=test_val_rng)

        m_val = np.stack([x_val, y_val])[i_val]

        exp_obs_logps = np.stack([norm_sp.logpdf(x_val), gamma_sp.logpdf(y_val)])[
            i_val
        ].sum()
        exp_obs_logps += bern_sp.logpmf(i_val).sum()

        logp_vals = M_logp_fn(p_val, m_val, i_val)

        np.testing.assert_almost_equal(logp_vals, exp_obs_logps, decimal=decimals)


@pytest.mark.parametrize(
    "p_val, size",
    [
        # (np.array(0.0, dtype=aesara.config.floatX), ()),
        # (np.array(1.0, dtype=aesara.config.floatX), ()),
        # (np.array(0.0, dtype=aesara.config.floatX), (2,)),
        # (np.array(1.0, dtype=aesara.config.floatX), (2, 1)),
        # (np.array(1.0, dtype=aesara.config.floatX), (2, 3)),
        (np.array([0.1, 0.9], dtype=aesara.config.floatX), (2, 3)),
    ],
)
def test_hetero_mixture_nonscalar(p_val, size):
    srng = at.random.RandomStream(29833)

    X_rv = srng.normal(0, 1, size=size, name="X")
    Y_rv = srng.gamma(0.5, 0.5, size=size, name="Y")

    if np.ndim(p_val) == 0:
        p_at = at.scalar("p")
        p_at.tag.test_value = p_val

        I_rv = srng.bernoulli(p_at, size=size, name="I")
    else:
        p_at = at.vector("p")
        p_at.tag.test_value = np.array(p_val, dtype=aesara.config.floatX)
        I_rv = srng.categorical(p_at, size=size, name="I")
        p_val_1 = p_val[1]

    i_vv = I_rv.clone()
    i_vv.name = "i"

    M_rv = at.stack([X_rv, Y_rv])[I_rv]
    M_rv.name = "M"

    m_vv = M_rv.clone()
    m_vv.name = "m"

    M_logp = joint_logprob(M_rv, {M_rv: m_vv, I_rv: i_vv})

    M_logp_fn = aesara.function([p_at, m_vv, i_vv], M_logp)

    assert_no_rvs(M_logp_fn.maker.fgraph.outputs[0])

    decimals = 6 if aesara.config.floatX == "float64" else 4

    test_val_rng = np.random.RandomState(3238)

    bern_sp = sp.bernoulli(p_val_1)
    norm_sp = sp.norm(loc=0, scale=1)
    gamma_sp = sp.gamma(0.5, scale=1.0 / 0.5)

    for i in range(10):
        i_val = bern_sp.rvs(size=size, random_state=test_val_rng)
        x_val = norm_sp.rvs(size=size, random_state=test_val_rng)
        y_val = gamma_sp.rvs(size=size, random_state=test_val_rng)

        exp_obs_logps = np.stack([norm_sp.logpdf(x_val), gamma_sp.logpdf(y_val)])[
            i_val
        ].sum()
        exp_obs_logps += bern_sp.logpmf(i_val).sum()

        m_val = np.stack([x_val, y_val])[i_val]
        logp_vals = M_logp_fn(p_val, m_val, i_val)

        np.testing.assert_almost_equal(logp_vals, exp_obs_logps, decimal=decimals)
