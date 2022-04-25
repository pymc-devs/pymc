import aesara
import aesara.tensor as at
import aesara.tensor.random as atr
import numpy as np
import pytest
import scipy.stats as sp
from aesara.compile.mode import get_default_mode

from aeppl import joint_logprob
from aeppl.dists import dirac_delta, discrete_markov_chain, switching_process
from aeppl.logprob import logprob
from tests.utils import simulate_poiszero_hmm


def poisson_zero_process(mu=None, states=None, srng=None, **kwargs):
    """A Poisson-Dirac-delta (at zero) mixture process.

    The first mixture component (at index 0) is the Dirac-delta at zero, and
    the second mixture component is the Poisson random variable.

    Parameters
    ----------
    mu: tensor
        The Poisson rate(s)
    states: tensor
        A vector of integer 0-1 states that indicate which component of
        the mixture is active at each point/time.
    """
    mu = at.as_tensor_variable(mu)
    states = at.as_tensor_variable(states)

    # NOTE: This creates distributions that are *not* part of a `Model`
    return switching_process(
        [dirac_delta(at.as_tensor(0, dtype=np.int64)), srng.poisson(mu)],
        states,
        **kwargs
    )


def test_dirac_delta():
    fn = aesara.function(
        [], dirac_delta(at.as_tensor(1)), mode=get_default_mode().excluding("useless")
    )
    with pytest.warns(UserWarning, match=".*DiracDelta.*"):
        assert np.array_equal(fn(), 1)


def test_simulate_poiszero_hmm():
    poiszero_sim = simulate_poiszero_hmm(30, 5000, seed=230)

    assert poiszero_sim.keys() == {"P_tt", "S_t", "p_1", "p_0", "Y_t", "pi_0", "S_0"}

    y_test = poiszero_sim["Y_t"].squeeze()
    nonzeros_idx = poiszero_sim["S_t"] > 0

    assert np.all(y_test[nonzeros_idx] > 0)
    assert np.all(y_test[~nonzeros_idx] == 0)


def test_discrete_markov_chain_random_seed():

    test_Gamma_base = np.array([[[0.5, 0.5], [0.5, 0.5]]])
    test_Gamma = np.broadcast_to(test_Gamma_base, (10, 2, 2))
    test_gamma_0 = np.r_[0.5, 0.5]

    dmc, updates = discrete_markov_chain(test_Gamma, test_gamma_0)

    dmc_fn = aesara.function((), dmc, updates=updates)

    test_sample_1 = dmc_fn()

    assert len(set(test_sample_1)) > 1

    test_sample_2 = dmc_fn()

    assert not np.array_equal(test_sample_1, test_sample_2)


def test_discrete_markov_chain_random():
    # A single transition matrix and initial probabilities vector for each
    # element in the state sequence
    test_Gamma_base = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    test_Gamma = np.broadcast_to(test_Gamma_base, (10, 2, 2))
    test_gamma_0 = np.r_[0.0, 1.0]

    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0)
    test_sample = aesara.function((), sample, updates=updates)()
    assert np.all(test_sample == 1)

    sample, updates = discrete_markov_chain(test_Gamma, 1.0 - test_gamma_0, size=10)
    test_sample = aesara.function((), sample, updates=updates)()
    assert np.all(test_sample == 0)

    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0, size=12)
    test_sample = aesara.function((), sample, updates=updates)()
    assert test_sample.shape == (12, 10)

    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0, size=2)
    test_sample = aesara.function((), sample, updates=updates)()
    assert np.array_equal(
        test_sample, np.stack([np.ones(10), np.ones(10)], 0).astype(int)
    )

    # Now, the same set-up, but--this time--generate two state sequences
    # samples
    test_Gamma_base = np.array([[[0.8, 0.2], [0.2, 0.8]]])
    test_Gamma = np.broadcast_to(test_Gamma_base, (10, 2, 2))
    test_gamma_0 = np.r_[0.2, 0.8]
    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0, size=2)
    test_sample = aesara.function((), sample, updates=updates)()
    # TODO: Fix the seed, and make sure there's at least one 0 and 1?
    assert test_sample.shape == (2, 10)

    # Two transition matrices--for two distinct state sequences--and one vector
    # of initial probs.
    test_Gamma_base = np.stack(
        [np.array([[[1.0, 0.0], [0.0, 1.0]]]), np.array([[[1.0, 0.0], [0.0, 1.0]]])]
    )
    test_Gamma = np.broadcast_to(test_Gamma_base, (2, 10, 2, 2))
    test_gamma_0 = np.r_[0.0, 1.0]

    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0)
    test_sample = aesara.function((), sample, updates=updates)()
    assert np.array_equal(
        test_sample, np.stack([np.ones(10), np.ones(10)], 0).astype(int)
    )
    assert test_sample.shape == (2, 10)

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0, size=3)
    test_sample = aesara.function((), sample, updates=updates)()
    assert np.array_equal(
        test_sample,
        np.tile(np.stack([np.ones(10), np.ones(10)], 0).astype(int), (3, 1, 1)),
    )
    assert test_sample.shape == (3, 2, 10)

    # Two transition matrices and initial probs. for two distinct state
    # sequences
    test_Gamma_base = np.stack(
        [np.array([[[1.0, 0.0], [0.0, 1.0]]]), np.array([[[1.0, 0.0], [0.0, 1.0]]])]
    )
    test_Gamma = np.broadcast_to(test_Gamma_base, (2, 10, 2, 2))
    test_gamma_0 = np.stack([np.r_[0.0, 1.0], np.r_[1.0, 0.0]])
    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0)
    test_sample = aesara.function((), sample, updates=updates)()
    assert np.array_equal(
        test_sample, np.stack([np.ones(10), np.zeros(10)], 0).astype(int)
    )
    assert test_sample.shape == (2, 10)

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0, size=3)
    test_sample = aesara.function((), sample, updates=updates)()
    assert np.array_equal(
        test_sample,
        np.tile(np.stack([np.ones(10), np.zeros(10)], 0).astype(int), (3, 1, 1)),
    )
    assert test_sample.shape == (3, 2, 10)

    # "Time"-varying transition matrices with a single vector of initial
    # probabilities
    test_Gamma = np.stack(
        [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        ],
        axis=0,
    )
    test_gamma_0 = np.r_[1, 0]

    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0)
    test_sample = aesara.function((), sample, updates=updates)()
    assert np.array_equal(test_sample, np.r_[1, 0, 0])

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0, size=3)
    test_sample = aesara.function((), sample, updates=updates)()
    assert np.array_equal(test_sample, np.tile(np.r_[1, 0, 0].astype(int), (3, 1)))

    # "Time"-varying transition matrices with two initial
    # probabilities vectors
    test_Gamma = np.stack(
        [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        ],
        axis=0,
    )
    test_gamma_0 = np.array([[1, 0], [0, 1]])

    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0)
    test_sample = aesara.function((), sample, updates=updates)()
    assert np.array_equal(test_sample, np.array([[1, 0, 0], [0, 1, 1]]))

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0, size=3)
    test_sample = aesara.function((), sample, updates=updates)()
    assert np.array_equal(
        test_sample, np.tile(np.array([[1, 0, 0], [0, 1, 1]]).astype(int), (3, 1, 1))
    )

    # Two "Time"-varying transition matrices with two initial
    # probabilities vectors
    test_Gamma = np.stack(
        [
            [
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.array([[1.0, 0.0], [0.0, 1.0]]),
            ],
            [
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                np.array([[0.0, 1.0], [1.0, 0.0]]),
            ],
        ],
        axis=0,
    )
    test_gamma_0 = np.array([[1, 0], [0, 1]])

    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0)
    test_sample = aesara.function((), sample, updates=updates)()
    assert np.array_equal(test_sample, np.array([[1, 0, 0], [1, 1, 0]]))

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    sample, updates = discrete_markov_chain(test_Gamma, test_gamma_0, size=3)
    test_sample = aesara.function((), sample, updates=updates)()
    assert np.array_equal(
        test_sample, np.tile(np.array([[1, 0, 0], [1, 1, 0]]).astype(int), (3, 1, 1))
    )


@pytest.mark.parametrize(
    "Gammas, gamma_0, obs, exp_res",
    [
        pytest.param(
            np.array([[[0.0, 1.0], [1.0, 0.0]]]),
            np.r_[1.0, 0.0],
            np.r_[1, 0, 1, 0],
            # 0
            None,
            id=(
                "A single transition matrix and initial probabilities vector for each "
                "element in the state sequence"
            ),
        ),
        pytest.param(
            np.stack(
                [
                    np.array([[0.0, 1.0], [1.0, 0.0]]),
                    np.array([[0.0, 1.0], [1.0, 0.0]]),
                    np.array([[0.0, 1.0], [1.0, 0.0]]),
                    np.array([[0.0, 1.0], [1.0, 0.0]]),
                ],
                axis=0,
            ),
            np.r_[1.0, 0.0],
            np.r_[1, 0, 1, 0],
            # 0,
            None,
            id='"Time"-varying transition matrices with a single vector of initial probabilities',
        ),
        pytest.param(
            np.stack(
                [
                    np.array([[0.1, 0.9], [0.5, 0.5]]),
                    np.array([[0.2, 0.8], [0.6, 0.4]]),
                    np.array([[0.3, 0.7], [0.7, 0.3]]),
                    np.array([[0.4, 0.6], [0.8, 0.2]]),
                ],
                axis=0,
            ),
            np.r_[0.3, 0.7],
            np.r_[1, 0, 1, 0],
            None,
            id=(
                '"Time"-varying transition matrices with a single vector of initial '
                "probabilities, but--this time--with better test values"
            ),
        ),
        pytest.param(
            np.array([[[0.0, 1.0], [1.0, 0.0]]]),
            np.r_[0.5, 0.5],
            np.array([[1, 0, 1, 0], [0, 1, 0, 1]]),
            # np.array([1, 1, 1, 1]),
            None,
            marks=pytest.mark.xfail(
                reason=("Broadcasting for logp not supported"),
                raises=NotImplementedError,
            ),
            id="Static transition matrix and two state sequences",
        ),
        pytest.param(
            np.stack(
                [
                    np.array([[0.0, 1.0], [1.0, 0.0]]),
                    np.array([[0.0, 1.0], [1.0, 0.0]]),
                    np.array([[0.0, 1.0], [1.0, 0.0]]),
                    np.array([[0.0, 1.0], [1.0, 0.0]]),
                ],
                axis=0,
            ),
            np.r_[0.5, 0.5],
            np.array([[1, 0, 1, 0], [0, 1, 0, 1]]),
            # np.array([1, 1, 1, 1]),
            None,
            marks=pytest.mark.xfail(
                reason=("Broadcasting for logp not supported"),
                raises=NotImplementedError,
            ),
            id="Time-varying transition matrices and two state sequences",
        ),
        pytest.param(
            np.stack(
                [
                    [
                        np.array([[0.0, 1.0], [1.0, 0.0]]),
                        np.array([[0.0, 1.0], [1.0, 0.0]]),
                        np.array([[0.0, 1.0], [1.0, 0.0]]),
                        np.array([[0.0, 1.0], [1.0, 0.0]]),
                    ],
                    [
                        np.array([[1.0, 0.0], [0.0, 1.0]]),
                        np.array([[1.0, 0.0], [0.0, 1.0]]),
                        np.array([[1.0, 0.0], [0.0, 1.0]]),
                        np.array([[1.0, 0.0], [0.0, 1.0]]),
                    ],
                ],
                axis=0,
            ),
            np.r_[0.5, 0.5],
            np.array([[1, 0, 1, 0], [0, 0, 0, 0]]),
            # np.array([1, 1, 1, 1]),
            None,
            marks=pytest.mark.xfail(
                reason=("Broadcasting for logp not supported"),
                raises=NotImplementedError,
            ),
            id="Two sets of time-varying transition matrices and two state sequences",
        ),
        pytest.param(
            np.stack(
                [
                    [
                        np.array([[0.0, 1.0], [1.0, 0.0]]),
                        np.array([[0.0, 1.0], [1.0, 0.0]]),
                        np.array([[0.0, 1.0], [1.0, 0.0]]),
                        np.array([[0.0, 1.0], [1.0, 0.0]]),
                    ],
                    [
                        np.array([[1.0, 0.0], [0.0, 1.0]]),
                        np.array([[1.0, 0.0], [0.0, 1.0]]),
                        np.array([[1.0, 0.0], [0.0, 1.0]]),
                        np.array([[1.0, 0.0], [0.0, 1.0]]),
                    ],
                ],
                axis=0,
            ),
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([[1, 0, 1, 0], [0, 0, 0, 0]]),
            # np.array([1, 1, 1, 1]),
            None,
            marks=pytest.mark.xfail(
                reason=("Broadcasting for logp not supported"),
                raises=NotImplementedError,
            ),
            id=(
                "Two sets of time-varying transition matrices--via `gamma_0` "
                "broadcasting--and two state sequences"
            ),
        ),
    ],
)
def test_discrete_Markov_chain_logp(Gammas, gamma_0, obs, exp_res):
    test_dist, _ = discrete_markov_chain(Gammas, gamma_0)

    test_logp_at = logprob(test_dist, at.as_tensor(obs))
    test_logp_val = test_logp_at.eval()

    if exp_res is None:

        def logp_single_chain(Gammas, gamma_0, obs):
            state_transitions = np.stack([obs[:-1], obs[1:]]).T

            p_S_0_to_1 = gamma_0.dot(Gammas[0])

            p_S_obs = np.empty_like(obs, dtype=np.float64)
            p_S_obs[0] = p_S_0_to_1[obs[0]]

            for t, (S_tm1, S_t) in enumerate(state_transitions):
                p_S_obs[t + 1] = Gammas[t, S_tm1, S_t]

            return np.log(p_S_obs)

        logp_fn = np.vectorize(logp_single_chain, signature="(n,m,m),(m),(n)->(n)")

        Gammas = np.broadcast_to(Gammas, (obs.shape[0],) + Gammas.shape[-2:])
        exp_res = logp_fn(Gammas, gamma_0, obs)

    assert np.allclose(test_logp_val, exp_res)


def test_switching_process_random():
    test_states = np.r_[0, 0, 1, 1, 0, 1]
    mu_zero_nonzero = [at.as_tensor(0), at.as_tensor(1)]
    test_dist = switching_process(mu_zero_nonzero, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)
    test_sample = test_dist.eval()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(test_sample[test_states > 0] > 0)

    test_states_sized = np.broadcast_to(test_states, (5,) + test_states.shape)
    test_sample = switching_process(mu_zero_nonzero, test_states_sized).eval()
    assert np.array_equal(test_sample.shape, (5,) + test_states.shape)
    assert np.all(test_sample[..., test_states > 0] > 0)

    test_states = at.lvector("states")
    test_states.tag.test_value = np.r_[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    test_dist = switching_process(mu_zero_nonzero, test_states)
    assert np.array_equal(
        test_dist.shape.eval({test_states: test_states.tag.test_value}),
        test_states.tag.test_value.shape,
    )
    test_states_sized = at.broadcast_to(test_states, (1,) + tuple(test_states.shape))
    test_sample = switching_process(mu_zero_nonzero, test_states_sized).eval(
        {test_states: test_states.tag.test_value}
    )
    assert np.array_equal(test_sample.shape, (1,) + test_states.tag.test_value.shape)
    assert np.all(test_sample[..., test_states.tag.test_value > 0] > 0)

    test_states = np.r_[0, 0, 1, 1, 0, 1]
    test_mus = [at.as_tensor(i) for i in range(6)]
    test_dist = switching_process(test_mus, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)
    test_sample = test_dist.eval()
    assert np.array_equal(test_sample.shape, test_states.shape)
    assert np.all(test_sample[..., test_states > 0] > 0)

    test_states = np.c_[0, 0, 1, 1, 0, 1].T
    test_mus = np.arange(1, 6).astype(np.float64)
    # One of the states has emissions that are a sequence of five Dirac delta
    # distributions on the values 1 to 5 (i.e. the one with values
    # `test_mus`), and the other is just a single delta at 0.  A single state
    # sample from this emissions mixture is a length five array of zeros or the
    # values 1 to 5.
    # Instead of specifying a state sequence containing only one state, we use
    # six state sequences--each of length one.  This should give us six samples
    # of either five zeros or the values 1 to 5.
    test_dist = switching_process(
        [at.as_tensor(0), at.as_tensor(test_mus)], test_states
    )
    assert np.array_equal(test_dist.shape.eval(), (6, 5))
    test_sample = test_dist.eval()
    assert np.array_equal(test_sample.shape, test_dist.shape.eval())
    sample_mus = test_sample[np.where(test_states > 0)[0]]
    assert np.all(sample_mus == test_mus)

    test_states = np.c_[0, 0, 1, 1, 0, 1]
    test_mus = np.arange(1, 7).astype(np.float64)
    test_dist = switching_process(
        [at.as_tensor(0), at.as_tensor(test_mus)], test_states
    )
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)

    test_states = np.r_[0, 0, 1, 1, 0, 1]
    test_states_sized = np.broadcast_to(test_states, (3,) + test_states.shape)
    test_sample = switching_process(
        [at.as_tensor(0), at.as_tensor(test_mus)], test_states_sized
    ).eval()
    assert np.array_equal(test_sample.shape, (3,) + test_mus.shape)
    assert np.all(test_sample.sum(0)[..., test_states > 0] > 0)

    # Some misc. tests
    rng = aesara.shared(np.random.RandomState(2023532), borrow=True)

    test_states = np.r_[2, 0, 1, 2, 0, 1]
    test_dists = [
        at.as_tensor(0),
        atr.poisson(100.0, rng=rng),
        atr.poisson(1000.0, rng=rng),
    ]
    test_dist = switching_process(test_dists, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)

    test_sample = test_dist.eval()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(test_sample[test_states == 0] == 0)
    assert np.all(0 < test_sample[test_states == 1])
    assert np.all(test_sample[test_states == 1] < 1000)
    assert np.all(100 < test_sample[test_states == 2])

    test_states = np.r_[2, 0, 1, 2, 0, 1]
    test_mus = np.r_[100, 100, 500, 100, 100, 100]
    test_dists = [
        at.as_tensor(0),
        atr.poisson(test_mus, rng=rng),
        atr.poisson(10000.0, rng=rng),
    ]
    test_dist = switching_process(test_dists, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)

    test_sample = test_dist.eval()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(200 < test_sample[2] < 600)
    assert np.all(0 < test_sample[5] < 200)
    assert np.all(5000 < test_sample[test_states == 2])

    # Try a continuous mixture
    test_states = np.r_[2, 0, 1, 2, 0, 1]
    test_dists = [
        atr.normal(0.0, 1.0, rng=rng),
        atr.normal(100.0, 1.0, rng=rng),
        atr.normal(1000.0, 1.0, rng=rng),
    ]
    test_dist = switching_process(test_dists, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)

    test_sample = test_dist.eval()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(test_sample[test_states == 0] < 10)
    assert np.all(50 < test_sample[test_states == 1])
    assert np.all(test_sample[test_states == 1] < 150)
    assert np.all(900 < test_sample[test_states == 2])

    # Make sure we can use a large number of distributions in the mixture
    test_states = np.ones(50, dtype=np.int64)
    test_dists = [at.as_tensor(i) for i in range(50)]
    test_dist = switching_process(test_dists, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)

    test_states = np.r_[2, 0, 1, 2, 0, 1]
    test_dists = [atr.poisson(0.0), atr.poisson(100.0), atr.poisson(1000.0)]
    test_dist = switching_process(test_dists, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)

    test_sample = test_dist.eval()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(test_sample[test_states == 0] == 0)
    assert np.all(0 < test_sample[test_states == 1])
    assert np.all(test_sample[test_states == 1] < 1000)
    assert np.all(100 < test_sample[test_states == 2])

    test_mus = np.r_[100, 100, 500, 100, 100, 100]
    test_dists = [
        dirac_delta(at.as_tensor(0)),
        atr.poisson(test_mus),
        atr.poisson(10000.0),
    ]
    test_dist = switching_process(test_dists, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)

    test_sample = test_dist.eval()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(200 < test_sample[2] < 600)
    assert np.all(0 < test_sample[5] < 200)
    assert np.all(5000 < test_sample[test_states == 2])


def test_switching_process_logp():

    srng = atr.RandomStream(2023532)

    states_rv = srng.categorical([1 / 3, 1 / 3, 1 / 3], size=6, name="states")
    states_vv = states_rv.clone()
    states_vv.name = "states_vv"

    states_vals = np.r_[2, 0, 1, 2, 0, 1]
    states_vv.tag.test_value = states_vals

    test_dists = [
        dirac_delta(at.as_tensor(0, dtype=np.int64)),
        srng.poisson(100),
        srng.poisson(1000),
    ]
    test_dist = switching_process(test_dists, states_rv)
    sw_vv = test_dist.clone()
    sw_vv.name = "sw_vv"

    test_logp = joint_logprob({test_dist: sw_vv, states_rv: states_vv}, sum=False)
    obs_vals = np.array([1000, 0, 100, 1000, 0, 100], dtype=np.int64)
    test_logp_val = test_logp.eval({sw_vv: obs_vals, states_vv: states_vals})

    np.testing.assert_array_almost_equal(
        test_logp_val[states_vals == 0], np.array([np.log(1 / 3)] * 2)
    )
    np.testing.assert_array_almost_equal(
        test_logp_val[states_vals == 1],
        np.array([np.log(1 / 3) + sp.poisson(100).logpmf(100)] * 2),
        decimal=5,
    )
    np.testing.assert_array_almost_equal(
        test_logp_val[states_vals == 2],
        np.array([np.log(1 / 3) + sp.poisson(1000).logpmf(1000)] * 2),
        decimal=4,
    )

    # Evaluate multiple observed state sequences in an extreme case
    states_rv = srng.categorical([1 / 2, 1 / 2], size=(10, 4), name="states")
    states_vv = states_rv.clone()
    states_vv.name = "states_vv"

    test_dist = switching_process(
        [
            dirac_delta(at.as_tensor(0, dtype=np.int64)),
            dirac_delta(at.as_tensor(1, dtype=np.int64)),
        ],
        states_rv,
    )
    test_dist.name = "switching_dist"

    test_obs = at.tile(np.arange(4), (10, 1)).astype(np.int64)

    test_logp = joint_logprob({test_dist: test_obs, states_rv: states_vv}, sum=False)

    exp_logp = np.tile(
        np.array([np.log(0.5)] + [-np.inf] * 3, dtype=aesara.config.floatX), (10, 1)
    )
    test_logp_val = test_logp.eval({states_vv: np.zeros((10, 4)).astype(np.int64)})
    assert np.array_equal(test_logp_val, exp_logp)


def test_poisson_zero_process_model():
    srng = atr.RandomStream(seed=2023532)

    test_mean = at.repeat(at.as_tensor(1000.0), 20)
    states = srng.bernoulli(0.5, size=20, name="states")

    Y = poisson_zero_process(test_mean, states, srng=srng)

    # We want to make sure that the sampled states and observations correspond,
    # because, if there are any zero states with non-zero observations, we know
    # that the sampled states weren't actually used to draw the observations,
    # and that's a big problem
    sample_fn = aesara.function([], [states, Y])

    fgraph = sample_fn.maker.fgraph
    nodes = list(fgraph.apply_nodes)
    bernoulli_nodes = set(
        n for n in nodes if isinstance(n.op, type(at.random.bernoulli))
    )
    assert len(bernoulli_nodes) == 1

    for i in range(100):
        test_states, test_Y = sample_fn()
        assert np.all(0 < test_Y[..., test_states > 0])
        assert np.all(test_Y[..., test_states > 0] < 10000)
        # Make sure we're sampling different values each time
        unique_nonzero_Y = set(y for y in test_Y if y > 0)
        assert len(unique_nonzero_Y) > 1
