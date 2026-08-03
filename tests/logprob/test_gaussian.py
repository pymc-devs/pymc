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
import numpy as np
import pytensor.tensor as pt
import pytest

from pymc.distributions import Exponential, MvNormal, Normal
from pymc.logprob.basic import conditional_logp, logp


def assert_logp_equal(y, ref, test_value, subs=None):
    """Total logp of ``y`` matches the reference distribution at ``test_value``."""
    subs = subs or {}
    value = y.type()
    y_logp = logp(y, value)
    ref_logp = logp(ref, value)
    np.testing.assert_allclose(
        y_logp.eval({value: test_value, **subs}).sum(),
        ref_logp.eval({value: test_value, **subs}).sum(),
    )


@pytest.mark.parametrize("D, K", [(4, 2), (5, 1), (3, 3)])
def test_low_rank_guide(D, K):
    """The low-rank ADVI guide ``loc + W @ ek + d * ed`` is recognized as MvNormal."""
    rng = np.random.default_rng(623)
    loc_v = rng.normal(size=D)
    W_v = rng.normal(size=(D, K))
    d_v = np.abs(rng.normal(size=D)) + 0.5

    loc = pt.tensor("loc", shape=(D,))
    W = pt.tensor("W", shape=(D, K))
    d = pt.tensor("d", shape=(D,))
    ek = Normal.dist(0, 1, shape=(K,))
    ed = Normal.dist(0, 1, shape=(D,))
    u = loc + W @ ek + d * ed

    ref = MvNormal.dist(mu=loc, cov=W @ W.T + pt.diag(d**2))
    assert_logp_equal(u, ref, rng.normal(size=D), {loc: loc_v, W: W_v, d: d_v})


def test_sum_of_two_full_normals():
    """``loc + A @ z1 + B @ z2`` -> MvNormal(loc, A Aᵀ + B Bᵀ)."""
    rng = np.random.default_rng(1)
    D = 5
    loc_v, A_v, B_v = rng.normal(size=D), rng.normal(size=(D, D)), rng.normal(size=(D, D))

    loc = pt.tensor("loc", shape=(D,))
    A = pt.tensor("A", shape=(D, D))
    B = pt.tensor("B", shape=(D, D))
    z1 = Normal.dist(0, 1, shape=(D,))
    z2 = Normal.dist(0, 1, shape=(D,))
    u = loc + A @ z1 + B @ z2

    ref = MvNormal.dist(mu=loc, cov=A @ A.T + B @ B.T)
    assert_logp_equal(u, ref, rng.normal(size=D), {loc: loc_v, A: A_v, B: B_v})


def test_sum_of_two_diagonal_normals():
    """Adding two independent diagonal Normals yields the joint (diagonal) MvNormal."""
    rng = np.random.default_rng(2)
    D = 4
    x = Normal.dist(mu=1.0, sigma=2.0, shape=(D,))
    y = Normal.dist(mu=-0.5, sigma=0.7, shape=(D,))
    ref = MvNormal.dist(mu=np.full(D, 0.5), cov=np.eye(D) * (2.0**2 + 0.7**2))
    assert_logp_equal(x + y, ref, rng.normal(size=D))


def test_mvnormal_plus_normal_promotion():
    """A full MvNormal plus an independent diagonal Normal stays MvNormal."""
    rng = np.random.default_rng(3)
    D = 4
    A_v = rng.normal(size=(D, D))
    cov_v = A_v @ A_v.T + np.eye(D)
    mv = MvNormal.dist(mu=np.zeros(D), cov=cov_v)
    ed = Normal.dist(0, 0.5, shape=(D,))
    ref = MvNormal.dist(mu=np.zeros(D), cov=cov_v + np.eye(D) * 0.25)
    assert_logp_equal(mv + ed, ref, rng.normal(size=D))


def test_linear_regression_marginal():
    """Analytic marginal ``y = X @ beta + sigma * eps`` over Gaussian latents."""
    rng = np.random.default_rng(4)
    N, P, sigma = 6, 3, 0.8
    X_v = rng.normal(size=(N, P))
    mu_beta = rng.normal(size=P)
    sd_beta = np.abs(rng.normal(size=P)) + 0.3

    X = pt.tensor("X", shape=(N, P))
    beta = Normal.dist(mu=pt.as_tensor(mu_beta), sigma=pt.as_tensor(sd_beta), shape=(P,))
    eps = Normal.dist(0, 1, shape=(N,))
    y = X @ beta + sigma * eps

    ref_cov = X @ pt.as_tensor(np.diag(sd_beta**2)) @ X.T + sigma**2 * pt.eye(N)
    ref = MvNormal.dist(mu=X @ pt.as_tensor(mu_beta), cov=ref_cov)
    assert_logp_equal(y, ref, rng.normal(size=N), {X: X_v})


def test_batched_low_rank():
    """Leading batch dims are carried through the moment propagation."""
    rng = np.random.default_rng(5)
    Bz, D, K = 4, 5, 2
    loc_v = rng.normal(size=(Bz, D))
    W_v = rng.normal(size=(Bz, D, K))
    d_v = np.abs(rng.normal(size=(Bz, D))) + 0.5

    loc = pt.tensor("loc", shape=(Bz, D))
    W = pt.tensor("W", shape=(Bz, D, K))
    d = pt.tensor("d", shape=(Bz, D))
    ek = Normal.dist(0, 1, shape=(Bz, K))
    ed = Normal.dist(0, 1, shape=(Bz, D))
    u = loc + (W @ ek[..., None])[..., 0] + d * ed

    ref = MvNormal.dist(mu=loc, cov=W @ W.mT + pt.eye(D) * d[..., None, :] ** 2)
    assert_logp_equal(u, ref, rng.normal(size=(Bz, D)), {loc: loc_v, W: W_v, d: d_v})


def test_full_rank_square_still_uses_matmul():
    """A square ``L @ z`` keeps deriving (via MeasurableMatMul), unaffected."""
    rng = np.random.default_rng(6)
    D = 3
    loc_v, L_v = rng.normal(size=D), rng.normal(size=(D, D))

    loc = pt.tensor("loc", shape=(D,))
    L = pt.tensor("L", shape=(D, D))
    z = Normal.dist(0, 1, shape=(D,))
    u = loc + L @ z

    ref = MvNormal.dist(mu=loc, cov=L @ L.T)
    assert_logp_equal(u, ref, rng.normal(size=D), {loc: loc_v, L: L_v})


@pytest.mark.parametrize(
    "build",
    [
        # non-Gaussian leaf in the affine combination
        lambda D, K, loc, W, d, ed: loc + W @ Exponential.dist(1.0, shape=(K,)) + d * ed,
        # non-linear op inside the affine path
        lambda D, K, loc, W, d, ed: loc + W @ Normal.dist(0, 1, shape=(K,)) + pt.exp(ed),
    ],
)
def test_bails_cleanly(build):
    D, K = 5, 2
    loc = pt.tensor("loc", shape=(D,))
    W = pt.tensor("W", shape=(D, K))
    d = pt.tensor("d", shape=(D,))
    ed = Normal.dist(0, 1, shape=(D,))
    y = build(D, K, loc, W, d, ed)
    with pytest.raises((NotImplementedError, RuntimeError)):
        conditional_logp({y: y.type()})


def test_bails_on_correlated_leaves():
    """Sharing a leaf across summands violates independence -> bail."""
    D = 4
    z = Normal.dist(0, 1, shape=(D,))
    with pytest.raises((NotImplementedError, RuntimeError)):
        conditional_logp({z + 2 * z: pt.vector("v", shape=(D,))})
