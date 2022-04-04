import numpy as np
from aesara.graph.basic import walk
from aesara.graph.op import HasInnerGraph

from aeppl.abstract import MeasurableVariable


def assert_no_rvs(var):
    """Assert that there are no `MeasurableVariable` nodes in a graph."""

    def expand(r):
        owner = r.owner
        if owner:
            inputs = list(reversed(owner.inputs))

            if isinstance(owner.op, HasInnerGraph):
                inputs += owner.op.inner_outputs

            return inputs

    for v in walk([var], expand, False):
        if v.owner and isinstance(v.owner.op, MeasurableVariable):
            raise AssertionError(f"Variable {v} is a MeasurableVariable")


def simulate_poiszero_hmm(
    N, mu=10.0, pi_0_a=np.r_[1, 1], p_0_a=np.r_[5, 1], p_1_a=np.r_[1, 1], seed=None
):
    rng = np.random.default_rng(seed)

    p_0 = rng.dirichlet(p_0_a)
    p_1 = rng.dirichlet(p_1_a)

    Gammas = np.stack([p_0, p_1])
    Gammas = np.broadcast_to(Gammas, (N,) + Gammas.shape)

    pi_0 = rng.dirichlet(pi_0_a)
    s_0 = rng.choice(pi_0.shape[0], p=pi_0)
    s_tm1 = s_0

    y_samples = np.empty((N,), dtype=np.int64)
    s_samples = np.empty((N,), dtype=np.int64)

    for i in range(N):
        s_t = rng.choice(Gammas.shape[-1], p=Gammas[i, s_tm1])
        s_samples[i] = s_t
        s_tm1 = s_t

        if s_t == 1:
            y_samples[i] = rng.poisson(mu)
        else:
            y_samples[i] = 0

    sample_point = {
        "Y_t": y_samples,
        "p_0": p_0,
        "p_1": p_1,
        "S_t": s_samples,
        "P_tt": Gammas,
        "S_0": s_0,
        "pi_0": pi_0,
    }

    return sample_point
