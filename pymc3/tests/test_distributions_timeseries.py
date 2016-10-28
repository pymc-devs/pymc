from __future__ import division

from ..model import Model
from ..distributions.continuous import Flat, Normal
from ..distributions.timeseries import EulerMaruyama
from ..tuning.starting import find_MAP
from ..sampling import sample, sample_ppc
from ..step_methods.nuts import NUTS

import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def _gen_sde_path(sde, pars, dt, n, x0):
    xs = [x0]
    wt = np.random.normal(size=(n,) if isinstance(x0, float) else (n, x0.size))
    for i in range(n):
        f, g = sde(xs[-1], *pars)
        xs.append(
            xs[-1] + f * dt + np.sqrt(dt) * g * wt[i]
        )
    return np.array(xs)


def test_linear():
    lam = -0.78
    sig2 = 5e-3
    N = 300
    dt = 1e-1
    sde = lambda x, lam: (lam * x, sig2)
    x = _gen_sde_path(sde, (lam,), dt, N, 5.0)
    z = x + np.random.randn(x.size) * 5e-3
    # build model
    with Model() as model:
        lamh = Flat('lamh')
        xh = EulerMaruyama('xh', dt, sde, (lamh,), shape=N + 1, testval=x)
        zh = Normal('zh', mu=xh, sd=5e-3, observed=z)
    # invert
    with model:
        start = find_MAP(vars=[xh], fmin=fmin_l_bfgs_b)
        warmup = sample(200, NUTS(scaling=start))
        trace = sample(1000, NUTS(scaling=warmup[-1], gamma=0.25), start=warmup[-1])
    ppc = sample_ppc(trace, model=model)
    # test
    p95 = [2.5, 97.5]
    lo, hi = np.percentile(trace[lamh], p95, axis=0)
    assert (lo < lam) and (lam < hi)
    lo, hi = np.percentile(ppc['zh'], p95, axis=0)
    assert ((lo < z) * (z < hi)).mean() > 0.95
