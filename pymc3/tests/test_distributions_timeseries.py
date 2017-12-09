from __future__ import division

from ..model import Model
from ..distributions.continuous import Flat, Normal
from ..distributions.timeseries import EulerMaruyama, AR
from ..sampling import sample, sample_ppc
from ..theanof import floatX

import numpy as np

def test_AR():
    # AR1
    data = np.array([0.3,1,2,3,4])
    phi = np.array([0.99])
    with Model() as t:
        y = AR('y', phi, sd=1, shape=len(data))
        z = Normal('z', mu=phi*data[:-1], sd=1, shape=len(data)-1)
    ar_like = t['y'].logp({'z':data[1:], 'y': data})
    reg_like = t['z'].logp({'z':data[1:], 'y': data})
    np.testing.assert_allclose(ar_like, reg_like)

    # AR1 + constant
    with Model() as t:
        y = AR('y', [0.3, phi], sd=1, shape=len(data), constant=True)
        z = Normal('z', mu=0.3 + phi*data[:-1], sd=1, shape=len(data)-1)
    ar_like = t['y'].logp({'z':data[1:], 'y': data})
    reg_like = t['z'].logp({'z':data[1:], 'y': data})
    np.testing.assert_allclose(ar_like, reg_like)

    # AR2
    phi = np.array([0.84, 0.10])
    with Model() as t:
        y = AR('y', phi, sd=1, shape=len(data))
        z = Normal('z', mu=phi[0]*data[1:-1]+phi[1]*data[:-2], sd=1, shape=len(data)-2)
    ar_like = t['y'].logp({'z':data[2:], 'y': data})
    reg_like = t['z'].logp({'z':data[2:], 'y': data})
    np.testing.assert_allclose(ar_like, reg_like)



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
    x = floatX(_gen_sde_path(sde, (lam,), dt, N, 5.0))
    z = x + np.random.randn(x.size) * sig2
    # build model
    with Model() as model:
        lamh = Flat('lamh')
        xh = EulerMaruyama('xh', dt, sde, (lamh,), shape=N + 1, testval=x)
        Normal('zh', mu=xh, sd=sig2, observed=z)
    # invert
    with model:
        trace = sample(init='advi+adapt_diag', chains=1)

    ppc = sample_ppc(trace, model=model)
    # test
    p95 = [2.5, 97.5]
    lo, hi = np.percentile(trace[lamh], p95, axis=0)
    assert (lo < lam) and (lam < hi)
    lo, hi = np.percentile(ppc['zh'], p95, axis=0)
    assert ((lo < z) * (z < hi)).mean() > 0.95
