"""
Comparing different samplers on a correlated bivariate normal distribution.

"""


import numpy as np
import time
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt


USE_XY = True

def run(steppers, p):
    steppers = set(steppers)
    traces = {}
    effn = {}
    runtimes = {}

    with pm.Model() as model:
        if USE_XY:
            x = pm.Flat('x')
            y = pm.Flat('y')
            mu = np.array([0.,0.])
            cov = np.array([[1.,p],[p,1.]])
            z = pm.MvNormal.dist(mu=mu, cov=cov, shape=(2,)).logp(tt.stack([x,y]))
            p = pm.Potential('logp_xy', z)
            start = {'x': 0, 'y': 0}
        else:
            mu = np.array([0.,0.])
            cov = np.array([[1.,p],[p,1.]])
            z = pm.MvNormal('z', mu=mu, cov=cov, shape=(2,))
            start={'z': [0, 0]}

        for step_cls in steppers:
            name = step_cls.__name__
            print('\r\nStep method: {}'.format(name))
            t_start = time.time()
            mt = pm.sample(
                draws=10000,
                njobs=1,
                chains=6,
                step=step_cls(),
                start=start
            )
            runtimes[name] = time.time() - t_start
            traces[name] = mt
            en = pm.diagnostics.effective_n(mt)
            if USE_XY:
                effn[name] = np.mean(en['x']) / len(mt) / mt.nchains
            else:
                effn[name] = np.mean(en['z']) / len(mt) / mt.nchains
    return traces, effn, runtimes


if __name__ == '__main__':
    methods = [
        pm.Metropolis,
        pm.NUTS,
        pm.DEMetropolis
    ]
    names = [c.__name__ for c in methods]
    df_effectiven = pd.DataFrame(columns=['p'] + names)
    df_effectiven['p'] = [.0,.9]
    df_effectiven = df_effectiven.set_index('p')

    df_runtime = pd.DataFrame(columns=['p'] + names)
    df_runtime['p'] = [.0,.9]
    df_runtime = df_runtime.set_index('p')

    for p in df_effectiven.index:
        trace, rate, runtime = run(methods, p)
        for name in names:
            df_effectiven.set_value(p, name, rate[name])
            df_runtime.set_value(p, name, runtime[name])

    print('\r\nEffective sample size [0...1]')
    print(df_effectiven.T)

    print('\r\nRuntime [s]')
    print(df_runtime.T)
