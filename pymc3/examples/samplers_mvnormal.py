"""
Comparing different samplers on a correlated bivariate normal distribution.

"""


import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt




def run(steppers, p):
    steppers = set(steppers)
    traces = {}
    effn = {}

    with pm.Model() as model:
        mu = np.array([0.,0.])
        cov = np.array([[1.,p],[p,1.]])
        z = pm.MvNormal('z', mu=mu, cov=cov, shape=(2,))

        for step_cls in steppers:
            name = step_cls.__name__
            print('Step method: {}'.format(name))
            mt = pm.sample(
                draws=10000,
                njobs=1,
                chains=6,
                step=step_cls(),
                start={'z': [0, 0]}
            )
            traces[name] = mt
            en = pm.diagnostics.effective_n(mt)
            effn[name] = np.mean(en['z']) / len(mt) / mt.nchains
    return traces, effn


if __name__ == '__main__':
    methods = [pm.Metropolis, pm.NUTS]
    names = [c.__name__ for c in methods]
    df = pd.DataFrame(columns=['p'] + names)
    df['p'] = [.0,.9]
    df = df.set_index('p')
    rates = []
    for p in df.index:
        trace, rate = run(methods, p)
        for name in names:
            df.set_value(p, name, rate[name])

    print('Effective sample size [0...1]')
    print(df)
