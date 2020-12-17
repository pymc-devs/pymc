"""
Comparing different samplers on a correlated bivariate normal distribution.

This example will sample a bivariate normal with Metropolis, NUTS and DEMetropolis
at two correlations (0, 0.9) and print out the effective sample sizes, runtime and
normalized effective sampling rates.
"""


import time

import numpy as np
import pandas as pd
import theano.tensor as tt

import pymc3 as pm

# with this flag one can switch between defining the bivariate normal as
# either a 2D MvNormal (USE_XY = False) split up the two dimensions into
# two variables 'x' and 'y'.  The latter is recommended because it highlights
# different behaviour with respect to blocking.
USE_XY = True


def run(steppers, p):
    steppers = set(steppers)
    traces = {}
    effn = {}
    runtimes = {}

    with pm.Model() as model:
        if USE_XY:
            x = pm.Flat("x")
            y = pm.Flat("y")
            mu = np.array([0.0, 0.0])
            cov = np.array([[1.0, p], [p, 1.0]])
            z = pm.MvNormal.dist(mu=mu, cov=cov, shape=(2,)).logp(tt.stack([x, y]))
            pot = pm.Potential("logp_xy", z)
            start = {"x": 0, "y": 0}
        else:
            mu = np.array([0.0, 0.0])
            cov = np.array([[1.0, p], [p, 1.0]])
            z = pm.MvNormal("z", mu=mu, cov=cov, shape=(2,))
            start = {"z": [0, 0]}

        for step_cls in steppers:
            name = step_cls.__name__
            t_start = time.time()
            mt = pm.sample(draws=10000, chains=16, parallelize=False, step=step_cls(), start=start)
            runtimes[name] = time.time() - t_start
            print("{} samples across {} chains".format(len(mt) * mt.nchains, mt.nchains))
            traces[name] = mt
            en = pm.ess(mt)
            print(f"effective: {en}\r\n")
            if USE_XY:
                effn[name] = np.mean(en["x"]) / len(mt) / mt.nchains
            else:
                effn[name] = np.mean(en["z"]) / len(mt) / mt.nchains
    return traces, effn, runtimes


if __name__ == "__main__":
    methods = [pm.Metropolis, pm.Slice, pm.NUTS, pm.DEMetropolis]
    names = [c.__name__ for c in methods]

    df_base = pd.DataFrame(columns=["p"] + names)
    df_base["p"] = [0.0, 0.9]
    df_base = df_base.set_index("p")

    df_effectiven = df_base.copy()
    df_runtime = df_base.copy()
    df_performance = df_base.copy()

    for p in df_effectiven.index:
        trace, rate, runtime = run(methods, p)
        for name in names:
            df_effectiven.set_value(p, name, rate[name])
            df_runtime.set_value(p, name, runtime[name])
            df_performance.set_value(p, name, rate[name] / runtime[name])

    print("\r\nEffective sample size [0...1]")
    print(df_effectiven.T.to_string(float_format="{:.3f}".format))

    print("\r\nRuntime [s]")
    print(df_runtime.T.to_string(float_format="{:.1f}".format))

    if "NUTS" in names:
        print("\r\nNormalized effective sampling rate [0...1]")
        df_performance = df_performance.T / df_performance.loc[0]["NUTS"]
    else:
        print("\r\nNormalized effective sampling rate [1/s]")
        df_performance = df_performance.T
    print(df_performance.to_string(float_format="{:.3f}".format))
