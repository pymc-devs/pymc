#
# Demonstrates the usage of hierarchical partial pooling
# See http://mc-stan.org/documentation/case-studies/pool-binary-trials.html for more details
#

import numpy as np

import pymc3 as pm


def build_model():
    data = np.loadtxt(
        pm.get_data("efron-morris-75-data.tsv"), delimiter="\t", skiprows=1, usecols=(2, 3)
    )

    atbats = pm.floatX(data[:, 0])
    hits = pm.floatX(data[:, 1])

    N = len(hits)

    # we want to bound the kappa below
    BoundedKappa = pm.Bound(pm.Pareto, lower=1.0)

    with pm.Model() as model:
        phi = pm.Uniform("phi", lower=0.0, upper=1.0)
        kappa = BoundedKappa("kappa", alpha=1.0001, m=1.5)
        thetas = pm.Beta("thetas", alpha=phi * kappa, beta=(1.0 - phi) * kappa, shape=N)
        ys = pm.Binomial("ys", n=atbats, p=thetas, observed=hits)
    return model


def run(n=2000):
    model = build_model()
    with model:
        trace = pm.sample(n, target_accept=0.99)

    pm.traceplot(trace)


if __name__ == "__main__":
    run()
