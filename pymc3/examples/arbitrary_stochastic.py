import numpy as np
import pymc3 as pm
import theano.tensor as tt


def build_model():
    # data
    failure = np.array([0., 1.])
    value = np.array([1., 0.])
    
    # custom log-liklihood
    def logp(failure, value):
        return tt.sum(failure * tt.log(lam) - lam * value)
    
    # model
    with pm.Model() as model:
        lam = pm.Exponential('lam', 1.)
        pm.DensityDist('x', logp, observed={'failure': failure, 'value': value})
    return model


def run(n_samples=3000):
    model = build_model()
    with model:
        trace = pm.sample(n_samples)
    return trace

if __name__ == "__main__":
    run()
