import numpy as np
import pymc3 as pm
import theano.tensor as tt


def build_model():
    with pm.Model() as model:
        lam = pm.Exponential('lam', 1)
        failure = np.array([0, 1])
        value = np.array([1, 0])

        def logp(failure, value):
            return tt.sum(failure * np.log(lam) - lam * value)
        pm.DensityDist('x', logp, observed={'failure': failure, 'value': value})
    return model


def run(n_samples=3000):
    model = build_model()
    start = model.test_point
    h = pm.find_hessian(start, model=model)
    step = pm.Metropolis(model.vars, h, blocked=True, model=model)
    trace = pm.sample(n_samples, step=step, start=start, model=model)
    return trace

if __name__ == "__main__":
    run()
