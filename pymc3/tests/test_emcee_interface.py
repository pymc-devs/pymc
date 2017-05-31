import pymc3 as pm
import numpy as np
import pytest

def setup_model(yshape):
    yparams = np.linspace(0, 1, yshape)

    with pm.Model() as model:
        x = pm.Normal('x', 0, 1)
        y = pm.Normal('y', yparams, yparams, shape=yshape)
    return model

def correct_instatiation(ndraws, nparticles, step, yshape):
    with setup_model(yshape) as model:
        from pymc3.external.emcee_samplers import sample
        trace = sample(ndraws, step=step, nparticles=nparticles)
    return model, trace

def test_correct_instantiation():
    ndraws = 500
    nparticles = 10
    yshape = 3
    model, trace = correct_instatiation(ndraws, nparticles, 'affine_invariant', yshape)

    assert all(k == i for k, i in zip(trace._straces.keys(), np.arange(0, nparticles, dtype=int)))
    for tr in trace._straces.values():
        assert tr['x'].shape == (ndraws, nparticles)
        assert tr['y'].shape == (ndraws, nparticles, yshape)


