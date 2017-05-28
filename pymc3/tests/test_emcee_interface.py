import pymc3 as pm
import numpy as np
from pymc3.external.emcee import AffineInvariantEnsemble

import pytest



def setup_model(yshape):
    yparams = np.linspace(0, 1, yshape)

    with pm.Model() as model:
        x = pm.Normal('x', 0, 1)
        y = pm.Normal('y', yparams, yparams, shape=yshape)
    return model

def correct_instatiation(ndraws, nwalkers, yshape):
    with setup_model(yshape) as model:
        step = AffineInvariantEnsemble(nwalkers=nwalkers)
        trace = pm.sample(ndraws, step=step)
    return model, step, trace

def test_correct_instantiation():
    ndraws = 500
    nwalkers = 10
    yshape = 3
    model, step, trace = correct_instatiation(ndraws, nwalkers, yshape)

    assert all(k == i for k, i in zip(trace._straces.keys(), np.arange(0, nwalkers, dtype=int)))
    for tr in trace._straces.values():
        assert tr['x'].shape == (ndraws, nwalkers)
        assert tr['y'].shape == (ndraws, nwalkers, yshape)


