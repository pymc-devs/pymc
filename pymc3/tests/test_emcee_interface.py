import numpy as np
import pymc3 as pm

import pytest


def setup_model(yshape):
    if yshape is not None:
        yparams = np.linspace(0, 1, yshape)
    else:
        yparams = 1
        yshape = ()

    with pm.Model() as model:
        x = pm.Normal('x', 0, 1)
        y = pm.Normal('y', yparams, yparams, shape=yshape)
    return model


def correct_instatiation(ndraws, nparticles, step, yshape, init):
    with setup_model(yshape) as model:
        trace = pm.sample(ndraws, step=step(nparticles=nparticles), init=init, n_init=1000)
    return model, trace


@pytest.fixture(params=[None, 3, 1])
def yshape(request):
    return request.param


@pytest.fixture(params=[None, 10])
def nparticles(request):
    return request.param


@pytest.fixture(params=['advi', 'random'])
def init(request):
    return request.param

@pytest.fixture()
def affine_invariant_ensemble():
    from pymc3.external.emcee.step_methods import AffineInvariantEnsemble
    return AffineInvariantEnsemble


def test_correct_instantiation(yshape, nparticles, init, affine_invariant_ensemble):
    ndraws = 100
    model, trace = correct_instatiation(ndraws, nparticles, affine_invariant_ensemble, yshape, init)

    if yshape is None:
        ylen = 1
        yshape = tuple()
    else:
        ylen = yshape
        yshape = (yshape, )
    if nparticles is None:
        nparticles = (ylen + 1) * 2
    assert all(k == i for k, i in zip(trace._straces.keys(), np.arange(0, nparticles, dtype=int)))
    assert trace['x'].shape == (ndraws * nparticles, )
    assert trace['y'].shape == (ndraws * nparticles, ) + yshape




