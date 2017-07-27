import pytest
import functools
import numpy as np
from theano import theano, tensor as tt

from .conftest import not_raises
import pymc3 as pm
from pymc3.variational.approximations import (
    MeanField, FullRank, NormalizingFlow, Empirical
)
from pymc3.variational.opvi import Approximation


pytestmark = pytest.mark.usefixtures(
    'strict_float32',
    'seeded_test'
)


@pytest.fixture('module')
def three_var_model():
    with pm.Model() as model:
        pm.Normal('one', shape=(10, 2))
        pm.Normal('two', shape=(10, ))
        pm.Normal('three', shape=(10, 1, 2))
    return model


@pytest.mark.parametrize(
    ['raises', 'grouping'],
    [
        (not_raises(), {MeanField: None}),
        (not_raises(), {FullRank: None, MeanField: ['one']}),
        (not_raises(), {MeanField: ['one'], FullRank: ['two'], NormalizingFlow: ['three']}),
        (pytest.raises(ValueError, match='Found duplicates'),
            {MeanField: ['one'], FullRank: ['two', 'one'], NormalizingFlow: ['three']}),
        (pytest.raises(ValueError, match='No approximation is specified'), {MeanField: ['one', 'two']}),
        (not_raises(), {MeanField: ['one'], FullRank: ['two', 'three']}),
    ]
)
def test_init_groups(three_var_model, raises, grouping):
    with raises, three_var_model:
        approxes, groups = zip(*grouping.items())
        groups = [list(map(functools.partial(getattr, three_var_model), g))
                  if g is not None else None
                  for g in groups]
        inited_groups = [a(group=g) for a, g in zip(approxes, groups)]
        approx = Approximation(inited_groups)
        for ig, g in zip(inited_groups, groups):
            if g is None:
                pass
            else:
                assert set(g) == set(ig.group)
        else:
            assert approx.ndim == three_var_model.ndim


@pytest.fixture(params=[
        ({}, {MeanField: (None, {})}),
        ({}, {FullRank: (None, {}), MeanField: (['one'], {})}),
        ({}, {MeanField: (['one'], {}), FullRank: (['two'], {}),
              NormalizingFlow: (['three'], {'flow': 'scale-hh*2-planar-radial-loc'})}),
        ({}, {MeanField: (['one'], {}), FullRank: (['two', 'three'], {})}),
        ({}, {MeanField: (['one'], {}), Empirical.from_noise: (['two', 'three'], {'size': 100})})
])
def three_var_groups(request, three_var_model):
    kw, grouping = request.param
    approxes, groups = zip(*grouping.items())
    groups, gkwargs = zip(*groups)
    groups = [list(map(functools.partial(getattr, three_var_model), g))
              if g is not None else None
              for g in groups]
    inited_groups = [a(group=g, model=three_var_model, **gk) for a, g, gk in zip(approxes, groups, gkwargs)]
    return inited_groups


@pytest.fixture
def three_var_approx(three_var_model, three_var_groups):
    approx = Approximation(three_var_groups, model=three_var_model)
    return approx


def test_sample_simple(three_var_approx):
    trace = three_var_approx.sample(500)
    assert set(trace.varnames) == {'one', 'two', 'three'}
    assert len(trace) == 500
    assert trace[0]['one'].shape == (10, 2)
    assert trace[0]['two'].shape == (10, )
    assert trace[0]['three'].shape == (10, 1, 2)


@pytest.fixture('module')
def aevb_initial():
    return theano.shared(np.random.rand(3, 7).astype('float32'))


@pytest.fixture(
    params=[
        (MeanField, {}),
        (FullRank, {}),
        (NormalizingFlow, {'flow': 'scale'}),
        (NormalizingFlow, {'flow': 'loc'}),
        (NormalizingFlow, {'flow': 'hh'}),
        (NormalizingFlow, {'flow': 'planar'}),
        (NormalizingFlow, {'flow': 'radial'}),
        (NormalizingFlow, {'flow': 'radial-loc'})
    ]
)
def three_var_aevb_groups(request, three_var_model, aevb_initial):
    dsize = np.prod(three_var_model.one.dshape[1:])
    cls, kw = request.param
    spec = cls.get_param_spec_for(d=dsize, **kw)
    params = dict()
    for k, v in spec.items():
        if isinstance(k, int):
            params[k] = dict()
            for k_i, v_i in v.items():
                params[k][k_i] = aevb_initial.dot(np.random.rand(7, *v_i).astype('float32'))
        else:
            params[k] = aevb_initial.dot(np.random.rand(7, *v).astype('float32'))
    aevb_g = cls([three_var_model.one], params=params, model=three_var_model, local=True)
    return [aevb_g, MeanField(model=three_var_model)]


@pytest.fixture
def three_var_aevb_approx(three_var_model, three_var_aevb_groups):
    approx = Approximation(three_var_aevb_groups, model=three_var_model)
    return approx


def test_sample_aevb(three_var_aevb_approx, aevb_initial):
    aevb_initial.set_value(np.random.rand(7, 7).astype('float32'))
    trace = three_var_aevb_approx.sample(500)
    assert set(trace.varnames) == {'one', 'two', 'three'}
    assert len(trace) == 500
    assert trace[0]['one'].shape == (7, 2)
    assert trace[0]['two'].shape == (10, )
    assert trace[0]['three'].shape == (10, 1, 2)

    aevb_initial.set_value(np.random.rand(13, 7).astype('float32'))
    trace = three_var_aevb_approx.sample(500)
    assert set(trace.varnames) == {'one', 'two', 'three'}
    assert len(trace) == 500
    assert trace[0]['one'].shape == (13, 2)
    assert trace[0]['two'].shape == (10,)
    assert trace[0]['three'].shape == (10, 1, 2)
