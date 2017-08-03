import pytest
import functools
import numpy as np
from theano import theano

from .conftest import not_raises
import pymc3 as pm
from pymc3.variational.approximations import (
    MeanFieldGroup, FullRankGroup, NormalizingFlowGroup, EmpiricalGroup
)
from pymc3.variational.opvi import Approximation, Group


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
        (not_raises(), {MeanFieldGroup: None}),
        (not_raises(), {FullRankGroup: None, MeanFieldGroup: ['one']}),
        (not_raises(), {MeanFieldGroup: ['one'], FullRankGroup: ['two'], NormalizingFlowGroup: ['three']}),
        (pytest.raises(ValueError, match='Found duplicates'),
            {MeanFieldGroup: ['one'], FullRankGroup: ['two', 'one'], NormalizingFlowGroup: ['three']}),
        (pytest.raises(ValueError, match='No approximation is specified'), {MeanFieldGroup: ['one', 'two']}),
        (not_raises(), {MeanFieldGroup: ['one'], FullRankGroup: ['two', 'three']}),
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
        ({}, {MeanFieldGroup: (None, {})}),
        ({}, {FullRankGroup: (None, {}), MeanFieldGroup: (['one'], {})}),
        ({}, {MeanFieldGroup: (['one'], {}), FullRankGroup: (['two'], {}),
              NormalizingFlowGroup: (['three'], {'flow': 'scale-hh*2-planar-radial-loc'})}),
        ({}, {MeanFieldGroup: (['one'], {}), FullRankGroup: (['two', 'three'], {})}),
        ({}, {MeanFieldGroup: (['one'], {}), EmpiricalGroup.from_noise: (['two', 'three'], {'size': 100})})
],
    ids=lambda t: ', '.join('%s: %s' % (k.__name__, v[0]) for k, v in t[1].items())
)
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
        (MeanFieldGroup, {}),
        (FullRankGroup, {}),
        (NormalizingFlowGroup, {'flow': 'scale'}),
        (NormalizingFlowGroup, {'flow': 'loc'}),
        (NormalizingFlowGroup, {'flow': 'hh'}),
        (NormalizingFlowGroup, {'flow': 'planar'}),
        (NormalizingFlowGroup, {'flow': 'radial'}),
        (NormalizingFlowGroup, {'flow': 'radial-loc'})
    ],
    ids=lambda t: '{c} : {d}'.format(c=t[0].__name__, d=t[1])
)
def parametric_grouped_approxes(request):
    return request.param


@pytest.fixture
def three_var_aevb_groups(parametric_grouped_approxes, three_var_model, aevb_initial):
    dsize = np.prod(three_var_model.one.dshape[1:])
    cls, kw = parametric_grouped_approxes
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
    return [aevb_g, MeanFieldGroup(model=three_var_model)]


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


def test_logq_mini_1_sample_1_var(parametric_grouped_approxes, three_var_model):
    cls, kw = parametric_grouped_approxes
    approx = cls([three_var_model.one], model=three_var_model, **kw)
    logq = approx.logq
    logq = approx.set_size_and_deterministic(logq, 1, 0)
    logq.eval()


def test_logq_mini_2_sample_2_var(parametric_grouped_approxes, three_var_model):
    cls, kw = parametric_grouped_approxes
    approx = cls([three_var_model.one, three_var_model.two], model=three_var_model, **kw)
    logq = approx.logq
    logq = approx.set_size_and_deterministic(logq, 2, 0)
    logq.eval()


def test_logq_mini_sample_aevb(three_var_aevb_groups):
    approx = three_var_aevb_groups[0]
    logq, symbolic_logq = approx.set_size_and_deterministic([approx.logq, approx.symbolic_logq], 3, 0)
    e = logq.eval()
    es = symbolic_logq.eval()
    assert e.shape == ()
    assert es.shape == (3,)


def test_logq_aevb(three_var_aevb_approx):
    approx = three_var_aevb_approx
    logq, symbolic_logq = approx.set_size_and_deterministic([approx.logq, approx.symbolic_logq], 1, 0)
    e = logq.eval()
    es = symbolic_logq.eval()
    assert e.shape == ()
    assert es.shape == (1,)

    logq, symbolic_logq = approx.set_size_and_deterministic([approx.logq, approx.symbolic_logq], 2, 0)
    e = logq.eval()
    es = symbolic_logq.eval()
    assert e.shape == ()
    assert es.shape == (2,)


def test_logq_globals(three_var_approx):
    if not three_var_approx.HAS_LOGQ:
        pytest.skip('%s does not implement logq' % three_var_approx)
    approx = three_var_approx
    logq, symbolic_logq = approx.set_size_and_deterministic([approx.logq, approx.symbolic_logq], 1, 0)
    e = logq.eval()
    es = symbolic_logq.eval()
    assert e.shape == ()
    assert es.shape == (1,)

    logq, symbolic_logq = approx.set_size_and_deterministic([approx.logq, approx.symbolic_logq], 2, 0)
    e = logq.eval()
    es = symbolic_logq.eval()
    assert e.shape == ()
    assert es.shape == (2,)


@pytest.mark.parametrize(
    'raises, vfam, type_, kw',
    [
        (not_raises(), 'mean_field', MeanFieldGroup, {}),
        (not_raises(), 'mf', MeanFieldGroup, {}),
        (not_raises(), 'full_rank', FullRankGroup, {}),
        (not_raises(), 'fr', FullRankGroup, {}),
        (not_raises(), 'FR', FullRankGroup, {}),
        (not_raises(), 'loc', NormalizingFlowGroup, {}),
        (not_raises(), 'scale', NormalizingFlowGroup, {}),
        (not_raises(), 'hh', NormalizingFlowGroup, {}),
        (not_raises(), 'planar', NormalizingFlowGroup, {}),
        (not_raises(), 'radial', NormalizingFlowGroup, {}),
        (not_raises(), 'scale-loc', NormalizingFlowGroup, {}),
        (pytest.raises(ValueError, match='Need `trace` or `size`'), 'empirical', EmpiricalGroup, {}),
        (not_raises(), 'empirical', EmpiricalGroup, {'size': 100}),
    ]
)
def test_group_api_vfam(three_var_model, raises, vfam, type_, kw):
    with three_var_model, raises:
        g = Group([three_var_model.one], vfam, **kw)
        assert isinstance(g, type_)
        assert not hasattr(g, '_kwargs')
        if isinstance(g, NormalizingFlowGroup):
            assert isinstance(g.flow, pm.flows.AbstractFlow)
            assert g.flow.formula == vfam


@pytest.mark.parametrize(
    'raises, params, type_, kw, formula',
    [
        (not_raises(),
         dict(mu=np.ones((10, 2), 'float32'), rho=np.ones((10, 2), 'float32')),
         MeanFieldGroup, {}, None),

        (not_raises(),
         dict(mu=np.ones((10, 2), 'float32'),
              L_tril=np.ones(
                  FullRankGroup.get_param_spec_for(d=np.prod((10, 2)))['L_tril'],
                  'float32'
              )),
         FullRankGroup, {}, None),

        (not_raises(),
         {0: dict(loc=np.ones((10, 2), 'float32'))},
         NormalizingFlowGroup, {}, 'loc'),

        (not_raises(),
         {0: dict(log_scale=np.ones((10, 2), 'float32'))},
         NormalizingFlowGroup, {}, 'scale'),

        (not_raises(),
         {0: dict(v=np.ones((10, 2), 'float32'),)},
         NormalizingFlowGroup, {}, 'hh'),

        (not_raises(),
         {0: dict(u=np.ones((10, 2), 'float32'),
                  w=np.ones((10, 2), 'float32'),
                  b=1.)},
         NormalizingFlowGroup, {}, 'planar'),

        (not_raises(),
         {0: dict(z_ref=np.ones((10, 2), 'float32'),
                  a=1.,
                  b=1.)},
         NormalizingFlowGroup, {}, 'radial'),

        (not_raises(),
         {0: dict(log_scale=np.ones((10, 2), 'float32')),
          1: dict(loc=np.ones((10, 2), 'float32'))},
         NormalizingFlowGroup, {}, 'scale-loc'),

        (not_raises(), dict(histogram=np.ones((20, 10, 2))), EmpiricalGroup, {}, None),
    ]
)
def test_group_api_params(three_var_model, raises, params, type_, kw, formula):
    with three_var_model, raises:
        g = Group([three_var_model.one], params=params, **kw)
        assert isinstance(g, type_)
        if isinstance(g, NormalizingFlowGroup):
            assert g.flow.formula == formula
        if g.HAS_LOGQ:
            # should work as well
            logq = g.logq
            logq = g.set_size_and_deterministic(logq, 1, 0)
            logq.eval()
