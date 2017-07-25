import pytest
import functools
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
        ({}, {MeanField: (['one'], {}), FullRank: (['two'], {}), NormalizingFlow: (['three'], {})}),
        ({}, {MeanField: (['one'], {}), FullRank: (['two', 'three'], {})}),
])
def three_var_approx(request, three_var_model):
    kw, grouping = request.param
    approxes, groups = zip(*grouping.items())
    groups, gkwargs = zip(*groups)
    groups = [list(map(functools.partial(getattr, three_var_model), g))
              if g is not None else None
              for g in groups]
    inited_groups = [a(group=g, model=three_var_model, **gk) for a, g, gk in zip(approxes, groups, gkwargs)]
    approx = Approximation(inited_groups, model=three_var_model)
    return approx


def test_sample_simple(three_var_approx):
    trace = three_var_approx.sample(500)
    assert set(trace.varnames) == {'one', 'two', 'three'}
    assert len(trace) == 500
