import pytest
from theano import theano, tensor as tt
import numpy as np
import pandas as pd
import numpy.testing as npt
import unittest

import pymc3 as pm
from pymc3.distributions import HalfCauchy, Normal, transforms
from pymc3 import Potential, Deterministic
from pymc3.model import ValueGradFunction


class NewModel(pm.Model):
    def __init__(self, name='', model=None):
        super().__init__(name, model)
        assert pm.modelcontext(None) is self
        # 1) init variables with Var method
        self.Var('v1', pm.Normal.dist())
        self.v2 = pm.Normal('v2', mu=0, sigma=1)
        # 2) Potentials and Deterministic variables with method too
        # be sure that names will not overlap with other same models
        pm.Deterministic('d', tt.constant(1))
        pm.Potential('p', tt.constant(1))


class DocstringModel(pm.Model):
    def __init__(self, mean=0, sigma=1, name='', model=None):
        super().__init__(name, model)
        self.Var('v1', Normal.dist(mu=mean, sigma=sigma))
        Normal('v2', mu=mean, sigma=sigma)
        Normal('v3', mu=mean, sigma=HalfCauchy('sd', beta=10, testval=1.))
        Deterministic('v3_sq', self.v3 ** 2)
        Potential('p1', tt.constant(1))


class TestBaseModel:
    def test_setattr_properly_works(self):
        with pm.Model() as model:
            pm.Normal('v1')
            assert len(model.vars) == 1
            with pm.Model('sub') as submodel:
                submodel.Var('v1', pm.Normal.dist())
                assert hasattr(submodel, 'v1')
                assert len(submodel.vars) == 1
            assert len(model.vars) == 2
            with submodel:
                submodel.Var('v2', pm.Normal.dist())
                assert hasattr(submodel, 'v2')
                assert len(submodel.vars) == 2
            assert len(model.vars) == 3

    def test_context_passes_vars_to_parent_model(self):
        with pm.Model() as model:
            # a set of variables is created
            NewModel()
            # another set of variables are created but with prefix 'another'
            usermodel2 = NewModel(name='another')
            # you can enter in a context with submodel
            with usermodel2:
                usermodel2.Var('v3', pm.Normal.dist())
                pm.Normal('v4')
                # this variable is created in parent model too
        assert 'another_v2' in model.named_vars
        assert 'another_v3' in model.named_vars
        assert 'another_v3' in usermodel2.named_vars
        assert 'another_v4' in model.named_vars
        assert 'another_v4' in usermodel2.named_vars
        assert hasattr(usermodel2, 'v3')
        assert hasattr(usermodel2, 'v2')
        assert hasattr(usermodel2, 'v4')
        # When you create a class based model you should follow some rules
        with model:
            m = NewModel('one_more')
        assert m.d is model['one_more_d']
        assert m['d'] is model['one_more_d']
        assert m['one_more_d'] is model['one_more_d']


class TestNested:
    def test_nest_context_works(self):
        with pm.Model() as m:
            new = NewModel()
            with new:
                assert pm.modelcontext(None) is new
            assert pm.modelcontext(None) is m
        assert 'v1' in m.named_vars
        assert 'v2' in m.named_vars

    def test_named_context(self):
        with pm.Model() as m:
            NewModel(name='new')
        assert 'new_v1' in m.named_vars
        assert 'new_v2' in m.named_vars

    def test_docstring_example1(self):
        usage1 = DocstringModel()
        assert 'v1' in usage1.named_vars
        assert 'v2' in usage1.named_vars
        assert 'v3' in usage1.named_vars
        assert 'v3_sq' in usage1.named_vars
        assert len(usage1.potentials), 1

    def test_docstring_example2(self):
        with pm.Model() as model:
            DocstringModel(name='prefix')
        assert 'prefix_v1' in model.named_vars
        assert 'prefix_v2' in model.named_vars
        assert 'prefix_v3' in model.named_vars
        assert 'prefix_v3_sq' in model.named_vars
        assert len(model.potentials), 1

    def test_duplicates_detection(self):
        with pm.Model():
            DocstringModel(name='prefix')
            with pytest.raises(ValueError):
                DocstringModel(name='prefix')

    def test_model_root(self):
        with pm.Model() as model:
            assert model is model.root
            with pm.Model() as sub:
                assert model is sub.root


class TestObserved:
    def test_observed_rv_fail(self):
        with pytest.raises(TypeError):
            with pm.Model():
                x = Normal('x')
                Normal('n', observed=x)

    def test_observed_type(self):
        X_ = np.random.randn(100, 5)
        X = pm.floatX(theano.shared(X_))
        with pm.Model():
            x1 = pm.Normal('x1', observed=X_)
            x2 = pm.Normal('x2', observed=X)

        assert x1.type == X.type
        assert x2.type == X.type


class TestTheanoConfig:
    def test_set_testval_raise(self):
        with theano.configparser.change_flags(compute_test_value='off'):
            with pm.Model():
                assert theano.config.compute_test_value == 'raise'
            assert theano.config.compute_test_value == 'off'

    def test_nested(self):
        with theano.configparser.change_flags(compute_test_value='off'):
            with pm.Model(theano_config={'compute_test_value': 'ignore'}):
                assert theano.config.compute_test_value == 'ignore'
                with pm.Model(theano_config={'compute_test_value': 'warn'}):
                    assert theano.config.compute_test_value == 'warn'
                assert theano.config.compute_test_value == 'ignore'
            assert theano.config.compute_test_value == 'off'


def test_duplicate_vars():
    with pytest.raises(ValueError) as err:
        with pm.Model():
            pm.Normal('a')
            pm.Normal('a')
    err.match('already exists')

    with pytest.raises(ValueError) as err:
        with pm.Model():
            pm.Normal('a')
            pm.Normal('a', transform=transforms.log)
    err.match('already exists')

    with pytest.raises(ValueError) as err:
        with pm.Model():
            a = pm.Normal('a')
            pm.Potential('a', a**2)
    err.match('already exists')

    with pytest.raises(ValueError) as err:
        with pm.Model():
            pm.Binomial('a', 10, .5)
            pm.Normal('a', transform=transforms.log)
    err.match('already exists')


def test_empty_observed():
    data = pd.DataFrame(np.ones((2, 3)) / 3)
    data.values[:] = np.nan
    with pm.Model():
        a = pm.Normal('a', observed=data)
        npt.assert_allclose(a.tag.test_value, np.zeros((2, 3)))
        b = pm.Beta('b', alpha=1, beta=1, observed=data)
        npt.assert_allclose(b.tag.test_value, np.ones((2, 3)) / 2)


class TestValueGradFunction(unittest.TestCase):
    def test_no_extra(self):
        a = tt.vector('a')
        a.tag.test_value = np.zeros(3, dtype=a.dtype)
        a.dshape = (3,)
        a.dsize = 3
        f_grad = ValueGradFunction(a.sum(), [a], [], mode='FAST_COMPILE')
        assert f_grad.size == 3

    def test_invalid_type(self):
        a = tt.ivector('a')
        a.tag.test_value = np.zeros(3, dtype=a.dtype)
        a.dshape = (3,)
        a.dsize = 3
        with pytest.raises(TypeError) as err:
            ValueGradFunction(a.sum(), [a], [], mode='FAST_COMPILE')
        err.match('Invalid dtype')

    def setUp(self):
        extra1 = tt.iscalar('extra1')
        extra1_ = np.array(0, dtype=extra1.dtype)
        extra1.tag.test_value = extra1_
        extra1.dshape = tuple()
        extra1.dsize = 1

        val1 = tt.vector('val1')
        val1_ = np.zeros(3, dtype=val1.dtype)
        val1.tag.test_value = val1_
        val1.dshape = (3,)
        val1.dsize = 3

        val2 = tt.matrix('val2')
        val2_ = np.zeros((2, 3), dtype=val2.dtype)
        val2.tag.test_value = val2_
        val2.dshape = (2, 3)
        val2.dsize = 6

        self.val1, self.val1_ = val1, val1_
        self.val2, self.val2_ = val2, val2_
        self.extra1, self.extra1_ = extra1, extra1_

        self.cost = extra1 * val1.sum() + val2.sum()

        self.f_grad = ValueGradFunction(
            self.cost, [val1, val2], [extra1], mode='FAST_COMPILE')

    def test_extra_not_set(self):
        with pytest.raises(ValueError) as err:
            self.f_grad.get_extra_values()
        err.match('Extra values are not set')

        with pytest.raises(ValueError) as err:
            self.f_grad(np.zeros(self.f_grad.size, dtype=self.f_grad.dtype))
        err.match('Extra values are not set')

    def test_grad(self):
        self.f_grad.set_extra_values({'extra1': 5})
        array = np.ones(self.f_grad.size, dtype=self.f_grad.dtype)
        val, grad = self.f_grad(array)
        assert val == 21
        npt.assert_allclose(grad, [5, 5, 5, 1, 1, 1, 1, 1, 1])

    def test_bij(self):
        self.f_grad.set_extra_values({'extra1': 5})
        array = np.ones(self.f_grad.size, dtype=self.f_grad.dtype)
        point = self.f_grad.array_to_dict(array)
        assert len(point) == 2
        npt.assert_allclose(point['val1'], 1)
        npt.assert_allclose(point['val2'], 1)

        array2 = self.f_grad.dict_to_array(point)
        npt.assert_allclose(array2, array)
        point_ = self.f_grad.array_to_full_dict(array)
        assert len(point_) == 3
        assert point_['extra1'] == 5

    def test_edge_case(self):
        # Edge case discovered in #2948
        ndim = 3
        with pm.Model() as m:
            pm.Lognormal('sigma',
                         mu=np.zeros(ndim),
                         tau=np.ones(ndim),
                         shape=ndim)  # variance for the correlation matrix
            pm.HalfCauchy('nu', beta=10)
            step = pm.NUTS()

        func = step._logp_dlogp_func
        func.set_extra_values(m.test_point)
        q = func.dict_to_array(m.test_point)
        logp, dlogp = func(q)
        assert logp.size == 1
        assert dlogp.size == 4
        npt.assert_allclose(dlogp, 0., atol=1e-5)
