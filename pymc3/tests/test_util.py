import pytest

import theano
import pymc3 as pm
import numpy as np
from numpy.testing import assert_almost_equal
from .helpers import SeededTest
from pymc3.distributions.transforms import Transform
from pymc3.util import DependenceDAG, WrapAsHashable


class TestTransformName(object):
    cases = [
        ('var', 'var_test__'),
        ('var_test_', 'var_test__test__')
    ]
    transform_name = 'test'

    def test_get_transformed_name(self):
        test_transform = Transform()
        test_transform.name = self.transform_name
        for name, transformed in self.cases:
            assert pm.util.get_transformed_name(name,
                                                test_transform) == transformed

    def test_is_transformed_name(self):
        for name, transformed in self.cases:
            assert pm.util.is_transformed_name(transformed)
            assert not pm.util.is_transformed_name(name)

    def test_get_untransformed_name(self):
        for name, transformed in self.cases:
            assert pm.util.get_untransformed_name(transformed) == name
            with pytest.raises(ValueError):
                pm.util.get_untransformed_name(name)


class TestUpdateStartVals(SeededTest):
    def setup_method(self):
        super(TestUpdateStartVals, self).setup_method()

    def test_soft_update_all_present(self):
        start = {'a': 1, 'b': 2}
        test_point = {'a': 3, 'b': 4}
        pm.util.update_start_vals(start, test_point, model=None)
        assert start == {'a': 1, 'b': 2}

    def test_soft_update_one_missing(self):
        start = {'a': 1, }
        test_point = {'a': 3, 'b': 4}
        pm.util.update_start_vals(start, test_point, model=None)
        assert start == {'a': 1, 'b': 4}

    def test_soft_update_empty(self):
        start = {}
        test_point = {'a': 3, 'b': 4}
        pm.util.update_start_vals(start, test_point, model=None)
        assert start == test_point

    def test_soft_update_transformed(self):
        with pm.Model() as model:
            pm.Exponential('a', 1)
        start = {'a': 2.}
        test_point = {'a_log__': 0}
        pm.util.update_start_vals(start, test_point, model)
        assert_almost_equal(np.exp(start['a_log__']), start['a'])

    def test_soft_update_parent(self):
        with pm.Model() as model:
            a = pm.Uniform('a', lower=0., upper=1.)
            b = pm.Uniform('b', lower=2., upper=3.)
            pm.Uniform('lower', lower=a, upper=3.)
            pm.Uniform('upper', lower=0., upper=b)
            pm.Uniform('interv', lower=a, upper=b)

        start = {'a': .3, 'b': 2.1, 'lower': 1.4, 'upper': 1.4, 'interv': 1.4}
        test_point = {'lower_interval__': -0.3746934494414109,
                      'upper_interval__': 0.693147180559945,
                      'interv_interval__': 0.4519851237430569}
        pm.util.update_start_vals(start, model.test_point, model)
        assert_almost_equal(start['lower_interval__'],
                            test_point['lower_interval__'])
        assert_almost_equal(start['upper_interval__'],
                            test_point['upper_interval__'])
        assert_almost_equal(start['interv_interval__'],
                            test_point['interv_interval__'])


class TestDependenceDAG(object):
    def setup_method(self):
        self.obs = np.random.randn(1000,) + 2
        with pm.Model() as model:
            self.a = pm.Normal('a', mu=0, sd=100)
            self.b = pm.Normal('b', mu=self.a, sd=1e-8)
            self.c = pm.Normal('c', mu=self.a, sd=1e-8)
            self.d = pm.Deterministic('d', self.b + self.c)
            self.e = pm.Normal('e', mu=self.d, sd=1, observed=self.obs)
        self.model = model
        self.expected_full_dag = DependenceDAG()
        self.expected_full_dag.add(self.e)

    def test_built_DependenceDAG(self):
        assert self.expected_full_dag == self.model.variable_dependence_dag

    def test_get_sub_dag(self):
        dag = self.model.variable_dependence_dag
        sub1 = dag.get_sub_dag(self.a)
        assert len(sub1.nodes) == 1
        assert sub1.check_integrity()

        sub2 = dag.get_sub_dag([self.a])
        assert len(sub2.nodes) == 1
        assert sub2.check_integrity()
        assert sub1 == sub2

        sub3 = dag.get_sub_dag([self.e])
        assert len(sub3.nodes) == 5
        assert sub3.check_integrity()
        assert sub3 == dag

        hard = dag.get_sub_dag([self.e,
                                theano.tensor.exp(self.b +
                                                  self.e * self.e) * self.e *
                                self.b + self.a])
        assert len(hard.nodes) == 6
        assert hard.check_integrity()
        new_node_depth = [hard.depth[n] for n in hard
                          if n not in self.model.basic_RVs +
                          self.model.deterministics][0]
        assert new_node_depth == 4
        assert hard.get_sub_dag(self.e) == dag

        params = [self.d,
                  0.,
                  np.zeros((10, 2), dtype=np.float32),
                  theano.tensor.constant(5.354),
                  theano.shared(np.array([2, 6, 8])),
                  ]
        with_non_theano, index = dag.get_sub_dag(params,
                                                 return_index=True)
        assert with_non_theano.check_integrity()
        assert with_non_theano.get_sub_dag(self.d) == dag.get_sub_dag([self.d])
        assert index[0] == params[0]
        assert all([isinstance(index[i], WrapAsHashable)
                    for i in range(1, len(params))])
        for i in range(1, len(params)):
            supplied = params[i]
            obj = index[i]
            assert isinstance(obj, WrapAsHashable)
            assert obj in with_non_theano
            if obj.node_is_hashable:
                assert obj.node == supplied
            else:
                assert id(obj.node) == id(params[i])
            obj_value = obj.get_value()
            if isinstance(supplied, theano.tensor.sharedvar.SharedVariable):
                expected_value = supplied.get_value()
            elif isinstance(supplied, theano.tensor.TensorConstant):
                expected_value = supplied.value
            else:
                expected_value = supplied
            if isinstance(obj_value, np.ndarray):
                assert np.all(obj_value == expected_value)
            else:
                assert obj_value == expected_value

        wnt2 = with_non_theano.copy()
        assert with_non_theano == wnt2
        wnt2, node2 = wnt2.add(params[-1], force=True, return_added_node=True)
        assert wnt2 == with_non_theano
        assert node2 == index[len(params) - 1]
