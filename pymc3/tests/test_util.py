import pytest

import pymc3 as pm
import numpy as np
from numpy.testing import assert_almost_equal
from .helpers import SeededTest
from pymc3.distributions.transforms import Transform


class TestTransformName:
    cases = [
        ('var', 'var_test__'),
        ('var_test_', 'var_test__test__')
    ]
    transform_name = 'test'

    def test_get_transformed_name(self):
        test_transform = Transform()
        test_transform.name = self.transform_name
        for name, transformed in self.cases:
            assert pm.util.get_transformed_name(name, test_transform) == transformed

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
        super().setup_method()

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

        start = {'a': .3, 'b': 2.1, 'lower': 1.4, 'upper': 1.4, 'interv':1.4}
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


