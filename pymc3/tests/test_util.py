import pytest
import numpy as np
from pymc3.distributions.transforms import Transform
import pymc3 as pm


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


class TestMinibatch(object):
    data = np.random.rand(30, 10, 40, 10, 50)

    def test_1d(self):
        mb = pm.Minibatch(self.data, 20)
        assert mb.minibatch_shape == (20, 10, 40, 10, 50)

    def test_2d(self):
        with pytest.raises(TypeError):
            pm.Minibatch(self.data, (10, 5))
        mb = pm.Minibatch(self.data, [(10, 42), (4, 42)])
        assert mb.minibatch_shape == (10, 4, 40, 10, 50)

    def test_special(self):
        mb = pm.Minibatch(self.data, [(10, 42), None, (4, 42)])
        assert mb.minibatch_shape == (10, 10, 4, 10, 50)
        mb = pm.Minibatch(self.data, [(10, 42), Ellipsis, (4, 42)])
        assert mb.minibatch_shape == (10, 10, 40, 10, 4)
