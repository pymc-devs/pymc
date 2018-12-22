import numpy as np
import pandas as pd
import theano.tensor as tt
from pymc3.glm import utils
import pytest


class TestUtils:
    def setup_method(self):
        self.data = pd.DataFrame(dict(a=[1, 2, 3], b=[4, 5, 6]))

    def assertMatrixLabels(self, m, l, mt=None, lt=None):
        assert np.all(
                np.equal(
                    m.eval(),
                    mt if mt is not None else self.data.values
                )
            )
        assert l == list(lt or self.data.columns)

    def test_numpy_init(self):
        m, l = utils.any_to_tensor_and_labels(self.data.values)
        self.assertMatrixLabels(m, l, lt=['x0', 'x1'])
        m, l = utils.any_to_tensor_and_labels(self.data.values, labels=['x2', 'x3'])
        self.assertMatrixLabels(m, l, lt=['x2', 'x3'])

    def test_pandas_init(self):
        m, l = utils.any_to_tensor_and_labels(self.data)
        self.assertMatrixLabels(m, l)
        m, l = utils.any_to_tensor_and_labels(self.data, labels=['x2', 'x3'])
        self.assertMatrixLabels(m, l, lt=['x2', 'x3'])

    def test_dict_input(self):
        m, l = utils.any_to_tensor_and_labels(self.data.to_dict('dict'))
        self.assertMatrixLabels(m, l, mt=self.data[l].values, lt=l)

        m, l = utils.any_to_tensor_and_labels(self.data.to_dict('series'))
        self.assertMatrixLabels(m, l, mt=self.data[l].values, lt=l)

        m, l = utils.any_to_tensor_and_labels(self.data.to_dict('list'))
        self.assertMatrixLabels(m, l, mt=self.data[l].values, lt=l)

        inp = {k: tt.as_tensor_variable(v) for k, v in self.data.to_dict('series').items()}
        m, l = utils.any_to_tensor_and_labels(inp)
        self.assertMatrixLabels(m, l, mt=self.data[l].values, lt=l)

    def test_list_input(self):
        m, l = utils.any_to_tensor_and_labels(self.data.values.tolist())
        self.assertMatrixLabels(m, l, lt=['x0', 'x1'])
        m, l = utils.any_to_tensor_and_labels(self.data.values.tolist(), labels=['x2', 'x3'])
        self.assertMatrixLabels(m, l, lt=['x2', 'x3'])

    def test_tensor_input(self):
        m, l = utils.any_to_tensor_and_labels(
            tt.as_tensor_variable(self.data.values.tolist()),
            labels=['x0', 'x1']
        )
        self.assertMatrixLabels(m, l, lt=['x0', 'x1'])
        m, l = utils.any_to_tensor_and_labels(
            tt.as_tensor_variable(self.data.values.tolist()),
            labels=['x2', 'x3'])
        self.assertMatrixLabels(m, l, lt=['x2', 'x3'])

    def test_user_mistakes(self):
        # no labels for tensor variable
        with pytest.raises(
            ValueError):
            utils.any_to_tensor_and_labels(tt.as_tensor_variable(self.data.values.tolist()))
        # len of labels is bad
        with pytest.raises(
            ValueError):
            utils.any_to_tensor_and_labels(self.data.values.tolist(),
            labels=['x'])
