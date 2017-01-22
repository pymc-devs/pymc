import unittest
import numpy as np
import pandas as pd
import theano.tensor as tt
from pymc3.glm import utils


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(dict(a=[1, 2, 3], b=[4, 5, 6]))

    def assertMatrixLabels(self, m, l, mt=None, lt=None):
        self.assertTrue(
            np.all(
                np.equal(
                    m.eval(),
                    mt if mt is not None else self.data.as_matrix()
                )
            )
        )
        self.assertEqual(l, list(lt or self.data.columns))

    def test_numpy_init(self):
        m, l = utils.any_to_tensor_and_labels(self.data.as_matrix())
        self.assertMatrixLabels(m, l, lt=['x0', 'x1'])
        m, l = utils.any_to_tensor_and_labels(self.data.as_matrix(), labels=['x2', 'x3'])
        self.assertMatrixLabels(m, l, lt=['x2', 'x3'])

    def test_pandas_init(self):
        m, l = utils.any_to_tensor_and_labels(self.data)
        self.assertMatrixLabels(m, l)
        m, l = utils.any_to_tensor_and_labels(self.data, labels=['x2', 'x3'])
        self.assertMatrixLabels(m, l, lt=['x2', 'x3'])

    def test_dict_input(self):
        m, l = utils.any_to_tensor_and_labels(self.data.to_dict('dict'))
        self.assertMatrixLabels(m, l, mt=self.data.as_matrix(l), lt=l)

        m, l = utils.any_to_tensor_and_labels(self.data.to_dict('series'))
        self.assertMatrixLabels(m, l, mt=self.data.as_matrix(l), lt=l)

        m, l = utils.any_to_tensor_and_labels(self.data.to_dict('list'))
        self.assertMatrixLabels(m, l, mt=self.data.as_matrix(l), lt=l)

        inp = {k: tt.as_tensor_variable(v) for k, v in self.data.to_dict('series').items()}
        m, l = utils.any_to_tensor_and_labels(inp)
        self.assertMatrixLabels(m, l, mt=self.data.as_matrix(l), lt=l)

    def test_list_input(self):
        m, l = utils.any_to_tensor_and_labels(self.data.as_matrix().tolist())
        self.assertMatrixLabels(m, l, lt=['x0', 'x1'])
        m, l = utils.any_to_tensor_and_labels(self.data.as_matrix().tolist(), labels=['x2', 'x3'])
        self.assertMatrixLabels(m, l, lt=['x2', 'x3'])

    def test_tensor_input(self):
        m, l = utils.any_to_tensor_and_labels(
            tt.as_tensor_variable(self.data.as_matrix().tolist()),
            labels=['x0', 'x1']
        )
        self.assertMatrixLabels(m, l, lt=['x0', 'x1'])
        m, l = utils.any_to_tensor_and_labels(
            tt.as_tensor_variable(self.data.as_matrix().tolist()),
            labels=['x2', 'x3'])
        self.assertMatrixLabels(m, l, lt=['x2', 'x3'])

    def test_user_mistakes(self):
        # no labels for tensor variable
        self.assertRaises(
            ValueError,
            utils.any_to_tensor_and_labels,
            tt.as_tensor_variable(self.data.as_matrix().tolist())
        )
        # len of labels is bad
        self.assertRaises(
            ValueError,
            utils.any_to_tensor_and_labels,
            self.data.as_matrix().tolist(),
            labels=['x']
        )
