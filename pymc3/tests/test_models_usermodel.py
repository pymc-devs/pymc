import unittest
import theano.tensor as tt
import pymc3 as pm
from pymc3.models.base import UserModel


class TestBaseModel(unittest.TestCase):
    class NewModel(UserModel):
        def __init__(self, name='', model=None):
            super(TestBaseModel.NewModel, self).__init__(name, model)
            # 1) init variables with Var method
            self.Var('v1', pm.Normal.dist())
            # 2) Potentials and Deterministic variables with method too
            # be sure that names will not overlap with other same models
            self.Deterministic('d', tt.constant(1))
            self.Potential('p', tt.constant(1))
            # avoid pm.Normal(...) initialisation as names can overlap

    def test_context_works(self):
        with pm.Model() as model:
            pm.Normal('v1')
            self.assertEqual(len(model.vars), 1)
            with UserModel('sub') as submodel:
                submodel.Var('v1', pm.Normal.dist())
                self.assertTrue(hasattr(submodel, 'v1'))
                self.assertEqual(len(submodel.vars), 1)
            self.assertEqual(len(model.vars), 2)
            with submodel:
                submodel.Var('v2', pm.Normal.dist())
                self.assertTrue(hasattr(submodel, 'v2'))
                self.assertEqual(len(submodel.vars), 2)
            self.assertEqual(len(model.vars), 3)

    def test_doc_example(self):
        with pm.Model() as model:
            # a set of variables is created
            self.NewModel()
            # another set of variables are created but with prefix 'another'
            usermodel2 = self.NewModel(name='another')
            # you can enter in a context with submodel
            with usermodel2:
                usermodel2.Var('v2', pm.Normal.dist())
                # this variable is created in parent model too

        # When you create a class based model you should follow some rules
        with model:
            m = self.NewModel('one_more')
            self.assertTrue(m.d is model['one_more_d'])
