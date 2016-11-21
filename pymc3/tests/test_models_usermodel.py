import unittest
import theano.tensor as tt
import pymc3 as pm
from pymc3.models.base import UserModel


class NewModel(pm.Model):
    def __init__(self, name='', model=None):
        super(NewModel, self).__init__(name, model)
        assert pm.modelcontext(None) is self
        # 1) init variables with Var method
        self.Var('v1', pm.Normal.dist())
        self.v2 = pm.Normal('v2', 0, 1)
        # 2) Potentials and Deterministic variables with method too
        # be sure that names will not overlap with other same models
        pm.Deterministic('d', tt.constant(1))
        pm.Potential('p', tt.constant(1))


class TestBaseModel(unittest.TestCase):
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
        self.assertIn('another_v2', model.named_vars)
        self.assertIn('another_v3', model.named_vars)
        self.assertIn('another_v3', usermodel2.named_vars)
        self.assertIn('another_v4', model.named_vars)
        self.assertIn('another_v4', usermodel2.named_vars)
        self.assertTrue(hasattr(usermodel2, 'v3'))
        self.assertTrue(hasattr(usermodel2, 'v2'))
        self.assertTrue(hasattr(usermodel2, 'v4'))
        # When you create a class based model you should follow some rules
        with model:
            m = NewModel('one_more')
            self.assertTrue(m.d is model['one_more_d'])


class TestNested(unittest.TestCase):
    def test_nest_context(self):
        with pm.Model() as m:
            new = NewModel()
            with new:
                self.assertTrue(
                    pm.modelcontext(None) is new
                )
            self.assertTrue(
                pm.modelcontext(None) is m
            )
        self.assertIn('v1', m.named_vars)
        self.assertIn('v2', m.named_vars)

    def test_named_context(self):
        with pm.Model() as m:
            NewModel(name='new')
        self.assertIn('new_v1', m.named_vars)
        self.assertIn('new_v2', m.named_vars)
