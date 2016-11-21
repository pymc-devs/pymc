from collections import OrderedDict
from ..model import modelcontext
import pymc3.model as _model


__all__ = [
    'UserModel'
]


class UserModel(_model.Model):
    """Base class for class based model specification
    It is supposed to be used by pymc3 contributors
    for simplifying usage of common bayesian models.
    Class based model has same attributes as pm.Model

    The only difference is that `add_random_variable`
    is not suggested for usage, you should prefer `named_var`
    as more consistent with renaming inside a model

    Usage
    -----
    # When you create a class based model you should follow some rules
    class NewModel(UserModel):
        def __init__(self, name='', model=None):
            # 0) init base class first
            super(NewModel, self).__init__(name, model)
            # 1) create variables with Var method
            self.Var('v1', pm.Normal.dist())
            # 2) create Potentials and Deterministic variables via method too
            # Then be sure that names will not overlap with other same models
            self.Deterministic('d', tt.constant(1))
            self.Potential('p', tt.constant(1))
            # 3) avoid pm.Normal(...) initialization as names can overlap
            # 4) instead of `add_random_variable` use `named_var` internally

    with pm.Model() as model:
        # a set of variables is created
        NewModel()
        # another set of variables is created but with prefix 'another'
        usermodel2 = NewModel(name='another')
        # you can enter in a context with submodel

    with usermodel2:
        usermodel2.Var('v2', pm.Normal.dist())
        # this variable is created in parent model too
    # this works too
    usermodel2.Var('v3', pm.Normal.dist())
    with model:
        m = NewModel('one_more')
        print(m.d is model['one_more_d'])   # True
    """
