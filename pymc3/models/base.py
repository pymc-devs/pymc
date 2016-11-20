from collections import OrderedDict
from ..model import modelcontext
import pymc3.model as _model


__all__ = [
    'UserModel'
]

_rv_types_ = (
    _model.FreeRV,
    _model.ObservedRV,
    _model.MultiObservedRV,
    _model.TransformedRV,
)


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
            super(TestBaseModel.NewModel, self).__init__(name, model)
            # 1) init variables with Var method
            self.Var('v1', pm.Normal.dist())
            # 2) Potentials and Deterministic variables via method too
            # be sure that names will not overlap with other same models
            self.Deterministic('d', tt.constant(1))
            self.Potential('p', tt.constant(1))
            # avoid pm.Normal(...) initialisation as names can overlap
            # instead of `add_random_variable` use `named_var` internally

    with pm.Model() as model:
        # a set of variables is created
        NewModel()
        # another set of variables are created but with prefix 'another'
        usermodel2 = NewModel(name='another')
        # you can enter in a context with submodel
    with usermodel2:
        usermodel2.Var('v2', pm.Normal.dist())
        # this variable is created in parent model too

    with model:
        m = NewModel('one_more')
        print(m.d is model['one_more_d'])   # True
    """

    def Deterministic(self, name, var):
        """Simple helper to declare deterministic variable inside a class"""
        label = '{}{}'.format(self.name, name)
        var = _model.Deterministic(label, var, self.model)
        self.deterministics.append(var)
        return self.named_var(name, var)

    def Potential(self, name, var):
        """Simple helper to declare potential variable inside a class"""
        label = '{}{}'.format(self.name, name)
        var = _model.Potential(label, var, self.model)
        self.potentials.append(var)
        return var

    def Var(self, name, dist, data=None, test_val=None):
        """Create and add (un)observed random variable to the model with an
        appropriate prior distribution. Also adds prefix to the name

        Parameters
        ----------
        name : str
        dist : distribution for the random variable
        data : array_like (optional)
           If data is provided, the variable is observed. If None,
           the variable is unobserved.
        test_val : test value for variable

        Returns
        -------
        FreeRV or ObservedRV
        """
        assert name not in self.named_vars, \
            'Cannot create duplicate var: {}'.format(name)
        label = '{}{}'.format(self.name, name)

        fv = len(self.model.free_RVs)
        ov = len(self.model.observed_RVs)
        det = len(self.model.deterministics)
        pot = len(self.model.potentials)
        mv = len(self.model.missing_values)

        var = self.model.Var(
            name=label,
            dist=dist,
            data=data
        )

        self.free_RVs += self.model.free_RVs[fv:]
        self.observed_RVs += self.model.observed_RVs[ov:]
        self.deterministics += self.model.model.deterministics[det:]
        self.potentials += self.model.model.potentials[pot:]
        self.missing_values += self.model.missing_values[mv:]

        if test_val is not None:
            var.tag.test_value = test_val
        return self.named_var(name, var)

    def __init__(self, name='', model=None):
        # name should be used as prefix for
        # all variables specified within a model
        self.named_vars = OrderedDict()
        if name:
            name = '%s_' % name
        # overrides logic of base pm.Model __init__
        self.model = modelcontext(model)
        self.name = name
        self.named_vars = {}
        # declaring lists just for code completion
        self.free_RVs = []
        self.observed_RVs = []
        self.deterministics = []
        self.potentials = []
        self.missing_values = []

    def named_var(self, name, var):
        """When user provides ready variable - do not create new one,
         just register it,

        Parameters
        ----------
        name : str - inner name for variable in the model
        var : FreeRV or ObservedRV

        Returns
        -------
        FreeRV or ObservedRV
        """
        assert name not in self.named_vars, \
            'Cannot create duplicate var: {}'.format(name)
        self.named_vars[name] = var
        if not hasattr(self, name):
            setattr(self, name, var)
        return var

