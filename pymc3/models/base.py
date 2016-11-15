from ..model import modelcontext
from collections import OrderedDict


class UserModel(object):
    """Base class for model specification
    It is supposed to be used by pymc3 contributors
    for simplifying usage of common bayesian models
    """

    def __init__(self, name=''):
        # name should be used as prefix for
        # all variables specified within a model
        self.vars = OrderedDict()
        if name:
            name = '%s_' % name
        self.name = name

    @property
    def model(self):
        """Shortcut to model"""
        return modelcontext(None)

    def new_var(self, name, dist, data=None, test_val=None):
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
        assert name not in self.vars, \
            'Cannot create duplicate var: {}'.format(name)
        label = '{}{}'.format(self.name, name)
        var = self.model.Var(label,
                             dist=dist, data=data)
        if test_val is not None:
            var.tag.test_value = test_val
        return self.add_var(name, var)

    def add_var(self, name, var):
        """When user provides ready variable - do not create new one,
         just register it

        Parameters
        ----------
        name : str - inner name for the model
        var : FreeRV or ObservedRV

        Returns
        -------
        FreeRV or ObservedRV
        """
        assert name not in self.vars, \
            'Cannot create duplicate var: {}'.format(name)
        self.vars[name] = var
        return var

    def __getitem__(self, item):
        return self.vars[item]
