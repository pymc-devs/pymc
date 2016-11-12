from ..model import modelcontext
from collections import OrderedDict


class UserModel(object):
    """Base class for model specification
    It is supposed to be used by pymc3 contributors
    for simplifying usage of common bayesian models
    """
    def __init__(self, name):
        # name should be used as prefix for
        # all variables specified within a model
        self.vars = OrderedDict()
        self.name = name

    @property
    def model(self):
        """Shortcut to model"""
        return modelcontext(None)

    def add_var(self, name, dist, data=None):
        """Create and add (un)observed random variable to the model with an
        appropriate prior distribution. Also adds prefix to the name

        Parameters
        ----------
        name : str
        dist : distribution for the random variable
        data : array_like (optional)
           If data is provided, the variable is observed. If None,
           the variable is unobserved.

        Returns
        -------
        FreeRV or ObservedRV
        """
        var = self.model.Var('{}_{}'.format(self, name, name), dist=dist, data=data)
        self.vars[name] = var
        return var
