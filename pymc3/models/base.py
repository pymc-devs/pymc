from collections import OrderedDict
from ..model import modelcontext
from ..vartypes import typefilter, discrete_types, continuous_types
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


class UserModel(object):
    """Base class for model specification
    It is supposed to be used by pymc3 contributors
    for simplifying usage of common bayesian models
    """

    def __init__(self, name=''):
        # name should be used as prefix for
        # all variables specified within a model
        self.named_vars = OrderedDict()
        if name:
            name = '%s_' % name
        self.name = name
        self.observed_RVs = list()
        self.free_RVs = list()
        self.deterministics = list()

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
        assert name not in self.named_vars, \
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
        name : str - inner name for variable in the model
        var : FreeRV or ObservedRV

        Returns
        -------
        FreeRV or ObservedRV
        """
        assert name not in self.named_vars, \
            'Cannot create duplicate var: {}'.format(name)
        if not isinstance(var, _rv_types_):
            var = _model.Deterministic(
                '{}{}'.format(self.name, name),
                var=var
            )
        self._register_var(var)
        self.named_vars[name] = var
        return var

    def __getitem__(self, item):
        return self.named_vars[item]

    @property
    def vars(self):
        """List of unobserved random variables used as inputs to the model
        (which excludes deterministics).
        """
        return self.free_RVs

    @property
    def basic_RVs(self):
        """List of random variables the model is defined in terms of
        (which excludes deterministics).
        """
        return self.free_RVs + self.observed_RVs

    @property
    def unobserved_RVs(self):
        """List of all random variable, including deterministic ones."""
        return self.vars + self.deterministics

    @property
    def disc_vars(self):
        """All the discrete variables in the model"""
        return list(typefilter(self.vars, discrete_types))

    @property
    def cont_vars(self):
        """All the continuous variables in the model"""
        return list(typefilter(self.vars, continuous_types))

    def _register_var(self, var):
        """Adds variable to variables stack"""
        if isinstance(var, (_model.ObservedRV, _model.MultiObservedRV)):
            self.observed_RVs.append(var)
        elif isinstance(var, _model.FreeRV):
            self.free_RVs.append(var)
        else:
            self.deterministics.append(var)
