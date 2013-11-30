from .point import *
from .vartypes import *

from theano import theano, tensor as t, function

import numpy as np
from functools import wraps
from .theanof import *

from .memoize import memoize

__all__ = ['Model', 'compilef', 'fn', 'pointfn', 'modelcontext', 'Point', 'Deterministic']


class Context(object):
    def __enter__(self):
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls, "contexts"):
            cls.contexts = []

        return cls.contexts

    @classmethod
    def get_context(cls):
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            raise TypeError("No context on context stack")


def modelcontext(model):
    if model is None:
        return Model.get_context()
    return model


class Model(Context):
    """
    Base class for encapsulation of the variables and
    likelihood factors of a model.
    """

    def __init__(self):
        self.named_vars = {}
        self.free_RVs = []
        self.observed_RVs = [] 
        self.deterministics = []
        self.potentials = []

    @property
    @memoize
    def logp(self):
        """
        log-probability of the model

        Parameters
        ----------

        self : Model

        Returns
        -------

        logp : Theano scalar

        """
        factors = [var.logp for var in (self.free_RVs + self.observed_RVs)] + self.potentials
        return t.add(*map(t.sum, factors))

    @property 
    def vars(self): 
        return self.free_RVs

    @property 
    def unobserved_RVs(self): 
        return self.free_RVs + self.deterministics 

    @property
    def logpc(self):
        """Compiled log probability density function"""
        return compilef(self.logp)

    def dlogpc(self, vars=None):
        """Compiled log probability density gradient function"""
        return compilef(gradient(self.logp, vars))

    def d2logpc(self, vars=None):
        """Compiled log probability density hessian function"""
        return compilef(hessian(self.logp, vars))

    @property
    def test_point(self):
        """Test point used to check that the model doesn't generate errors"""
        return Point(((var, var.tag.test_value) for var in self.vars),
                     model=self)

    @property
    def cont_vars(self):
        """All the continuous variables in the model"""
        return list(typefilter(self.vars, continuous_types))

    """
    these functions add random variables
    """
    def Data(self, name, dist, data):
        var = Data(name, args, dist)
        self.observed_RVs.append(var)
        self.add_random_variable(var)
        return var

    def Var(self, name, dist):
        var = dist.makevar(name)
        self.free_RVs.append(var)
        self.add_random_variable(var)
        return var

    def TransformedVar(self, name, dist, trans):
        tvar = self.Var(trans.name + '_' + name, trans.apply(dist))

        return Deterministic(name, trans.backward(tvar)), tvar

    def AddPotential(self, potential):

        self.potentials.append(potential)

    def add_random_variable(self, var):
        self.named_vars[var.name] = var
        if not hasattr(self, var.name):
            setattr(self, var.name, var)
        return var

    def __getitem__(self, key):
        return self.named_vars[key]

class Data(object): 
    def __init__(self, name, data, distribution):
        self.name = name

        if hasattr(data, 'values'):
            # Incase obs is a Series or DataFrame
            data = data.values
        self.data = map(t.constant, as_iterargs(data))

        self.distribution = distribution
        self.logp = self.distribution.logp(*data_args)

    def __str__(self): 
        return self.name

def Point(*args, **kwargs):

    """
    Build a point. Uses same args as dict() does.
    Filters out variables not in the model. All keys are strings.

    Parameters
    ----------
        *args, **kwargs
            arguments to build a dict
    """
    model = modelcontext(kwargs.get('model'))
    kwargs.pop('model', None)

    args = [a for a in args]
    try:
        d = dict(*args, **kwargs)
    except Exception as e:
        raise TypeError(
            "can't turn " + str(args) + " and " + str(kwargs) +
            " into a dict. " + str(e))

    varnames = list(map(str, model.vars))
    return dict((str(k), np.array(v))
                for (k, v) in d.items()
                if str(k) in varnames)

@memoize
def fn(outs, mode=None):
    """
    Compiles a Theano function which returns `outs` and takes the variable
    ancestors of `outs` as inputs.

    Parameters
    ----------
    outs : Theano variable or iterable of Theano variables
    mode : Theano compilation mode

    Returns
    -------
    Compiled Theano function
    """
    return function(inputvars(outs), outs,
                 allow_input_downcast=True,
                 on_unused_input='ignore',
                 mode=mode)

@memoize
def pointfn(outs, mode=None):
    """
    Compiles a Theano function which returns `outs` and takes the variable
    ancestors of `outs` as inputs.

    Parameters
    ----------
    outs : Theano variable or iterable of Theano variables
    mode : Theano compilation mode

    Returns
    -------
    Compiled Theano function as point function.
    """
    return PointFunc(fn(outs, mode))

compilef = pointfn 


def Deterministic(name, var, model=None):
    """
    Create a named deterministic variable

    Parameters
    ----------
        name : str
        var : theano variables
    Returns
    -------
        n : var but with name name
    """
    var.name = name
    modelcontext(model).add_random_variable(var)
    return var


def as_iterargs(data):
    if isinstance(data, tuple):
        return data
    if hasattr(data, 'columns'):  # data frames
        return [np.asarray(data[c]) for c in data.columns]
    else:
        return [data]

# theano stuff
theano.config.warn.sum_div_dimshuffle_bug = False
theano.config.compute_test_value = 'raise'
