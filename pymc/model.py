from .point import *
from .vartypes import *

from theano import theano, tensor as t, function
from theano.tensor.var import TensorVariable

import numpy as np
from functools import wraps
from .theanof import *
from inspect import getargspec

from .memoize import memoize

__all__ = ['Model', 'Factor', 'compilef', 'fn', 'fastfn', 'modelcontext', 'Point', 'Deterministic']


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

class Factor(object): 
    @property
    def logp(self):
        """Compiled log probability density function"""
        return self.model.fn(self.logpt)

    @property
    def logp_elemwise(self):
        return self.model.fn(self.logp_elemwiset)

    def dlogp(self, vars=None):
        """Compiled log probability density gradient function"""
        return self.model.fn(gradient(self.logpt, vars))

    def d2logp(self, vars=None):
        """Compiled log probability density hessian function"""
        return self.model.fn(hessian(self.logpt, vars))

    @property
    def fastlogp(self):
        """Compiled log probability density function"""
        return self.model.fastfn(self.logpt)

    def fastdlogp(self, vars=None):
        """Compiled log probability density gradient function"""
        return self.model.fastfn(gradient(self.logpt, vars))

    def fastd2logp(self, vars=None):
        """Compiled log probability density hessian function"""
        return self.model.fastfn(hessian(self.logpt, vars))

    @property
    def logpt(self):
        return t.sum(self.logp_elemwiset)

class Model(Context, Factor):
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
        self.model = self

    @property
    @memoize
    def logpt(self):
        """
        log-probability of the model

        Parameters
        ----------

        self : Model

        Returns
        -------

        logp : Theano scalar

        """
        factors = [var.logpt for var in self.basic_RVs] + self.potentials
        return t.add(*map(t.sum, factors))

    @property 
    def vars(self): 
        return self.free_RVs

    @property
    def basic_RVs(self):
        return (self.free_RVs + self.observed_RVs)

    @property 
    def unobserved_RVs(self): 
        return self.free_RVs + self.deterministics 


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
    def Var(self, name, dist, data=None):
        if data is None: 
            var = FreeRV(name=name, distribution=dist, model=self) 
            self.free_RVs.append(var)
        else: 
            var = ObservedRV(name=name, data=data, distribution=dist, model=self)
            self.observed_RVs.append(var)
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

    def __getitem__(self, key):
        return self.named_vars[key]

    @memoize
    def makefn(self, outs, mode=None):
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
        return function(self.vars, outs,
                     allow_input_downcast=True,
                     on_unused_input='ignore',
                     mode=mode)

    def fn(self, outs, mode=None):
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
        return LoosePointFunc(self.makefn(outs, mode))

    def fastfn(self, outs, mode=None):
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
        return FastPointFunc(self.makefn(outs, mode))

def fn(outs, mode=None, model=None):
    model = modelcontext(model)
    return model.fn(outs,mode)

def fastfn(outs, mode=None, model=None):
    model = modelcontext(model)
    return model.fastfn(outs,mode)


def Point(*args, **kwargs):

    """
    Build a point. Uses same args as dict() does.
    Filters out variables not in the model. All keys are strings.

    Parameters
    ----------
        *args, **kwargs
            arguments to build a dict
    """
    model = modelcontext(kwargs.pop('model', None))

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

class FastPointFunc(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, state):
        return self.f(**state)

class LoosePointFunc(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        point = Point(*args, **kwargs)
        return self.f(**point)


compilef = fastfn 


class FreeRV(Factor, TensorVariable):
    def __init__(self, type=None, owner=None, index=None, name=None, distribution=None, model=None):
        if type is None:
            type = distribution.type
        TensorVariable.__init__(self, type, owner, index, name)

        if distribution is not None:
            self.dshape = tuple(distribution.shape)
            self.dsize = int(np.prod(distribution.shape))
            self.distribution = distribution
            testval = distribution.default(distribution.testval)
            self.tag.test_value = np.ones(
                distribution.shape, distribution.dtype) * get_test_val(distribution, testval)
            self.logp_elemwiset = distribution.logp(self)
            self.model = model

class ObservedRV(Factor):
    def __init__(self, name, data, distribution, model):
        self.name = name
        data = getattr(data, 'values', data) #handle pandas
        args = as_iterargs(data)

        if len(args) > 1:
            params = getargspec(distribution.logp).args
            args = [t.constant(d, name=name + "_" + param) 
                    for d,param in zip(args,params) ]
        else: 
            args = [t.constant(args[0], name=name)]
            
        self.logp_elemwiset = distribution.logp(*args)
        self.model = model

def get_test_val(dist, val):
    try:
        val = getattr(dist, val)
    except TypeError:
        pass

    if hasattr(val, '__call__'):
        val = val(dist)

    if isinstance(val, TensorVariable):
        return val.tag.test_value
    else:
        return val


def as_iterargs(data):
    if isinstance(data, tuple):
        return data
    if hasattr(data, 'columns'):  # data frames
        return [np.asarray(data[c]) for c in data.columns]
    else:
        return [data]



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



# theano stuff
theano.config.warn.sum_div_dimshuffle_bug = False
theano.config.compute_test_value = 'raise'
