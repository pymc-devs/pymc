from point import *
from vartypes import *

from theano import theano, tensor as t, function
from theano.gof.graph import inputs

import numpy as np
from functools import wraps

__all__ = ['Model', 'compilef', 'gradient', 'hessian', 'modelcontext', 'Point']


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
        self.vars = []
        self.factors = []

    @property
    def logp(model):
        """
        log-probability of the model

        Parameters
        ----------

        model : Model

        Returns
        -------

        logp : Theano scalar

        """
        return t.add(*map(t.sum, model.factors))

    @property
    def logpc(model):
        """Compiled log probability density function"""
        return compilef(model.logp)

    def dlogpc(model, vars=None):
        """Compiled log probability density gradient function"""
        return compilef(gradient(model.logp, vars))

    def d2logpc(model, vars=None):
        """Compiled log probability density hessian function"""
        return compilef(hessian(model.logp, vars))

    @property
    def test_point(self):
        """Test point used to check that the model doesn't generate errors"""
        return Point(((var, var.tag.test_value) for var in self.vars), model=self)

    @property
    def cont_vars(model):
        """All the continuous variables in the model"""
        return typefilter(model.vars, continuous_types)

    """
    these functions add random variables
    """
    def Data(model, data, dist):
        if  hasattr(data, 'values'):
            # Incase obs is a Series or DataFrame
            data = data.values
        args = map(t.constant, as_iterargs(data))
        model.factors.append(dist.logp(*args))

    def Var(model, name, dist):
        var = dist.makevar(name)

        model.vars.append(var)
        model.factors.append(dist.logp(var))
        return var

    def TransformedVar(model, name, dist, trans):
        tvar = model.Var(trans.name + '_' + name, trans.apply(dist))

        return named(trans.backward(tvar), name), tvar

    def AddPotential(model, potential):
        model.factors.append(potential)


def Point(*args, **kwargs):
    """
    Build a point. Uses same args as dict() does.
    Filters out variables not in the model. All keys are strings.

    Parameters
    ----------
        *args, **kwargs
            arguments to build a dict
    """
    if 'model' in kwargs:
        model = kwargs['model']
        del kwargs['model']
    else:
        model = Model.get_context()

    d = dict(*args, **kwargs)
    varnames = map(str, model.vars)
    return dict((str(k), np.array(v))
                for (k, v) in d.iteritems()
                if str(k) in varnames)


def compilef(outs, mode=None):
    """
    Compiles a Theano function which returns `outs` and takes the variable ancestors of `outs` as inputs.

    Parameters
    ----------
    outs : Theano variable or iterable of Theano variables
    mode : Theano compilation mode

    Returns
    -------
    Compiled Theano function
    """
    return PointFunc(
        function(inputvars(outs), outs,
                 allow_input_downcast=True,
                 on_unused_input='ignore',
                 mode=mode)
    )


def named(var, name):
    var.name = name
    return var


def as_iterargs(data):
    if isinstance(data, tuple):
        return data
    if hasattr(data, 'columns'):  # data frames
        return [np.asarray(data[c]) for c in data.columns]
    else:
        return [data]


def makeiter(a):
    if isinstance(a, (tuple, list)):
        return a
    else:
        return [a]


def inputvars(a):
    return [v for v in inputs(makeiter(a)) if isinstance(v, t.TensorVariable)]

"""
Theano derivative functions
"""


def cont_inputs(f):
    return typefilter(inputvars(f), continuous_types)


def gradient1(f, v):
    """flat gradient of f wrt v"""
    return t.flatten(t.grad(f, v, disconnected_inputs='warn'))


def gradient(f, vars=None):
    if not vars:
        vars = cont_inputs(f)

    return t.concatenate([gradient1(f, v) for v in vars], axis=0)


def jacobian1(f, v):
    """jacobian of f wrt v"""
    f = t.flatten(f)
    idx = t.arange(f.shape[0])

    def grad_i(i):
        return gradient1(f[i], v)

    return theano.map(grad_i, idx)[0]


def jacobian(f, vars=None):
    if not vars:
        vars = cont_inputs(f)

    return t.concatenate([jacobian1(f, v) for v in vars], axis=1)


def hessian(f, vars=None):
    return -jacobian(gradient(f, vars), vars)


# theano stuff
theano.config.warn.sum_div_dimshuffle_bug = False
theano.config.compute_test_value = 'raise'
