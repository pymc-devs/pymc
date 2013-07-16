from point import *
from vartypes import *

from theano import theano, tensor as t, function

import numpy as np
from functools import wraps
from theanof import *

from memoize import memoize

__all__ = ['Model', 'compilef', 'modelcontext', 'Point']


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
    @memoize
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
    try:
        d = dict(*args, **kwargs)
    except Exception as e:
        raise TypeError(
            "can't turn " + str(args) + " and " + str(kwargs) +
            " into a dict. " + str(e))

    varnames = map(str, model.vars)
    return dict((str(k), np.array(v))
                for (k, v) in d.iteritems()
                if str(k) in varnames)

@memoize
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
