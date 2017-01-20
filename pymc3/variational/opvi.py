import collections
import theano.tensor as tt
from theano.ifelse import ifelse
import theano
import pymc3 as pm
from ..theanof import tt_rng, memoize, change_flags
from ..blocking import ArrayOrdering
from ..distributions.dist_math import rho2sd, log_normal
from ..math import flatten_list
from ..model import modelcontext
import numpy as np

# helper class
FlatView = collections.namedtuple('FlatView', 'input, replacements')
# shortcut for zero grad
Z = theano.gradient.zero_grad


def flatten_model(model, vars=None):
    """Helper function for flattening model input"""
    if vars is None:
        vars = model.free_RVs
    order = ArrayOrdering(vars)
    inputvar = tt.vector('flat_view', dtype=theano.config.floatX)
    inputvar.tag.test_value = flatten_list(vars).tag.test_value
    replacements = {model.named_vars[name]: inputvar[slc].reshape(shape).astype(dtype)
                    for name, slc, shape, dtype in order.vmap}
    flat_view = FlatView(inputvar, replacements)
    view = {vm.var: vm for vm in order.vmap}
    return order, flat_view, view


class Operator(object):
    def __init__(self, model, approx):
        self.model = model
        self.approx = approx
        model = modelcontext(model)
        self.check_model(model)
        self.model = model
        self.order = approx.order
        self.flat_view = approx.flat_input
        self.view = approx.view

    def logp(self, z):
        p = theano.clone(self.model.logpt, self.flat_view.replacements)
        p = theano.clone(p, {self.input: z})
        return p

    def logq(self, z):
        return self.approx.logq(z)

    @staticmethod
    def check_model(model):
        """
        Checks that model is valid for variational inference
        """
        vars_ = [var for var in model.vars if not isinstance(var, pm.model.ObservedRV)]
        if any([var.dtype in pm.discrete_types for var in vars_]):
            raise ValueError('Model should not include discrete RVs')

    @property
    def input(self):
        """
        Shortcut to flattened input
        """
        return self.flat_view.input

    def apply(self, f):
        raise NotImplementedError

    def __call__(self, f):
        if not isinstance(f, TestFunction) and callable(f):
            f = TestFunction.from_function(f)
        else:
            raise ValueError('Need callable or TestFunction class, got %r' % f)
        return ObjectiveFunction(self, f)

    def obj(self, f):
        return lambda z: theano.clone(self.apply(f), {self.input: z})


class TestFunction(object):
    @property
    def params(self):
        return []

    def __call__(self, z):
        raise NotImplementedError

    @classmethod
    def from_function(cls, f):
        obj = TestFunction()
        obj.__call__ = f
        return obj


class ObjectiveFunction(object):
    def __init__(self, op, tf):
        self.obj = op.obj(tf)
        self.test_params = tf.params
        self.obj_params = op.approx.params

    def __call__(self, z):
        return self.obj(z)


class Approximation(object):
    def __init__(self, local_rv=None, model=None):
        if local_rv is None:
            local_rv = {}
        self.model = modelcontext(model)

        def get_transformed(v):
            if hasattr(v, 'transformed'):
                return v.transformed
            return v

        known = {get_transformed(k): v for k, v in local_rv.items()}
        self.known = known
        self.local_vars = [v for v in model.free_RVs if v in known]
        self.global_vars = [v for v in model.free_RVs if v not in known]
        self.order, self.flat_view, self.view = flatten_model(
            model=self.model,
            vars=self.local_vars + self.global_vars
        )

    def to_flat_input(self, node):
        """
        Replaces vars with flattened view stored in self.input
        """
        return theano.clone(node, self.flat_view.replacements, strict=False)

    @property
    def params(self):
        return []

    def random(self, size=None):
        raise NotImplementedError

    def logq(self):
        raise NotImplementedError
