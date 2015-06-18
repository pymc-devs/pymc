from .vartypes import *

from theano import theano, tensor as t, function
from theano.tensor.var import TensorVariable

import numpy as np
from functools import wraps
from .theanof import *
from inspect import getargspec

from .memoize import memoize

__all__ = ['Model', 'Factor', 'compilef', 'fn', 'fastfn', 'modelcontext', 'Point', 'Deterministic', 'Potential']


class Context(object):
    """Functionality for objects that put themselves in a context using the `with` statement."""
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
        """Return the deepest context on the stack."""
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            raise TypeError("No context on context stack")


def modelcontext(model):
    """return the given model or try to find it in the context if there was none supplied."""
    if model is None:
        return Model.get_context()
    return model

class Factor(object):
    """Common functionality for objects with a log probability density associated with them."""
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
        """Theano scalar of log-probability of the model"""
        return t.sum(self.logp_elemwiset)

class Model(Context, Factor):
    """Encapsulates the variables and likelihood factors of a model."""

    def __init__(self):
        self.named_vars = {}
        self.free_RVs = []
        self.observed_RVs = []
        self.deterministics = []
        self.potentials = []
        self.missing_values = []
        self.model = self

    @property
    @memoize
    def logpt(self):
        """Theano scalar of log-probability of the model"""
        factors = [var.logpt for var in self.basic_RVs] + self.potentials
        return t.add(*map(t.sum, factors))

    @property
    def vars(self):
        """List of unobserved random variables used as inputs to the model (which excludes deterministics)."""
        return self.free_RVs

    @property
    def basic_RVs(self):
        """List of random variables the model is defined in terms of (which excludes deterministics)."""
        return (self.free_RVs + self.observed_RVs)

    @property
    def unobserved_RVs(self):
        """List of all random variable, including deterministic ones."""
        return self.vars + self.deterministics


    @property
    def test_point(self):
        """Test point used to check that the model doesn't generate errors"""
        return Point(((var, var.tag.test_value) for var in self.vars),
                     model=self)

    @property
    def cont_vars(self):
        """All the continuous variables in the model"""
        return list(typefilter(self.vars, continuous_types))

    def Var(self, name, dist, data=None):
        """Create and add (un)observed random variable to the model with an appropriate prior distribution.

        Parameters
        ----------
            name : str
            dist : distribution for the random variable
            data : arraylike (optional)
               if data is provided, the variable is observed. If None, the variable is unobserved.
        Returns
        -------
            FreeRV or ObservedRV
        """
        if data is None:
            if getattr(dist, "transform", None) is None:
                var = FreeRV(name=name, distribution=dist, model=self)
                self.free_RVs.append(var)
            else:
                var = TransformedRV(name=name, distribution=dist, model=self, transform=dist.transform) 
                self.deterministics.append(var)
                return var
        elif isinstance(data, dict):
            var = MultiObservedRV(name=name, data=data, distribution=dist, model=self)
            self.observed_RVs.append(var)
            if var.missing_values:
                self.free_RVs += var.missing_values
                self.missing_values += var.missing_values
                for v in var.missing_values:
                    self.named_vars[v.name] = v
        else: 
            var = ObservedRV(name=name, data=data, distribution=dist, model=self)
            self.observed_RVs.append(var)
            if var.missing_values:
                self.free_RVs.append(var.missing_values)
                self.missing_values.append(var.missing_values)
                self.named_vars[var.missing_values.name] = var.missing_values

        self.add_random_variable(var)
        return var


    def add_random_variable(self, var):
        """Add a random variable to the named variables of the model."""
        self.named_vars[var.name] = var
        if not hasattr(self, var.name):
            setattr(self, var.name, var)

    def __getitem__(self, key):
        return self.named_vars[key]

    @memoize
    def makefn(self, outs, mode=None, *args, **kwargs):
        """Compiles a Theano function which returns `outs` and takes the variable
        ancestors of `outs` as inputs.

        Parameters
        ----------
        outs : Theano variable or iterable of Theano variables
        mode : Theano compilation mode

        Returns
        -------
        Compiled Theano function"""
        return function(self.vars, outs,
                     allow_input_downcast=True,
                     on_unused_input='ignore',
                     accept_inplace=True,
                     mode=mode, *args, **kwargs)

    def fn(self, outs, mode=None, *args, **kwargs):
        """Compiles a Theano function which returns the values of `outs` and takes values of model
        vars as arguments.

        Parameters
        ----------
        outs : Theano variable or iterable of Theano variables
        mode : Theano compilation mode

        Returns
        -------
        Compiled Theano function"""
        return LoosePointFunc(self.makefn(outs, mode, *args, **kwargs), self)

    def fastfn(self, outs, mode=None, *args, **kwargs):
        """Compiles a Theano function which returns `outs` and takes values of model
        vars as a dict as an argument.

        Parameters
        ----------
        outs : Theano variable or iterable of Theano variables
        mode : Theano compilation mode

        Returns
        -------
        Compiled Theano function as point function."""
        f = self.makefn(outs, mode, *args, **kwargs)
        return FastPointFunc(f)

    def profile(self, outs, n=1000, point=None, profile=True, *args, **kwargs):
        """Compiles and profiles a Theano function which returns `outs` and takes values of model
        vars as a dict as an argument.

        Parameters
        ----------
        outs : Theano variable or iterable of Theano variables
        n : int, default 1000
            Number of iterations to run
        point : point
            Point to pass to the function
        profile : True or ProfileStats
        *args, **kwargs 
            Compilation args

        Returns
        -------
        ProfileStats
            Use .summary() to print stats."""
        f = self.makefn(outs, profile=profile, *args, **kwargs)
        if point is None: 
            point = self.test_point

        for i in range(n):
            f(**point)

        return f.profile

        

def fn(outs, mode=None, model=None, *args, **kwargs):
    """Compiles a Theano function which returns the values of `outs` and takes values of model
    vars as arguments.

    Parameters
    ----------
    outs : Theano variable or iterable of Theano variables
    mode : Theano compilation mode

    Returns
    -------
    Compiled Theano function"""
    model = modelcontext(model)
    return model.fn(outs,mode, *args, **kwargs)

def fastfn(outs, mode=None, model=None):
    """Compiles a Theano function which returns `outs` and takes values of model
    vars as a dict as an argument.

    Parameters
    ----------
    outs : Theano variable or iterable of Theano variables
    mode : Theano compilation mode

    Returns
    -------
    Compiled Theano function as point function."""
    model = modelcontext(model)
    return model.fastfn(outs,mode)


def Point(*args, **kwargs):
    """Build a point. Uses same args as dict() does.
    Filters out variables not in the model. All keys are strings.

    Parameters
    ----------
        *args, **kwargs
            arguments to build a dict"""
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
    """Wraps so a function so it takes a dict of arguments instead of arguments."""
    def __init__(self, f):
        self.f = f

    def __call__(self, state):
        return self.f(**state)

class LoosePointFunc(object):
    """Wraps so a function so it takes a dict of arguments instead of arguments
    but can still take arguments."""
    def __init__(self, f, model):
        self.f = f
        self.model = model

    def __call__(self, *args, **kwargs):
        point = Point(model=self.model, *args, **kwargs)
        return self.f(**point)

compilef = fastfn


class FreeRV(Factor, TensorVariable):
    """Unobserved random variable that a model is specified in terms of."""
    def __init__(self, type=None, owner=None, index=None, name=None, distribution=None, model=None):
        """
        Parameters
        ----------

        type : theano type (optional)
        owner : theano owner (optional)

        name : str
        distribution : Distribution
        model : Model"""
        if type is None:
            type = distribution.type
        super(FreeRV, self).__init__(type, owner, index, name)

        if distribution is not None:
            self.dshape = tuple(distribution.shape)
            self.dsize = int(np.prod(distribution.shape))
            self.distribution = distribution
            self.tag.test_value = np.ones(
                distribution.shape, distribution.dtype) * distribution.default()
            self.logp_elemwiset = distribution.logp(self)
            self.model = model

def pandas_to_array(data):
    if hasattr(data, 'values'): #pandas
        if data.isnull().any().any(): #missing values
            return np.ma.MaskedArray(data.values, data.isnull().values)
        else: 
            return data.values
    elif hasattr(data, 'mask'):
        return data
    elif isinstance(data, theano.gof.graph.Variable):
        return data 
    else:
        return np.asarray(data)
        

def as_tensor(data, name,model, dtype):
    data = pandas_to_array(data).astype(dtype)

    if hasattr(data, 'mask'): 
        from .distributions import NoDistribution
        fakedist = NoDistribution.dist(shape=data.mask.sum(), dtype=dtype, testval=data.mean().astype(dtype))
        missing_values = FreeRV(name=name + '_missing', distribution=fakedist, model=model)

        constant = t.as_tensor_variable(data.filled())

        dataTensor = theano.tensor.set_subtensor(constant[data.mask.nonzero()], missing_values) 
        dataTensor.missing_values = missing_values
        return dataTensor
    else:
        data = t.as_tensor_variable(data, name=name)
        data.missing_values = None
        return data

class ObservedRV(Factor, TensorVariable):
    """Observed random variable that a model is specified in terms of.
    Potentially partially observed.
    """
    def __init__(self, type=None, owner=None, index=None, name=None, data=None, distribution=None, model=None):
        """
        Parameters
        ----------

        type : theano type (optional)
        owner : theano owner (optional)

        name : str
        distribution : Distribution
        model : Model
        """
        from .distributions import TensorType
        if type is None:
            data = pandas_to_array(data)
            type = TensorType(distribution.dtype, data.shape)

        super(TensorVariable, self).__init__(type, None, None, name)

        if distribution is not None:
            data = as_tensor(data, name,model,distribution.dtype) 
            self.missing_values = data.missing_values

            self.logp_elemwiset = distribution.logp(data)
            self.model = model
            self.distribution = distribution

            #make this RV a view on the combined missing/nonmissing array
            theano.gof.Apply(theano.compile.view_op, inputs=[data], outputs=[self])

            self.tag.test_value = theano.compile.view_op(data).tag.test_value

class MultiObservedRV(Factor):
    """Observed random variable that a model is specified in terms of.
    Potentially partially observed.
    """
    def __init__(self, name, data, distribution, model):
        """
        Parameters
        ----------

        type : theano type (optional)
        owner : theano owner (optional)

        name : str
        distribution : Distribution
        model : Model
        """
        self.name = name

        self.data = { name : as_tensor(data, name, model, distribution.dtype) for name, data in data.items()}

        self.missing_values = [ data.missing_values for data in self.data.values() if data.missing_values is not None]
        self.logp_elemwiset = distribution.logp(**self.data)
        self.model = model
        self.distribution = distribution

def Deterministic(name, var, model=None):
    """Create a named deterministic variable

    Parameters
    ----------
        name : str
        var : theano variables
    Returns
    -------
        n : var but with name name"""
    var.name = name
    modelcontext(model).deterministics.append(var)
    modelcontext(model).add_random_variable(var)
    return var

def Potential(name, var, model=None):
    """Add an arbitrary factor potential to the model likelihood

    Parameters
    ----------
        name : str
        var : theano variables
    Returns
    -------
        var : var, with name attribute
    """

    var.name = name
    modelcontext(model).potentials.append(var)
    return var

class TransformedRV(TensorVariable):
    def __init__(self, type=None, owner=None, index=None, name=None, distribution=None, model=None, transform=None):
        """
        Parameters
        ----------

        type : theano type (optional)
        owner : theano owner (optional)

        name : str
        distribution : Distribution
        model : Model"""
        if type is None:
            type = distribution.type
        super(TransformedRV, self).__init__(type, owner, index, name)

        if distribution is not None:
            self.model = model

            self.transformed = model.Var(name + "_" + transform.name, transform.apply(distribution))

            normalRV = transform.backward(self.transformed)

            theano.Apply(theano.compile.view_op, inputs=[normalRV], outputs=[self])
            self.tag.test_value = normalRV.tag.test_value



def as_iterargs(data):
    if isinstance(data, tuple):
        return data
    else:
        return [data]

# theano stuff
theano.config.warn.sum_div_dimshuffle_bug = False
theano.config.compute_test_value = 'raise'
