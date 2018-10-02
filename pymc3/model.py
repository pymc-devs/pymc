import collections
import functools
import itertools
import threading
import re
import six
from copy import copy
try:
    from collections.abc import Hashable
except ImportError:
    from collections import Hashable

import numpy as np
import networkx
from pandas import Series
import scipy.sparse as sps
import theano.sparse as sparse
from theano import theano, tensor as tt
from theano.tensor.var import TensorVariable

from pymc3.theanof import set_theano_conf, floatX
import pymc3 as pm
from pymc3.math import flatten_list
from .memoize import memoize, WithMemoization
from .theanof import gradient, hessian, inputvars, generator
from .vartypes import typefilter, discrete_types, continuous_types, isgenerator
from .blocking import DictToArrayBijection, ArrayOrdering
from .util import get_transformed_name, WrapAsHashable

__all__ = [
    'Model', 'Factor', 'compilef', 'fn', 'fastfn', 'modelcontext',
    'Point', 'Deterministic', 'Potential'
]

FlatView = collections.namedtuple('FlatView', 'input, replacements, view')


class InstanceMethod(object):
    """Class for hiding references to instance methods so they can be pickled.

    >>> self.method = InstanceMethod(some_object, 'method_name')
    """

    def __init__(self, obj, method_name):
        self.obj = obj
        self.method_name = method_name

    def __call__(self, *args, **kwargs):
        return getattr(self.obj, self.method_name)(*args, **kwargs)


def incorporate_methods(source, destination, methods, default=None,
                        wrapper=None, override=False):
    """
    Add attributes to a destination object which points to
    methods from from a source object.

    Parameters
    ----------
    source : object
        The source object containing the methods.
    destination : object
        The destination object for the methods.
    methods : list of str
        Names of methods to incorporate.
    default : object
        The value used if the source does not have one of the listed methods.
    wrapper : function
        An optional function to allow the source method to be
        wrapped. Should take the form my_wrapper(source, method_name)
        and return a single value.
    override : bool
        If the destination object already has a method/attribute
        an AttributeError will be raised if override is False (the default).
    """
    for method in methods:
        if hasattr(destination, method) and not override:
            raise AttributeError("Cannot add method {!r}".format(method) +
                                 "to destination object as it already exists. "
                                 "To prevent this error set 'override=True'.")
        if hasattr(source, method):
            if wrapper is None:
                setattr(destination, method, getattr(source, method))
            else:
                setattr(destination, method, wrapper(source, method))
        else:
            setattr(destination, method, None)


class Context(object):
    """Functionality for objects that put themselves in a context using
    the `with` statement.
    """
    contexts = threading.local()

    def __enter__(self):
        type(self).get_contexts().append(self)
        # self._theano_config is set in Model.__new__
        if hasattr(self, '_theano_config'):
            self._old_theano_config = set_theano_conf(self._theano_config)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()
        # self._theano_config is set in Model.__new__
        if hasattr(self, '_old_theano_config'):
            set_theano_conf(self._old_theano_config)

    @classmethod
    def get_contexts(cls):
        # no race-condition here, cls.contexts is a thread-local object
        # be sure not to override contexts in a subclass however!
        if not hasattr(cls.contexts, 'stack'):
            cls.contexts.stack = []
        return cls.contexts.stack

    @classmethod
    def get_context(cls):
        """Return the deepest context on the stack."""
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            raise TypeError("No context on context stack")


def modelcontext(model):
    """return the given model or try to find it in the context if there was
    none supplied.
    """
    if model is None:
        return Model.get_context()
    return model


class Factor(object):
    """Common functionality for objects with a log probability density
    associated with them.
    """
    def __init__(self, *args, **kwargs):
        super(Factor, self).__init__(*args, **kwargs)

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
    def logp_nojac(self):
        return self.model.fn(self.logp_nojact)

    def dlogp_nojac(self, vars=None):
        """Compiled log density gradient function, without jacobian terms."""
        return self.model.fn(gradient(self.logp_nojact, vars))

    def d2logp_nojac(self, vars=None):
        """Compiled log density hessian function, without jacobian terms."""
        return self.model.fn(hessian(self.logp_nojact, vars))

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
    def fastlogp_nojac(self):
        return self.model.fastfn(self.logp_nojact)

    def fastdlogp_nojac(self, vars=None):
        """Compiled log density gradient function, without jacobian terms."""
        return self.model.fastfn(gradient(self.logp_nojact, vars))

    def fastd2logp_nojac(self, vars=None):
        """Compiled log density hessian function, without jacobian terms."""
        return self.model.fastfn(hessian(self.logp_nojact, vars))

    @property
    def logpt(self):
        """Theano scalar of log-probability of the model"""
        if getattr(self, 'total_size', None) is not None:
            logp = self.logp_sum_unscaledt * self.scaling
        else:
            logp = self.logp_sum_unscaledt
        if self.name is not None:
            logp.name = '__logp_%s' % self.name
        return logp

    @property
    def logp_nojact(self):
        """Theano scalar of log-probability, excluding jacobian terms."""
        if getattr(self, 'total_size', None) is not None:
            logp = tt.sum(self.logp_nojac_unscaledt) * self.scaling
        else:
            logp = tt.sum(self.logp_nojac_unscaledt)
        if self.name is not None:
            logp.name = '__logp_%s' % self.name
        return logp


class InitContextMeta(type):
    """Metaclass that executes `__init__` of instance in it's context"""
    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        with instance:  # appends context
            instance.__init__(*args, **kwargs)
        return instance


def withparent(meth):
    """Helper wrapper that passes calls to parent's instance"""
    def wrapped(self, *args, **kwargs):
        res = meth(self, *args, **kwargs)
        if getattr(self, 'parent', None) is not None:
            getattr(self.parent, meth.__name__)(*args, **kwargs)
        return res
    # Unfortunately functools wrapper fails
    # when decorating built-in methods so we
    # need to fix that improper behaviour
    wrapped.__name__ = meth.__name__
    return wrapped


class treelist(list):
    """A list that passes mutable extending operations used in Model
    to parent list instance.
    Extending treelist you will also extend its parent
    """
    def __init__(self, iterable=(), parent=None):
        super(treelist, self).__init__(iterable)
        assert isinstance(parent, list) or parent is None
        self.parent = parent
        if self.parent is not None:
            self.parent.extend(self)
    # typechecking here works bad
    append = withparent(list.append)
    __iadd__ = withparent(list.__iadd__)
    extend = withparent(list.extend)

    def tree_contains(self, item):
        if isinstance(self.parent, treedict):
            return (list.__contains__(self, item) or
                    self.parent.tree_contains(item))
        elif isinstance(self.parent, list):
            return (list.__contains__(self, item) or
                    self.parent.__contains__(item))
        else:
            return list.__contains__(self, item)

    def __setitem__(self, key, value):
        raise NotImplementedError('Method is removed as we are not'
                                  ' able to determine '
                                  'appropriate logic for it')

    def __imul__(self, other):
        t0 = len(self)
        list.__imul__(self, other)
        if self.parent is not None:
            self.parent.extend(self[t0:])


class treedict(dict):
    """A dict that passes mutable extending operations used in Model
    to parent dict instance.
    Extending treedict you will also extend its parent
    """
    def __init__(self, iterable=(), parent=None, **kwargs):
        super(treedict, self).__init__(iterable, **kwargs)
        assert isinstance(parent, dict) or parent is None
        self.parent = parent
        if self.parent is not None:
            self.parent.update(self)
    # typechecking here works bad
    __setitem__ = withparent(dict.__setitem__)
    update = withparent(dict.update)

    def tree_contains(self, item):
        # needed for `add_random_variable` method
        if isinstance(self.parent, treedict):
            return (dict.__contains__(self, item) or
                    self.parent.tree_contains(item))
        elif isinstance(self.parent, dict):
            return (dict.__contains__(self, item) or
                    self.parent.__contains__(item))
        else:
            return dict.__contains__(self, item)


class ValueGradFunction(object):
    """Create a theano function that computes a value and its gradient.

    Parameters
    ----------
    cost : theano variable
        The value that we compute with its gradient.
    grad_vars : list of named theano variables or None
        The arguments with respect to which the gradient is computed.
    extra_vars : list of named theano variables or None
        Other arguments of the function that are assumed constant. They
        are stored in shared variables and can be set using
        `set_extra_values`.
    dtype : str, default=theano.config.floatX
        The dtype of the arrays.
    casting : {'no', 'equiv', 'save', 'same_kind', 'unsafe'}, default='no'
        Casting rule for casting `grad_args` to the array dtype.
        See `numpy.can_cast` for a description of the options.
        Keep in mind that we cast the variables to the array *and*
        back from the array dtype to the variable dtype.
    kwargs
        Extra arguments are passed on to `theano.function`.

    Attributes
    ----------
    size : int
        The number of elements in the parameter array.
    profile : theano profiling object or None
        The profiling object of the theano function that computes value and
        gradient. This is None unless `profile=True` was set in the
        kwargs.
    """
    def __init__(self, cost, grad_vars, extra_vars=None, dtype=None,
                 casting='no', **kwargs):
        if extra_vars is None:
            extra_vars = []

        names = [arg.name for arg in grad_vars + extra_vars]
        if any(name is None for name in names):
            raise ValueError('Arguments must be named.')
        if len(set(names)) != len(names):
            raise ValueError('Names of the arguments are not unique.')

        if cost.ndim > 0:
            raise ValueError('Cost must be a scalar.')

        self._grad_vars = grad_vars
        self._extra_vars = extra_vars
        self._extra_var_names = set(var.name for var in extra_vars)
        self._cost = cost
        self._ordering = ArrayOrdering(grad_vars)
        self.size = self._ordering.size
        self._extra_are_set = False
        if dtype is None:
            dtype = theano.config.floatX
        self.dtype = dtype
        for var in self._grad_vars:
            if not np.can_cast(var.dtype, self.dtype, casting):
                raise TypeError('Invalid dtype for variable %s. Can not '
                                'cast to %s with casting rule %s.'
                                % (var.name, self.dtype, casting))
            if not np.issubdtype(var.dtype, np.floating):
                raise TypeError('Invalid dtype for variable %s. Must be '
                                'floating point but is %s.'
                                % (var.name, var.dtype))

        givens = []
        self._extra_vars_shared = {}
        for var in extra_vars:
            shared = theano.shared(var.tag.test_value, var.name + '_shared__')
            self._extra_vars_shared[var.name] = shared
            givens.append((var, shared))

        self._vars_joined, self._cost_joined = self._build_joined(
            self._cost, grad_vars, self._ordering.vmap)

        grad = tt.grad(self._cost_joined, self._vars_joined)
        grad.name = '__grad'

        inputs = [self._vars_joined]

        self._theano_function = theano.function(
            inputs, [self._cost_joined, grad], givens=givens, **kwargs)

    def set_extra_values(self, extra_vars):
        self._extra_are_set = True
        for var in self._extra_vars:
            self._extra_vars_shared[var.name].set_value(extra_vars[var.name])

    def get_extra_values(self):
        if not self._extra_are_set:
            raise ValueError('Extra values are not set.')

        return {var.name: self._extra_vars_shared[var.name].get_value()
                for var in self._extra_vars}

    def __call__(self, array, grad_out=None, extra_vars=None):
        if extra_vars is not None:
            self.set_extra_values(extra_vars)

        if not self._extra_are_set:
            raise ValueError('Extra values are not set.')

        if array.shape != (self.size,):
            raise ValueError('Invalid shape for array. Must be %s but is %s.'
                             % ((self.size,), array.shape))

        if grad_out is None:
            out = np.empty_like(array)
        else:
            out = grad_out

        logp, dlogp = self._theano_function(array)
        if grad_out is None:
            return logp, dlogp
        else:
            out[...] = dlogp
            return logp

    @property
    def profile(self):
        """Profiling information of the underlying theano function."""
        return self._theano_function.profile

    def dict_to_array(self, point):
        """Convert a dictionary with values for grad_vars to an array."""
        array = np.empty(self.size, dtype=self.dtype)
        for varmap in self._ordering.vmap:
            array[varmap.slc] = point[varmap.var].ravel().astype(self.dtype)
        return array

    def array_to_dict(self, array):
        """Convert an array to a dictionary containing the grad_vars."""
        if array.shape != (self.size,):
            raise ValueError('Array should have shape (%s,) but has %s'
                             % (self.size, array.shape))
        if array.dtype != self.dtype:
            raise ValueError('Array has invalid dtype. Should be %s but is %s'
                             % (self._dtype, self.dtype))
        point = {}
        for varmap in self._ordering.vmap:
            data = array[varmap.slc].reshape(varmap.shp)
            point[varmap.var] = data.astype(varmap.dtyp)

        return point

    def array_to_full_dict(self, array):
        """Convert an array to a dictionary with grad_vars and extra_vars."""
        point = self.array_to_dict(array)
        for name, var in self._extra_vars_shared.items():
            point[name] = var.get_value()
        return point

    def _build_joined(self, cost, args, vmap):
        args_joined = tt.vector('__args_joined')
        args_joined.tag.test_value = np.zeros(self.size, dtype=self.dtype)

        joined_slices = {}
        for vmap in vmap:
            sliced = args_joined[vmap.slc].reshape(vmap.shp)
            sliced.name = vmap.var
            joined_slices[vmap.var] = sliced

        replace = {var: joined_slices[var.name] for var in args}
        return args_joined, theano.clone(cost, replace=replace)


class Model(six.with_metaclass(InitContextMeta, Context, Factor, WithMemoization)):
    """Encapsulates the variables and likelihood factors of a model.

    Model class can be used for creating class based models. To create
    a class based model you should inherit from :class:`~.Model` and
    override :meth:`~.__init__` with arbitrary definitions (do not
    forget to call base class :meth:`__init__` first).

    Parameters
    ----------
    name : str
        name that will be used as prefix for names of all random
        variables defined within model
    model : Model
        instance of Model that is supposed to be a parent for the new
        instance. If ``None``, context will be used. All variables
        defined within instance will be passed to the parent instance.
        So that 'nested' model contributes to the variables and
        likelihood factors of parent model.
    theano_config : dict
        A dictionary of theano config values that should be set
        temporarily in the model context. See the documentation
        of theano for a complete list. Set config key
        ``compute_test_value`` to `raise` if it is None.

    Examples
    --------

    How to define a custom model

    .. code-block:: python

        class CustomModel(Model):
            # 1) override init
            def __init__(self, mean=0, sd=1, name='', model=None):
                # 2) call super's init first, passing model and name
                # to it name will be prefix for all variables here if
                # no name specified for model there will be no prefix
                super(CustomModel, self).__init__(name, model)
                # now you are in the context of instance,
                # `modelcontext` will return self you can define
                # variables in several ways note, that all variables
                # will get model's name prefix

                # 3) you can create variables with Var method
                self.Var('v1', Normal.dist(mu=mean, sd=sd))
                # this will create variable named like '{prefix_}v1'
                # and assign attribute 'v1' to instance created
                # variable can be accessed with self.v1 or self['v1']

                # 4) this syntax will also work as we are in the
                # context of instance itself, names are given as usual
                Normal('v2', mu=mean, sd=sd)

                # something more complex is allowed, too
                half_cauchy = HalfCauchy('sd', beta=10, testval=1.)
                Normal('v3', mu=mean, sd=half_cauchy)

                # Deterministic variables can be used in usual way
                Deterministic('v3_sq', self.v3 ** 2)

                # Potentials too
                Potential('p1', tt.constant(1))

        # After defining a class CustomModel you can use it in several
        # ways

        # I:
        #   state the model within a context
        with Model() as model:
            CustomModel()
            # arbitrary actions

        # II:
        #   use new class as entering point in context
        with CustomModel() as model:
            Normal('new_normal_var', mu=1, sd=0)

        # III:
        #   just get model instance with all that was defined in it
        model = CustomModel()

        # IV:
        #   use many custom models within one context
        with Model() as model:
            CustomModel(mean=1, name='first')
            CustomModel(mean=2, name='second')
    """
    def __new__(cls, *args, **kwargs):
        # resolves the parent instance
        instance = super(Model, cls).__new__(cls)
        if kwargs.get('model') is not None:
            instance._parent = kwargs.get('model')
        elif cls.get_contexts():
            instance._parent = cls.get_contexts()[-1]
        else:
            instance._parent = None
        theano_config = kwargs.get('theano_config', None)
        if theano_config is None or 'compute_test_value' not in theano_config:
            theano_config = {'compute_test_value': 'raise'}
        instance._theano_config = theano_config
        return instance

    def __init__(self, name='', model=None, theano_config=None):
        self.name = name
        self.dependence_dag = networkx.DiGraph()
        if self.parent is not None:
            self.named_vars = treedict(parent=self.parent.named_vars)
            self.free_RVs = treelist(parent=self.parent.free_RVs)
            self.observed_RVs = treelist(parent=self.parent.observed_RVs)
            self.deterministics = treelist(parent=self.parent.deterministics)
            self.potentials = treelist(parent=self.parent.potentials)
            self.missing_values = treelist(parent=self.parent.missing_values)
        else:
            self.named_vars = treedict()
            self.free_RVs = treelist()
            self.observed_RVs = treelist()
            self.deterministics = treelist()
            self.potentials = treelist()
            self.missing_values = treelist()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'dependence_dag' not in state:
            self.dependence_dag = build_dependence_dag_from_model(self)

    @property
    def model(self):
        return self

    @property
    def parent(self):
        return self._parent

    @property
    def root(self):
        model = self
        while not model.isroot:
            model = model.parent
        return model

    @property
    def isroot(self):
        return self.parent is None

    @property
    @memoize(bound=True)
    def bijection(self):
        vars = inputvars(self.cont_vars)

        bij = DictToArrayBijection(ArrayOrdering(vars),
                                   self.test_point)

        return bij

    @property
    def dict_to_array(self):
        return self.bijection.map

    @property
    def ndim(self):
        return sum(var.dsize for var in self.free_RVs)

    @property
    def logp_array(self):
        return self.bijection.mapf(self.fastlogp)

    @property
    def dlogp_array(self):
        vars = inputvars(self.cont_vars)
        return self.bijection.mapf(self.fastdlogp(vars))

    def logp_dlogp_function(self, grad_vars=None, **kwargs):
        if grad_vars is None:
            grad_vars = list(typefilter(self.free_RVs, continuous_types))
        else:
            for var in grad_vars:
                if var.dtype not in continuous_types:
                    raise ValueError("Can only compute the gradient of "
                                     "continuous types: %s" % var)
        varnames = [var.name for var in grad_vars]
        extra_vars = [var for var in self.free_RVs if var.name not in varnames]
        return ValueGradFunction(self.logpt, grad_vars, extra_vars, **kwargs)

    @property
    def logpt(self):
        """Theano scalar of log-probability of the model"""
        with self:
            factors = [var.logpt for var in self.basic_RVs] + self.potentials
            logp = tt.sum([tt.sum(factor) for factor in factors])
            if self.name:
                logp.name = '__logp_%s' % self.name
            else:
                logp.name = '__logp'
            return logp

    @property
    def logp_nojact(self):
        """Theano scalar of log-probability of the model"""
        with self:
            factors = [var.logp_nojact for var in self.basic_RVs] + self.potentials
            logp = tt.sum([tt.sum(factor) for factor in factors])
            if self.name:
                logp.name = '__logp_nojac_%s' % self.name
            else:
                logp.name = '__logp_nojac'
            return logp

    @property
    def varlogpt(self):
        """Theano scalar of log-probability of the unobserved random variables
           (excluding deterministic)."""
        with self:
            factors = [var.logpt for var in self.free_RVs]
            return tt.sum(factors)

    @property
    def datalogpt(self):
        with self:
            factors = [var.logpt for var in self.observed_RVs]
            factors += [tt.sum(factor) for factor in self.potentials]
            return tt.sum(factors)

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
    def test_point(self):
        """Test point used to check that the model doesn't generate errors"""
        return Point(((var, var.tag.test_value) for var in self.vars),
                     model=self)

    @property
    def disc_vars(self):
        """All the discrete variables in the model"""
        return list(typefilter(self.vars, discrete_types))

    @property
    def cont_vars(self):
        """All the continuous variables in the model"""
        return list(typefilter(self.vars, continuous_types))

    def Var(self, name, dist, data=None, total_size=None):
        """Create and add (un)observed random variable to the model with an
        appropriate prior distribution.

        Parameters
        ----------
        name : str
        dist : distribution for the random variable
        data : array_like (optional)
           If data is provided, the variable is observed. If None,
           the variable is unobserved.
        total_size : scalar
            upscales logp of variable with ``coef = total_size/var.shape[0]``

        Returns
        -------
        FreeRV or ObservedRV
        """
        name = self.name_for(name)
        if data is None:
            if getattr(dist, "transform", None) is None:
                with self:
                    var = FreeRV(name=name, distribution=dist,
                                 total_size=total_size, model=self)
                self.free_RVs.append(var)
            else:
                with self:
                    var = TransformedRV(name=name, distribution=dist,
                                        transform=dist.transform,
                                        total_size=total_size,
                                        model=self)
                pm._log.debug('Applied {transform}-transform to {name}'
                              ' and added transformed {orig_name} to model.'.format(
                                transform=dist.transform.name,
                                name=name,
                                orig_name=get_transformed_name(name, dist.transform))
                              )
                self.deterministics.append(var)
                self.add_random_variable(var)
                return var
        elif isinstance(data, dict):
            with self:
                var = MultiObservedRV(name=name, data=data, distribution=dist,
                                      total_size=total_size, model=self)
            self.observed_RVs.append(var)
            if var.missing_values:
                self.free_RVs += var.missing_values
                self.missing_values += var.missing_values
                for v in var.missing_values:
                    self.named_vars[v.name] = v
        else:
            with self:
                var = ObservedRV(name=name, data=data,
                                 distribution=dist,
                                 total_size=total_size, model=self)
            self.observed_RVs.append(var)
            if var.missing_values:
                self.free_RVs.append(var.missing_values)
                self.missing_values.append(var.missing_values)
                self.named_vars[var.missing_values.name] = var.missing_values

        self.add_random_variable(var)
        return var

    def add_random_variable(self, var, accept_cons_shared=False):
        """Add a random variable to the named variables of the model."""
        if self.named_vars.tree_contains(var.name):
            raise ValueError(
                "Variable name {} already exists.".format(var.name))
        self.named_vars[var.name] = var
        if not hasattr(self, self.name_of(var.name)):
            setattr(self, self.name_of(var.name), var)
        # The model should automatically construct a DependenceDAG instance
        # that encodes the relations between its variables
        add_to_dependence_dag(self.dependence_dag,
                              var,
                              accept_cons_shared=accept_cons_shared)

    @property
    def prefix(self):
        return '%s_' % self.name if self.name else ''

    def name_for(self, name):
        """Checks if name has prefix and adds if needed
        """
        if self.prefix:
            if not name.startswith(self.prefix):
                return '{}{}'.format(self.prefix, name)
            else:
                return name
        else:
            return name

    def name_of(self, name):
        """Checks if name has prefix and deletes if needed
        """
        if not self.prefix or not name:
            return name
        elif name.startswith(self.prefix):
            return name[len(self.prefix):]
        else:
            return name

    def __getitem__(self, key):
        try:
            return self.named_vars[key]
        except KeyError as e:
            try:
                return self.named_vars[self.name_for(key)]
            except KeyError:
                raise e

    def makefn(self, outs, mode=None, *args, **kwargs):
        """Compiles a Theano function which returns ``outs`` and takes the variable
        ancestors of ``outs`` as inputs.

        Parameters
        ----------
        outs : Theano variable or iterable of Theano variables
        mode : Theano compilation mode

        Returns
        -------
        Compiled Theano function
        """
        with self:
            return theano.function(self.vars, outs,
                                   allow_input_downcast=True,
                                   on_unused_input='ignore',
                                   accept_inplace=True,
                                   mode=mode, *args, **kwargs)

    def fn(self, outs, mode=None, *args, **kwargs):
        """Compiles a Theano function which returns the values of ``outs``
        and takes values of model vars as arguments.

        Parameters
        ----------
        outs : Theano variable or iterable of Theano variables
        mode : Theano compilation mode

        Returns
        -------
        Compiled Theano function
        """
        return LoosePointFunc(self.makefn(outs, mode, *args, **kwargs), self)

    def fastfn(self, outs, mode=None, *args, **kwargs):
        """Compiles a Theano function which returns ``outs`` and takes values
        of model vars as a dict as an argument.

        Parameters
        ----------
        outs : Theano variable or iterable of Theano variables
        mode : Theano compilation mode

        Returns
        -------
        Compiled Theano function as point function.
        """
        f = self.makefn(outs, mode, *args, **kwargs)
        return FastPointFunc(f)

    def profile(self, outs, n=1000, point=None, profile=True, *args, **kwargs):
        """Compiles and profiles a Theano function which returns ``outs`` and
        takes values of model vars as a dict as an argument.

        Parameters
        ----------
        outs : Theano variable or iterable of Theano variables
        n : int, default 1000
            Number of iterations to run
        point : point
            Point to pass to the function
        profile : True or ProfileStats
        args, kwargs
            Compilation args

        Returns
        -------
        ProfileStats
            Use .summary() to print stats.
        """
        f = self.makefn(outs, profile=profile, *args, **kwargs)
        if point is None:
            point = self.test_point

        for _ in range(n):
            f(**point)

        return f.profile

    def flatten(self, vars=None, order=None, inputvar=None):
        """Flattens model's input and returns:

        FlatView with
            * input vector variable
            * replacements ``input_var -> vars``
            * view `{variable: VarMap}`

        Parameters
        ----------
        vars : list of variables or None
            if None, then all model.free_RVs are used for flattening input
        order : ArrayOrdering
            Optional, use predefined ordering
        inputvar : tt.vector
            Optional, use predefined inputvar

        Returns
        -------
        flat_view
        """
        if vars is None:
            vars = self.free_RVs
        if order is None:
            order = ArrayOrdering(vars)
        if inputvar is None:
            inputvar = tt.vector('flat_view', dtype=theano.config.floatX)
            if theano.config.compute_test_value != 'off':
                if vars:
                    inputvar.tag.test_value = flatten_list(vars).tag.test_value
                else:
                    inputvar.tag.test_value = np.asarray([], inputvar.dtype)
        replacements = {self.named_vars[name]: (inputvar[slc].reshape(shape).astype(dtype))
                        for name, slc, shape, dtype in order.vmap}
        view = {vm.var: vm for vm in order.vmap}
        flat_view = FlatView(inputvar, replacements, view)
        return flat_view

    def check_test_point(self, test_point=None, round_vals=2):
        """Checks log probability of test_point for all random variables in the model.

        Parameters
        ----------
        test_point : Point
            Point to be evaluated.
            if None, then all model.test_point is used
        round_vals : int
            Number of decimals to round log-probabilities

        Returns
        -------
        Pandas Series
        """
        if test_point is None:
            test_point = self.test_point

        return Series({RV.name: np.round(RV.logp(self.test_point), round_vals) for RV in self.basic_RVs},
                      name='Log-probability of test_point')

    def _repr_latex_(self, name=None, dist=None):
        tex_vars = []
        for rv in itertools.chain(self.unobserved_RVs, self.observed_RVs):
            rv_tex = rv.__latex__()
            if rv_tex is not None:
                array_rv = rv_tex.replace(r'\sim', r'&\sim &').strip('$')
                tex_vars.append(array_rv)
        return r'''$$
            \begin{{array}}{{rcl}}
            {}
            \end{{array}}
            $$'''.format('\\\\'.join(tex_vars))

    __latex__ = _repr_latex_


def fn(outs, mode=None, model=None, *args, **kwargs):
    """Compiles a Theano function which returns the values of ``outs`` and
    takes values of model vars as arguments.

    Parameters
    ----------
    outs : Theano variable or iterable of Theano variables
    mode : Theano compilation mode

    Returns
    -------
    Compiled Theano function
    """
    model = modelcontext(model)
    return model.fn(outs, mode, *args, **kwargs)


def fastfn(outs, mode=None, model=None):
    """Compiles a Theano function which returns ``outs`` and takes values of model
    vars as a dict as an argument.

    Parameters
    ----------
    outs : Theano variable or iterable of Theano variables
    mode : Theano compilation mode

    Returns
    -------
    Compiled Theano function as point function.
    """
    model = modelcontext(model)
    return model.fastfn(outs, mode)


def Point(*args, **kwargs):
    """Build a point. Uses same args as dict() does.
    Filters out variables not in the model. All keys are strings.

    Parameters
    ----------
    args, kwargs
        arguments to build a dict
    """
    model = modelcontext(kwargs.pop('model', None))
    args = list(args)
    try:
        d = dict(*args, **kwargs)
    except Exception as e:
        raise TypeError(
            "can't turn {} and {} into a dict. {}".format(args, kwargs, e))
    return dict((str(k), np.array(v)) for k, v in d.items()
                if str(k) in map(str, model.vars))


class FastPointFunc(object):
    """Wraps so a function so it takes a dict of arguments instead of arguments.
    """

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


def _get_scaling(total_size, shape, ndim):
    """
    Gets scaling constant for logp

    Parameters
    ----------
    total_size : int or list[int]
    shape : shape
        shape to scale
    ndim : int
        ndim hint

    Returns
    -------
    scalar
    """
    if total_size is None:
        coef = floatX(1)
    elif isinstance(total_size, int):
        if ndim >= 1:
            denom = shape[0]
        else:
            denom = 1
        coef = floatX(total_size) / floatX(denom)
    elif isinstance(total_size, (list, tuple)):
        if not all(isinstance(i, int) for i in total_size if (i is not Ellipsis and i is not None)):
            raise TypeError('Unrecognized `total_size` type, expected '
                            'int or list of ints, got %r' % total_size)
        if Ellipsis in total_size:
            sep = total_size.index(Ellipsis)
            begin = total_size[:sep]
            end = total_size[sep+1:]
            if Ellipsis in end:
                raise ValueError('Double Ellipsis in `total_size` is restricted, got %r' % total_size)
        else:
            begin = total_size
            end = []
        if (len(begin) + len(end)) > ndim:
            raise ValueError('Length of `total_size` is too big, '
                             'number of scalings is bigger that ndim, got %r' % total_size)
        elif (len(begin) + len(end)) == 0:
            return floatX(1)
        if len(end) > 0:
            shp_end = shape[-len(end):]
        else:
            shp_end = np.asarray([])
        shp_begin = shape[:len(begin)]
        begin_coef = [floatX(t) / shp_begin[i] for i, t in enumerate(begin) if t is not None]
        end_coef = [floatX(t) / shp_end[i] for i, t in enumerate(end) if t is not None]
        coefs = begin_coef + end_coef
        coef = tt.prod(coefs)
    else:
        raise TypeError('Unrecognized `total_size` type, expected '
                        'int or list of ints, got %r' % total_size)
    return tt.as_tensor(floatX(coef))


class FreeRV(Factor, TensorVariable):
    """Unobserved random variable that a model is specified in terms of."""

    def __init__(self, type=None, owner=None, index=None, name=None,
                 distribution=None, total_size=None, model=None):
        """
        Parameters
        ----------
        type : theano type (optional)
        owner : theano owner (optional)
        name : str
        distribution : Distribution
        model : Model
        total_size : scalar Tensor (optional)
            needed for upscaling logp
        """
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
            # The logp might need scaling in minibatches.
            # This is done in `Factor`.
            self.logp_sum_unscaledt = distribution.logp_sum(self)
            self.logp_nojac_unscaledt = distribution.logp_nojac(self)
            self.total_size = total_size
            self.model = model
            self.scaling = _get_scaling(total_size, self.shape, self.ndim)

            incorporate_methods(source=distribution, destination=self,
                                methods=['random'],
                                wrapper=InstanceMethod)

    def _repr_latex_(self, name=None, dist=None):
        if self.distribution is None:
            return None
        if name is None:
            name = self.name
        if dist is None:
            dist = self.distribution
        return self.distribution._repr_latex_(name=name, dist=dist)

    __latex__ = _repr_latex_

    @property
    def init_value(self):
        """Convenience attribute to return tag.test_value"""
        return self.tag.test_value


def pandas_to_array(data):
    if hasattr(data, 'values'):  # pandas
        if data.isnull().any().any():  # missing values
            ret = np.ma.MaskedArray(data.values, data.isnull().values)
        else:
            ret = data.values
    elif hasattr(data, 'mask'):
        ret = data
    elif isinstance(data, theano.gof.graph.Variable):
        ret = data
    elif sps.issparse(data):
        ret = data
    elif isgenerator(data):
        ret = generator(data)
    else:
        ret = np.asarray(data)
    return pm.floatX(ret)


def as_tensor(data, name, model, distribution):
    dtype = distribution.dtype
    data = pandas_to_array(data).astype(dtype)

    if hasattr(data, 'mask'):
        from .distributions import NoDistribution
        testval = np.broadcast_to(distribution.default(), data.shape)[data.mask]
        fakedist = NoDistribution.dist(shape=data.mask.sum(), dtype=dtype,
                                       testval=testval, parent_dist=distribution)
        missing_values = FreeRV(name=name + '_missing', distribution=fakedist,
                                model=model)
        constant = tt.as_tensor_variable(data.filled())

        dataTensor = tt.set_subtensor(
            constant[data.mask.nonzero()], missing_values)
        dataTensor.missing_values = missing_values
        return dataTensor
    elif sps.issparse(data):
        data = sparse.basic.as_sparse(data, name=name)
        data.missing_values = None
        return data
    else:
        data = tt.as_tensor_variable(data, name=name)
        data.missing_values = None
        return data


class ObservedRV(Factor, TensorVariable):
    """Observed random variable that a model is specified in terms of.
    Potentially partially observed.
    """

    def __init__(self, type=None, owner=None, index=None, name=None, data=None,
                 distribution=None, total_size=None, model=None):
        """
        Parameters
        ----------
        type : theano type (optional)
        owner : theano owner (optional)
        name : str
        distribution : Distribution
        model : Model
        total_size : scalar Tensor (optional)
            needed for upscaling logp
        """
        from .distributions import TensorType

        if hasattr(data, 'type') and isinstance(data.type, tt.TensorType):
            type = data.type

        if type is None:
            data = pandas_to_array(data)
            type = TensorType(distribution.dtype, data.shape)

        self.observations = data

        super(ObservedRV, self).__init__(type, owner, index, name)

        if distribution is not None:
            data = as_tensor(data, name, model, distribution)

            self.missing_values = data.missing_values
            self.logp_elemwiset = distribution.logp(data)
            # The logp might need scaling in minibatches.
            # This is done in `Factor`.
            self.logp_sum_unscaledt = distribution.logp_sum(data)
            self.logp_nojac_unscaledt = distribution.logp_nojac(data)
            self.total_size = total_size
            self.model = model
            self.distribution = distribution

            # make this RV a view on the combined missing/nonmissing array
            theano.gof.Apply(theano.compile.view_op,
                             inputs=[data], outputs=[self])
            self.tag.test_value = theano.compile.view_op(data).tag.test_value
            self.scaling = _get_scaling(total_size, data.shape, data.ndim)

    def _repr_latex_(self, name=None, dist=None):
        if self.distribution is None:
            return None
        if name is None:
            name = self.name
        if dist is None:
            dist = self.distribution
        return self.distribution._repr_latex_(name=name, dist=dist)

    __latex__ = _repr_latex_

    @property
    def init_value(self):
        """Convenience attribute to return tag.test_value"""
        return self.tag.test_value


class MultiObservedRV(Factor):
    """Observed random variable that a model is specified in terms of.
    Potentially partially observed.
    """

    def __init__(self, name, data, distribution, total_size=None, model=None):
        """
        Parameters
        ----------
        type : theano type (optional)
        owner : theano owner (optional)
        name : str
        distribution : Distribution
        model : Model
        total_size : scalar Tensor (optional)
            needed for upscaling logp
        """
        self.name = name
        self.data = {name: as_tensor(data, name, model, distribution)
                     for name, data in data.items()}

        self.missing_values = [datum.missing_values for datum in self.data.values()
                               if datum.missing_values is not None]
        self.logp_elemwiset = distribution.logp(**self.data)
        # The logp might need scaling in minibatches.
        # This is done in `Factor`.
        self.logp_sum_unscaledt = distribution.logp_sum(**self.data)
        self.logp_nojac_unscaledt = distribution.logp_nojac(**self.data)
        self.total_size = total_size
        self.model = model
        self.distribution = distribution
        self.scaling = _get_scaling(total_size, self.logp_elemwiset.shape, self.logp_elemwiset.ndim)


def _walk_up_rv(rv):
    """Walk up theano graph to get inputs for deterministic RV."""
    all_rvs = []
    parents = list(itertools.chain(*[j.inputs for j in rv.get_parents()]))
    if parents:
        for parent in parents:
            all_rvs.extend(_walk_up_rv(parent))
    else:
        if rv.name:
            all_rvs.append(r'\text{%s}' % rv.name)
        else:
            all_rvs.append(r'\text{Constant}')
    return all_rvs


def _latex_repr_rv(rv):
    """Make latex string for a Deterministic variable"""
    return (r'$\text{%s} \sim \text{Deterministic}(%s)$' % (rv.name, r',~'.join(_walk_up_rv(rv))))


def Deterministic(name, var, model=None):
    """Create a named deterministic variable

    Parameters
    ----------
    name : str
    var : theano variables

    Returns
    -------
    var : var, with name attribute
    """
    model = modelcontext(model)
    var = var.copy(model.name_for(name))
    model.deterministics.append(var)
    model.add_random_variable(var, accept_cons_shared=True)
    var._repr_latex_ = functools.partial(_latex_repr_rv, var)
    var.__latex__ = var._repr_latex_
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
    model = modelcontext(model)
    var.name = model.name_for(name)
    model.potentials.append(var)
    model.add_random_variable(var, accept_cons_shared=True)
    return var


class TransformedRV(TensorVariable):
    """
    Parameters
    ----------

    type : theano type (optional)
    owner : theano owner (optional)
    name : str
    distribution : Distribution
    model : Model
    total_size : scalar Tensor (optional)
        needed for upscaling logp
    """

    def __init__(self, type=None, owner=None, index=None, name=None,
                 distribution=None, model=None, transform=None,
                 total_size=None):
        if type is None:
            type = distribution.type
        super(TransformedRV, self).__init__(type, owner, index, name)

        self.transformation = transform

        if distribution is not None:
            self.model = model
            self.distribution = distribution
            self.dshape = tuple(distribution.shape)
            self.dsize = int(np.prod(distribution.shape))

            transformed_name = get_transformed_name(name, transform)

            self.transformed = model.Var(
                transformed_name, transform.apply(distribution), total_size=total_size)

            normalRV = transform.backward(self.transformed)

            theano.Apply(theano.compile.view_op, inputs=[
                         normalRV], outputs=[self])
            self.tag.test_value = normalRV.tag.test_value
            self.scaling = _get_scaling(total_size, self.shape, self.ndim)
            incorporate_methods(source=distribution, destination=self,
                                methods=['random'],
                                wrapper=InstanceMethod)

    def _repr_latex_(self, name=None, dist=None):
        if self.distribution is None:
            return None
        if name is None:
            name = self.name
        if dist is None:
            dist = self.distribution
        return self.distribution._repr_latex_(name=name, dist=dist)

    __latex__ = _repr_latex_

    @property
    def init_value(self):
        """Convenience attribute to return tag.test_value"""
        return self.tag.test_value


def as_iterargs(data):
    if isinstance(data, tuple):
        return data
    else:
        return [data]


def all_continuous(vars):
    """Check that vars not include discrete variables, excepting
    ObservedRVs.  """
    vars_ = [var for var in vars if not isinstance(var, pm.model.ObservedRV)]
    if any([var.dtype in pm.discrete_types for var in vars_]):
        return False
    else:
        return True


class ConstantNodeException(Exception):
    pass


def not_shared_or_constant_variable(x):
    return (isinstance(x, theano.Variable) and
            not(isinstance(x, theano.Constant) or
                isinstance(x, theano.tensor.sharedvar.SharedVariable))
            ) or (isinstance(x, (FreeRV, MultiObservedRV, TransformedRV)))


_theano_autonamed_ops = [('.+\.T$',
                          theano.tensor.elemwise.DimShuffle,
                          None),
                         ('max$',
                          theano.tensor.basic.MaxAndArgmax,
                          None),
                         ('argmax$',
                          theano.tensor.basic.MaxAndArgmax,
                          None),
                         ('mean$',
                          theano.tensor.elemwise.Elemwise,
                          'Elemwise{true_div,no_inplace}'),
                         ('var$',
                          theano.tensor.elemwise.Elemwise,
                          'Elemwise{true_div,no_inplace}'),
                         ('std$',
                          theano.tensor.elemwise.Elemwise,
                          'Elemwise{sqrt,no_inplace}'),
                         ]


def get_first_level_conditionals(root):
    """Performs a breadth first search on the supplied root node's logpt or
    transformed logpt graph searching for named input nodes, which are
    different from the supplied root. Each explored branch will stop when
    either when it ends or when it finds its first named node.

    Parameters
    ----------
    root: theano.Variable (mandatory)
        The node from which to get the transformed.logpt or logpt and perform
        the search. If root does not have either of these attributes, the
        function returns None.

    Returns
    -------
    conditional_on : set, with named nodes that are not theano.Constant nor
    SharedVariable. The input `root` is conditionally dependent on these nodes
    and is one step away from them in the bayesian network that specifies the
    relationships, hence the name `get_first_level_conditionals`.
    """
    transformed = getattr(root, 'transformed', None)
    try:
        cond = transformed.logpt
    except AttributeError:
        cond = getattr(root, 'logpt', None)
    if cond is None:
        return None
    conditional_on = set()
    queue = copy(getattr(cond.owner, 'inputs', []))
    while queue:
        parent = queue.pop(0)
        if (parent is not None and getattr(parent, 'name', None) is not None
                and not_shared_or_constant_variable(parent)):
            # We don't include as a conditional relation either logpt depending
            # on root or on transformed because they are both deterministic
            # relations
            if parent == root or parent == transformed:
                continue
            # Some theano ops place default names to their associated output
            # nodes. We want to ignore them, so we test that the name pattern
            # matches with the ones we don't want, that the op is an instance
            # of the conflicting op, and, in the case of mean, var, and std,
            # where the op is built from other ops, that the op.name matches
            # the default one.
            must_ignore = False
            for name_pattern, owner_type, owner_name in _theano_autonamed_ops:
                if re.match(name_pattern, parent.name):
                    owner = getattr(parent, 'owner', None)
                    if owner is not None and isinstance(owner.op, owner_type):
                        if (owner_name is None or
                                getattr(owner.op, 'name', None) == owner_name):
                            must_ignore = True
                            break
            if must_ignore:
                continue
            conditional_on.add(parent)
        else:
            parent_owner = getattr(parent, 'owner', None)
            queue.extend(getattr(parent_owner, 'inputs', []))
    if not conditional_on:
        return None
    return conditional_on


def add_to_dependence_dag(dag, node, return_added_node=False, force=False,
                          accept_cons_shared=False, inplace=True):
    """Add a node and its conditional and deterministic parents along with
    their relations, recursively into the `networkx.DiGraph` instance.

    Parameters
    ----------
    dag: networkx.DiGraph instance (mandatory)
        The Digraph instance on which to add the node. By default, the addition
        is done inplace, but if `inplace=False` is supplied, it is performed
        on a copy of the graph.
    node: The variable to add to the DAG (mandatory)
        By default `theano.Variable`'s, which could be a pymc random
        variable, Deterministic or Potential, are allowed. TensorConstants
        and SharedVariables are not allowed by default, but this behavior
        can be changed with either the `force` or `accept_cons_shared`
        inputs.
        Other unhashable types are only accepted if `force=True`, and they
        are wrapped by `WrapAsHashable` instances.
    accept_cons_shared: bool (optional)
        If True, `theano` `TensorConstant`s and `theano` `SharedVariable`s
        are allowed to be added to the DAG. These are treated separately
        a priori because `_draw_value` handles these cases differently.
    force: bool (optional)
        If True, any type of node, except None, is allowed to be added.
    return_added_node: bool (optional)
        If True, the node which was added to the DAG is returned along with
        the `DiGraph` instance. This may be useful because the added
        node may be a `WrapAsHashable` instance which wraps to inputed node
        depending on its type.

    Returns
    -------
    The `DiGraph` instance to which the node was added.
    If `return_added_node` is `True`, the `DiGraph` instance is
    packed into a tuple along with the node that was actually added into
    the DAG. Usually this node is the input node, but depending on the
    node's type, it could be a `WrapAsHashable` instance.
    """
    if node is None:
        raise TypeError('None is not allowed to be added as a node in '
                        'variable dependency graph.')
    if not isinstance(dag, networkx.DiGraph):
        raise TypeError('Input `dag` must be an instance of networkx.DiGraph. '
                        'Got {}, which has type {}, instead.'.
                        format(dag, type(dag)))
    if not inplace:
        dag = dag.copy()
    if not isinstance(node, (theano.Variable, MultiObservedRV)):
        if not force:
            raise TypeError(
                "By default, it is not allowed to add nodes that "
                "are not `theano.Variable`'s nor pymc3 random variates "
                "to a variable dependence graph."
                "Got node `{}` of type `{}`. "
                "However, this kind of node could be added by "
                "passing `force=True` to `add`. It would be "
                "wrapped by a `WrapAsHashable` instead. This "
                "wrapped node can be returned by `add` by passing "
                "`return_added_node=True`.".
                format(node, type(node)))
        node = WrapAsHashable(node)
    elif not (not_shared_or_constant_variable(node) or
              hasattr(node, 'distribution')):
        if not (force or accept_cons_shared):
            raise ConstantNodeException(
                'Supplied node, of type `{}`, does not have a '
                '`distribution` attribute or is an instance of a `theano` '
                '`Constant` or `SharedVariable`. This node could be '
                'accepted by passing either `force=True` or '
                '`accept_cons_shared=True` to `add`.'.
                format(type(node)))
    if not isinstance(node, Hashable):
        node = WrapAsHashable(node)
    if node in dag:
        # Should we raise a warning with we attempt to add a node that is
        # already in the DAG??
        if return_added_node:
            return dag, node
        else:
            return dag

    # Add node into the nodes set and then initiate all node relations and
    # values to their defaults
    dag.add_node(node)

    # Try to get the conditional parents of node and add them
    cond = get_first_level_conditionals(node)
    if cond is not None:
        for conditional_parent in walk_down_ownership(cond):
            if conditional_parent not in dag:
                try:
                    add_to_dependence_dag(dag, conditional_parent)
                except ConstantNodeException:
                    continue
            dag.add_edge(conditional_parent,
                         node,
                         conditional=True,
                         deterministic=False)

    # Try to get the deterministic parents of node and add them
    if not_shared_or_constant_variable(node):
        for deterministic_parent in walk_down_ownership([node], ignore=True):
            if deterministic_parent not in dag:
                try:
                    add_to_dependence_dag(dag, deterministic_parent)
                except ConstantNodeException:
                    continue
            dag.add_edge(deterministic_parent,
                         node,
                         conditional=False,
                         deterministic=True)
    if not networkx.is_directed_acyclic_graph(dag):
        raise RuntimeError('The dependence graph is no longer a directed '
                           'acyclic graph (DAG). The addition of node `{}`, '
                           'of type `{}`, and the edges from its predecessors '
                           'introduced a loop inside the variable dependence '
                           'graph. Consider raising an issue with developers.'.
                           format(node, type(node)))

    if return_added_node:
        return dag, node
    return dag


def walk_down_ownership(node_list, ignore=False):
    """This function goes through an iterable of nodes provided in
    `node_list`, yielding the non None named nodes in the ownership graph.
    With the optional input `ignore`, a node without a name can be yielded.
    """
    for node in node_list:
        if hasattr(node, 'name') and node.name is not None and not ignore:
            yield node
        elif not_shared_or_constant_variable(node):
            owner = getattr(node, 'owner', None)
            if owner is not None:
                for parent in walk_down_ownership(owner.inputs):
                    yield parent


def get_sub_dag(dag, input_nodes, force=True, return_index=False):
    """Get a new DiGraph instance which is like a right outer join
    of `dag` with a list of input nodes provided in `input_nodes`.
    What this means is that it will look for the `input_nodes` inside
    `dag`, the nodes which are contained in `dag`, along with all their
    predecessors will be used to get a subgraph of `dag`. Then, the
    remaining nodes will be added to the `DiGraph` with add_to_dependence_dag.
    Finally, this instance is returned. In summary, it copies the shared
    part of `dag`, given the nodes in `input_nodes`, and then adds onto
    that.

    Parameters
    ----------
    dag: DiGraph instance (mandatory)
        Dependence DAG DiGraph on which to perform the right outer join
        operation.
    input_nodes: list or scalar (mandatory)
        If it is a scalar `input_nodes` will be converted to a list as
        `[input_nodes]`. `input_nodes` is a list of nodes that will be
        used to create a new `DependenceDAG` instance. The part of the DAG
        that is shared with `self`, will be copied, and the rest will be
        added.
    force: bool (optional)
        If True, the nodes that must be added, will be added with the
        force flag set to True. [Default is True]
    return_index: bool (optional)
        If True, this function will also return a dictionary of indices
        to nodes `{index: node}`. Each key will be the position of the
        node provided in `input_nodes` and the value will be the node
        that was added to the `dag`, which could either be a
        `WrapAsHashable` instance or `input_nodes[index]` itself.

    Returns
    -------
    A new `DiGraph` instance that results from the right outer join
    operation.
    If `return_index` is `True`, the returned `DiGraph` instance is
    packed into a tuple along with the dictionary of indices to nodes
    `{index: node}`. Each key will be the position of the node provided in
    `input_nodes` and the value will be the node that was added to the
    returned `DiGraph` instance, which could either be a
    `WrapAsHashable` instance or `input_nodes[index]` itself.
    """
    if not isinstance(dag, networkx.DiGraph):
        raise TypeError('Input `dag` must be an instance of networkx.DiGraph. '
                        'Got {}, which has type {}, instead.'.
                        format(dag, type(dag)))
    if not isinstance(input_nodes, list):
        input_nodes = [input_nodes]
    index = {}
    copied_bunch = set()
    nodes_to_add = []
    for i, node in enumerate(input_nodes):
        if node in dag:
            index[i] = node
            copied_bunch.add(node)
            copied_bunch.update(networkx.ancestors(dag, node))
        else:
            nodes_to_add.append((i, node))

    subgraph = dag.subgraph(copied_bunch).copy()
    for node_index, node in nodes_to_add:
        _, added_node = add_to_dependence_dag(subgraph,
                                              node,
                                              return_added_node=True,
                                              force=force)
        if node_index is not None:
            index[node_index] = added_node
    if return_index:
        return subgraph, index
    return subgraph


def matching_dependence_dags(a, b):
    """A helper function inteaded to be used during debugging. Is true is
    two DiGraph instances that represent a dependence DAG contain the same
    nodes, the same edges and the `deterministic` and `conditional` attributes
    of all edges are the same.
    """
    if (not isinstance(a, networkx.DiGraph) or
            not isinstance(a, networkx.DiGraph)):
        return False
    if len(a.nodes) != len(b.nodes):
        return False
    for n in b.nodes:
        if n not in a:
            return False
        for edge in b.edges(n, data=True):
            if not a.has_edge(edge[0], edge[1]):
                return False
            data = a[edge[0]][edge[1]]
            if (data['conditional'] != edge[2]['conditional'] or
                    data['deterministic'] != edge[2]['deterministic']):
                return False
    return True


def build_dependence_dag_from_model(model):
    dag = networkx.DiGraph()
    for node in model.free_RVs + model.deterministics + model.observed_RVs:
        add_to_dependence_dag(dag, node, accept_cons_shared=True)
    return dag
