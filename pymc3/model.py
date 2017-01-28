import threading
import six

import numpy as np
import theano
import theano.tensor as tt
from theano.tensor.var import TensorVariable

import pymc3 as pm
from .memoize import memoize
from .theanof import gradient, hessian, inputvars, generator
from .vartypes import typefilter, discrete_types, continuous_types, isgenerator
from .blocking import DictToArrayBijection, ArrayOrdering

__all__ = [
    'Model', 'Factor', 'compilef', 'fn', 'fastfn', 'modelcontext',
    'Point', 'Deterministic', 'Potential'
]


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


def get_named_nodes(graph):
    """Get the named nodes in a theano graph
    (i.e., nodes whose name attribute is not None).

    Parameters
    ----------
    graph - a theano node

    Returns:
    A dictionary of name:node pairs.
    """
    return _get_named_nodes(graph, {})


def _get_named_nodes(graph, nodes):
    if graph.owner is None:
        if graph.name is not None:
            nodes.update({graph.name: graph})
    else:
        for i in graph.owner.inputs:
            nodes.update(_get_named_nodes(i, nodes))
    return nodes


class Context(object):
    """Functionality for objects that put themselves in a context using
    the `with` statement.
    """
    contexts = threading.local()

    def __enter__(self):
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()

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

        return tt.sum(self.logp_elemwiset) * self.scaling

    @property
    def scaling(self):
        total_size = getattr(self, 'total_size', None)
        if total_size is None:
            coef = tt.constant(1)
        else:
            if self.logp_elemwiset.ndim >= 1:
                denom = self.logp_elemwiset.shape[0]
            else:
                denom = 1
            coef = tt.as_tensor(total_size) / denom
        return coef


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


class Model(six.with_metaclass(InitContextMeta, Context, Factor)):
    """Encapsulates the variables and likelihood factors of a model.

    Model class can be used for creating class based models. To create
    a class based model you should inherit from `Model` and
    override `__init__` with arbitrary definitions
    (do not forget to call base class `__init__` first).

    Parameters
    ----------
    name : str, default '' - name that will be used as prefix for
        names of all random variables defined within model
    model : Model, default None - instance of Model that is
        supposed to be a parent for the new instance. If None,
        context will be used. All variables defined within instance
        will be passed to the parent instance. So that 'nested' model
        contributes to the variables and likelihood factors of
        parent model.

    Examples
    --------
    # How to define a custom model
    class CustomModel(Model):
        # 1) override init
        def __init__(self, mean=0, sd=1, name='', model=None):
            # 2) call super's init first, passing model and name to it
            # name will be prefix for all variables here
            # if no name specified for model there will be no prefix
            super(CustomModel, self).__init__(name, model)
            # now you are in the context of instance,
            # `modelcontext` will return self
            # you can define variables in several ways
            # note, that all variables will get model's name prefix

            # 3) you can create variables with Var method
            self.Var('v1', Normal.dist(mu=mean, sd=sd))
            # this will create variable named like '{prefix_}v1'
            # and assign attribute 'v1' to instance
            # created variable can be accessed with self.v1 or self['v1']

            # 4) this syntax will also work as we are in the context
            # of instance itself, names are given as usual
            Normal('v2', mu=mean, sd=sd)

            # something more complex is allowed too
            Normal('v3', mu=mean, sd=HalfCauchy('sd', beta=10, testval=1.))

            # Deterministic variables can be used in usual way
            Deterministic('v3_sq', self.v3 ** 2)
            # Potentials too
            Potential('p1', tt.constant(1))

    # After defining a class CustomModel you can use it in several ways

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
        instance = object.__new__(cls)
        if kwargs.get('model') is not None:
            instance._parent = kwargs.get('model')
        elif cls.get_contexts():
            instance._parent = cls.get_contexts()[-1]
        else:
            instance._parent = None
        return instance

    def __init__(self, name='', model=None):
        self.name = name
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
    @memoize
    def bijection(self):
        vars = inputvars(self.cont_vars)

        bij = DictToArrayBijection(ArrayOrdering(vars),
                                   self.test_point)

        return bij

    @property
    @memoize
    def dict_to_array(self):
        return self.bijection.map

    @property
    def ndim(self):
        return self.dict_to_array(self.test_point).shape[0]

    @property
    @memoize
    def logp_array(self):
        return self.bijection.mapf(self.fastlogp)

    @property
    @memoize
    def dlogp_array(self):
        vars = inputvars(self.cont_vars)
        return self.bijection.mapf(self.fastdlogp(vars))

    @property
    @memoize
    def logpt(self):
        """Theano scalar of log-probability of the model"""
        factors = [var.logpt for var in self.basic_RVs] + self.potentials
        return tt.add(*map(tt.sum, factors))

    @property
    def varlogpt(self):
        """Theano scalar of log-probability of the unobserved random variables
           (excluding deterministic)."""
        factors = [var.logpt for var in self.vars]
        return tt.add(*map(tt.sum, factors))

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
            upscales logp of variable with :math:`coef = total_size/var.shape[0]`

        Returns
        -------
        FreeRV or ObservedRV
        """
        name = self.name_for(name)
        if data is None:
            if getattr(dist, "transform", None) is None:
                var = FreeRV(name=name, distribution=dist,
                             total_size=total_size, model=self)
                self.free_RVs.append(var)
            else:
                var = TransformedRV(name=name, distribution=dist,
                                    transform=dist.transform,
                                    total_size=total_size,
                                    model=self)
                pm._log.debug('Applied {transform}-transform to {name}'
                              ' and added transformed {orig_name} to model.'.format(
                                transform=dist.transform.name,
                                name=name,
                                orig_name='{}_{}_'.format(name, dist.transform.name)))
                self.deterministics.append(var)
                return var
        elif isinstance(data, dict):
            var = MultiObservedRV(name=name, data=data, distribution=dist,
                                  total_size=total_size, model=self)
            self.observed_RVs.append(var)
            if var.missing_values:
                self.free_RVs += var.missing_values
                self.missing_values += var.missing_values
                for v in var.missing_values:
                    self.named_vars[v.name] = v
        else:
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

    def add_random_variable(self, var):
        """Add a random variable to the named variables of the model."""
        if self.named_vars.tree_contains(var.name):
            raise ValueError(
                "Variable name {} already exists.".format(var.name))
        self.named_vars[var.name] = var
        if not hasattr(self, self.name_of(var.name)):
            setattr(self, self.name_of(var.name), var)

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
        Compiled Theano function
        """
        return theano.function(self.vars, outs,
                               allow_input_downcast=True,
                               on_unused_input='ignore',
                               accept_inplace=True,
                               mode=mode, *args, **kwargs)

    def fn(self, outs, mode=None, *args, **kwargs):
        """Compiles a Theano function which returns the values of `outs`
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
        """Compiles a Theano function which returns `outs` and takes values
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
        """Compiles and profiles a Theano function which returns `outs` and
        takes values of model vars as a dict as an argument.

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
            Use .summary() to print stats.
        """
        f = self.makefn(outs, profile=profile, *args, **kwargs)
        if point is None:
            point = self.test_point

        for _ in range(n):
            f(**point)

        return f.profile


def fn(outs, mode=None, model=None, *args, **kwargs):
    """Compiles a Theano function which returns the values of `outs` and
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
    """Compiles a Theano function which returns `outs` and takes values of model
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
    *args, **kwargs
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
            self.total_size = total_size
            self.model = model

            incorporate_methods(source=distribution, destination=self,
                                methods=['random'],
                                wrapper=InstanceMethod)

    @property
    def init_value(self):
        """Convenience attribute to return tag.test_value"""
        return self.tag.test_value


def pandas_to_array(data):
    if hasattr(data, 'values'):  # pandas
        if data.isnull().any().any():  # missing values
            return np.ma.MaskedArray(data.values, data.isnull().values)
        else:
            return data.values
    elif hasattr(data, 'mask'):
        return data
    elif isinstance(data, theano.gof.graph.Variable):
        return data
    elif isgenerator(data):
        return generator(data)
    else:
        return np.asarray(data)


def as_tensor(data, name, model, distribution):
    dtype = distribution.dtype
    data = pandas_to_array(data).astype(dtype)

    if hasattr(data, 'mask'):
        from .distributions import NoDistribution
        testval = distribution.testval or data.mean().astype(dtype)
        fakedist = NoDistribution.dist(shape=data.mask.sum(), dtype=dtype,
                                       testval=testval, parent_dist=distribution)
        missing_values = FreeRV(name=name + '_missing', distribution=fakedist,
                                model=model)
        constant = tt.as_tensor_variable(data.filled())

        dataTensor = tt.set_subtensor(
            constant[data.mask.nonzero()], missing_values)
        dataTensor.missing_values = missing_values
        return dataTensor
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
        if type is None:
            data = pandas_to_array(data)
            type = TensorType(distribution.dtype, data.shape)

        super(TensorVariable, self).__init__(type, None, None, name)

        if distribution is not None:
            data = as_tensor(data, name, model, distribution)

            self.missing_values = data.missing_values

            self.logp_elemwiset = distribution.logp(data)
            self.total_size = total_size
            self.model = model
            self.distribution = distribution

            # make this RV a view on the combined missing/nonmissing array
            theano.gof.Apply(theano.compile.view_op,
                             inputs=[data], outputs=[self])

            self.tag.test_value = theano.compile.view_op(data).tag.test_value

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
        self.total_size = total_size
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
    n : var but with name name
    """
    model = modelcontext(model)
    var.name = model.name_for(name)
    model.deterministics.append(var)
    model.add_random_variable(var)
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
    return var


class TransformedRV(TensorVariable):

    def __init__(self, type=None, owner=None, index=None, name=None,
                 distribution=None, model=None, transform=None,
                 total_size=None):
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
        super(TransformedRV, self).__init__(type, owner, index, name)

        if distribution is not None:
            self.model = model

            transformed_name = "{}_{}_".format(name, transform.name)
            self.transformed = model.Var(
                transformed_name, transform.apply(distribution), total_size=total_size)

            normalRV = transform.backward(self.transformed)

            theano.Apply(theano.compile.view_op, inputs=[
                         normalRV], outputs=[self])
            self.tag.test_value = normalRV.tag.test_value

            incorporate_methods(source=distribution, destination=self,
                                methods=['random'],
                                wrapper=InstanceMethod)

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
    """Check that vars not include discrete variables, excepting ObservedRVs.
    """
    vars_ = [var for var in vars if not isinstance(var, pm.model.ObservedRV)]
    if any([var.dtype in pm.discrete_types for var in vars_]):
        return False
    else:
        return True

# theano stuff
theano.config.warn.sum_div_dimshuffle_bug = False
theano.config.compute_test_value = 'raise'
