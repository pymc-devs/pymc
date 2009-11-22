"""The point of Container.py is to provide a function Container which converts
any old thing A to thing B which looks and acts just like A, but it has a
'value' attribute. B.value looks and acts just like A but every variable
'inside' B has been replaced by its value. Examples:

    class MyObject(object):
        def __init__(self):
            self.x = Uninformative('x',0)
            self.y = 3

    A = MyObject()
    B = Container(A)
    B.x
    B.value.x


    A = [Uninformative('x',0), 3]
    B = Container(A)
    B
    B.value

Should work even with nested inputs:

    class MyObject(object):
        def __init__(self):
            self.x = [Uninformative('x',0), 5]
            self.y = 3

    A = MyObject()
    B = Container(A)


In addition, container objects file away the objects they contain into the
following sets: stochastics, deterministics, variables, nodes, containers, data, step methods.
These flattened representations are useful for things like cache checking.
"""

from Node import Node, ContainerBase, Variable, StochasticBase, DeterministicBase, PotentialBase, ContainerRegistry
from copy import copy
from numpy import ndarray, array, zeros, shape, arange, where, dtype, Inf
from pymc.Container_values import LCValue, DCValue, ACValue, OCValue
from types import ModuleType
import pdb

__all__ = ['Container', 'DictContainer', 'TupleContainer', 'ListContainer', 'SetContainer', 'ObjectContainer', 'ArrayContainer']

def filter_dict(obj):
    filtered_dict = {}
    for item in obj.__dict__.iteritems():
        if isinstance(item[1], Node) or isinstance(item[1], ContainerBase):
            filtered_dict[item[0]] = item[1]
    return filtered_dict

def Container(*args):
    """
    C = Container(iterable)
    C = Container(module)
    C = Container(object)
    C = Container(obj_1, obj_2, obj_3, ...)

    Wraps an iterable object (currently a list, set, tuple, dictionary
    or ndarray), or a module or other object, or just a sequence of objects,
    in a subclass of ContainerBase and returns it.

    Iterable subclasses of ContainerBase strive to emulate the iterables they
    wrap, with one important difference: They have a value attribute.
    A container's value attribute behaves like the container itself, but
    it replaces every PyMC variable it contains with that variable's value.

    Example:

        @stochastic
        def A(value=0., mu=3, tau=2):
            return normal_like(value, mu, tau)

        C = Container([A, 15.2])

        will yield the following:
        C[0] = A
        C.value[0] = A.value
        C[1] = C.value[1] = 15.2


    The primary reason containers exist is to allow nodes to have large
    sets of parents without the need to refer to each of the parents by name.
    Example:

        x = []

        @stochastic
        def x_0(value=0, mu=0, tau=2):
            return normal_like(value, mu, tau)

        x.append(x_0)
        last_x = x_0

        for i in range(1,N):
            @stochastic
            def x_now(value=0, mu = last_x, tau=2):
                return normal_like(value, mu, tau)

            x_now.__name__ = 'x[%i]' % i
            last_x = x_now

            x.append(x_now)

        @stochastic
        def y(value=0, mu = x, tau = 100):

            mean_sum = 0
            for i in range(len(mu)):
                mean_sum = mean_sum + mu[i]

            return normal_like(value, mean_sum, tau)

    x.value will be passed into y's log-probability function as argument mu,
    so mu[i] will return x.value[i] = x[i].value. Stochastic y
    will cache the values of each element of x, and will evaluate whether it
    needs to recompute based on all of them.

    :SeeAlso:
      ListContainer, TupleContainer, SetContainer, ArrayContainer, DictContainer,
      ObjectContainer
    """

    if len(args)==1:
        iterable = args[0]
    else:
        iterable = args

    if isinstance(iterable, ContainerBase):
        return iterable

    for container_class, containing_classes in ContainerRegistry:
        if any([isinstance(iterable, containing_class) for containing_class in containing_classes]):
            return container_class(iterable)

    # Wrap mutable objects
    if hasattr(iterable, '__dict__'):
        return ObjectContainer(iterable.__dict__)

    # Otherwise raise an error.
    raise ValueError, 'No container classes available for class ' + iterable.__class__.__name__ + ', see Container.py for examples on how to write one.'

def file_items(container, iterable):
    """
    Files away objects into the appropriate attributes of the container.
    """

    # container._value = copy(iterable)

    container.nodes = set()
    container.variables = set()
    container.deterministics = set()
    container.stochastics = set()
    container.potentials = set()
    container.observed_stochastics = set()

    # containers needs to be a list to hold unhashable items.
    container.containers = []

    i=-1

    for item in iterable:

        # If this is a dictionary, switch from key to item.
        if isinstance(iterable, dict):
            key = item
            item = iterable[key]
        # Item counter
        else:
    	    i += 1

        # If the item isn't iterable, file it away.
        if isinstance(item, Variable):
            container.variables.add(item)
            if isinstance(item, StochasticBase):
                if item.observed:
                    container.observed_stochastics.add(item)
                else:
                    container.stochastics.add(item)
            elif isinstance(item, DeterministicBase):
                container.deterministics.add(item)
        elif isinstance(item, PotentialBase):
            container.potentials.add(item)

        elif isinstance(item, ContainerBase):
            container.assimilate(item)
            container.containers.append(item)

        # Wrap internal containers
        elif hasattr(item, '__iter__'):

            # If this is a non-object-valued ndarray, don't container-ize it.
            if isinstance(item, ndarray):
                if item.dtype!=dtype('object'):
                    continue

            # If the item is iterable, wrap it in a container. Replace the item
            # with the wrapped version.
            try:
                new_container = Container(item)
            except:
                continue

            # Update all of container's variables, potentials, etc. with the new wrapped
            # iterable's. This process recursively unpacks nested iterables.
            container.assimilate(new_container)

            if isinstance(container, dict):
                container.replace(key, new_container)
            elif isinstance(container, tuple):
                return container[:i] + (new_container,) + container[i+1:]
            else:
                container.replace(item, new_container, i)

    container.nodes = container.potentials | container.variables

    # 'Freeze' markov blanket, moral neighbors, coparents of all constituent stochastics
    # for future use
    for attr in ['moral_neighbors', 'markov_blanket', 'coparents']:
        setattr(container, attr, {})
    for s in container.stochastics:
        for attr in ['moral_neighbors', 'markov_blanket', 'coparents']:
            getattr(container, attr)[s] = getattr(s, attr)

value_doc = 'A copy of self, with all variables replaced by their values.'

def sort_list(container, _value):
    val_ind = []
    val_obj = []
    nonval_ind = []
    nonval_obj = []
    for i in xrange(len(_value)):
        obj = _value[i]
        if isinstance(obj, Variable) or isinstance(obj, ContainerBase):
            val_ind.append(i)
            val_obj.append(obj)
        else:
            nonval_ind.append(i)
            nonval_obj.append(obj)
    # In case val_obj is only a single array, avert confusion.
    # Leave this even though it's confusing!
    val_obj.append(None)
    nonval_obj.append(None)
    container.n_val = len(val_ind)
    container.n_nonval = len(nonval_ind)
    container.val_ind = array(val_ind, dtype='int32')
    container.val_obj = val_obj
    container.nonval_ind = array(nonval_ind, dtype='int32')
    container.nonval_obj = array(nonval_obj, dtype=object)
    container.LCValue = LCValue(container)

class SetContainer(ContainerBase, frozenset):
    """
    SetContainers are containers that wrap sets.

    :Parameters:
      iterable : set.

    :Attributes:
      value : set
        A copy of self, with all variables replaced with their values.
      nodes : set
        All the stochastics, deterministics and potentials self contains.
      deterministics : set
        All the deterministics self contains.
      stochastics : set
        All the stochastics self contains with observed=False.
      potentials : set
        All the potentials self contains.
      observed_stochastics : set
        All the stochastics self contains with observed=True.
      containers : list
        All the containers self contains.

    :Note:
      - nodes, deterministics, etc. include all the objects in nested
        containers.
      - value replaces objects in nested containers.

    :SeeAlso:
      Container, ListContainer, DictContainer, ArrayContainer, TupleContainer,
      ObjectContainer
    """
    register=True
    change_methods = []
    containing_classes = [set, frozenset]
    def __init__(self, iterable):
        self.new_iterable = set(iterable)
        file_items(self, self.new_iterable)
        ContainerBase.__init__(self, self.new_iterable)
        self._value = list(self)
        sort_list(self, self._value)

    def replace(self, item, new_container, i):
        self.new_iterable.discard(item)
        self.new_iterable.add(new_container)

    def get_value(self):
        self.LCValue.run()
        return set(self._value)

    value = property(fget = get_value, doc=value_doc)

class TupleContainer(ContainerBase, tuple):
    """
    TupleContainers are containers that wrap tuples.

    :Parameters:
      iterable : tuple.

    :Attributes:
      value : tuple
        A copy of self, with all variables replaced with their values.
      nodes : set
        All the stochastics, deterministics and potentials self contains.
      deterministics : set
        All the deterministics self contains.
      stochastics : set
        All the stochastics self contains with observed=False.
      potentials : set
        All the potentials self contains.
      observed_stochastics : set
        All the stochastics self contains with observed=True.
      containers : list
        All the containers self contains.

    :Note:
      - nodes, deterministics, etc. include all the objects in nested
        containers.
      - value replaces objects in nested containers.

    :SeeAlso:
      Container, ListContainer, DictContainer, ArrayContainer, SetContainer,
      ObjectContainer
    """
    register=True
    change_methods = []
    containing_classes = [tuple]

    def __init__(self, iterable):
        new_tup = file_items(self, iterable)
        if len(self.containers)>0:
            raise NotImplementedError, """We have not figured out how to satisfactorily implement nested TupleContainers.
The reason is there is no way to change an element of a tuple after it has been created.
Even the Python-C API makes this impossible by checking that a tuple is new
before allowing you to change one of its elements."""
        ContainerBase.__init__(self, iterable)
        file_items(self, iterable)
        self._value = list(self)
        sort_list(self, self._value)

    def replace(self, item, new_container, i):
        list.__setitem__(self, i, new_container)

    def get_value(self):
        self.LCValue.run()
        return tuple(self._value)

    value = property(fget = get_value, doc=value_doc)

class ListContainer(ContainerBase, list):
    """
    ListContainers are containers that wrap lists.

    :Parameters:
      iterable : list.

    :Attributes:
      value : list
        A copy of self, with all variables replaced with their values.
      nodes : set
        All the stochastics, deterministics and potentials self contains.
      deterministics : set
        All the deterministics self contains.
      stochastics : set
        All the stochastics self contains with observed=False.
      potentials : set
        All the potentials self contains.
      observed_stochastics : set
        All the stochastics self contains with observed=True.
      containers : list
        All the containers self contains.

    :Note:
      - nodes, deterministics, etc. include all the objects in nested
        containers.
      - value replaces objects in nested containers.

    :SeeAlso:
      Container, TupleContainer, DictContainer, ArrayContainer, SetContainer,
      ObjectContainer
    """
    change_methods = ['__setitem__', '__delitem__', '__setslice__', '__delslice__', '__iadd__', '__imul__', 'append', 'extend', 'insert', 'pop', 'remove', 'reverse', 'sort']
    containing_classes = [list]
    register=True
    def __init__(self, iterable):
        list.__init__(self, iterable)
        ContainerBase.__init__(self, iterable)
        file_items(self, iterable)
        self._value = list(self)
        sort_list(self, self._value)

    def replace(self, item, new_container, i):
        list.__setitem__(self, i, new_container)

    def get_value(self):
        self.LCValue.run()
        return self._value

    value = property(fget = get_value, doc=value_doc)

class DictContainer(ContainerBase, dict):
    """
    DictContainers are containers that wrap dictionaries.
    Modules are converted into DictContainers, and variables' and potentials'
    Parents objects are DictContainers also.

    :Parameters:
      iterable : dictionary or object with a __dict__.

    :Attributes:
      value : dictionary
        A copy of self, with all variables replaced with their values.
      nodes : set
        All the stochastics, deterministics and potentials self contains.
      deterministics : set
        All the deterministics self contains.
      stochastics : set
        All the stochastics self contains with observed=False.
      potentials : set
        All the potentials self contains.
      observed_stochastics : set
        All the stochastics self contains with observed=True.
      containers : list
        All the containers self contains.

    :Note:
      - nodes, deterministics, etc. include all the objects in nested
        containers.
      - value replaces objects in nested containers.

    :SeeAlso:
      Container, ListContainer, TupleContainer, ArrayContainer, SetContainer,
      ObjectContainer
    """
    change_methods = ['__setitem__', '__delitem__', 'clear', 'pop', 'popitem', 'update']
    containing_classes = [dict]
    register=True
    def __init__(self, iterable):
        dict.__init__(self, iterable)
        ContainerBase.__init__(self, iterable)
        self._value = copy(iterable)
        file_items(self, iterable)

        self.val_keys = []
        self.val_obj = []
        self.nonval_keys = []
        self.nonval_obj = []
        self._value = {}
        for key, obj in self.iteritems():
            if isinstance(obj, Variable) or isinstance(obj, ContainerBase):
                self.val_keys.append(key)
                self.val_obj.append(obj)
            else:
                self.nonval_keys.append(key)
                self.nonval_obj.append(obj)
        # In case val_obj is only a single array, avert confusion.
        # Leave this even though it's confusing!
        self.val_obj.append(None)
        self.nonval_obj.append(None)

        self.n_val = len(self.val_keys)
        self.val_keys = array(self.val_keys, dtype=object)
        # self.val_obj = array(self.val_obj, dtype=object)
        self.n_nonval = len(self.nonval_keys)
        self.nonval_keys = array(self.nonval_keys, dtype=object)
        self.nonval_obj = array(self.nonval_obj, dtype=object)
        self.DCValue = DCValue(self)

    def replace(self, key, new_container):
        dict.__setitem__(self, key, new_container)

    def get_value(self):
        # DCValue(self)
        self.DCValue.run()
        return self._value

    value = property(fget = get_value, doc=value_doc)

def conservative_update(obj, dict):
    for k in dict.iterkeys():
        if not hasattr(obj, k):
            try:
                setattr(obj, k, dict[k])
            except:
                pass

class ObjectContainer(ContainerBase):
    """
    ObjectContainers wrap non-iterable objects.

    Contents of the input iterable, or attributes of the input object,
    are exposed as attributes of the object.

    :Parameters:
      iterable : dictionary or object with a __dict__.

    :Attributes:
      value : object
        A copy of self, with all variables replaced with their values.
      nodes : set
        All the stochastics, deterministics and potentials self contains.
      deterministics : set
        All the deterministics self contains.
      stochastics : set
        All the stochastics self contains with observed=False.
      potentials : set
        All the potentials self contains.
      observed_stochastics : set
        All the stochastics self contains with observed=True.
      containers : list
        All the containers self contains.

    :Note:
      - nodes, deterministics, etc. include all the objects in nested
        containers.
      - value replaces objects in nested containers.

    :SeeAlso:
      Container, ListContainer, DictContainer, ArrayContainer, SetContainer,
      TupleContainer
    """
    register=False
    def __init__(self, input):

        if isinstance(input, dict):
            input_to_file = input
            conservative_update(self, input_to_file)
            # self.__dict__.update(input_to_file)

        elif hasattr(input,'__iter__'):
            input_to_file = input

        else: # Modules, objects, etc.
            input_to_file = input.__dict__
            conservative_update(self, input_to_file)
            # self.__dict__.update(input_to_file)

        self._dict_container = DictContainer(self.__dict__)
        file_items(self, input_to_file)

        self._value = copy(self)
        ContainerBase.__init__(self, input)
        self.OCValue = OCValue(self)


    def replace(self, item, new_container, key):
        dict.__setitem__(self.__dict__, key, new_container)

    def _get_value(self):
        self.OCValue.run()
        return self._value
    value = property(fget = _get_value, doc=value_doc)


class ArrayContainer(ContainerBase, ndarray):
    """
    ArrayContainers wrap Numerical Python ndarrays. These are full
    ndarray subclasses, and should support all of ndarrays'
    functionality.

    :Parameters:
      iterable : array.

    :Attributes:
      value : array.
        A copy of self, with all variables replaced with their values.
      nodes : set
        All the stochastics, deterministics and potentials self contains.
      deterministics : set
        All the deterministics self contains.
      stochastics : set
        All the stochastics self contains with observed=False.
      potentials : set
        All the potentials self contains.
      observed_stochastics : set
        All the stochastics self contains with observed=True.
      containers : list
        All the containers self contains.

    :Note:
      - nodes, deterministics, etc. include all the objects in nested
        containers.
      - value replaces objects in nested containers.

    :SeeAlso:
      Container, ListContainer, DictContainer, ObjectContainer, SetContainer,
      TupleContainer
    """

    register=True
    change_methods = []
    containing_classes = [ndarray]
    def __new__(subtype, array_in):
        if not array_in.dtype == dtype('object'):
            raise ValueError, 'Cannot create container from array whose dtype is not object.'

        C = array(array_in, copy=True).view(subtype)
        C_ravel = C.ravel()
        ContainerBase.__init__(C, array_in)

        # Sort out contents and wrap internal containers.
        file_items(C, C_ravel)
        C._value = C.copy()
        C._ravelledvalue = C._value.ravel()

        # An array range to keep around.
        C.iterrange = arange(len(C_ravel))

        val_ind = []
        val_obj = []
        nonval_ind = []
        nonval_obj = []
        for i in xrange(len(C_ravel)):
            obj = C_ravel[i]
            if isinstance(obj, Variable) or isinstance(obj, ContainerBase):
                val_ind.append(i)
                val_obj.append(obj)
            else:
                nonval_ind.append(i)
                nonval_obj.append(obj)
        val_obj.append(None)
        C.val_ind = array(val_ind, dtype='int32')
        C.val_obj = val_obj
        C.n_val = len(val_ind)
        nonval_obj.append(None)
        C.nonval_ind = array(nonval_ind, dtype='int32')
        C.nonval_obj = array(nonval_obj, dtype=object)
        C.n_nonval = len(nonval_ind)


        C.flags['W'] = False
        C.ACValue = ACValue(C)

        return C

    def replace(self, item, new_container, i):
        ndarray.__setitem__(self.ravel(), i, new_container)

    # This method converts self to self.value.
    def get_value(self):
        self.ACValue.run()
        return self._value

    value = property(fget = get_value, doc=value_doc)

