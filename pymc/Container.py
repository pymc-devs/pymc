"""
The point of Container.py is to provide a function Container which converts 
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

from Node import Node, ContainerBase, Variable, StochasticBase, DeterministicBase, PotentialBase
from copy import copy
from numpy import ndarray, array, zeros, shape, arange, where, dtype, Inf
from pymc.Container_values import LCValue, TCValue, DCValue, ACValue, OCValue
from types import ModuleType


__all__ = ['Container', 'DictContainer', 'ListContainer', 'TupleContainer', 'SetContainer', 'ObjectContainer']

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
        
    if isinstance(iterable, Node):
        return ListTupleContainer([iterable])
    
    # Wrap sets
    if isinstance(iterable, set):
        return SetContainer(iterable)
    
    # Wrap lists and tuples
    elif isinstance(iterable, tuple): 
        return TupleContainer(iterable)
        
    elif isinstance(iterable, list):
        return ListContainer(iterable)

    # Dictionaries
    elif isinstance(iterable, dict):
        return DictContainer(iterable)
    
    # Arrays of dtype=object
    elif isinstance(iterable, ndarray):
        if iterable.dtype == dtype('object'):
            return ArrayContainer(iterable) 
    
    # Wrap modules
    elif isinstance(iterable, ModuleType):
        return DictContainer(filter_dict(iterable))
        
    # Wrap mutable objects
    elif hasattr(iterable, '__dict__'):
        return ObjectContainer(iterable.__dict__)
        
    # Otherwise raise an error.
    raise ValueError, 'No container classes available for class ' + iterable.__class__.__name__ + ', see Container.py for examples on how to write one.'

def file_items(container, iterable):
    """
    Files away objects into the appropriate attributes of the container.
    """

    container._value = copy(iterable)
    
    container.nodes = set()
    container.variables = set()
    container.deterministics = set()
    container.stochastics = set()
    container.potentials = set()
    container.data_stochastics = set()

    
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
                if item.isdata:
                    container.data_stochastics.add(item)
                else:
                    container.stochastics.add(item)
            elif isinstance(item, DeterministicBase):
                container.deterministics.add(item)
        elif isinstance(item, PotentialBase):
            container.potentials.add(item)

        elif isinstance(item, ContainerBase):
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
            if isinstance(container, dict):
                container.replace(key, new_container)
            elif isinstance(container, tuple):
                return iterable[:i] + (new_container,) + iterable[i+1:]
            else:
                container.replace(item, new_container, i)

            # Update all of container's variables, potentials, etc. with the new wrapped
            # iterable's. This process recursively unpacks nested iterables.
            container.containers.append(new_container)
            container.variables.update(new_container.variables)
            container.stochastics.update(new_container.stochastics)
            container.potentials.update(new_container.potentials)
            container.deterministics.update(new_container.deterministics)
            container.data_stochastics.update(new_container.data_stochastics)




    container.nodes = container.potentials | container.variables
    
value_doc = 'A copy of self, with all variables replaced by their values.'    

class SetContainer(ContainerBase, set):
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
        All the stochastics self contains with isdata=False.
      potentials : set
        All the potentials self contains.
      data_stochastics : set
        All the stochastics self contains with isdata=True.
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
    def __init__(self, iterable):
        set.__init__(self, iterable)
        ContainerBase.__init__(self, iterable)
        for item in iterable:
            if isinstance(item, Variable) or isinstance(item, ContainerBase):
                try:
                    hash(item.value)
                except TypeError:
                    raise TypeError, 'Only objects with hashable values may be included in SetContainers.\n'\
                                    + item.__repr__() + ' has value of type ' +  item.value.__class__.__name__\
                                     + '\nwhich is not hashable.'
        file_items(self, iterable)
        
    def replace(self, item, new_container, i):
        self.discard(item)
        self.add(new_container)
        
    def get_value(self):
        _value = set(self)
        for item in self:
            if isinstance(item, Variable) or isinstance(item, ContainerBase):
                set.discard(_value, item)
                set.add(_value, item.value)
                
        return _value

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
        All the stochastics self contains with isdata=False.
      potentials : set
        All the potentials self contains.
      data_stochastics : set
        All the stochastics self contains with isdata=True.
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
    def __init__(self, iterable):
        tuple.__init__(self, file_items(self, iterable))
        # ContainerBase.__init__(self, iterable)        
        # file_items(self, iterable)

        self.isval = []
        for i in xrange(len(self)):
            if isinstance(self[i], Variable) or isinstance(self[i], ContainerBase):
                self.isval.append(True)
            else:
                self.isval.append(False)

    def get_value(self):
        return TCValue(self)

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
        All the stochastics self contains with isdata=False.
      potentials : set
        All the potentials self contains.
      data_stochastics : set
        All the stochastics self contains with isdata=True.
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
    def __init__(self, iterable):
        list.__init__(self, iterable)
        ContainerBase.__init__(self, iterable)        
        file_items(self, iterable)
        
        self.val_ind = []   
        self.nonval_ind = []
        for i in xrange(len(self)):
            if isinstance(self[i], Variable) or isinstance(self[i], ContainerBase):
                self.val_ind.append(i)
            else:
                self.nonval_ind.append(i)
                
        self.n_val = len(self.val_ind)
        self.n_nonval = len(self) - self.n_val

    def replace(self, item, new_container, i):
        list.__setitem__(self, i, new_container)
        
    def get_value(self):
        LCValue(self)
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
        All the stochastics self contains with isdata=False.
      potentials : set
        All the potentials self contains.
      data_stochastics : set
        All the stochastics self contains with isdata=True.
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
    def __init__(self, iterable):
        dict.__init__(self, iterable)
        ContainerBase.__init__(self, iterable)        
        file_items(self, iterable)
        
        self.val_keys = []   
        self.nonval_keys = []
        for key in self.keys():
            if isinstance(self[key], Variable) or isinstance(self[key], ContainerBase):
                self.val_keys.append(key)
            else:
                self.nonval_keys.append(key)
                
        self.n_val = len(self.val_keys)
        self.n_nonval = len(self) - self.n_val
        
    def replace(self, key, new_container):
        dict.__setitem__(self, key, new_container)
        
    def get_value(self):
        DCValue(self)
        return self._value

    value = property(fget = get_value, doc=value_doc)

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
        All the stochastics self contains with isdata=False.
      potentials : set
        All the potentials self contains.
      data_stochastics : set
        All the stochastics self contains with isdata=True.
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
    def __init__(self, input):

        if isinstance(input, dict):
            input_to_file = input
            self.__dict__.update(input_to_file)

        elif hasattr(input,'__iter__'):
            input_to_file = input

        else: # Modules, objects, etc.
            input_to_file = filter_dict(input)
            self.__dict__.update(input_to_file)

        self._dict_container = DictContainer(self.__dict__)  
        file_items(self, input_to_file)
        
        self._value = copy(self)
        ContainerBase.__init__(self, input)
	
        
    def replace(self, item, new_container, key):
        dict.__setitem__(self.__dict__, key, new_container)

    def _get_value(self):
        OCValue(self)
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
        All the stochastics self contains with isdata=False.
      potentials : set
        All the potentials self contains.
      data_stochastics : set
        All the stochastics self contains with isdata=True.
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
    
    data=set()
    
    def __new__(subtype, array_in):

        C = array(array_in, copy=True)
        
        C = C.view(subtype)
        ContainerBase.__init__(C, array_in)
                
        # Ravelled versions of self, self.value, and self._pymc_finder.
        C._ravelleddata = array(array_in, copy=True).ravel()
        
        # Sort out contents and wrap internal containers.
        file_items(C, C._ravelleddata)
        C._value = array_in.copy()        
        C._ravelledvalue = C._value.ravel()
        
        # An array range to keep around.        
        C.iterrange = arange(len(C.ravel()))
        
        C.val_ind = []
        C.nonval_ind = []
        for i in xrange(len(C._ravelleddata)):
            if isinstance(C._ravelleddata[i], Variable) or isinstance(C._ravelleddata[i], ContainerBase):
                C.val_ind.append(i)
            else:
                C.nonval_ind.append(i)
        
        C.val_ind = array(C.val_ind, copy=True, dtype=int)
        C.nonval_ind = array(C.nonval_ind, copy=True, dtype=int)
        
        C.n_val = len(C.val_ind)
        C.n_nonval = len(C.nonval_ind)
        
        C.flags['W'] = False
        
        return C

    def replace(self, item, new_container, i):
        ndarray.__setitem__(self._ravelleddata,i, new_container)

    # This method converts self to self.value.
    def get_value(self):
        ACValue(self)
        return self._value
                
    value = property(fget = get_value, doc=value_doc)
