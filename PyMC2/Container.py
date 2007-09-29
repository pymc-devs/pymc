from PyMCObjects import Parameter, Node, Potential
from PyMCBase import PyMCBase, ContainerBase, Variable
from SamplingMethods import SamplingMethod
from copy import copy
from numpy import ndarray, array, zeros, shape, arange, where

def Container(iterable = None, name = 'container', **args):
    """
    C = Container(iterable[, name])
    
    Wraps an iterable object (currently a list, set, tuple, dictionary 
    or ndarray) in a subclass of ContainerBase and returns it.
    
    Subclasses of ContainerBase strive to emulate the iterables they wrap,
    with two important differences:
        - They are read-only
        - They have a value attribute.
        
    A container's value attribute behaves like the container itself, but
    it replaces every PyMC variable it contains with that variable's value.
    
    Example:
    
        @parameter
        def A(value=0., mu=3, tau=2):
            return normal_like(value, mu, tau)
        
        C = Container([A, 15.2])
    
        will yield the following:
        C[0] = A
        C.value[0] = A.value
        C[1] = C.value[1] = A
    
    
    The primary reason containers exist is to allow parameters to have large
    sets of parents without the need to refer to each of the parents by name.
    Example:
    
        x = []
    
        @parameter
        def x_0(value=0, mu=0, tau=2):
            return normal_like(value, mu, tau)
    
        x.append(x_0)
        last_x = x_0
    
        for i in range(1,N):          
            @parameter
            def x_now(value=0, mu = last_x, tau=2):
                return normal_like(value, mu, tau)
                
            x_now.__name__ = 'x\_%i' % i
            last_x = x_now
        
            x.append(x_now)
        
        x = Container(x, name = 'x')
        
        @parameter
        def y(value=0, mu = x, tau = 100):

            mean_sum = 0
            for i in range(len(mu)):
                mean_sum = mean_sum + mu[i]

            return normal_like(value, mean_sum, tau)
        
    x.value will be passed into y's log-probability function as argument mu, 
    so mu[i] will return x.value[i] = x_i.value = x[i].value. Parameter y
    will cache the values of each element of x, and will evaluate whether it
    needs to recompute based on all of them.
    
    :SeeAlso: ListTupleContainer, SetContainer, ArrayContainer, DictContainer 
    """
    
    # Wrap classes and modules
    if len(args)>0:
        return ListTupleContainer(args,name)
    
    elif hasattr(iterable, '__dict__'):
        if name == 'container' and hasattr(iterable, '__name__'):
            name = iterable.__name__
        filtered_dict = {}
        for item in iterable.__dict__.iteritems():
            if isinstance(item[1], Variable) or isinstance(item[1], ContainerBase):
                filtered_dict[item[0]] = item[1]
                
        return DictContainer(filtered_dict, name)
        
    # Wrap sets
    elif isinstance(iterable, set):
        return SetContainer(iterable, name)
    
    # # Wrap lists and tuples
    elif isinstance(iterable, tuple) or isinstance(iterable, list):
        return ListTupleContainer(iterable, name)

    elif isinstance(iterable, dict):
        return DictContainer(iterable, name)
        
    elif isinstance(iterable, ndarray):
        return ArrayContainer(iterable, name) 
        
    # Otherwise raise an error.
    else:
        raise ValueError, 'No container classes available for class ' + iterable.__class__.__name__ + 'see Container.py for examples on how to write one.'

def Container_init(container, iterable):
    """
    Files away objects into the appropriate attributes of the container.
    """
    container.all_objects = set()
    container.variables = set()
    container.nodes = set()
    container.parameters = set()
    container.potentials = set()
    container.data = set()
    container.sampling_methods = set()
    
    i=0
    
    for item in iterable:
        
        # If this is a dictionary, switch from key to item.
        if isinstance(iterable, dict):
            item = iterable[item]

        if hasattr(item,'__iter__'):
            
            # If the item is iterable, wrap it in a container. Replace the item
            # with the wrapped version.
            new_container = Container(item)
            container.replace(item, new_container, i)

            # Update all of container's variables, potentials, etc. with the new wrapped
            # iterable's. This process recursively unpacks nested iterables.
            container.variables.update(new_container.variables)
            container.parameters.update(new_container.parameters)
            container.potentials.update(new_container.potentials)
            container.nodes.update(new_container.nodes)
            container.data.update(new_container.data)
            container.all_objects.update(new_container.all_objects)
            container.sampling_methods.update(new_container.sampling_methods)

        else:
            
            # If the item isn't iterable, file it away.
            container.all_objects.add(item)
            if isinstance(item, Variable):
                container.variables.add(item)
                if isinstance(item, Parameter):
                    if item.isdata:
                        container.data.add(item)
                    else:
                        container.parameters.add(item)
                elif isinstance(item, Node):
                    container.nodes.add(item)
            elif isinstance(item, Potential):
                container.potentials.add(item)
            elif isinstance(item, SamplingMethod):
                container.sampling_methods.add(item)
        i += 1

    container.pymc_objects = container.potentials | container.variables
    

class SetContainer(ContainerBase, set):
    """
    SetContainers are containers that wrap sets.
    
    :SeeAlso: Container, ListContainer, DictContainer, ArrayContainer
    """
    def __init__(self, iterable, name='container'):
        set.__init__(self, iterable)
        Container_init(self, iterable)
        
    def replace(self, item, new_container, i):
        self.discard(item)
        self.add(new_container)
        
    def get_value(self):
        _value = copy(self)
        for item in _value:
            if isinstance(item, Variable) or isinstance(item, ContainerBase):
                _value.discard(item)
                _value.add(item.value)
                
        return _value

    value = property(fget = get_value)
        
class ListTupleContainer(ContainerBase, list):
    """
    ListContainers are containers that wrap lists and tuples. 
    They act more like lists than tuples.
    
    :SeeAlso: Container, ListTupleContainer, DictContainer, ArrayContainer
    """
    def __init__(self, iterable, name='container'):
        list.__init__(self, iterable)
        Container_init(self, iterable)
        
    def replace(self, item, new_container, i):
        self[i] = new_container
        
    def get_value(self):
        _value = copy(self)
        for i in xrange(len(_value)):
            item = _value[i]
            if isinstance(item, Variable) or isinstance(item, ContainerBase):
                _value[i] = item.value

        return _value

    value = property(fget = get_value)

class DictContainer(ContainerBase, dict):
    """
    DictContainers are containers that wrap dictionaries. 
    
    :SeeAlso: Container, ListTupleContainer, SetContainer, ArrayContainer
    """
    def __init__(self, iterable, name='container'):
        dict.__init__(self, iterable)
        Container_init(self, iterable)
        
    def replace(self, item, new_container, i):
        self[self.keys()[i]] = new_container
        
    def get_value(self):
        _value = copy(self)
        for key in _value.iterkeys():
            item = _value[key]
            if isinstance(item, Variable) or isinstance(item, ContainerBase):
                _value[key] = item.value

        return _value

    value = property(fget = get_value)        

class ArrayContainer(ContainerBase, ndarray):
    """
    ArrayContainers wrap Numerical Python ndarrays. These are full 
    ndarray subclasses, and should support all of ndarrays' 
    functionality.
    
    :SeeAlso: Container, SetContainer, ListDictContainer
    """
    
    data=set()
    
    def __new__(subtype, array_in, name='container'):

        C = array(array_in, copy=True)
        
        C = C.view(subtype)
        
        ContainerBase.__init__(C)
        
        C._value = array_in.copy()
        C.name = name
        
        # Ravelled versions of self, self.value, and self._pymc_finder.
        C._ravelleddata = C.ravel()
        C._ravelledvalue = C._value.ravel()
        
        # An array range to keep around.        
        C.iterrange = arange(len(C.ravel()))
        
        C._pymc_finder = ()
        for i in xrange(len(C._ravelleddata)):
            if isinstance(C._ravelleddata[i], Variable):
                C._pymc_finder += (i,)
        Container_init(C, C._ravelleddata)

        C.flags['W'] = False
        
        return C

    def replace(self, item, new_container, i):
        self._ravelleddata[i] = new_container
        self._pymc_finder += (i,)

    # This method converts self to self.value.
    def get_value(self):
        for index in self._pymc_finder:
            self._ravelledvalue[index] = self._ravelleddata[index].value
        return self._value
                
    value = property(fget = get_value)
