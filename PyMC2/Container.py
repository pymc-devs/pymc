# from PyMCObjects import ParameterBase, NodeBase, PotentialBase
from PyMCBase import PyMCBase, ContainerBase, Variable, ParameterBase, NodeBase, PotentialBase, SamplingMethodBase
from copy import copy
from numpy import ndarray, array, zeros, shape, arange, where
from Container_values import LTCValue, DCValue, ACValue

# TODO: Bring docs for Container and Model up to date.
# TODO: Pass modules to DictContainer specially, and make an ObjectContainer whose value attribute makes a shallow copy.

def Container(*args):
    """
    C = Container(iterable[, name])
    
    Wraps an iterable object (currently a list, set, tuple, dictionary 
    or ndarray) in a subclass of ContainerBase and returns it.
    
    If the iterable is not recognized, a dictionary container will be made
    from its __dict__ attribute.
    
    
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
    so mu[i] will return x.value[i] = x_i.value = x[i].value. ParameterBase y
    will cache the values of each element of x, and will evaluate whether it
    needs to recompute based on all of them.
    
    :SeeAlso: ListTupleContainer, SetContainer, ArrayContainer, DictContainer 
    """
    
    if len(args)==1:
        iterable = args[0]
    else:
        iterable = args
    
    # Wrap sets
    if isinstance(iterable, set):
        return SetContainer(iterable)
    
    # # Wrap lists and tuples
    elif isinstance(iterable, tuple) or isinstance(iterable, list):
        return ListTupleContainer(iterable)

    elif isinstance(iterable, dict):
        return DictContainer(iterable)
        
    elif isinstance(iterable, ndarray):
        return ArrayContainer(iterable) 
    
    # Wrap modules
    elif hasattr(iterable, '__dict__'):
        if hasattr(iterable, '__name__'):
            name = iterable.__name__
        filtered_dict = {}
        for item in iterable.__dict__.iteritems():
            if isinstance(item[1], PyMCBase):
                filtered_dict[item[0]] = item[1]

        return DictContainer(filtered_dict)
        
    # Otherwise raise an error.
    else:
        raise ValueError, 'No container classes available for class ' + iterable.__class__.__name__ + 'see Container.py for examples on how to write one.'

 #TODO Consider changing containers to list so it can hold nonhashables
def file_items(container, iterable):
    """
    Files away objects into the appropriate attributes of the container.
    """
    container._value = copy(iterable)
    container.__name__ = container.__class__.__name__
    
    # all_objects needs to be a list because some may be unhashable.
    container.all_objects = []
    container.pymc_objects = set()
    container.variables = set()
    container.nodes = set()
    container.parameters = set()
    container.potentials = set()
    container.data = set()
    container.sampling_methods = set()
    container.container_refs = set()
    
    class ContainerReference(object):
        def __init__(self, container):
            self.owner = container
    container.reference = ContainerReference(container)
    
    i=0
    

    for item in iterable:
    
        
        # If this is a dictionary, switch from key to item.
        if isinstance(iterable, dict):
            item = iterable[item]
            
        container.all_objects.append(item)

        if hasattr(item,'__iter__'):
            
            # If the item is iterable, wrap it in a container. Replace the item
            # with the wrapped version.
            new_container = Container(item)
            container.replace(item, new_container, i)

            # Update all of container's variables, potentials, etc. with the new wrapped
            # iterable's. This process recursively unpacks nested iterables.
            container.container_refs.add(new_container.reference)
            container.variables.update(new_container.variables)
            container.parameters.update(new_container.parameters)
            container.potentials.update(new_container.potentials)
            container.nodes.update(new_container.nodes)
            container.data.update(new_container.data)
            container.sampling_methods.update(new_container.sampling_methods)

        else:
            
            # If the item isn't iterable, file it away.
            if isinstance(item, Variable):
                container.variables.add(item)
                if isinstance(item, ParameterBase):
                    if item.isdata:
                        container.data.add(item)
                    else:
                        container.parameters.add(item)
                elif isinstance(item, NodeBase):
                    container.nodes.add(item)
            elif isinstance(item, PotentialBase):
                container.potentials.add(item)
            elif isinstance(item, SamplingMethodBase):
                container.sampling_methods.add(item)
        i += 1

    container.pymc_objects = container.potentials | container.variables
    
    container.parents = {}
    for pymc_object in container.pymc_objects:
        for key in pymc_object.parents.keys():
            if isinstance(pymc_object.parents[key],Variable):
                if not pymc_object.parents[key] in container.pymc_objects:
                    container.parents[pymc_object.__name__ + '_' + key] = pymc_object.parents[key]
                
    container.children = set()
    for variable in container.variables:
        for child in variable.children:
            if not child in container.pymc_objects:
                container.children.add(child)
    
    

class SetContainer(ContainerBase, set):
    """
    SetContainers are containers that wrap sets.
    
    :SeeAlso: Container, ListContainer, DictContainer, ArrayContainer
    """
    def __init__(self, iterable, name='container'):
        set.__init__(self, iterable)
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
        _value = copy(self)
        for item in _value:
            if isinstance(item, Variable) or isinstance(item, ContainerBase):
                set.discard(_value, item)
                set.add(_value, item.value)
                
        return _value

    value = property(fget = get_value)
        

class ListTupleContainer(ContainerBase, list):
    """
    ListContainers are containers that wrap lists and tuples. 
    They act more like lists than tuples.
    
    :SeeAlso: Container, ListTupleContainer, DictContainer, ArrayContainer
    """
    def __init__(self, iterable):
        list.__init__(self, iterable)
        file_items(self, iterable)
        self._value = list(self._value)
        
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
        LTCValue(self)
        return self._value

    value = property(fget = get_value)


class DictContainer(ContainerBase, dict):
    """
    DictContainers are containers that wrap dictionaries. 
    
    :SeeAlso: Container, ListTupleContainer, SetContainer, ArrayContainer
    """
    def __init__(self, iterable):
        dict.__init__(self, iterable)
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
        
    def replace(self, item, new_container, i):
        dict.__setitem__(self, self.keys()[i], new_container)
        
    def get_value(self):
        DCValue(self)
        return self._value

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
        
        C.__name__ = name
        
        # Ravelled versions of self, self.value, and self._pymc_finder.
        C._ravelleddata = C.ravel()
        
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
        
        C.val_ind = array(C.val_ind)
        C.nonval_ind = array(C.nonval_ind)
        
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
                
    value = property(fget = get_value)
