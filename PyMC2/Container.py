from PyMCObjects import Parameter, Node, Potential
from PyMCBase import PyMCBase, ContainerBase, Variable
from SamplingMethods import SamplingMethod
from copy import copy
from numpy import ndarray, array, zeros, shape, arange

# TODO: Eliminate item-filing code duplication.
# TODO: Can you just subclass the iterables and ContainerBase? Probably.
def Container(iterable, name = 'container'):
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
    
    :SeeAlso: ListDictContainer, SetContainer, ArrayContainer
    """
    
    # Wrap arrays
    if isinstance(iterable, ndarray):
        return ArrayContainer(iterable, name)
        
    # Wrap sets
    elif isinstance(iterable, set):
        return SetContainer(iterable, name)
    
    # Wrap lists and dictionaries
    elif hasattr(iterable, '__getitem__'):
        return ListDictContainer(iterable, name)
        
    # Otherwise raise an error.
    else:
        raise ValueError, 'No container classes available for class ' + iterable.__class__

class SetContainer(ContainerBase):
    """
    SetContainers are containers that wrap sets. They are iterable.
    More set methods will eventually be implemented in SetContainer.
    
    :SeeAlso: Container, ListDictContainer, ArrayContainer
    """
    def __init__(self, iterable, name = 'container'):

        ContainerBase.__init__(self)
        
        self._value = copy(iterable)
        self.__name__ = name

        for item in iterable:

            if hasattr(item,'__iter__'):
                
                # If the item is iterable, wrap it in a container. Replace the item
                # with the wrapped version.
                new_container = Container(item)
                self._value.discard(item)
                self._value.add(new_container)

                # Update all of self's variables, potentials, etc. with the new wrapped
                # iterable's. This process recursively unpacks nested iterables.
                self.variables.update(new_container.variables)
                self.parameters.update(new_container.parameters)
                self.potentials.update(new_container.potentials)
                self.nodes.update(new_container.nodes)
                self.data.update(new_container.data)
                self.all_objects.update(new_container.all_objects)
                self.sampling_methods.update(new_container.sampling_methods)

            else:
                
                # If the item isn't iterable, file it away.
                self.all_objects.add(item)
                if isinstance(item, Variable):
                    self.variables.add(item)
                    if isinstance(item, Parameter):
                        if item.isdata:
                            self.data.add(item)
                        else:
                            self.parameters.add(item)
                    elif isinstance(item, Node):
                        self.nodes.add(item)
                elif isinstance(item, Potential):
                    self.potentials.add(item)
                elif isinstance(item, SamplingMethod):
                    self.sampling_methods.add(item)

        self.pymc_objects = self.potentials | self.variables

    def __iter__(self):
        return self.all_objects.__iter__()

    # This method converts self to self.value.
    def get_value(self):
        return_set = copy(self._value)
        for item in return_set:
            if isinstance(item, Variable) or isinstance(item, ContainerBase):
                return_set.discard(item)
                return_set.add(item.value)
            else:
                return_set.add(item)
                
        return return_set

    value = property(fget = get_value)

    

class ListDictContainer(ContainerBase):
    """
    ListDictContainers are containers that wrap lists and dictionaries.
    They support indexing and iteration. More methods will eventually be
    implemented.
    
    :SeeAlso: Container, SetContainer, ArrayContainer
    """
    def __init__(self, iterable, name = 'container'):
        
        ContainerBase.__init__(self)

        self._value = copy(iterable)
        self.__name__ = name

        for i in xrange(len(iterable)):
            item = iterable[i]

            if hasattr(item,'__iter__'):
                
                # If the item is iterable, wrap it in a container. Replace the item
                # with the wrapped version.
                new_container = Container(item)
                self._value[i] = new_container
                
                # Update all of self's variables, potentials, etc. with the new wrapped
                # iterable's. This process recursively unpacks nested iterables.                
                self.variables.update(new_container.variables)
                self.parameters.update(new_container.parameters)
                self.potentials.update(new_container.potentials)
                self.nodes.update(new_container.nodes)
                self.data.update(new_container.data)
                self.all_objects.update(new_container.all_objects)
                self.sampling_methods.update(new_container.sampling_methods)

            else:
                            
                # If the item isn't iterable, file it away.
                self.all_objects.add(item)
                if isinstance(item, Variable):
                    self.variables.add(item)
                    if isinstance(item, Parameter):
                        if item.isdata:
                            self.data.add(item)
                        else:
                            self.parameters.add(item)
                    elif isinstance(item, Node):
                        self.nodes.add(item)
                elif isinstance(item, Potential):
                    self.potentials.add(item)
                elif isinstance(item, SamplingMethod):
                    self.sampling_methods.add(item)
                    
        self.pymc_objects = self.potentials | self.variables
        
        # Initialize self._value as a list/dictionary value container object
        self._value = ListDictValueContainer(self._value)
            

    def get_value(self):
        return self._value
        
    def __getitem__(self,index):
        return self._value._value[index]
        
    value = property(fget = get_value)


class ListDictValueContainer(object):
    """
    The value attribute of a ListDictContainer.
    
    Every PyMC variable contained in a ListDictContainer
    is replaced by its value in corresponding 
    ListDictValueContainers.
    
    :SeeAlso: ListDictContainer
    """
    
    def __init__(self, value):
        self._value = copy(value)

    def __getitem__(self,index):
        item = self._value[index]
        if isinstance(item, Variable) or isinstance(item, ContainerBase):
            return item.value
        else:
            return item



class ArrayContainer(ContainerBase, ndarray):
    """
    ArrayContainers wrap Numerical Python ndarrays. These are full 
    ndarray subclasses, and should support all of ndarrays' 
    functionality.
    
    :SeeAlso: Container, SetContainer, ListDictContainer
    """
    data=set()
    
    def __init__(self, array_in, name):
        pass
    
    def __new__(subtype, array_in, name):

        C = array(array_in, copy=True)
        
        C = C.view(subtype)
        
        ContainerBase.__init__(C)
        
        C._value = array_in.copy()
        C.name = name
        
        # A boolean array indicating which elements are PyMC variables or containers.
        C._pymc_finder = zeros(C.shape,dtype=bool)
        
        # Ravelled versions of self, self.value, and self._pymc_finder.
        C._ravelleddata = C.ravel()
        C._ravelledvalue = C._value.ravel()
        C._ravelledfinder = C._pymc_finder.ravel()
        
        # An array range to keep around.        
        C.iterrange = arange(len(C.ravel()))

        for i in C.iterrange:            
            item = C._ravelleddata[i]
            
            if hasattr(item,'__iter__'):

                # If the item is iterable, wrap it in a container. Replace the item
                # with the wrapped version.
                new_container = Container(item)
                C._ravelleddata[i] = new_container
                
                # Remember that item is a container.
                C._ravelledfinder[i] = True                   

                # Update all of self's variables, potentials, etc. with the new wrapped
                # iterable's. This process recursively unpacks nested iterables.
                C.variables.update(new_container.variables)
                C.parameters.update(new_container.parameters)
                C.potentials.update(new_container.potentials)
                C.nodes.update(new_container.nodes)
                C.data.update(new_container.data)
                C.all_objects.update(new_container.all_objects)
                C.sampling_methods.update(new_container.sampling_methods)
                             
            
            else:                
                # If the item isn't iterable, file it away.

                C.all_objects.add(item)
            
                if isinstance(item, Variable):
                    
                    # Remember that this item is a variable.
                    C._ravelledfinder[i] = True   
                    C.variables.add(item)

                    if isinstance(item, Parameter):
                        if item.isdata:
                            C.data.add(item)
                        else:
                            C.parameters.add(item)
                    elif isinstance(item, Node):
                        C.nodes.add(item)

                elif isinstance(item, Potential):
                    C.potentials.add(item)
                    
                elif isinstance(item, SamplingMethod):
                    C.sampling_methods.add(item)

        # Make self not writable.
        C.flags['W'] = False
        C.pymc_objects = C.variables | C.potentials
        
        return C

    # This method converts self to self.value.
    def get_value(self):
        for i in self.iterrange:
            if self._ravelledfinder[i]:
                self._ravelledvalue[i] = self._ravelleddata[i].value
        return self._value
                
    value = property(fget = get_value)
