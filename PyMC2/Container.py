from PyMCObjects import Parameter, Node, Potential
from PyMCBase import PyMCBase, ContainerBase, Variable
from SamplingMethods import SamplingMethod
from copy import copy
from numpy import ndarray, array, zeros, shape, arange

def Container(iterable, name = 'container'):
    """
    Takes any iterable in its constructor. On instantiation, it
    recursively 'unpacks' the iterable and adds all PyMC objects
    it finds to its set 'variables.' It also adds sampling
    methods to its set 'sampling_methods'.

    When Model's constructor finds a Container, it should
    add all the objects in the sets to its own sets of the same names.

    When Parameter/ Node's constructor's 'parents' dictionary contains 
    a Container, it's retained as a parent.

    However, the parent_pointers array will contain the members of the
    set variables, and all the members of that set will have the
    parameter/ node added as a child.

    Ultimately, the 'value' attribute should return a copy of the input
    iterable, but with references to all PyMC objects replaced with
    references to their values, for ease of programming log-probability
    functions. That is, if A is a list of parameters,

    M=Container(A)
    M.value[3]

    should return the _value_ of the fourth parameter, not the Parameter
    object itself. That means that in a log-probability or eval function

    M[3]

    would serve the same function.

    To do this: The constructor doesn't unpack the entire iterator, it
    just explores the top layer. When it finds an iterable, it creates
    a _new_ container object around the iterable and stores the
    container object.

    That way, the value query can automatically recur and explore the
    nested iterable. The problem reduces to figuring out how to go
    through an iterable and replace references to PyMCObjects and
    containers with their 'value' attributes.

    This might be easiest in C. I don't know if writing
    O=&PyIter_Next
    *O=Py_GetAttr(*O,"value")
    would do it. Probably not, PyIter_Next would probably return a copy 
    of the pointer from the underlying iterator.

    Ultimately, the 'value' attribute should return a copy of the input
    iterable, but with references to all PyMC objects replaced with
    references to their values, for ease of programming log-probability
    functions. That is, if A is a list of parameters,

    M=Container(A)
    M.value[3]

    should return the _value_ of the fourth parameter, not the Parameter
    object itself. That means that in a log-probability or eval function

    M[3]

    would serve the same function. Who knows how to actually do this...
    """
    
    if isinstance(iterable, ndarray):
        return ArrayContainer(iterable, name)
    elif isinstance(iterable, set):
        return SetContainer(iterable, name)
    elif hasattr(iterable, '__getitem__'):
        return ListDictContainer(iterable, name)
    else:
        raise ValueError, 'No container classes available for class ' + iterable.__class__

class SetContainer(ContainerBase):
    def __init__(self, iterable, name = 'container'):

        ContainerBase.__init__(self)
        
        self._value = copy(iterable)
        self.__name__ = name

        for item in iterable:
            # Recursively unpacks nested iterables.
            if hasattr(item,'__iter__'):
                
                new_container = Container(item)
                self._value.discard(item)
                self._value.add(new_container)

                self.variables.update(new_container.variables)
                self.parameters.update(new_container.parameters)
                self.potentials.update(new_container.potentials)
                self.nodes.update(new_container.nodes)
                self.data.update(new_container.data)
                self.all_objects.update(new_container.all_objects)
                self.sampling_methods.update(new_container.sampling_methods)
            else:
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
        # self._value = SetValueContainer(self._value)

    def __iter__(self):
        return self.all_objects.__iter__()

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
    def __init__(self, iterable, name = 'container'):
        
        ContainerBase.__init__(self)

        self._value = copy(iterable)
        self.__name__ = name

        for i in xrange(len(iterable)):
            item = iterable[i]
            # Recursively unpacks nested iterables.
            if hasattr(item,'__iter__'):

                new_container = Container(item)
                self._value[i] = new_container

                self.variables.update(new_container.variables)
                self.parameters.update(new_container.parameters)
                self.potentials.update(new_container.potentials)
                self.nodes.update(new_container.nodes)
                self.data.update(new_container.data)
                self.all_objects.update(new_container.all_objects)
                self.sampling_methods.update(new_container.sampling_methods)
            else:
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
        self._value = ListDictValueContainer(self._value)
            

    def get_value(self):
        return self._value
        
    def __getitem__(self,index):
        return self._value._value[index]
        
    value = property(fget = get_value)


class ListDictValueContainer(object):
    
    def __init__(self, value):
        self._value = copy(value)

    def __getitem__(self,index):
        item = self._value[index]
        if isinstance(item, Variable) or isinstance(item, ContainerBase):
            return item.value
        else:
            return item



class ArrayContainer(ContainerBase, ndarray):
    
    data=set()
    
    def __init__(self, array_in, name):
        pass
    
    def __new__(subtype, array_in, name):

        C = array(array_in, copy=True)
        
        C = C.view(subtype)
        
        ContainerBase.__init__(C)
        
        C._value = array_in.copy()
        C.name = name
        
        C._pymc_finder = zeros(C.shape,dtype=bool)
        
        C._ravelleddata = C.ravel()
        C._ravelledvalue = C._value.ravel()
        C._ravelledfinder = C._pymc_finder.ravel()
                
        C.iterrange = arange(len(C.ravel()))

        for i in C.iterrange:            
            item = C._ravelleddata[i]
            
            if hasattr(item,'__iter__'):

                new_container = Container(item)
                C._ravelleddata[i] = new_container
                C._ravelledfinder[i] = True                   

                C.variables.update(new_container.variables)
                C.parameters.update(new_container.parameters)
                C.potentials.update(new_container.potentials)
                C.nodes.update(new_container.nodes)
                C.data.update(new_container.data)
                C.all_objects.update(new_container.all_objects)
                C.sampling_methods.update(new_container.sampling_methods)
                             
            
            else:
                C.all_objects.add(item)
            
                if isinstance(item, Variable):
    
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

        C.flags['W'] = False
        C.pymc_objects = C.parameters | C.nodes | C.potentials
        
        return C

    def get_value(self):
        for i in self.iterrange:
            if self._ravelledfinder[i]:
                self._ravelledvalue[i] = self._ravelleddata[i].value
        return self._value
                
    value = property(fget = get_value)
