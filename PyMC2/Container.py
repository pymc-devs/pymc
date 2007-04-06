from PyMCObjects import Parameter, Node
from PyMCBase import PyMCBase, ContainerBase
from copy import copy
from numpy import ndarray


class Container(ContainerBase):
    def __init__(self, iterable, name = 'container'):
        """
        Takes any iterable in its constructor. On instantiation, it
        recursively 'unpacks' the iterable and adds all PyMC objects
        it finds to its set 'pymc_objects.' It also adds sampling
        methods to its set 'sampling_methods'.

        When Model's constructor finds a Container, it should
        add all the objects in the sets to its own sets of the same names.

        When Parameter/ Node's constructor's 'parents' dictionary contains 
        a Container, it's retained as a parent.

        However, the parent_pointers array will contain the members of the
        set pymc_objects, and all the members of that set will have the
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
        if not hasattr(iterable,'__iter__'):
            raise ValueError, 'PyMC object containers can only be made from iterables.'
        self.value = ValueContainer(iterable)
        self.all_objects = set()
        self.pymc_objects = set()
        self.nodes = set()
        self.parameters = set()
        self.data = set()
        self.__name__ = name

        for i in xrange(len(iterable)):
            item = iterable[i]
            # Recursively unpacks nested iterables.
            if hasattr(item,'__getitem__'):

                new_container = Container(item)
                self.value[i] = new_container

                self.pymc_objects.update(new_container.pymc_objects)
                self.parameters.update(new_container.parameters)
                self.nodes.update(new_container.nodes)
                self.data.update(new_container.data)
                self.all_objects.update(new_container.all_objects)
            else:
                self.all_objects.add(item)
                if isinstance(item, PyMCBase):
                    self.pymc_objects.add(item)
                    if isinstance(item, Parameter):
                        if item.isdata:
                            self.data.add(item)
                        else:
                            self.parameters.add(item)
                    elif isinstance(item, Node):
                        self.nodes.add(item)
            

        def get_value(self):
            return self.value


"""
Anand, here is something that works for Parameters and Nodes, 
but not for SamplingMethods since they don't have a value attribute.
"""

class ValueContainer(object):
    
    def __init__(self, value):
        self._value = copy(value)

    def __getitem__(self,index):
        item = self._value[index]
        if isinstance(item, PyMCBase) or isinstance(item, Container):
            return item.value
        else:
            return item

    def __setitem__(self, index, value):
        self._value.__setitem__(index, value)

    # These aren't working yet.
    def __getslice__(self, slice):
        val = []
        j=0
        for i in xrange(i.start, i.stop, i.step):
            val[j] = self.__getitem__(i)
            j += 1 

        return val

    def __setslice__(self, slice, value):
        return self._value.__setslice__(slice, value)


class ArraySubclassContainer(Container, ndarray):
    
    """
    Would we prefer to go with this? Kind of square, but simple.
    """
    
    def __new__(subtype, array_in):

        C.data = array_in.copy()
        C._value = array_in.copy()
        
        C._pymc_finder = zeros(shape(C.data),dtype=bool)
        
        C._ravelleddata = C.data.ravel()
        C._ravelledvalue = C._value.ravel()
        C._ravelledfinder = C._pymc_finder.ravel()
        
        C.pymc_objects = set()
        C.parameters = set()
        C.nodes = set()
        C.data = set()
        C.sampling_methods = set()
        
        C.iterrange = arange(len(C.data.ravel()))

        for i in C.iterrange:
            item = C._ravelleddata[i]
            if isinstance(item, PyMCBase):
    
                C._ravelledfinder[i] = True   
                C.pymc_objects.add(item)

                if isinstance(item, Parameter):
                    if item.isdata:
                        C.data.add(item)
                    else:
                        C.parameters.add(item)
                elif isinstance(item, Node):
                    C.nodes.add(item)

            elif isinstance(item, ndarray):
                self._ravelledvalue[i] = ArraySubclassContainer(item)  
                C._ravelledfinder[i] = True
            
        return C.data.view(subtype)
        
    def __array__(self):
        return self.data

    def get_value(self):
        for i in self.iterrange:
            if self._ravelledfinder[i]:
                self._value[i] = self.data[i].value
                
    value = property(fget = get_value)
