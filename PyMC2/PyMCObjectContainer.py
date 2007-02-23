from AbstractBase import *
from SamplingMethods import SamplingMethod
from copy import copy
from numpy import ndarray


class PyMCObjectContainer(ContainerBase):
    def __init__(self, iterable, name = 'container'):
        """
        Takes any iterable in its constructor. On instantiation, it
        recursively 'unpacks' the iterable and adds all PyMC objects
        it finds to its set 'pymc_objects.' It also adds sampling
        methods to its set 'sampling_methods'.

        When Model's constructor finds a PyMCObjectContainer, it should
        add all the objects in the sets to its own sets of the same names.

        When Parameter/ Node's constructor's 'parents' dictionary contains 
        a PyMCObjectContainer, it's retained as a parent.

        However, the parent_pointers array will contain the members of the
        set pymc_objects, and all the members of that set will have the
        parameter/ node added as a child.

        Ultimately, the 'value' attribute should return a copy of the input
        iterable, but with references to all PyMC objects replaced with
        references to their values, for ease of programming log-probability
        functions. That is, if A is a list of parameters,
        
        M=PyMCObjectContainer(A)
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
        
        M=PyMCObjectContainer(A)
        M.value[3]
        
        should return the _value_ of the fourth parameter, not the Parameter
        object itself. That means that in a log-probability or eval function
        
        M[3]
        
        would serve the same function. Who knows how to actually do this...
        """
        if not hasattr(iterable,'__iter__'):
            raise ValueError, 'PyMC object containers can only be made from iterables.'
        self.value = ValueContainer(iterable)
        self.pymc_objects = set()
        self.nodes = set()
        self.parameters = set()
        self.data = set()
        self.__name__ = name

        for i in xrange(len(iterable)):
            item = iterable[i]
            # Recursively unpacks nested iterables.
            if isinstance(item, PyMCBase):
                self.pymc_objects.add(item)
                if isinstance(item, ParameterBase):
                    if item.isdata:
                        self.data.add(item)
                    else:
                        self.parameters.add(item)
                elif isinstance(item, NodeBase):
                    self.nodes.add(item)
            elif isinstance(item, SamplingMethod):
                self.sampling_methods.add(item)

            elif hasattr(item,'__getitem__'):

                new_container = PyMCObjectContainer(item)
                self.value[i] = new_container

                self.pymc_objects.update(new_container.pymc_objects)
                self.parameters.update(new_container.parameters)
                self.nodes.update(new_container.nodes)
                self.data.update(new_container.data)

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
        if isinstance(item, PyMCBase) or isinstance(item, ContainerBase):
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


def test():
    from PyMC2.examples import DisasterModel as DM
    A = [[DM.e, DM.s], [DM.l, DM.D, 3.], 54.323]
    C = PyMCObjectContainer(A)
    return C

class ArraySubclassContainer(ContainerBase, ndarray):
    
    """
    Would we prefer to go with this? Kind of square, but
    simple.
    """
    
    def __new__(subtype, array_in):

        subtype.data = array_in.copy()
        subtype._value = array_in.copy()
        
        subtype._pymc_finder = zeros(shape(subtype.data),dtype=bool)
        
        subtype._ravelleddata = subtype.data.ravel()
        subtype._ravelledvalue = subtype._value.ravel()
        subtype._ravelledfinder = subtype._pymc_finder.ravel()
        
        subtype.pymc_objects = set()
        subtype.parameters = set()
        subtype.nodes = set()
        subtype.data = set()
        subtype.sampling_methods = set()
        
        subtype.iterrange = arange(len(subtype.data.ravel()))

        for i in subtype.iterrange:
            item = subtype._ravelleddata[i]
            if isinstance(item, PyMCBase):
    
                subtype._ravelledfinder[i] = True   
                subtype.pymc_objects.add(item)

                if isinstance(item, ParameterBase):
                    if item.isdata:
                        subtype.data.add(item)
                    else:
                        subtype.parameters.add(item)
                elif isinstance(item, NodeBase):
                    subtype.nodes.add(item)

            elif isinstance(item, SamplingMethod):
                subtype.sampling_methods.add(item)

            elif isinstance(item, ndarray):
                self._ravelledvalue[i] = ArraySubclassContainer(item)  
                subtype._ravelledfinder[i] = True
            
        return subtype.data.view(subtype)
        
    def __array__(self):
        return self.data

    def get_value(self):
        for i in self.iterrange:
            if self._ravelledfinder[i]:
                self._value[i] = self.data[i].value
                
    value = property(fget = get_value)