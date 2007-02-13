
from AbstractBase import *
from SamplingMethods import SamplingMethod


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
        
        would serve the same function. Who knows how to actually do this...
        """
        if not hasattr(iterable,'__iter__'):
            raise ValueError, 'PyMC object containers can only be made from iterables.'
        self.value = ValueContainer(iterable)
        self.pymc_objects = set()
        self.nodes = set()
        self.parameters = set()
        self.data = set()
        self.sampling_methods = set()
        self.unpack(iterable)
        self.__name__ = name
    
    def unpack(self, iterable):
        # Recursively unpacks nested iterables.
        for item in iterable:
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
            elif hasattr(item,'__iter__'):
                self.unpack(item)


"""
Anand, here is something that works for Parameters and Nodes, 
but not for SamplingMethods since they don't have a value attribute.
"""
class ValueContainer(object):
    def __init__(self, value):
        self._value = value
    def __getitem__(self,index):
        return self._value[index].value


def test():
    from examples import DisasterModel as DM
    A = [DM.e, DM.s, DM.l, DM.S, DM.D]
    C = PyMCObjectContainer(A)
    print C.value[0]
    print C.value[1]
    print C.value[2]
    print C.value[3]
    return C
