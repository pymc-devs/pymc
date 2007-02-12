from PyMCObjects import *

class PyMCObjectContainer(object):
    def __init__(self, iterable, name = 'container'):
        """
        Takes any iterable in its constructor. On instantiation, it
        recursively 'unpacks' the iterable and adds all PyMC objects
        it finds to its set 'pymc_objects.' It also adds sampling
        methods to its set 'sampling_methods.

        When Model's constructor finds a PyMCObjectContainer, it should
        add all the objects in the sets to its own sets of the same names.

        When Parameter/ Node's constructor's 'parents' dictionary contains 
        a, PyMCObjectContainer, it's retained as a parent. The 'value'
        attribute returns the iterable that was passed to the constructor,
        for ease of programming log-probability functions.

        However, the parent_pointers array will contain the members of the
        set pymc_objects, and all the members of that set will have the
        parameter/ node added as a child.
        """
        if not hasattr(iterable,'__iter__'):
            raise ValueError, 'PyMC object containers can only be made from iterables.'
        self.value = iterable
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
                if isinstance(item, Parameter):
                    if item.isdata:
                        self.data.add(item)
                    else:
                        self.parameters.add(item)
                elif isinstance(item, Node):
                    self.nodes.add(item)
            elif isinstance(item, SamplingMethod):
                self.sampling_methods.add(item)
            elif hasattr(item,'__iter__'):
                self.unpack(item)