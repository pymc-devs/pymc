import pkgutil
import io
from copy import copy
import theano.tensor as tt
from .vartypes import isgenerator

__all__ = ['get_data_file', 'DataGenerator']


def get_data_file(pkg, path):
    """Returns a file object for a package data file.

    Parameters
    ----------
    pkg : str
        dotted package hierarchy. e.g. "pymc3.examples"
    path : str 
        file path within package. e.g. "data/wells.dat"
    Returns 
    -------
    BytesIO of the data
    """

    return io.BytesIO(pkgutil.get_data(pkg, path))


class DataGenerator(object):
    """
    Helper class that helps to infer data type of generator with looking
    at the first item, preserving the order of the resulting generator
    """

    class Variable(tt.TensorVariable):
        def __init__(self, op, type, name=None):
            super(DataGenerator.Variable, self).__init__(type=type, name=name)
            self.op = op

        def set_gen(self, gen):
            self.op.set_gen(gen)

        def set_default(self, value):
            self.op.set_default(value)

        def clone(self):
            cp = self.__class__(self.op, self.type, self.name)
            cp.tag = copy(self.tag)
            return cp

    def make_variable(self, gop, name=None):
        var = self.Variable(gop, self.tensortype, name)
        var.tag.test_value = self.test_value
        return var

    def __init__(self, generator):
        if not isgenerator(generator):
            raise TypeError('Object should be generator like')
        self.test_value = copy(next(generator))
        # make pickling potentially possible
        self._yielded_test_value = False
        self.gen = generator
        self.tensortype = tt.TensorType(
            self.test_value.dtype,
            ((False, ) * self.test_value.ndim))

    # python3 generator
    def __next__(self):
        if not self._yielded_test_value:
            self._yielded_test_value = True
            return self.test_value
        else:
            return copy(next(self.gen))

    # python2 generator
    next = __next__

    def __iter__(self):
        return self

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))
