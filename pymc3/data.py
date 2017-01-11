import itertools
import pkgutil
import io

import theano.tensor as tt

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
    def __init__(self, iterable):
        if hasattr(iterable, '__len__'):
            generator = itertools.cycle(iterable)
        else:
            generator = iter(iterable)
        self.test_value = next(generator)
        self.gen = itertools.chain([self.test_value], generator)
        self.tensortype = tt.TensorType(
            self.test_value.dtype,
            ((False, ) * self.test_value.ndim))

    def __next__(self):
        return next(self.gen)

    def __iter__(self):
        return self

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))
