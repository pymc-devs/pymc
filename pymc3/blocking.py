"""
pymc3.blocking

Classes for working with subsets of parameters.
"""
import copy
import numpy as np
import collections

__all__ = ['ArrayOrdering', 'DictToArrayBijection', 'DictToVarBijection']

VarMap = collections.namedtuple('VarMap', 'var, slc, shp, dtyp')
DataMap = collections.namedtuple('DataMap', 'list_ind, slc, shp, dtype')


# TODO Classes and methods need to be fully documented.


class ArrayOrdering(object):
    """
    An ordering for an array space
    """

    def __init__(self, vars):
        self.vmap = []
        dim = 0

        for var in vars:
            slc = slice(dim, dim + var.dsize)
            self.vmap.append(VarMap(str(var), slc, var.dshape, var.dtype))
            dim += var.dsize

        self.dimensions = dim


class DictToArrayBijection(object):
    """
    A mapping between a dict space and an array space
    """

    def __init__(self, ordering, dpoint):
        self.ordering = ordering
        self.dpt = dpoint

        # determine smallest float dtype that will fit all data
        if all([x.dtyp == 'float16' for x in ordering.vmap]):
            self.array_dtype = 'float16'
        elif all([x.dtyp == 'float32' for x in ordering.vmap]):
            self.array_dtype = 'float32'
        else:
            self.array_dtype = 'float64'

    def map(self, dpt):
        """
        Maps value from dict space to array space

        Parameters
        ----------
        dpt : dict
        """
        apt = np.empty(self.ordering.dimensions, dtype=self.array_dtype)
        for var, slc, _, _ in self.ordering.vmap:
            apt[slc] = dpt[var].ravel()
        return apt

    def rmap(self, apt):
        """
        Maps value from array space to dict space

        Parameters
        ----------
        apt : array
        """
        dpt = self.dpt.copy()

        for var, slc, shp, dtyp in self.ordering.vmap:
            dpt[var] = np.atleast_1d(apt)[slc].reshape(shp).astype(dtyp)

        return dpt

    def mapf(self, f):
        """
         function f : DictSpace -> T to ArraySpace -> T

        Parameters
        ----------

        f : dict -> T

        Returns
        -------
        f : array -> T
        """
        return Compose(f, self.rmap)


class ListArrayOrdering(object):
    """
    An ordering for a list to an array space. Takes also non theano.tensors.
    Modified from pymc3 blocking.

    Parameters
    ----------
    list_arrays : list
        :class:`numpy.ndarray` or :class:`theano.tensor.Tensor`
    intype : str
        defining the input type 'tensor' or 'numpy'
    """

    def __init__(self, list_arrays, intype='numpy'):
        self.vmap = []
        dim = 0

        count = 0
        for array in list_arrays:
            if intype == 'tensor':
                array = array.tag.test_value
            elif intype == 'numpy':
                pass

            slc = slice(dim, dim + array.size)
            self.vmap.append(DataMap(
                count, slc, array.shape, array.dtype))
            dim += array.size
            count += 1

        self.dimensions = dim


class ListToArrayBijection(object):
    """
    A mapping between a List of arrays and an array space

    Parameters
    ----------
    ordering : :class:`ListArrayOrdering`
    list_arrays : list
        of :class:`numpy.ndarray`
    """

    def __init__(self, ordering, list_arrays):
        self.ordering = ordering
        self.list_arrays = list_arrays

    def fmap(self, list_arrays):
        """
        Maps values from List space to array space

        Parameters
        ----------
        list_arrays : list
            of :class:`numpy.ndarray`

        Returns
        -------
        array : :class:`numpy.ndarray`
            single array comprising all the input arrays
        """

        array = np.empty(self.ordering.dimensions)
        for list_ind, slc, _, _ in self.ordering.vmap:
            array[slc] = list_arrays[list_ind].ravel()
        return array

    def rmap(self, array):
        """
        Maps value from array space to List space
        Inverse operation of fmap.

        Parameters
        ----------
        array : :class:`numpy.ndarray`

        Returns
        -------
        a_list : list
            of :class:`numpy.ndarray`
        """

        a_list = copy.copy(self.list_arrays)

        for list_ind, slc, shp, dtype in self.ordering.vmap:
            a_list[list_ind] = np.atleast_1d(
                                        array)[slc].reshape(shp).astype(dtype)

        return a_list


class DictToVarBijection(object):
    """
    A mapping between a dict space and the array space for one element within the dict space
    """

    def __init__(self, var, idx, dpoint):
        self.var = str(var)
        self.idx = idx
        self.dpt = dpoint

    def map(self, dpt):
        return dpt[self.var][self.idx]

    def rmap(self, apt):
        dpt = self.dpt.copy()

        dvar = dpt[self.var].copy()
        dvar[self.idx] = apt

        dpt[self.var] = dvar

        return dpt

    def mapf(self, f):
        return Compose(f, self.rmap)


class Compose(object):
    """
    Compose two functions in a pickleable way
    """

    def __init__(self, fa, fb):
        self.fa = fa
        self.fb = fb

    def __call__(self, x):
        return self.fa(self.fb(x))
