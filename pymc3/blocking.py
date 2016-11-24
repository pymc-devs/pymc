"""
pymc3.blocking

Classes for working with subsets of parameters.
"""
import numpy as np
import collections

__all__ = ['ArrayOrdering', 'DictToArrayBijection', 'DictToVarBijection']

VarMap = collections.namedtuple('VarMap', 'var, slc, shp, dtyp')

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
