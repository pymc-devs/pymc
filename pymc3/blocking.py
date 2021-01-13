#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
pymc3.blocking

Classes for working with subsets of parameters.
"""
import collections
import copy

import numpy as np

from pymc3.util import get_var_name

__all__ = ["ArrayOrdering", "DictToArrayBijection", "DictToVarBijection"]

VarMap = collections.namedtuple("VarMap", "var, slc, shp, dtyp")
DataMap = collections.namedtuple("DataMap", "list_ind, slc, shp, dtype, name")


# TODO Classes and methods need to be fully documented.


class ArrayOrdering:
    """
    An ordering for an array space
    """

    def __init__(self, vars):
        self.vmap = []
        self.by_name = {}
        self.size = 0

        for var in vars:
            name = var.name
            if name is None:
                raise ValueError("Unnamed variable in ArrayOrdering.")
            if name in self.by_name:
                raise ValueError("Name of variable not unique: %s." % name)
            if not hasattr(var, "dshape") or not hasattr(var, "dsize"):
                raise ValueError("Shape of variable not known %s" % name)

            slc = slice(self.size, self.size + var.dsize)
            varmap = VarMap(name, slc, var.dshape, var.dtype)
            self.vmap.append(varmap)
            self.by_name[name] = varmap
            self.size += var.dsize

    def __getitem__(self, key):
        return self.by_name[key]


class DictToArrayBijection:
    """
    A mapping between a dict space and an array space
    """

    def __init__(self, ordering, dpoint):
        self.ordering = ordering
        self.dpt = dpoint

        # determine smallest float dtype that will fit all data
        if all([x.dtyp == "float16" for x in ordering.vmap]):
            self.array_dtype = "float16"
        elif all([x.dtyp == "float32" for x in ordering.vmap]):
            self.array_dtype = "float32"
        else:
            self.array_dtype = "float64"

    def map(self, dpt):
        """
        Maps value from dict space to array space

        Parameters
        ----------
        dpt: dict
        """
        apt = np.empty(self.ordering.size, dtype=self.array_dtype)
        for var, slc, _, _ in self.ordering.vmap:
            apt[slc] = dpt[var].ravel()
        return apt

    def rmap(self, apt):
        """
        Maps value from array space to dict space

        Parameters
        ----------
        apt: array
        """
        dpt = self.dpt.copy()

        for var, slc, shp, dtyp in self.ordering.vmap:
            dpt[var] = np.atleast_1d(apt)[slc].reshape(shp).astype(dtyp)

        return dpt

    def mapf(self, f):
        """
         function f: DictSpace -> T to ArraySpace -> T

        Parameters
        ----------

        f: dict -> T

        Returns
        -------
        f: array -> T
        """
        return Compose(f, self.rmap)


class ListArrayOrdering:
    """
    An ordering for a list to an array space. Takes also non theano.tensors.
    Modified from pymc3 blocking.

    Parameters
    ----------
    list_arrays: list
        :class:`numpy.ndarray` or :class:`theano.tensor.Tensor`
    intype: str
        defining the input type 'tensor' or 'numpy'
    """

    def __init__(self, list_arrays, intype="numpy"):
        if intype not in {"tensor", "numpy"}:
            raise ValueError("intype not in {'tensor', 'numpy'}")
        self.vmap = []
        self.intype = intype
        self.size = 0
        for array in list_arrays:
            if self.intype == "tensor":
                name = array.name
                array = array.tag.test_value
            else:
                name = "numpy"

            slc = slice(self.size, self.size + array.size)
            self.vmap.append(DataMap(len(self.vmap), slc, array.shape, array.dtype, name))
            self.size += array.size


class ListToArrayBijection:
    """
    A mapping between a List of arrays and an array space

    Parameters
    ----------
    ordering: :class:`ListArrayOrdering`
    list_arrays: list
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
        list_arrays: list
            of :class:`numpy.ndarray`

        Returns
        -------
        array: :class:`numpy.ndarray`
            single array comprising all the input arrays
        """

        array = np.empty(self.ordering.size)
        for list_ind, slc, _, _, _ in self.ordering.vmap:
            array[slc] = list_arrays[list_ind].ravel()
        return array

    def dmap(self, dpt):
        """
        Maps values from dict space to List space

        Parameters
        ----------
        list_arrays: list
            of :class:`numpy.ndarray`

        Returns
        -------
        point
        """
        a_list = copy.copy(self.list_arrays)

        for list_ind, _, _, _, var in self.ordering.vmap:
            a_list[list_ind] = dpt[var].ravel()

        return a_list

    def rmap(self, array):
        """
        Maps value from array space to List space
        Inverse operation of fmap.

        Parameters
        ----------
        array: :class:`numpy.ndarray`

        Returns
        -------
        a_list: list
            of :class:`numpy.ndarray`
        """

        a_list = copy.copy(self.list_arrays)

        for list_ind, slc, shp, dtype, _ in self.ordering.vmap:
            a_list[list_ind] = np.atleast_1d(array)[slc].reshape(shp).astype(dtype)

        return a_list


class DictToVarBijection:
    """
    A mapping between a dict space and the array space for one element within the dict space
    """

    def __init__(self, var, idx, dpoint):
        self.var = get_var_name(var)
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


class Compose:
    """
    Compose two functions in a pickleable way
    """

    def __init__(self, fa, fb):
        self.fa = fa
        self.fb = fb

    def __call__(self, x):
        return self.fa(self.fb(x))
