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

import collections
import io
import os
import pkgutil
import urllib.request

from copy import copy
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import theano
import theano.tensor as tt

from theano.graph.basic import Apply

import pymc3 as pm

__all__ = [
    "get_data",
    "GeneratorAdapter",
    "Minibatch",
    "align_minibatches",
    "Data",
]
BASE_URL = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/{filename}"


def get_data(filename):
    """Returns a BytesIO object for a package data file.

    Parameters
    ----------
    filename: str
        file to load

    Returns
    -------
    BytesIO of the data
    """
    data_pkg = "pymc3.tests"
    try:
        content = pkgutil.get_data(data_pkg, os.path.join("data", filename))
    except FileNotFoundError:
        with urllib.request.urlopen(BASE_URL.format(filename=filename)) as handle:
            content = handle.read()
    return io.BytesIO(content)


class GenTensorVariable(tt.TensorVariable):
    def __init__(self, op, type, name=None):
        super().__init__(type=type, name=name)
        self.op = op

    def set_gen(self, gen):
        self.op.set_gen(gen)

    def set_default(self, value):
        self.op.set_default(value)

    def clone(self):
        cp = self.__class__(self.op, self.type, self.name)
        cp.tag = copy(self.tag)
        return cp


class GeneratorAdapter:
    """
    Helper class that helps to infer data type of generator with looking
    at the first item, preserving the order of the resulting generator
    """

    def make_variable(self, gop, name=None):
        var = GenTensorVariable(gop, self.tensortype, name)
        var.tag.test_value = self.test_value
        return var

    def __init__(self, generator):
        if not pm.vartypes.isgenerator(generator):
            raise TypeError("Object should be generator like")
        self.test_value = pm.smartfloatX(copy(next(generator)))
        # make pickling potentially possible
        self._yielded_test_value = False
        self.gen = generator
        self.tensortype = tt.TensorType(self.test_value.dtype, ((False,) * self.test_value.ndim))

    # python3 generator
    def __next__(self):
        if not self._yielded_test_value:
            self._yielded_test_value = True
            return self.test_value
        else:
            return pm.smartfloatX(copy(next(self.gen)))

    # python2 generator
    next = __next__

    def __iter__(self):
        return self

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))


class Minibatch(tt.TensorVariable):
    """Multidimensional minibatch that is pure TensorVariable

    Parameters
    ----------
    data: np.ndarray
        initial data
    batch_size: ``int`` or ``List[int|tuple(size, random_seed)]``
        batch size for inference, random seed is needed
        for child random generators
    dtype: ``str``
        cast data to specific type
    broadcastable: tuple[bool]
        change broadcastable pattern that defaults to ``(False, ) * ndim``
    name: ``str``
        name for tensor, defaults to "Minibatch"
    random_seed: ``int``
        random seed that is used by default
    update_shared_f: ``callable``
        returns :class:`ndarray` that will be carefully
        stored to underlying shared variable
        you can use it to change source of
        minibatches programmatically
    in_memory_size: ``int`` or ``List[int|slice|Ellipsis]``
        data size for storing in ``theano.shared``

    Attributes
    ----------
    shared: shared tensor
        Used for storing data
    minibatch: minibatch tensor
        Used for training

    Notes
    -----
    Below is a common use case of Minibatch with variational inference.
    Importantly, we need to make PyMC3 "aware" that a minibatch is being used in inference.
    Otherwise, we will get the wrong :math:`logp` for the model.
    the density of the model ``logp`` that is affected by Minibatch. See more in the examples below.
    To do so, we need to pass the ``total_size`` parameter to the observed node, which correctly scales
    the density of the model ``logp`` that is affected by Minibatch. See more in the examples below.

    Examples
    --------
    Consider we have `data` as follows:

    >>> data = np.random.rand(100, 100)

    if we want a 1d slice of size 10 we do

    >>> x = Minibatch(data, batch_size=10)

    Note that your data is cast to ``floatX`` if it is not integer type
    But you still can add the ``dtype`` kwarg for :class:`Minibatch`
    if you need more control.

    If we want 10 sampled rows and columns
    ``[(size, seed), (size, seed)]`` we can use

    >>> x = Minibatch(data, batch_size=[(10, 42), (10, 42)], dtype='int32')
    >>> assert str(x.dtype) == 'int32'


    Or, more simply, we can use the default random seed = 42
    ``[size, size]``

    >>> x = Minibatch(data, batch_size=[10, 10])


    In the above, `x` is a regular :class:`TensorVariable` that supports any math operations:


    >>> assert x.eval().shape == (10, 10)


    You can pass the Minibatch `x` to your desired model:

    >>> with pm.Model() as model:
    ...     mu = pm.Flat('mu')
    ...     sd = pm.HalfNormal('sd')
    ...     lik = pm.Normal('lik', mu, sd, observed=x, total_size=(100, 100))


    Then you can perform regular Variational Inference out of the box


    >>> with model:
    ...     approx = pm.fit()


    Important note: :class:``Minibatch`` has ``shared``, and ``minibatch`` attributes
    you can call later:

    >>> x.set_value(np.random.laplace(size=(100, 100)))

    and minibatches will be then from new storage
    it directly affects ``x.shared``.
    A less convenient convenient, but more explicit, way to achieve the same
    thing:

    >>> x.shared.set_value(pm.floatX(np.random.laplace(size=(100, 100))))

    The programmatic way to change storage is as follows
    I import ``partial`` for simplicity
    >>> from functools import partial
    >>> datagen = partial(np.random.laplace, size=(100, 100))
    >>> x = Minibatch(datagen(), batch_size=10, update_shared_f=datagen)
    >>> x.update_shared()

    To be more concrete about how we create a minibatch, here is a demo:
    1. create a shared variable

        >>> shared = theano.shared(data)

    2. take a random slice of size 10:

        >>> ridx = pm.tt_rng().uniform(size=(10,), low=0, high=data.shape[0]-1e-10).astype('int64')

    3) take the resulting slice:

        >>> minibatch = shared[ridx]

    That's done. Now you can use this minibatch somewhere else.
    You can see that the implementation does not require a fixed shape
    for the shared variable. Feel free to use that if needed.
    *FIXME: What is "that" which we can use here?  A fixed shape?  Should this say
    "but feel free to put a fixed shape on the shared variable, if appropriate?"*

    Suppose you need to make some replacements in the graph, e.g. change the minibatch to testdata

    >>> node = x ** 2  # arbitrary expressions on minibatch `x`
    >>> testdata = pm.floatX(np.random.laplace(size=(1000, 10)))

    Then you should create a `dict` with replacements:

    >>> replacements = {x: testdata}
    >>> rnode = theano.clone(node, replacements)
    >>> assert (testdata ** 2 == rnode.eval()).all()

    *FIXME: In the following, what is the **reason** to replace the Minibatch variable with
    its shared variable?  And in the following, the `rnode` is a **new** node, not a modification
    of a previously existing node, correct?*
    To replace a minibatch with its shared variable you should do
    the same things. The Minibatch variable is accessible through the `minibatch` attribute.
    For example

    >>> replacements = {x.minibatch: x.shared}
    >>> rnode = theano.clone(node, replacements)

    For more complex slices some more code is needed that can seem not so clear

    >>> moredata = np.random.rand(10, 20, 30, 40, 50)

    The default ``total_size`` that can be passed to ``PyMC3`` random node
    is then ``(10, 20, 30, 40, 50)`` but can be less verbose in some cases

    1. Advanced indexing, ``total_size = (10, Ellipsis, 50)``

        >>> x = Minibatch(moredata, [2, Ellipsis, 10])

        We take the slice only for the first and last dimension

        >>> assert x.eval().shape == (2, 20, 30, 40, 10)

    2. Skipping a particular dimension, ``total_size = (10, None, 30)``:

        >>> x = Minibatch(moredata, [2, None, 20])
        >>> assert x.eval().shape == (2, 20, 20, 40, 50)

    3. Mixing both of these together, ``total_size = (10, None, 30, Ellipsis, 50)``:

        >>> x = Minibatch(moredata, [2, None, 20, Ellipsis, 10])
        >>> assert x.eval().shape == (2, 20, 20, 40, 10)
    """

    RNG = collections.defaultdict(list)  # type: Dict[str, List[Any]]

    @theano.config.change_flags(compute_test_value="raise")
    def __init__(
        self,
        data,
        batch_size=128,
        dtype=None,
        broadcastable=None,
        name="Minibatch",
        random_seed=42,
        update_shared_f=None,
        in_memory_size=None,
    ):
        if dtype is None:
            data = pm.smartfloatX(np.asarray(data))
        else:
            data = np.asarray(data, dtype)
        in_memory_slc = self.make_static_slices(in_memory_size)
        self.shared = theano.shared(data[in_memory_slc])
        self.update_shared_f = update_shared_f
        self.random_slc = self.make_random_slices(self.shared.shape, batch_size, random_seed)
        minibatch = self.shared[self.random_slc]
        if broadcastable is None:
            broadcastable = (False,) * minibatch.ndim
        minibatch = tt.patternbroadcast(minibatch, broadcastable)
        self.minibatch = minibatch
        super().__init__(self.minibatch.type, None, None, name=name)
        Apply(theano.compile.view_op, inputs=[self.minibatch], outputs=[self])
        self.tag.test_value = copy(self.minibatch.tag.test_value)

    def rslice(self, total, size, seed):
        if size is None:
            return slice(None)
        elif isinstance(size, int):
            rng = pm.tt_rng(seed)
            Minibatch.RNG[id(self)].append(rng)
            return rng.uniform(size=(size,), low=0.0, high=pm.floatX(total) - 1e-16).astype("int64")
        else:
            raise TypeError("Unrecognized size type, %r" % size)

    def __del__(self):
        del Minibatch.RNG[id(self)]

    @staticmethod
    def make_static_slices(user_size):
        if user_size is None:
            return [Ellipsis]
        elif isinstance(user_size, int):
            return slice(None, user_size)
        elif isinstance(user_size, (list, tuple)):
            slc = list()
            for i in user_size:
                if isinstance(i, int):
                    slc.append(i)
                elif i is None:
                    slc.append(slice(None))
                elif i is Ellipsis:
                    slc.append(Ellipsis)
                elif isinstance(i, slice):
                    slc.append(i)
                else:
                    raise TypeError("Unrecognized size type, %r" % user_size)
            return slc
        else:
            raise TypeError("Unrecognized size type, %r" % user_size)

    def make_random_slices(self, in_memory_shape, batch_size, default_random_seed):
        if batch_size is None:
            return [Ellipsis]
        elif isinstance(batch_size, int):
            slc = [self.rslice(in_memory_shape[0], batch_size, default_random_seed)]
        elif isinstance(batch_size, (list, tuple)):

            def check(t):
                if t is Ellipsis or t is None:
                    return True
                else:
                    if isinstance(t, (tuple, list)):
                        if not len(t) == 2:
                            return False
                        else:
                            return isinstance(t[0], int) and isinstance(t[1], int)
                    elif isinstance(t, int):
                        return True
                    else:
                        return False

            # end check definition
            if not all(check(t) for t in batch_size):
                raise TypeError(
                    "Unrecognized `batch_size` type, expected "
                    "int or List[int|tuple(size, random_seed)] where "
                    "size and random seed are both ints, got %r" % batch_size
                )
            batch_size = [(i, default_random_seed) if isinstance(i, int) else i for i in batch_size]
            shape = in_memory_shape
            if Ellipsis in batch_size:
                sep = batch_size.index(Ellipsis)
                begin = batch_size[:sep]
                end = batch_size[sep + 1 :]
                if Ellipsis in end:
                    raise ValueError(
                        "Double Ellipsis in `batch_size` is restricted, got %r" % batch_size
                    )
                if len(end) > 0:
                    shp_mid = shape[sep : -len(end)]
                    mid = [tt.arange(s) for s in shp_mid]
                else:
                    mid = []
            else:
                begin = batch_size
                end = []
                mid = []
            if (len(begin) + len(end)) > len(in_memory_shape.eval()):
                raise ValueError(
                    "Length of `batch_size` is too big, "
                    "number of ints is bigger that ndim, got %r" % batch_size
                )
            if len(end) > 0:
                shp_end = shape[-len(end) :]
            else:
                shp_end = np.asarray([])
            shp_begin = shape[: len(begin)]
            slc_begin = [
                self.rslice(shp_begin[i], t[0], t[1]) if t is not None else tt.arange(shp_begin[i])
                for i, t in enumerate(begin)
            ]
            slc_end = [
                self.rslice(shp_end[i], t[0], t[1]) if t is not None else tt.arange(shp_end[i])
                for i, t in enumerate(end)
            ]
            slc = slc_begin + mid + slc_end
        else:
            raise TypeError("Unrecognized size type, %r" % batch_size)
        return pm.theanof.ix_(*slc)

    def update_shared(self):
        if self.update_shared_f is None:
            raise NotImplementedError("No `update_shared_f` was provided to `__init__`")
        self.set_value(np.asarray(self.update_shared_f(), self.dtype))

    def set_value(self, value):
        self.shared.set_value(np.asarray(value, self.dtype))

    def clone(self):
        ret = self.type()
        ret.name = self.name
        ret.tag = copy(self.tag)
        return ret


def align_minibatches(batches=None):
    if batches is None:
        for rngs in Minibatch.RNG.values():
            for rng in rngs:
                rng.seed()
    else:
        for b in batches:
            if not isinstance(b, Minibatch):
                raise TypeError("{b} is not a Minibatch")
            for rng in Minibatch.RNG[id(b)]:
                rng.seed()


class Data:
    """Data container class that wraps the theano ``SharedVariable`` class
    and lets the model be aware of its inputs and outputs.

    Parameters
    ----------
    name: str
        The name for this variable
    value: {List, np.ndarray, pd.Series, pd.Dataframe}
        A value to associate with this variable
    dims: {str, tuple of str}, optional, default=None
        Dimension names of the random variables (as opposed to the shapes of these
        random variables). Use this when `value` is a pandas Series or DataFrame. The
        `dims` will then be the name of the Series / DataFrame's columns. See ArviZ
        documentation for more information about dimensions and coordinates:
        https://arviz-devs.github.io/arviz/notebooks/Introduction.html
    export_index_as_coords: bool, optional, default=False
        If True, the `Data` container will try to infer what the coordinates should be
        if there is an index in `value`.

    Examples
    --------

    >>> import pymc3 as pm
    >>> import numpy as np
    >>> # We generate 10 datasets
    >>> true_mu = [np.random.randn() for _ in range(10)]
    >>> observed_data = [mu + np.random.randn(20) for mu in true_mu]

    >>> with pm.Model() as model:
    ...     data = pm.Data('data', observed_data[0])
    ...     mu = pm.Normal('mu', 0, 10)
    ...     pm.Normal('y', mu=mu, sigma=1, observed=data)

    >>> # Generate one trace for each dataset
    >>> traces = []
    >>> for data_vals in observed_data:
    ...     with model:
    ...         # Switch out the observed dataset
    ...         pm.set_data({'data': data_vals})
    ...         traces.append(pm.sample())

    To set the value of the data container variable, check out
    :func:`pymc3.model.set_data()`.

    For more information, take a look at this example notebook
    https://docs.pymc.io/notebooks/data_container.html
    """

    def __new__(self, name, value, *, dims=None, export_index_as_coords=False):
        if isinstance(value, list):
            value = np.array(value)

        # Add data container to the named variables of the model.
        try:
            model = pm.Model.get_context()
        except TypeError:
            raise TypeError(
                "No model on context stack, which is needed to instantiate a data container. "
                "Add variable inside a 'with model:' block."
            )
        name = model.name_for(name)

        # `pm.model.pandas_to_array` takes care of parameter `value` and
        # transforms it to something digestible for pymc3
        shared_object = theano.shared(pm.model.pandas_to_array(value), name)

        if isinstance(dims, str):
            dims = (dims,)
        if not (dims is None or len(dims) == shared_object.ndim):
            raise pm.exceptions.ShapeError(
                "Length of `dims` must match the dimensions of the dataset.",
                actual=len(dims),
                expected=shared_object.ndim,
            )

        coords = self.set_coords(model, value, dims)

        if export_index_as_coords:
            model.add_coords(coords)

        # To draw the node for this variable in the graphviz Digraph we need
        # its shape.
        shared_object.dshape = tuple(shared_object.shape.eval())
        if dims is not None:
            shape_dims = model.shape_from_dims(dims)
            if shared_object.dshape != shape_dims:
                raise pm.exceptions.ShapeError(
                    "Data shape does not match with specified `dims`.",
                    actual=shared_object.dshape,
                    expected=shape_dims,
                )

        model.add_random_variable(shared_object, dims=dims)

        return shared_object

    @staticmethod
    def set_coords(model, value, dims=None):
        coords = {}

        # If value is a df or a series, we interpret the index as coords:
        if isinstance(value, (pd.Series, pd.DataFrame)):
            dim_name = None
            if dims is not None:
                dim_name = dims[0]
            if dim_name is None and value.index.name is not None:
                dim_name = value.index.name
            if dim_name is not None:
                coords[dim_name] = value.index

        # If value is a df, we also interpret the columns as coords:
        if isinstance(value, pd.DataFrame):
            dim_name = None
            if dims is not None:
                dim_name = dims[1]
            if dim_name is None and value.columns.name is not None:
                dim_name = value.columns.name
            if dim_name is not None:
                coords[dim_name] = value.columns

        if isinstance(value, np.ndarray) and dims is not None:
            if len(dims) != value.ndim:
                raise pm.exceptions.ShapeError(
                    "Invalid data shape. The rank of the dataset must match the "
                    "length of `dims`.",
                    actual=value.shape,
                    expected=value.ndim,
                )
            for size, dim in zip(value.shape, dims):
                coord = model.coords.get(dim, None)
                if coord is None:
                    coords[dim] = pd.RangeIndex(size, name=dim)

        return coords
