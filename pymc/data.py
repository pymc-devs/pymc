#   Copyright 2023 The PyMC Developers
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

import io
import urllib.request
import warnings

from copy import copy
from typing import Dict, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import xarray as xr

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.raise_op import Assert
from pytensor.scalar import Cast
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.basic import IntegersRV
from pytensor.tensor.subtensor import AdvancedSubtensor
from pytensor.tensor.type import TensorType
from pytensor.tensor.variable import TensorConstant, TensorVariable

import pymc as pm

from pymc.pytensorf import convert_observed_data

__all__ = [
    "get_data",
    "GeneratorAdapter",
    "Minibatch",
    "Data",
    "ConstantData",
    "MutableData",
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
    with urllib.request.urlopen(BASE_URL.format(filename=filename)) as handle:
        content = handle.read()
    return io.BytesIO(content)


class GenTensorVariable(TensorVariable):
    def __init__(self, op, type, name=None):
        super().__init__(type=type, owner=None, name=name)
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
        self.tensortype = TensorType(self.test_value.dtype, ((False,) * self.test_value.ndim))

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


class MinibatchIndexRV(IntegersRV):
    _print_name = ("minibatch_index", r"\operatorname{minibatch\_index}")

    # Work-around for https://github.com/pymc-devs/pytensor/issues/97
    def make_node(self, rng, *args, **kwargs):
        if rng is None:
            rng = pytensor.shared(np.random.default_rng())
        return super().make_node(rng, *args, **kwargs)


minibatch_index = MinibatchIndexRV()


def is_minibatch(v: TensorVariable) -> bool:
    return (
        isinstance(v.owner.op, AdvancedSubtensor)
        and isinstance(v.owner.inputs[1].owner.op, MinibatchIndexRV)
        and valid_for_minibatch(v.owner.inputs[0])
    )


def valid_for_minibatch(v: TensorVariable) -> bool:
    return (
        v.owner is None
        # The only PyTensor operation we allow on observed data is type casting
        # Although we could allow for any graph that does not depend on other RVs
        or (
            isinstance(v.owner.op, Elemwise)
            and v.owner.inputs[0].owner is None
            and isinstance(v.owner.op.scalar_op, Cast)
        )
    )


def assert_all_scalars_equal(scalar, *scalars):
    if len(scalars) == 0:
        return scalar
    else:
        return Assert(
            "All variables shape[0] in Minibatch should be equal, check your Minibatch(data1, data2, ...) code"
        )(scalar, pt.all([pt.eq(scalar, s) for s in scalars]))


def Minibatch(variable: TensorVariable, *variables: TensorVariable, batch_size: int):
    """Get random slices from variables from the leading dimension.

    Parameters
    ----------
    variable: TensorVariable
    variables: TensorVariable
    batch_size: int

    Examples
    --------
    >>> data1 = np.random.randn(100, 10)
    >>> data2 = np.random.randn(100, 20)
    >>> mdata1, mdata2 = Minibatch(data1, data2, batch_size=10)
    """

    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")

    tensor, *tensors = tuple(map(pt.as_tensor, (variable, *variables)))
    upper = assert_all_scalars_equal(*[t.shape[0] for t in (tensor, *tensors)])
    slc = minibatch_index(0, upper, size=batch_size)
    for i, v in enumerate((tensor, *tensors)):
        if not valid_for_minibatch(v):
            raise ValueError(
                f"{i}: {v} is not valid for Minibatch, only constants or constants.astype(dtype) are allowed"
            )
    result = tuple([v[slc] for v in (tensor, *tensors)])
    for i, r in enumerate(result):
        r.name = f"minibatch.{i}"
    return result if tensors else result[0]


def determine_coords(
    model,
    value: Union[pd.DataFrame, pd.Series, xr.DataArray],
    dims: Optional[Sequence[Optional[str]]] = None,
    coords: Optional[Dict[str, Union[Sequence, np.ndarray]]] = None,
) -> Tuple[Dict[str, Union[Sequence, np.ndarray]], Sequence[Optional[str]]]:
    """Determines coordinate values from data or the model (via ``dims``)."""
    if coords is None:
        coords = {}

    dim_name = None
    # If value is a df or a series, we interpret the index as coords:
    if hasattr(value, "index"):
        if dims is not None:
            dim_name = dims[0]
        if dim_name is None and value.index.name is not None:
            dim_name = value.index.name
        if dim_name is not None:
            coords[dim_name] = value.index

    # If value is a df, we also interpret the columns as coords:
    if hasattr(value, "columns"):
        if dims is not None:
            dim_name = dims[1]
        if dim_name is None and value.columns.name is not None:
            dim_name = value.columns.name
        if dim_name is not None:
            coords[dim_name] = value.columns

    if isinstance(value, xr.DataArray):
        if dims is not None:
            for dim in dims:
                dim_name = dim
                # str is applied because dim entries may be None
                coords[str(dim_name)] = cast(xr.DataArray, value[dim]).to_numpy()

    if isinstance(value, np.ndarray) and dims is not None:
        if len(dims) != value.ndim:
            raise pm.exceptions.ShapeError(
                "Invalid data shape. The rank of the dataset must match the " "length of `dims`.",
                actual=value.shape,
                expected=value.ndim,
            )
        for size, dim in zip(value.shape, dims):
            coord = model.coords.get(dim, None)
            if coord is None and dim is not None:
                coords[dim] = range(size)

    if dims is None:
        # TODO: Also determine dim names from the index
        dims = [None] * np.ndim(value)

    return coords, dims


def ConstantData(
    name: str,
    value,
    *,
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Dict[str, Union[Sequence, np.ndarray]]] = None,
    export_index_as_coords=False,
    infer_dims_and_coords=False,
    **kwargs,
) -> TensorConstant:
    """Alias for ``pm.Data(..., mutable=False)``.

    Registers the ``value`` as a :class:`~pytensor.tensor.TensorConstant` with the model.
    For more information, please reference :class:`pymc.Data`.
    """
    if export_index_as_coords:
        infer_dims_and_coords = export_index_as_coords
        warnings.warn(
            "Deprecation warning: 'export_index_as_coords; is deprecated and will be removed in future versions. Please use 'infer_dims_and_coords' instead.",
            DeprecationWarning,
        )

    var = Data(
        name,
        value,
        dims=dims,
        coords=coords,
        infer_dims_and_coords=infer_dims_and_coords,
        mutable=False,
        **kwargs,
    )
    return cast(TensorConstant, var)


def MutableData(
    name: str,
    value,
    *,
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Dict[str, Union[Sequence, np.ndarray]]] = None,
    export_index_as_coords=False,
    infer_dims_and_coords=False,
    **kwargs,
) -> SharedVariable:
    """Alias for ``pm.Data(..., mutable=True)``.

    Registers the ``value`` as a :class:`~pytensor.compile.sharedvalue.SharedVariable`
    with the model. For more information, please reference :class:`pymc.Data`.
    """
    if export_index_as_coords:
        infer_dims_and_coords = export_index_as_coords
        warnings.warn(
            "Deprecation warning: 'export_index_as_coords; is deprecated and will be removed in future versions. Please use 'infer_dims_and_coords' instead.",
            DeprecationWarning,
        )

    var = Data(
        name,
        value,
        dims=dims,
        coords=coords,
        infer_dims_and_coords=infer_dims_and_coords,
        mutable=True,
        **kwargs,
    )
    return cast(SharedVariable, var)


def Data(
    name: str,
    value,
    *,
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Dict[str, Union[Sequence, np.ndarray]]] = None,
    export_index_as_coords=False,
    infer_dims_and_coords=False,
    mutable: Optional[bool] = None,
    **kwargs,
) -> Union[SharedVariable, TensorConstant]:
    """Data container that registers a data variable with the model.

    Depending on the ``mutable`` setting (default: True), the variable
    is registered as a :class:`~pytensor.compile.sharedvalue.SharedVariable`,
    enabling it to be altered in value and shape, but NOT in dimensionality using
    :func:`pymc.set_data`.

    To set the value of the data container variable, check out
    :meth:`pymc.Model.set_data`.

    When making predictions or doing posterior predictive sampling, the shape of the
    registered data variable will most likely need to be changed.  If you encounter an
    PyTensor shape mismatch error, refer to the documentation for
    :meth:`pymc.model.set_data`.

    For more information, read the notebook :ref:`nb:data_container`.

    Parameters
    ----------
    name : str
        The name for this variable.
    value : array_like or pandas.Series, pandas.Dataframe
        A value to associate with this variable.
    dims : str or tuple of str, optional
        Dimension names of the random variables (as opposed to the shapes of these
        random variables). Use this when ``value`` is a pandas Series or DataFrame. The
        ``dims`` will then be the name of the Series / DataFrame's columns. See ArviZ
        documentation for more information about dimensions and coordinates:
        :ref:`arviz:quickstart`.
        If this parameter is not specified, the random variables will not have dimension
        names.
    coords : dict, optional
        Coordinate values to set for new dimensions introduced by this ``Data`` variable.
    export_index_as_coords : bool
        Deprecated, previous version of "infer_dims_and_coords"
    infer_dims_and_coords : bool, default=False
        If True, the ``Data`` container will try to infer what the coordinates
        and dimension names should be if there is an index in ``value``.
    mutable : bool, optional
        Switches between creating a :class:`~pytensor.compile.sharedvalue.SharedVariable`
        (``mutable=True``) vs. creating a :class:`~pytensor.tensor.TensorConstant`
        (``mutable=False``).
        Consider using :class:`pymc.ConstantData` or :class:`pymc.MutableData` as less
        verbose alternatives to ``pm.Data(..., mutable=...)``.
        If this parameter is not specified, the value it takes will depend on the
        version of the package. Since ``v4.1.0`` the default value is
        ``mutable=False``, with previous versions having ``mutable=True``.
    **kwargs : dict, optional
        Extra arguments passed to :func:`pytensor.shared`.

    Examples
    --------
    >>> import pymc as pm
    >>> import numpy as np
    >>> # We generate 10 datasets
    >>> true_mu = [np.random.randn() for _ in range(10)]
    >>> observed_data = [mu + np.random.randn(20) for mu in true_mu]

    >>> with pm.Model() as model:
    ...     data = pm.MutableData('data', observed_data[0])
    ...     mu = pm.Normal('mu', 0, 10)
    ...     pm.Normal('y', mu=mu, sigma=1, observed=data)

    >>> # Generate one trace for each dataset
    >>> idatas = []
    >>> for data_vals in observed_data:
    ...     with model:
    ...         # Switch out the observed dataset
    ...         model.set_data('data', data_vals)
    ...         idatas.append(pm.sample())
    """
    if coords is None:
        coords = {}

    if isinstance(value, list):
        value = np.array(value)

    # Add data container to the named variables of the model.
    model = pm.Model.get_context(error_if_none=False)
    if model is None:
        raise TypeError(
            "No model on context stack, which is needed to instantiate a data container. "
            "Add variable inside a 'with model:' block."
        )
    name = model.name_for(name)

    # `convert_observed_data` takes care of parameter `value` and
    # transforms it to something digestible for PyTensor.
    arr = convert_observed_data(value)
    if isinstance(arr, np.ma.MaskedArray):
        raise NotImplementedError(
            "Masked arrays or arrays with `nan` entries are not supported. "
            "Pass them directly to `observed` if you want to trigger auto-imputation"
        )

    if mutable is None:
        warnings.warn(
            "The `mutable` kwarg was not specified. Before v4.1.0 it defaulted to `pm.Data(mutable=True)`,"
            " which is equivalent to using `pm.MutableData()`."
            " In v4.1.0 the default changed to `pm.Data(mutable=False)`, equivalent to `pm.ConstantData`."
            " Use `pm.ConstantData`/`pm.MutableData` or pass `pm.Data(..., mutable=False/True)` to avoid this warning.",
            UserWarning,
        )
        mutable = False
    if mutable:
        x = pytensor.shared(arr, name, **kwargs)
    else:
        x = pt.as_tensor_variable(arr, name, **kwargs)

    if isinstance(dims, str):
        dims = (dims,)
    if not (dims is None or len(dims) == x.ndim):
        raise pm.exceptions.ShapeError(
            "Length of `dims` must match the dimensions of the dataset.",
            actual=len(dims),
            expected=x.ndim,
        )

    # Optionally infer coords and dims from the input value.
    if export_index_as_coords:
        infer_dims_and_coords = export_index_as_coords
        warnings.warn(
            "Deprecation warning: 'export_index_as_coords; is deprecated and will be removed in future versions. Please use 'infer_dims_and_coords' instead.",
            DeprecationWarning,
        )

    if infer_dims_and_coords:
        coords, dims = determine_coords(model, value, dims)

    if dims:
        if not mutable:
            # Use the dimension lengths from the before it was tensorified.
            # These can still be tensors, but in many cases they are numeric.
            xshape = np.shape(arr)
        else:
            xshape = x.shape
        # Register new dimension lengths
        for d, dname in enumerate(dims):
            if dname not in model.dim_lengths:
                model.add_coord(
                    name=dname,
                    # Note: Coordinate values can't be taken from
                    # the value, because it could be N-dimensional.
                    values=coords.get(dname, None),
                    mutable=mutable,
                    length=xshape[d],
                )

    model.add_named_variable(x, dims=dims)

    return x
