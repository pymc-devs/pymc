#   Copyright 2024 - present The PyMC Developers
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
import typing
import urllib.request
import warnings

from collections.abc import Sequence
from copy import copy
from functools import singledispatch

import narwhals as nw
import numpy as np
import pytensor
import pytensor.tensor as pt
import xarray as xr

from narwhals.typing import IntoDataFrame, IntoSeries
from pytensor.compile import SharedVariable
from pytensor.compile.builders import OpFromGraph
from pytensor.graph.basic import Variable
from pytensor.raise_op import Assert
from pytensor.tensor.random.basic import IntegersRV
from pytensor.tensor.variable import TensorConstant, TensorVariable
from pytensor.xtensor.type import XTensorConstant
from scipy.sparse import sparray, spmatrix

from pymc.exceptions import ShapeError
from pymc.pytensorf import convert_data, rvs_in_graph
from pymc.vartypes import isgenerator

if typing.TYPE_CHECKING:
    from pymc.model.core import Model

InputDataType: typing.TypeAlias = (
    np.ndarray
    | np.ma.MaskedArray
    | list
    | sparray
    | spmatrix
    | IntoDataFrame
    | IntoSeries
    | xr.DataArray
    | Variable
)

__all__ = [
    "Data",
    "Minibatch",
    "get_data",
]
BASE_URL = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/{filename}"


def get_data(filename):
    """Return a BytesIO object for a package data file.

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


class MinibatchIndexRV(IntegersRV):
    _print_name = ("minibatch_index", r"\operatorname{minibatch\_index}")


minibatch_index = MinibatchIndexRV()


class MinibatchOp(OpFromGraph):
    """Encapsulate Minibatch random draws in an opaque OFG."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, inline=True)

    def __str__(self):
        return "Minibatch"


def is_valid_observed(v) -> bool:
    if not isinstance(v, Variable):
        # Non-symbolic constant
        return True

    if v.owner is None:
        # Symbolic root variable (constant or not)
        return True

    return (
        not rvs_in_graph(v)
        # Or Minibatch
        or (
            isinstance(v.owner.op, MinibatchOp)
            and all(is_valid_observed(inp) for inp in v.owner.inputs)
        )
    )


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

    tensors = tuple(map(pt.as_tensor, (variable, *variables)))
    for i, v in enumerate(tensors):
        if not is_valid_observed(v):
            raise ValueError(
                f"{i}: {v} is not valid for Minibatch, only non-random variables are allowed"
            )

    upper = tensors[0].shape[0]
    if len(tensors) > 1:
        upper = Assert(
            "All variables shape[0] in Minibatch should be equal, check your Minibatch(data1, data2, ...) code"
        )(upper, pt.all([pt.eq(upper, other_tensor.shape[0]) for other_tensor in tensors[1:]]))

    rng = pytensor.shared(np.random.default_rng())
    rng_update, mb_indices = minibatch_index(0, upper, size=batch_size, rng=rng).owner.outputs
    mb_tensors = [tensor[mb_indices] for tensor in tensors]

    # Wrap graph in OFG so it's easily identifiable and not rewritten accidentally
    *mb_tensors, _ = MinibatchOp([*tensors, rng], [*mb_tensors, rng_update])(*tensors, rng)
    for i, r in enumerate(mb_tensors[:-1]):
        r.name = f"minibatch.{i}"

    return mb_tensors if len(variables) else mb_tensors[0]


@singledispatch
def prepare_user_data(data):
    try:
        # We can't explicitly dispatch on all possible DataFrame/Series types supported by narwhals without making them
        # package dependencies. So in the generic function we can just try to convert using narwhals, then
        # dispatch based on the native Narwhals types if it works.
        df = nw.from_native(data, allow_series=True)
        return prepare_user_data(df)
    except TypeError:
        raise TypeError(f"Cannot convert data of type {type(data)}")


@prepare_user_data.register(float | int | complex | bool)
def _prepare_scalar(data: float | int | complex | bool) -> np.ndarray:
    return np.array(data)


@prepare_user_data.register(nw.DataFrame)
def _prepare_dataframe(data: nw.DataFrame) -> np.ndarray | np.ma.MaskedArray:
    array = data.to_numpy()

    # Some backends (like polars) distinguish between NaN and null/missing values.
    # We consider both as missing values to be masked.
    mask = data.with_columns(nw.all().fill_nan(None).is_null()).to_numpy()
    if mask.any():
        return np.ma.MaskedArray(array, mask=mask)
    return array


@prepare_user_data.register(nw.Series)
def _prepare_series(data: nw.Series) -> np.ndarray | np.ma.MaskedArray:
    array = data.to_numpy()

    # Some backends (like polars) distinguish between NaN and null/missing values.
    # We consider both as missing values to be masked.
    mask = data.fill_nan(None).is_null().to_numpy()
    if mask.any():
        return np.ma.MaskedArray(array, mask=mask)
    return array


@prepare_user_data.register(nw.LazyFrame)
def _prepare_lazyframe(data: nw.LazyFrame) -> np.ndarray | np.ma.MaskedArray:
    df = data.collect()
    return prepare_user_data(df)


@prepare_user_data.register(np.ndarray)
def _prepare_ndarray(data: np.ndarray) -> np.ndarray | np.ma.MaskedArray:
    mask = np.isnan(data)
    if mask.any():
        return np.ma.MaskedArray(data, mask=mask)
    return data


@prepare_user_data.register(list)
def _prepare_list(data: list) -> np.ndarray | np.ma.MaskedArray:
    array = np.array(data)
    return prepare_user_data(array)


@prepare_user_data.register(np.ma.MaskedArray)
def _prepare_masked_array(data: np.ma.MaskedArray) -> np.ndarray | np.ma.MaskedArray:
    if not data.mask.any():
        return data.filled()
    return data


@prepare_user_data.register(sparray)
@prepare_user_data.register(spmatrix)
def _prepare_sparse(data: sparray | spmatrix) -> sparray | spmatrix:
    # TODO: Handle missing values?
    return data


@prepare_user_data.register(xr.DataArray)
def _prepare_data_array(data: xr.DataArray) -> np.ndarray | np.ma.MaskedArray:
    values = data.to_numpy()
    return prepare_user_data(values)


@prepare_user_data.register(TensorConstant | XTensorConstant)
def _prepare_pytensor_variable(
    data: TensorConstant | XTensorConstant,
) -> np.ndarray | np.ma.MaskedArray:
    return prepare_user_data(data.data)


@prepare_user_data.register(SharedVariable)
def _prepare_shared_variable(data: SharedVariable) -> np.ndarray | np.ma.MaskedArray:
    return prepare_user_data(data.get_value(borrow=True))


def Data(
    name: str,
    value: InputDataType,
    *,
    dims: Sequence[str] | None = None,
    coords: dict[str, Sequence | np.ndarray] | None = None,
    infer_dims_and_coords: bool = False,
    model: "Model | None" = None,
    **kwargs,
) -> SharedVariable:
    """Create a data container that registers a data variable with the model.

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
    value : array_like or narwhals-compatible DataFrame/Series
        A value to associate with this variable. Accepts numpy arrays, lists, or any
        narwhals-compatible DataFrame/Series (pandas, polars, dask, pyarrow, etc.).
        Will be converted to a numpy array.
    dims : str or tuple of str, optional
        Dimension names of the data variable. See ArviZ documentation for more
        information about dimensions and coordinates: :ref:`arviz:quickstart`.
        If this parameter is not specified, the data variable will not have dimension
        names.
    coords : dict, optional
        Coordinate values to set for new dimensions introduced by this ``Data`` variable.

        .. warning::
            This parameter is deprecated and will be removed in future versions. Add coordinates
            explicitly by passing them to :class:`pymc.Model` during model creation instead.

    infer_dims_and_coords : bool, default=False
        If True, the ``Data`` container will try to infer what the coordinates
        and dimension names should be if there is an index in ``value``.

        .. warning::
            This parameter is deprecated and will be removed in future versions. Add coordinates
            explicitly by passing them to :class:`pymc.Model` during model creation instead.

    model : pymc.Model, optional
        Model to which to add the data variable. If not specified, the data variable
        will be added to the model on the context stack.
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
    ...     data = pm.Data("data", observed_data[0])
    ...     mu = pm.Normal("mu", 0, 10)
    ...     pm.Normal("y", mu=mu, sigma=1, observed=data)

    >>> # Generate one trace for each dataset
    >>> idatas = []
    >>> for data_vals in observed_data:
    ...     with model:
    ...         # Switch out the observed dataset
    ...         model.set_data("data", data_vals)
    ...         idatas.append(pm.sample())
    """
    from pymc.model.core import modelcontext

    if coords is not None:
        warnings.warn(
            "`coords` parameter in `pm.Data` is deprecated and will be removed in future versions. It is "
            "no longer respected. Add coordinates explicitly by passing them to `pymc.Model` during model creation "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if infer_dims_and_coords:
        warnings.warn(
            "`infer_dims_and_coords` parameter in `pm.Data` is deprecated and will be removed in future "
            "versions. It is no longer respected. Add coordinates explicitly by passing them to `pymc.Model` during "
            "model creation instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    try:
        model = modelcontext(model)
    except TypeError:
        raise TypeError(
            "No model on context stack, which is needed to instantiate a data container. "
            "Add variable inside a 'with model:' block."
        )
    name = model.name_for(name)

    # Transform `value` it to something digestible for PyTensor.
    if isgenerator(value):
        raise NotImplementedError(
            "Generator type data is no longer supported with pm.Data.",
            # It messes up InferenceData and can't be the input to a SharedVariable.
        )

    value = prepare_user_data(value)
    arr = convert_data(value)

    if isinstance(arr, np.ma.MaskedArray):
        raise NotImplementedError(
            "Masked arrays or arrays with `nan` entries are not supported. "
            "Pass them directly to `observed` if you want to trigger auto-imputation"
        )

    x = pytensor.shared(arr, name, **kwargs)

    if isinstance(dims, str):
        dims = (dims,)
    if not (dims is None or len(dims) == x.ndim):
        raise ShapeError(
            "Length of `dims` must match the dimensions of the dataset.",
            actual=len(dims),
            expected=x.ndim,
        )

    if dims:
        xshape = x.shape
        # Register new dimension lengths
        for d, dname in enumerate(dims):
            if dname not in model.dim_lengths and dname is not None:
                model.add_coord(
                    name=dname,
                    # Note: Coordinate values can't be taken from
                    # the value, because it could be N-dimensional.
                    values=None,
                    length=xshape[d],
                )

    model.register_data_var(x, dims=dims)

    return x
