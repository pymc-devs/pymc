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

from collections.abc import Sequence
from copy import copy

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.compile.builders import OpFromGraph
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Variable
from pytensor.raise_op import Assert
from pytensor.tensor.random.basic import IntegersRV
from pytensor.tensor.variable import TensorVariable

from pymc.exceptions import ShapeError
from pymc.pytensorf import convert_data, rvs_in_graph
from pymc.vartypes import isgenerator

if typing.TYPE_CHECKING:
    from pymc.model.core import Model

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

    rng = pt.random.shared_rng(seed=None)
    rng_update, mb_indices = minibatch_index(
        0, upper, size=batch_size, rng=rng, return_next_rng=True
    )
    mb_tensors = [tensor[mb_indices] for tensor in tensors]

    # Wrap graph in OFG so it's easily identifiable and not rewritten accidentally
    *mb_tensors, _ = MinibatchOp([*tensors, rng], [*mb_tensors, rng_update])(*tensors, rng)
    for i, r in enumerate(mb_tensors[:-1]):
        r.name = f"minibatch.{i}"

    return mb_tensors if len(variables) else mb_tensors[0]


def Data(
    name: str,
    value,
    *,
    dims: Sequence[str] | None = None,
    model: "Model | None" = None,
    **kwargs,
) -> SharedVariable:
    """Create a data container that registers a data variable with the model.

    The variable is registered as a :class:`~pytensor.compile.sharedvalue.SharedVariable`,
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
    value : array-like, DataFrame, or Series
        A value to associate with this variable. Accepts numpy arrays, lists,
        scipy sparse matrices, xarray DataArrays, and any DataFrame or Series
        supported by `narwhals <https://narwhals-dev.github.io/narwhals/>`_
        (pandas, polars, dask, pyarrow, modin, cudf, ibis, ...). The value is
        converted to a numpy array.
    dims : str or tuple of str, optional
        Dimension names of the data variable. See ArviZ documentation for more
        information about dimensions and coordinates: :ref:`arviz:quickstart`.
        If not specified, the data variable will not have dimension names.
    model : pymc.Model, optional
        Model to which to add the data variable. If not specified, the data
        variable will be added to the model on the context stack.
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

    try:
        model = modelcontext(model)
    except TypeError:
        raise TypeError(
            "No model on context stack, which is needed to instantiate a data container. "
            "Add variable inside a 'with model:' block."
        )
    name = model.name_for(name)

    if isgenerator(value):
        raise NotImplementedError(
            "Generator type data is no longer supported with pm.Data.",
            # It messes up InferenceData and can't be the input to a SharedVariable.
        )

    if isinstance(value, list | tuple):
        # Promote here so the inferred dtype (e.g. int64 for an index list) survives
        # `convert_data`, instead of being floatX'd by the "no dtype" fallback.
        value = np.asarray(value)

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
        for d, dname in enumerate(dims):
            if dname not in model.dim_lengths and dname is not None:
                model.add_coord(name=dname, values=None, length=xshape[d])

    model.register_data_var(x, dims=dims)

    return x
