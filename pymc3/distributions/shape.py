#   Copyright 2021 The PyMC Developers
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
import warnings

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from aesara.graph.basic import Variable
from aesara.tensor.var import TensorVariable

from pymc3.aesaraf import change_rv_size, pandas_to_array
from pymc3.exceptions import ShapeError, ShapeWarning

# User-provided can be lazily specified as scalars
Shape = Union[int, TensorVariable, Sequence[Union[int, TensorVariable, type(Ellipsis)]]]
Dims = Union[str, Sequence[Union[str, None, type(Ellipsis)]]]
Size = Union[int, TensorVariable, Sequence[Union[int, TensorVariable]]]

# After conversion to vectors
WeakShape = Union[TensorVariable, Tuple[Union[int, TensorVariable, type(Ellipsis)], ...]]
WeakDims = Tuple[Union[str, None, type(Ellipsis)], ...]

# After Ellipsis were substituted
StrongShape = Union[TensorVariable, Tuple[Union[int, TensorVariable], ...]]
StrongDims = Sequence[Union[str, None]]
StrongSize = Union[TensorVariable, Tuple[Union[int, TensorVariable], ...]]


def convert_dims(dims: Dims) -> Optional[WeakDims]:
    """ Process a user-provided dims variable into None or a valid dims tuple. """
    if dims is None:
        return None

    if isinstance(dims, str):
        dims = (dims,)
    elif isinstance(dims, (list, tuple)):
        dims = tuple(dims)
    else:
        raise ValueError(f"The `dims` parameter must be a tuple, str or list. Actual: {type(dims)}")

    if any(d == Ellipsis for d in dims[:-1]):
        raise ValueError(f"Ellipsis in `dims` may only appear in the last position. Actual: {dims}")

    return dims


def convert_shape(shape: Shape) -> Optional[WeakShape]:
    """ Process a user-provided shape variable into None or a valid shape object. """
    if shape is None:
        return None

    if isinstance(shape, int) or (isinstance(shape, TensorVariable) and shape.ndim == 0):
        shape = (shape,)
    elif isinstance(shape, (list, tuple)):
        shape = tuple(shape)
    else:
        raise ValueError(
            f"The `shape` parameter must be a tuple, TensorVariable, int or list. Actual: {type(shape)}"
        )

    if isinstance(shape, tuple) and any(s == Ellipsis for s in shape[:-1]):
        raise ValueError(
            f"Ellipsis in `shape` may only appear in the last position. Actual: {shape}"
        )

    return shape


def convert_size(size: Size) -> Optional[StrongSize]:
    """ Process a user-provided size variable into None or a valid size object. """
    if size is None:
        return None

    if isinstance(size, int) or (isinstance(size, TensorVariable) and size.ndim == 0):
        size = (size,)
    elif isinstance(size, (list, tuple)):
        size = tuple(size)
    else:
        raise ValueError(
            f"The `size` parameter must be a tuple, TensorVariable, int or list. Actual: {type(size)}"
        )

    if isinstance(size, tuple) and Ellipsis in size:
        raise ValueError(f"The `size` parameter cannot contain an Ellipsis. Actual: {size}")

    return size


def resize_from_dims(
    dims: WeakDims, ndim_implied: int, model
) -> Tuple[int, StrongSize, StrongDims]:
    """Determines a potential resize shape from a `dims` tuple.

    Parameters
    ----------
    dims : array-like
        A vector of dimension names, None or Ellipsis.
    ndim_implied : int
        Number of RV dimensions that were implied from its inputs alone.
    model : pm.Model
        The current model on stack.

    Returns
    -------
    ndim_resize : int
        Number of dimensions that should be added through resizing.
    resize_shape : array-like
        The shape of the new dimensions.
    """
    if Ellipsis in dims:
        # Auto-complete the dims tuple to the full length.
        # We don't have a way to know the names of implied
        # dimensions, so they will be `None`.
        dims = (*dims[:-1], *[None] * ndim_implied)

    ndim_resize = len(dims) - ndim_implied

    # All resize dims must be known already (numerically or symbolically).
    unknowndim_resize_dims = set(dims[:ndim_resize]) - set(model.dim_lengths)
    if unknowndim_resize_dims:
        raise KeyError(
            f"Dimensions {unknowndim_resize_dims} are unknown to the model and cannot be used to specify a `size`."
        )

    # The numeric/symbolic resize tuple can be created using model.RV_dim_lengths
    resize_shape = tuple(model.dim_lengths[dname] for dname in dims[:ndim_resize])
    return ndim_resize, resize_shape, dims


def resize_from_observed(
    observed, ndim_implied: int
) -> Tuple[int, StrongSize, Union[np.ndarray, Variable]]:
    """Determines a potential resize shape from observations.

    Parameters
    ----------
    observed : scalar, array-like
        The value of the `observed` kwarg to the RV creation.
    ndim_implied : int
        Number of RV dimensions that were implied from its inputs alone.

    Returns
    -------
    ndim_resize : int
        Number of dimensions that should be added through resizing.
    resize_shape : array-like
        The shape of the new dimensions.
    observed : scalar, array-like
        Observations as numpy array or `Variable`.
    """
    if not hasattr(observed, "shape"):
        observed = pandas_to_array(observed)
    ndim_resize = observed.ndim - ndim_implied
    resize_shape = tuple(observed.shape[d] for d in range(ndim_resize))
    return ndim_resize, resize_shape, observed


def find_size(shape=None, size=None, ndim_supp=None):
    """Determines the size keyword argument for creating a Distribution.

    Parameters
    ----------
    shape : tuple
        A tuple specifying the final shape of a distribution
    size : tuple
        A tuple specifying the size of a distribution
    ndim_supp : int
        The support dimension of the distribution.
        0 if a univariate distribution, 1 if a multivariate distribution.

    Returns
    -------
    create_size : int
        The size argument to be passed to the distribution
    ndim_expected : int
        Number of dimensions expected after distribution was created
    ndim_batch : int
        Number of batch dimensions
    ndim_supp : int
        Number of support dimensions
    """

    shape = convert_shape(shape)
    size = convert_size(size)

    ndim_expected = None
    ndim_batch = None
    create_size = None

    if shape is not None:
        if Ellipsis in shape:
            # Ellipsis short-hands all implied dimensions. Therefore
            # we don't know how many dimensions to expect.
            ndim_expected = ndim_batch = None
            # Create the RV with its implied shape and resize later
            create_size = None
        else:
            ndim_expected = len(tuple(shape))
            ndim_batch = ndim_expected - ndim_supp
            create_size = tuple(shape)[:ndim_batch]
    elif size is not None:
        ndim_expected = ndim_supp + len(tuple(size))
        ndim_batch = ndim_expected - ndim_supp
        create_size = size

    return create_size, ndim_expected, ndim_batch, ndim_supp


def maybe_resize(
    rv_out,
    rv_op,
    dist_params,
    ndim_expected,
    ndim_batch,
    ndim_supp,
    size=None,
    shape=None,
    **kwargs,
):
    """Resize a distribution if necessary.

    Parameters
    ----------
    rv_out : RandomVariable
        The RandomVariable to be resized if necessary
    rv_op : RandomVariable.__class__
        The RandomVariable class to recreate it
    dist_params : dict
        Input parameters to recreate the RandomVariable
    ndim_expected : int
        Number of dimensions expected after distribution was created
    ndim_batch : int
        Number of batch dimensions
    ndim_supp : int
        The support dimension of the distribution.
        0 if a univariate distribution, 1 if a multivariate distribution.
    size : tuple
        A tuple specifying the size of a distribution
    shape : tuple
        A tuple specifying the final shape of a distribution

    Returnsfind
    -------
    rv_out : int
        The size argument to be passed to the distribution
    """
    ndim_actual = rv_out.ndim
    ndims_unexpected = ndim_actual != ndim_expected

    if shape is not None and ndims_unexpected:
        if Ellipsis in shape:
            # Resize and we're done!
            rv_out = change_rv_size(rv_var=rv_out, new_size=shape[:-1], expand=True)
        else:
            # This is rare, but happens, for example, with MvNormal(np.ones((2, 3)), np.eye(3), shape=(2, 3)).
            # Recreate the RV without passing `size` to created it with just the implied dimensions.
            rv_out = rv_op(*dist_params, size=None, **kwargs)

            # Now resize by any remaining "extra" dimensions that were not implied from support and parameters
            if rv_out.ndim < ndim_expected:
                expand_shape = shape[: ndim_expected - rv_out.ndim]
                rv_out = change_rv_size(rv_var=rv_out, new_size=expand_shape, expand=True)
            if not rv_out.ndim == ndim_expected:
                raise ShapeError(
                    f"Failed to create the RV with the expected dimensionality. "
                    f"This indicates a severe problem. Please open an issue.",
                    actual=ndim_actual,
                    expected=ndim_batch + ndim_supp,
                )

    # Warn about the edge cases where the RV Op creates more dimensions than
    # it should based on `size` and `RVOp.ndim_supp`.
    if size is not None and ndims_unexpected:
        warnings.warn(
            f"You may have expected a ({len(tuple(size))}+{ndim_supp})-dimensional RV, but the resulting RV will be {ndim_actual}-dimensional."
            ' To silence this warning use `warnings.simplefilter("ignore", pm.ShapeWarning)`.',
            ShapeWarning,
        )

    return rv_out
