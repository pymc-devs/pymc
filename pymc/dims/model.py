#   Copyright 2025 - present The PyMC Developers
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
from collections.abc import Callable

from pytensor.tensor import TensorVariable
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable

from pymc.data import Data as RegularData
from pymc.distributions.shape_utils import (
    Dims,
    DimsWithEllipsis,
    convert_dims,
    convert_dims_with_ellipsis,
)
from pymc.model.core import Deterministic as RegularDeterministic
from pymc.model.core import Model, modelcontext
from pymc.model.core import Potential as RegularPotential


def Data(
    name: str, value, dims: Dims = None, model: Model | None = None, **kwargs
) -> XTensorVariable:
    """Wrapper around pymc.Data that returns an XtensorVariable.

    Dimensions are required if the input is not a scalar.
    These are always forwarded to the model object.
    """
    model = modelcontext(model)
    dims = convert_dims(dims)  # type: ignore[assignment]

    with model:
        value = RegularData(name, value, dims=dims, **kwargs)  # type: ignore[arg-type]

    dims = model.named_vars_to_dims[value.name]
    if dims is None and value.ndim > 0:
        raise ValueError("pymc.dims.Data requires dims to be specified for non-scalar data.")

    return as_xtensor(value, dims=dims, name=name)  # type: ignore[arg-type]


def _register_and_return_xtensor_variable(
    name: str,
    value: TensorVariable | XTensorVariable,
    dims: DimsWithEllipsis | None,
    model: Model | None,
    registration_func: Callable,
) -> XTensorVariable:
    if isinstance(value, XTensorVariable):
        dims = convert_dims_with_ellipsis(dims)
        if dims is not None:
            # If dims are provided, apply a transpose to align with the user expectation
            value = value.transpose(*dims)
        # Regardless of whether dims are provided, we now have them
        dims = value.type.dims
    else:
        value = as_xtensor(value, dims=dims, name=name)  # type: ignore[arg-type]
    return registration_func(name, value, dims=dims, model=model)


def Deterministic(
    name: str, value, dims: DimsWithEllipsis | None = None, model: Model | None = None
) -> XTensorVariable:
    """Wrapper around pymc.Deterministic that returns an XtensorVariable.

    If the input is already an XTensorVariable, dims are optional. If dims are provided, the variable is aligned with them with a transpose.
    If the input is not an XTensorVariable, it is converted to one using `as_xtensor`. Dims are required if the input is not a scalar.

    The dimensions of the resulting XTensorVariable are always forwarded to the model object.
    """
    return _register_and_return_xtensor_variable(name, value, dims, model, RegularDeterministic)


def Potential(
    name: str, value, dims: DimsWithEllipsis | None = None, model: Model | None = None
) -> XTensorVariable:
    """Wrapper around pymc.Potential that returns an XtensorVariable.

    If the input is already an XTensorVariable, dims are optional. If dims are provided, the variable is aligned with them with a transpose.
    If the input is not an XTensorVariable, it is converted to one using `as_xtensor`. Dims are required if the input is not a scalar.

    The dimensions of the resulting XTensorVariable are always forwarded to the model object.
    """
    return _register_and_return_xtensor_variable(name, value, dims, model, RegularPotential)
