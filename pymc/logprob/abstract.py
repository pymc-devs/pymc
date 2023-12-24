#   Copyright 2024 The PyMC Developers
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
#
#   MIT License
#
#   Copyright (c) 2021-2022 aesara-devs
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import abc

from typing import Tuple, Union
from collections.abc import Sequence
from functools import singledispatch

from pytensor.graph.op import Op
from pytensor.graph.utils import MetaType
from pytensor.tensor import TensorVariable
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.op import RandomVariable


@singledispatch
def _logprob(
    op: Op,
    values: Sequence[TensorVariable],
    *inputs: TensorVariable,
    **kwargs,
):
    """Create a graph for the log-density/mass of a ``RandomVariable``.

    This function dispatches on the type of ``op``, which should be a subclass
    of ``RandomVariable``.  If you want to implement new density/mass graphs
    for a ``RandomVariable``, register a new function on this dispatcher.

    """
    raise NotImplementedError(f"Logprob method not implemented for {op}")


def _logprob_helper(rv, *values, **kwargs):
    """Helper that calls `_logprob` dispatcher."""
    logprob = _logprob(rv.owner.op, values, *rv.owner.inputs, **kwargs)

    name = rv.name
    if (not name) and (len(values) == 1):
        name = values[0].name
    if name:
        if isinstance(logprob, (list, tuple)):
            for i, term in enumerate(logprob):
                term.name = f"{name}_logprob.{i}"
        else:
            logprob.name = f"{name}_logprob"

    return logprob


@singledispatch
def _logcdf(
    op: Op,
    value: TensorVariable,
    *inputs: TensorVariable,
    **kwargs,
):
    """Create a graph for the logcdf of a ``RandomVariable``.

    This function dispatches on the type of ``op``, which should be a subclass
    of ``RandomVariable``.  If you want to implement new logcdf graphs
    for a ``RandomVariable``, register a new function on this dispatcher.
    """
    raise NotImplementedError(f"LogCDF method not implemented for {op}")


def _logcdf_helper(rv, value, **kwargs):
    """Helper that calls `_logcdf` dispatcher."""
    logcdf = _logcdf(rv.owner.op, value, *rv.owner.inputs, name=rv.name, **kwargs)

    if rv.name:
        logcdf.name = f"{rv.name}_logcdf"

    return logcdf


@singledispatch
def _icdf(
    op: Op,
    value: TensorVariable,
    *inputs: TensorVariable,
    **kwargs,
):
    """Create a graph for the inverse CDF of a `RandomVariable`.

    This function dispatches on the type of `op`, which should be a subclass
    of `RandomVariable`.
    """
    raise NotImplementedError(f"Inverse CDF method not implemented for {op}")


def _icdf_helper(rv, value, **kwargs):
    """Helper that calls `_icdf` dispatcher."""
    rv_icdf = _icdf(rv.owner.op, value, *rv.owner.inputs, **kwargs)

    if rv.name:
        rv_icdf.name = f"{rv.name}_icdf"

    return rv_icdf


from enum import Enum, auto


class MeasureType(Enum):
    Discrete = auto()
    Continuous = auto()
    Mixed = auto()


class MeasurableVariable(abc.ABC):
    """A variable that can be assigned a measure/log-probability"""

    def __init__(
        self,
        *args,
        ndim_supp: Union[int, Tuple[int]],
        supp_axes: Tuple[Union[int, Tuple[int]]],
        measure_type: Union[MeasureType, Tuple[MeasureType]],
        **kwargs,
    ):
        self.ndim_supp = ndim_supp
        self.supp_axes = supp_axes
        self.measure_type = measure_type
        super().__init__(*args, **kwargs)


MeasurableVariable.register(RandomVariable)


class MeasurableElemwise(MeasurableVariable, Elemwise):
    """Base class for Measurable Elemwise variables"""

    valid_scalar_types: tuple[MetaType, ...] = ()

    def __init__(self, scalar_op, *args, **kwargs):
        if not isinstance(scalar_op, self.valid_scalar_types):
            raise TypeError(
                f"scalar_op {scalar_op} is not valid for class {self.__class__}. "
                f"Acceptable types are {self.valid_scalar_types}"
            )
        super().__init__(scalar_op, *args, **kwargs)


def get_measure_type_info(
    base_var,
) -> Tuple[
    Union[int, Tuple[int]], Tuple[Union[int, Tuple[int]]], Union[MeasureType, Tuple[MeasureType]]
]:
    from pymc.logprob.utils import DiracDelta

    if not isinstance(base_var, MeasurableVariable):
        base_op = base_var.owner.op
        index = base_var.owner.outputs.index(base_var)
    else:
        base_op = base_var
    if not isinstance(base_op, MeasurableVariable):
        raise TypeError("base_op must be a RandomVariable or MeasurableVariable")

    if isinstance(base_op, DiracDelta):
        ndim_supp = 0
        supp_axes = ()
        measure_type = MeasureType.Discrete
        return ndim_supp, supp_axes, measure_type

    if isinstance(base_op, RandomVariable):
        ndim_supp = base_op.ndim_supp
        supp_axes = tuple(range(-ndim_supp, 0))
        measure_type = (
            MeasureType.Continuous if base_op.dtype.startswith("float") else MeasureType.Discrete
        )
        return base_op.ndim_supp, supp_axes, measure_type
    else:
        # We'll need this for operators like scan and IfElse
        if isinstance(base_op.ndim_supp, tuple):
            if len(base_var.owner.outputs) != len(base_op.ndim_supp):
                raise NotImplementedError("length of outputs and meta-properties is different")
            return base_op.ndim_supp[index], base_op.supp_axes, base_op.measure_type
        return base_op.ndim_supp, base_op.supp_axes, base_op.measure_type
