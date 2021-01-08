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

from itertools import product

import numpy as np
import pytest
import theano
import theano.tensor as tt

from pymc3.theanof import _conversion_map, take_along_axis
from pymc3.vartypes import int_types

FLOATX = str(theano.config.floatX)
INTX = str(_conversion_map[FLOATX])


def _make_along_axis_idx(arr_shape, indices, axis):
    # compute dimensions to iterate over
    if str(indices.dtype) not in int_types:
        raise IndexError("`indices` must be an integer array")
    shape_ones = (1,) * indices.ndim
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))

    # build a fancy index, consisting of orthogonal aranges, with the
    # requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1 :]
            fancy_index.append(np.arange(n).reshape(ind_shape))

    return tuple(fancy_index)


if hasattr(np, "take_along_axis"):
    np_take_along_axis = np.take_along_axis
else:

    def np_take_along_axis(arr, indices, axis):
        if arr.shape[axis] <= 32:
            # We can safely test with numpy's choose
            arr = np.moveaxis(arr, axis, 0)
            indices = np.moveaxis(indices, axis, 0)
            out = np.choose(indices, arr)
            return np.moveaxis(out, 0, axis)
        else:
            # numpy's choose cannot handle such a large axis so we
            # just use the implementation of take_along_axis. This is kind of
            # cheating because our implementation is the same as the one below
            if axis < 0:
                _axis = arr.ndim + axis
            else:
                _axis = axis
            if _axis < 0 or _axis >= arr.ndim:
                raise ValueError(f"Supplied axis {axis} is out of bounds")
            return arr[_make_along_axis_idx(arr.shape, indices, _axis)]


class TestTakeAlongAxis:
    def setup_class(self):
        self.inputs_buffer = dict()
        self.output_buffer = dict()
        self.func_buffer = dict()

    def _input_tensors(self, shape):
        ndim = len(shape)
        arr = tt.TensorType(FLOATX, [False] * ndim)("arr")
        indices = tt.TensorType(INTX, [False] * ndim)("indices")
        arr.tag.test_value = np.zeros(shape, dtype=FLOATX)
        indices.tag.test_value = np.zeros(shape, dtype=INTX)
        return arr, indices

    def get_input_tensors(self, shape):
        ndim = len(shape)
        try:
            return self.inputs_buffer[ndim]
        except KeyError:
            arr, indices = self._input_tensors(shape)
            self.inputs_buffer[ndim] = arr, indices
            return arr, indices

    def _output_tensor(self, arr, indices, axis):
        return take_along_axis(arr, indices, axis)

    def get_output_tensors(self, shape, axis):
        ndim = len(shape)
        try:
            return self.output_buffer[(ndim, axis)]
        except KeyError:
            arr, indices = self.get_input_tensors(shape)
            out = self._output_tensor(arr, indices, axis)
            self.output_buffer[(ndim, axis)] = out
            return out

    def _function(self, arr, indices, out):
        return theano.function([arr, indices], [out])

    def get_function(self, shape, axis):
        ndim = len(shape)
        try:
            return self.func_buffer[(ndim, axis)]
        except KeyError:
            arr, indices = self.get_input_tensors(shape)
            out = self.get_output_tensors(shape, axis)
            func = self._function(arr, indices, out)
            self.func_buffer[(ndim, axis)] = func
            return func

    @staticmethod
    def get_input_values(shape, axis, samples):
        arr = np.random.randn(*shape).astype(FLOATX)
        size = list(shape)
        size[axis] = samples
        size = tuple(size)
        indices = np.random.randint(low=0, high=shape[axis], size=size, dtype=INTX)
        return arr, indices

    @pytest.mark.parametrize(
        ["shape", "axis", "samples"],
        product(
            [
                (1,),
                (3,),
                (3, 1),
                (3, 2),
                (1, 1),
                (1, 2),
                (40, 40),  # choose fails here
                (5, 1, 1),
                (5, 1, 2),
                (5, 3, 1),
                (5, 3, 2),
            ],
            [0, -1],
            [1, 10],
        ),
        ids=str,
    )
    def test_take_along_axis(self, shape, axis, samples):
        arr, indices = self.get_input_values(shape, axis, samples)
        func = self.get_function(shape, axis)
        assert np.allclose(np_take_along_axis(arr, indices, axis=axis), func(arr, indices)[0])

    @pytest.mark.parametrize(
        ["shape", "axis", "samples"],
        product(
            [
                (1,),
                (3,),
                (3, 1),
                (3, 2),
                (1, 1),
                (1, 2),
                (40, 40),  # choose fails here
                (5, 1, 1),
                (5, 1, 2),
                (5, 3, 1),
                (5, 3, 2),
            ],
            [0, -1],
            [1, 10],
        ),
        ids=str,
    )
    def test_take_along_axis_grad(self, shape, axis, samples):
        if axis < 0:
            _axis = len(shape) + axis
        else:
            _axis = axis
        # Setup the theano function
        t_arr, t_indices = self.get_input_tensors(shape)
        t_out2 = theano.grad(
            tt.sum(self._output_tensor(t_arr ** 2, t_indices, axis)),
            t_arr,
        )
        func = theano.function([t_arr, t_indices], [t_out2])

        # Test that the gradient gives the same output as what is expected
        arr, indices = self.get_input_values(shape, axis, samples)
        expected_grad = np.zeros_like(arr)
        slicer = [slice(None)] * len(shape)
        for i in range(indices.shape[axis]):
            slicer[axis] = i
            inds = indices[slicer].reshape(shape[:_axis] + (1,) + shape[_axis + 1 :])
            inds = _make_along_axis_idx(shape, inds, _axis)
            expected_grad[inds] += 1
        expected_grad *= 2 * arr
        out = func(arr, indices)[0]
        assert np.allclose(out, expected_grad)

    @pytest.mark.parametrize("axis", [-4, 4], ids=str)
    def test_axis_failure(self, axis):
        arr, indices = self.get_input_tensors((3, 1))
        with pytest.raises(ValueError):
            take_along_axis(arr, indices, axis=axis)

    def test_ndim_failure(self):
        arr = tt.TensorType(FLOATX, [False] * 3)("arr")
        indices = tt.TensorType(INTX, [False] * 2)("indices")
        arr.tag.test_value = np.zeros((1,) * arr.ndim, dtype=FLOATX)
        indices.tag.test_value = np.zeros((1,) * indices.ndim, dtype=INTX)
        with pytest.raises(ValueError):
            take_along_axis(arr, indices)

    def test_dtype_failure(self):
        arr = tt.TensorType(FLOATX, [False] * 3)("arr")
        indices = tt.TensorType(FLOATX, [False] * 3)("indices")
        arr.tag.test_value = np.zeros((1,) * arr.ndim, dtype=FLOATX)
        indices.tag.test_value = np.zeros((1,) * indices.ndim, dtype=FLOATX)
        with pytest.raises(IndexError):
            take_along_axis(arr, indices)
