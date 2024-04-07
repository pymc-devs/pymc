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


import numpy as np

from pytensor import tensor as pt
from scipy import stats as stats

from pymc.logprob import icdf, logcdf, logp
from pymc.logprob.rewriting import construct_ir_fgraph


def scipy_logprob(obs, p):
    if p.ndim > 1:
        if p.ndim > obs.ndim:
            obs = obs[((None,) * (p.ndim - obs.ndim) + (Ellipsis,))]
        elif p.ndim < obs.ndim:
            p = p[((None,) * (obs.ndim - p.ndim) + (Ellipsis,))]

        pattern = (p.ndim - 1, *range(p.ndim - 1))
        return np.log(np.take_along_axis(p.transpose(pattern), obs, 0))
    else:
        return np.log(p[obs])


def create_pytensor_params(dist_params, obs, size):
    dist_params_at = []
    for p in dist_params:
        p_aet = pt.as_tensor(p).type()
        p_aet.tag.test_value = p
        dist_params_at.append(p_aet)

    size_at = []
    for s in size:
        s_aet = pt.iscalar()
        s_aet.tag.test_value = s
        size_at.append(s_aet)

    obs_at = pt.as_tensor(obs).type()
    obs_at.tag.test_value = obs

    return dist_params_at, obs_at, size_at


def scipy_logprob_tester(
    rv_var, obs, dist_params, test_fn=None, check_broadcastable=True, test="logprob"
):
    """Test for correspondence between `RandomVariable` and NumPy shape and
    broadcast dimensions.
    """
    if test_fn is None:
        name = getattr(rv_var.owner.op, "name", None)

        if name is None:
            name = rv_var.__name__

        test_fn = getattr(stats, name)

    if test == "logprob":
        pytensor_res = logp(rv_var, pt.as_tensor(obs))
    elif test == "logcdf":
        pytensor_res = logcdf(rv_var, pt.as_tensor(obs))
    elif test == "icdf":
        pytensor_res = icdf(rv_var, pt.as_tensor(obs))
    else:
        raise ValueError(f"test must be one of (logprob, logcdf, icdf), got {test}")

    pytensor_res_val = pytensor_res.eval(dist_params)

    numpy_res = np.asarray(test_fn(obs, *dist_params.values()))

    assert pytensor_res.type.numpy_dtype.kind == numpy_res.dtype.kind

    if check_broadcastable:
        numpy_shape = np.shape(numpy_res)
        numpy_bcast = [s == 1 for s in numpy_shape]
        np.testing.assert_array_equal(pytensor_res.type.broadcastable, numpy_bcast)

    np.testing.assert_array_equal(pytensor_res_val.shape, numpy_res.shape)

    np.testing.assert_array_almost_equal(pytensor_res_val, numpy_res, 4)


def measure_type_info_helper(rv, vv):
    """Extract measurable information from rv"""
    fgraph, _, _ = construct_ir_fgraph({rv: vv})
    node = fgraph.outputs[0].owner
    ndim_supp = node.op.ndim_supp
    supp_axes = node.op.supp_axes
    measure_type = node.op.measure_type

    return ndim_supp, supp_axes, measure_type
