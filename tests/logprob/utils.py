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

from typing import Optional

import numpy as np

from pytensor import tensor as pt
from pytensor.tensor.var import TensorVariable
from scipy import stats as stats

from pymc.logprob import factorized_joint_logprob, icdf, logcdf, logp
from pymc.logprob.abstract import get_measurable_outputs
from pymc.logprob.utils import ignore_logprob


def joint_logprob(*args, sum: bool = True, **kwargs) -> Optional[TensorVariable]:
    """Create a graph representing the joint log-probability/measure of a graph.

    This function calls `factorized_joint_logprob` and returns the combined
    log-probability factors as a single graph.

    Parameters
    ----------
    sum: bool
        If ``True`` each factor is collapsed to a scalar via ``sum`` before
        being joined with the remaining factors. This may be necessary to
        avoid incorrect broadcasting among independent factors.

    """
    logprob = factorized_joint_logprob(*args, **kwargs)
    if not logprob:
        return None
    if len(logprob) == 1:
        logprob = tuple(logprob.values())[0]
        if sum:
            return pt.sum(logprob)
        return logprob
    if sum:
        return pt.sum([pt.sum(factor) for factor in logprob.values()])
    return pt.add(*logprob.values())


def simulate_poiszero_hmm(
    N, mu=10.0, pi_0_a=np.r_[1, 1], p_0_a=np.r_[5, 1], p_1_a=np.r_[1, 1], seed=None
):
    rng = np.random.default_rng(seed)

    p_0 = rng.dirichlet(p_0_a)
    p_1 = rng.dirichlet(p_1_a)

    Gammas = np.stack([p_0, p_1])
    Gammas = np.broadcast_to(Gammas, (N,) + Gammas.shape)

    pi_0 = rng.dirichlet(pi_0_a)
    s_0 = rng.choice(pi_0.shape[0], p=pi_0)
    s_tm1 = s_0

    y_samples = np.empty((N,), dtype=np.int64)
    s_samples = np.empty((N,), dtype=np.int64)

    for i in range(N):
        s_t = rng.choice(Gammas.shape[-1], p=Gammas[i, s_tm1])
        s_samples[i] = s_t
        s_tm1 = s_t

        if s_t == 1:
            y_samples[i] = rng.poisson(mu)
        else:
            y_samples[i] = 0

    sample_point = {
        "Y_t": y_samples,
        "p_0": p_0,
        "p_1": p_1,
        "S_t": s_samples,
        "P_tt": Gammas,
        "S_0": s_0,
        "pi_0": pi_0,
    }

    return sample_point


def scipy_logprob(obs, p):
    if p.ndim > 1:
        if p.ndim > obs.ndim:
            obs = obs[((None,) * (p.ndim - obs.ndim) + (Ellipsis,))]
        elif p.ndim < obs.ndim:
            p = p[((None,) * (obs.ndim - p.ndim) + (Ellipsis,))]

        pattern = (p.ndim - 1,) + tuple(range(p.ndim - 1))
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


def test_ignore_logprob_basic():
    x = Normal.dist()
    (measurable_x_out,) = get_measurable_outputs(x.owner.op, x.owner)
    assert measurable_x_out is x.owner.outputs[1]

    new_x = ignore_logprob(x)
    assert new_x is not x
    assert isinstance(new_x.owner.op, Normal)
    assert type(new_x.owner.op).__name__ == "UnmeasurableNormalRV"
    # Confirm that it does not have measurable output
    assert get_measurable_outputs(new_x.owner.op, new_x.owner) is None

    # Test that it will not clone a variable that is already unmeasurable
    new_new_x = ignore_logprob(new_x)
    assert new_new_x is new_x


def test_ignore_logprob_model():
    # logp that does not depend on input
    def logp(value, x):
        return value

    with Model() as m:
        x = Normal.dist()
        y = CustomDist("y", x, logp=logp)
    with pytest.warns(
        UserWarning,
        match="Found a random variable that was neither among the observations "
        "nor the conditioned variables",
    ):
        joint_logp(
            [y],
            rvs_to_values={y: y.type()},
            rvs_to_transforms={},
        )

    # The above warning should go away with ignore_logprob.
    with Model() as m:
        x = ignore_logprob(Normal.dist())
        y = CustomDist("y", x, logp=logp)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert joint_logp(
            [y],
            rvs_to_values={y: y.type()},
            rvs_to_transforms={},
        )
