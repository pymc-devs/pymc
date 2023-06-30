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
import numpy as np
import pytensor
import pytensor.tensor as pt
import scipy.stats as st

from pymc import logp


def test_measurable_broadcast():
    b_shape = pt.vector("b_shape", shape=(3,), dtype=int)

    x = pt.random.normal(size=(3, 1))
    bcast_x = pt.broadcast_to(x, shape=b_shape)
    bcast_x.name = "bcast_x"

    bcast_x_value = bcast_x.clone()
    logp_bcast_x = logp(bcast_x, bcast_x_value)
    logp_fn = pytensor.function([b_shape, bcast_x_value], logp_bcast_x, on_unused_input="ignore")

    # assert_allclose also asserts shapes match (if neither is scalar)
    np.testing.assert_allclose(
        logp_fn([1, 3, 1], np.zeros((1, 3, 1))),
        st.norm.logpdf(np.zeros((1, 3, 1))),
    )
    np.testing.assert_allclose(
        logp_fn([1, 3, 5], np.zeros((1, 3, 5))),
        st.norm.logpdf(np.zeros((1, 3, 1))),
    )
    np.testing.assert_allclose(
        logp_fn([2, 3, 5], np.broadcast_to(np.arange(3).reshape(1, 3, 1), (2, 3, 5))),
        st.norm.logpdf(np.arange(3).reshape(1, 3, 1)),
    )
    # Invalid broadcast value
    np.testing.assert_array_equal(
        logp_fn([1, 3, 5], np.arange(3 * 5).reshape(1, 3, 5)),
        np.full(shape=(1, 3, 1), fill_value=-np.inf),
    )
