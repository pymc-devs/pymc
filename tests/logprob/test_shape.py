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
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
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

    # The expanded and broadcast dimensions are consumed like support dimensions:
    # the logp has the base variable's remaining batch shape
    # (assert_allclose also asserts shapes match, if neither is scalar)
    np.testing.assert_allclose(
        logp_fn([1, 3, 1], np.zeros((1, 3, 1))),
        st.norm.logpdf(np.zeros(3)),
    )
    np.testing.assert_allclose(
        logp_fn([1, 3, 5], np.zeros((1, 3, 5))),
        st.norm.logpdf(np.zeros(3)),
    )
    np.testing.assert_allclose(
        logp_fn([2, 3, 5], np.broadcast_to(np.arange(3).reshape(1, 3, 1), (2, 3, 5))),
        st.norm.logpdf(np.arange(3)),
    )
    # Invalid broadcast value
    np.testing.assert_array_equal(
        logp_fn([1, 3, 5], np.arange(3 * 5).reshape(1, 3, 5)),
        np.full(shape=(3,), fill_value=-np.inf),
    )
    # The invalidity check is elementwise over the base batch dimensions: an
    # inconsistent row only invalidates its own logp
    partially_valid = np.broadcast_to(np.arange(3).reshape(1, 3, 1), (1, 3, 5)).copy()
    partially_valid[0, 1, 3] = 99.0
    np.testing.assert_allclose(
        logp_fn([1, 3, 5], partially_valid),
        np.where([True, False, True], st.norm.logpdf(np.arange(3)), -np.inf),
    )


def test_measurable_broadcast_multivariate():
    x = pt.random.dirichlet(pt.ones(3), size=(1,))
    bcast_x = pt.broadcast_to(x, (5, 3))

    bcast_x_value = bcast_x.clone()
    logp_bcast_x = logp(bcast_x, bcast_x_value)

    rng = np.random.default_rng(170)
    row = rng.dirichlet(np.ones(3))
    valid_value = np.broadcast_to(row, (5, 3))
    valid_logp = logp_bcast_x.eval({bcast_x_value: valid_value})
    assert valid_logp.shape == ()
    np.testing.assert_allclose(
        valid_logp,
        st.dirichlet(np.ones(3)).logpdf(row),
    )

    invalid_value = rng.dirichlet(np.ones(3), size=(5,))
    np.testing.assert_array_equal(
        logp_bcast_x.eval({bcast_x_value: invalid_value}),
        -np.inf,
    )


def test_broadcast_not_measurable_behind_other_ops():
    # The broadcast dimensions are degenerate copies; other rewrites would treat them
    # as independent entries (e.g., counting the jacobian of the exp once per copy),
    # so the broadcast is only measurable when directly valued
    x = pt.random.normal()
    y = pt.exp(pt.broadcast_to(x, (3,)))
    with pytest.raises(NotImplementedError):
        logp(y, y.clone())
