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
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pymc.distributions import Beta, Dirichlet, MvNormal, MvStudentT, Normal, StudentT
from pymc.logprob.basic import logp


@pytest.mark.parametrize("univariate", [True, False])
def test_complete_set_subtensor(univariate):
    if univariate:
        rv0 = Normal.dist(mu=-10)
        rv1 = StudentT.dist(nu=3, mu=0)
        rv2 = Normal.dist(mu=10, sigma=3)
        rv34 = Beta.dist(alpha=[np.pi, np.e], beta=[1, 1])
        base = pt.empty((5,))
        test_val = [2, 0, -2, 0.25, 0.5]
    else:
        rv0 = MvNormal.dist(mu=[-11, -9], cov=pt.eye(2))
        rv1 = MvStudentT.dist(nu=3, mu=[-1, 1], cov=pt.eye(2))
        rv2 = MvNormal.dist(mu=[9, 11], cov=pt.eye(2) * 3)
        rv34 = Dirichlet.dist(a=[[np.pi, 1], [np.e, 1]])
        base = pt.empty((3, 2))
        test_val = [[2, 0], [0, -2], [-2, 2], [0.25, 0.75], [0.5, 0.5]]

    # fmt: off
    rv = (
        # Boolean indexing
        base[np.array([True, False, False, False, False])].set(rv0)
        # Slice indexing
        [1:2].set(rv1)
        # Integer indexing
        [2].set(rv2)
        # Vector indexing
        [[3, 4]].set(rv34)
    )
    # fmt: on
    ref_rv = pt.join(0, [rv0], [rv1], [rv2], rv34)

    np.testing.assert_allclose(
        logp(rv, test_val).eval(),
        logp(ref_rv, test_val).eval(),
    )


def test_partial_set_subtensor():
    rv123 = Normal.dist(mu=[-10, 0, 10])

    # When base is empty, it doesn't matter what the missing values are
    base = pt.empty((5,))
    rv = base[:3].set(rv123)

    np.testing.assert_allclose(
        logp(rv, [0, 0, 0, 1, np.pi]).eval(),
        [*logp(rv123, [0, 0, 0]).eval(), 0, 0],
    )

    # Otherwise they should match
    base = pt.ones((5,))
    rv = base[:3].set(rv123)

    np.testing.assert_allclose(
        logp(rv, [0, 0, 0, 1, np.pi]).eval(),
        [*logp(rv123, [0, 0, 0]).eval(), 0, -np.inf],
    )


def test_overwrite_set_subtensor():
    """Test that order of overwriting in the generative graph is respected."""
    x = Normal.dist(mu=[0, 1, 2])
    y = x[1:].set(Normal.dist([10, 20]))
    z = y[2:].set(Normal.dist([300]))

    np.testing.assert_allclose(
        logp(z, [0, 0, 0]).eval(),
        logp(Normal.dist([0, 10, 300]), [0, 0, 0]).eval(),
    )


def test_mixed_dimensionality_set_subtensor():
    x = Normal.dist(mu=0, size=(3, 2))
    y = x[1].set(MvNormal.dist(mu=[1, 1], cov=np.eye(2)))
    z = y[2].set(Normal.dist(mu=2, size=(2,)))

    # Because `y` is multivariate the last dimension of `z` must be summed over
    test_val = np.zeros((3, 2))
    logp_eval = logp(z, test_val).eval()
    assert logp_eval.shape == (3,)
    np.testing.assert_allclose(
        logp_eval,
        logp(Normal.dist(mu=[[0, 0], [1, 1], [2, 2]]), test_val).sum(-1).eval(),
    )


def test_invalid_indexing_core_dims():
    x = pt.empty((2, 2))
    rv = MvNormal.dist(cov=np.eye(2))
    vv = x.type()

    match_msg = "Indexing along core dimensions of multivariate SetSubtensor not supported"

    y = x[[0, 1], [1, 0]].set(rv)
    with pytest.raises(NotImplementedError, match=match_msg):
        logp(y, vv)

    y = x[np.array([[False, True], [True, False]])].set(rv)
    with pytest.raises(NotImplementedError, match=match_msg):
        logp(y, vv)

    # Univariate indexing above multivariate core dims also not supported
    z = y[0].set(rv)[0, 1].set(Normal.dist())
    with pytest.raises(NotImplementedError, match=match_msg):
        logp(z, vv)


def test_invalid_broadcasted_set_subtensor():
    rv_bcast = Normal.dist(mu=0)
    base = pt.empty((5,))

    rv = base[:3].set(rv_bcast)
    vv = rv.type()

    # Broadcasting is known at write time, and PyMC does not attempt to make SetSubtensor measurable
    with pytest.raises(NotImplementedError):
        logp(rv, vv)

    mask = pt.tensor(shape=(5,), dtype=bool)
    rv = base[mask].set(rv_bcast)

    # Broadcasting is only known at runtime, and PyMC raises an error when it happens
    logp_rv = logp(rv, vv)
    fn = pytensor.function([mask, vv], logp_rv)
    test_vv = np.zeros(5)

    np.testing.assert_allclose(
        fn([False, False, True, False, False], test_vv),
        [0, 0, -0.91893853, 0, 0],
    )

    with pytest.raises(
        NotImplementedError,
        match="Measurable SetSubtensor not supported when set value is broadcasted.",
    ):
        fn([False, False, True, False, True], test_vv)
