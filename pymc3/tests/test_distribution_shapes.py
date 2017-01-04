from __future__ import division

from ..model import Model
from ..distributions import Normal

import numpy as np
from nose.tools import raises


def test_univariate_shape_info():
    with Model() as model:
        # Test scalar case with replications.
        x = Normal('x', 0, 1, size=(3, 2))

        # Check resulting objects' shapes
        np.testing.assert_array_equal(x.distribution.shape, (3, 2))
        np.testing.assert_array_equal(np.shape(x.tag.test_value), (3, 2))
        np.testing.assert_array_equal(np.shape(x.random()), (3, 2))

        # Check shape info
        np.testing.assert_array_equal(x.distribution.shape_supp.eval(), ())
        assert x.distribution.ndim_supp == 0
        np.testing.assert_array_equal(x.distribution.shape_ind.eval(), ())
        assert x.distribution.ndim_ind == 0
        np.testing.assert_array_equal(x.distribution.shape_reps.eval(), (3, 2))
        assert x.distribution.ndim_reps == 2

    with Model() as model:
        # Test scalar case with empty size input.
        x = Normal('x', 0, 1, size=())

        # Check resulting objects' shapes
        np.testing.assert_array_equal(x.distribution.shape, ())
        np.testing.assert_array_equal(np.shape(x.tag.test_value), ())
        # FIXME: `Distribution.random` adds an unnecessary dimension.
        # np.testing.assert_array_equal(np.shape(x.random()), ())

        # Check shape info
        np.testing.assert_array_equal(x.distribution.bcast, ())
        np.testing.assert_array_equal(x.distribution.shape_supp.eval(), ())
        assert x.distribution.ndim_supp == 0
        np.testing.assert_array_equal(x.distribution.shape_ind.eval(), ())
        assert x.distribution.ndim_ind == 0
        np.testing.assert_array_equal(x.distribution.shape_reps.eval(), ())
        assert x.distribution.ndim_reps == 0

    with Model() as model:
        # Test independent terms alone.
        x = Normal('x', mu=np.arange(0, 2), tau=np.arange(1, 3))

        # Check resulting objects' shapes
        np.testing.assert_array_equal(x.distribution.shape, (2,))
        np.testing.assert_array_equal(np.shape(x.tag.test_value), (2,))
        np.testing.assert_array_equal(np.shape(x.random()), (2,))

        # Check shape info
        np.testing.assert_array_equal(x.distribution.bcast, (False,))
        np.testing.assert_array_equal(x.distribution.shape_supp.eval(), ())
        assert x.distribution.ndim_supp == 0
        np.testing.assert_array_equal(x.distribution.shape_ind.eval(), (2,))
        assert x.distribution.ndim_ind == 1
        np.testing.assert_array_equal(x.distribution.shape_reps.eval(), ())
        assert x.distribution.ndim_reps == 0

    with Model() as model:
        # Test independent terms and replication.
        x = Normal('x', mu=np.arange(0, 2), tau=np.arange(1, 3),
                   size=(3, 2))

        # Check resulting objects' shapes
        np.testing.assert_array_equal(x.distribution.shape, (3, 2, 2))
        np.testing.assert_array_equal(np.shape(x.tag.test_value), (3, 2, 2))
        np.testing.assert_array_equal(np.shape(x.random()), (3, 2, 2))

        # Check shape info
        np.testing.assert_array_equal(x.distribution.bcast,
                                      (False, False, False))
        np.testing.assert_array_equal(x.distribution.shape_supp.eval(), ())
        assert x.distribution.ndim_supp == 0
        np.testing.assert_array_equal(x.distribution.shape_ind.eval(), (2,))
        assert x.distribution.ndim_ind == 1
        np.testing.assert_array_equal(x.distribution.shape_reps.eval(), (3, 2))
        assert x.distribution.ndim_reps == 2

    with Model() as model:
        # Test broadcasting among the independent terms.
        x = Normal('x', mu=np.arange(0, 2), tau=1, size=(3, 2))

        # Check resulting objects' shapes
        np.testing.assert_array_equal(x.distribution.shape, (3, 2, 2))
        np.testing.assert_array_equal(np.shape(x.tag.test_value), (3, 2, 2))
        np.testing.assert_array_equal(np.shape(x.random()), (3, 2, 2))

        # Check shape info
        np.testing.assert_array_equal(x.distribution.bcast,
                                      (False, False, False))
        np.testing.assert_array_equal(x.distribution.shape_supp.eval(), ())
        assert x.distribution.ndim_supp == 0
        np.testing.assert_array_equal(x.distribution.shape_ind.eval(), (2,))
        assert x.distribution.ndim_ind == 1
        np.testing.assert_array_equal(x.distribution.shape_reps.eval(), (3, 2))
        assert x.distribution.ndim_reps == 2

    with Model() as model:
        # Test broadcasting among the independent terms, where independent
        # terms are determined by a non-default test value parameter.
        x = Normal('x', mu=0, tau=np.r_[1, 2], size=(3, 2))

        # Check resulting objects' shapes
        np.testing.assert_array_equal(x.distribution.shape, (3, 2, 2))
        np.testing.assert_array_equal(np.shape(x.tag.test_value), (3, 2, 2))
        np.testing.assert_array_equal(np.shape(x.random()), (3, 2, 2))

        # Check shape info
        np.testing.assert_array_equal(x.distribution.bcast,
                                      (False, False, False))
        np.testing.assert_array_equal(x.distribution.shape_supp.eval(), ())
        assert x.distribution.ndim_supp == 0
        np.testing.assert_array_equal(x.distribution.shape_ind.eval(), (2,))
        assert x.distribution.ndim_ind == 1
        np.testing.assert_array_equal(x.distribution.shape_reps.eval(), (3, 2))
        assert x.distribution.ndim_reps == 2


# TODO
# def test_multivariate_shape_info():
#     with Model() as model:
#         x = MvNormal('x', np.ones((3,)), np.zeros((3, 3)))
#
#     with Model() as model:
#         x = MvNormal('x', np.ones((3,)), np.zeros((3,)), size=(3, 2))
#
#     with Model() as model:
#         x = MvNormal('x', np.ones((3, 2)), np.zeros((3, 3)))
#
#     with Model() as model:
#         x = Dirichlet('x', np.r_[1, 1, 1], size=(3, 2))
#
#     with Model() as model:
#         x = Multinomial('x', 2, np.r_[0.5, 0.5], size=(3, 2))
#
#     with Model() as model:
#         x = Multinomial('x', np.r_[2, 2], np.r_[0.5, 0.5])
#
#     with Model() as model:
#         x = Multinomial('x', 2, np.r_[[0.5, 0.5], [0.1, 0.9]])
