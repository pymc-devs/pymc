import numpy as np
import theano.tensor as tt

from ..distributions.dist_math import alltrue


def test_alltrue():
    assert alltrue([]).eval()
    assert alltrue([True]).eval()
    assert alltrue([tt.ones(10)]).eval()
    assert alltrue([tt.ones(10),
                    5 * tt.ones(101)]).eval()
    assert alltrue([np.ones(10),
                    5 * tt.ones(101)]).eval()
    assert alltrue([np.ones(10),
                    True,
                    5 * tt.ones(101)]).eval()
    assert alltrue([np.array([1, 2, 3]),
                    True,
                    5 * tt.ones(101)]).eval()

    assert not alltrue([False]).eval()
    assert not alltrue([tt.zeros(10)]).eval()
    assert not alltrue([True,
                        False]).eval()
    assert not alltrue([np.array([0, -1]),
                        tt.ones(60)]).eval()
    assert not alltrue([np.ones(10),
                        False,
                        5 * tt.ones(101)]).eval()


def test_alltrue_shape():
    vals = [True, tt.ones(10), tt.zeros(5)]

    assert alltrue(vals).eval().shape == ()
