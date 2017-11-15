import numpy as np
from numpy import inf
from pymc3.tuning import scaling, find_MAP
from . import models


def test_adjust_precision():
    a = np.array([-10, -.01, 0, 10, 1e300, -inf, inf])
    a1 = scaling.adjust_precision(a)
    assert all((a1 > 0) & (a1 < 1e200))


def test_guess_scaling():
    start, model, _ = models.non_normal(n=5)
    a1 = scaling.guess_scaling(start, model=model)
    assert all((a1 > 0) & (a1 < 1e200))


def test_mle_jacobian():
    """Test MAP / MLE estimation for distributions with flat priors."""
    truth = 10.0  # Simple normal model should give mu=10.0

    start, model, _ = models.simple_normal(bounded_prior=False)
    with model:
        map_estimate = find_MAP(method="BFGS", model=model)

    rtol = 1E-5  # this rtol should work on both floatX precisions
    np.testing.assert_allclose(map_estimate["mu_i"], truth, rtol=rtol)

    start, model, _ = models.simple_normal(bounded_prior=True)
    with model:
        map_estimate = find_MAP(method="BFGS", model=model)

    np.testing.assert_allclose(map_estimate["mu_i"], truth, rtol=rtol)
