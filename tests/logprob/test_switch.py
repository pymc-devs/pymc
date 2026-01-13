from typing import cast

import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor.compile.function import function
from pytensor.tensor.variable import TensorVariable

import pymc as pm

from pymc.logprob.basic import logp
from pymc.logprob.utils import ParameterValueError


def test_switch_non_overlapping_logp_matches_change_of_variables():
    scale = pt.scalar("scale")
    x = pm.Normal.dist(mu=0, sigma=1, size=(3,))
    y = cast(TensorVariable, pt.switch(x > 0, x, scale * x))

    vv = pt.vector("vv")

    logp_y = logp(y, vv, warn_rvs=False)
    inv = cast(TensorVariable, pt.switch(pt.gt(vv, 0), vv, vv / scale))
    expected = logp(x, inv, warn_rvs=False) + cast(
        TensorVariable,
        pt.switch(pt.gt(vv, 0), 0.0, -cast(TensorVariable, pt.log(scale))),
    )

    logp_y_fn = function([vv, scale], logp_y)
    expected_fn = function([vv, scale], expected)

    v = np.array([-2.0, 0.0, 1.5])
    np.testing.assert_allclose(logp_y_fn(v, 0.5), expected_fn(v, 0.5))

    # No warning-based shortcuts: also match under default warn_rvs (scalar case)
    x_s = pm.Normal.dist(mu=0, sigma=1)
    y_s = cast(TensorVariable, pt.switch(x_s > 0, x_s, scale * x_s))

    v_pos = 1.2
    np.testing.assert_allclose(logp(y_s, v_pos).eval({scale: 0.5}), logp(x_s, v_pos).eval())

    v_neg = -2.0
    np.testing.assert_allclose(
        logp(y_s, v_neg).eval({scale: 0.5}),
        logp(x_s, v_neg / 0.5).eval() - np.log(0.5),
    )

    # boundary point (measure-zero for continuous RVs): should still produce a finite logp
    assert np.isfinite(logp(y_s, 0.0, warn_rvs=False).eval({scale: 0.5}))


def test_switch_non_overlapping_requires_positive_scale():
    scale = pt.scalar("scale")
    x = pm.Normal.dist(mu=0, sigma=1)
    y = cast(TensorVariable, pt.switch(x > 0, x, scale * x))

    with pytest.raises(ParameterValueError, match="switch non-overlapping scale > 0"):
        logp(y, -1.0, warn_rvs=False).eval({scale: -0.5})

    with pytest.raises(ParameterValueError, match="switch non-overlapping scale > 0"):
        logp(y, -1.0, warn_rvs=False).eval({scale: 0.0})


def test_switch_non_overlapping_does_not_rewrite_if_x_replicated_by_condition():
    scale = pt.scalar("scale")
    x = pm.Normal.dist(mu=0, sigma=1, size=(3,))
    cond = (x[None, :] > 0) & pt.ones((2, 1), dtype="bool")
    y = cast(TensorVariable, pt.switch(cond, x, scale * x))

    with pytest.raises(NotImplementedError, match="Logprob method not implemented for Switch"):
        logp(y, np.zeros((2, 3)), warn_rvs=False).eval({scale: 0.5})


def test_switch_non_overlapping_does_not_rewrite_if_scale_broadcasts_x():
    x = pm.Normal.dist(mu=0, sigma=1)
    scale = pt.vector("scale")
    y = cast(TensorVariable, pt.switch(x > 0, x, scale * x))

    with pytest.raises(NotImplementedError, match="Logprob method not implemented for Switch"):
        logp(y, np.zeros((3,)), warn_rvs=False).eval({scale: np.array([0.5, 0.5, 0.5])})


def test_switch_non_overlapping_does_not_apply_to_discrete_rv():
    a = pt.scalar("a")
    x = pm.Poisson.dist(3)
    y = cast(TensorVariable, pt.switch(x > 0, x, a * x))

    with pytest.raises(NotImplementedError, match="Logprob method not implemented for Switch"):
        logp(y, 1, warn_rvs=False).eval({a: 0.5})
