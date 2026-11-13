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
import pytest

from pymc.variational.updates import (
    adadelta,
    adagrad,
    adagrad_window,
    adam,
    adamax,
    momentum,
    nesterov_momentum,
    rmsprop,
    sgd,
)

_a = pytensor.shared(1.0)
_b = _a * 2

_m = pytensor.shared(np.empty((10,), pytensor.config.floatX))
_n = _m.sum()
_m2 = pytensor.shared(np.empty((10, 10, 10), pytensor.config.floatX))
_n2 = _b + _n + _m2.sum()


@pytest.mark.parametrize(
    "opt",
    [sgd, momentum, nesterov_momentum, adagrad, rmsprop, adadelta, adam, adamax, adagrad_window],
    ids=[
        "sgd",
        "momentum",
        "nesterov_momentum",
        "adagrad",
        "rmsprop",
        "adadelta",
        "adam",
        "adamax",
        "adagrad_window",
    ],
)
@pytest.mark.parametrize(
    "getter",
    [
        lambda t: t,  # all params -> ok
        lambda t: (None, t[1]),  # missing loss -> fail
        lambda t: (t[0], None),  # missing params -> fail
        lambda t: (None, None),
    ],  # all missing -> partial
    ids=["all_params", "missing_loss", "missing_params", "all_missing"],
)
@pytest.mark.parametrize("kwargs", [{}, {"learning_rate": 1e-2}], ids=["without_args", "with_args"])
@pytest.mark.parametrize(
    "loss_and_params",
    [(_b, [_a]), (_n, [_m]), (_n2, [_a, _m, _m2])],
    ids=["scalar", "matrix", "mixed"],
)
def test_updates_fast(opt, loss_and_params, kwargs, getter):
    loss, param = getter(loss_and_params)
    args = {}
    args.update(**kwargs)
    args.update({"loss_or_grads": loss, "params": param})
    if loss is None and param is None:
        updates = opt(**args)
        # Here we should get new callable
        assert callable(updates)
        # And be able to get updates
        updates = opt(_b, [_a])
        assert isinstance(updates, dict)
    # case when both are None is above
    elif loss is None or param is None:
        # Here something goes wrong and user provides not full set of [params + loss_or_grads]
        # We raise Value error
        with pytest.raises(ValueError):
            opt(**args)
    else:
        # Usual call to optimizer, old behaviour
        updates = opt(**args)
        assert isinstance(updates, dict)


def test_adagrad_window_excludes_current_gradient():
    # Regression test for #8444: with n_win=1 the rule after the first
    # (stock) step is constant-scale SGD, lr * g / sqrt(epsilon).
    p = pytensor.shared(np.array(1.0))
    gval = pytensor.shared(np.array(0.0))
    updates = adagrad_window([gval], [p], learning_rate=0.1, epsilon=0.1, n_win=1)
    f = pytensor.function([], [], updates=updates)
    gval.set_value(np.array(2.0))
    f()
    assert p.get_value() == pytest.approx(1.0 - 0.1 * 2.0 / np.sqrt(4.0 + 0.1))
    gval.set_value(np.array(-1.0))
    f()
    assert p.get_value() == pytest.approx(
        1.0 - 0.1 * 2.0 / np.sqrt(4.0 + 0.1) + 0.1 * 1.0 / np.sqrt(0.1)
    )
