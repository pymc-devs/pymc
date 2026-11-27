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


def test_adagrad_window_normaliser():
    """Ensure adagrad_window excludes the current step's gradient from the normalizer (#8444)."""
    param = pytensor.shared(0.0)
    loss = 2.0 * param  # grad = 2.0
    lr = 1.0
    eps = 0.25
    n_win = 5

    updates = adagrad_window(loss, [param], learning_rate=lr, epsilon=eps, n_win=n_win)
    step = pytensor.function([], updates=updates)

    # Step 1:
    # Initial buffer `accu` is 0. Normalizer should only include prior accumulated gradients,
    # so denom = sqrt(0 + eps) = 0.5.
    # If the current gradient was inside the normalizer, denom would be sqrt(2^2 + 0.25) = sqrt(4.25).
    step()
    expected_step1 = 0.0 - lr * 2.0 / np.sqrt(eps)
    np.testing.assert_allclose(param.get_value(), expected_step1)

    # Step 2:
    # Buffer `accu` now contains [4.0, 0, 0, 0, 0], sum is 4.0.
    # Current gradient (2.0) should not be added before normalizing,
    # so denom = sqrt(4.0 + eps) = sqrt(4.25).
    # If current gradient were included, denom would be sqrt(4.0 + 4.0 + 0.25) = sqrt(8.25).
    step()
    expected_step2 = expected_step1 - lr * 2.0 / np.sqrt(4.0 + eps)
    np.testing.assert_allclose(param.get_value(), expected_step2)

