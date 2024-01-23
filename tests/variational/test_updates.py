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
import pytest

import pymc as pm

from pymc.variational.updates import (
    adadelta,
    adagrad,
    adagrad_window,
    adam,
    adamax,
    exponential_decay_scheduler,
    momentum,
    nesterov_momentum,
    reduce_lr_on_plateau_scheduler,
    rmsprop,
    sgd,
)

OPTIMIZERS = [
    sgd,
    momentum,
    nesterov_momentum,
    adagrad,
    rmsprop,
    adadelta,
    adam,
    adamax,
    adagrad_window,
]

OPTIMIZER_NAMES = [
    "sgd",
    "momentum",
    "nesterov_momentum",
    "adagrad",
    "rmsprop",
    "adadelta",
    "adam",
    "adamax",
    "adagrad_window",
]

_a = pytensor.shared(1.0)
_b = _a * 2

_m = pytensor.shared(np.empty((10,), pytensor.config.floatX))
_n = _m.sum()
_m2 = pytensor.shared(np.empty((10, 10, 10), pytensor.config.floatX))
_n2 = _b + _n + _m2.sum()


@pytest.mark.parametrize("opt", OPTIMIZERS, ids=OPTIMIZER_NAMES)
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
@pytest.mark.parametrize(
    "kwargs", [dict(), dict(learning_rate=1e-2)], ids=["without_args", "with_args"]
)
@pytest.mark.parametrize(
    "loss_and_params",
    [(_b, [_a]), (_n, [_m]), (_n2, [_a, _m, _m2])],
    ids=["scalar", "matrix", "mixed"],
)
def test_updates_fast(opt, loss_and_params, kwargs, getter):
    with pytensor.config.change_flags(compute_test_value="ignore"):
        loss, param = getter(loss_and_params)
        args = dict()
        args.update(**kwargs)
        args.update(dict(loss_or_grads=loss, params=param))
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


@pytest.fixture()
def regression_model():
    rng = np.random.default_rng(1)

    X = rng.normal(size=(100,))
    intercept, coef = rng.normal(100, size=(2,))
    noise = rng.normal(size=(100,))
    y = intercept + coef * X + noise

    with pm.Model() as model:
        a = pm.Normal(
            "a",
        )
        b = pm.Normal("b")

        mu = a + b * X
        pm.Normal("y", mu=mu, sigma=1, observed=y)

    return model


SCHEDULER_PARAMS = [
    (1, 0.5, 1, 1e-8, False),
    (1, 0.5, 1, 1e-8, True),
    (1, 0.5, 2, 1e-8, False),
    (1, 0.5, 2, 1e-8, True),
]
SCHEDULER_IDS = [
    f"initial_lr={x[0]}, decay_rate={x[1]}, decay_steps={x[2]}, min_lr={x[3]}, staircase={x[4]}"
    for x in SCHEDULER_PARAMS
]


@pytest.mark.parametrize("optimizer", OPTIMIZERS, ids=OPTIMIZER_NAMES)
@pytest.mark.parametrize("scheduler_args", SCHEDULER_PARAMS, ids=SCHEDULER_IDS)
def test_exponential_decay_scheduler(regression_model, optimizer, scheduler_args):
    initial_lr, decay_rate, decay_steps, min_lr, staircase = scheduler_args
    opt = optimizer(learning_rate=initial_lr)
    scheduled_optimizer = exponential_decay_scheduler(
        opt, decay_steps, decay_rate, min_lr, staircase
    )

    with regression_model:
        advi = pm.ADVI()

    updates = advi.objective.updates(obj_optimizer=scheduled_optimizer)
    inputs = list(updates.keys())
    outputs = list(updates.values())

    old_names = [x.name for x in inputs]
    new_names = [x.name for x in outputs]

    assert all([expected_name in old_names for expected_name in ["learning_rate", "t"]])
    assert all(
        [expected_name in new_names for expected_name in ["learning_rate__updated", "t__updated"]]
    )

    lr_idx = old_names.index("learning_rate")
    t_idx = old_names.index("t")

    step_func = pytensor.function(
        [], [outputs[lr_idx], outputs[t_idx]], updates=updates, mode="FAST_COMPILE"
    )

    steps = np.vstack([step_func() for _ in range(10)])

    def floor_div(x, y):
        return x // y

    div_func = floor_div if staircase else np.divide

    expected_decay = np.maximum(
        initial_lr * decay_rate ** (div_func(np.arange(10), decay_steps)), min_lr
    )

    np.testing.assert_allclose(steps[:, 0], expected_decay)
    np.testing.assert_allclose(steps[:, 1], np.arange(1, 11))


def test_reduce_lr_on_plateau_scheduler(regression_model):
    opt = pm.adam(learning_rate=1)
    factor = 0.1
    patience = 10
    min_lr = 1e-6
    cooldown = 10
    scheduled_optimizer = reduce_lr_on_plateau_scheduler(
        opt, factor=factor, patience=patience, min_lr=min_lr, cooldown=cooldown
    )
    with regression_model:
        advi = pm.ADVI()

        updates = advi.objective.updates(obj_optimizer=scheduled_optimizer)
        inputs = list(updates.keys())
        outputs = list(updates.values())

        old_names = [x.name for x in inputs]
        new_names = [x.name for x in outputs]

        expected_names = ["best_loss", "cooldown_counter", "wait", "learning_rate"]

        assert all([expected_name in old_names for expected_name in expected_names])
        assert all([f"{expected_name}__updated" in new_names for expected_name in expected_names])

        outputs_of_interest = [
            outputs[new_names.index(f"{expected_name}__updated")]
            for expected_name in expected_names
        ]

        tracker = pm.callbacks.Tracker(
            best_loss=outputs_of_interest[0].eval,
            cooldown_counter=outputs_of_interest[1].eval,
            wait=outputs_of_interest[2].eval,
            learning_rate=outputs_of_interest[3].eval,
        )
        approx = advi.fit(1000, callbacks=[tracker], obj_optimizer=scheduled_optimizer)

    # Best loss only decreases
    assert np.all(np.diff(np.stack(tracker.hist["best_loss"])) <= 0)

    # Learning_rate only decreases
    assert np.all(np.diff(np.stack(tracker.hist["learning_rate"])) <= 0)

    # Wait is never greater than patience
    assert np.all(np.stack(tracker.hist["wait"]) <= patience)

    # Cooldown_counter is never greater than cooldown
    assert np.all(np.stack(tracker.hist["cooldown_counter"]) <= cooldown)
