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

import pymc as pm

from pymc.variational.callbacks import CheckLossConvergence, CheckParametersConvergence, Tracker


@pytest.mark.parametrize("diff", ["relative", "absolute"])
@pytest.mark.parametrize("ord", [1, 2, np.inf])
def test_callbacks_convergence(diff, ord):
    cb = CheckParametersConvergence(every=1, diff=diff, ord=ord)

    class _approx:
        params = (pytensor.shared(np.asarray([1, 2, 3])),)

    approx = _approx()

    with pytest.raises(StopIteration):
        cb(approx, None, 1)
        cb(approx, None, 10)


def test_tracker_callback():
    import time

    tracker = Tracker(
        ints=lambda *t: t[-1],
        ints2=lambda ap, h, j: j,
        time=time.time,
    )
    for i in range(10):
        tracker(None, None, i)
    assert "time" in tracker.hist
    assert "ints" in tracker.hist
    assert "ints2" in tracker.hist
    assert len(tracker["ints"]) == len(tracker["ints2"]) == len(tracker["time"]) == 10
    assert tracker["ints"] == tracker["ints2"] == list(range(10))
    tracker = Tracker(bad=lambda t: t)  # bad signature
    with pytest.raises(TypeError):
        tracker(None, None, 1)


def run_monitor(monitor, losses):
    """Feed a loss trace step by step; return the stop step or None."""
    losses = np.asarray(losses, dtype=float)
    for i in range(len(losses)):
        try:
            monitor(None, losses[: i + 1], i)
        except StopIteration:
            return i
    return None


def improving_then_plateau(n_improve, n_plateau, step=1.0, noise=0.5, seed=0):
    """Loss trace that decreases by ~step per iteration, then goes flat."""
    rng = np.random.default_rng(seed)
    deltas = np.concatenate(
        [
            rng.normal(step, noise, size=n_improve),
            rng.normal(0.0, noise, size=n_plateau),
        ]
    )
    return 1000.0 - np.cumsum(deltas)


def test_no_trigger_before_min_steps():
    """A flat-from-the-start trace must never fire inside the arming window."""
    rng = np.random.default_rng(1)
    losses = 100.0 + rng.normal(0.0, 0.5, size=2000)
    monitor = CheckLossConvergence(min_steps=1000)
    stop = run_monitor(monitor, losses)
    assert stop is None or stop >= 1000


def test_triggers_on_plateau_after_arming():
    """Improvement dies at a known step; the monitor fires within a bounded delay."""
    t_star = 1500
    losses = improving_then_plateau(n_improve=t_star, n_plateau=2000)
    monitor = CheckLossConvergence(min_steps=500)
    stop = run_monitor(monitor, losses)
    assert stop is not None, "monitor never fired on a clear plateau"
    assert t_star <= stop <= t_star + 500


def test_no_trigger_on_steady_improvement():
    """A trace that keeps improving must run the full horizon."""
    losses = improving_then_plateau(n_improve=10_000, n_plateau=0)
    monitor = CheckLossConvergence(min_steps=500)
    assert run_monitor(monitor, losses) is None


def test_stopiteration_message():
    """The stop reason names the class, the step, and the statistic."""
    losses = improving_then_plateau(n_improve=1000, n_plateau=2000)
    monitor = CheckLossConvergence(min_steps=200)
    with pytest.raises(StopIteration, match=r"CheckLossConvergence: converged at step \d+"):
        for i in range(len(losses)):
            monitor(None, losses[: i + 1], i)


def test_nonfinite_losses_skipped_and_counted():
    """NaN/inf losses are ignored without corrupting the statistics."""
    losses = improving_then_plateau(n_improve=800, n_plateau=1200)
    losses[100] = np.nan
    losses[200] = np.inf
    monitor = CheckLossConvergence(min_steps=300)
    stop = run_monitor(monitor, losses)
    assert monitor.n_nonfinite == 2
    assert stop is not None  # still detects the plateau


def test_none_losses_raises_typeerror():
    """score=False (losses=None) produces an actionable error, not a silent no-op."""
    monitor = CheckLossConvergence()
    with pytest.raises(TypeError, match=r"score=True"):
        monitor(None, None, 0)


def test_sigma_adapts_to_scale_change():
    """A 10x jump in noise scale mid-stream must not fake a convergence signal."""
    rng = np.random.default_rng(3)
    deltas = np.concatenate(
        [
            rng.normal(1.0, 0.2, size=3000),
            rng.normal(10.0, 2.0, size=3000),  # still improving, just rescaled
        ]
    )
    losses = 1000.0 - np.cumsum(deltas)
    monitor = CheckLossConvergence(min_steps=500)
    assert run_monitor(monitor, losses) is None


def test_sigma_floor_prevents_z_blowup():
    """Exactly-constant losses (MAD -> 0) stay finite and stop cleanly."""
    losses = np.full(3000, 42.0)
    monitor = CheckLossConvergence(min_steps=100)
    stop = run_monitor(monitor, losses)
    # constant loss really is converged; S grows by kappa per step after arming
    assert stop is not None
    expected = 100 + int(monitor.h / monitor.kappa)
    assert abs(stop - expected) <= 3


def test_pm_fit_integration_smoke():
    """End to end inside pm.fit: early stop returns the partial approximation."""
    rng = np.random.default_rng(0)
    data = rng.normal(1.0, 1.0, size=200)
    with pm.Model():
        mu = pm.Normal("mu", 0.0, 1.0)
        pm.Normal("obs", mu, 1.0, observed=data)
        monitor = CheckLossConvergence(min_steps=200, halflife=50.0, h=10.0)
        approx = pm.fit(
            5000,
            callbacks=[monitor],
            progressbar=False,
            random_seed=0,
            obj_optimizer=pm.adam(learning_rate=0.1),
        )
    # a conjugate normal model plateaus quickly; the monitor must have stopped early
    assert len(approx.hist) < 5000
    assert np.isfinite(approx.hist[-50:]).all()
