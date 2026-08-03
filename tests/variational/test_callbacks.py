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

from pymc.variational.callbacks import (
    _SQRT_PI_OVER_2,
    CheckLossConvergence,
    CheckParametersConvergence,
    Tracker,
)


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


def test_nonfinite_losses_skipped_and_the_run_reset():
    """NaN/inf losses are ignored without corrupting the statistics.

    The tolerance is on a *run* of them, so a finite loss clears the count; a cumulative
    count kills a long healthy fit that emits one +inf every so often.
    """
    losses = improving_then_plateau(n_improve=800, n_plateau=1200)
    losses[100] = np.nan
    losses[101] = np.inf
    monitor = CheckLossConvergence(min_steps=300)
    stop = run_monitor(monitor, losses)
    assert monitor.n_nonfinite == 0
    assert stop is not None  # still detects the plateau


def test_scattered_nonfinite_losses_never_stop_a_healthy_fit():
    """One +inf every tenth step is not a stalled fit, whatever their total."""
    losses = 10_000.0 - np.arange(3000.0)
    losses[::10] = np.inf
    assert run_monitor(CheckLossConvergence(min_steps=100), losses) is None


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


def test_the_scale_constant_is_not_a_unit_variance_normalizer():
    """``sqrt(pi)/2`` recovers ``sd(delta)`` only when the increments are independent.

    ``delta`` is itself a first difference, so on a trace whose *levels* are i.i.d. --
    what a plateaued ADVI trace looks like, its increments correlating at about -0.5 --
    the estimate lands on ``sqrt(3) * sigma`` while ``sd(delta)`` is ``sqrt(2) * sigma``,
    putting ``z`` at 0.82 sd. Both cases are real, so no one factor makes ``z``
    unit-variance: ``kappa`` and ``h`` are calibrated against the constant as it stands,
    and rescaling it moves the stall boundary with every ``z``.
    """
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, size=50_000)
    for losses, expected in ((noise, np.sqrt(3.0)), (np.cumsum(noise), 1.0)):
        monitor = CheckLossConvergence(min_steps=10**9, halflife=2000.0)
        run_monitor(monitor, losses)
        assert monitor._scale * _SQRT_PI_OVER_2 == pytest.approx(expected, rel=0.03)


def test_sigma_floor_prevents_z_blowup():
    """Exactly-constant losses (MAD -> 0) stay finite and stop cleanly."""
    losses = np.full(3000, 42.0)
    monitor = CheckLossConvergence(min_steps=100)
    stop = run_monitor(monitor, losses)
    # constant loss really is converged; S grows by kappa per step after arming
    assert stop is not None
    expected = 100 + int(monitor.h / monitor.kappa)
    assert abs(stop - expected) <= 3


@pytest.mark.parametrize(
    "slope, noise", [(1.0, 0.0), (1.0, 1.0), (5.0, 1.0)], ids=["clean", "noisy", "steep"]
)
def test_rising_loss_is_not_called_convergence(slope, noise):
    """A diverging fit is the opposite of a converged one and must not be reported as one."""
    rng = np.random.default_rng(4)
    losses = slope * np.arange(4000.0) + rng.normal(0.0, noise, size=4000)
    monitor = CheckLossConvergence()
    with pytest.raises(
        StopIteration, match=r"CheckLossConvergence: the loss is trending up.*negate it"
    ):
        for i in range(len(losses)):
            monitor(None, losses[: i + 1], i)


@pytest.mark.parametrize("seed", range(4))
def test_a_rise_of_one_noise_sd_per_step_is_still_called_divergence(seed):
    """The label has to survive a rise that noise nearly hides, not just an obvious one.

    A loss climbing by about one noise sd per step leaves the smoothed z between
    -0.45 and -0.62, so a divergence threshold loose enough to sit in that band
    reports the run as converged. Parametrized because the margin is what matters.
    """
    rng = np.random.default_rng(seed)
    losses = np.arange(6000.0) + rng.normal(0.0, 1.0, size=6000)
    monitor = CheckLossConvergence()
    with pytest.raises(StopIteration, match="trending up, not converging"):
        for i in range(len(losses)):
            monitor(None, losses[: i + 1], i)


def test_a_converged_plateau_is_never_labelled_diverging():
    """The mirror of the test above: the threshold must leave real plateaus alone."""
    for seed in range(4):
        losses = improving_then_plateau(n_improve=1500, n_plateau=2500, seed=seed)
        monitor = CheckLossConvergence(min_steps=500)
        with pytest.raises(StopIteration, match="converged at step"):
            for i in range(len(losses)):
                monitor(None, losses[: i + 1], i)


def test_a_burst_of_reversals_does_not_end_a_still_improving_fit():
    """Under a plain ``kappa - z`` each upward step contributes ``kappa + |z|``, so a
    three-step burst covers most of ``h`` on its own while the fit is still improving."""
    rng = np.random.default_rng(7)
    deltas = rng.normal(2.0, 0.3, size=1000)
    deltas[400:403] = -20.0
    losses = 1000.0 - np.cumsum(deltas)
    assert run_monitor(CheckLossConvergence(min_steps=200), losses) is None


def test_one_spike_does_not_end_a_still_improving_fit():
    """A heavy-tailed excursion inflates the raw scale for hundreds of steps if unbounded."""
    losses = 10_000.0 - np.arange(3000.0)
    assert run_monitor(CheckLossConvergence(), losses) is None
    spiked = losses.copy()
    spiked[1500] += 1e6
    assert run_monitor(CheckLossConvergence(), spiked) is None


def test_a_one_step_jump_does_not_turn_a_plateau_into_a_divergence():
    """Winsorizing z bounds what a single step can contribute to the trend estimate.

    Unclipped, one upward jump holds the smoothed z below the divergence threshold for
    hundreds of steps, so the plateau that follows is reported as a diverging fit.
    """
    losses = improving_then_plateau(n_improve=1500, n_plateau=2500)
    losses[1400:] += 1e5
    monitor = CheckLossConvergence(min_steps=300)
    with pytest.raises(StopIteration, match="converged at step"):
        for i in range(len(losses)):
            monitor(None, losses[: i + 1], i)


def test_overflowing_loss_does_not_poison_the_scale():
    """Successive differences of huge finite losses overflow; the scale must survive it."""
    losses = 10_000.0 - np.arange(3000.0)
    losses[1200] = 1e308
    assert run_monitor(CheckLossConvergence(), losses) is None


def test_persistently_nonfinite_loss_stops():
    """pm.fit aborts on NaN but runs to completion on +inf, so the monitor has to call it."""
    monitor = CheckLossConvergence(min_steps=100)
    stop = run_monitor(monitor, np.full(2000, np.inf))
    assert stop == monitor.min_steps  # min_steps tolerated, then the next one stops


@pytest.mark.parametrize(
    "kwargs",
    [
        {"h": np.nan},
        {"kappa": np.nan},
        {"halflife": np.inf},
        {"z_clip": np.nan},
        {"sigma_floor": 0.0},
        {"sigma_floor": -1.0},
        {"min_steps": 2.7},
    ],
)
def test_nonfinite_or_nonpositive_parameters_rejected(kwargs):
    """Each of these silently disables the stop rule, flips the sign of z, or is
    silently truncated to something the caller did not ask for."""
    with pytest.raises(ValueError):
        CheckLossConvergence(**kwargs)


def test_stop_step_tracks_plateau_onset():
    """The stop step follows the plateau, not the clock: later onset, later stop.

    The detection delay has to stay bounded as the improving phase lengthens. A CUSUM
    that is not floored at zero banks a deficit proportional to that length, so the
    same plateau takes ever longer to clear ``h`` -- monotone in the wrong way.
    """
    for seed in range(3):
        stops = []
        for t_star in (800, 1600, 2400, 3200):
            losses = improving_then_plateau(n_improve=t_star, n_plateau=1200, seed=seed)
            stop = run_monitor(CheckLossConvergence(min_steps=300), losses)
            assert stop is not None, f"no stop for onset {t_star} (seed {seed})"
            assert 20 <= stop - t_star <= 400, f"delay {stop - t_star} at onset {t_star}"
            stops.append(stop)
        assert stops == sorted(stops)


@pytest.mark.parametrize(
    "scale, offset",
    [(1e-6, 0.0), (1e6, 0.0), (1e-3, 5e3), (1e3, -1e6)],
    ids=["tiny", "huge", "shrunk_shifted", "grown_shifted"],
)
def test_stop_step_is_invariant_to_affine_relabelling(scale, offset):
    """Only increments measured in units of their own noise reach the decision rule.

    ELBO magnitudes differ by orders of magnitude between models, which is the whole
    reason for dividing by sigma; an affine relabelling of one trace must not move the
    stop by a single step.
    """
    losses = improving_then_plateau(n_improve=1500, n_plateau=2000)
    reference = run_monitor(CheckLossConvergence(min_steps=300), losses)
    assert reference is not None
    assert run_monitor(CheckLossConvergence(min_steps=300), losses * scale + offset) == reference


_STILL_IMPROVING = {
    "linear": lambda rng, n: rng.normal(1.0, 1.0, size=n),
    "power_law": lambda rng, n: (
        5.0 * np.arange(1.0, n + 1.0) ** -0.25 + rng.normal(0.0, 0.5, size=n)
    ),
    "heteroscedastic": lambda rng, n: 1.0 + rng.normal(0.0, 1.0, size=n) * np.linspace(0.1, 1.0, n),
    "student_t": lambda rng, n: 1.0 + 0.5 * rng.standard_t(df=3, size=n),
}


@pytest.mark.parametrize("family", sorted(_STILL_IMPROVING))
def test_still_improving_traces_do_not_raise_a_false_alarm(family):
    """Fifty traces per family whose improvement never dies cost at most one false alarm.

    Every family ends with improvement still well clear of the stall boundary, so a
    stop is a truncated fit. One lucky trace shape says nothing about the calibrated
    0.8% worst-family rate; a kappa or h that drifts off it shows up here first.
    """
    make_deltas = _STILL_IMPROVING[family]
    fired = []
    for seed in range(50):
        deltas = make_deltas(np.random.default_rng(seed), 3000)
        stop = run_monitor(CheckLossConvergence(), 1000.0 - np.cumsum(deltas))
        if stop is not None:
            fired.append(stop)
    assert len(fired) <= 1, f"{family}: {len(fired)} false alarms in 50 traces, at {fired}"


@pytest.mark.parametrize("min_steps, slope", [(50, 0.5), (200, 3.0), (500, 1e4)])
def test_a_deterministic_ramp_stops_at_the_analytic_step(min_steps, slope):
    """A noiseless ramp drives |z| to z_clip, fixing the stop at min_steps + h / kappa.

    Uphill ``max(z, 0)`` is zero and S gains exactly kappa per armed step, whatever the
    slope. Reading the rise as negative improvement would charge ``kappa + z_clip`` and
    stop nine times sooner than a converged trace does; downhill the allowance never
    covers the improvement and S stays pinned at zero forever.
    """
    monitor = CheckLossConvergence(min_steps=min_steps)
    expected = min_steps + int(monitor.h / monitor.kappa)
    ramp = slope * np.arange(1500.0)
    with pytest.raises(StopIteration, match=rf"trending up.*\(step {expected},"):
        for i in range(len(ramp)):
            monitor(None, 7.0 + ramp[: i + 1], i)
    assert run_monitor(CheckLossConvergence(min_steps=min_steps), 7.0 - ramp) is None


@pytest.mark.parametrize("rate", [1.0, 5.0, 100.0])
def test_a_maximized_objective_has_to_be_negated(rate):
    """Handed a raw ELBO the monitor answers a question about its own settings.

    Every step is uphill, so ``max(z, 0)`` is zero and S gains the full allowance per
    armed step: the stop lands at ``min_steps + h / kappa`` however fast the fit is
    improving, and faster improvement stops sooner. Negated, the same trace runs on.
    """
    rng = np.random.default_rng(0)
    elbo = rate * np.arange(4000.0) + rng.normal(0.0, 1.0, size=4000)
    assert run_monitor(CheckLossConvergence(), -elbo) is None
    monitor = CheckLossConvergence()
    assert run_monitor(monitor, elbo) == pytest.approx(
        monitor.min_steps + monitor.h / monitor.kappa, abs=10
    )


@pytest.mark.parametrize(
    "rate, expected",
    [
        (0.05, "converged"),
        (0.2, "converged"),
        (0.5, "trending up"),
        (1.0, "trending up"),
        (3.0, "trending up"),
    ],
)
def test_divergence_label_tracks_the_rise_rate(rate, expected):
    """The label turns over between rises of 0.25 and 0.3 noise sd per step, and stays.

    With i.i.d. noise on the loss level the successive-difference scale estimates
    sqrt(3) sigma, so the smoothed z settles near ``-0.58 * rate`` and crosses
    ``-_rise_tol`` at about 0.29. Pinned from both sides: a threshold loosened twofold
    reads a half-sd-per-step climb as a plateau, a tightened one starts calling honest
    plateaus divergent.
    """
    rng = np.random.default_rng(0)
    losses = rate * np.arange(2000.0) + rng.normal(0.0, 1.0, size=2000)
    monitor = CheckLossConvergence()
    with pytest.raises(StopIteration, match=expected):
        for i in range(len(losses)):
            monitor(None, losses[: i + 1], i)


@pytest.mark.parametrize("min_steps", [300, 600, 1000, 1500])
def test_min_steps_arms_the_cusum_and_nothing_else(min_steps):
    """Arming earlier cannot move the stop of a trace that improves through the window.

    S is floored at zero while the loss falls, so every min_steps below the plateau
    onset has to give the same stop step, not merely a similar one. The scale and trend
    estimates are meanwhile fed by every step: skipping that work before arming makes
    the answer a function of min_steps.
    """
    losses = improving_then_plateau(n_improve=2000, n_plateau=1500)
    reference = run_monitor(CheckLossConvergence(min_steps=100), losses)
    assert reference is not None
    assert run_monitor(CheckLossConvergence(min_steps=min_steps), losses) == reference


def test_a_fired_monitor_is_not_silently_reusable():
    """A monitor that has stopped once keeps saying so instead of re-judging a new fit.

    Nothing drops S back under h, so a second fit stops at its first step rather than
    running to a different answer. One instance per fit is the contract; the cost of
    breaking it is loud, not a plausible wrong stop halfway through.
    """
    monitor = CheckLossConvergence(min_steps=300)
    assert run_monitor(monitor, improving_then_plateau(1000, 1500)) is not None
    still_improving = improving_then_plateau(3000, 0, seed=5)
    assert run_monitor(CheckLossConvergence(min_steps=300), still_improving) is None
    assert run_monitor(monitor, still_improving) == 0


def test_only_the_newest_loss_is_read():
    """Handing over one loss per step decides identically to handing over the history.

    pm.fit passes the growing hist; the monitor's own state is the running summary, so
    reaching further back than the last entry would double-count what it already holds.
    """
    losses = improving_then_plateau(n_improve=1200, n_plateau=1500)
    monitor = CheckLossConvergence(min_steps=300)
    windowed = None
    for i in range(len(losses)):
        try:
            monitor(None, losses[i : i + 1], i)
        except StopIteration:
            windowed = i
            break
    assert windowed is not None
    assert windowed == run_monitor(CheckLossConvergence(min_steps=300), losses)


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
