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
"""Trainer: drive variational inference over a DataLoader with no user callbacks."""

import numpy as np
import pytest

import pymc as pm

from pymc.variational.streaming import DataLoader, Trainer
from tests.variational.streaming_helpers import chunked_factory


def test_trainer_end_to_end_matches_in_ram_minibatch():
    """End-to-end: Trainer-driven streaming ADVI reproduces in-RAM pm.Minibatch ADVI.

    Exercises the whole API: a pm.Data placeholder, total_size=len(loader), and a
    Trainer that streams minibatches into the placeholder with set_data while the
    user writes no callbacks. Runs long enough to cycle the loader across epochs.
    """
    seed = 0
    rng = np.random.default_rng(seed)
    N, bs = 60_000, 2048
    X = rng.normal(size=(N, 2))
    b_true = np.array([0.3, -1.1, 0.7])
    y = (rng.random(N) < 1 / (1 + np.exp(-(b_true[0] + X @ b_true[1:])))).astype("float64")
    data = np.column_stack([X, y])

    with pm.Model():
        b = pm.Normal("b", 0, 3, shape=3)
        xb, zb, yb = pm.Minibatch(X[:, 0].copy(), X[:, 1].copy(), y, batch_size=bs)
        pm.Bernoulli("o", logit_p=b[0] + b[1] * xb + b[2] * zb, observed=yb, total_size=N)
        ap = pm.fit(
            6000,
            method="advi",
            obj_optimizer=pm.adam(learning_rate=0.02),
            progressbar=False,
            random_seed=seed,
        )
        in_ram = ap.sample(400).posterior["b"].values.reshape(-1, 3).mean(0)

    loader = DataLoader(
        chunked_factory(data, 20_000),
        batch_size=bs,
        shuffle=True,
        buffer_size=40_000,
        seed=seed,
        sample_shape=(3,),
        total_size=N,
    )
    with pm.Model() as model:
        b = pm.Normal("b", 0, 3, shape=3)
        batch = pm.Data("batch", np.zeros((bs, 3)))
        pm.Bernoulli(
            "o",
            logit_p=b[0] + b[1] * batch[:, 0] + b[2] * batch[:, 1],
            observed=batch[:, 2],
            total_size=len(loader),
        )
        ap = Trainer(
            method="advi",
            dataloader=loader,
            data_name="batch",
            obj_optimizer=pm.adam(learning_rate=0.02),
        ).fit(6000, random_seed=seed)
        stream = ap.sample(400).posterior["b"].values.reshape(-1, 3).mean(0)

    np.testing.assert_allclose(in_ram, stream, atol=0.1)


def test_trainer_streams_into_placeholder():
    """The Trainer seeds the pm.Data placeholder before step 0 (pm.fit runs
    callbacks after each step) and overwrites it each step; after fitting it holds
    a real batch, not the zero seed."""
    data = np.ones((4, 1))
    loader = DataLoader(lambda: iter([data] * 100), batch_size=4, sample_shape=(1,), total_size=4)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        Trainer(method="advi", dataloader=loader, data_name="batch").fit(
            5, progressbar=False, random_seed=0
        )
    np.testing.assert_array_equal(model["batch"].get_value(), data)


def test_trainer_raises_when_loader_cannot_restart():
    """A source that streams one epoch and then comes back empty cannot be cycled;
    the Trainer surfaces a clear error instead of training on stale data."""
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 1:
            yield np.zeros((4, 1))

    loader = DataLoader(factory, batch_size=4, sample_shape=(1,), total_size=4)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        with pytest.raises(RuntimeError, match="yielded no batches"):
            Trainer(method="advi", dataloader=loader, data_name="batch").fit(
                5, progressbar=False, random_seed=0
            )


def test_trainer_rejects_non_dataloader():
    """The isinstance guard fires before any model lookup."""
    with pytest.raises(TypeError, match="DataLoader"):
        Trainer(method="advi", dataloader=object()).fit(10)


def test_trainer_appends_user_callbacks_and_streams_distinct_batches():
    """User callbacks (e.g. convergence trackers) compose with the internal
    advance callback instead of colliding on the keyword, and the placeholder
    holds a different batch on successive steps. Also exercises the default
    data_name ("batch")."""
    blocks = [np.full((4, 1), float(i)) for i in range(60)]
    loader = DataLoader(lambda: iter(blocks), batch_size=4, sample_shape=(1,), total_size=240)
    seen = []
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 1.0)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", x, 1.0, observed=batch[:, 0], total_size=len(loader))
        Trainer(method="advi", dataloader=loader).fit(
            5, callbacks=[lambda *_: seen.append(float(model["batch"].get_value()[0, 0]))]
        )
    assert len(seen) == 5
    assert len(set(seen)) > 1


def test_trainer_accepts_inference_instance():
    """An Inference instance is forwarded to pm.fit unchanged; it is bound to
    the model it was built under, so the Trainer only streams the batches."""
    data = np.ones((4, 1))
    loader = DataLoader(lambda: iter([data] * 50), batch_size=4, sample_shape=(1,), total_size=4)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        approx = Trainer(method=pm.ADVI(random_seed=0), dataloader=loader).fit(5)
    assert len(approx.hist) == 5
    np.testing.assert_array_equal(model["batch"].get_value(), data)


def test_constructor_fit_kwargs_take_random_seed():
    """random_seed works as a constructor default, as the docstring promises,
    and a per-call value overrides the constructor's."""
    data = np.ones((4, 1))

    def fit_with(ctor_kwargs, fit_kwargs):
        loader = DataLoader(
            lambda: iter([data] * 50), batch_size=4, sample_shape=(1,), total_size=4
        )
        with pm.Model():
            mu = pm.Normal("mu", 0, 1)
            batch = pm.Data("batch", np.zeros((4, 1)))
            pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
            return Trainer(method="advi", dataloader=loader, data_name="batch", **ctor_kwargs).fit(
                5, **fit_kwargs
            )

    a = fit_with({"random_seed": 7}, {})
    b = fit_with({"random_seed": 0}, {"random_seed": 7})
    np.testing.assert_array_equal(a.hist, b.hist)


def test_fit_consumes_exactly_n_batches():
    """fit(n) consumes exactly n minibatches: one seeds the placeholder before
    step 0 and the advance after the final step is skipped, so an (n+1)-th
    batch is never fetched."""
    blocks = [np.full((2, 1), float(i)) for i in range(2)]
    loader = DataLoader(lambda: iter(blocks), batch_size=2, sample_shape=(1,), total_size=4)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((2, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        Trainer(method="advi", dataloader=loader).fit(3, random_seed=0)
    assert loader.batches_seen == 3
    assert loader.rows_streamed == 6


def test_fit_one_step_on_single_batch_one_shot_source():
    """A finite stream with exactly the batches needed must not be over-consumed:
    fit(1) on a one-batch, one-shot source trains and returns instead of failing
    on a post-final restart."""
    loader = DataLoader(iter([np.ones((2, 1))]), batch_size=2, sample_shape=(1,), total_size=2)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((2, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        approx = Trainer(method="advi", dataloader=loader).fit(1, random_seed=0)
    assert len(approx.hist) == 1
    assert loader.batches_seen == 1


def test_refine_after_fit_resumes_the_stream():
    """Inference.refine replays pm.fit's saved callbacks. Because the advance
    skips only fit's own final step (and not every step past n), refine resumes
    advancing the stream instead of going permanently dead on the last batch.

    refine does not re-seed, so its first step still trains on the batch fit left
    in the placeholder; this pins that resume-not-reseed behavior with distinct
    batch markers rather than claiming every refine step is fresh.
    """
    blocks = [np.full((4, 1), float(i)) for i in range(50)]
    loader = DataLoader(lambda: iter(blocks), batch_size=4, sample_shape=(1,), total_size=4)
    sets = []
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        original = model.set_data
        model.set_data = lambda name, values, *a, **k: (  # type: ignore[method-assign]
            sets.append(float(np.asarray(values)[0, 0])),
            original(name, values, *a, **k),
        )[1]
        inference = pm.ADVI(random_seed=0)
        Trainer(method=inference, dataloader=loader).fit(3)
        assert sets == [0.0, 1.0, 2.0]  # fit seeds 0, advances to 1 and 2, skips its last
        sets.clear()
        inference.refine(4, progressbar=False)
    # refine resumes from where the stream stopped (3, 4, 5, ...), not stuck on 2
    assert sets == [3.0, 4.0, 5.0, 6.0]
    assert loader.batches_seen == 7


def test_total_size_check_fires_when_fit_ends_at_pass_boundary():
    """fit(n) with n exactly the batches in one pass still runs the total_size
    sanity check: the stream is kept one batch ahead, so stopping at the
    boundary does not abandon the check right before it would fire."""
    data = np.zeros((40, 1))
    loader = DataLoader(chunked_factory(data, 10), batch_size=10, sample_shape=(1,), total_size=400)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((10, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        with pytest.warns(UserWarning, match="disagrees with"):
            Trainer(method="advi", dataloader=loader).fit(4, random_seed=0)


def test_fit_rejects_nonpositive_n():
    """fit consumes the seed batch before pm.fit could reject n itself, so a
    non-positive n is refused up front, before touching the stream."""
    loader = DataLoader(
        lambda: iter([np.zeros((2, 1))]), batch_size=2, sample_shape=(1,), total_size=2
    )
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((2, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        with pytest.raises(ValueError, match="positive integer"):
            Trainer(method="advi", dataloader=loader).fit(0)
    assert loader.batches_seen == 0


def test_unknown_data_name_raises_before_consuming():
    """A data_name that is not in the model raises a guided KeyError before any
    batch is pulled from the loader."""
    loader = DataLoader(
        lambda: iter([np.zeros((4, 1))] * 3), batch_size=4, sample_shape=(1,), total_size=4
    )
    with pm.Model():
        pm.Normal("mu", 0, 1)
        with pytest.raises(KeyError, match="pm.Data placeholder"):
            Trainer(method="advi", dataloader=loader, data_name="nope").fit(2)
    assert loader.batches_seen == 0
    assert loader.rows_streamed == 0
