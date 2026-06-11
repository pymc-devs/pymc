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
import pytest

import pymc as pm

from pymc.variational.streaming import (
    DataLoader,
    IterableDataset,
    Trainer,
    shuffle_buffer,
)


def _chunks(data, size):
    def factory():
        for i in range(0, len(data), size):
            yield data[i : i + size]

    return factory


def test_plain_loader_rebatches_arbitrary_blocks():
    """Blocks of 3 with batch_size=4 are re-batched in order; the trailing rows
    that cannot fill a final batch are dropped (drop_last semantics)."""
    data = np.arange(20, dtype="float64").reshape(10, 2)
    ds = DataLoader(_chunks(data, 3), batch_size=4, sample_shape=(2,), total_size=10)
    batches = list(ds)
    assert [b.shape for b in batches] == [(4, 2), (4, 2)]
    np.testing.assert_array_equal(np.concatenate(batches), data[:8])


def test_raw_array_source_like_vi_rework_sketch():
    """A raw array works directly, as in the VI-rework sketch
    ``Dataloader(np.random.normal(...), batch_size=...)``: rows are yielded one
    sample at a time, re-batched, and counted as rows by total_size='auto'."""
    data = np.arange(40, dtype="float64").reshape(20, 2)
    with pytest.warns(UserWarning, match="counting pass"):
        ds = DataLoader(data, batch_size=8, sample_shape=(2,), total_size="auto")
    assert ds.total_size == 20
    batches = list(ds)
    assert [b.shape for b in batches] == [(8, 2), (8, 2)]
    np.testing.assert_array_equal(np.concatenate(batches), data[:16])


def test_wrong_sample_shape_rejected():
    """A source whose trailing shape does not match sample_shape raises."""
    data = np.zeros((12, 3))
    ds = DataLoader(_chunks(data, 4), batch_size=4, sample_shape=(2,), total_size=12)
    with pytest.raises(ValueError, match="source yielded shape"):
        next(iter(ds))


def test_total_size_none_warns_at_construction():
    """total_size=None disables the N/batch_size rescaling, so it warns."""
    data = np.zeros((8, 1))
    with pytest.warns(UserWarning, match="total_size=None"):
        DataLoader(_chunks(data, 4), batch_size=4, sample_shape=(1,))


def test_preprocess_fn_applied():
    """preprocess_fn transforms each batch before it is yielded."""
    data = np.ones((8, 1))
    ds = DataLoader(
        _chunks(data, 4),
        batch_size=4,
        sample_shape=(1,),
        total_size=8,
        preprocess_fn=lambda b: b * 3.0,
    )
    np.testing.assert_array_equal(next(iter(ds)), np.full((4, 1), 3.0))


def test_shuffle_buffer_conserves_rows_with_non_dividing_chunks():
    """Chunk and buffer sizes that do not divide batch_size must not lose or
    duplicate rows; the remainder is carried into the next buffer fill."""
    data = np.arange(140, dtype="float64").reshape(140, 1)
    src = shuffle_buffer(_chunks(data, 7), buffer_size=55, batch_size=10, seed=0)
    batches = list(src())
    assert all(b.shape == (10, 1) for b in batches)
    seen = np.sort(np.concatenate([b.ravel() for b in batches]))
    np.testing.assert_array_equal(seen, data.ravel())


def test_shuffle_buffer_does_not_mutate_source():
    """Shuffling happens on an owned copy, never in place on the source arrays."""
    data = np.arange(100, dtype="float64").reshape(100, 1)
    original = data.copy()
    src = shuffle_buffer(_chunks(data, 25), buffer_size=40, batch_size=10, seed=1)
    list(src())
    np.testing.assert_array_equal(data, original)


def test_dataloader_shuffle_true_yields_full_batches():
    """shuffle=True wraps the source in a bounded shuffle_buffer; one epoch yields
    full batches and conserves every row when N divides batch_size."""
    data = np.arange(120, dtype="float64").reshape(120, 1)
    ds = DataLoader(
        _chunks(data, 8),
        batch_size=10,
        shuffle=True,
        buffer_size=40,
        seed=0,
        sample_shape=(1,),
        total_size=120,
    )
    batches = list(ds)
    assert all(b.shape == (10, 1) for b in batches)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([b.ravel() for b in batches])), data.ravel()
    )


def test_total_size_rescales_logp_like_minibatch():
    """total_size=len(loader) scales the observed minibatch log-likelihood by
    exactly N / batch_size, through the same create_minibatch_rv mechanism as
    pm.Minibatch: logp(scaled) == logp(plain) * N / batch_size."""
    rng = np.random.default_rng(0)
    N, bs = 1000, 16
    data = rng.normal(size=(bs, 1))
    loader = DataLoader(lambda: iter([data]), batch_size=bs, sample_shape=(1,), total_size=N)

    with pm.Model() as scaled:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", data)
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
    with pm.Model() as plain:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1, observed=data[:, 0])

    point = {"mu": np.array(0.3)}
    obs_scaled = scaled.compile_logp(scaled.observed_RVs)(point)
    obs_plain = plain.compile_logp(plain.observed_RVs)(point)
    np.testing.assert_allclose(obs_scaled, obs_plain * (N / bs), rtol=1e-6)


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
        _chunks(data, 20_000),
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


def test_len_returns_total_size():
    """len(loader) is the dataset row count N, the value total_size needs."""
    data = np.zeros((40, 1))
    loader = DataLoader(_chunks(data, 8), batch_size=8, sample_shape=(1,), total_size=40)
    assert len(loader) == 40


def test_len_raises_when_total_size_none():
    """With total_size=None there is no N to hand the model, so len() raises
    rather than silently skipping the N/batch_size rescaling."""
    data = np.ones((4, 1))
    with pytest.warns(UserWarning, match="total_size=None"):
        loader = DataLoader(lambda: iter([data] * 5), batch_size=4, sample_shape=(1,))
    with pytest.raises(TypeError, match="total_size=None"):
        len(loader)


def test_iter_yields_clean_batches_and_reiterates():
    """__iter__ yields validated (batch_size, *sample_shape) batches and can be
    re-iterated for another epoch."""
    data = np.arange(40, dtype="float64").reshape(40, 1)
    loader = DataLoader(_chunks(data, 10), batch_size=10, sample_shape=(1,), total_size=40)
    e1 = list(loader)
    e2 = list(loader)
    assert len(e1) == 4 and all(b.shape == (10, 1) for b in e1)
    np.testing.assert_array_equal(np.sort(np.concatenate([b.ravel() for b in e1])), data.ravel())
    np.testing.assert_array_equal(np.sort(np.concatenate([b.ravel() for b in e2])), data.ravel())


def test_total_size_zero_raises():
    """total_size=0 is falsy and would silently skip the rescaling, so it raises."""
    data = np.zeros((8, 1))
    with pytest.raises(ValueError, match="positive integer"):
        DataLoader(_chunks(data, 4), batch_size=4, sample_shape=(1,), total_size=0)


def test_total_size_negative_raises():
    """A negative total_size would flip the sign of the data log-likelihood."""
    data = np.zeros((8, 1))
    with pytest.raises(ValueError, match="positive integer"):
        DataLoader(_chunks(data, 4), batch_size=4, sample_shape=(1,), total_size=-100)


def test_shuffle_buffer_small_buffer_conserves_rows():
    """buffer_size < batch_size must not silently discard the dataset: the buffer
    accumulates to at least batch_size before emitting."""
    data = np.arange(120, dtype="float64").reshape(120, 1)
    src = shuffle_buffer(_chunks(data, 7), buffer_size=3, batch_size=10, seed=0)
    batches = list(src())
    assert batches
    assert all(b.shape == (10, 1) for b in batches)
    seen = np.sort(np.concatenate([b.ravel() for b in batches]))
    np.testing.assert_array_equal(seen, data.ravel())


def test_shuffle_buffer_rejects_nonpositive_sizes():
    """Zero or negative buffer/batch sizes raise at construction."""
    data = np.zeros((10, 1))
    with pytest.raises(ValueError, match="buffer_size"):
        shuffle_buffer(_chunks(data, 5), buffer_size=0, batch_size=4)
    with pytest.raises(ValueError, match="batch_size"):
        shuffle_buffer(_chunks(data, 5), buffer_size=10, batch_size=0)


def test_accepts_numpy_integer_sizes_rejects_bool():
    """The positive-int check uses numbers.Integral: numpy ints pass, bool does not."""
    data = np.zeros((8, 1))
    ds = DataLoader(
        _chunks(data, 4), batch_size=np.int64(4), sample_shape=(1,), total_size=np.int64(8)
    )
    assert next(iter(ds)).shape == (4, 1)
    assert ds.batch_size == 4
    with pytest.raises(ValueError):
        DataLoader(_chunks(data, 4), batch_size=True, sample_shape=(1,), total_size=8)


def test_shuffle_buffer_draws_fresh_permutation_each_epoch():
    """A seeded buffer must not replay one fixed permutation every epoch; each
    epoch reshuffles while conserving rows."""
    data = np.arange(60, dtype="float64").reshape(60, 1)
    factory = shuffle_buffer(_chunks(data, 10), buffer_size=60, batch_size=10, seed=0)
    epoch1 = np.concatenate([b.ravel() for b in factory()])
    epoch2 = np.concatenate([b.ravel() for b in factory()])
    assert not np.array_equal(epoch1, epoch2)
    np.testing.assert_array_equal(np.sort(epoch1), data.ravel())
    np.testing.assert_array_equal(np.sort(epoch2), data.ravel())


def test_shuffle_buffer_seed_reproducible_across_runs():
    """The same seed gives an identical first-epoch order across constructions."""
    data = np.arange(60, dtype="float64").reshape(60, 1)
    a = np.concatenate(
        [
            b.ravel()
            for b in shuffle_buffer(_chunks(data, 10), buffer_size=60, batch_size=10, seed=7)()
        ]
    )
    b = np.concatenate(
        [
            b.ravel()
            for b in shuffle_buffer(_chunks(data, 10), buffer_size=60, batch_size=10, seed=7)()
        ]
    )
    np.testing.assert_array_equal(a, b)


def test_sizes_normalized_to_python_int():
    """Numpy integer sizes are stored as plain Python ints so total_size is
    accepted downstream by create_minibatch_rv."""
    data = np.zeros((8, 1))
    ds = DataLoader(
        _chunks(data, 4), batch_size=np.int64(4), sample_shape=(1,), total_size=np.int64(8)
    )
    assert type(ds.batch_size) is int
    assert type(ds.total_size) is int


def test_numpy_total_size_accepted_by_observed_rv():
    """A numpy-integer total_size used to reach create_minibatch_rv and raise; the
    normalized value must build and compile a valid observed RV."""
    data = np.zeros((4, 1), dtype="float64")
    loader = DataLoader(
        lambda: iter([data]), batch_size=4, sample_shape=(1,), total_size=np.int64(4)
    )
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", data)
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
    model.compile_logp(model.observed_RVs)({"mu": np.array(0.0)})


def test_factory_returning_reiterable_is_accepted():
    """A zero-arg factory may return any iterable (e.g. a list), not just an
    iterator."""
    data = [np.zeros((4, 1), dtype="float64")]
    ds = DataLoader(lambda: data, batch_size=4, sample_shape=(1,), total_size=4)
    assert next(iter(ds)).shape == (4, 1)


def test_scalar_samples_are_batched():
    """With sample_shape=() a 0-D yield is one scalar sample, exactly what
    iterating a raw 1-D array produces; the loader batches scalars."""
    data = np.arange(6, dtype="float64")
    ds = DataLoader(data, batch_size=3, sample_shape=(), total_size=6)
    batches = list(ds)
    assert [b.shape for b in batches] == [(3,), (3,)]
    np.testing.assert_array_equal(np.concatenate(batches), data)


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


def test_iterable_dataset_base_is_abstract():
    """The base class is a contract: __iter__ must be overridden."""
    with pytest.raises(NotImplementedError):
        iter(IterableDataset())
