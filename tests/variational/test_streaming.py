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

from pymc.variational.streaming import StreamingDataset, shuffle_buffer


def _chunks(data, size):
    def factory():
        for i in range(0, len(data), size):
            yield data[i : i + size]

    return factory


def test_advance_shape_and_counters():
    data = np.arange(40, dtype="float64").reshape(20, 2)
    ds = StreamingDataset(_chunks(data, 4), batch_size=4, sample_shape=(2,), total_size=20)
    assert ds.batches_seen == 0
    ds.advance()
    assert ds.as_tensor().get_value().shape == (4, 2)
    assert ds.batches_seen == 1 and ds.rows_streamed == 4
    ds.advance()
    assert ds.batches_seen == 2 and ds.rows_streamed == 8


def test_wrong_batch_shape_rejected():
    data = np.zeros((10, 2))
    ds = StreamingDataset(_chunks(data, 3), batch_size=4, sample_shape=(2,), total_size=10)
    with pytest.raises(ValueError, match="does not match batch_size"):
        ds.advance()


def test_total_size_none_warns_at_construction():
    data = np.zeros((8, 1))
    with pytest.warns(UserWarning, match="total_size=None"):
        StreamingDataset(_chunks(data, 4), batch_size=4, sample_shape=(1,))


def test_cycle_true_restarts_source():
    data = np.arange(8, dtype="float64").reshape(8, 1)
    ds = StreamingDataset(
        _chunks(data, 4), batch_size=4, sample_shape=(1,), total_size=8, cycle=True
    )
    for _ in range(4):  # two epochs worth
        ds.advance()
    assert ds.batches_seen == 4


def test_cycle_false_raises_when_exhausted():
    data = np.arange(8, dtype="float64").reshape(8, 1)
    ds = StreamingDataset(
        _chunks(data, 4), batch_size=4, sample_shape=(1,), total_size=8, cycle=False
    )
    ds.advance()
    ds.advance()
    with pytest.raises(StopIteration):
        ds.advance()


def test_preprocess_fn_applied():
    data = np.ones((8, 1))
    ds = StreamingDataset(
        _chunks(data, 4),
        batch_size=4,
        sample_shape=(1,),
        total_size=8,
        preprocess_fn=lambda b: b * 3.0,
    )
    ds.advance()
    np.testing.assert_array_equal(ds.as_tensor().get_value(), np.full((4, 1), 3.0))


def test_shuffle_buffer_conserves_rows_non_dividing():
    # buffer_size and chunk size deliberately do NOT divide batch_size: the
    # carry-over must not lose or duplicate any row (regression for the drop bug).
    data = np.arange(140, dtype="float64").reshape(140, 1)
    src = shuffle_buffer(_chunks(data, 7), buffer_size=55, batch_size=10, seed=0)
    batches = list(src())
    assert all(b.shape == (10, 1) for b in batches)
    seen = np.sort(np.concatenate([b.ravel() for b in batches]))
    # 140 rows, batch 10 -> 14 full batches, nothing dropped (140 % 10 == 0)
    np.testing.assert_array_equal(seen, data.ravel())


def test_shuffle_buffer_does_not_mutate_source():
    data = np.arange(100, dtype="float64").reshape(100, 1)
    original = data.copy()
    src = shuffle_buffer(_chunks(data, 25), buffer_size=40, batch_size=10, seed=1)
    list(src())
    np.testing.assert_array_equal(data, original)  # source untouched


def test_total_size_rescales_logp_like_minibatch():
    # observed=buf[:, k] + total_size=N must scale the observed log-likelihood by
    # N / batch_size via the existing create_minibatch_rv path -- pin this without
    # training anything.
    rng = np.random.default_rng(0)
    N, bs = 1000, 16
    data = rng.normal(size=(bs, 1))
    ds = StreamingDataset(lambda: iter([data]), batch_size=bs, sample_shape=(1,), total_size=N)
    ds.advance()

    with pm.Model() as scaled:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1, observed=ds.as_tensor()[:, 0], total_size=ds.total_size)
    with pm.Model() as plain:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1, observed=data[:, 0])  # no total_size

    point = {"mu": np.array(0.3)}
    obs_scaled = scaled.compile_logp(scaled.observed_RVs)(point)
    obs_plain = plain.compile_logp(plain.observed_RVs)(point)
    np.testing.assert_allclose(obs_scaled, obs_plain * (N / bs), rtol=1e-6)


def test_equivalence_with_in_ram_minibatch():
    """End-to-end: streaming ADVI reproduces in-RAM pm.Minibatch ADVI."""
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

    ds = StreamingDataset(
        shuffle_buffer(_chunks(data, 20_000), buffer_size=40_000, batch_size=bs, seed=seed),
        batch_size=bs,
        sample_shape=(3,),
        total_size=N,
    )
    ds.advance()
    with pm.Model():
        b = pm.Normal("b", 0, 3, shape=3)
        buf = ds.as_tensor()
        pm.Bernoulli(
            "o",
            logit_p=b[0] + b[1] * buf[:, 0] + b[2] * buf[:, 1],
            observed=buf[:, 2],
            total_size=ds.total_size,
        )
        ap = pm.fit(
            6000,
            method="advi",
            obj_optimizer=pm.adam(learning_rate=0.02),
            callbacks=[ds.fit_callback()],
            progressbar=False,
            random_seed=seed,
        )
        stream = ap.sample(400).posterior["b"].values.reshape(-1, 3).mean(0)

    np.testing.assert_allclose(in_ram, stream, atol=0.1)


def test_total_size_zero_raises():
    # total_size=0 is falsy: it slips a None-only check and the model's truthy
    # `if total_size:` guard, silently skipping the N/batch_size rescaling.
    data = np.zeros((8, 1))
    with pytest.raises(ValueError, match="positive integer"):
        StreamingDataset(_chunks(data, 4), batch_size=4, sample_shape=(1,), total_size=0)


def test_total_size_negative_raises():
    # negative total_size is truthy but yields a negative scaling coefficient
    # (the data log-likelihood's sign flips, so VI maximizes mis-fit).
    data = np.zeros((8, 1))
    with pytest.raises(ValueError, match="positive integer"):
        StreamingDataset(_chunks(data, 4), batch_size=4, sample_shape=(1,), total_size=-100)


def test_shuffle_buffer_small_buffer_conserves_rows():
    # buffer_size < batch_size must NOT silently discard the dataset: the buffer
    # accumulates to at least batch_size before emitting (regression for the
    # early-return data-loss bug).
    data = np.arange(120, dtype="float64").reshape(120, 1)
    src = shuffle_buffer(_chunks(data, 7), buffer_size=3, batch_size=10, seed=0)
    batches = list(src())
    assert batches, "buffer_size < batch_size silently produced zero batches"
    assert all(b.shape == (10, 1) for b in batches)
    seen = np.sort(np.concatenate([b.ravel() for b in batches]))
    np.testing.assert_array_equal(seen, data.ravel())  # 120 % 10 == 0, nothing dropped


def test_shuffle_buffer_rejects_nonpositive_sizes():
    data = np.zeros((10, 1))
    with pytest.raises(ValueError, match="buffer_size"):
        shuffle_buffer(_chunks(data, 5), buffer_size=0, batch_size=4)
    with pytest.raises(ValueError, match="batch_size"):
        shuffle_buffer(_chunks(data, 5), buffer_size=10, batch_size=0)


def test_accepts_numpy_integer_sizes_rejects_bool():
    # the positive-int check uses numbers.Integral: numpy ints are valid, bool is not.
    data = np.zeros((8, 1))
    ds = StreamingDataset(
        _chunks(data, 4), batch_size=np.int64(4), sample_shape=(1,), total_size=np.int64(8)
    )
    ds.advance()
    assert ds.batch_size == 4
    with pytest.raises(ValueError):
        StreamingDataset(_chunks(data, 4), batch_size=True, sample_shape=(1,), total_size=8)


def test_shuffle_buffer_reshuffles_across_epochs():
    # a seeded buffer must NOT replay one fixed permutation every epoch (that
    # would weaken shuffling under cycle=True); each epoch reshuffles, but rows
    # are conserved.
    data = np.arange(60, dtype="float64").reshape(60, 1)
    factory = shuffle_buffer(_chunks(data, 10), buffer_size=60, batch_size=10, seed=0)
    epoch1 = np.concatenate([b.ravel() for b in factory()])
    epoch2 = np.concatenate([b.ravel() for b in factory()])
    assert not np.array_equal(epoch1, epoch2)  # different order across epochs
    np.testing.assert_array_equal(np.sort(epoch1), data.ravel())  # but conserves rows
    np.testing.assert_array_equal(np.sort(epoch2), data.ravel())


def test_shuffle_buffer_seed_reproducible_across_runs():
    # same seed => identical first-epoch order across independent constructions.
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
    # numpy integer sizes must be stored as plain Python ints so ds.total_size is
    # accepted downstream by create_minibatch_rv (regression for the np.int64 trap).
    data = np.zeros((8, 1))
    ds = StreamingDataset(
        _chunks(data, 4), batch_size=np.int64(4), sample_shape=(1,), total_size=np.int64(8)
    )
    assert type(ds.batch_size) is int
    assert type(ds.total_size) is int


def test_numpy_total_size_accepted_by_observed_rv():
    # a stored np.int64 total_size used to reach create_minibatch_rv and raise
    # "Invalid type for total_size"; it must now build a valid observed RV.
    data = np.zeros((4, 1), dtype="float64")
    ds = StreamingDataset(
        lambda: iter([data]), batch_size=4, sample_shape=(1,), total_size=np.int64(4)
    )
    ds.advance()
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1, observed=ds.as_tensor()[:, 0], total_size=ds.total_size)
    # compiling the observed logp exercises the create_minibatch_rv scaling path
    model.compile_logp(model.observed_RVs)({"mu": np.array(0.0)})


def test_factory_returning_reiterable_is_accepted():
    # a zero-arg factory may return ANY iterable (e.g. a list), not just an
    # iterator; advance() used to crash with "'list' object is not an iterator".
    data = [np.zeros((4, 1), dtype="float64")]
    ds = StreamingDataset(lambda: data, batch_size=4, sample_shape=(1,), total_size=4)
    ds.advance()
    assert ds.as_tensor().get_value().shape == (4, 1)


def test_scalar_batch_rejected_with_clear_error():
    # a 0-D batch used to raise an opaque IndexError on batch.shape[0].
    ds = StreamingDataset(
        lambda: iter([np.array(1.0)]), batch_size=1, sample_shape=(), total_size=1
    )
    with pytest.raises(ValueError, match="leading batch dimension"):
        ds.advance()


def test_fit_callback_seeds_buffer_by_default():
    # PyMC runs callbacks AFTER each step, so the buffer must be seeded before the
    # first step; fit_callback() seeds on creation unless seed=False.
    data = np.ones((4, 1))
    ds = StreamingDataset(lambda: iter([data, data]), batch_size=4, sample_shape=(1,), total_size=8)
    assert ds.batches_seen == 0
    ds.fit_callback()  # default seed=True
    assert ds.batches_seen == 1
    np.testing.assert_array_equal(ds.as_tensor().get_value(), data)  # not the zero placeholder


def test_fit_callback_seed_false_does_not_advance():
    data = np.ones((4, 1))
    ds = StreamingDataset(lambda: iter([data]), batch_size=4, sample_shape=(1,), total_size=4)
    ds.fit_callback(seed=False)
    assert ds.batches_seen == 0
