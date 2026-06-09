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
"""total_size='auto' resolution + the rows_streamed sanity warning."""

import warnings

import numpy as np
import pytest

from pymc.variational.streaming import (
    DataLoader,
    IterableDataset,
    parquet_source,
    shuffle_buffer,
)


def _factory(data, size):
    """A re-readable zero-arg factory yielding `size`-row chunks of `data`."""

    def f():
        for i in range(0, len(data), size):
            yield data[i : i + size]

    return f


def test_auto_counts_finite_source():
    # no .n_rows -> auto does one counting pass and resolves the true N.
    data = np.arange(60, dtype="float64").reshape(60, 1)
    with pytest.warns(UserWarning, match="counting pass"):
        ds = DataLoader(_factory(data, 7), batch_size=10, sample_shape=(1,), total_size="auto")
    assert ds.total_size == 60


def test_auto_uses_n_rows_fast_path():
    # source advertises .n_rows -> auto trusts it WITHOUT counting (the factory only
    # really yields 8 rows, but n_rows says 999; auto must return 999).
    data = np.zeros((8, 1))
    f = _factory(data, 4)
    f.n_rows = 999
    ds = DataLoader(f, batch_size=4, sample_shape=(1,), total_size="auto")
    assert ds.total_size == 999


def test_auto_rejects_one_shot_iterator():
    # a bare generator is consumed by counting -> auto must refuse it.
    data = np.zeros((20, 1))
    one_shot = (data[i : i + 4] for i in range(0, 20, 4))
    with pytest.raises(ValueError, match="re-readable"):
        DataLoader(one_shot, batch_size=4, sample_shape=(1,), total_size="auto")


def test_shuffle_buffer_forwards_n_rows_for_auto():
    # shuffle_buffer must forward a known .n_rows so total_size="auto" works through
    # the explicit shuffle_buffer(parquet_source(...)) composition WITHOUT a counting
    # pass (the realistic way power users wrap a Parquet source).
    data = np.arange(40, dtype="float64").reshape(40, 1)
    src = _factory(data, 8)
    src.n_rows = 40
    wrapped = shuffle_buffer(src, buffer_size=20, batch_size=10, seed=0)
    assert wrapped.n_rows == 40

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # a counting pass would warn -> fail
        ds = DataLoader(wrapped, batch_size=10, sample_shape=(1,), total_size="auto")
    assert ds.total_size == 40


def test_dataloader_shuffle_auto_resolves_via_n_rows():
    # DataLoader(shuffle=True, total_size="auto") must resolve N from the source's
    # .n_rows WITHOUT a counting pass, even though shuffle wraps the source.
    data = np.arange(40, dtype="float64").reshape(40, 1)
    src = _factory(data, 8)
    src.n_rows = 40
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ds = DataLoader(
            src,
            batch_size=10,
            shuffle=True,
            buffer_size=20,
            seed=0,
            sample_shape=(1,),
            total_size="auto",
        )
    assert ds.total_size == 40


def test_shuffle_buffer_without_n_rows_has_no_attribute():
    # a plain source without .n_rows must not gain a bogus one.
    data = np.arange(40, dtype="float64").reshape(40, 1)
    wrapped = shuffle_buffer(_factory(data, 8), buffer_size=20, batch_size=10, seed=0)
    assert not hasattr(wrapped, "n_rows")


def test_auto_rejects_factory_returning_same_one_shot_iterator():
    # a "factory" that hands back the SAME already-consumable iterator each call is
    # not re-readable: the counting pass consumes it and advance() would get nothing.
    data = np.zeros((20, 1))
    one_shot = (data[i : i + 4] for i in range(0, 20, 4))
    with pytest.raises(ValueError, match="fresh iterator"):
        DataLoader(lambda: one_shot, batch_size=4, sample_shape=(1,), total_size="auto")


def test_auto_rejects_bad_n_rows():
    f = _factory(np.zeros((8, 1)), 4)
    f.n_rows = 0
    with pytest.raises(ValueError, match="n_rows must be a positive integer"):
        DataLoader(f, batch_size=4, sample_shape=(1,), total_size="auto")


def test_sanity_warns_on_grossly_wrong_total_size():
    # one full pass = 20 rows, but total_size=100 -> at the first epoch boundary, warn.
    data = np.arange(20, dtype="float64").reshape(20, 1)
    ds = DataLoader(_factory(data, 4), batch_size=4, sample_shape=(1,), total_size=100)
    with pytest.warns(UserWarning, match="disagrees with"):
        for _ in range(6):  # 5 batches = one epoch, the 6th crosses the boundary
            ds.advance()


def test_sanity_silent_when_total_size_matches():
    data = np.arange(20, dtype="float64").reshape(20, 1)
    ds = DataLoader(_factory(data, 4), batch_size=4, sample_shape=(1,), total_size=20)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # any UserWarning fails the test
        for _ in range(6):
            ds.advance()


def test_parquet_source_n_rows_from_metadata(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rng = np.random.default_rng(0)
    total = 0
    for i in range(3):
        n = 100 + 50 * i
        total += n
        block = rng.normal(size=(n, 2))
        pq.write_table(
            pa.table({"a": block[:, 0], "b": block[:, 1]}),
            f"{tmp_path}/part_{i:02d}.parquet",
        )
    src = parquet_source(str(tmp_path))
    assert isinstance(src, IterableDataset)  # parquet_source is a dataset now
    assert src.n_rows == total  # read from metadata, no data scan

    # and total_size='auto' picks it up for free (no counting pass / warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ds = DataLoader(src, batch_size=10, sample_shape=(2,), total_size="auto")
    assert ds.total_size == total


def test_auto_counts_unshuffled_source_when_shuffling_non_divisible():
    # total_size="auto" with shuffle=True must count the UNSHUFFLED source: the
    # shuffle buffer drops the final partial batch, so counting through it would
    # undercount N by up to batch_size-1. N=125 is not divisible by batch_size=10.
    data = np.arange(125, dtype="float64").reshape(125, 1)
    with pytest.warns(UserWarning, match="counting pass"):
        ds = DataLoader(
            _factory(data, 125),  # one chunk, NO .n_rows -> forces a counting pass
            batch_size=10,
            shuffle=True,
            buffer_size=30,
            seed=0,
            sample_shape=(1,),
            total_size="auto",
        )
    assert ds.total_size == 125  # exact N, not 120 (was undercounted via the shuffle wrap)


def test_stream_batches_updates_counters_and_warns_on_wrong_total_size():
    # The accounting-aware stream the Trainer iterates (loader._stream_batches) must
    # update the public counters AND fire the one-shot sanity check at the epoch
    # boundary -- so a grossly wrong hand-passed total_size is still caught on the
    # Trainer's primary path, not only via advance(); plain iteration stays pure.
    data = np.arange(40, dtype="float64").reshape(20, 2)
    ds = DataLoader(
        _factory(data, 5),  # 4 chunks of 5 rows
        batch_size=5,
        sample_shape=(2,),
        total_size=10_000,  # grossly wrong vs the 20 rows actually streamed
    )
    assert ds.batches_seen == 0 and ds.rows_streamed == 0
    list(ds)  # plain __iter__ must NOT mutate counters
    assert ds.batches_seen == 0 and ds.rows_streamed == 0
    with pytest.warns(UserWarning, match="disagrees with"):
        batches = list(ds._stream_batches())  # one epoch through the Trainer's path
    assert len(batches) == 4
    assert ds.batches_seen == 4
    assert ds.rows_streamed == 20
