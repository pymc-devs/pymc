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
    """Without .n_rows, 'auto' does one counting pass and resolves the true N."""
    data = np.arange(60, dtype="float64").reshape(60, 1)
    with pytest.warns(UserWarning, match="counting pass"):
        ds = DataLoader(_factory(data, 7), batch_size=10, sample_shape=(1,), total_size="auto")
    assert ds.total_size == 60


def test_auto_uses_n_rows_fast_path():
    """A source-advertised .n_rows is trusted without a counting pass."""
    data = np.zeros((8, 1))
    f = _factory(data, 4)
    f.n_rows = 999
    ds = DataLoader(f, batch_size=4, sample_shape=(1,), total_size="auto")
    assert ds.total_size == 999


def test_auto_rejects_one_shot_iterator():
    """A bare generator would be consumed by the counting pass, so 'auto' refuses it."""
    data = np.zeros((20, 1))
    one_shot = (data[i : i + 4] for i in range(0, 20, 4))
    with pytest.raises(ValueError, match="re-readable"):
        DataLoader(one_shot, batch_size=4, sample_shape=(1,), total_size="auto")


def test_shuffle_buffer_forwards_n_rows_for_auto():
    """shuffle_buffer forwards a known .n_rows so total_size='auto' works through
    an explicit shuffle_buffer(parquet_source(...)) composition without counting."""
    data = np.arange(40, dtype="float64").reshape(40, 1)
    src = _factory(data, 8)
    src.n_rows = 40
    wrapped = shuffle_buffer(src, buffer_size=20, batch_size=10, seed=0)
    assert wrapped.n_rows == 40

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ds = DataLoader(wrapped, batch_size=10, sample_shape=(1,), total_size="auto")
    assert ds.total_size == 40


def test_dataloader_shuffle_auto_resolves_via_n_rows():
    """DataLoader(shuffle=True, total_size='auto') resolves N from the source's
    .n_rows without a counting pass, even though shuffle wraps the source."""
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
    """A source without .n_rows must not gain a bogus one through the wrapper."""
    data = np.arange(40, dtype="float64").reshape(40, 1)
    wrapped = shuffle_buffer(_factory(data, 8), buffer_size=20, batch_size=10, seed=0)
    assert not hasattr(wrapped, "n_rows")


def test_auto_rejects_factory_returning_same_one_shot_iterator():
    """A factory that returns the same already-consumed iterator each call is not
    re-readable; the counting pass detects and refuses it."""
    data = np.zeros((20, 1))
    one_shot = (data[i : i + 4] for i in range(0, 20, 4))
    with pytest.raises(ValueError, match="fresh iterator"):
        DataLoader(lambda: one_shot, batch_size=4, sample_shape=(1,), total_size="auto")


def test_auto_rejects_bad_n_rows():
    """A non-positive source .n_rows is rejected instead of trusted."""
    f = _factory(np.zeros((8, 1)), 4)
    f.n_rows = 0
    with pytest.raises(ValueError, match="n_rows must be a positive integer"):
        DataLoader(f, batch_size=4, sample_shape=(1,), total_size="auto")


def test_sanity_warns_on_grossly_wrong_total_size():
    """A hand-passed total_size that grossly disagrees with the rows actually
    streamed in one pass triggers the one-shot warning at the epoch boundary."""
    data = np.arange(20, dtype="float64").reshape(20, 1)
    ds = DataLoader(_factory(data, 4), batch_size=4, sample_shape=(1,), total_size=100)
    with pytest.warns(UserWarning, match="disagrees with"):
        list(ds._stream_batches())


def test_sanity_silent_when_total_size_matches():
    """No warning when total_size matches the rows streamed in one pass."""
    data = np.arange(20, dtype="float64").reshape(20, 1)
    ds = DataLoader(_factory(data, 4), batch_size=4, sample_shape=(1,), total_size=20)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        list(ds._stream_batches())


def test_parquet_source_n_rows_from_metadata(tmp_path):
    """parquet_source reads n_rows from file metadata (no data scan) and
    total_size='auto' picks it up without a counting pass."""
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
    assert isinstance(src, IterableDataset)
    assert src.n_rows == total

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ds = DataLoader(src, batch_size=10, sample_shape=(2,), total_size="auto")
    assert ds.total_size == total


def test_parquet_source_columns_and_shard_order(tmp_path):
    """columns= selects a column subset and shards are read in sorted path order."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    for i in range(2):
        pq.write_table(
            pa.table(
                {"a": [float(i)] * 2, "b": [9.0] * 2, "c": [float(10 + i)] * 2},
            ),
            f"{tmp_path}/part_{i}.parquet",
        )
    src = parquet_source(str(tmp_path), columns=["a", "c"])
    blocks = list(src)
    assert [b.shape for b in blocks] == [(2, 2), (2, 2)]
    np.testing.assert_array_equal(blocks[0][:, 0], [0.0, 0.0])
    np.testing.assert_array_equal(blocks[1][:, 1], [11.0, 11.0])


def test_parquet_source_empty_dir_raises(tmp_path):
    """A directory with no matching Parquet files raises a clear error."""
    pytest.importorskip("pyarrow")
    with pytest.raises(ValueError, match="no Parquet files match"):
        parquet_source(str(tmp_path))


def test_auto_counts_unshuffled_source_when_shuffling_non_divisible():
    """total_size='auto' with shuffle=True counts the unshuffled source: the
    shuffle buffer drops the trailing partial batch, so counting through it would
    undercount N by up to batch_size - 1 (here 125 vs 120)."""
    data = np.arange(125, dtype="float64").reshape(125, 1)
    with pytest.warns(UserWarning, match="counting pass"):
        ds = DataLoader(
            _factory(data, 125),
            batch_size=10,
            shuffle=True,
            buffer_size=30,
            seed=0,
            sample_shape=(1,),
            total_size="auto",
        )
    assert ds.total_size == 125


def test_stream_batches_updates_counters_and_warns_on_wrong_total_size():
    """The accounting stream the Trainer iterates updates the public counters and
    fires the one-shot total_size sanity check at the epoch boundary, while plain
    __iter__ stays side-effect-free."""
    data = np.arange(40, dtype="float64").reshape(20, 2)
    ds = DataLoader(
        _factory(data, 5),
        batch_size=5,
        sample_shape=(2,),
        total_size=10_000,
    )
    assert ds.batches_seen == 0 and ds.rows_streamed == 0
    list(ds)
    assert ds.batches_seen == 0 and ds.rows_streamed == 0
    with pytest.warns(UserWarning, match="disagrees with"):
        batches = list(ds._stream_batches())
    assert len(batches) == 4
    assert ds.batches_seen == 4
    assert ds.rows_streamed == 20
