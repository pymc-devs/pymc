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
"""Out-of-core minibatching for variational inference.

``pm.Minibatch`` random-indexes an array that is *fully resident in memory*; its
peak memory is therefore O(N) in the dataset size.  ``StreamingDataset`` instead
feeds minibatches from an arbitrary source (a generator, a directory of Parquet
shards, ...) into a small fixed-size ``pytensor.shared`` buffer, so peak memory is
O(buffer) -- the batch buffer plus, if used, the shuffle buffer -- and
independent of N.  The unbiased-gradient rescaling is the *same* as for
``pm.Minibatch``: pass ``total_size=N`` to the observed distribution and PyMC
scales the minibatch log-likelihood by ``N / batch_size`` through the existing
:func:`~pymc.variational.minibatch_rv.create_minibatch_rv`.

The one extra obligation relative to ``pm.Minibatch`` is **shuffling**.
``pm.Minibatch`` draws a fresh uniform index over all N rows every step, so its
minibatches are i.i.d. by construction.  A streaming source is only as well
mixed as the order it yields rows in: reading time/row-ordered data through a
*bounded* buffer is merely a block-shuffle and biases the variational posterior.
Pre-shuffle the data once (or interleave shards) and/or use :func:`shuffle_buffer`.

Example
-------
.. code-block:: python

    import pymc as pm
    from pymc.variational.streaming import StreamingDataset, shuffle_buffer

    N = 10_000_000  # rows on disk; never all in memory at once


    def chunks():  # yields (rows, n_features+1) float64 blocks off disk
        for shard in shards:
            yield read(shard)


    ds = StreamingDataset(
        shuffle_buffer(chunks, buffer_size=1_000_000, batch_size=4096, seed=0),
        batch_size=4096,
        sample_shape=(4,),  # 3 features + 1 observed column
        total_size=N,
    )
    ds.advance()  # seed the buffer

    with pm.Model():
        b = pm.Normal("b", 0.0, 3.0, shape=4)
        buf = ds.as_tensor()  # (batch_size, 4) shared
        logit = b[0] + b[1] * buf[:, 0] + b[2] * buf[:, 1] + b[3] * buf[:, 2]
        pm.Bernoulli("y", logit_p=logit, observed=buf[:, 3], total_size=ds.total_size)
        approx = pm.fit(20_000, method="advi", callbacks=[ds.fit_callback()])
"""

from __future__ import annotations

import warnings

from collections.abc import Callable, Iterable, Iterator

import numpy as np
import pytensor
import pytensor.tensor as pt


class StreamingDataset:
    """Feed minibatches to variational inference from an out-of-core source.

    Parameters
    ----------
    source : Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]]
        Yields ``np.ndarray`` batches of shape ``(batch_size, *sample_shape)``.
        Pass a zero-arg *callable* (a factory) so the stream can be restarted
        when ``cycle=True``; a bare generator can only be consumed once.
    batch_size : int
        Leading dimension of every yielded batch (and of the buffer).
    sample_shape : tuple of int, default ()
        Trailing shape of a single observation. ``()`` for scalar observations,
        ``(k,)`` to stream ``k`` columns (e.g. features + the observed column).
    dtype : str, default "float64"
        Dtype of the shared buffer. If it differs from ``pytensor.config.floatX``
        the model will insert a per-step cast on the observed tensor.
    total_size : int, optional
        The true dataset size ``N``. Pass it to the observed distribution as
        ``total_size=ds.total_size`` so the minibatch log-likelihood is rescaled
        by ``N / batch_size`` (the same mechanism as ``pm.Minibatch``). Unlike
        ``pm.Minibatch`` it cannot be inferred from a resident array, so it must
        be supplied; a warning is issued at construction if it is left ``None``.
    preprocess_fn : callable, optional
        Pure transform applied to each batch before it lands in the buffer.
    cycle : bool, default True
        Restart the source when exhausted (the usual case: many epochs). If
        ``False``, :meth:`advance` raises ``StopIteration`` once exhausted.
    name : str
        Name of the underlying ``pytensor.shared`` variable.
    """

    def __init__(
        self,
        source: Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]],
        *,
        batch_size: int,
        sample_shape: tuple[int, ...] = (),
        dtype: str = "float64",
        total_size: int | None = None,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        cycle: bool = True,
        name: str = "streaming_buffer",
    ):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        if total_size is None:
            warnings.warn(
                "StreamingDataset created with total_size=None: the minibatch "
                "log-likelihood will NOT be rescaled and the posterior will be "
                "biased. Pass total_size=N (the true dataset size).",
                UserWarning,
                stacklevel=2,
            )

        self._source_factory = _make_factory(source)
        self._source_iter: Iterator[np.ndarray] = self._source_factory()
        self._batch_size = batch_size
        self._sample_shape = tuple(sample_shape)
        self._dtype = dtype
        self._total_size = total_size
        self._preprocess_fn = preprocess_fn
        self._cycle = cycle

        self._batches_seen = 0
        self._rows_streamed = 0

        self._shared = pytensor.shared(
            np.zeros((batch_size, *self._sample_shape), dtype=dtype), name=name
        )

    # ----- read-only state ---------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def total_size(self) -> int | None:
        """The dataset size ``N`` (pass to the distribution's ``total_size``)."""
        return self._total_size

    @property
    def batches_seen(self) -> int:
        return self._batches_seen

    @property
    def rows_streamed(self) -> int:
        """Total rows pushed through the buffer (grows past ``N`` across epochs)."""
        return self._rows_streamed

    # ----- the model-facing tensor ------------------------------------------

    def as_tensor(self) -> pt.TensorVariable:
        """The ``pytensor.shared`` buffer the model observes (mutates each step)."""
        return self._shared

    # ----- the only mutator --------------------------------------------------

    def advance(self) -> None:
        """Pull the next batch from the source into the buffer."""
        batch = self._next_batch()
        if self._preprocess_fn is not None:
            batch = self._preprocess_fn(batch)
        self._validate(batch)
        # Own a fresh contiguous copy before borrowing into the shared variable:
        # the source may legitimately yield *views* into a reused array, so we
        # must not alias it. np.array(copy default) guarantees an owned array.
        arr = np.array(batch, dtype=self._dtype)
        self._shared.set_value(arr, borrow=True)
        self._batches_seen += 1
        self._rows_streamed += int(arr.shape[0])

    def fit_callback(self) -> Callable:
        """A 3-arg callback ``(approx, losses, i)`` for ``pm.fit(callbacks=...)``."""

        def _cb(*_):
            self.advance()

        return _cb

    # ----- iterator sugar ----------------------------------------------------

    def __iter__(self) -> StreamingDataset:
        return self

    def __next__(self) -> np.ndarray:
        self.advance()
        return self._shared.get_value(borrow=False)  # an owned copy, safe to keep

    # ----- internals ---------------------------------------------------------

    def _next_batch(self) -> np.ndarray:
        try:
            return next(self._source_iter)
        except StopIteration:
            if not self._cycle:
                raise
            self._source_iter = self._source_factory()
            return next(self._source_iter)

    def _validate(self, batch: np.ndarray) -> None:
        if not isinstance(batch, np.ndarray):
            raise TypeError(f"expected np.ndarray batch, got {type(batch).__name__}")
        if batch.shape[0] != self._batch_size:
            raise ValueError(
                f"batch shape[0] = {batch.shape[0]} does not match batch_size = "
                f"{self._batch_size}; partial batches are not allowed (drop them in "
                "the source, e.g. via shuffle_buffer)."
            )
        if batch.shape[1:] != self._sample_shape:
            raise ValueError(
                f"batch sample-shape {batch.shape[1:]} does not match declared "
                f"sample_shape={self._sample_shape}"
            )


def shuffle_buffer(
    chunk_source: Callable[[], Iterator[np.ndarray]],
    *,
    buffer_size: int,
    batch_size: int,
    seed: int | None = None,
) -> Callable[[], Iterator[np.ndarray]]:
    """Wrap a chunk source into a shuffled, fixed-size batch source.

    Accumulates rows from ``chunk_source`` into a buffer of at least
    ``buffer_size`` rows, shuffles it, and yields ``batch_size`` slices; rows that
    do not fill a final batch are **carried over** into the next buffer (never
    dropped) until the source is exhausted, at which point a single trailing
    partial batch (< ``batch_size`` rows) is dropped. This approximates i.i.d.
    minibatches from an *unordered* or pre-shuffled stream.

    It does **not** by itself fix a strongly time/row-ordered stream (a bounded
    buffer only block-shuffles such data) -- pre-shuffle on disk, or interleave
    shards into ``chunk_source``, for that. Note ``buffer_size`` is a *lower*
    bound: a single yielded chunk larger than ``buffer_size`` is taken whole, so
    peak buffer memory is ``max(buffer_size, largest_chunk_rows)``.
    """

    def factory() -> Iterator[np.ndarray]:
        rng = np.random.default_rng(seed)
        it = chunk_source()
        carry: np.ndarray | None = None  # leftover (< batch_size) from last fill
        exhausted = False
        while not exhausted:
            bufs: list[np.ndarray] = []
            have = 0
            if carry is not None:
                bufs.append(carry)
                have += carry.shape[0]
                carry = None
            for arr in it:
                a = np.asarray(arr)
                bufs.append(a)
                have += a.shape[0]
                if have >= buffer_size:
                    break
            else:
                exhausted = True  # for-loop ran to completion: source is done
            if have < batch_size:
                return  # nothing left that can form a batch
            buf = np.concatenate(bufs, axis=0)  # always a fresh, owned copy
            rng.shuffle(buf)
            n_full = buf.shape[0] // batch_size
            for i in range(n_full):
                yield buf[i * batch_size : (i + 1) * batch_size]
            rem = buf.shape[0] - n_full * batch_size
            carry = buf[n_full * batch_size :].copy() if rem else None

    return factory


def _make_factory(
    source: Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]],
) -> Callable[[], Iterator[np.ndarray]]:
    """Coerce ``source`` into a zero-arg callable returning a fresh iterator.

    A callable that is not itself an iterator is treated as the factory; a bare
    iterator is wrapped (and refuses a second epoch); any other iterable is
    re-``iter``-ed each epoch.
    """
    if callable(source) and not isinstance(source, Iterator):
        return source  # type: ignore[return-value]
    if isinstance(source, Iterator):
        consumed = {"done": False}

        def _factory() -> Iterator[np.ndarray]:
            if consumed["done"]:
                raise RuntimeError(
                    "source is a bare iterator and cycle=True was requested; pass a "
                    "zero-arg factory or a re-iterable instead"
                )
            consumed["done"] = True
            return source

        return _factory
    return lambda: iter(source)
