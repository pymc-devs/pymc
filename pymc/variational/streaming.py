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

``pm.Minibatch`` random-indexes an array that is fully resident in memory; its
peak memory is therefore O(N) in the dataset size. This module instead streams
minibatches from an out-of-core source into a ``pm.Data`` placeholder, so peak
memory is set by the batch, the source chunk, and the optional shuffle buffer,
independent of N.

The API follows PyTorch's ``torch.utils.data``:

* :class:`IterableDataset`: a re-iterable, out-of-core source of rows
  (e.g. :func:`parquet_source` over a directory of shards). It never loads the
  whole dataset; it yields it a chunk at a time.
* :class:`DataLoader`: turns a dataset into fixed-size (optionally shuffled)
  minibatches; it is iterable (the minibatch stream) and sized. Note ``len(loader)``
  is the row count ``N`` (what the observed distribution needs for ``total_size``),
  not the batch count ``torch.utils.data.DataLoader.__len__`` returns.
* :class:`Trainer`: drives variational inference (ADVI, ...) over a
  ``DataLoader`` with no user-facing callbacks;
  ``Trainer(method=..., dataloader=...).fit(n)`` streams each minibatch into the
  model's ``pm.Data`` placeholder with ``set_data``.

With bounded source chunks the full data never sits in RAM at once. The model
graph observes only a ``(batch_size, *sample_shape)`` ``pm.Data`` placeholder
that the ``Trainer`` overwrites with the next minibatch every step. Passing a
directory of Parquet shards far larger than RAM still gives a model whose
resident footprint is one batch (:func:`parquet_source` reads one row group at
a time).

The unbiased-gradient rescaling is the same as for ``pm.Minibatch``: the
observed log-likelihood must be scaled by ``N / batch_size`` through the existing
:func:`~pymc.variational.minibatch_rv.create_minibatch_rv`. ``N`` is exactly
``len(loader)`` (the loader is sized; ``len`` returns the row count ``N``), so the
model passes ``total_size=len(loader)``. (Folding that scaling into the inference
step, so it drops out of the model body, is the next step in PyMC's VI rework.)

Batches have exactly ``batch_size`` rows, so each pass drops the final
``N mod batch_size`` rows (torch's ``drop_last``). With ``shuffle=True`` that
remainder is re-drawn every epoch, so all rows participate across epochs; with
a source that replays a fixed order, the same rows are dropped every pass (after
a one-time on-disk pre-shuffle that fixed remainder is a random subset).

One difference from ``pm.Minibatch`` is shuffling.
``pm.Minibatch`` draws a fresh uniform index over all N rows every step, so its
minibatches are i.i.d. by construction. A streaming source is only as well
mixed as the order it yields rows in: reading time/row-ordered data through a
bounded buffer is merely a block-shuffle, and the resulting non-representative
minibatches can bias the variational posterior.
Pre-shuffle the data once on disk (or interleave shards) and/or pass
``shuffle=True``.

Examples
--------
.. code-block:: python

    import numpy as np
    import pymc as pm
    from pymc.variational.streaming import DataLoader, Trainer, parquet_source

    # The data was pre-shuffled on disk once (see the module note on shuffling),
    # so the loader streams it sequentially. The full table stays on disk.
    loader = DataLoader(
        parquet_source("shuffled/"),  # an IterableDataset over the shards
        batch_size=4096,
        sample_shape=(4,),  # 3 features + 1 observed column
        total_size="auto",  # infer N from Parquet metadata; N == len(loader)
    )

    with pm.Model() as model:
        b = pm.Normal("b", 0.0, 3.0, shape=4)
        batch = pm.Data("batch", np.zeros((4096, 4)))  # placeholder for one minibatch
        logit = b[0] + b[1] * batch[:, 0] + b[2] * batch[:, 1] + b[3] * batch[:, 2]
        pm.Bernoulli("y", logit_p=logit, observed=batch[:, 3], total_size=len(loader))

    # No callbacks: the Trainer streams each minibatch into "batch" with set_data.
    with model:
        approx = Trainer(method="advi", dataloader=loader, data_name="batch").fit(20_000)
"""

from __future__ import annotations

import glob
import numbers
import os
import warnings

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from pymc.model import modelcontext
from pymc.variational.inference import Inference
from pymc.variational.inference import fit as _fit

__all__ = ["DataLoader", "IterableDataset", "Trainer", "parquet_source", "shuffle_buffer"]


def _is_positive_int(value: object) -> bool:
    """True for a strictly positive integer (incl. numpy integer types), excluding bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and int(value) > 0


class IterableDataset:
    """A re-iterable, out-of-core source of rows, like ``torch.utils.data.IterableDataset``.

    Subclass and implement :meth:`__iter__` to yield ``np.ndarray`` blocks of rows
    (shape ``(rows, *sample_shape)``); :class:`DataLoader` re-batches those blocks
    into fixed-size minibatches. ``__iter__`` must return a fresh iterator each
    call so the dataset can be replayed across epochs.

    Optionally set :attr:`n_rows` (the total row count, if known cheaply, e.g.
    from file metadata) so a :class:`DataLoader` with ``total_size="auto"`` can
    resolve ``N`` without a counting pass.

    A plain zero-arg factory (``Callable[[], Iterator[np.ndarray]]``) or any
    re-iterable is also accepted directly by :class:`DataLoader`; this base class
    is only needed when you want to attach behavior or ``n_rows`` to a custom
    source.
    """

    n_rows: int | None = None

    def __iter__(self) -> Iterator[np.ndarray]:
        raise NotImplementedError("IterableDataset subclasses must implement __iter__")


class DataLoader:
    """Turn an out-of-core dataset into fixed-size minibatches for variational inference.

    Like ``torch.utils.data.DataLoader``, it batches (and optionally
    shuffles) an :class:`IterableDataset` into the minibatch stream that
    :class:`Trainer` feeds to the model. It is iterable and sized (``len(loader)``
    is the dataset size ``N``). With bounded source chunks the full dataset is
    never resident at once.

    Parameters
    ----------
    dataset : IterableDataset | Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]]
        The source of rows. An :class:`IterableDataset`, a re-iterable (including a
        plain ``np.ndarray``), or a zero-arg factory returning a fresh iterator
        (preferred, so the stream can be restarted each epoch). It may yield single
        samples (e.g. the rows of a raw array) or blocks of any size; the loader
        re-batches them, in order, to exactly ``batch_size`` rows. Trailing rows
        that do not fill a final batch are dropped at the end of a pass, like
        ``drop_last=True`` in PyTorch (required here because the model observes a
        fixed-shape placeholder). With ``shuffle=True`` the dropped remainder
        differs per epoch; with a fixed replay order it is the same rows every
        pass.
    batch_size : int
        Leading dimension of every yielded minibatch.
    shuffle : bool, default False
        If ``True``, wrap the source in a bounded :func:`shuffle_buffer` of
        ``buffer_size`` rows. This only approximates i.i.d. batches for an
        already unordered stream; a bounded buffer cannot fix strongly
        time/row-ordered data (pre-shuffle on disk for that; see the module
        docstring).
    buffer_size : int, optional
        Shuffle-buffer size in rows when ``shuffle=True``. Defaults to
        ``50 * batch_size``. Ignored when ``shuffle=False``. A buffer at least
        as large as the dataset holds all of it in memory (a full shuffle).
    seed : int, optional
        Seed for the shuffle buffer (ignored when ``shuffle=False``).
    sample_shape : tuple of int, optional
        Trailing shape of a single observation. ``()`` for scalar observations,
        ``(k,)`` to stream ``k`` columns (e.g. features + the observed column).
        Defaults to ``dataset.shape[1:]`` for a raw ``np.ndarray`` source (its
        rows are the samples, like torch's ``TensorDataset``), else ``()``.
    dtype : str, default "float64"
        Dtype each prepared batch is cast to; match the dtype of the ``pm.Data``
        placeholder the batches are streamed into.
    total_size : int or "auto", optional
        The true dataset size ``N`` (a positive integer), or ``"auto"`` to infer
        it (from the source's ``n_rows`` if available, else a single counting
        pass). Pass it on to the observed distribution as
        ``total_size=len(loader)`` so the minibatch log-likelihood is rescaled by
        ``N / batch_size`` (the same mechanism as ``pm.Minibatch``). Unlike
        ``pm.Minibatch`` it cannot be inferred from a resident array; ``None``
        warns at construction and a non-positive value raises (it would otherwise
        silently disable or invert the rescaling).
    preprocess_fn : callable, optional
        Pure transform applied to each batch before validation (e.g.
        normalization). It must preserve the row count and ``sample_shape``;
        to select columns, do it at the source instead
        (``parquet_source(columns=...)``).
    """

    def __init__(
        self,
        dataset: IterableDataset | Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]],
        *,
        batch_size: int,
        shuffle: bool = False,
        buffer_size: int | None = None,
        seed: int | None = None,
        sample_shape: tuple[int, ...] | None = None,
        dtype: str = "float64",
        total_size: int | str | None = None,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        if not _is_positive_int(batch_size):
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if sample_shape is None:
            # A raw array is rows-of-samples; without this default a 2-D array
            # would be read as blocks of scalars and silently flattened.
            sample_shape = dataset.shape[1:] if isinstance(dataset, np.ndarray) else ()
        sample_shape = tuple(sample_shape)

        raw_factory = _make_factory(dataset)
        source_factory = raw_factory
        if shuffle:
            if buffer_size is None:
                buffer_size = 50 * int(batch_size)
            # shuffle_buffer concatenates yields along the leading axis, so single
            # samples must be promoted to one-row blocks before shuffling.
            source_factory = shuffle_buffer(
                _block_factory(raw_factory, sample_shape),
                buffer_size=buffer_size,
                batch_size=batch_size,
                seed=seed,
            )
        self._source_factory = source_factory

        if isinstance(total_size, str):
            if total_size != "auto":
                raise ValueError(f"total_size string must be 'auto', got {total_size!r}")
            # Count the unshuffled source: the shuffle wrapper drops the trailing
            # partial batch, so counting through it would undercount N.
            total_size = _auto_total_size(raw_factory, dataset, sample_shape)
        elif total_size is None:
            warnings.warn(
                "DataLoader created with total_size=None: the minibatch "
                "log-likelihood will not be rescaled and the posterior will be "
                "biased. Pass total_size=N (the true dataset size) or total_size='auto'.",
                UserWarning,
                stacklevel=2,
            )
        elif not _is_positive_int(total_size):
            # 0 is falsy (the rescaling would be silently skipped) and a negative
            # value flips the sign of the data log-likelihood; raise on both.
            raise ValueError(
                "total_size must be a positive integer (the true dataset size N) so "
                "the minibatch log-likelihood is rescaled by N / batch_size; got "
                f"{total_size!r}."
            )

        # Plain Python ints: create_minibatch_rv rejects np.int64 for total_size.
        self._batch_size = int(batch_size)
        self._sample_shape = sample_shape
        self._dtype = dtype
        self._total_size = None if total_size is None else int(total_size)
        self._preprocess_fn = preprocess_fn

        self._batches_seen = 0
        self._rows_streamed = 0
        self._warned_size = False

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
        """Total rows streamed into the model (grows past ``N`` across epochs)."""
        return self._rows_streamed

    def _rebatched(self) -> Iterator[np.ndarray]:
        """A fresh pass of exactly ``batch_size``-row batches from the source."""
        return _rebatch(self._source_factory(), self._batch_size, self._sample_shape)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield one epoch of validated ``(batch_size, *sample_shape)`` minibatches.

        The same batches the :class:`Trainer` streams into the model's ``pm.Data``
        placeholder (it consumes them through an accounting wrapper, so plain
        iteration leaves the counters untouched). Re-iterate the loader for
        another epoch.
        """
        for batch in self._rebatched():
            yield self._prepare(batch)

    def __len__(self) -> int:
        """The dataset size ``N`` (row count); pass it to the distribution's ``total_size``.

        ``total_size=len(loader)`` is how the model gets the ``N / batch_size``
        rescaling. Note this returns the row count ``N``, not the batch count
        that ``torch.utils.data.DataLoader.__len__`` returns; ``total_size``
        needs ``N``. :attr:`total_size` is the same value.
        """
        if self._total_size is None:
            raise TypeError(
                "len(DataLoader) is the dataset size N, but this loader was built with "
                "total_size=None; construct it with total_size=N or total_size='auto'."
            )
        return self._total_size

    def _stream_batches(self) -> Iterator[np.ndarray]:
        """One epoch of prepared minibatches, with accounting (the Trainer's path).

        Like :meth:`__iter__` but it updates :attr:`batches_seen` /
        :attr:`rows_streamed` and runs the one-shot ``total_size`` sanity check on
        the pass's final batch. The rebatcher is kept one batch ahead so the check
        still fires when a fit stops exactly at the pass boundary; without the
        lookahead the generator would be abandoned right before its epilogue.
        :meth:`__iter__` stays side-effect-free so plain iteration does not mutate
        counters.
        """
        seen_this_pass = 0
        it = self._rebatched()
        batch = next(it, None)
        while batch is not None:
            following = next(it, None)
            prepared = self._prepare(batch)
            self._batches_seen += 1
            self._rows_streamed += int(prepared.shape[0])
            seen_this_pass += int(prepared.shape[0])
            if following is None:
                self._maybe_warn_total_size(seen_this_pass)
            yield prepared
            batch = following

    def _prepare(self, batch: np.ndarray) -> np.ndarray:
        """Preprocess, validate, and return an owned copy of one batch.

        A source may legitimately yield views into a reused array; the copy
        prevents the consumer from aliasing it.
        """
        if self._preprocess_fn is not None:
            batch = self._preprocess_fn(batch)
        self._validate(batch)
        return np.array(batch, dtype=self._dtype)

    def _maybe_warn_total_size(self, seen: int) -> None:
        """Warn once if ``total_size`` is inconsistent with the rows of one full pass.

        ``seen`` is the row count of the pass that just completed (not the
        cumulative :attr:`rows_streamed`, which keeps growing across partial
        streams and earlier fits). A correct ``N`` satisfies
        ``seen <= N < seen + batch_size`` after a full pass (the trailing partial
        batch is dropped), so that window never warns; outside it a 10% slack
        absorbs sources that are only approximately sized.
        """
        if self._warned_size or self._total_size is None:
            return
        self._warned_size = True
        if not seen or seen <= self._total_size < seen + self._batch_size:
            return
        if abs(self._total_size - seen) > 0.1 * seen:
            warnings.warn(
                f"total_size={self._total_size} disagrees with the {seen} rows streamed "
                f"in one full pass; the N/batch_size rescaling, and therefore the "
                f"posterior width, is likely wrong. Pass the true dataset size (or, if "
                f"'auto' resolved it from the source's n_rows, fix that attribute).",
                UserWarning,
                stacklevel=3,
            )

    def _validate(self, batch: np.ndarray) -> None:
        if not isinstance(batch, np.ndarray):
            raise TypeError(f"expected np.ndarray batch, got {type(batch).__name__}")
        if batch.ndim < 1:
            raise ValueError(
                "batch needs a leading batch dimension; got a scalar array with "
                f"shape {batch.shape}."
            )
        if batch.shape[0] != self._batch_size:
            raise ValueError(
                f"batch shape[0] = {batch.shape[0]} does not match batch_size = {self._batch_size}."
            )
        if batch.shape[1:] != self._sample_shape:
            raise ValueError(
                f"batch sample-shape {batch.shape[1:]} does not match declared "
                f"sample_shape={self._sample_shape}"
            )


class Trainer:
    """Drive variational inference over a :class:`DataLoader` without user callbacks.

    Follows the design in PyMC's variational-inference rework and PyTorch
    Lightning: the ``Trainer`` owns the training loop, the
    :class:`DataLoader` owns batching (and ``len(dataloader)`` is the dataset size
    ``N``), and the model owns the math. The model exposes a ``pm.Data`` placeholder;
    the ``Trainer`` streams minibatches into it with ``model.set_data`` once per
    step; no user callbacks are needed.

    Parameters
    ----------
    method : str or Inference, default "advi"
        Variational method, forwarded to :func:`pymc.fit`: a name (``"advi"``,
        ``"fullrank_advi"``, ...) or an :class:`~pymc.variational.inference.Inference`
        instance. ``pm.fit`` applies ``model`` and ``random_seed`` only to a name;
        an instance is already bound to a model, so configure it at construction
        (e.g. ``ADVI(random_seed=...)``).
    dataloader : DataLoader
        The minibatch source. ``len(dataloader)`` is ``N``; the model should pass
        it to the observed distribution's ``total_size``.
    model : pymc.Model, optional
        Defaults to the model on the context stack.
    data_name : str, default "batch"
        Name of the ``pm.Data`` placeholder minibatches are streamed into. Must
        match the name used for ``pm.Data(name, ...)`` in the model.
    **fit_kwargs
        Default keyword arguments forwarded to :func:`pymc.fit` (e.g.
        ``obj_optimizer``); per-call kwargs to :meth:`fit` override them.

    Notes
    -----
    The per-step ``set_data`` currently lives in the ``Trainer``. Once the VI
    rework's ``Inference.step(batch)`` lands it moves there, at which point the
    ``total_size`` rescaling can be derived from ``len(dataloader)`` and dropped
    from the model body entirely.

    Examples
    --------
    .. code-block:: python

        loader = DataLoader(
            parquet_source("shuffled/"), batch_size=4096, sample_shape=(4,), total_size="auto"
        )
        with pm.Model() as model:
            b = pm.Normal("b", 0.0, 3.0, shape=4)
            batch = pm.Data("batch", np.zeros((4096, 4)))  # placeholder
            logit = b[0] + b[1] * batch[:, 0] + b[2] * batch[:, 1] + b[3] * batch[:, 2]
            pm.Bernoulli("y", logit_p=logit, observed=batch[:, 3], total_size=len(loader))
            approx = Trainer(method="advi", dataloader=loader, data_name="batch").fit(20_000)
    """

    def __init__(
        self,
        *,
        method: str | Inference = "advi",
        dataloader: DataLoader,
        model=None,
        data_name: str = "batch",
        **fit_kwargs,
    ):
        self.method = method
        self.dataloader = dataloader
        self.model = model
        self.data_name = data_name
        self._fit_kwargs = fit_kwargs

    def fit(self, n: int = 10_000, **kwargs):
        """Fit for ``n`` steps, streaming minibatches into the model's placeholder.

        Exactly ``n`` minibatches are fed to the model: the first seeds the
        placeholder before step 0, and the advance after the final step is skipped.
        The accounting stream reads one batch ahead so the pass-size check can fire
        at a pass boundary, so a re-readable source (the only kind the loader
        accepts) may be read one batch past the ``n`` the model uses. Keyword
        arguments are forwarded to :func:`pymc.fit` on top of the constructor's
        ``fit_kwargs`` (per-call wins); ``progressbar`` defaults to ``False``
        unless either sets it.

        Returns
        -------
        :class:`Approximation`
            The fitted approximation, as returned by :func:`pymc.fit`.
        """
        if not _is_positive_int(n):
            raise ValueError(f"n must be a positive integer (the number of fit steps), got {n!r}")
        loader = self.dataloader
        if not isinstance(loader, DataLoader):
            raise TypeError(
                f"Trainer needs a DataLoader for `dataloader`, got {type(loader).__name__}."
            )
        model = modelcontext(self.model)
        if self.data_name not in model:
            # Checked before the stream starts so no batch is consumed (and no
            # counter advances) on a typo.
            raise KeyError(
                f"data_name {self.data_name!r} is not a variable in the model; it "
                f"must name the pm.Data placeholder the minibatches are streamed into."
            )

        def _stream() -> Iterator[np.ndarray]:
            while True:
                empty = True
                for batch in loader._stream_batches():
                    empty = False
                    yield batch
                if empty:
                    raise RuntimeError("dataloader yielded no batches")

        batches = _stream()
        # Seed the placeholder before step 0: pm.fit runs callbacks after each step,
        # so without this the first step would train on the placeholder's contents.
        model.set_data(self.data_name, next(batches))

        steps_done = 0

        def _advance(*_):
            # pm.fit fires callbacks after every step including the last; skip the
            # advance on this fit's final step so exactly n batches are consumed.
            # Only that one call is skipped (not every call past n): Inference.refine
            # replays the saved callbacks and must keep streaming fresh batches.
            nonlocal steps_done
            steps_done += 1
            if steps_done != n:
                model.set_data(self.data_name, next(batches))

        merged = {**self._fit_kwargs, **kwargs}
        merged.setdefault("progressbar", False)
        # User callbacks (e.g. convergence trackers) are appended after the
        # internal advance instead of colliding with it on the keyword.
        user_callbacks = merged.pop("callbacks", None) or []
        return _fit(
            n,
            method=self.method,
            model=model,
            callbacks=[_advance, *user_callbacks],
            **merged,
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
    do not fill a final batch are carried over into the next buffer (never
    dropped) until the source is exhausted, at which point a single trailing
    partial batch (< ``batch_size`` rows) is dropped. This approximates i.i.d.
    minibatches from an unordered or pre-shuffled stream.

    :class:`DataLoader` calls this for you when ``shuffle=True``; use it directly
    when you want explicit control over ``buffer_size`` independently of the
    loader.

    It does not by itself fix a strongly time/row-ordered stream (a bounded
    buffer only block-shuffles such data); pre-shuffle on disk, or interleave
    shards into ``chunk_source``, for that. ``buffer_size`` is a lower bound:
    each fill accumulates at least ``max(buffer_size, batch_size)`` rows before
    shuffling (so a ``buffer_size`` smaller than ``batch_size`` still yields full
    batches; the final fill stops at whatever the source has left), and the chunk
    that crosses the threshold is kept whole, so the buffer holds fewer than
    ``max(buffer_size, batch_size)`` plus one chunk's rows. Concatenating a fill
    into one shuffleable array transiently allocates a second copy of those
    rows, so peak allocation is about twice that bound.

    Each epoch (each call of the returned factory) draws a fresh permutation from
    a sub-stream of ``seed``, so the shuffle order differs across epochs while
    staying reproducible for a given ``seed``.
    """
    if not _is_positive_int(batch_size):
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if not _is_positive_int(buffer_size):
        raise ValueError(f"buffer_size must be a positive integer, got {buffer_size!r}")
    seed_seq = np.random.SeedSequence(seed)

    def factory() -> Iterator[np.ndarray]:
        # A fresh sub-stream per epoch: re-iterating reshuffles instead of
        # replaying one fixed permutation, yet stays reproducible per seed.
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        # A factory may return a re-iterable (a list of chunks, ...); normalize so
        # each buffer fill continues one stream instead of restarting it forever.
        it = iter(chunk_source())
        carry: np.ndarray | None = None
        exhausted = False
        # Accumulate at least one batch even when buffer_size < batch_size,
        # otherwise the guard below would silently discard the whole stream.
        target = max(buffer_size, batch_size)
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
                if have >= target:
                    break
            else:
                exhausted = True
            if have < batch_size:
                # Only reachable once the source is exhausted: drop the final
                # partial batch.
                return
            buf = np.concatenate(bufs, axis=0)
            rng.shuffle(buf)
            n_full = buf.shape[0] // batch_size
            for i in range(n_full):
                yield buf[i * batch_size : (i + 1) * batch_size]
            rem = buf.shape[0] - n_full * batch_size
            carry = buf[n_full * batch_size :].copy() if rem else None

    # Forward a known row count so total_size="auto" stays metadata-cheap
    # through the shuffle wrapper.
    source_n_rows = getattr(chunk_source, "n_rows", None)
    if source_n_rows is not None:
        factory.n_rows = source_n_rows  # type: ignore[attr-defined]

    return factory


def _promote_to_block(a: np.ndarray, sample_shape: tuple[int, ...]) -> np.ndarray:
    """Return ``a`` as a ``(rows, *sample_shape)`` block; a single sample becomes one row."""
    if a.shape == sample_shape:
        return a[None, ...]
    if a.ndim != len(sample_shape) + 1 or a.shape[1:] != sample_shape:
        raise ValueError(
            f"source yielded shape {a.shape}; expected one sample of shape "
            f"{sample_shape} or a (rows, *sample_shape) block; if the source is "
            f"right, declare its trailing shape with DataLoader(sample_shape=...)"
        )
    return a


def _block_factory(
    factory: Callable[[], Iterator[np.ndarray]],
    sample_shape: tuple[int, ...],
) -> Callable[[], Iterator[np.ndarray]]:
    """Wrap ``factory`` so every yield is a block, promoting single samples.

    :func:`shuffle_buffer` counts and concatenates yields along the leading axis,
    so single-sample yields (e.g. the rows of a raw array) must be promoted to
    one-row blocks before shuffling. A known ``.n_rows`` is forwarded.
    """

    def f() -> Iterator[np.ndarray]:
        for arr in factory():
            yield _promote_to_block(np.asarray(arr), sample_shape)

    n_rows = getattr(factory, "n_rows", None)
    if n_rows is not None:
        f.n_rows = n_rows  # type: ignore[attr-defined]
    return f


def _rebatch(
    blocks: Iterable[np.ndarray],
    batch_size: int,
    sample_shape: tuple[int, ...],
) -> Iterator[np.ndarray]:
    """Slice a stream of samples/blocks into exact ``batch_size``-row batches, in order.

    Accepts single samples (shape ``sample_shape``, e.g. the rows of a raw array)
    and blocks of any size (shape ``(rows, *sample_shape)``), carrying remainders
    across blocks so no row is lost mid-stream. Trailing rows that do not fill a
    final batch are dropped when the stream ends (``drop_last=True`` behavior; the
    model observes a fixed-shape placeholder, so a partial batch cannot be fed).
    Sources that already yield exact ``batch_size`` blocks (e.g.
    :func:`shuffle_buffer`) pass through without copying.
    """
    buf: list[np.ndarray] = []
    have = 0
    for arr in blocks:
        a = _promote_to_block(np.asarray(arr), sample_shape)
        buf.append(a)
        have += a.shape[0]
        if have < batch_size:
            continue
        merged = np.concatenate(buf, axis=0) if len(buf) > 1 else buf[0]
        n_full = merged.shape[0] // batch_size
        for i in range(n_full):
            yield merged[i * batch_size : (i + 1) * batch_size]
        rem = merged.shape[0] - n_full * batch_size
        buf = [merged[n_full * batch_size :].copy()] if rem else []
        have = rem


def _make_factory(
    source: Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]],
) -> Callable[[], Iterator[np.ndarray]]:
    """Coerce ``source`` into a zero-arg callable returning a fresh iterator.

    A callable that is not itself an iterator is treated as the factory; a bare
    iterator is wrapped (and refuses a second epoch); any other iterable (incl. an
    :class:`IterableDataset`) is re-``iter``-ed each epoch. A known ``.n_rows`` is
    forwarded onto the returned factory so ``total_size="auto"`` stays cheap.
    """
    if callable(source) and not isinstance(source, Iterator):
        # A factory may return any iterable (a list of batches, a generator, ...);
        # normalize so the loader always pulls from a true iterator.
        def _factory() -> Iterator[np.ndarray]:
            return iter(source())  # type: ignore[operator]

    elif isinstance(source, Iterator):
        consumed = {"done": False}

        def _factory() -> Iterator[np.ndarray]:
            if consumed["done"]:
                raise RuntimeError(
                    "source is a bare iterator and was already consumed; the loader "
                    "restarts the stream each epoch, so pass a zero-arg factory or a "
                    "re-iterable instead"
                )
            consumed["done"] = True
            return source

    else:

        def _factory() -> Iterator[np.ndarray]:
            return iter(source)

    n_rows = getattr(source, "n_rows", None)
    if n_rows is not None:
        _factory.n_rows = n_rows  # type: ignore[attr-defined]
    return _factory


def _auto_total_size(
    factory: Callable[[], Iterator[np.ndarray]],
    source: object,
    sample_shape: tuple[int, ...] = (),
) -> int:
    """Resolve ``total_size="auto"``: a source ``.n_rows`` (cheap) else a counting pass.

    Fast path: if ``source`` advertises ``.n_rows`` (e.g. :func:`parquet_source`, which
    reads it from Parquet metadata without scanning the data) use it directly. Otherwise
    do a single counting pass over a finite, re-readable source. A bare one-shot iterator
    cannot be auto-counted (counting consumes it) and an infinite stream would make the
    pass hang; both must pass ``total_size`` explicitly.
    """
    n = getattr(source, "n_rows", None)
    if n is None:
        n = getattr(factory, "n_rows", None)
    if n is not None:
        if not _is_positive_int(n):
            raise ValueError(f"source.n_rows must be a positive integer, got {n!r}")
        return int(n)
    if isinstance(source, Iterator):
        raise ValueError(
            "total_size='auto' needs a re-readable source (a zero-arg factory or an "
            "iterable), not a one-shot iterator; pass total_size=N explicitly instead."
        )
    warnings.warn(
        "total_size='auto' is doing a full counting pass over the source; for a cheap "
        "path use a source exposing .n_rows (e.g. parquet_source, from Parquet metadata).",
        UserWarning,
        stacklevel=3,
    )
    first_iter = factory()
    count = 0
    for chunk in first_iter:
        a = np.asarray(chunk)
        # A yield of shape exactly `sample_shape` is one sample, not a block.
        count += 1 if a.shape == sample_shape else int(a.shape[0])
    if count <= 0:
        raise ValueError("total_size='auto' counted 0 rows (empty or non-re-readable source).")
    # A genuine factory yields a fresh, non-empty stream each call; one that
    # returns the same exhausted iterator (or a new generator over consumed
    # state) would leave the loader with nothing to stream. The probe costs one
    # chunk, which the counting pass has already dwarfed.
    second_iter = factory()
    if second_iter is first_iter or next(second_iter, None) is None:
        raise ValueError(
            "total_size='auto' counted rows but the factory's next stream was empty "
            "(it returns the same one-shot iterator, or closes over an already-"
            "consumed one); pass a factory that creates a fresh iterator each call, "
            "or total_size=N explicitly."
        )
    return count


class _ParquetDataset(IterableDataset):
    """An :class:`IterableDataset` over a directory of Parquet shards.

    Yields one ``(rows, n_columns)`` array per row group (so peak read memory is
    one row group, not one file), in the fixed column order chosen at
    construction, and exposes :attr:`n_rows` read from Parquet metadata (no data
    scan).
    """

    def __init__(self, paths: list[str], columns: list[str], n_rows: int):
        self._paths = paths
        self._columns = columns
        self.n_rows = n_rows

    def __iter__(self) -> Iterator[np.ndarray]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for path in self._paths:
            file = pq.ParquetFile(path)
            schema = file.schema_arrow
            missing = [c for c in self._columns if c not in schema.names]
            if missing:
                # read_row_group(columns=...) silently drops unknown names, so a
                # malformed shard must be named here, not surface as a bare
                # KeyError with no path.
                raise ValueError(f"columns {missing} not found in {path!r}")
            non_numeric = [
                c
                for c in self._columns
                if not (
                    pa.types.is_integer(schema.field(c).type)
                    or pa.types.is_floating(schema.field(c).type)
                    or pa.types.is_boolean(schema.field(c).type)
                )
            ]
            if non_numeric:
                # parquet_source validates types against the first shard only; a
                # later shard whose column turned non-numeric would otherwise
                # become an object array and fail at the batch cast with no path.
                raise ValueError(
                    f"columns {non_numeric} in {path!r} are not numeric and cannot be "
                    f"streamed into a float batch; select numeric columns with columns=."
                )
            for i in range(file.metadata.num_row_groups):
                table = file.read_row_group(i, columns=self._columns)
                # Stack by the frozen column names, not the file's own order, so
                # a shard with a permuted schema cannot silently swap features.
                yield np.column_stack([table.column(c).to_numpy() for c in self._columns])


def parquet_source(
    directory: str,
    *,
    columns: list[str] | None = None,
    pattern: str = "*.parquet",
) -> _ParquetDataset:
    """An :class:`IterableDataset` over a directory of Parquet files.

    Yields one ``(rows, n_columns)`` array per row group (one or more per file),
    so peak read memory is one row group, not one file. The column order is
    frozen at construction — ``columns`` if given, else the first file's schema
    order — and every shard is read in that order, so a shard with a permuted
    schema cannot silently reorder features mid-stream. Carries an ``n_rows``
    attribute read from Parquet metadata (no data scan) so that
    ``DataLoader(parquet_source(dir), ..., total_size="auto")`` resolves the
    dataset size for free. Pass ``shuffle=True`` to the :class:`DataLoader` (or
    wrap in :func:`shuffle_buffer`) to get shuffled batches.
    """
    # pyarrow is an optional dependency, so it is imported on use.
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise ValueError(f"no Parquet files match {os.path.join(directory, pattern)!r}")
    schema = pq.read_schema(paths[0])
    if columns is None:
        columns = list(schema.names)
    else:
        missing = sorted(set(columns) - set(schema.names))
        if missing:
            raise ValueError(
                f"columns {missing} not found in {paths[0]!r}; available: {sorted(schema.names)}"
            )
    non_numeric = [
        c
        for c in columns
        if not (
            pa.types.is_integer(schema.field(c).type)
            or pa.types.is_floating(schema.field(c).type)
            or pa.types.is_boolean(schema.field(c).type)
        )
    ]
    if non_numeric:
        # A string/dictionary column would turn whole chunks object-dtype and only
        # fail later at the batch cast, without naming the column.
        raise ValueError(
            f"columns {non_numeric} in {paths[0]!r} are not numeric and cannot be "
            f"streamed into a float batch; select numeric columns with columns=."
        )
    n_rows = sum(pq.read_metadata(p).num_rows for p in paths)
    return _ParquetDataset(paths, columns, n_rows)
