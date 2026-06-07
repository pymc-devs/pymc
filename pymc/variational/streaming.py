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
peak memory is therefore O(N) in the dataset size.  This module instead feeds
minibatches from an out-of-core source into a small fixed-size
``pytensor.shared`` buffer, so peak memory is O(buffer) -- the batch buffer plus,
if used, the shuffle buffer -- and independent of N.

The API deliberately mirrors PyTorch's ``torch.utils.data`` so the mental model
transfers directly:

* :class:`IterableDataset` -- a re-iterable, out-of-core source of rows
  (e.g. :func:`parquet_source` over a directory of shards). It never loads the
  whole dataset; it yields it a chunk at a time.
* :class:`DataLoader` -- turns a dataset into fixed-size (optionally shuffled)
  minibatches; it is iterable (the minibatch stream) and sized (``len(loader)``
  is the dataset size ``N``), exactly like a PyTorch one.
* :class:`Trainer` -- drives variational inference (ADVI, ...) over a
  ``DataLoader`` with **no user-facing callbacks**;
  ``Trainer(method=..., dataloader=...).fit(n)`` streams each minibatch into the
  model's ``pm.Data`` placeholder with ``set_data``.

**The full data never enters RAM.** The model graph observes only a
``(batch_size, *sample_shape)`` ``pm.Data`` *placeholder* that the ``Trainer``
overwrites with the next minibatch every step. Passing a directory of 122 GB of
Parquet shards still gives a model whose resident footprint is one batch.

The unbiased-gradient rescaling is the *same* as for ``pm.Minibatch``: the
observed log-likelihood must be scaled by ``N / batch_size`` through the existing
:func:`~pymc.variational.minibatch_rv.create_minibatch_rv`. ``N`` is exactly
``len(loader)`` -- a :class:`DataLoader` is sized like a PyTorch one -- so the
model passes ``total_size=len(loader)``. (Folding that scaling into the inference
step, so it drops out of the model body, is the next step in PyMC's VI rework.)

The one extra obligation relative to ``pm.Minibatch`` is **shuffling**.
``pm.Minibatch`` draws a fresh uniform index over all N rows every step, so its
minibatches are i.i.d. by construction.  A streaming source is only as well
mixed as the order it yields rows in: reading time/row-ordered data through a
*bounded* buffer is merely a block-shuffle and biases the variational posterior.
Pre-shuffle the data once on disk (or interleave shards) and/or pass
``shuffle=True``.

Example
-------
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
        batch = pm.Data("batch", np.zeros((4096, 4)))  # placeholder -- the ONLY data in RAM
        logit = b[0] + b[1] * batch[:, 0] + b[2] * batch[:, 1] + b[3] * batch[:, 2]
        pm.Bernoulli("y", logit_p=logit, observed=batch[:, 3], total_size=len(loader))

    # No callbacks: the Trainer streams each minibatch into "batch" with set_data.
    with model:
        approx = Trainer(method="advi", dataloader=loader, data_name="batch").fit(20_000)
"""

from __future__ import annotations

import numbers
import warnings

from collections.abc import Callable, Iterable, Iterator

import numpy as np
import pytensor
import pytensor.tensor as pt


def _is_positive_int(value: object) -> bool:
    """True for a strictly positive integer (incl. numpy integer types), excluding bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and value > 0


class IterableDataset:
    """A re-iterable, out-of-core source of rows -- the analogue of ``torch.utils.data.IterableDataset``.

    Subclass and implement :meth:`__iter__` to yield ``np.ndarray`` blocks of rows
    (shape ``(rows, *sample_shape)``); :class:`DataLoader` re-batches those blocks
    into fixed-size minibatches. ``__iter__`` must return a **fresh** iterator each
    call so the dataset can be replayed across epochs.

    Optionally set :attr:`n_rows` (the total row count, if known cheaply -- e.g.
    from file metadata) so a :class:`DataLoader` with ``total_size="auto"`` can
    resolve ``N`` without a counting pass.

    A plain zero-arg factory (``Callable[[], Iterator[np.ndarray]]``) or any
    re-iterable is also accepted directly by :class:`DataLoader`; this base class
    is only needed when you want to attach behaviour or ``n_rows`` to a custom
    source.
    """

    n_rows: int | None = None

    def __iter__(self) -> Iterator[np.ndarray]:
        raise NotImplementedError("IterableDataset subclasses must implement __iter__")


class DataLoader:
    """Turn an out-of-core dataset into fixed-size minibatches for variational inference.

    The analogue of ``torch.utils.data.DataLoader``: it batches (and optionally
    shuffles) an :class:`IterableDataset` into the minibatch stream that
    :class:`Trainer` feeds to the model. It is iterable and sized (``len(loader)``
    is the dataset size ``N``). The full dataset never enters memory -- only one
    ``(batch_size, *sample_shape)`` batch does.

    Parameters
    ----------
    dataset : IterableDataset | Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]]
        The source of rows. An :class:`IterableDataset`, a re-iterable, or a
        zero-arg *factory* returning a fresh iterator (preferred, so the stream can
        be restarted when ``cycle=True``). It may yield individual rows or
        multi-row blocks; the loader re-batches to exactly ``batch_size`` rows.
    batch_size : int
        Leading dimension of every yielded minibatch (and of the buffer).
    shuffle : bool, default False
        If ``True``, wrap the source in a bounded :func:`shuffle_buffer` of
        ``buffer_size`` rows. This only approximates i.i.d. batches for an
        *already unordered* stream; a bounded buffer cannot fix strongly
        time/row-ordered data (pre-shuffle on disk for that -- see the module
        docstring).
    buffer_size : int, optional
        Shuffle-buffer size in rows when ``shuffle=True``. Defaults to
        ``50 * batch_size``. Ignored when ``shuffle=False``.
    seed : int, optional
        Seed for the shuffle buffer (ignored when ``shuffle=False``).
    sample_shape : tuple of int, default ()
        Trailing shape of a single observation. ``()`` for scalar observations,
        ``(k,)`` to stream ``k`` columns (e.g. features + the observed column).
    dtype : str, default "float64"
        Dtype of the shared buffer. If it differs from ``pytensor.config.floatX``
        the model will insert a per-step cast on the observed tensor.
    total_size : int or "auto", optional
        The true dataset size ``N`` (a positive integer), or ``"auto"`` to infer
        it (from the source's ``n_rows`` if available, else a single counting
        pass). Pass it on to the observed distribution as
        ``total_size=loader.total_size`` so the minibatch log-likelihood is
        rescaled by ``N / batch_size`` (the same mechanism as ``pm.Minibatch``).
        Unlike ``pm.Minibatch`` it cannot be inferred from a resident array;
        ``None`` warns at construction and a non-positive value raises (it would
        otherwise silently disable or invert the rescaling).
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
        dataset: IterableDataset | Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]],
        *,
        batch_size: int,
        shuffle: bool = False,
        buffer_size: int | None = None,
        seed: int | None = None,
        sample_shape: tuple[int, ...] = (),
        dtype: str = "float64",
        total_size: int | str | None = None,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        cycle: bool = True,
        name: str = "dataloader_buffer",
    ):
        if not _is_positive_int(batch_size):
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

        source_factory = _make_factory(dataset)
        if shuffle:
            if buffer_size is None:
                buffer_size = 50 * int(batch_size)
            # shuffle_buffer forwards a known .n_rows, so total_size="auto" still
            # resolves cheaply through the shuffle wrapper.
            source_factory = shuffle_buffer(
                source_factory, buffer_size=buffer_size, batch_size=batch_size, seed=seed
            )
        self._source_factory = source_factory

        if isinstance(total_size, str):
            if total_size != "auto":
                raise ValueError(f"total_size string must be 'auto', got {total_size!r}")
            # Resolve N automatically: a source-provided .n_rows (cheap, e.g. from
            # parquet_source's metadata) else one counting pass over a finite,
            # re-readable source. One-shot / infinite sources cannot be auto-counted.
            total_size = _auto_total_size(self._source_factory, dataset)
        elif total_size is None:
            warnings.warn(
                "DataLoader created with total_size=None: the minibatch "
                "log-likelihood will NOT be rescaled and the posterior will be "
                "biased. Pass total_size=N (the true dataset size) or total_size='auto'.",
                UserWarning,
                stacklevel=2,
            )
        elif not _is_positive_int(total_size):
            # A non-positive total_size is silently dangerous: 0 is falsy, so the
            # model never wraps the observed RV and the N/batch_size rescaling is
            # skipped (posterior collapses toward the prior); a negative value
            # yields a negative scaling coefficient that flips the data
            # log-likelihood's sign (VI then maximizes mis-fit). Reject it loudly.
            raise ValueError(
                "total_size must be a positive integer (the true dataset size N) so "
                "the minibatch log-likelihood is rescaled by N / batch_size; got "
                f"{total_size!r}."
            )

        self._source_iter: Iterator[np.ndarray] = self._source_factory()
        # Normalize integer-like sizes to plain Python ints. ``_is_positive_int``
        # accepts numpy integers (via ``numbers.Integral``), but the downstream
        # ``create_minibatch_rv`` type-checks ``isinstance(total_size, int)`` and
        # would raise on a stored ``np.int64`` ("Invalid type for total_size").
        self._batch_size = int(batch_size)
        self._sample_shape = tuple(sample_shape)
        self._dtype = dtype
        self._total_size = None if total_size is None else int(total_size)
        self._preprocess_fn = preprocess_fn
        self._cycle = cycle

        self._batches_seen = 0
        self._rows_streamed = 0
        self._warned_size = False  # the sanity check below fires at most once

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

    # ----- iteration: the minibatch stream ----------------------------------

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield one epoch of validated ``(batch_size, *sample_shape)`` minibatches.

        This is the stream :class:`Trainer` pushes into the model's ``pm.Data``
        placeholder via ``set_data``. Re-iterate the loader for another epoch.
        """
        for batch in self._source_factory():
            yield self._prepare(batch)

    def __len__(self) -> int:
        """The dataset size ``N`` -- pass to the observed distribution's ``total_size``.

        Sized like a PyTorch ``DataLoader``; ``total_size=len(loader)`` is how the
        model gets the ``N / batch_size`` rescaling.
        """
        if self._total_size is None:
            raise TypeError(
                "len(DataLoader) is the dataset size N, but this loader was built with "
                "total_size=None; construct it with total_size=N or total_size='auto'."
            )
        return self._total_size

    # ----- shared-buffer path (advanced; the Trainer uses pm.Data instead) ---

    def as_tensor(self) -> pt.TensorVariable:
        """A ``pytensor.shared`` buffer the model can observe directly.

        An alternative to the recommended ``pm.Data`` + :class:`Trainer` path: drive
        ``pm.fit`` yourself and call :meth:`advance` (e.g. from a callback) to refill
        this buffer each step.
        """
        return self._shared

    def advance(self) -> None:
        """Pull the next batch from the source into the :meth:`as_tensor` buffer."""
        arr = self._prepare(self._next_batch())
        self._shared.set_value(arr, borrow=True)
        self._batches_seen += 1
        self._rows_streamed += int(arr.shape[0])

    # ----- internals ---------------------------------------------------------

    def _prepare(self, batch: np.ndarray) -> np.ndarray:
        """Preprocess, validate, and return an *owned* copy of one batch.

        The owned copy matters: a source may legitimately yield *views* into a
        reused array, so neither the buffer nor an iteration consumer may alias it.
        """
        if self._preprocess_fn is not None:
            batch = self._preprocess_fn(batch)
        self._validate(batch)
        return np.array(batch, dtype=self._dtype)  # np.array(copy default) owns it

    def _next_batch(self) -> np.ndarray:
        try:
            return next(self._source_iter)
        except StopIteration:
            if not self._cycle:
                raise
            # First exhaustion == one full pass: rows_streamed now equals the real
            # row count, so we can sanity-check the user's total_size for free.
            self._maybe_warn_total_size()
            self._source_iter = self._source_factory()
            return next(self._source_iter)

    def _maybe_warn_total_size(self) -> None:
        """Warn once if total_size grossly disagrees with the rows seen in one pass."""
        if self._warned_size or self._total_size is None:
            return
        self._warned_size = True
        seen = self._rows_streamed
        if seen and abs(self._total_size - seen) > 0.1 * seen:
            warnings.warn(
                f"total_size={self._total_size} disagrees with the {seen} rows streamed "
                f"in one full pass; the N/batch_size rescaling -- and therefore the "
                f"posterior width -- is likely wrong. Pass the true dataset size, or "
                f"total_size='auto'.",
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
                f"batch shape[0] = {batch.shape[0]} does not match batch_size = "
                f"{self._batch_size}; partial batches are not allowed (drop them in "
                "the source, e.g. via shuffle=True / shuffle_buffer)."
            )
        if batch.shape[1:] != self._sample_shape:
            raise ValueError(
                f"batch sample-shape {batch.shape[1:]} does not match declared "
                f"sample_shape={self._sample_shape}"
            )


class Trainer:
    """Drive variational inference over a :class:`DataLoader` -- without callbacks.

    Follows the design in PyMC's variational-inference rework (Grabowski, *VI
    Overview*) and PyTorch Lightning: the ``Trainer`` owns the training loop, the
    :class:`DataLoader` owns batching (and ``len(dataloader)`` is the dataset size
    ``N``), and the model owns the math. The model exposes a ``pm.Data`` placeholder;
    the ``Trainer`` streams minibatches into it with ``model.set_data`` once per
    step, so the user wires up **no** callbacks (the design Rob asked for).

    .. code-block:: python

        import numpy as np

        loader = DataLoader(
            parquet_source("shuffled/"), batch_size=4096, sample_shape=(4,), total_size="auto"
        )
        with pm.Model() as model:
            b = pm.Normal("b", 0.0, 3.0, shape=4)
            batch = pm.Data("batch", np.zeros((4096, 4)))  # placeholder
            logit = b[0] + b[1] * batch[:, 0] + b[2] * batch[:, 1] + b[3] * batch[:, 2]
            pm.Bernoulli("y", logit_p=logit, observed=batch[:, 3], total_size=len(loader))
            approx = Trainer(method="advi", dataloader=loader, data_name="batch").fit(20_000)

    Parameters
    ----------
    method : str, default "advi"
        Variational method, forwarded to :func:`pymc.fit` (``"advi"``,
        ``"fullrank_advi"``, ...). When PyMC's VI rework lands this will also accept
        an inference *instance* (e.g. ``ADVI()``); a string drives today's ``pm.fit``.
    dataloader : DataLoader
        The minibatch source. ``len(dataloader)`` is ``N`` -- the model should pass
        it to the observed distribution's ``total_size``.
    model : pymc.Model, optional
        Defaults to the model on the context stack.
    data_name : str, default "data"
        Name of the ``pm.Data`` placeholder minibatches are streamed into.
    **fit_kwargs
        Default keyword arguments forwarded to :func:`pymc.fit` (e.g.
        ``obj_optimizer``); per-call kwargs to :meth:`fit` override them.

    Notes
    -----
    This is the *starting point* Rob suggested: the per-step ``set_data`` logic
    lives in the ``Trainer``. The longer-term plan is to fold it into the inference
    object's ``step(batch)`` once the VI rework lands, at which point the
    ``total_size`` rescaling can be derived from ``len(dataloader)`` and dropped
    from the model body entirely.
    """

    def __init__(
        self,
        *,
        method: str = "advi",
        dataloader: DataLoader,
        model=None,
        data_name: str = "data",
        **fit_kwargs,
    ):
        self.method = method
        self.dataloader = dataloader
        self.model = model
        self.data_name = data_name
        self._fit_kwargs = fit_kwargs

    def fit(
        self,
        n: int = 10_000,
        *,
        random_seed: int | None = None,
        progressbar: bool = False,
        **kwargs,
    ):
        """Fit for ``n`` steps, streaming minibatches into the model's placeholder.

        Returns whatever :func:`pymc.fit` returns for the chosen method.
        """
        from pymc.model import modelcontext
        from pymc.variational.inference import fit as _fit

        loader = self.dataloader
        if not isinstance(loader, DataLoader):
            raise TypeError(
                f"Trainer needs a DataLoader for `dataloader`, got {type(loader).__name__}."
            )
        model = modelcontext(self.model)

        # An endless minibatch stream: re-iterate the loader across epochs.
        def _stream() -> Iterator[np.ndarray]:
            while True:
                empty = True
                for batch in loader:
                    empty = False
                    yield batch
                if empty:
                    raise RuntimeError("dataloader yielded no batches")

        batches = _stream()
        # Seed the placeholder before step 0: pm.fit runs callbacks AFTER each step,
        # so without this the first step would train on the placeholder's contents.
        model.set_data(self.data_name, next(batches))

        def _advance(*_):
            model.set_data(self.data_name, next(batches))

        merged = {**self._fit_kwargs, **kwargs}
        return _fit(
            n,
            method=self.method,
            model=model,
            random_seed=random_seed,
            progressbar=progressbar,
            callbacks=[_advance],
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
    do not fill a final batch are **carried over** into the next buffer (never
    dropped) until the source is exhausted, at which point a single trailing
    partial batch (< ``batch_size`` rows) is dropped. This approximates i.i.d.
    minibatches from an *unordered* or pre-shuffled stream.

    :class:`DataLoader` calls this for you when ``shuffle=True``; use it directly
    when you want explicit control over ``buffer_size`` independently of the
    loader.

    It does **not** by itself fix a strongly time/row-ordered stream (a bounded
    buffer only block-shuffles such data) -- pre-shuffle on disk, or interleave
    shards into ``chunk_source``, for that. ``buffer_size`` is a *lower* bound: the
    buffer always accumulates at least ``max(buffer_size, batch_size)`` rows before
    emitting (so a ``buffer_size`` smaller than ``batch_size`` still yields full
    batches instead of silently dropping the stream), and a single chunk larger
    than that is taken whole, so peak buffer memory is
    ``max(buffer_size, batch_size, largest_chunk_rows)``.

    Each epoch (each call of the returned factory) draws a fresh permutation from
    a sub-stream of ``seed``, so the shuffle order differs across epochs -- a
    seeded buffer must not replay one fixed order forever -- while staying
    reproducible for a given ``seed``.
    """
    if not _is_positive_int(batch_size):
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if not _is_positive_int(buffer_size):
        raise ValueError(f"buffer_size must be a positive integer, got {buffer_size!r}")
    seed_seq = np.random.SeedSequence(seed)

    def factory() -> Iterator[np.ndarray]:
        # Spawn a fresh sub-stream per epoch so re-iterating (cycle=True) reshuffles
        # rather than replaying one fixed permutation forever; still reproducible
        # across runs for a given seed.
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        it = chunk_source()
        carry: np.ndarray | None = None  # leftover (< batch_size) from last fill
        exhausted = False
        # Accumulate at least one full batch's worth even when buffer_size <
        # batch_size: otherwise the inner loop would break early with fewer than
        # batch_size rows and the `have < batch_size` guard below would silently
        # discard the entire stream.
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
                exhausted = True  # for-loop ran to completion: source is done
            if have < batch_size:
                # Only reachable once the source is exhausted: drop the final
                # sub-batch remainder (it cannot form a full batch).
                return
            buf = np.concatenate(bufs, axis=0)  # always a fresh, owned copy
            rng.shuffle(buf)
            n_full = buf.shape[0] // batch_size
            for i in range(n_full):
                yield buf[i * batch_size : (i + 1) * batch_size]
            rem = buf.shape[0] - n_full * batch_size
            carry = buf[n_full * batch_size :].copy() if rem else None

    # Forward a known row count (e.g. parquet_source's .n_rows from Parquet
    # metadata) to the wrapped factory, so
    # ``DataLoader(source, shuffle=True, total_size="auto")`` resolves N for free
    # instead of doing a counting pass. The only discrepancy is the single dropped
    # trailing partial batch (< batch_size rows), well within the auto-size
    # sanity tolerance.
    source_n_rows = getattr(chunk_source, "n_rows", None)
    if source_n_rows is not None:
        factory.n_rows = source_n_rows  # type: ignore[attr-defined]

    return factory


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
        # A factory may return any iterable (a list of batches, a generator, ...),
        # not only an iterator; normalize so ``__next__`` always has an iterator to
        # pull from (a bare ``list`` would otherwise fail ``next(...)``).
        def _factory() -> Iterator[np.ndarray]:
            return iter(source())  # type: ignore[operator]

    elif isinstance(source, Iterator):
        consumed = {"done": False}

        def _factory() -> Iterator[np.ndarray]:
            if consumed["done"]:
                raise RuntimeError(
                    "source is a bare iterator and cycle=True was requested; pass a "
                    "zero-arg factory or a re-iterable instead"
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
) -> int:
    """Resolve ``total_size="auto"``: a source ``.n_rows`` (cheap) else a counting pass.

    Fast path: if ``source`` advertises ``.n_rows`` (e.g. :func:`parquet_source`, which
    reads it from Parquet metadata without scanning the data) use it directly. Otherwise
    do a single counting pass over a finite, re-readable source. A bare one-shot iterator
    cannot be auto-counted (counting consumes it) and an infinite stream would make the
    pass hang -- both must pass ``total_size`` explicitly.
    """
    n = getattr(source, "n_rows", None)
    if n is None:
        # The user's source may not carry .n_rows even when the (shuffle-wrapped)
        # factory does; fall back to the factory's own forwarded count.
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
        count += int(np.asarray(chunk).shape[0])
    if count <= 0:
        raise ValueError("total_size='auto' counted 0 rows (empty or non-re-readable source).")
    if factory() is first_iter:
        # A genuine factory yields a FRESH iterator each call; one that returns the
        # same (now-exhausted) iterator would leave advance() with nothing to pull.
        raise ValueError(
            "total_size='auto' got a factory that returns the same one-shot iterator "
            "each call; pass a factory that creates a fresh iterator each call, or "
            "total_size=N explicitly."
        )
    return count


class _ParquetDataset(IterableDataset):
    """An :class:`IterableDataset` over a directory of Parquet shards.

    Yields one ``(rows, n_columns)`` ``float64`` array per file and exposes
    :attr:`n_rows` read from Parquet *metadata* (no data scan).
    """

    def __init__(self, paths: list[str], columns: list[str] | None, n_rows: int):
        self._paths = paths
        self._columns = columns
        self.n_rows = n_rows

    def __iter__(self) -> Iterator[np.ndarray]:
        import pyarrow.parquet as pq

        for path in self._paths:
            table = pq.read_table(path, columns=self._columns)
            yield np.column_stack([table.column(c).to_numpy() for c in table.column_names])


def parquet_source(
    directory: str,
    *,
    columns: list[str] | None = None,
    pattern: str = "*.parquet",
) -> _ParquetDataset:
    """An :class:`IterableDataset` over a directory of Parquet files.

    Yields one ``(rows, n_columns)`` ``float64`` array per file, and carries an
    ``n_rows`` attribute read from Parquet *metadata* (no data scan) so that
    ``DataLoader(parquet_source(dir), ..., total_size="auto")`` resolves the
    dataset size for free. Pass ``shuffle=True`` to the :class:`DataLoader` (or
    wrap in :func:`shuffle_buffer`) to get shuffled batches.
    """
    import glob as _glob
    import os

    import pyarrow.parquet as pq

    paths = sorted(_glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise ValueError(f"no Parquet files match {os.path.join(directory, pattern)!r}")
    n_rows = sum(pq.read_metadata(p).num_rows for p in paths)
    return _ParquetDataset(paths, columns, n_rows)
