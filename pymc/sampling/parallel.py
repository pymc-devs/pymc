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

import ctypes
import logging
import multiprocessing
import multiprocessing.sharedctypes
import platform
import time
import traceback

from collections import namedtuple
from collections.abc import Sequence
from contextlib import nullcontext
from typing import cast

import cloudpickle
import numpy as np

from rich.theme import Theme
from threadpoolctl import threadpool_limits

from pymc.backends.zarr import ZarrChain
from pymc.blocking import DictToArrayBijection
from pymc.exceptions import SamplingError
from pymc.progress_bar import ProgressBarManager, default_progress_theme
from pymc.util import (
    RandomGeneratorState,
    get_state_from_generator,
    random_generator_from_state,
)

logger = logging.getLogger(__name__)


class ParallelSamplingError(Exception):
    def __init__(self, message, chain):
        super().__init__(message)
        self._chain = chain


# Taken from https://hg.python.org/cpython/rev/c4f92b597074
class RemoteTraceback(Exception):
    def __init__(self, tb):
        self.tb = tb

    def __str__(self):
        """Return a string representation of the object."""
        return self.tb


class ExceptionWithTraceback:
    def __init__(self, exc, tb):
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = "".join(tb)
        self.exc = exc
        self.tb = f'\n"""\n{tb}"""'

    def __reduce__(self):
        """Return a tuple to pickle."""
        return rebuild_exc, (self.exc, self.tb)


def rebuild_exc(exc, tb):
    exc.__cause__ = RemoteTraceback(tb)
    return exc


# Messages
# ('writing_done', is_last, sample_idx, tuning, stats)
# ('error', *exception_info)

# ('abort', reason)
# ('write_next',)
# ('start',)


class _Process:
    """Separate process for each chain.

    We communicate with the main process using a pipe,
    and send finished samples using shared memory.
    """

    def __init__(
        self,
        name: str,
        msg_pipe,
        step_method,
        step_method_is_pickled,
        shared_point,
        draws: int,
        tune: int,
        rng_state: RandomGeneratorState,
        blas_cores,
        chain: int,
        mp_start_method: str,
        zarr_chains: list[ZarrChain] | bytes | None = None,
        zarr_chains_is_pickled: bool = False,
    ):
        # Because of https://github.com/numpy/numpy/issues/27727, we can't send
        # the rng instance to the child process because pickling (copying) looses
        # the seed sequence state information. For this reason, we send a
        # RandomGeneratorState instead.
        rng = random_generator_from_state(rng_state)
        self._msg_pipe = msg_pipe
        self._step_method = step_method
        self._step_method_is_pickled = step_method_is_pickled
        self.chain = chain
        self._zarr_recording = False
        self._zarr_chain: ZarrChain | None = None
        if zarr_chains_is_pickled:
            self._zarr_chain = cloudpickle.loads(zarr_chains)[self.chain]
        elif zarr_chains is not None:
            self._zarr_chain = cast(list[ZarrChain], zarr_chains)[self.chain]
        self._zarr_recording = self._zarr_chain is not None

        self._shared_point = shared_point
        self._rng = rng
        self._draws = draws
        self._tune = tune
        self._blas_cores = blas_cores
        self._mp_start_method = mp_start_method

    def _unpickle_step_method(self):
        unpickle_error = (
            "The model could not be unpickled. This is required for sampling "
            "with more than one core and multiprocessing context spawn "
            "or forkserver."
        )
        if self._step_method_is_pickled:
            try:
                self._step_method = cloudpickle.loads(self._step_method)
            except Exception:
                raise ValueError(unpickle_error)

    def run(self):
        # Only apply threadpool_limits for non-fork methods.
        with (
            nullcontext()
            if self._mp_start_method == "fork"
            else threadpool_limits(limits=self._blas_cores)
        ):
            try:
                # We do not create this in __init__, as pickling this
                # would destroy the shared memory.
                self._unpickle_step_method()
                self._link_step_to_zarrchain()
                self._point = self._make_numpy_refs()
                self._start_loop()
            except KeyboardInterrupt:
                pass
            except BaseException as e:
                e = ExceptionWithTraceback(e, e.__traceback__)
                # Send is not blocking so we have to force a wait for the abort
                # message
                self._msg_pipe.send(("error", e))
                self._wait_for_abortion()
            finally:
                self._msg_pipe.close()

    def _link_step_to_zarrchain(self):
        if self._zarr_recording:
            self._zarr_chain.link_stepper(self._step_method)

    def _wait_for_abortion(self):
        while True:
            msg = self._recv_msg()
            if msg[0] == "abort":
                break

    def _make_numpy_refs(self):
        point = {}
        # XXX: I'm assuming that the processes are properly synchronized...
        for name, (array, shape, dtype) in self._shared_point.items():
            point[name] = np.frombuffer(array, dtype).reshape(shape)
        return point

    def _write_point(self, point):
        # XXX: What do we do when the underlying points change shape?
        for name, vals in point.items():
            self._point[name][...] = vals

    def _recv_msg(self):
        return self._msg_pipe.recv()

    def _start_loop(self):
        zarr_recording = self._zarr_recording
        self._step_method.set_rng(self._rng)

        draw = 0
        tuning = True

        msg = self._recv_msg()
        if msg[0] == "abort":
            raise KeyboardInterrupt()
        if msg[0] != "start":
            raise ValueError("Unexpected msg " + msg[0])

        while True:
            if draw == self._tune:
                self._step_method.stop_tuning()
                tuning = False

            if draw < self._draws + self._tune:
                try:
                    point, stats = self._step_method.step(self._point)
                except SamplingError as e:
                    e = ExceptionWithTraceback(e, e.__traceback__)
                    self._msg_pipe.send(("error", e))
            else:
                return

            msg = self._recv_msg()
            if msg[0] == "abort":
                raise KeyboardInterrupt()
            elif msg[0] == "write_next":
                if zarr_recording:
                    self._zarr_chain.record(point, stats)
                self._write_point(point)
                is_last = draw + 1 == self._draws + self._tune
                self._msg_pipe.send(("writing_done", is_last, draw, tuning, stats))
                draw += 1
            else:
                raise ValueError("Unknown message " + msg[0])


def _run_process(*args):
    _Process(*args).run()


class ProcessAdapter:
    """Control a Chain process from the main thread."""

    def __init__(
        self,
        draws: int,
        tune: int,
        step_method,
        step_method_pickled,
        chain: int,
        rng: np.random.Generator,
        start: dict[str, np.ndarray],
        blas_cores,
        mp_ctx,
        zarr_chains: list[ZarrChain] | None = None,
        zarr_chains_pickled: bytes | None = None,
    ):
        self.chain = chain
        process_name = f"worker_chain_{chain}"
        self._msg_pipe, remote_conn = multiprocessing.Pipe()

        self._shared_point = {}
        self._point = {}

        for name, shape, size, dtype in DictToArrayBijection.map(start).point_map_info:
            byte_size = size * dtype.itemsize
            if byte_size != ctypes.c_size_t(byte_size).value:
                raise ValueError(f"Variable {name} is too large")

            array = mp_ctx.RawArray("c", byte_size)
            self._shared_point[name] = (array, shape, dtype)
            array_np = np.frombuffer(array, dtype).reshape(shape)
            array_np[...] = start[name]
            self._point[name] = array_np

        self._readable = True
        self._num_samples = 0

        zarr_chains_send: list[ZarrChain] | bytes | None = None
        if zarr_chains_pickled is not None:
            zarr_chains_send = zarr_chains_pickled
        elif zarr_chains is not None:
            if mp_ctx.get_start_method() == "spawn":
                raise ValueError(
                    "please provide a pre-pickled zarr_chains when multiprocessing start method is 'spawn'"
                )
            zarr_chains_send = zarr_chains

        if step_method_pickled is not None:
            step_method_send = step_method_pickled
        else:
            if mp_ctx.get_start_method() == "spawn":
                raise ValueError(
                    "please provide a pre-pickled step method when multiprocessing start method is 'spawn'"
                )
            step_method_send = step_method

        self._process = mp_ctx.Process(
            daemon=True,
            name=process_name,
            target=_run_process,
            args=(
                process_name,
                remote_conn,
                step_method_send,
                step_method_pickled is not None,
                self._shared_point,
                draws,
                tune,
                get_state_from_generator(rng),
                blas_cores,
                self.chain,
                mp_ctx.get_start_method(),
                zarr_chains_send,
                zarr_chains_pickled is not None,
            ),
        )
        self._process.start()
        # Close the remote pipe, so that we get notified if the other
        # end is closed.
        remote_conn.close()

    @property
    def shared_point_view(self):
        """May only be written to or read between a `recv_draw` call from the process and a `write_next` or `abort` call."""
        if not self._readable:
            raise RuntimeError()
        return self._point

    def _send(self, msg, *args):
        try:
            self._msg_pipe.send((msg, *args))
        except Exception:
            # try to receive an error message
            message = None
            try:
                message = self._msg_pipe.recv()
            except Exception:
                pass
            if message is not None and message[0] == "error":
                old_error = message[1]
                if old_error is not None:
                    error = ParallelSamplingError(
                        f"Chain {self.chain} failed with: {old_error}", self.chain
                    )
                else:
                    error = RuntimeError(f"Chain {self.chain} failed.")
                raise error from old_error
            raise

    def start(self):
        self._send("start")

    def write_next(self):
        self._readable = False
        self._send("write_next")

    def abort(self):
        self._send("abort")

    def join(self, timeout=None):
        self._process.join(timeout)

    def terminate(self):
        self._process.terminate()

    @staticmethod
    def recv_draw(processes, timeout=3600):
        if not processes:
            raise ValueError("No processes.")
        pipes = [proc._msg_pipe for proc in processes]
        ready = multiprocessing.connection.wait(pipes)
        if not ready:
            raise multiprocessing.TimeoutError("No message from samplers.")
        idxs = {id(proc._msg_pipe): proc for proc in processes}
        proc = idxs[id(ready[0])]
        msg = ready[0].recv()

        if msg[0] == "error":
            old_error = msg[1]
            if old_error is not None:
                error = ParallelSamplingError(
                    f"Chain {proc.chain} failed with: {old_error}", proc.chain
                )
            else:
                error = RuntimeError(f"Chain {proc.chain} failed.")
            raise error from old_error
        elif msg[0] == "writing_done":
            proc._readable = True
            proc._num_samples += 1
            return (proc,) + msg[1:]
        else:
            raise ValueError("Sampler sent bad message.")

    @staticmethod
    def terminate_all(processes, patience=2):
        for process in processes:
            try:
                process.abort()
            except Exception:
                pass

        start_time = time.time()
        try:
            for process in processes:
                timeout = time.time() + patience - start_time
                if timeout < 0:
                    raise multiprocessing.TimeoutError()
                process.join(timeout)
        except multiprocessing.TimeoutError:
            logger.warning(
                "Chain processes did not terminate as expected. Terminating forcefully..."
            )
            for process in processes:
                process.terminate()
            for process in processes:
                process.join()


Draw = namedtuple("Draw", ["chain", "is_last", "draw_idx", "tuning", "stats", "point"])


class ParallelSampler:
    def __init__(
        self,
        *,
        draws: int,
        tune: int,
        chains: int,
        cores: int,
        rngs: Sequence[np.random.Generator],
        start_points: Sequence[dict[str, np.ndarray]],
        step_method,
        progressbar: bool = True,
        progressbar_theme: Theme | None = default_progress_theme,
        blas_cores: int | None = None,
        mp_ctx=None,
        zarr_chains: list[ZarrChain] | None = None,
    ):
        if any(len(arg) != chains for arg in [rngs, start_points]):
            raise ValueError(f"Number of rngs and start_points must be {chains}.")

        if mp_ctx is None or isinstance(mp_ctx, str):
            # Closes issue https://github.com/pymc-devs/pymc/issues/3849
            # Related issue https://github.com/pymc-devs/pymc/issues/5339
            if mp_ctx is None and platform.system() == "Darwin":
                if platform.processor() == "arm":
                    mp_ctx = "fork"
                    logger.debug(
                        "mp_ctx is set to 'fork' for MacOS with ARM architecture. "
                        + "This might cause unexpected behavior with JAX, which is inherently multithreaded."
                    )
                else:
                    mp_ctx = "forkserver"

            mp_ctx = multiprocessing.get_context(mp_ctx)

        step_method_pickled = None
        zarr_chains_pickled = None
        self.zarr_recording = False
        if zarr_chains is not None:
            assert all(isinstance(zarr_chain, ZarrChain) for zarr_chain in zarr_chains)
            self.zarr_recording = True
        if mp_ctx.get_start_method() != "fork":
            step_method_pickled = cloudpickle.dumps(step_method, protocol=-1)
            if zarr_chains is not None:
                zarr_chains_pickled = cloudpickle.dumps(zarr_chains, protocol=-1)

        self._samplers = [
            ProcessAdapter(
                draws,
                tune,
                step_method,
                step_method_pickled,
                chain,
                rng,
                start,
                blas_cores,
                mp_ctx,
                zarr_chains=zarr_chains,
                zarr_chains_pickled=zarr_chains_pickled,
            )
            for chain, rng, start in zip(range(chains), rngs, start_points)
        ]

        self._inactive = self._samplers.copy()
        self._finished: list[ProcessAdapter] = []
        self._active: list[ProcessAdapter] = []
        self._max_active = cores

        self._in_context = False
        self._progress = ProgressBarManager(
            step_method=step_method,
            chains=chains,
            draws=draws,
            tune=tune,
            progressbar=progressbar,
            progressbar_theme=progressbar_theme,
        )

    def _make_active(self):
        while self._inactive and len(self._active) < self._max_active:
            proc = self._inactive.pop(0)
            proc.start()
            proc.write_next()
            self._active.append(proc)

    def __iter__(self):
        """Return an iterator over draws."""
        if not self._in_context:
            raise ValueError("Use ParallelSampler as context manager.")
        self._make_active()

        with self._progress:
            while self._active:
                draw = ProcessAdapter.recv_draw(self._active)
                proc, is_last, draw, tuning, stats = draw

                self._progress.update(
                    chain_idx=proc.chain, is_last=is_last, draw=draw, tuning=tuning, stats=stats
                )

                if is_last:
                    proc.join()
                    self._active.remove(proc)
                    self._finished.append(proc)
                    self._make_active()

                # We could also yield proc.shared_point_view directly,
                # and only call proc.write_next() after the yield returns.
                # This seems to be faster overally though, as the worker
                # loses less time waiting.
                point = {name: val.copy() for name, val in proc.shared_point_view.items()}

                # Already called for new proc in _make_active
                if not is_last:
                    proc.write_next()

                yield Draw(proc.chain, is_last, draw, tuning, stats, point)

    def __enter__(self):
        """Enter the context manager."""
        self._in_context = True
        return self

    def __exit__(self, *args):
        """Exit the context manager."""
        ProcessAdapter.terminate_all(self._samplers)


def _cpu_count():
    """Try to guess the number of CPUs in the system.

    We use the number provided by psutil if that is installed.
    If not, we use the number provided by multiprocessing, but assume
    that half of the cpus are only hardware threads and ignore those.
    """
    try:
        cpus = multiprocessing.cpu_count() // 2
    except NotImplementedError:
        cpus = 1
    return cpus
