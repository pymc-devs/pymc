#   Copyright 2020 The PyMC Developers
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
from typing import Dict, List, Sequence

import cloudpickle
import numpy as np

from fastprogress.fastprogress import progress_bar

from pymc import aesaraf
from pymc.blocking import DictToArrayBijection
from pymc.exceptions import SamplingError

logger = logging.getLogger("pymc")


class ParallelSamplingError(Exception):
    def __init__(self, message, chain, warnings=None):
        super().__init__(message)
        if warnings is None:
            warnings = []
        self._chain = chain
        self._warnings = warnings


# Taken from https://hg.python.org/cpython/rev/c4f92b597074
class RemoteTraceback(Exception):
    def __init__(self, tb):
        self.tb = tb

    def __str__(self):
        return self.tb


class ExceptionWithTraceback:
    def __init__(self, exc, tb):
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = "".join(tb)
        self.exc = exc
        self.tb = '\n"""\n%s"""' % tb

    def __reduce__(self):
        return rebuild_exc, (self.exc, self.tb)


def rebuild_exc(exc, tb):
    exc.__cause__ = RemoteTraceback(tb)
    return exc


# Messages
# ('writing_done', is_last, sample_idx, tuning, stats, warns)
# ('error', warnings, *exception_info)

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
        seed,
    ):
        self._msg_pipe = msg_pipe
        self._step_method = step_method
        self._step_method_is_pickled = step_method_is_pickled
        self._shared_point = shared_point
        self._seed = seed
        self._at_seed = seed + 1
        self._draws = draws
        self._tune = tune

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
        try:
            # We do not create this in __init__, as pickling this
            # would destroy the shared memory.
            self._unpickle_step_method()
            self._point = self._make_numpy_refs()
            self._start_loop()
        except KeyboardInterrupt:
            pass
        except BaseException as e:
            e = ExceptionWithTraceback(e, e.__traceback__)
            # Send is not blocking so we have to force a wait for the abort
            # message
            self._msg_pipe.send(("error", None, e))
            self._wait_for_abortion()
        finally:
            self._msg_pipe.close()

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
        np.random.seed(self._seed)
        aesaraf.set_at_rng(self._at_seed)

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
                    point, stats = self._compute_point()
                except SamplingError as e:
                    warns = self._collect_warnings()
                    e = ExceptionWithTraceback(e, e.__traceback__)
                    self._msg_pipe.send(("error", warns, e))
            else:
                return

            msg = self._recv_msg()
            if msg[0] == "abort":
                raise KeyboardInterrupt()
            elif msg[0] == "write_next":
                self._write_point(point)
                is_last = draw + 1 == self._draws + self._tune
                if is_last:
                    warns = self._collect_warnings()
                else:
                    warns = None
                self._msg_pipe.send(("writing_done", is_last, draw, tuning, stats, warns))
                draw += 1
            else:
                raise ValueError("Unknown message " + msg[0])

    def _compute_point(self):
        if self._step_method.generates_stats:
            point, stats = self._step_method.step(self._point)
        else:
            point = self._step_method.step(self._point)
            stats = None
        return point, stats

    def _collect_warnings(self):
        if hasattr(self._step_method, "warnings"):
            return self._step_method.warnings()
        else:
            return []


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
        seed,
        start: Dict[str, np.ndarray],
        mp_ctx,
    ):
        self.chain = chain
        process_name = "worker_chain_%s" % chain
        self._msg_pipe, remote_conn = multiprocessing.Pipe()

        self._shared_point = {}
        self._point = {}

        for name, shape, dtype in DictToArrayBijection.map(start).point_map_info:
            size = 1
            for dim in shape:
                size *= int(dim)
            size *= dtype.itemsize
            if size != ctypes.c_size_t(size).value:
                raise ValueError("Variable %s is too large" % name)

            array = mp_ctx.RawArray("c", size)
            self._shared_point[name] = (array, shape, dtype)
            array_np = np.frombuffer(array, dtype).reshape(shape)
            array_np[...] = start[name]
            self._point[name] = array_np

        self._readable = True
        self._num_samples = 0

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
                seed,
            ),
        )
        self._process.start()
        # Close the remote pipe, so that we get notified if the other
        # end is closed.
        remote_conn.close()

    @property
    def shared_point_view(self):
        """May only be written to or read between a `recv_draw`
        call from the process and a `write_next` or `abort` call.
        """
        if not self._readable:
            raise RuntimeError()
        return self._point

    def _send(self, msg, *args):
        try:
            self._msg_pipe.send((msg, *args))
        except Exception:
            # try to recive an error message
            message = None
            try:
                message = self._msg_pipe.recv()
            except Exception:
                pass
            if message is not None and message[0] == "error":
                warns, old_error = message[1:]
                if warns is not None:
                    error = ParallelSamplingError(str(old_error), self.chain, warns)
                else:
                    error = RuntimeError("Chain %s failed." % self.chain)
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
            warns, old_error = msg[1:]
            if warns is not None:
                error = ParallelSamplingError(str(old_error), proc.chain, warns)
            else:
                error = RuntimeError("Chain %s failed." % proc.chain)
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
            logger.warn(
                "Chain processes did not terminate as expected. " "Terminating forcefully..."
            )
            for process in processes:
                process.terminate()
            for process in processes:
                process.join()


Draw = namedtuple("Draw", ["chain", "is_last", "draw_idx", "tuning", "stats", "point", "warnings"])


class ParallelSampler:
    def __init__(
        self,
        draws: int,
        tune: int,
        chains: int,
        cores: int,
        seeds: list,
        start_points: Sequence[Dict[str, np.ndarray]],
        step_method,
        start_chain_num: int = 0,
        progressbar: bool = True,
        mp_ctx=None,
    ):

        if any(len(arg) != chains for arg in [seeds, start_points]):
            raise ValueError("Number of seeds and start_points must be %s." % chains)

        if mp_ctx is None or isinstance(mp_ctx, str):
            # Closes issue https://github.com/pymc-devs/pymc/issues/3849
            if platform.system() == "Darwin":
                mp_ctx = "forkserver"
            mp_ctx = multiprocessing.get_context(mp_ctx)

        step_method_pickled = None
        if mp_ctx.get_start_method() != "fork":
            step_method_pickled = cloudpickle.dumps(step_method, protocol=-1)

        self._samplers = [
            ProcessAdapter(
                draws,
                tune,
                step_method,
                step_method_pickled,
                chain + start_chain_num,
                seed,
                start,
                mp_ctx,
            )
            for chain, seed, start in zip(range(chains), seeds, start_points)
        ]

        self._inactive = self._samplers.copy()
        self._finished: List[ProcessAdapter] = []
        self._active: List[ProcessAdapter] = []
        self._max_active = cores

        self._in_context = False
        self._start_chain_num = start_chain_num

        self._progress = None
        self._divergences = 0
        self._total_draws = 0
        self._desc = "Sampling {0._chains:d} chains, {0._divergences:,d} divergences"
        self._chains = chains
        if progressbar:
            self._progress = progress_bar(range(chains * (draws + tune)), display=progressbar)
            self._progress.comment = self._desc.format(self)

    def _make_active(self):
        while self._inactive and len(self._active) < self._max_active:
            proc = self._inactive.pop(0)
            proc.start()
            proc.write_next()
            self._active.append(proc)

    def __iter__(self):
        if not self._in_context:
            raise ValueError("Use ParallelSampler as context manager.")
        self._make_active()

        if self._active and self._progress:
            self._progress.update(self._total_draws)

        while self._active:
            draw = ProcessAdapter.recv_draw(self._active)
            proc, is_last, draw, tuning, stats, warns = draw
            self._total_draws += 1
            if not tuning and stats and stats[0].get("diverging"):
                self._divergences += 1
                if self._progress:
                    self._progress.comment = self._desc.format(self)
            if self._progress:
                self._progress.update(self._total_draws)

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

            yield Draw(proc.chain, is_last, draw, tuning, stats, point, warns)

    def __enter__(self):
        self._in_context = True
        return self

    def __exit__(self, *args):
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
