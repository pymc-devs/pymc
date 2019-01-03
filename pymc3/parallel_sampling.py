import multiprocessing
import multiprocessing.sharedctypes
import ctypes
import time
import logging
from collections import namedtuple
import traceback
from pymc3.exceptions import SamplingError
import errno

import numpy as np

from . import theanof

logger = logging.getLogger("pymc3")


def _get_broken_pipe_exception():
    import sys
    if sys.platform == 'win32':
        return RuntimeError("The communication pipe between the main process "
                            "and its spawned children is broken.\n"
                            "In Windows OS, this usually means that the child "
                            "process raised an exception while it was being "
                            "spawned, before it was setup to communicate to "
                            "the main process.\n"
                            "The exceptions raised by the child process while "
                            "spawning cannot be caught or handled from the "
                            "main process, and when running from an IPython or "
                            "jupyter notebook interactive kernel, the child's "
                            "exception and traceback appears to be lost.\n"
                            "A known way to see the child's error, and try to "
                            "fix or handle it, is to run the problematic code "
                            "as a batch script from a system's Command Prompt. "
                            "The child's exception will be printed to the "
                            "Command Promt's stderr, and it should be visible "
                            "above this error and traceback.\n"
                            "Note that if running a jupyter notebook that was "
                            "invoked from a Command Prompt, the child's "
                            "exception should have been printed to the Command "
                            "Prompt on which the notebook is running.")
    else:
        return None


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


class _Process(multiprocessing.Process):
    """Seperate process for each chain.
    We communicate with the main process using a pipe,
    and send finished samples using shared memory.
    """

    def __init__(self, name, msg_pipe, step_method, shared_point, draws, tune, seed):
        super().__init__(daemon=True, name=name)
        self._msg_pipe = msg_pipe
        self._step_method = step_method
        self._shared_point = shared_point
        self._seed = seed
        self._tt_seed = seed + 1
        self._draws = draws
        self._tune = tune

    def run(self):
        try:
            # We do not create this in __init__, as pickling this
            # would destroy the shared memory.
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
        shape_dtypes = self._step_method.vars_shape_dtype
        point = {}
        for name, (shape, dtype) in shape_dtypes.items():
            array = self._shared_point[name]
            self._shared_point[name] = array
            point[name] = np.frombuffer(array, dtype).reshape(shape)
        return point

    def _write_point(self, point):
        for name, vals in point.items():
            self._point[name][...] = vals

    def _recv_msg(self):
        return self._msg_pipe.recv()

    def _start_loop(self):
        np.random.seed(self._seed)
        theanof.set_tt_rng(self._tt_seed)

        draw = 0
        tuning = True

        msg = self._recv_msg()
        if msg[0] == "abort":
            raise KeyboardInterrupt()
        if msg[0] != "start":
            raise ValueError("Unexpected msg " + msg[0])

        while True:
            if draw < self._draws + self._tune:
                try:
                    point, stats = self._compute_point()
                except SamplingError as e:
                    warns = self._collect_warnings()
                    e = ExceptionWithTraceback(e, e.__traceback__)
                    self._msg_pipe.send(("error", warns, e))
            else:
                return

            if draw == self._tune:
                self._step_method.stop_tuning()
                tuning = False

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
                self._msg_pipe.send(
                    ("writing_done", is_last, draw, tuning, stats, warns)
                )
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


class ProcessAdapter:
    """Control a Chain process from the main thread."""

    def __init__(self, draws, tune, step_method, chain, seed, start):
        self.chain = chain
        process_name = "worker_chain_%s" % chain
        self._msg_pipe, remote_conn = multiprocessing.Pipe()

        self._shared_point = {}
        self._point = {}
        for name, (shape, dtype) in step_method.vars_shape_dtype.items():
            size = 1
            for dim in shape:
                size *= int(dim)
            size *= dtype.itemsize
            if size != ctypes.c_size_t(size).value:
                raise ValueError("Variable %s is too large" % name)

            array = multiprocessing.sharedctypes.RawArray("c", size)
            self._shared_point[name] = array
            array_np = np.frombuffer(array, dtype).reshape(shape)
            array_np[...] = start[name]
            self._point[name] = array_np

        self._readable = True
        self._num_samples = 0

        self._process = _Process(
            process_name,
            remote_conn,
            step_method,
            self._shared_point,
            draws,
            tune,
            seed,
        )
        # We fork right away, so that the main process can start tqdm threads
        try:
            self._process.start()
        except IOError as e:
            # Something may have gone wrong during the fork / spawn
            if e.errno == errno.EPIPE:
                exc = _get_broken_pipe_exception()
                if exc is not None:
                    # Sleep a little to give the child process time to flush
                    # all its error message
                    time.sleep(0.2)
                    raise exc
            raise

    @property
    def shared_point_view(self):
        """May only be written to or read between a `recv_draw`
        call from the process and a `write_next` or `abort` call.
        """
        if not self._readable:
            raise RuntimeError()
        return self._point

    def start(self):
        self._msg_pipe.send(("start",))

    def write_next(self):
        self._readable = False
        self._msg_pipe.send(("write_next",))

    def abort(self):
        self._msg_pipe.send(("abort",))

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
            except EOFError:
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
                "Chain processes did not terminate as expected. "
                "Terminating forcefully..."
            )
            for process in processes:
                process.terminate()
            for process in processes:
                process.join()


Draw = namedtuple(
    "Draw", ["chain", "is_last", "draw_idx", "tuning", "stats", "point", "warnings"]
)


class ParallelSampler:
    def __init__(
        self,
        draws,
        tune,
        chains,
        cores,
        seeds,
        start_points,
        step_method,
        start_chain_num=0,
        progressbar=True,
    ):
        if progressbar:
            import tqdm

            tqdm_ = tqdm.tqdm

        if any(len(arg) != chains for arg in [seeds, start_points]):
            raise ValueError("Number of seeds and start_points must be %s." % chains)

        self._samplers = [
            ProcessAdapter(
                draws, tune, step_method, chain + start_chain_num, seed, start
            )
            for chain, seed, start in zip(range(chains), seeds, start_points)
        ]

        self._inactive = self._samplers.copy()
        self._finished = []
        self._active = []
        self._max_active = cores

        self._in_context = False
        self._start_chain_num = start_chain_num

        self._progress = None
        if progressbar:
            self._progress = tqdm_(
                total=chains * (draws + tune),
                unit="draws",
                desc="Sampling %s chains" % chains,
            )

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

        while self._active:
            draw = ProcessAdapter.recv_draw(self._active)
            proc, is_last, draw, tuning, stats, warns = draw
            if self._progress is not None:
                self._progress.update()

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
        if self._progress is not None:
            self._progress.close()
