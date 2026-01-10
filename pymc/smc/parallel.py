#   Copyright 2026 - present The PyMC Developers
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
import logging
import multiprocessing
import time

from collections import defaultdict, namedtuple
from collections.abc import Sequence

import cloudpickle
import numpy as np

from rich.theme import Theme

from pymc import Model
from pymc.distributions.custom import CustomDistRV, CustomSymbolicDistRV
from pymc.distributions.distribution import _support_point
from pymc.logprob.abstract import _icdf, _logcdf, _logprob
from pymc.progress_bar import CustomProgress, default_progress_theme
from pymc.sampling.parallel import ExceptionWithTraceback, ParallelSamplingError
from pymc.util import RandomGeneratorState, get_state_from_generator, random_generator_from_state

logger = logging.getLogger(__name__)

SMCResult = namedtuple(
    "SMCResult", ["chain", "is_last", "stage", "beta", "trace", "sample_stats", "sample_settings"]
)


class _SMCProcess:
    """Separate process for each SMC chain.

    Follows the pattern from parallel.py but adapted for SMC sampling.
    """

    def __init__(
        self,
        name: str,
        msg_pipe,
        draws: int,
        kernel,
        kernel_pickled: bool,
        start,
        model,
        model_pickled: bool,
        rng_state: RandomGeneratorState,
        chain: int,
        kernel_kwargs: dict,
        custom_methods: dict,
    ):
        self._msg_pipe = msg_pipe
        self._draws = draws
        self._kernel = kernel
        self._kernel_pickled = kernel_pickled
        self._start = start
        self._model = model
        self._model_pickled = model_pickled
        self._rng = random_generator_from_state(rng_state)
        self.chain = chain
        self._kernel_kwargs = kernel_kwargs
        self._custom_methods = custom_methods

    def _unpickle_objects(self):
        """Unpickle model and kernel if needed."""
        if self._model_pickled:
            self._model = cloudpickle.loads(self._model)
        if self._kernel_pickled:
            self._kernel = cloudpickle.loads(self._kernel)
        if self._start is not None:
            self._start = cloudpickle.loads(self._start)

        # Unpickle kernel_kwargs
        self._kernel_kwargs = {
            key: cloudpickle.loads(value) for key, value in self._kernel_kwargs.items()
        }

    def run(self):
        try:
            self._unpickle_objects()
            self._register_custom_methods()
            self._run_smc()
        except KeyboardInterrupt:
            pass
        except BaseException as e:
            e = ExceptionWithTraceback(e, e.__traceback__)
            self._msg_pipe.send(("error", e))
            self._wait_for_abortion()
        finally:
            self._msg_pipe.close()

    def _register_custom_methods(self):
        """Register custom distribution methods for this process."""
        for cls, (logprob, logcdf, icdf, support_point) in self._custom_methods.items():
            cls = cloudpickle.loads(cls)
            if logprob is not None:
                _logprob.register(cls, cloudpickle.loads(logprob))
            if logcdf is not None:
                _logcdf.register(cls, cloudpickle.loads(logcdf))
            if icdf is not None:
                _icdf.register(cls, cloudpickle.loads(icdf))
            if support_point is not None:
                _support_point.register(cls, cloudpickle.loads(support_point))

    def _wait_for_abortion(self):
        """Wait for abort message from main process."""
        while True:
            msg = self._recv_msg()
            if msg[0] == "abort":
                break

    def _recv_msg(self):
        """Receive message from main process."""
        return self._msg_pipe.recv()

    def _run_smc(self):
        """Run the SMC sampling algorithm."""
        msg = self._recv_msg()
        if msg[0] == "abort":
            raise KeyboardInterrupt()
        if msg[0] != "start":
            raise ValueError("Unexpected msg " + msg[0])

        smc = self._kernel(
            draws=self._draws,
            start=self._start,
            model=self._model,
            random_seed=self._rng,
            **self._kernel_kwargs,
        )

        smc._initialize_kernel()
        smc.setup_kernel()

        stage = 0
        sample_stats = defaultdict(list)

        while smc.beta < 1:
            smc.update_beta_and_weights()

            self._msg_pipe.send(("progress", stage, smc.beta))

            msg = self._recv_msg()
            if msg[0] == "abort":
                raise KeyboardInterrupt()
            elif msg[0] != "continue":
                raise ValueError("Unknown message " + msg[0])

            smc.resample()
            smc.tune()
            smc.mutate()

            # Collect sample stats
            for stat, value in smc.sample_stats().items():
                sample_stats[stat].append(value)

            stage += 1

        trace = smc._posterior_to_trace(self.chain)
        sample_settings = smc.sample_settings()

        result = cloudpickle.dumps(
            (stage, smc.beta, trace, dict(sample_stats), sample_settings), protocol=-1
        )
        self._msg_pipe.send(("done", result))


def _run_smc_process(*args):
    """Entry point for SMC process."""
    _SMCProcess(*args).run()


class SMCProcessAdapter:
    """Adapter to control an SMC process from the main thread."""

    def __init__(
        self,
        draws: int,
        kernel,
        kernel_pickled,
        chain: int,
        rng: np.random.Generator,
        start,
        model,
        model_pickled: bool,
        mp_ctx,
        kernel_kwargs: dict,
        custom_methods: dict,
    ):
        self.chain = chain
        process_name = f"smc_worker_chain_{chain}"
        self._msg_pipe, remote_conn = multiprocessing.Pipe()

        self._process = mp_ctx.Process(
            daemon=True,
            name=process_name,
            target=_run_smc_process,
            args=(
                process_name,
                remote_conn,
                draws,
                kernel,
                kernel_pickled,
                start,
                model,
                model_pickled,
                get_state_from_generator(rng),
                chain,
                kernel_kwargs,
                custom_methods,
            ),
        )
        self._process.start()
        remote_conn.close()

        self._current_stage = 0
        self._current_beta = 0.0

    def _send(self, msg, *args):
        """Send message to the worker process."""
        try:
            self._msg_pipe.send((msg, *args))
        except Exception:
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
        """Send start signal to the worker."""
        self._send("start")

    def continue_sampling(self):
        """Send continue signal to the worker."""
        self._send("continue")

    def abort(self):
        """Send abort signal to the worker."""
        self._send("abort")

    def join(self, timeout=None):
        """Join the worker process."""
        self._process.join(timeout)

    def terminate(self):
        """Terminate the worker process."""
        self._process.terminate()

    @staticmethod
    def recv_message(processes, timeout=3600):
        """Receive a message from any of the worker processes."""
        if not processes:
            raise ValueError("No processes.")
        pipes = [proc._msg_pipe for proc in processes]
        ready = multiprocessing.connection.wait(pipes, timeout=timeout)
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
        elif msg[0] == "progress":
            proc._current_stage = msg[1]
            proc._current_beta = msg[2]
            return (proc, "progress", msg[1], msg[2])
        elif msg[0] == "done":
            # Unpickle the results
            stage, beta, trace, sample_stats, sample_settings = cloudpickle.loads(msg[1])
            return (proc, "done", stage, beta, trace, sample_stats, sample_settings)
        else:
            raise ValueError("Sampler sent bad message: " + msg[0])

    @staticmethod
    def terminate_all(processes, patience=2):
        """Terminate all worker processes."""
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


class ParallelSMCSampler:
    """Parallel sampler for SMC chains."""

    def __init__(
        self,
        *,
        draws: int,
        kernel,
        chains: int,
        cores: int,
        rngs: Sequence[np.random.Generator],
        start_points: Sequence[dict[str, np.ndarray] | None],
        model: Model,
        progressbar: bool = True,
        progressbar_theme: Theme | None = default_progress_theme,
        mp_ctx,
        kernel_kwargs: dict,
    ):
        if any(len(arg) != chains for arg in [rngs, start_points]):
            raise ValueError(f"Number of rngs and start_points must be {chains}.")

        custom_methods = _find_custom_dist_dispatch_methods(model)

        kernel_pickled = None
        model_pickled = False
        start_points_pickled = [None] * chains

        if mp_ctx.get_start_method() != "fork":
            kernel_pickled = True
            kernel = cloudpickle.dumps(kernel, protocol=-1)
            model = cloudpickle.dumps(model, protocol=-1)
            model_pickled = True
            start_points_pickled = [
                cloudpickle.dumps(start, protocol=-1) if start is not None else None
                for start in start_points
            ]

        kernel_kwargs_pickled = {
            key: cloudpickle.dumps(value, protocol=-1) for key, value in kernel_kwargs.items()
        }

        self._samplers = [
            SMCProcessAdapter(
                draws,
                kernel,
                kernel_pickled,
                chain,
                rng,
                start_pickled,
                model,
                model_pickled,
                mp_ctx,
                kernel_kwargs_pickled,
                custom_methods,
            )
            for chain, rng, start_pickled in zip(range(chains), rngs, start_points_pickled)
        ]

        self._inactive = self._samplers.copy()
        self._finished: list[SMCProcessAdapter] = []
        self._active: list[SMCProcessAdapter] = []
        self._max_active = cores

        self._in_context = False

        self._progressbar = progressbar
        self._progressbar_theme = progressbar_theme
        self._chains = chains

    def _make_active(self):
        """Start inactive processes up to the maximum number of active processes."""
        while self._inactive and len(self._active) < self._max_active:
            proc = self._inactive.pop(0)
            proc.start()
            self._active.append(proc)

    def __iter__(self):
        """Iterate over SMC results."""
        if not self._in_context:
            raise ValueError("Use ParallelSMCSampler as context manager.")

        self._make_active()

        from rich.console import Console
        from rich.progress import SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

        with CustomProgress(
            TextColumn("{task.description}"),
            SpinnerColumn(),
            TimeRemainingColumn(),
            TextColumn("/"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[status]}"),
            console=Console(theme=self._progressbar_theme),
            disable=not self._progressbar,
        ) as progress:
            task_ids = {}
            for proc in self._samplers:
                task_id = progress.add_task(f"Chain {proc.chain}", status="Stage: 0 Beta: 0.0000")
                task_ids[proc.chain] = task_id

            while self._active:
                result = SMCProcessAdapter.recv_message(self._active, timeout=3600)
                proc = result[0]
                msg_type = result[1]

                if msg_type == "progress":
                    stage, beta = result[2], result[3]

                    progress.update(
                        task_ids[proc.chain],
                        status=f"Stage: {stage} Beta: {beta:.4f}",
                        refresh=True,
                    )

                    proc.continue_sampling()

                elif msg_type == "done":
                    stage, beta, trace, sample_stats, sample_settings = result[2:]

                    progress.update(
                        task_ids[proc.chain],
                        status=f"Stage: {stage} Beta: {beta:.4f}",
                        refresh=True,
                    )

                    proc.join()
                    self._active.remove(proc)
                    self._finished.append(proc)
                    self._make_active()

                    yield SMCResult(
                        proc.chain, True, stage, beta, trace, sample_stats, sample_settings
                    )

    def __enter__(self):
        """Enter the context manager."""
        self._in_context = True
        return self

    def __exit__(self, *args):
        """Exit the context manager."""
        SMCProcessAdapter.terminate_all(self._samplers)


def _find_custom_dist_dispatch_methods(model):
    custom_methods = {}
    for rv in model.basic_RVs:
        rv_type = rv.owner.op
        cls = type(rv_type)
        if isinstance(rv_type, CustomDistRV | CustomSymbolicDistRV):
            custom_methods[cloudpickle.dumps(cls)] = (
                cloudpickle.dumps(_logprob.registry.get(cls, None)),
                cloudpickle.dumps(_logcdf.registry.get(cls, None)),
                cloudpickle.dumps(_icdf.registry.get(cls, None)),
                cloudpickle.dumps(_support_point.registry.get(cls, None)),
            )

    return custom_methods
