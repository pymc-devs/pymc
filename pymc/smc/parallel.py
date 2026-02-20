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

from collections import namedtuple
from collections.abc import Sequence
from contextlib import nullcontext

import cloudpickle
import numpy as np

from rich.theme import Theme
from threadpoolctl import threadpool_limits

from pymc.progress_bar import SMCProgressBarManager, default_progress_theme
from pymc.sampling.parallel import ExceptionWithTraceback, ParallelSamplingError
from pymc.smc.kernels import SMC_KERNEL
from pymc.util import RandomGeneratorState, get_state_from_generator, random_generator_from_state

logger = logging.getLogger(__name__)

SMCResult = namedtuple(
    "SMCResult",
    [
        "chain",
        "is_last",
        "stage",
        "beta",
        "tempered_posterior",
        "var_info",
        "variables",
        "sample_stats",
        "sample_settings",
    ],
)


class _SMCProcess:
    """Separate process for each SMC chain.

    Follows the pattern from parallel.py but adapted for SMC sampling.
    """

    def __init__(
        self,
        name: str,
        msg_pipe,
        kernel,
        start,
        rng_state: RandomGeneratorState,
        chain: int,
        mp_start_method: str,
        blas_cores: int | None,
    ):
        self._msg_pipe = msg_pipe
        self._kernel = kernel
        self._start = start
        self._rng = random_generator_from_state(rng_state)
        self.chain = chain
        self._mp_start_method = mp_start_method
        self._blas_cores = blas_cores

    def _unpickle_objects(self):
        """Unpickle kernel and start point."""
        self._kernel = cloudpickle.loads(self._kernel)
        if self._start is not None:
            self._start = cloudpickle.loads(self._start)

    def run(self):
        with (
            nullcontext()
            if self._mp_start_method == "fork"
            else threadpool_limits(limits=self._blas_cores)
        ):
            self._unpickle_objects()

            try:
                self._run_smc()
            except KeyboardInterrupt:
                pass
            except BaseException as e:
                e = ExceptionWithTraceback(e, e.__traceback__)
                self._msg_pipe.send(("error", e))
                self._wait_for_abortion()
            finally:
                self._msg_pipe.close()

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

        smc = self._kernel
        smc.initialize(self._start, self._rng)

        stage = 0
        sample_stats: dict[str, list] = {stat: [] for stat in smc.stats_dtypes_shapes}  # type: ignore[annotation-unchecked]

        while smc.beta < 1:
            smc.update_beta_and_weights()

            self._msg_pipe.send(("progress", stage, smc.beta))

            msg = self._recv_msg()
            if msg[0] == "abort":
                raise KeyboardInterrupt()
            elif msg[0] != "continue":
                raise ValueError("Unknown message " + msg[0])

            for stat, value in smc.step().items():
                sample_stats[stat].append(value)

            stage += 1

        result = cloudpickle.dumps(
            (
                stage,
                smc.beta,
                smc.tempered_posterior,
                smc.var_info,
                smc.variables,
                sample_stats,
                smc.sample_settings(),
            ),
            protocol=-1,
        )
        self._msg_pipe.send(("done", result))


def _run_smc_process(*args):
    """Entry point for SMC process."""
    _SMCProcess(*args).run()


class SMCProcessAdapter:
    """Adapter to control an SMC process from the main thread."""

    def __init__(
        self,
        kernel,
        chain: int,
        rng: np.random.Generator,
        start,
        mp_ctx,
        blas_cores: int | None,
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
                kernel,
                start,
                get_state_from_generator(rng),
                chain,
                mp_ctx.get_start_method(),
                blas_cores,
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
            stage, beta, tempered_posterior, var_info, variables, sample_stats, sample_settings = (
                cloudpickle.loads(msg[1])
            )
            return (
                proc,
                "done",
                stage,
                beta,
                tempered_posterior,
                var_info,
                variables,
                sample_stats,
                sample_settings,
            )
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
        kernel: SMC_KERNEL,
        chains: int,
        cores: int,
        rngs: Sequence[np.random.Generator],
        start_points: list[dict[str, np.ndarray] | None],
        mp_ctx: multiprocessing.context.BaseContext,
        progressbar: bool = True,
        progressbar_theme: Theme | None = default_progress_theme,
        blas_cores: int | None = None,
    ):
        if any(len(arg) != chains for arg in [rngs, start_points]):
            raise ValueError(f"Number of rngs and start_points must be {chains}.")

        self._kernel = kernel
        kernel_pickled = cloudpickle.dumps(kernel, protocol=-1)
        start_points_pickled = [
            cloudpickle.dumps(start, protocol=-1) if start is not None else None
            for start in start_points
        ]

        self._samplers = [
            SMCProcessAdapter(
                kernel_pickled,
                chain,
                rng,
                start_pickled,
                mp_ctx,
                blas_cores,
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

        with SMCProgressBarManager(
            kernel=self._kernel,
            chains=self._chains,
            progressbar=self._progressbar,
            progressbar_theme=self._progressbar_theme,
        ) as progress_manager:
            chain_betas = {proc.chain: 0.0 for proc in self._samplers}

            while self._active:
                result = SMCProcessAdapter.recv_message(self._active, timeout=3600)
                proc = result[0]
                msg_type = result[1]

                if msg_type == "progress":
                    stage, beta = result[2], result[3]
                    old_beta = chain_betas[proc.chain]
                    chain_betas[proc.chain] = beta

                    progress_manager.update(proc.chain, stage, beta, old_beta)
                    proc.continue_sampling()

                elif msg_type == "done":
                    (
                        stage,
                        beta,
                        tempered_posterior,
                        var_info,
                        variables,
                        sample_stats,
                        sample_settings,
                    ) = result[2:]

                    progress_manager.update(proc.chain, stage, beta, is_last=True)

                    proc.join()
                    self._active.remove(proc)
                    self._finished.append(proc)
                    self._make_active()

                    yield SMCResult(
                        proc.chain,
                        True,
                        stage,
                        beta,
                        tempered_posterior,
                        var_info,
                        variables,
                        sample_stats,
                        sample_settings,
                    )

    def __enter__(self):
        """Enter the context manager."""
        self._in_context = True
        return self

    def __exit__(self, *args):
        """Exit the context manager."""
        SMCProcessAdapter.terminate_all(self._samplers)
