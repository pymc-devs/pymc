#   Copyright 2024 The PyMC Developers
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

"""Specializes on running MCMCs with population step methods."""

import logging
import warnings

from collections.abc import Iterator, Sequence
from copy import copy
from typing import TypeAlias

import cloudpickle
import numpy as np

from rich.progress import BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from pymc.backends.base import BaseTrace
from pymc.initial_point import PointType
from pymc.model import Model, modelcontext
from pymc.stats.convergence import log_warning_stats
from pymc.step_methods import CompoundStep
from pymc.step_methods.arraystep import (
    BlockedStep,
    PopulationArrayStepShared,
    StatsType,
)
from pymc.step_methods.metropolis import DEMetropolis
from pymc.util import CustomProgress

__all__ = ()


Step: TypeAlias = BlockedStep | CompoundStep


_log = logging.getLogger(__name__)


def _sample_population(
    *,
    initial_points: Sequence[PointType],
    draws: int,
    start: Sequence[PointType],
    rngs: Sequence[np.random.Generator],
    step: BlockedStep | CompoundStep,
    tune: int,
    model: Model,
    progressbar: bool = True,
    parallelize: bool = False,
    traces: Sequence[BaseTrace],
    **kwargs,
):
    """Perform sampling of a population of chains using the ``PopulationStepper``.

    Parameters
    ----------
    draws : int
        The number of samples to draw
    start : list
        Start points for each chain
    rngs: sequence of random Generators
        A list of :py:class:`~numpy.random.Generator` objects, one for each chain
    step : function
        Step function (should be or contain a population step method)
    tune : int
        Number of iterations to tune.
    model : Model (optional if in ``with`` context)
    progressbar : bool
        Show progress bars? (defaults to True)
    parallelize : bool
        Setting for multiprocess parallelization
    """
    warn_population_size(
        step=step,
        initial_points=initial_points,
        model=model,
        chains=len(traces),
    )

    sampling = _prepare_iter_population(
        draws=draws,
        step=step,
        start=start,
        parallelize=parallelize,
        traces=traces,
        tune=tune,
        model=model,
        rngs=rngs,
        progressbar=progressbar,
    )

    with CustomProgress(disable=not progressbar) as progress:
        task = progress.add_task("[red]Sampling...", total=draws)
        for _ in sampling:
            progress.update(task)

    return


def warn_population_size(
    *,
    step: BlockedStep | CompoundStep,
    initial_points: Sequence[PointType],
    model: Model,
    chains: int,
):
    """Emit informative errors/warnings for dangerously small population size."""
    has_demcmc = np.any(
        [
            isinstance(m, DEMetropolis)
            for m in (step.methods if isinstance(step, CompoundStep) else [step])
        ]
    )

    initial_point_model_size = sum(initial_points[0][n.name].size for n in model.value_vars)

    if has_demcmc and chains < 3:
        raise ValueError(
            "DEMetropolis requires at least 3 chains. "
            f"For this {initial_point_model_size}-dimensional model you should use ≥{initial_point_model_size + 1} chains"
        )
    if has_demcmc and chains <= initial_point_model_size:
        warnings.warn(
            "DEMetropolis should be used with more chains than dimensions! "
            f"(The model has {initial_point_model_size} dimensions.)",
            UserWarning,
            stacklevel=2,
        )
    return


class PopulationStepper:
    """Wraps population of step methods to step them in parallel with single or multiprocessing."""

    def __init__(self, steppers, parallelize: bool, progressbar: bool = True):
        """Use multiprocessing to parallelize chains.

        Falls back to sequential evaluation if multiprocessing fails.

        In the multiprocessing mode of operation, a new process is started for each
        chain/stepper and Pipes are used to communicate with the main process.

        Parameters
        ----------
        steppers : list
            A collection of independent step methods, one for each chain.
        parallelize : bool
            Indicates if parallelization via multiprocessing is desired.
        progressbar : bool
            Should we display a progress bar showing relative progress?
        """
        self.nchains = len(steppers)
        self.is_parallelized = False
        self._primary_ends = []
        self._processes = []
        self._steppers = steppers
        self._progress = None
        if parallelize:
            try:
                # configure a child process for each stepper
                _log.info(
                    "Attempting to parallelize chains to all cores. You can turn this off with `pm.sample(cores=1)`."
                )
                import multiprocessing

                with CustomProgress(
                    "[progress.description]{task.description}",
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeRemainingColumn(),
                    TextColumn("/"),
                    TimeElapsedColumn(),
                    disable=not progressbar,
                ) as self._progress:
                    for c, stepper in enumerate(steppers):
                        #     enumerate(progress_bar(steppers)) if progressbar else enumerate(steppers)
                        # ):
                        task = self._progress.add_task(description=f"Chain {c}")
                        secondary_end, primary_end = multiprocessing.Pipe()
                        stepper_dumps = cloudpickle.dumps(stepper, protocol=4)
                        process = multiprocessing.Process(
                            target=self.__class__._run_secondary,
                            args=(c, stepper_dumps, secondary_end, task, self._progress),
                            name=f"ChainWalker{c}",
                        )
                        # we want the child process to exit if the parent is terminated
                        process.daemon = True
                        # Starting the process might fail and takes time.
                        # By doing it in the constructor, the sampling progress bar
                        # will not be confused by the process start.
                        process.start()
                        self._primary_ends.append(primary_end)
                        self._processes.append(process)
                    self.is_parallelized = True
            except Exception:
                _log.info(
                    "Population parallelization failed. "
                    "Falling back to sequential stepping of chains."
                )
                _log.debug("Error was: ", exc_info=True)
        else:
            _log.info(
                "Chains are not parallelized. You can enable this by passing "
                "`pm.sample(cores=n)`, where n > 1."
            )
        return super().__init__()

    def __enter__(self):
        """Do nothing: processes are already started in ``__init__``."""
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self._processes) > 0:
            try:
                for primary_end in self._primary_ends:
                    primary_end.send(None)
                for process in self._processes:
                    process.join(timeout=3)
            except Exception:
                _log.warning("Termination failed.")
        return

    @staticmethod
    def _run_secondary(c, stepper_dumps, secondary_end, task, progress):
        """Perform stepping of a chain from a separate process.

        Parameters
        ----------
        c : int
            number of this chain
        stepper : BlockedStep
            a step method such as CompoundStep
        secondary_end : multiprocessing.connection.PipeConnection
            This is our connection to the main process
        task : progress.Task
            The progress task for this chain
        progress : progress.Progress
            The progress bar
        """
        try:
            stepper = cloudpickle.loads(stepper_dumps)
            # the stepper is not necessarily a PopulationArraySharedStep itself,
            # but rather a CompoundStep. PopulationArrayStepShared.population
            # has to be updated, therefore we identify the substeppers first.
            population_steppers = []
            for sm in stepper.methods if isinstance(stepper, CompoundStep) else [stepper]:
                if isinstance(sm, PopulationArrayStepShared):
                    population_steppers.append(sm)
            while True:
                incoming = secondary_end.recv()
                # receiving a None is the signal to exit
                if incoming is None:
                    break
                tune_stop, population = incoming
                if tune_stop:
                    stepper.stop_tuning()
                # forward the population to the PopulationArrayStepShared objects
                # This is necessary because due to the process fork, the population
                # object is no longer shared between the steppers.
                for popstep in population_steppers:
                    popstep.population = population
                update = stepper.step(population[c])
                progress.advance(task)
                secondary_end.send(update)
        except Exception:
            _log.exception(f"ChainWalker{c}")
        return

    def step(self, tune_stop: bool, population) -> list[tuple[PointType, StatsType]]:
        """Step the entire population of chains.

        Parameters
        ----------
        tune_stop : bool
            Indicates if the condition (i == tune) is fulfilled
        population : list
            Current Points of all chains

        Returns
        -------
        update : list
            List of (Point, stats) tuples for all chains
        """
        updates: list[tuple[PointType, StatsType]] = []
        if self.is_parallelized:
            for c in range(self.nchains):
                self._primary_ends[c].send((tune_stop, population))
            # Blockingly get the step outcomes
            for c in range(self.nchains):
                updates.append(self._primary_ends[c].recv())
        else:
            for c in range(self.nchains):
                if tune_stop:
                    self._steppers[c].stop_tuning()
                updates.append(self._steppers[c].step(population[c]))
        return updates


def _prepare_iter_population(
    *,
    draws: int,
    step,
    start: Sequence[PointType],
    parallelize: bool,
    traces: Sequence[BaseTrace],
    tune: int,
    rngs: Sequence[np.random.Generator],
    model=None,
    progressbar=True,
) -> Iterator[int]:
    """Prepare a PopulationStepper and traces for population sampling.

    Parameters
    ----------
    draws : int
        The number of samples to draw
    step : function
        Step function (should be or contain a population step method)
    start : list
        Start points for each chain
    parallelize : bool
        Setting for multiprocess parallelization
    tune : int
        Number of iterations to tune.
    rngs: sequence of random Generators
        A list of :py:class:`~numpy.random.Generator` objects, one for each chain
    model : Model (optional if in ``with`` context)
    progressbar : bool
        ``progressbar`` argument for the ``PopulationStepper``, (defaults to True)

    Returns
    -------
    _iter_population : generator
        Main sampling iterator yieling the iteration number.
    """
    nchains = len(start)
    model = modelcontext(model)
    draws = int(draws)

    if draws < 1:
        raise ValueError("Argument `draws` should be above 0.")

    # The initialization of traces, samplers and points must happen in the right order:
    # 1. population of points is created
    # 2. steppers are initialized and linked to the points object
    # 3. a PopulationStepper is configured for parallelized stepping

    # 1. create a population (points) that tracks each chain
    # it is updated as the chains are advanced
    population = [start[c] for c in range(nchains)]

    # 2. Set up the steppers
    steppers: list[Step] = []
    assert (
        len(rngs) == nchains
    ), f"There must be one random Generator per chain. Got {len(rngs)} instead of {nchains}"
    for c, rng in enumerate(rngs):
        # need independent samplers for each chain
        # it is important to copy the actual steppers (but not the delta_logp)
        if isinstance(step, CompoundStep):
            chainstep = CompoundStep([copy(m) for m in step.methods])
        else:
            chainstep = copy(step)
        chainstep.set_rng(rng)
        # link population samplers to the shared population state
        for sm in chainstep.methods if isinstance(step, CompoundStep) else [chainstep]:
            if isinstance(sm, PopulationArrayStepShared):
                sm.link_population(population, c)
        steppers.append(chainstep)

    # 3. configure the PopulationStepper (expensive call)
    popstep = PopulationStepper(steppers, parallelize, progressbar=progressbar)

    # Because the preparations above are expensive, the actual iterator is
    # in another method. This way the progbar will not be disturbed.
    return _iter_population(
        draws=draws, tune=tune, popstep=popstep, steppers=steppers, traces=traces, points=population
    )


def _iter_population(
    *,
    draws: int,
    tune: int,
    popstep: PopulationStepper,
    steppers,
    traces: Sequence[BaseTrace],
    points,
) -> Iterator[int]:
    """Iterate a ``PopulationStepper``.

    Parameters
    ----------
    draws : int
        number of draws per chain
    tune : int
        number of tuning steps
    popstep : PopulationStepper
        the helper object for (parallelized) stepping of chains
    steppers : list
        The step methods for each chain
    traces : list
        Traces for each chain
    points : list
        population of chain states

    Yields
    ------
    i
        Iteration number.
    """
    try:
        with popstep:
            # iterate draws of all chains
            for i in range(draws):
                # this call steps all chains and returns a list of (point, stats)
                # the `popstep` may interact with subprocesses internally
                updates = popstep.step(i == tune, points)

                # apply the update to the points and record to the traces
                for c, strace in enumerate(traces):
                    points[c], stats = updates[c]
                    strace.record(points[c], stats)
                    log_warning_stats(stats)
                # yield the state of all chains in parallel
                yield i
    except KeyboardInterrupt:
        for c, strace in enumerate(traces):
            strace.close()
            if hasattr(steppers[c], "report"):
                steppers[c].report._finalize(strace)
        raise
    except BaseException:
        for c, strace in enumerate(traces):
            strace.close()
        raise
    else:
        for c, strace in enumerate(traces):
            strace.close()
            if hasattr(steppers[c], "report"):
                steppers[c].report._finalize(strace)
