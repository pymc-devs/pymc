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

"""Functions for MCMC sampling."""

import collections.abc as abc
import logging
import pickle
import sys
import time
import warnings

from collections import defaultdict
from copy import copy
from typing import Any, Dict, Iterable, List, Optional, Set, Union, cast

import arviz
import numpy as np
import packaging
import theano.gradient as tg
import xarray

from arviz import InferenceData
from fastprogress.fastprogress import progress_bar

import pymc3 as pm

from pymc3.backends.base import BaseTrace, MultiTrace
from pymc3.backends.ndarray import NDArray
from pymc3.distributions.distribution import draw_values
from pymc3.distributions.posterior_predictive import fast_sample_posterior_predictive
from pymc3.exceptions import IncorrectArgumentsError, SamplingError
from pymc3.model import Model, Point, all_continuous, modelcontext
from pymc3.parallel_sampling import Draw, _cpu_count
from pymc3.step_methods import (
    NUTS,
    PGBART,
    BinaryGibbsMetropolis,
    BinaryMetropolis,
    CategoricalGibbsMetropolis,
    CompoundStep,
    DEMetropolis,
    HamiltonianMC,
    Metropolis,
    Slice,
)
from pymc3.step_methods.arraystep import BlockedStep, PopulationArrayStepShared
from pymc3.step_methods.hmc import quadpotential
from pymc3.util import (
    chains_and_samples,
    check_start_vals,
    dataset_to_point_list,
    get_default_varnames,
    get_untransformed_name,
    is_transformed_name,
    update_start_vals,
)
from pymc3.vartypes import discrete_types

sys.setrecursionlimit(10000)

__all__ = [
    "sample",
    "iter_sample",
    "sample_posterior_predictive",
    "sample_posterior_predictive_w",
    "init_nuts",
    "sample_prior_predictive",
    "fast_sample_posterior_predictive",
]

STEP_METHODS = (
    NUTS,
    HamiltonianMC,
    Metropolis,
    BinaryMetropolis,
    BinaryGibbsMetropolis,
    Slice,
    CategoricalGibbsMetropolis,
    PGBART,
)
Step = Union[BlockedStep, CompoundStep]

ArrayLike = Union[np.ndarray, List[float]]
PointType = Dict[str, np.ndarray]
PointList = List[PointType]
Backend = Union[BaseTrace, MultiTrace, NDArray]

_log = logging.getLogger("pymc3")


def instantiate_steppers(
    _model, steps: List[Step], selected_steps, step_kwargs=None
) -> Union[Step, List[Step]]:
    """Instantiate steppers assigned to the model variables.

    This function is intended to be called automatically from ``sample()``, but
    may be called manually.

    Parameters
    ----------
    model : Model object
        A fully-specified model object; legacy argument -- ignored
    steps : list
        A list of zero or more step function instances that have been assigned to some subset of
        the model's parameters.
    selected_steps : dict
        A dictionary that maps a step method class to a list of zero or more model variables.
    step_kwargs : dict
        Parameters for the samplers. Keys are the lower case names of
        the step method, values a dict of arguments. Defaults to None.

    Returns
    -------
    methods : list or step
        List of step methods associated with the model's variables, or step method
        if there is only one.
    """
    if step_kwargs is None:
        step_kwargs = {}

    used_keys = set()
    for step_class, vars in selected_steps.items():
        if vars:
            args = step_kwargs.get(step_class.name, {})
            used_keys.add(step_class.name)
            step = step_class(vars=vars, **args)
            steps.append(step)

    unused_args = set(step_kwargs).difference(used_keys)
    if unused_args:
        raise ValueError("Unused step method arguments: %s" % unused_args)

    if len(steps) == 1:
        return steps[0]

    return steps


def assign_step_methods(model, step=None, methods=STEP_METHODS, step_kwargs=None):
    """Assign model variables to appropriate step methods.

    Passing a specified model will auto-assign its constituent stochastic
    variables to step methods based on the characteristics of the variables.
    This function is intended to be called automatically from ``sample()``, but
    may be called manually. Each step method passed should have a
    ``competence()`` method that returns an ordinal competence value
    corresponding to the variable passed to it. This value quantifies the
    appropriateness of the step method for sampling the variable.

    Parameters
    ----------
    model : Model object
        A fully-specified model object
    step : step function or vector of step functions
        One or more step functions that have been assigned to some subset of
        the model's parameters. Defaults to ``None`` (no assigned variables).
    methods : vector of step method classes
        The set of step methods from which the function may choose. Defaults
        to the main step methods provided by PyMC3.
    step_kwargs : dict
        Parameters for the samplers. Keys are the lower case names of
        the step method, values a dict of arguments.

    Returns
    -------
    methods : list
        List of step methods associated with the model's variables.
    """
    steps = []
    assigned_vars = set()

    if step is not None:
        try:
            steps += list(step)
        except TypeError:
            steps.append(step)
        for step in steps:
            try:
                assigned_vars = assigned_vars.union(set(step.vars))
            except AttributeError:
                for method in step.methods:
                    assigned_vars = assigned_vars.union(set(method.vars))

    # Use competence classmethods to select step methods for remaining
    # variables
    selected_steps = defaultdict(list)
    for var in model.free_RVs:
        if var not in assigned_vars:
            # determine if a gradient can be computed
            has_gradient = var.dtype not in discrete_types
            if has_gradient:
                try:
                    tg.grad(model.logpt, var)
                except (AttributeError, NotImplementedError, tg.NullTypeGradError):
                    has_gradient = False
            # select the best method
            selected = max(
                methods,
                key=lambda method, var=var, has_gradient=has_gradient: method._competence(
                    var, has_gradient
                ),
            )
            selected_steps[selected].append(var)

    return instantiate_steppers(model, steps, selected_steps, step_kwargs)


def _print_step_hierarchy(s: Step, level=0) -> None:
    if isinstance(s, CompoundStep):
        _log.info(">" * level + "CompoundStep")
        for i in s.methods:
            _print_step_hierarchy(i, level + 1)
    else:
        varnames = ", ".join(
            [
                get_untransformed_name(v.name) if is_transformed_name(v.name) else v.name
                for v in s.vars
            ]
        )
        _log.info(">" * level + f"{s.__class__.__name__}: [{varnames}]")


def sample(
    draws=1000,
    step=None,
    init="auto",
    n_init=200000,
    start=None,
    trace=None,
    chain_idx=0,
    chains=None,
    cores=None,
    tune=1000,
    progressbar=True,
    model=None,
    random_seed=None,
    discard_tuned_samples=True,
    compute_convergence_checks=True,
    callback=None,
    jitter_max_retries=10,
    *,
    return_inferencedata=None,
    idata_kwargs: dict = None,
    mp_ctx=None,
    pickle_backend: str = "pickle",
    **kwargs,
):
    r"""Draw samples from the posterior using the given step methods.

    Multiple step methods are supported via compound step methods.

    Parameters
    ----------
    draws : int
        The number of samples to draw. Defaults to 1000. The number of tuned samples are discarded
        by default. See ``discard_tuned_samples``.
    init : str
        Initialization method to use for auto-assigned NUTS samplers.

        * auto: Choose a default initialization method automatically.
          Currently, this is ``jitter+adapt_diag``, but this can change in the future.
          If you depend on the exact behaviour, choose an initialization method explicitly.
        * adapt_diag: Start with a identity mass matrix and then adapt a diagonal based on the
          variance of the tuning samples. All chains use the test value (usually the prior mean)
          as starting point.
        * jitter+adapt_diag: Same as ``adapt_diag``, but add uniform jitter in [-1, 1] to the
          starting point in each chain.
        * advi+adapt_diag: Run ADVI and then adapt the resulting diagonal mass matrix based on the
          sample variance of the tuning samples.
        * advi+adapt_diag_grad: Run ADVI and then adapt the resulting diagonal mass matrix based
          on the variance of the gradients during tuning. This is **experimental** and might be
          removed in a future release.
        * advi: Run ADVI to estimate posterior mean and diagonal mass matrix.
        * advi_map: Initialize ADVI with MAP and use MAP as starting point.
        * map: Use the MAP as starting point. This is discouraged.
        * adapt_full: Adapt a dense mass matrix using the sample covariances

    step : function or iterable of functions
        A step function or collection of functions. If there are variables without step methods,
        step methods for those variables will be assigned automatically.  By default the NUTS step
        method will be used, if appropriate to the model; this is a good default for beginning
        users.
    n_init : int
        Number of iterations of initializer. Only works for 'ADVI' init methods.
    start : dict, or array of dict
        Starting point in parameter space (or partial point)
        Defaults to ``trace.point(-1))`` if there is a trace provided and model.test_point if not
        (defaults to empty dict). Initialization methods for NUTS (see ``init`` keyword) can
        overwrite the default.
    trace : backend, list, or MultiTrace
        This should be a backend instance, a list of variables to track, or a MultiTrace object
        with past values. If a MultiTrace object is given, it must contain samples for the chain
        number ``chain``. If None or a list of variables, the NDArray backend is used.
    chain_idx : int
        Chain number used to store sample in backend. If ``chains`` is greater than one, chain
        numbers will start here.
    chains : int
        The number of chains to sample. Running independent chains is important for some
        convergence statistics and can also reveal multiple modes in the posterior. If ``None``,
        then set to either ``cores`` or 2, whichever is larger.
    cores : int
        The number of chains to run in parallel. If ``None``, set to the number of CPUs in the
        system, but at most 4.
    tune : int
        Number of iterations to tune, defaults to 1000. Samplers adjust the step sizes, scalings or
        similar during tuning. Tuning samples will be drawn in addition to the number specified in
        the ``draws`` argument, and will be discarded unless ``discard_tuned_samples`` is set to
        False.
    progressbar : bool, optional default=True
        Whether or not to display a progress bar in the command line. The bar shows the percentage
        of completion, the sampling speed in samples per second (SPS), and the estimated remaining
        time until completion ("expected time of arrival"; ETA).
    model : Model (optional if in ``with`` context)
    random_seed : int or list of ints
        A list is accepted if ``cores`` is greater than one.
    discard_tuned_samples : bool
        Whether to discard posterior samples of the tune interval.
    compute_convergence_checks : bool, default=True
        Whether to compute sampler statistics like Gelman-Rubin and ``effective_n``.
    callback : function, default=None
        A function which gets called for every sample from the trace of a chain. The function is
        called with the trace and the current draw and will contain all samples for a single trace.
        the ``draw.chain`` argument can be used to determine which of the active chains the sample
        is drawn from.
        Sampling can be interrupted by throwing a ``KeyboardInterrupt`` in the callback.
    jitter_max_retries : int
        Maximum number of repeated attempts (per chain) at creating an initial matrix with uniform jitter
        that yields a finite probability. This applies to ``jitter+adapt_diag`` and ``jitter+adapt_full``
        init methods.
    return_inferencedata : bool, default=False
        Whether to return the trace as an :class:`arviz:arviz.InferenceData` (True) object or a `MultiTrace` (False)
        Defaults to `False`, but we'll switch to `True` in an upcoming release.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`arviz:arviz.from_pymc3`
    mp_ctx : multiprocessing.context.BaseContent
        A multiprocessing context for parallel sampling. See multiprocessing
        documentation for details.
    pickle_backend : str
        One of `'pickle'` or `'dill'`. The library used to pickle models
        in parallel sampling if the multiprocessing context is not of type
        `fork`.

    Returns
    -------
    trace : pymc3.backends.base.MultiTrace or arviz.InferenceData
        A ``MultiTrace`` or ArviZ ``InferenceData`` object that contains the samples.

    Notes
    -----
    Optional keyword arguments can be passed to ``sample`` to be delivered to the
    ``step_method``\ s used during sampling.

    If your model uses only one step method, you can address step method kwargs
    directly. In particular, the NUTS step method has several options including:

        * target_accept : float in [0, 1]. The step size is tuned such that we
          approximate this acceptance rate. Higher values like 0.9 or 0.95 often
          work better for problematic posteriors
        * max_treedepth : The maximum depth of the trajectory tree
        * step_scale : float, default 0.25
          The initial guess for the step size scaled down by :math:`1/n**(1/4)`

    If your model uses multiple step methods, aka a Compound Step, then you have
    two ways to address arguments to each step method:

    A. If you let ``sample()`` automatically assign the ``step_method``\ s,
       and you can correctly anticipate what they will be, then you can wrap
       step method kwargs in a dict and pass that to sample() with a kwarg set
       to the name of the step method.
       e.g. for a CompoundStep comprising NUTS and BinaryGibbsMetropolis,
       you could send:

       1. ``target_accept`` to NUTS: nuts={'target_accept':0.9}
       2. ``transit_p`` to BinaryGibbsMetropolis: binary_gibbs_metropolis={'transit_p':.7}

       Note that available names are:

        ``nuts``, ``hmc``, ``metropolis``, ``binary_metropolis``,
        ``binary_gibbs_metropolis``, ``categorical_gibbs_metropolis``,
        ``DEMetropolis``, ``DEMetropolisZ``, ``slice``

    B. If you manually declare the ``step_method``\ s, within the ``step``
       kwarg, then you can address the ``step_method`` kwargs directly.
       e.g. for a CompoundStep comprising NUTS and BinaryGibbsMetropolis,
       you could send ::

        step=[pm.NUTS([freeRV1, freeRV2], target_accept=0.9),
              pm.BinaryGibbsMetropolis([freeRV3], transit_p=.7)]

    You can find a full list of arguments in the docstring of the step methods.

    Examples
    --------
    .. code:: ipython

        In [1]: import pymc3 as pm
           ...: n = 100
           ...: h = 61
           ...: alpha = 2
           ...: beta = 2

        In [2]: with pm.Model() as model: # context management
           ...:     p = pm.Beta("p", alpha=alpha, beta=beta)
           ...:     y = pm.Binomial("y", n=n, p=p, observed=h)
           ...:     trace = pm.sample()

        In [3]: az.summary(trace, kind="stats")

        Out[3]:
            mean     sd  hdi_3%  hdi_97%
        p  0.609  0.047   0.528    0.699
    """
    model = modelcontext(model)
    if start is None:
        check_start_vals(model.test_point, model)
    else:
        if isinstance(start, dict):
            update_start_vals(start, model.test_point, model)
        else:
            for chain_start_vals in start:
                update_start_vals(chain_start_vals, model.test_point, model)
        check_start_vals(start, model)

    if cores is None:
        cores = min(4, _cpu_count())

    if chains is None:
        chains = max(2, cores)
    if isinstance(start, dict):
        start = [start] * chains
    if random_seed == -1:
        random_seed = None
    if chains == 1 and isinstance(random_seed, int):
        random_seed = [random_seed]
    if random_seed is None or isinstance(random_seed, int):
        if random_seed is not None:
            np.random.seed(random_seed)
        random_seed = [np.random.randint(2 ** 30) for _ in range(chains)]
    if not isinstance(random_seed, abc.Iterable):
        raise TypeError("Invalid value for `random_seed`. Must be tuple, list or int")

    if not discard_tuned_samples and not return_inferencedata:
        warnings.warn(
            "Tuning samples will be included in the returned `MultiTrace` object, which can lead to"
            " complications in your downstream analysis. Please consider to switch to `InferenceData`:\n"
            "`pm.sample(..., return_inferencedata=True)`",
            UserWarning,
        )

    if return_inferencedata is None:
        v = packaging.version.parse(pm.__version__)
        if v.release[0] > 3 or v.release[1] >= 10:  # type: ignore
            warnings.warn(
                "In an upcoming release, pm.sample will return an `arviz.InferenceData` object instead of a `MultiTrace` by default. "
                "You can pass return_inferencedata=True or return_inferencedata=False to be safe and silence this warning.",
                FutureWarning,
            )
        # set the default
        return_inferencedata = False

    if start is not None:
        for start_vals in start:
            _check_start_shape(model, start_vals)

    # small trace warning
    if draws == 0:
        msg = "Tuning was enabled throughout the whole trace."
        _log.warning(msg)
    elif draws < 500:
        msg = "Only %s samples in chain." % draws
        _log.warning(msg)

    draws += tune

    if model.ndim == 0:
        raise ValueError("The model does not contain any free variables.")

    if step is None and init is not None and all_continuous(model.vars):
        try:
            # By default, try to use NUTS
            _log.info("Auto-assigning NUTS sampler...")
            start_, step = init_nuts(
                init=init,
                chains=chains,
                n_init=n_init,
                model=model,
                random_seed=random_seed,
                progressbar=progressbar,
                jitter_max_retries=jitter_max_retries,
                **kwargs,
            )
            if start is None:
                start = start_
                check_start_vals(start, model)
        except (AttributeError, NotImplementedError, tg.NullTypeGradError):
            # gradient computation failed
            _log.info("Initializing NUTS failed. " "Falling back to elementwise auto-assignment.")
            _log.debug("Exception in init nuts", exec_info=True)
            step = assign_step_methods(model, step, step_kwargs=kwargs)
    else:
        step = assign_step_methods(model, step, step_kwargs=kwargs)

    if isinstance(step, list):
        step = CompoundStep(step)
    if start is None:
        start = {}
    if isinstance(start, dict):
        start = [start] * chains

    sample_args = {
        "draws": draws,
        "step": step,
        "start": start,
        "trace": trace,
        "chain": chain_idx,
        "chains": chains,
        "tune": tune,
        "progressbar": progressbar,
        "model": model,
        "random_seed": random_seed,
        "cores": cores,
        "callback": callback,
        "discard_tuned_samples": discard_tuned_samples,
    }
    parallel_args = {
        "pickle_backend": pickle_backend,
        "mp_ctx": mp_ctx,
    }

    sample_args.update(kwargs)

    has_population_samplers = np.any(
        [
            isinstance(m, PopulationArrayStepShared)
            for m in (step.methods if isinstance(step, CompoundStep) else [step])
        ]
    )

    parallel = cores > 1 and chains > 1 and not has_population_samplers
    t_start = time.time()
    if parallel:
        _log.info(f"Multiprocess sampling ({chains} chains in {cores} jobs)")
        _print_step_hierarchy(step)
        try:
            trace = _mp_sample(**sample_args, **parallel_args)
        except pickle.PickleError:
            _log.warning("Could not pickle model, sampling singlethreaded.")
            _log.debug("Pickling error:", exec_info=True)
            parallel = False
        except AttributeError as e:
            if not str(e).startswith("AttributeError: Can't pickle"):
                raise
            _log.warning("Could not pickle model, sampling singlethreaded.")
            _log.debug("Pickling error:", exec_info=True)
            parallel = False
    if not parallel:
        if has_population_samplers:
            has_demcmc = np.any(
                [
                    isinstance(m, DEMetropolis)
                    for m in (step.methods if isinstance(step, CompoundStep) else [step])
                ]
            )
            _log.info(f"Population sampling ({chains} chains)")
            if has_demcmc and chains < 3:
                raise ValueError(
                    "DEMetropolis requires at least 3 chains. "
                    "For this {}-dimensional model you should use ≥{} chains".format(
                        model.ndim, model.ndim + 1
                    )
                )
            if has_demcmc and chains <= model.ndim:
                warnings.warn(
                    "DEMetropolis should be used with more chains than dimensions! "
                    "(The model has {} dimensions.)".format(model.ndim),
                    UserWarning,
                )
            _print_step_hierarchy(step)
            trace = _sample_population(parallelize=cores > 1, **sample_args)
        else:
            _log.info(f"Sequential sampling ({chains} chains in 1 job)")
            _print_step_hierarchy(step)
            trace = _sample_many(**sample_args)

    t_sampling = time.time() - t_start
    # count the number of tune/draw iterations that happened
    # ideally via the "tune" statistic, but not all samplers record it!
    if "tune" in trace.stat_names:
        stat = trace.get_sampler_stats("tune", chains=0)
        # when CompoundStep is used, the stat is 2 dimensional!
        if len(stat.shape) == 2:
            stat = stat[:, 0]
        stat = tuple(stat)
        n_tune = stat.count(True)
        n_draws = stat.count(False)
    else:
        # these may be wrong when KeyboardInterrupt happened, but they're better than nothing
        n_tune = min(tune, len(trace))
        n_draws = max(0, len(trace) - n_tune)

    if discard_tuned_samples:
        trace = trace[n_tune:]

    # save metadata in SamplerReport
    trace.report._n_tune = n_tune
    trace.report._n_draws = n_draws
    trace.report._t_sampling = t_sampling

    if "variable_inclusion" in trace.stat_names:
        variable_inclusion = np.stack(trace.get_sampler_stats("variable_inclusion")).mean(0)
        trace.report.variable_importance = variable_inclusion / variable_inclusion.sum()

    n_chains = len(trace.chains)
    _log.info(
        f'Sampling {n_chains} chain{"s" if n_chains > 1 else ""} for {n_tune:_d} tune and {n_draws:_d} draw iterations '
        f"({n_tune*n_chains:_d} + {n_draws*n_chains:_d} draws total) "
        f"took {trace.report.t_sampling:.0f} seconds."
    )

    idata = None
    if compute_convergence_checks or return_inferencedata:
        ikwargs = dict(model=model, save_warmup=not discard_tuned_samples)
        if idata_kwargs:
            ikwargs.update(idata_kwargs)
        idata = arviz.from_pymc3(trace, **ikwargs)

    if compute_convergence_checks:
        if draws - tune < 100:
            warnings.warn("The number of samples is too small to check convergence reliably.")
        else:
            trace.report._run_convergence_checks(idata, model)
    trace.report._log_summary()

    if return_inferencedata:
        return idata
    else:
        return trace


def _check_start_shape(model, start):
    if not isinstance(start, dict):
        raise TypeError("start argument must be a dict or an array-like of dicts")
    e = ""
    for var in model.vars:
        if var.name in start.keys():
            var_shape = var.shape.tag.test_value
            start_var_shape = np.shape(start[var.name])
            if start_var_shape:
                if not np.array_equal(var_shape, start_var_shape):
                    e += "\nExpected shape {} for var '{}', got: {}".format(
                        tuple(var_shape), var.name, start_var_shape
                    )
            # if start var has no shape
            else:
                # if model var has a specified shape
                if var_shape.size > 0:
                    e += "\nExpected shape {} for var " "'{}', got scalar {}".format(
                        tuple(var_shape), var.name, start[var.name]
                    )

    if e != "":
        raise ValueError(f"Bad shape for start argument:{e}")


def _sample_many(
    draws,
    chain: int,
    chains: int,
    start: list,
    random_seed: list,
    step,
    callback=None,
    **kwargs,
):
    """Samples all chains sequentially.

    Parameters
    ----------
    draws: int
        The number of samples to draw
    chain: int
        Number of the first chain in the sequence.
    chains: int
        Total number of chains to sample.
    start: list
        Starting points for each chain
    random_seed: list
        A list of seeds, one for each chain
    step: function
        Step function

    Returns
    -------
    trace: MultiTrace
        Contains samples of all chains
    """
    traces: List[Backend] = []
    for i in range(chains):
        trace = _sample(
            draws=draws,
            chain=chain + i,
            start=start[i],
            step=step,
            random_seed=random_seed[i],
            callback=callback,
            **kwargs,
        )
        if trace is None:
            if len(traces) == 0:
                raise ValueError("Sampling stopped before a sample was created.")
            else:
                break
        elif len(trace) < draws:
            if len(traces) == 0:
                traces.append(trace)
            break
        else:
            traces.append(trace)
    return MultiTrace(traces)


def _sample_population(
    draws: int,
    chain: int,
    chains: int,
    start,
    random_seed,
    step,
    tune,
    model,
    progressbar: bool = True,
    parallelize=False,
    **kwargs,
):
    """Performs sampling of a population of chains using the ``PopulationStepper``.

    Parameters
    ----------
    draws : int
        The number of samples to draw
    chain : int
        The number of the first chain in the population
    chains : int
        The total number of chains in the population
    start : list
        Start points for each chain
    random_seed : int or list of ints, optional
        A list is accepted if more if ``cores`` is greater than one.
    step : function
        Step function (should be or contain a population step method)
    tune : int, optional
        Number of iterations to tune, if applicable (defaults to None)
    model : Model (optional if in ``with`` context)
    progressbar : bool
        Show progress bars? (defaults to True)
    parallelize : bool
        Setting for multiprocess parallelization

    Returns
    -------
    trace : MultiTrace
        Contains samples of all chains
    """
    sampling = _prepare_iter_population(
        draws,
        [chain + c for c in range(chains)],
        step,
        start,
        parallelize,
        tune=tune,
        model=model,
        random_seed=random_seed,
        progressbar=progressbar,
    )

    if progressbar:
        sampling = progress_bar(sampling, total=draws, display=progressbar)

    latest_traces = None
    for it, traces in enumerate(sampling):
        latest_traces = traces
    return MultiTrace(latest_traces)


def _sample(
    chain: int,
    progressbar: bool,
    random_seed,
    start,
    draws: int,
    step=None,
    trace=None,
    tune=None,
    model: Optional[Model] = None,
    callback=None,
    **kwargs,
):
    """Main iteration for singleprocess sampling.

    Multiple step methods are supported via compound step methods.

    Parameters
    ----------
    chain : int
        Number of the chain that the samples will belong to.
    progressbar : bool
        Whether or not to display a progress bar in the command line. The bar shows the percentage
        of completion, the sampling speed in samples per second (SPS), and the estimated remaining
        time until completion ("expected time of arrival"; ETA).
    random_seed : int or list of ints
        A list is accepted if ``cores`` is greater than one.
    start : dict
        Starting point in parameter space (or partial point)
    draws : int
        The number of samples to draw
    step : function
        Step function
    trace : backend, list, or MultiTrace
        This should be a backend instance, a list of variables to track, or a MultiTrace object
        with past values. If a MultiTrace object is given, it must contain samples for the chain
        number ``chain``. If None or a list of variables, the NDArray backend is used.
    tune : int, optional
        Number of iterations to tune, if applicable (defaults to None)
    model : Model (optional if in ``with`` context)

    Returns
    -------
    strace : pymc3.backends.base.BaseTrace
        A ``BaseTrace`` object that contains the samples for this chain.
    """
    skip_first = kwargs.get("skip_first", 0)

    sampling = _iter_sample(draws, step, start, trace, chain, tune, model, random_seed, callback)
    _pbar_data = {"chain": chain, "divergences": 0}
    _desc = "Sampling chain {chain:d}, {divergences:,d} divergences"
    if progressbar:
        sampling = progress_bar(sampling, total=draws, display=progressbar)
        sampling.comment = _desc.format(**_pbar_data)
    try:
        strace = None
        for it, (strace, diverging) in enumerate(sampling):
            if it >= skip_first and diverging:
                _pbar_data["divergences"] += 1
                if progressbar:
                    sampling.comment = _desc.format(**_pbar_data)
    except KeyboardInterrupt:
        pass
    return strace


def iter_sample(
    draws: int,
    step,
    start: Optional[Dict[Any, Any]] = None,
    trace=None,
    chain=0,
    tune: Optional[int] = None,
    model: Optional[Model] = None,
    random_seed: Optional[Union[int, List[int]]] = None,
    callback=None,
):
    """Generate a trace on each iteration using the given step method.

    Multiple step methods ared supported via compound step methods.  Returns the
    amount of time taken.

    Parameters
    ----------
    draws : int
        The number of samples to draw
    step : function
        Step function
    start : dict
        Starting point in parameter space (or partial point). Defaults to trace.point(-1)) if
        there is a trace provided and model.test_point if not (defaults to empty dict)
    trace : backend, list, or MultiTrace
        This should be a backend instance, a list of variables to track, or a MultiTrace object
        with past values. If a MultiTrace object is given, it must contain samples for the chain
        number ``chain``. If None or a list of variables, the NDArray backend is used.
    chain : int, optional
        Chain number used to store sample in backend. If ``cores`` is greater than one, chain numbers
        will start here.
    tune : int, optional
        Number of iterations to tune, if applicable (defaults to None)
    model : Model (optional if in ``with`` context)
    random_seed : int or list of ints, optional
        A list is accepted if more if ``cores`` is greater than one.
    callback :
        A function which gets called for every sample from the trace of a chain. The function is
        called with the trace and the current draw and will contain all samples for a single trace.
        the ``draw.chain`` argument can be used to determine which of the active chains the sample
        is drawn from.
        Sampling can be interrupted by throwing a ``KeyboardInterrupt`` in the callback.

    Yields
    ------
    trace : MultiTrace
        Contains all samples up to the current iteration

    Examples
    --------
    ::

        for trace in iter_sample(500, step):
            ...
    """
    sampling = _iter_sample(draws, step, start, trace, chain, tune, model, random_seed, callback)
    for i, (strace, _) in enumerate(sampling):
        yield MultiTrace([strace[: i + 1]])


def _iter_sample(
    draws,
    step,
    start=None,
    trace=None,
    chain=0,
    tune=None,
    model=None,
    random_seed=None,
    callback=None,
):
    """Generator for sampling one chain. (Used in singleprocess sampling.)

    Parameters
    ----------
    draws : int
        The number of samples to draw
    step : function
        Step function
    start : dict, optional
        Starting point in parameter space (or partial point). Defaults to trace.point(-1)) if
        there is a trace provided and model.test_point if not (defaults to empty dict)
    trace : backend, list, MultiTrace, or None
        This should be a backend instance, a list of variables to track, or a MultiTrace object
        with past values. If a MultiTrace object is given, it must contain samples for the chain
        number ``chain``. If None or a list of variables, the NDArray backend is used.
    chain : int, optional
        Chain number used to store sample in backend. If ``cores`` is greater than one, chain numbers
        will start here.
    tune : int, optional
        Number of iterations to tune, if applicable (defaults to None)
    model : Model (optional if in ``with`` context)
    random_seed : int or list of ints, optional
        A list is accepted if more if ``cores`` is greater than one.

    Yields
    ------
    strace : BaseTrace
        The trace object containing the samples for this chain
    diverging : bool
        Indicates if the draw is divergent. Only available with some samplers.
    """
    model = modelcontext(model)
    draws = int(draws)
    if random_seed is not None:
        np.random.seed(random_seed)
    if draws < 1:
        raise ValueError("Argument `draws` must be greater than 0.")

    if start is None:
        start = {}

    strace = _choose_backend(trace, chain, model=model)

    if len(strace) > 0:
        update_start_vals(start, strace.point(-1), model)
    else:
        update_start_vals(start, model.test_point, model)

    try:
        step = CompoundStep(step)
    except TypeError:
        pass

    point = Point(start, model=model)

    if step.generates_stats and strace.supports_sampler_stats:
        strace.setup(draws, chain, step.stats_dtypes)
    else:
        strace.setup(draws, chain)

    try:
        step.tune = bool(tune)
        if hasattr(step, "reset_tuning"):
            step.reset_tuning()
        for i in range(draws):
            stats = None
            diverging = False

            if i == 0 and hasattr(step, "iter_count"):
                step.iter_count = 0
            if i == tune:
                step = stop_tuning(step)
            if step.generates_stats:
                point, stats = step.step(point)
                if strace.supports_sampler_stats:
                    strace.record(point, stats)
                    diverging = i > tune and stats and stats[0].get("diverging")
                else:
                    strace.record(point)
            else:
                point = step.step(point)
                strace.record(point)
            if callback is not None:
                warns = getattr(step, "warnings", None)
                callback(
                    trace=strace,
                    draw=Draw(chain, i == draws, i, i < tune, stats, point, warns),
                )

            yield strace, diverging
    except KeyboardInterrupt:
        strace.close()
        if hasattr(step, "warnings"):
            warns = step.warnings()
            strace._add_warnings(warns)
        raise
    except BaseException:
        strace.close()
        raise
    else:
        strace.close()
        if hasattr(step, "warnings"):
            warns = step.warnings()
            strace._add_warnings(warns)


class PopulationStepper:
    """Wraps population of step methods to step them in parallel with single or multiprocessing."""

    def __init__(self, steppers, parallelize, progressbar=True):
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
        if parallelize:
            try:
                # configure a child process for each stepper
                _log.info(
                    "Attempting to parallelize chains to all cores. You can turn this off with `pm.sample(cores=1)`."
                )
                import multiprocessing

                for c, stepper in (
                    enumerate(progress_bar(steppers)) if progressbar else enumerate(steppers)
                ):
                    secondary_end, primary_end = multiprocessing.Pipe()
                    stepper_dumps = pickle.dumps(stepper, protocol=4)
                    process = multiprocessing.Process(
                        target=self.__class__._run_secondary,
                        args=(c, stepper_dumps, secondary_end),
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
                _log.debug("Error was: ", exec_info=True)
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
    def _run_secondary(c, stepper_dumps, secondary_end):
        """This method is started on a separate process to perform stepping of a chain.

        Parameters
        ----------
        c : int
            number of this chain
        stepper : BlockedStep
            a step method such as CompoundStep
        secondary_end : multiprocessing.connection.PipeConnection
            This is our connection to the main process
        """
        # re-seed each child process to make them unique
        np.random.seed(None)
        try:
            stepper = pickle.loads(stepper_dumps)
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
                    stop_tuning(stepper)
                # forward the population to the PopulationArrayStepShared objects
                # This is necessary because due to the process fork, the population
                # object is no longer shared between the steppers.
                for popstep in population_steppers:
                    popstep.population = population
                update = stepper.step(population[c])
                secondary_end.send(update)
        except Exception:
            _log.exception(f"ChainWalker{c}")
        return

    def step(self, tune_stop, population):
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
        updates = [None] * self.nchains
        if self.is_parallelized:
            for c in range(self.nchains):
                self._primary_ends[c].send((tune_stop, population))
            # Blockingly get the step outcomes
            for c in range(self.nchains):
                updates[c] = self._primary_ends[c].recv()
        else:
            for c in range(self.nchains):
                if tune_stop:
                    self._steppers[c] = stop_tuning(self._steppers[c])
                updates[c] = self._steppers[c].step(population[c])
        return updates


def _prepare_iter_population(
    draws: int,
    chains: list,
    step,
    start: list,
    parallelize: bool,
    tune=None,
    model=None,
    random_seed=None,
    progressbar=True,
):
    """Prepare a PopulationStepper and traces for population sampling.

    Parameters
    ----------
    draws : int
        The number of samples to draw
    chains : list
        The chain numbers in the population
    step : function
        Step function (should be or contain a population step method)
    start : list
        Start points for each chain
    parallelize : bool
        Setting for multiprocess parallelization
    tune : int, optional
        Number of iterations to tune, if applicable (defaults to None)
    model : Model (optional if in ``with`` context)
    random_seed : int or list of ints, optional
        A list is accepted if more if ``cores`` is greater than one.
    progressbar : bool
        ``progressbar`` argument for the ``PopulationStepper``, (defaults to True)

    Returns
    -------
    _iter_population : generator
        Yields traces of all chains at the same time
    """
    # chains contains the chain numbers, but for indexing we need indices...
    nchains = len(chains)
    model = modelcontext(model)
    draws = int(draws)
    if random_seed is not None:
        np.random.seed(random_seed)
    if draws < 1:
        raise ValueError("Argument `draws` should be above 0.")

    # The initialization of traces, samplers and points must happen in the right order:
    # 1. traces are initialized and update_start_vals configures variable transforms
    # 2. population of points is created
    # 3. steppers are initialized and linked to the points object
    # 4. traces are configured to track the sampler stats
    # 5. a PopulationStepper is configured for parallelized stepping

    # 1. prepare a BaseTrace for each chain
    traces = [_choose_backend(None, chain, model=model) for chain in chains]
    for c, strace in enumerate(traces):
        # initialize the trace size and variable transforms
        if len(strace) > 0:
            update_start_vals(start[c], strace.point(-1), model)
        else:
            update_start_vals(start[c], model.test_point, model)

    # 2. create a population (points) that tracks each chain
    # it is updated as the chains are advanced
    population = [Point(start[c], model=model) for c in range(nchains)]

    # 3. Set up the steppers
    steppers: List[Step] = []
    for c in range(nchains):
        # need indepenent samplers for each chain
        # it is important to copy the actual steppers (but not the delta_logp)
        if isinstance(step, CompoundStep):
            chainstep = CompoundStep([copy(m) for m in step.methods])
        else:
            chainstep = copy(step)
        # link population samplers to the shared population state
        for sm in chainstep.methods if isinstance(step, CompoundStep) else [chainstep]:
            if isinstance(sm, PopulationArrayStepShared):
                sm.link_population(population, c)
        steppers.append(chainstep)

    # 4. configure tracking of sampler stats
    for c in range(nchains):
        if steppers[c].generates_stats and traces[c].supports_sampler_stats:
            traces[c].setup(draws, c, steppers[c].stats_dtypes)
        else:
            traces[c].setup(draws, c)

    # 5. configure the PopulationStepper (expensive call)
    popstep = PopulationStepper(steppers, parallelize, progressbar=progressbar)

    # Because the preparations above are expensive, the actual iterator is
    # in another method. This way the progbar will not be disturbed.
    return _iter_population(draws, tune, popstep, steppers, traces, population)


def _iter_population(draws, tune, popstep, steppers, traces, points):
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
    traces : list
        List of trace objects of the individual chains
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
                    if steppers[c].generates_stats:
                        points[c], stats = updates[c]
                        if strace.supports_sampler_stats:
                            strace.record(points[c], stats)
                        else:
                            strace.record(points[c])
                    else:
                        points[c] = updates[c]
                        strace.record(points[c])
                # yield the state of all chains in parallel
                yield traces
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


def _choose_backend(trace, chain, **kwds) -> Backend:
    """Selects or creates a NDArray trace backend for a particular chain.

    Parameters
    ----------
    trace : BaseTrace, list, MultiTrace, or None
        This should be a BaseTrace, list of variables to track,
        or a MultiTrace object with past values.
        If a MultiTrace object is given, it must contain samples for the chain number ``chain``.
        If None or a list of variables, the NDArray backend is used.
    chain : int
        Number of the chain of interest.
    **kwds :
        keyword arguments to forward to the backend creation

    Returns
    -------
    trace : BaseTrace
        A trace object for the selected chain
    """
    if isinstance(trace, BaseTrace):
        return trace
    if isinstance(trace, MultiTrace):
        return trace._straces[chain]
    if trace is None:
        return NDArray(**kwds)

    return NDArray(vars=trace, **kwds)


def _mp_sample(
    draws: int,
    tune: int,
    step,
    chains: int,
    cores: int,
    chain: int,
    random_seed: list,
    start: list,
    progressbar=True,
    trace=None,
    model=None,
    callback=None,
    discard_tuned_samples=True,
    mp_ctx=None,
    pickle_backend="pickle",
    **kwargs,
):
    """Main iteration for multiprocess sampling.

    Parameters
    ----------
    draws : int
        The number of samples to draw
    tune : int, optional
        Number of iterations to tune, if applicable (defaults to None)
    step : function
        Step function
    chains : int
        The number of chains to sample.
    cores : int
        The number of chains to run in parallel.
    chain : int
        Number of the first chain.
    random_seed : list of ints
        Random seeds for each chain.
    start : list
        Starting points for each chain.
    progressbar : bool
        Whether or not to display a progress bar in the command line.
    trace : BaseTrace, list, MultiTrace or None
        This should be a backend instance, a list of variables to track, or a MultiTrace object
        with past values. If a MultiTrace object is given, it must contain samples for the chain
        number ``chain``. If None or a list of variables, the NDArray backend is used.
    model : Model (optional if in ``with`` context)
    callback : Callable
        A function which gets called for every sample from the trace of a chain. The function is
        called with the trace and the current draw and will contain all samples for a single trace.
        the ``draw.chain`` argument can be used to determine which of the active chains the sample
        is drawn from.
        Sampling can be interrupted by throwing a ``KeyboardInterrupt`` in the callback.

    Returns
    -------
    trace : pymc3.backends.base.MultiTrace
        A ``MultiTrace`` object that contains the samples for all chains.
    """
    import pymc3.parallel_sampling as ps

    # We did draws += tune in pm.sample
    draws -= tune

    traces = []
    for idx in range(chain, chain + chains):
        if trace is not None:
            strace = _choose_backend(copy(trace), idx, model=model)
        else:
            strace = _choose_backend(None, idx, model=model)
        # for user supply start value, fill-in missing value if the supplied
        # dict does not contain all parameters
        update_start_vals(start[idx - chain], model.test_point, model)
        if step.generates_stats and strace.supports_sampler_stats:
            strace.setup(draws + tune, idx + chain, step.stats_dtypes)
        else:
            strace.setup(draws + tune, idx + chain)
        traces.append(strace)

    sampler = ps.ParallelSampler(
        draws,
        tune,
        chains,
        cores,
        random_seed,
        start,
        step,
        chain,
        progressbar,
        mp_ctx=mp_ctx,
        pickle_backend=pickle_backend,
    )
    try:
        try:
            with sampler:
                for draw in sampler:
                    trace = traces[draw.chain - chain]
                    if trace.supports_sampler_stats and draw.stats is not None:
                        trace.record(draw.point, draw.stats)
                    else:
                        trace.record(draw.point)
                    if draw.is_last:
                        trace.close()
                        if draw.warnings is not None:
                            trace._add_warnings(draw.warnings)

                    if callback is not None:
                        callback(trace=trace, draw=draw)

        except ps.ParallelSamplingError as error:
            trace = traces[error._chain - chain]
            trace._add_warnings(error._warnings)
            for trace in traces:
                trace.close()

            multitrace = MultiTrace(traces)
            multitrace._report._log_summary()
            raise
        return MultiTrace(traces)
    except KeyboardInterrupt:
        if discard_tuned_samples:
            traces, length = _choose_chains(traces, tune)
        else:
            traces, length = _choose_chains(traces, 0)
        return MultiTrace(traces)[:length]
    finally:
        for trace in traces:
            trace.close()


def _choose_chains(traces, tune):
    """
    Filter and slice traces such that (n_traces * len(shortest_trace)) is maximized.

    We get here after a ``KeyboardInterrupt``, and so the different
    traces have different lengths. We therefore pick the number of
    traces such that (number of traces) * (length of shortest trace)
    is maximised.
    """
    if tune is None:
        tune = 0

    if not traces:
        return []

    lengths = [max(0, len(trace) - tune) for trace in traces]
    if not sum(lengths):
        raise ValueError("Not enough samples to build a trace.")

    idxs = np.argsort(lengths)
    l_sort = np.array(lengths)[idxs]

    use_until = np.argmax(l_sort * np.arange(1, l_sort.shape[0] + 1)[::-1])
    final_length = l_sort[use_until]

    return [traces[idx] for idx in idxs[use_until:]], final_length + tune


def stop_tuning(step):
    """Stop tuning the current step method."""
    step.stop_tuning()
    return step


class _DefaultTrace:
    """
    Utility for collecting samples into a dictionary.

    Name comes from its similarity to ``defaultdict``:
    entries are lazily created.

    Parameters
    ----------
    samples : int
        The number of samples that will be collected, per variable,
        into the trace.

    Attributes
    ----------
    trace_dict : Dict[str, np.ndarray]
        A dictionary constituting a trace.  Should be extracted
        after a procedure has filled the `_DefaultTrace` using the
        `insert()` method
    """

    trace_dict: Dict[str, np.ndarray] = {}
    _len: Optional[int] = None

    def __init__(self, samples: int):
        self._len = samples
        self.trace_dict = {}

    def insert(self, k: str, v, idx: int):
        """
        Insert `v` as the value of the `idx`th sample for the variable `k`.

        Parameters
        ----------
        k: str
            Name of the variable.
        v: anything that can go into a numpy array (including a numpy array)
            The value of the `idx`th sample from variable `k`
        ids: int
            The index of the sample we are inserting into the trace.
        """
        value_shape = np.shape(v)

        # initialize if necessary
        if k not in self.trace_dict:
            array_shape = (self._len,) + value_shape
            self.trace_dict[k] = np.empty(array_shape, dtype=np.array(v).dtype)

        # do the actual insertion
        if value_shape == ():
            self.trace_dict[k][idx] = v
        else:
            self.trace_dict[k][idx, :] = v


def sample_posterior_predictive(
    trace,
    samples: Optional[int] = None,
    model: Optional[Model] = None,
    var_names: Optional[List[str]] = None,
    size: Optional[int] = None,
    keep_size: Optional[bool] = False,
    random_seed=None,
    progressbar: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate posterior predictive samples from a model given a trace.

    Parameters
    ----------
    trace : backend, list, xarray.Dataset, arviz.InferenceData, or MultiTrace
        Trace generated from MCMC sampling, or a list of dicts (eg. points or from find_MAP()),
        or xarray.Dataset (eg. InferenceData.posterior or InferenceData.prior)
    samples : int
        Number of posterior predictive samples to generate. Defaults to one posterior predictive
        sample per posterior sample, that is, the number of draws times the number of chains. It
        is not recommended to modify this value; when modified, some chains may not be represented
        in the posterior predictive sample.
    model : Model (optional if in ``with`` context)
        Model used to generate ``trace``
    vars : iterable
        Variables for which to compute the posterior predictive samples.
        Deprecated: please use ``var_names`` instead.
    var_names : Iterable[str]
        Names of variables for which to compute the posterior predictive samples.
    size : int
        The number of random draws from the distribution specified by the parameters in each
        sample of the trace. Not recommended unless more than ndraws times nchains posterior
        predictive samples are needed.
    keep_size : bool, optional
        Force posterior predictive sample to have the same shape as posterior and sample stats
        data: ``(nchains, ndraws, ...)``. Overrides samples and size parameters.
    random_seed : int
        Seed for the random number generator.
    progressbar : bool
        Whether or not to display a progress bar in the command line. The bar shows the percentage
        of completion, the sampling speed in samples per second (SPS), and the estimated remaining
        time until completion ("expected time of arrival"; ETA).

    Returns
    -------
    samples : dict
        Dictionary with the variable names as keys, and values numpy arrays containing
        posterior predictive samples.
    """

    _trace: Union[MultiTrace, PointList]
    if isinstance(trace, InferenceData):
        _trace = dataset_to_point_list(trace.posterior)
    elif isinstance(trace, xarray.Dataset):
        _trace = dataset_to_point_list(trace)
    else:
        _trace = trace

    nchain: int
    len_trace: int
    if isinstance(trace, (InferenceData, xarray.Dataset)):
        nchain, len_trace = chains_and_samples(trace)
    else:
        len_trace = len(_trace)
        try:
            nchain = _trace.nchains
        except AttributeError:
            nchain = 1

    if keep_size and samples is not None:
        raise IncorrectArgumentsError("Should not specify both keep_size and samples arguments")
    if keep_size and size is not None:
        raise IncorrectArgumentsError("Should not specify both keep_size and size arguments")

    if samples is None:
        if isinstance(_trace, MultiTrace):
            samples = sum(len(v) for v in _trace._straces.values())
        elif isinstance(_trace, list) and all(isinstance(x, dict) for x in _trace):
            # this is a list of points
            samples = len(_trace)
        else:
            raise TypeError(
                "Do not know how to compute number of samples for trace argument of type %s"
                % type(_trace)
            )

    assert samples is not None
    if samples < len_trace * nchain:
        warnings.warn(
            "samples parameter is smaller than nchains times ndraws, some draws "
            "and/or chains may not be represented in the returned posterior "
            "predictive sample"
        )

    model = modelcontext(model)

    if model.potentials:
        warnings.warn(
            "The effect of Potentials on other parameters is ignored during posterior predictive sampling. "
            "This is likely to lead to invalid or biased predictive samples.",
            UserWarning,
        )

    if var_names is not None:
        vars_ = [model[x] for x in var_names]
    else:
        vars_ = model.observed_RVs

    if random_seed is not None:
        np.random.seed(random_seed)

    indices = np.arange(samples)

    if progressbar:
        indices = progress_bar(indices, total=samples, display=progressbar)

    ppc_trace_t = _DefaultTrace(samples)
    try:
        for idx in indices:
            if nchain > 1:
                # the trace object will either be a MultiTrace (and have _straces)...
                if hasattr(_trace, "_straces"):
                    chain_idx, point_idx = np.divmod(idx, len_trace)
                    param = cast(MultiTrace, _trace)._straces[chain_idx % nchain].point(point_idx)
                # ... or a PointList
                else:
                    param = cast(PointList, _trace)[idx % (len_trace * nchain)]
            # there's only a single chain, but the index might hit it multiple times if
            # the number of indices is greater than the length of the trace.
            else:
                param = _trace[idx % len_trace]

            values = draw_values(vars_, point=param, size=size)
            for k, v in zip(vars_, values):
                ppc_trace_t.insert(k.name, v, idx)
    except KeyboardInterrupt:
        pass

    ppc_trace = ppc_trace_t.trace_dict
    if keep_size:
        for k, ary in ppc_trace.items():
            ppc_trace[k] = ary.reshape((nchain, len_trace, *ary.shape[1:]))

    return ppc_trace


def sample_posterior_predictive_w(
    traces,
    samples: Optional[int] = None,
    models: Optional[List[Model]] = None,
    weights: Optional[ArrayLike] = None,
    random_seed: Optional[int] = None,
    progressbar: bool = True,
):
    """Generate weighted posterior predictive samples from a list of models and
    a list of traces according to a set of weights.

    Parameters
    ----------
    traces : list or list of lists
        List of traces generated from MCMC sampling (xarray.Dataset, arviz.InferenceData, or
        MultiTrace), or a list of list containing dicts from find_MAP() or points. The number of
        traces should be equal to the number of weights.
    samples : int, optional
        Number of posterior predictive samples to generate. Defaults to the
        length of the shorter trace in traces.
    models : list of Model
        List of models used to generate the list of traces. The number of models should be equal to
        the number of weights and the number of observed RVs should be the same for all models.
        By default a single model will be inferred from ``with`` context, in this case results will
        only be meaningful if all models share the same distributions for the observed RVs.
    weights : array-like, optional
        Individual weights for each trace. Default, same weight for each model.
    random_seed : int, optional
        Seed for the random number generator.
    progressbar : bool, optional default True
        Whether or not to display a progress bar in the command line. The bar shows the percentage
        of completion, the sampling speed in samples per second (SPS), and the estimated remaining
        time until completion ("expected time of arrival"; ETA).

    Returns
    -------
    samples : dict
        Dictionary with the variables as keys. The values corresponding to the
        posterior predictive samples from the weighted models.
    """
    np.random.seed(random_seed)

    if isinstance(traces[0], InferenceData):
        n_samples = [
            trace.posterior.sizes["chain"] * trace.posterior.sizes["draw"] for trace in traces
        ]
        traces = [dataset_to_point_list(trace.posterior) for trace in traces]
    elif isinstance(traces[0], xarray.Dataset):
        n_samples = [trace.sizes["chain"] * trace.sizes["draw"] for trace in traces]
        traces = [dataset_to_point_list(trace) for trace in traces]
    else:
        n_samples = [len(i) * i.nchains for i in traces]

    if models is None:
        models = [modelcontext(models)] * len(traces)

    for model in models:
        if model.potentials:
            warnings.warn(
                "The effect of Potentials on other parameters is ignored during posterior predictive sampling. "
                "This is likely to lead to invalid or biased predictive samples.",
                UserWarning,
            )
            break

    if weights is None:
        weights = [1] * len(traces)

    if len(traces) != len(weights):
        raise ValueError("The number of traces and weights should be the same")

    if len(models) != len(weights):
        raise ValueError("The number of models and weights should be the same")

    length_morv = len(models[0].observed_RVs)
    if any(len(i.observed_RVs) != length_morv for i in models):
        raise ValueError("The number of observed RVs should be the same for all models")

    weights = np.asarray(weights)
    p = weights / np.sum(weights)

    min_tr = min(n_samples)

    n = (min_tr * p).astype("int")
    # ensure n sum up to min_tr
    idx = np.argmax(n)
    n[idx] = n[idx] + min_tr - np.sum(n)
    trace = []
    for i, j in enumerate(n):
        tr = traces[i]
        len_trace = len(tr)
        try:
            nchain = tr.nchains
        except AttributeError:
            nchain = 1

        indices = np.random.randint(0, nchain * len_trace, j)
        if nchain > 1:
            chain_idx, point_idx = np.divmod(indices, len_trace)
            for idx in zip(chain_idx, point_idx):
                trace.append(tr._straces[idx[0]].point(idx[1]))
        else:
            for idx in indices:
                trace.append(tr[idx])

    obs = [x for m in models for x in m.observed_RVs]
    variables = np.repeat(obs, n)

    lengths = list({np.atleast_1d(observed).shape for observed in obs})

    if len(lengths) == 1:
        size = [None for i in variables]
    elif len(lengths) > 2:
        raise ValueError("Observed variables could not be broadcast together")
    else:
        size = []
        x = np.zeros(shape=lengths[0])
        y = np.zeros(shape=lengths[1])
        b = np.broadcast(x, y)
        for var in variables:
            shape = np.shape(np.atleast_1d(var.distribution.default()))
            if shape != b.shape:
                size.append(b.shape)
            else:
                size.append(None)
    len_trace = len(trace)

    if samples is None:
        samples = len_trace

    indices = np.random.randint(0, len_trace, samples)

    if progressbar:
        indices = progress_bar(indices, total=samples, display=progressbar)

    try:
        ppc = defaultdict(list)
        for idx in indices:
            param = trace[idx]
            var = variables[idx]
            # TODO sample_posterior_predictive_w is currently only work for model with
            # one observed.
            ppc[var.name].append(draw_values([var], point=param, size=size[idx])[0])

    except KeyboardInterrupt:
        pass
    else:
        return {k: np.asarray(v) for k, v in ppc.items()}


def sample_prior_predictive(
    samples=500,
    model: Optional[Model] = None,
    var_names: Optional[Iterable[str]] = None,
    random_seed=None,
) -> Dict[str, np.ndarray]:
    """Generate samples from the prior predictive distribution.

    Parameters
    ----------
    samples : int
        Number of samples from the prior predictive to generate. Defaults to 500.
    model : Model (optional if in ``with`` context)
    var_names : Iterable[str]
        A list of names of variables for which to compute the posterior predictive
        samples. Defaults to both observed and unobserved RVs.
    random_seed : int
        Seed for the random number generator.

    Returns
    -------
    dict
        Dictionary with variable names as keys. The values are numpy arrays of prior
        samples.
    """
    model = modelcontext(model)

    if model.potentials:
        warnings.warn(
            "The effect of Potentials on other parameters is ignored during prior predictive sampling. "
            "This is likely to lead to invalid or biased predictive samples.",
            UserWarning,
        )

    if var_names is None:
        prior_pred_vars = model.observed_RVs
        prior_vars = (
            get_default_varnames(model.unobserved_RVs, include_transformed=True) + model.potentials
        )
        vars_: Set[str] = {var.name for var in prior_vars + prior_pred_vars}
    else:
        vars_ = set(var_names)

    if random_seed is not None:
        np.random.seed(random_seed)
    names = get_default_varnames(vars_, include_transformed=False)
    # draw_values fails with auto-transformed variables. transform them later!
    values = draw_values([model[name] for name in names], size=samples)

    data = {k: v for k, v in zip(names, values)}
    if data is None:
        raise AssertionError("No variables sampled: attempting to sample %s" % names)

    prior: Dict[str, np.ndarray] = {}
    for var_name in vars_:
        if var_name in data:
            prior[var_name] = data[var_name]
        elif is_transformed_name(var_name):
            untransformed = get_untransformed_name(var_name)
            if untransformed in data:
                prior[var_name] = model[untransformed].transformation.forward_val(
                    data[untransformed]
                )
    return prior


def _init_jitter(model, chains, jitter_max_retries):
    """Apply a uniform jitter in [-1, 1] to the test value as starting point in each chain.

    pymc3.util.check_start_vals is used to test whether the jittered starting values produce
    a finite log probability. Invalid values are resampled unless `jitter_max_retries` is achieved,
    in which case the last sampled values are returned.

    Parameters
    ----------
    model : pymc3.Model
    chains : int
    jitter_max_retries : int
        Maximum number of repeated attempts at initializing values (per chain).

    Returns
    -------
    start : ``pymc3.model.Point``
        Starting point for sampler
    """
    start = []
    for _ in range(chains):
        for i in range(jitter_max_retries + 1):
            mean = {var: val.copy() for var, val in model.test_point.items()}
            for val in mean.values():
                val[...] += 2 * np.random.rand(*val.shape) - 1

            if i < jitter_max_retries:
                try:
                    check_start_vals(mean, model)
                except SamplingError:
                    pass
                else:
                    break

        start.append(mean)
    return start


def init_nuts(
    init="auto",
    chains=1,
    n_init=500000,
    model=None,
    random_seed=None,
    progressbar=True,
    jitter_max_retries=10,
    **kwargs,
):
    """Set up the mass matrix initialization for NUTS.

    NUTS convergence and sampling speed is extremely dependent on the
    choice of mass/scaling matrix. This function implements different
    methods for choosing or adapting the mass matrix.

    Parameters
    ----------
    init : str
        Initialization method to use.

        * auto: Choose a default initialization method automatically.
          Currently, this is ``jitter+adapt_diag``, but this can change in the future. If you
          depend on the exact behaviour, choose an initialization method explicitly.
        * adapt_diag: Start with a identity mass matrix and then adapt a diagonal based on the
          variance of the tuning samples. All chains use the test value (usually the prior mean)
          as starting point.
        * jitter+adapt_diag: Same as ``adapt_diag``, but use test value plus a uniform jitter in
          [-1, 1] as starting point in each chain.
        * advi+adapt_diag: Run ADVI and then adapt the resulting diagonal mass matrix based on the
          sample variance of the tuning samples.
        * advi+adapt_diag_grad: Run ADVI and then adapt the resulting diagonal mass matrix based
          on the variance of the gradients during tuning. This is **experimental** and might be
          removed in a future release.
        * advi: Run ADVI to estimate posterior mean and diagonal mass matrix.
        * advi_map: Initialize ADVI with MAP and use MAP as starting point.
        * map: Use the MAP as starting point. This is discouraged.
        * adapt_full: Adapt a dense mass matrix using the sample covariances. All chains use the
          test value (usually the prior mean) as starting point.
        * jitter+adapt_full: Same as ``adapt_full``, but use test value plus a uniform jitter in
          [-1, 1] as starting point in each chain.

    chains : int
        Number of jobs to start.
    n_init : int
        Number of iterations of initializer. Only works for 'ADVI' init methods.
    model : Model (optional if in ``with`` context)
    progressbar : bool
        Whether or not to display a progressbar for advi sampling.
    jitter_max_retries : int
        Maximum number of repeated attempts (per chain) at creating an initial matrix with uniform jitter
        that yields a finite probability. This applies to ``jitter+adapt_diag`` and ``jitter+adapt_full``
        init methods.
    **kwargs : keyword arguments
        Extra keyword arguments are forwarded to pymc3.NUTS.

    Returns
    -------
    start : ``pymc3.model.Point``
        Starting point for sampler
    nuts_sampler : ``pymc3.step_methods.NUTS``
        Instantiated and initialized NUTS sampler object
    """
    model = modelcontext(model)

    vars = kwargs.get("vars", model.vars)
    if set(vars) != set(model.vars):
        raise ValueError("Must use init_nuts on all variables of a model.")
    if not all_continuous(vars):
        raise ValueError("init_nuts can only be used for models with only " "continuous variables.")

    if not isinstance(init, str):
        raise TypeError("init must be a string.")

    if init is not None:
        init = init.lower()

    if init == "auto":
        init = "jitter+adapt_diag"

    _log.info(f"Initializing NUTS using {init}...")

    if random_seed is not None:
        random_seed = int(np.atleast_1d(random_seed)[0])
        np.random.seed(random_seed)

    cb = [
        pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff="absolute"),
        pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff="relative"),
    ]

    if init == "adapt_diag":
        start = [model.test_point] * chains
        mean = np.mean([model.dict_to_array(vals) for vals in start], axis=0)
        var = np.ones_like(mean)
        potential = quadpotential.QuadPotentialDiagAdapt(model.ndim, mean, var, 10)
    elif init == "jitter+adapt_diag":
        start = _init_jitter(model, chains, jitter_max_retries)
        mean = np.mean([model.dict_to_array(vals) for vals in start], axis=0)
        var = np.ones_like(mean)
        potential = quadpotential.QuadPotentialDiagAdapt(model.ndim, mean, var, 10)
    elif init == "advi+adapt_diag_grad":
        approx: pm.MeanField = pm.fit(
            random_seed=random_seed,
            n=n_init,
            method="advi",
            model=model,
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window,
        )
        start = approx.sample(draws=chains)
        start = list(start)
        stds = approx.bij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        mean = approx.bij.rmap(approx.mean.get_value())
        mean = model.dict_to_array(mean)
        weight = 50
        potential = quadpotential.QuadPotentialDiagAdaptGrad(model.ndim, mean, cov, weight)
    elif init == "advi+adapt_diag":
        approx = pm.fit(
            random_seed=random_seed,
            n=n_init,
            method="advi",
            model=model,
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window,
        )
        start = approx.sample(draws=chains)
        start = list(start)
        stds = approx.bij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        mean = approx.bij.rmap(approx.mean.get_value())
        mean = model.dict_to_array(mean)
        weight = 50
        potential = quadpotential.QuadPotentialDiagAdapt(model.ndim, mean, cov, weight)
    elif init == "advi":
        approx = pm.fit(
            random_seed=random_seed,
            n=n_init,
            method="advi",
            model=model,
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window,
        )
        start = approx.sample(draws=chains)
        start = list(start)
        stds = approx.bij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        potential = quadpotential.QuadPotentialDiag(cov)
    elif init == "advi_map":
        start = pm.find_MAP(include_transformed=True)
        approx = pm.MeanField(model=model, start=start)
        pm.fit(
            random_seed=random_seed,
            n=n_init,
            method=pm.KLqp(approx),
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window,
        )
        start = approx.sample(draws=chains)
        start = list(start)
        stds = approx.bij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        potential = quadpotential.QuadPotentialDiag(cov)
    elif init == "map":
        start = pm.find_MAP(include_transformed=True)
        cov = pm.find_hessian(point=start)
        start = [start] * chains
        potential = quadpotential.QuadPotentialFull(cov)
    elif init == "adapt_full":
        start = [model.test_point] * chains
        mean = np.mean([model.dict_to_array(vals) for vals in start], axis=0)
        cov = np.eye(model.ndim)
        potential = quadpotential.QuadPotentialFullAdapt(model.ndim, mean, cov, 10)
    elif init == "jitter+adapt_full":
        start = _init_jitter(model, chains, jitter_max_retries)
        mean = np.mean([model.dict_to_array(vals) for vals in start], axis=0)
        cov = np.eye(model.ndim)
        potential = quadpotential.QuadPotentialFullAdapt(model.ndim, mean, cov, 10)
    else:
        raise ValueError(f"Unknown initializer: {init}.")

    step = pm.NUTS(potential=potential, model=model, **kwargs)

    return start, step
