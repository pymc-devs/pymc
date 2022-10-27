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

import logging
import pickle
import sys
import time
import warnings

from collections import defaultdict
from copy import copy
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import aesara.gradient as tg
import cloudpickle
import numpy as np
import xarray

from aesara import tensor as at
from aesara.graph.basic import Apply, Constant, Variable, general_toposort, walk
from aesara.graph.fg import FunctionGraph
from aesara.tensor.random.var import (
    RandomGeneratorSharedVariable,
    RandomStateSharedVariable,
)
from aesara.tensor.sharedvar import SharedVariable
from arviz import InferenceData
from fastprogress.fastprogress import progress_bar
from typing_extensions import TypeAlias

import pymc as pm

from pymc.aesaraf import compile_pymc
from pymc.backends.arviz import _DefaultTrace
from pymc.backends.base import BaseTrace, MultiTrace
from pymc.backends.ndarray import NDArray
from pymc.blocking import DictToArrayBijection
from pymc.exceptions import SamplingError
from pymc.initial_point import (
    PointType,
    StartDict,
    filter_rvs_to_jitter,
    make_initial_point_fns_per_chain,
)
from pymc.model import Model, modelcontext
from pymc.parallel_sampling import Draw, _cpu_count
from pymc.stats.convergence import run_convergence_checks
from pymc.step_methods import NUTS, CompoundStep, DEMetropolis
from pymc.step_methods.arraystep import BlockedStep, PopulationArrayStepShared
from pymc.step_methods.hmc import quadpotential
from pymc.util import (
    dataset_to_point_list,
    get_default_varnames,
    get_untransformed_name,
    is_transformed_name,
    point_wrapper,
)
from pymc.vartypes import discrete_types

sys.setrecursionlimit(10000)

__all__ = [
    "sample",
    "iter_sample",
    "compile_forward_sampling_function",
    "sample_posterior_predictive",
    "sample_posterior_predictive_w",
    "init_nuts",
    "sample_prior_predictive",
    "draw",
]

Step: TypeAlias = Union[BlockedStep, CompoundStep]

ArrayLike: TypeAlias = Union[np.ndarray, List[float]]
PointList: TypeAlias = List[PointType]
Backend: TypeAlias = Union[BaseTrace, MultiTrace, NDArray]

RandomSeed = Optional[Union[int, Sequence[int], np.ndarray]]
RandomState = Union[RandomSeed, np.random.RandomState, np.random.Generator]

_log = logging.getLogger("pymc")


def instantiate_steppers(
    model, steps: List[Step], selected_steps, step_kwargs=None
) -> Union[Step, List[Step]]:
    """Instantiate steppers assigned to the model variables.

    This function is intended to be called automatically from ``sample()``, but
    may be called manually.

    Parameters
    ----------
    model : Model object
        A fully-specified model object.
    steps : list, array_like of shape (selected_steps, )
        A list of zero or more step function instances that have been assigned to some subset of
        the model's parameters.
    selected_steps : dict
        A dictionary that maps a step method class to a list of zero or more model variables.
    step_kwargs : dict, default=None
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
            step = step_class(vars=vars, model=model, **args)
            steps.append(step)

    unused_args = set(step_kwargs).difference(used_keys)
    if unused_args:
        raise ValueError("Unused step method arguments: %s" % unused_args)

    if len(steps) == 1:
        return steps[0]

    return steps


def assign_step_methods(model, step=None, methods=None, step_kwargs=None):
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
        A fully-specified model object.
    step : step function or iterable of step functions, optional
        One or more step functions that have been assigned to some subset of
        the model's parameters. Defaults to ``None`` (no assigned variables).
    methods : iterable of step method classes, optional
        The set of step methods from which the function may choose. Defaults
        to the main step methods provided by PyMC.
    step_kwargs : dict, optional
        Parameters for the samplers. Keys are the lower case names of
        the step method, values a dict of arguments.

    Returns
    -------
    methods : list
        List of step methods associated with the model's variables.
    """
    steps = []
    assigned_vars = set()

    if methods is None:
        methods = pm.STEP_METHODS

    if step is not None:
        try:
            steps += list(step)
        except TypeError:
            steps.append(step)
        for step in steps:
            assigned_vars = assigned_vars.union(set(step.vars))

    # Use competence classmethods to select step methods for remaining
    # variables
    selected_steps = defaultdict(list)
    model_logp = model.logp()

    for var in model.value_vars:
        if var not in assigned_vars:
            # determine if a gradient can be computed
            has_gradient = var.dtype not in discrete_types
            if has_gradient:
                try:
                    tg.grad(model_logp, var)
                except (NotImplementedError, tg.NullTypeGradError):
                    has_gradient = False

            # select the best method
            rv_var = model.values_to_rvs[var]
            selected = max(
                methods,
                key=lambda method, var=rv_var, has_gradient=has_gradient: method._competence(
                    var, has_gradient
                ),
            )
            selected_steps[selected].append(var)

    return instantiate_steppers(model, steps, selected_steps, step_kwargs)


def _print_step_hierarchy(s: Step, level: int = 0) -> None:
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


def all_continuous(vars):
    """Check that vars not include discrete variables, excepting observed RVs."""

    vars_ = [var for var in vars if not hasattr(var.tag, "observations")]

    if any([(var.dtype in discrete_types) for var in vars_]):
        return False
    else:
        return True


def _get_seeds_per_chain(
    random_state: RandomState,
    chains: int,
) -> Union[Sequence[int], np.ndarray]:
    """Obtain or validate specified integer seeds per chain.

    This function process different possible sources of seeding and returns one integer
    seed per chain:
    1. If the input is an integer and a single chain is requested, the input is
        returned inside a tuple.
    2. If the input is a sequence or NumPy array with as many entries as chains,
        the input is returned.
    3. If the input is an integer and multiple chains are requested, new unique seeds
        are generated from NumPy default Generator seeded with that integer.
    4. If the input is None new unique seeds are generated from an unseeded NumPy default
        Generator.
    5. If a RandomState or Generator is provided, new unique seeds are generated from it.

    Raises
    ------
    ValueError
        If none of the conditions above are met
    """

    def _get_unique_seeds_per_chain(integers_fn):
        seeds = []
        while len(set(seeds)) != chains:
            seeds = [int(seed) for seed in integers_fn(2**30, dtype=np.int64, size=chains)]
        return seeds

    if random_state is None or isinstance(random_state, int):
        if chains == 1 and isinstance(random_state, int):
            return (random_state,)
        return _get_unique_seeds_per_chain(np.random.default_rng(random_state).integers)
    if isinstance(random_state, np.random.Generator):
        return _get_unique_seeds_per_chain(random_state.integers)
    if isinstance(random_state, np.random.RandomState):
        return _get_unique_seeds_per_chain(random_state.randint)

    if not isinstance(random_state, (list, tuple, np.ndarray)):
        raise ValueError(f"The `seeds` must be array-like. Got {type(random_state)} instead.")

    if len(random_state) != chains:
        raise ValueError(
            f"Number of seeds ({len(random_state)}) does not match the number of chains ({chains})."
        )

    return random_state


def sample(
    draws: int = 1000,
    step=None,
    init: str = "auto",
    n_init: int = 200_000,
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
    trace: Optional[Union[BaseTrace, List[str]]] = None,
    chains: Optional[int] = None,
    cores: Optional[int] = None,
    tune: int = 1000,
    progressbar: bool = True,
    model=None,
    random_seed: RandomState = None,
    discard_tuned_samples: bool = True,
    compute_convergence_checks: bool = True,
    callback=None,
    jitter_max_retries: int = 10,
    *,
    return_inferencedata: bool = True,
    idata_kwargs: dict = None,
    mp_ctx=None,
    **kwargs,
) -> Union[InferenceData, MultiTrace]:
    r"""Draw samples from the posterior using the given step methods.

    Multiple step methods are supported via compound step methods.

    Parameters
    ----------
    draws : int
        The number of samples to draw. Defaults to 1000. The number of tuned samples are discarded
        by default. See ``discard_tuned_samples``.
    init : str
        Initialization method to use for auto-assigned NUTS samplers. See `pm.init_nuts` for a list
        of all options. This argument is ignored when manually passing the NUTS step method.
    step : function or iterable of functions
        A step function or collection of functions. If there are variables without step methods,
        step methods for those variables will be assigned automatically. By default the NUTS step
        method will be used, if appropriate to the model.
    n_init : int
        Number of iterations of initializer. Only works for 'ADVI' init methods.
    initvals : optional, dict, array of dict
        Dict or list of dicts with initial value strategies to use instead of the defaults from
        `Model.initial_values`. The keys should be names of transformed random variables.
        Initialization methods for NUTS (see ``init`` keyword) can overwrite the default.
    trace : backend or list
        This should be a backend instance, or a list of variables to track.
        If None or a list of variables, the NDArray backend is used.
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
        Model to sample from. The model needs to have free random variables.
    random_seed : int, array-like of int, RandomState or Generator, optional
        Random seed(s) used by the sampling steps. If a list, tuple or array of ints
        is passed, each entry will be used to seed each chain. A ValueError will be
        raised if the length does not match the number of chains.
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
        Maximum number of repeated attempts (per chain) at creating an initial matrix with uniform
        jitter that yields a finite probability. This applies to ``jitter+adapt_diag`` and
        ``jitter+adapt_full`` init methods.
    return_inferencedata : bool
        Whether to return the trace as an :class:`arviz:arviz.InferenceData` (True) object or a
        `MultiTrace` (False). Defaults to `True`.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data`
    mp_ctx : multiprocessing.context.BaseContent
        A multiprocessing context for parallel sampling.
        See multiprocessing documentation for details.

    Returns
    -------
    trace : pymc.backends.base.MultiTrace or arviz.InferenceData
        A ``MultiTrace`` or ArviZ ``InferenceData`` object that contains the samples.

    Notes
    -----
    Optional keyword arguments can be passed to ``sample`` to be delivered to the
    ``step_method``\ s used during sampling.

    For example:

       1. ``target_accept`` to NUTS: nuts={'target_accept':0.9}
       2. ``transit_p`` to BinaryGibbsMetropolis: binary_gibbs_metropolis={'transit_p':.7}

    Note that available step names are:

    ``nuts``, ``hmc``, ``metropolis``, ``binary_metropolis``,
    ``binary_gibbs_metropolis``, ``categorical_gibbs_metropolis``,
    ``DEMetropolis``, ``DEMetropolisZ``, ``slice``

    The NUTS step method has several options including:

        * target_accept : float in [0, 1]. The step size is tuned such that we
          approximate this acceptance rate. Higher values like 0.9 or 0.95 often
          work better for problematic posteriors. This argument can be passed directly to sample.
        * max_treedepth : The maximum depth of the trajectory tree
        * step_scale : float, default 0.25
          The initial guess for the step size scaled down by :math:`1/n**(1/4)`,
          where n is the dimensionality of the parameter space

    Alternatively, if you manually declare the ``step_method``\ s, within the ``step``
       kwarg, then you can address the ``step_method`` kwargs directly.
       e.g. for a CompoundStep comprising NUTS and BinaryGibbsMetropolis,
       you could send ::

        step=[pm.NUTS([freeRV1, freeRV2], target_accept=0.9),
              pm.BinaryGibbsMetropolis([freeRV3], transit_p=.7)]

    You can find a full list of arguments in the docstring of the step methods.

    Examples
    --------
    .. code:: ipython

        In [1]: import pymc as pm
           ...: n = 100
           ...: h = 61
           ...: alpha = 2
           ...: beta = 2

        In [2]: with pm.Model() as model: # context management
           ...:     p = pm.Beta("p", alpha=alpha, beta=beta)
           ...:     y = pm.Binomial("y", n=n, p=p, observed=h)
           ...:     idata = pm.sample()

        In [3]: az.summary(idata, kind="stats")

        Out[3]:
            mean     sd  hdi_3%  hdi_97%
        p  0.609  0.047   0.528    0.699
    """
    if "start" in kwargs:
        if initvals is not None:
            raise ValueError("Passing both `start` and `initvals` is not supported.")
        warnings.warn(
            "The `start` kwarg was renamed to `initvals` and can now do more. Please check the docstring.",
            FutureWarning,
            stacklevel=2,
        )
        initvals = kwargs.pop("start")
    if "target_accept" in kwargs:
        if "nuts" in kwargs and "target_accept" in kwargs["nuts"]:
            raise ValueError(
                "`target_accept` was defined twice. Please specify it either as a direct keyword argument or in the `nuts` kwarg."
            )
        if "nuts" in kwargs:
            kwargs["nuts"]["target_accept"] = kwargs.pop("target_accept")
        else:
            kwargs = {"nuts": {"target_accept": kwargs.pop("target_accept")}}

    model = modelcontext(model)
    if not model.free_RVs:
        raise SamplingError(
            "Cannot sample from the model, since the model does not contain any free variables."
        )

    if cores is None:
        cores = min(4, _cpu_count())

    if chains is None:
        chains = max(2, cores)

    if random_seed == -1:
        random_seed = None
    random_seed_list = _get_seeds_per_chain(random_seed, chains)

    if not discard_tuned_samples and not return_inferencedata:
        warnings.warn(
            "Tuning samples will be included in the returned `MultiTrace` object, which can lead to"
            " complications in your downstream analysis. Please consider to switch to `InferenceData`:\n"
            "`pm.sample(..., return_inferencedata=True)`",
            UserWarning,
            stacklevel=2,
        )

    # small trace warning
    if draws == 0:
        msg = "Tuning was enabled throughout the whole trace."
        _log.warning(msg)
    elif draws < 500:
        msg = "Only %s samples in chain." % draws
        _log.warning(msg)

    draws += tune

    auto_nuts_init = True
    if step is not None:
        if isinstance(step, CompoundStep):
            for method in step.methods:
                if isinstance(method, NUTS):
                    auto_nuts_init = False
        elif isinstance(step, NUTS):
            auto_nuts_init = False

    initial_points = None
    step = assign_step_methods(model, step, methods=pm.STEP_METHODS, step_kwargs=kwargs)

    if isinstance(step, list):
        step = CompoundStep(step)
    elif isinstance(step, NUTS) and auto_nuts_init:
        if "nuts" in kwargs:
            nuts_kwargs = kwargs.pop("nuts")
            [kwargs.setdefault(k, v) for k, v in nuts_kwargs.items()]
        _log.info("Auto-assigning NUTS sampler...")
        initial_points, step = init_nuts(
            init=init,
            chains=chains,
            n_init=n_init,
            model=model,
            random_seed=random_seed_list,
            progressbar=progressbar,
            jitter_max_retries=jitter_max_retries,
            tune=tune,
            initvals=initvals,
            **kwargs,
        )

    if initial_points is None:
        # Time to draw/evaluate numeric start points for each chain.
        ipfns = make_initial_point_fns_per_chain(
            model=model,
            overrides=initvals,
            jitter_rvs=filter_rvs_to_jitter(step),
            chains=chains,
        )
        initial_points = [ipfn(seed) for ipfn, seed in zip(ipfns, random_seed_list)]

    # One final check that shapes and logps at the starting points are okay.
    for ip in initial_points:
        model.check_start_vals(ip)
        _check_start_shape(model, ip)

    sample_args = {
        "draws": draws,
        "step": step,
        "start": initial_points,
        "trace": trace,
        "chains": chains,
        "tune": tune,
        "progressbar": progressbar,
        "model": model,
        "cores": cores,
        "callback": callback,
        "discard_tuned_samples": discard_tuned_samples,
    }
    parallel_args = {
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
    # At some point it was decided that PyMC should not set a global seed by default,
    # unless the user specified a seed. This is a symptom of the fact that PyMC samplers
    # are built around global seeding. This branch makes sure we maintain this unspoken
    # rule. See https://github.com/pymc-devs/pymc/pull/1395.
    if parallel:
        # For parallel sampling we can pass the list of random seeds directly, as
        # global seeding will only be called inside each process
        sample_args["random_seed"] = random_seed_list
    else:
        # We pass None if the original random seed was None. The single core sampler
        # methods will only set a global seed when it is not None.
        sample_args["random_seed"] = random_seed if random_seed is None else random_seed_list

    t_start = time.time()
    if parallel:
        _log.info(f"Multiprocess sampling ({chains} chains in {cores} jobs)")
        _print_step_hierarchy(step)
        try:
            mtrace = _mp_sample(**sample_args, **parallel_args)
        except pickle.PickleError:
            _log.warning("Could not pickle model, sampling singlethreaded.")
            _log.debug("Pickling error:", exc_info=True)
            parallel = False
        except AttributeError as e:
            if not str(e).startswith("AttributeError: Can't pickle"):
                raise
            _log.warning("Could not pickle model, sampling singlethreaded.")
            _log.debug("Pickling error:", exc_info=True)
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

            initial_point_model_size = sum(initial_points[0][n.name].size for n in model.value_vars)

            if has_demcmc and chains < 3:
                raise ValueError(
                    "DEMetropolis requires at least 3 chains. "
                    "For this {}-dimensional model you should use ≥{} chains".format(
                        initial_point_model_size, initial_point_model_size + 1
                    )
                )
            if has_demcmc and chains <= initial_point_model_size:
                warnings.warn(
                    "DEMetropolis should be used with more chains than dimensions! "
                    "(The model has {} dimensions.)".format(initial_point_model_size),
                    UserWarning,
                    stacklevel=2,
                )
            _print_step_hierarchy(step)
            mtrace = _sample_population(parallelize=cores > 1, **sample_args)
        else:
            _log.info(f"Sequential sampling ({chains} chains in 1 job)")
            _print_step_hierarchy(step)
            mtrace = _sample_many(**sample_args)

    t_sampling = time.time() - t_start
    # count the number of tune/draw iterations that happened
    # ideally via the "tune" statistic, but not all samplers record it!
    if "tune" in mtrace.stat_names:
        stat = mtrace.get_sampler_stats("tune", chains=0)
        # when CompoundStep is used, the stat is 2 dimensional!
        if len(stat.shape) == 2:
            stat = stat[:, 0]
        stat = tuple(stat)
        n_tune = stat.count(True)
        n_draws = stat.count(False)
    else:
        # these may be wrong when KeyboardInterrupt happened, but they're better than nothing
        n_tune = min(tune, len(mtrace))
        n_draws = max(0, len(mtrace) - n_tune)

    if discard_tuned_samples:
        mtrace = mtrace[n_tune:]

    # save metadata in SamplerReport
    mtrace.report._n_tune = n_tune
    mtrace.report._n_draws = n_draws
    mtrace.report._t_sampling = t_sampling

    n_chains = len(mtrace.chains)
    _log.info(
        f'Sampling {n_chains} chain{"s" if n_chains > 1 else ""} for {n_tune:_d} tune and {n_draws:_d} draw iterations '
        f"({n_tune*n_chains:_d} + {n_draws*n_chains:_d} draws total) "
        f"took {t_sampling:.0f} seconds."
    )
    mtrace.report._log_summary()

    idata = None
    if compute_convergence_checks or return_inferencedata:
        ikwargs = dict(model=model, save_warmup=not discard_tuned_samples)
        if idata_kwargs:
            ikwargs.update(idata_kwargs)
        idata = pm.to_inference_data(mtrace, **ikwargs)

        if compute_convergence_checks:
            if draws - tune < 100:
                warnings.warn(
                    "The number of samples is too small to check convergence reliably.",
                    stacklevel=2,
                )
            else:
                convergence_warnings = run_convergence_checks(idata, model)
                mtrace.report._add_warnings(convergence_warnings)

        if return_inferencedata:
            return idata
    return mtrace


def _check_start_shape(model, start: PointType):
    """Checks that the prior evaluations and initial points have identical shapes.

    Parameters
    ----------
    model : pm.Model
        The current model on context.
    start : dict
        The complete dictionary mapping (transformed) variable names to numeric initial values.
    """
    e = ""
    try:
        actual_shapes = model.eval_rv_shapes()
    except NotImplementedError as ex:
        warnings.warn(f"Unable to validate shapes: {ex.args[0]}", UserWarning)
        return
    for name, sval in start.items():
        ashape = actual_shapes.get(name)
        sshape = np.shape(sval)
        if ashape != tuple(sshape):
            e += f"\nExpected shape {ashape} for var '{name}', got: {sshape}"
    if e != "":
        raise ValueError(f"Bad shape in start point:{e}")


def _sample_many(
    draws: int,
    chains: int,
    start: Sequence[PointType],
    random_seed: Optional[Sequence[RandomSeed]],
    step,
    callback=None,
    **kwargs,
) -> MultiTrace:
    """Samples all chains sequentially.

    Parameters
    ----------
    draws: int
        The number of samples to draw
    chains: int
        Total number of chains to sample.
    start: list
        Starting points for each chain
    random_seed: list of random seeds, optional
        A list of seeds, one for each chain
    step: function
        Step function

    Returns
    -------
    mtrace: MultiTrace
        Contains samples of all chains
    """
    traces: List[BaseTrace] = []
    for i in range(chains):
        trace = _sample(
            draws=draws,
            chain=i,
            start=start[i],
            step=step,
            random_seed=None if random_seed is None else random_seed[i],
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
    chains: int,
    start: Sequence[PointType],
    random_seed: RandomSeed,
    step,
    tune: int,
    model,
    progressbar: bool = True,
    parallelize: bool = False,
    **kwargs,
) -> MultiTrace:
    """Performs sampling of a population of chains using the ``PopulationStepper``.

    Parameters
    ----------
    draws : int
        The number of samples to draw
    chains : int
        The total number of chains in the population
    start : list
        Start points for each chain
    random_seed : single random seed, optional
    step : function
        Step function (should be or contain a population step method)
    tune : int
        Number of iterations to tune.
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
    *,
    chain: int,
    progressbar: bool,
    random_seed: RandomSeed,
    start: PointType,
    draws: int,
    step=None,
    trace: Optional[Union[BaseTrace, List[str]]] = None,
    tune: int,
    model: Optional[Model] = None,
    callback=None,
    **kwargs,
) -> BaseTrace:
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
    random_seed : single random seed
    start : dict
        Starting point in parameter space (or partial point)
    draws : int
        The number of samples to draw
    step : function
        Step function
    trace : backend or list
        This should be a backend instance, or a list of variables to track.
        If None or a list of variables, the NDArray backend is used.
    tune : int
        Number of iterations to tune.
    model : Model (optional if in ``with`` context)

    Returns
    -------
    strace : BaseTrace
        A ``BaseTrace`` object that contains the samples for this chain.
    """
    skip_first = kwargs.get("skip_first", 0)

    trace = copy(trace)

    sampling_gen = _iter_sample(
        draws, step, start, trace, chain, tune, model, random_seed, callback
    )
    _pbar_data = {"chain": chain, "divergences": 0}
    _desc = "Sampling chain {chain:d}, {divergences:,d} divergences"
    if progressbar:
        sampling = progress_bar(sampling_gen, total=draws, display=progressbar)
        sampling.comment = _desc.format(**_pbar_data)
    else:
        sampling = sampling_gen
    try:
        strace = None
        for it, (strace, diverging) in enumerate(sampling):
            if it >= skip_first and diverging:
                _pbar_data["divergences"] += 1
                if progressbar:
                    sampling.comment = _desc.format(**_pbar_data)
    except KeyboardInterrupt:
        pass
    if strace is None:
        raise Exception("KeyboardInterrupt happened before the base trace was created.")
    return strace


def iter_sample(
    draws: int,
    step,
    start: PointType,
    trace=None,
    chain: int = 0,
    tune: int = 0,
    model: Optional[Model] = None,
    random_seed: RandomSeed = None,
    callback=None,
) -> Iterator[MultiTrace]:
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
        Starting point in parameter space (or partial point).
    trace : backend or list
        This should be a backend instance, or a list of variables to track.
        If None or a list of variables, the NDArray backend is used.
    chain : int, optional
        Chain number used to store sample in backend.
    tune : int, optional
        Number of iterations to tune (defaults to 0).
    model : Model (optional if in ``with`` context)
    random_seed : single random seed, optional
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
    draws: int,
    step,
    start: PointType,
    trace: Optional[Union[BaseTrace, List[str]]] = None,
    chain: int = 0,
    tune: int = 0,
    model=None,
    random_seed: RandomSeed = None,
    callback=None,
) -> Iterator[Tuple[BaseTrace, bool]]:
    """Generator for sampling one chain. (Used in singleprocess sampling.)

    Parameters
    ----------
    draws : int
        The number of samples to draw
    step : function
        Step function
    start : dict
        Starting point in parameter space (or partial point).
        Must contain numeric (transformed) initial values for all (transformed) free variables.
    trace : backend or list
        This should be a backend instance, or a list of variables to track.
        If None or a list of variables, the NDArray backend is used.
    chain : int, optional
        Chain number used to store sample in backend.
    tune : int, optional
        Number of iterations to tune (defaults to 0).
    model : Model (optional if in ``with`` context)
    random_seed : single random seed, optional

    Yields
    ------
    strace : BaseTrace
        The trace object containing the samples for this chain
    diverging : bool
        Indicates if the draw is divergent. Only available with some samplers.
    """
    model = modelcontext(model)
    draws = int(draws)

    if draws < 1:
        raise ValueError("Argument `draws` must be greater than 0.")

    if random_seed is not None:
        np.random.seed(random_seed)

    try:
        step = CompoundStep(step)
    except TypeError:
        pass

    point = start

    strace: BaseTrace = _init_trace(
        expected_length=draws + tune,
        step=step,
        chain_number=chain,
        trace=trace,
        model=model,
    )

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
                strace.record(point, stats)
                diverging = i > tune and stats and stats[0].get("diverging")
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
                    stepper_dumps = cloudpickle.dumps(stepper, protocol=4)
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

    def step(self, tune_stop: bool, population):
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
    step,
    start: Sequence[PointType],
    parallelize: bool,
    tune: int,
    model=None,
    random_seed: RandomSeed = None,
    progressbar=True,
) -> Iterator[Sequence[BaseTrace]]:
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
    model : Model (optional if in ``with`` context)
    random_seed : single random seed, optional
    progressbar : bool
        ``progressbar`` argument for the ``PopulationStepper``, (defaults to True)

    Returns
    -------
    _iter_population : generator
        Yields traces of all chains at the same time
    """
    nchains = len(start)
    model = modelcontext(model)
    draws = int(draws)

    if draws < 1:
        raise ValueError("Argument `draws` should be above 0.")

    if random_seed is not None:
        np.random.seed(random_seed)

    # The initialization of traces, samplers and points must happen in the right order:
    # 1. population of points is created
    # 2. steppers are initialized and linked to the points object
    # 3. traces are initialized
    # 4. a PopulationStepper is configured for parallelized stepping

    # 1. create a population (points) that tracks each chain
    # it is updated as the chains are advanced
    population = [start[c] for c in range(nchains)]

    # 2. Set up the steppers
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

    # 3. Initialize a BaseTrace for each chain
    traces: List[BaseTrace] = [
        _init_trace(
            expected_length=draws + tune,
            step=steppers[c],
            chain_number=c,
            trace=None,
            model=model,
        )
        for c in range(nchains)
    ]

    # 4. configure the PopulationStepper (expensive call)
    popstep = PopulationStepper(steppers, parallelize, progressbar=progressbar)

    # Because the preparations above are expensive, the actual iterator is
    # in another method. This way the progbar will not be disturbed.
    return _iter_population(draws, tune, popstep, steppers, traces, population)


def _iter_population(
    draws: int, tune: int, popstep: PopulationStepper, steppers, traces: Sequence[BaseTrace], points
) -> Iterator[Sequence[BaseTrace]]:
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
                        strace.record(points[c], stats)
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


def _choose_backend(trace: Optional[Union[BaseTrace, List[str]]], **kwds) -> BaseTrace:
    """Selects or creates a NDArray trace backend for a particular chain.

    Parameters
    ----------
    trace : BaseTrace, list, or None
        This should be a BaseTrace, or list of variables to track.
        If None or a list of variables, the NDArray backend is used.
    **kwds :
        keyword arguments to forward to the backend creation

    Returns
    -------
    trace : BaseTrace
        The incoming, or a brand new trace object.
    """
    if isinstance(trace, BaseTrace) and len(trace) > 0:
        raise ValueError("Continuation of traces is no longer supported.")
    if isinstance(trace, MultiTrace):
        raise ValueError("Starting from existing MultiTrace objects is no longer supported.")

    if isinstance(trace, BaseTrace):
        return trace
    if trace is None:
        return NDArray(**kwds)

    return NDArray(vars=trace, **kwds)


def _init_trace(
    *,
    expected_length: int,
    step: Step,
    chain_number: int,
    trace: Optional[Union[BaseTrace, List[str]]],
    model,
) -> BaseTrace:
    """Extracted helper function to create trace backends for each chain."""
    if trace is not None:
        strace = _choose_backend(copy(trace), model=model)
    else:
        strace = _choose_backend(None, model=model)

    if step.generates_stats:
        strace.setup(expected_length, chain_number, step.stats_dtypes)
    else:
        strace.setup(expected_length, chain_number)
    return strace


def _mp_sample(
    draws: int,
    tune: int,
    step,
    chains: int,
    cores: int,
    random_seed: Sequence[RandomSeed],
    start: Sequence[PointType],
    progressbar: bool = True,
    trace: Optional[Union[BaseTrace, List[str]]] = None,
    model=None,
    callback=None,
    discard_tuned_samples: bool = True,
    mp_ctx=None,
    **kwargs,
) -> MultiTrace:
    """Main iteration for multiprocess sampling.

    Parameters
    ----------
    draws : int
        The number of samples to draw
    tune : int
        Number of iterations to tune.
    step : function
        Step function
    chains : int
        The number of chains to sample.
    cores : int
        The number of chains to run in parallel.
    random_seed : list of random seeds
        Random seeds for each chain.
    start : list
        Starting points for each chain.
        Dicts must contain numeric (transformed) initial values for all (transformed) free variables.
    progressbar : bool
        Whether or not to display a progress bar in the command line.
    trace : BaseTrace, list, or None
        This should be a backend instance, or a list of variables to track
        If None or a list of variables, the NDArray backend is used.
    model : Model (optional if in ``with`` context)
    callback : Callable
        A function which gets called for every sample from the trace of a chain. The function is
        called with the trace and the current draw and will contain all samples for a single trace.
        the ``draw.chain`` argument can be used to determine which of the active chains the sample
        is drawn from.
        Sampling can be interrupted by throwing a ``KeyboardInterrupt`` in the callback.

    Returns
    -------
    mtrace : pymc.backends.base.MultiTrace
        A ``MultiTrace`` object that contains the samples for all chains.
    """
    import pymc.parallel_sampling as ps

    # We did draws += tune in pm.sample
    draws -= tune

    traces = [
        _init_trace(
            expected_length=draws + tune,
            step=step,
            chain_number=chain_number,
            trace=trace,
            model=model,
        )
        for chain_number in range(chains)
    ]

    sampler = ps.ParallelSampler(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        seeds=random_seed,
        start_points=start,
        step_method=step,
        progressbar=progressbar,
        mp_ctx=mp_ctx,
    )
    try:
        try:
            with sampler:
                for draw in sampler:
                    strace = traces[draw.chain]
                    if draw.stats is not None:
                        strace.record(draw.point, draw.stats)
                    else:
                        strace.record(draw.point)
                    if draw.is_last:
                        strace.close()
                        if draw.warnings is not None:
                            strace._add_warnings(draw.warnings)

                    if callback is not None:
                        callback(trace=trace, draw=draw)

        except ps.ParallelSamplingError as error:
            strace = traces[error._chain]
            strace._add_warnings(error._warnings)
            for strace in traces:
                strace.close()

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
        for strace in traces:
            strace.close()


def _choose_chains(traces: Sequence[BaseTrace], tune: int) -> Tuple[List[BaseTrace], int]:
    """
    Filter and slice traces such that (n_traces * len(shortest_trace)) is maximized.

    We get here after a ``KeyboardInterrupt``, and so the different
    traces have different lengths. We therefore pick the number of
    traces such that (number of traces) * (length of shortest trace)
    is maximised.
    """
    if not traces:
        raise ValueError("No traces to slice.")

    lengths = [max(0, len(trace) - tune) for trace in traces]
    if not sum(lengths):
        raise ValueError("Not enough samples to build a trace.")

    idxs = np.argsort(lengths)
    l_sort = np.array(lengths)[idxs]

    use_until = cast(int, np.argmax(l_sort * np.arange(1, l_sort.shape[0] + 1)[::-1]))
    final_length = l_sort[use_until]

    take_idx = cast(Sequence[int], idxs[use_until:])
    sliced_traces = [traces[idx] for idx in take_idx]
    return sliced_traces, final_length + tune


def stop_tuning(step):
    """Stop tuning the current step method."""
    step.stop_tuning()
    return step


def get_vars_in_point_list(trace, model):
    """Get the list of Variable instances in the model that have values stored in the trace."""
    if not isinstance(trace, MultiTrace):
        names_in_trace = list(trace[0])
    else:
        names_in_trace = trace.varnames
    vars_in_trace = [model[v] for v in names_in_trace if v in model]
    return vars_in_trace


def compile_forward_sampling_function(
    outputs: List[Variable],
    vars_in_trace: List[Variable],
    basic_rvs: Optional[List[Variable]] = None,
    givens_dict: Optional[Dict[Variable, Any]] = None,
    constant_data: Optional[Dict[str, np.ndarray]] = None,
    constant_coords: Optional[Set[str]] = None,
    **kwargs,
) -> Tuple[Callable[..., Union[np.ndarray, List[np.ndarray]]], Set[Variable]]:
    """Compile a function to draw samples, conditioned on the values of some variables.

    The goal of this function is to walk the aesara computational graph from the list
    of output nodes down to the root nodes, and then compile a function that will produce
    values for these output nodes. The compiled function will take as inputs the subset of
    variables in the ``vars_in_trace`` that are deemed to not be **volatile**.

    Volatile variables are variables whose values could change between runs of the
    compiled function or after inference has been run. These variables are:

    - Variables in the outputs list
    - ``SharedVariable`` instances that are not ``RandomStateSharedVariable`` or ``RandomGeneratorSharedVariable``, and whose values changed with respect to what they were at inference time
    - Variables that are in the `basic_rvs` list but not in the ``vars_in_trace`` list
    - Variables that are keys in the ``givens_dict``
    - Variables that have volatile inputs

    Concretely, this function can be used to compile a function to sample from the
    posterior predictive distribution of a model that has variables that are conditioned
    on ``MutableData`` instances. The variables that depend on the mutable data that have changed
    will be considered volatile, and as such, they wont be included as inputs into the compiled
    function. This means that if they have values stored in the posterior, these values will be
    ignored and new values will be computed (in the case of deterministics and potentials) or
    sampled (in the case of random variables).

    This function also enables a way to impute values for any variable in the computational
    graph that produces the desired outputs: the ``givens_dict``. This dictionary can be used
    to set the ``givens`` argument of the aesara function compilation. This will essentially
    replace a node in the computational graph with any other expression that has the same
    type as the desired node. Passing variables in the givens_dict is considered an intervention
    that might lead to different variable values from those that could have been seen during
    inference, as such, **any variable that is passed in the ``givens_dict`` will be considered
    volatile**.

    Parameters
    ----------
    outputs : List[aesara.graph.basic.Variable]
        The list of variables that will be returned by the compiled function
    vars_in_trace : List[aesara.graph.basic.Variable]
        The list of variables that are assumed to have values stored in the trace
    basic_rvs : Optional[List[aesara.graph.basic.Variable]]
        A list of random variables that are defined in the model. This list (which could be the
        output of ``model.basic_RVs``) should have a reference to the variables that should
        be considered as random variable instances. This includes variables that have
        a ``RandomVariable`` owner op, but also unpure random variables like Mixtures, or
        Censored distributions.
    givens_dict : Optional[Dict[aesara.graph.basic.Variable, Any]]
        A dictionary that maps tensor variables to the values that should be used to replace them
        in the compiled function. The types of the key and value should match or an error will be
        raised during compilation.
    constant_data : Optional[Dict[str, numpy.ndarray]]
        A dictionary that maps the names of ``MutableData`` or ``ConstantData`` instances to their
        corresponding values at inference time. If a model was created with ``MutableData``, these
        are stored as ``SharedVariable`` with the name of the data variable and a value equal to
        the initial data. At inference time, this information is stored in ``InferenceData``
        objects under the ``constant_data`` group, which allows us to check whether a
        ``SharedVariable`` instance changed its values after inference or not. If the values have
        changed, then the ``SharedVariable`` is assumed to be volatile. If it has not changed, then
        the ``SharedVariable`` is assumed to not be volatile. If a ``SharedVariable`` is not found
        in either ``constant_data`` or ``constant_coords``, then it is assumed to be volatile.
        Setting ``constant_data`` to ``None`` is equivalent to passing an empty dictionary.
    constant_coords : Optional[Set[str]]
        A set with the names of the mutable coordinates that have not changed their shape after
        inference. If a model was created with mutable coordinates, these are stored as
        ``SharedVariable`` with the name of the coordinate and a value equal to the length of said
        coordinate. This set let's us check if a ``SharedVariable`` is a mutated coordinate, in
        which case, it is considered volatile. If a ``SharedVariable`` is not found
        in either ``constant_data`` or ``constant_coords``, then it is assumed to be volatile.
        Setting ``constant_coords`` to ``None`` is equivalent to passing an empty set.

    Returns
    -------
    function: Callable
        Compiled forward sampling Aesara function
    volatile_basic_rvs: Set of Variable
        Set of all basic_rvs that were considered volatile and will be resampled when
        the function is evaluated
    """
    if givens_dict is None:
        givens_dict = {}

    if basic_rvs is None:
        basic_rvs = []

    if constant_data is None:
        constant_data = {}
    if constant_coords is None:
        constant_coords = set()

    # We define a helper function to check if shared values match to an array
    def shared_value_matches(var):
        try:
            old_array_value = constant_data[var.name]
        except KeyError:
            return var.name in constant_coords
        current_shared_value = var.get_value(borrow=True)
        return np.array_equal(old_array_value, current_shared_value)

    # We need a function graph to walk the clients and propagate the volatile property
    fg = FunctionGraph(outputs=outputs, clone=False)

    # Walk the graph from inputs to outputs and tag the volatile variables
    nodes: List[Variable] = general_toposort(
        fg.outputs, deps=lambda x: x.owner.inputs if x.owner else []
    )
    volatile_nodes: Set[Any] = set()
    for node in nodes:
        if (
            node in fg.outputs
            or node in givens_dict
            or (  # SharedVariables, except RandomState/Generators
                isinstance(node, SharedVariable)
                and not isinstance(node, (RandomStateSharedVariable, RandomGeneratorSharedVariable))
                and not shared_value_matches(node)
            )
            or (  # Basic RVs that are not in the trace
                node in basic_rvs and node not in vars_in_trace
            )
            or (  # Variables that have any volatile input
                node.owner and any(inp in volatile_nodes for inp in node.owner.inputs)
            )
        ):
            volatile_nodes.add(node)

    # Collect the function inputs by walking the graph from the outputs. Inputs will be:
    # 1. Random variables that are not volatile
    # 2. Variables that have no owner and are not constant or shared
    inputs = []

    def expand(node):
        if (
            (
                node.owner is None and not isinstance(node, (Constant, SharedVariable))
            )  # Variables without owners that are not constant or shared
            or node in vars_in_trace  # Variables in the trace
        ) and node not in volatile_nodes:
            # This test will include variables without owners, and that are not constant
            # or shared, because these nodes will never be considered volatile
            inputs.append(node)
        if node.owner:
            return node.owner.inputs

    # walk produces a generator, so we have to actually exhaust the generator in a list to walk
    # the entire graph
    list(walk(fg.outputs, expand))

    # Populate the givens list
    givens = [
        (
            node,
            value
            if isinstance(value, (Variable, Apply))
            else at.constant(value, dtype=getattr(node, "dtype", None), name=node.name),
        )
        for node, value in givens_dict.items()
    ]

    return (
        compile_pymc(inputs, fg.outputs, givens=givens, on_unused_input="ignore", **kwargs),
        set(basic_rvs) & (volatile_nodes - set(givens_dict)),  # Basic RVs that will be resampled
    )


def sample_posterior_predictive(
    trace,
    model: Optional[Model] = None,
    var_names: Optional[List[str]] = None,
    sample_dims: Optional[List[str]] = None,
    random_seed: RandomState = None,
    progressbar: bool = True,
    return_inferencedata: bool = True,
    extend_inferencedata: bool = False,
    predictions: bool = False,
    idata_kwargs: dict = None,
    compile_kwargs: dict = None,
) -> Union[InferenceData, Dict[str, np.ndarray]]:
    """Generate posterior predictive samples from a model given a trace.

    Parameters
    ----------
    trace : backend, list, xarray.Dataset, arviz.InferenceData, or MultiTrace
        Trace generated from MCMC sampling, or a list of dicts (eg. points or from find_MAP()),
        or xarray.Dataset (eg. InferenceData.posterior or InferenceData.prior)
    model : Model (optional if in ``with`` context)
        Model to be used to generate the posterior predictive samples. It will
        generally be the model used to generate the ``trace``, but it doesn't need to be.
    var_names : Iterable[str]
        Names of variables for which to compute the posterior predictive samples.
    sample_dims : list of str, optional
        Dimensions over which to loop and generate posterior predictive samples.
        When `sample_dims` is ``None`` (default) both "chain" and "draw" are considered sample
        dimensions. Only taken into account when `trace` is InferenceData or Dataset.
    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.
    progressbar : bool
        Whether or not to display a progress bar in the command line. The bar shows the percentage
        of completion, the sampling speed in samples per second (SPS), and the estimated remaining
        time until completion ("expected time of arrival"; ETA).
    return_inferencedata : bool, default True
        Whether to return an :class:`arviz:arviz.InferenceData` (True) object or a dictionary (False).
    extend_inferencedata : bool, default False
        Whether to automatically use :meth:`arviz.InferenceData.extend` to add the posterior predictive samples to
        ``trace`` or not. If True, ``trace`` is modified inplace but still returned.
    predictions : bool, default False
        Choose the function used to convert the samples to inferencedata. See ``idata_kwargs``
        for more details.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data` if ``predictions=False`` or to
        :func:`pymc.predictions_to_inference_data` otherwise.
    compile_kwargs: dict, optional
        Keyword arguments for :func:`pymc.aesaraf.compile_pymc`.

    Returns
    -------
    arviz.InferenceData or Dict
        An ArviZ ``InferenceData`` object containing the posterior predictive samples (default), or
        a dictionary with variable names as keys, and samples as numpy arrays.

    Examples
    --------
    Thin a sampled inferencedata by keeping 1 out of every 5 draws
    before passing it to sample_posterior_predictive

    .. code:: python

        thinned_idata = idata.sel(draw=slice(None, None, 5))
        with model:
            idata.extend(pymc.sample_posterior_predictive(thinned_idata))

    Generate 5 posterior predictive samples per posterior sample.

    .. code:: python

        expanded_data = idata.posterior.expand_dims(pred_id=5)
        with model:
            idata.extend(pymc.sample_posterior_predictive(expanded_data))
    """

    _trace: Union[MultiTrace, PointList]
    nchain: int
    if idata_kwargs is None:
        idata_kwargs = {}
    else:
        idata_kwargs = idata_kwargs.copy()
    if sample_dims is None:
        sample_dims = ["chain", "draw"]
    constant_data: Dict[str, np.ndarray] = {}
    trace_coords: Dict[str, np.ndarray] = {}
    if "coords" not in idata_kwargs:
        idata_kwargs["coords"] = {}
    idata: Optional[InferenceData] = None
    stacked_dims = None
    if isinstance(trace, InferenceData):
        _constant_data = getattr(trace, "constant_data", None)
        if _constant_data is not None:
            trace_coords.update({str(k): v.data for k, v in _constant_data.coords.items()})
            constant_data.update({str(k): v.data for k, v in _constant_data.items()})
        idata = trace
        trace = trace["posterior"]
    if isinstance(trace, xarray.Dataset):
        trace_coords.update({str(k): v.data for k, v in trace.coords.items()})
        _trace, stacked_dims = dataset_to_point_list(trace, sample_dims)
        nchain = 1
    elif isinstance(trace, MultiTrace):
        _trace = trace
        nchain = _trace.nchains
    elif isinstance(trace, list) and all(isinstance(x, dict) for x in trace):
        _trace = trace
        nchain = 1
    else:
        raise TypeError(f"Unsupported type for `trace` argument: {type(trace)}.")
    len_trace = len(_trace)

    if isinstance(_trace, MultiTrace):
        samples = sum(len(v) for v in _trace._straces.values())
    elif isinstance(_trace, list):
        # this is a list of points
        samples = len(_trace)
    else:
        raise TypeError(
            "Do not know how to compute number of samples for trace argument of type %s"
            % type(_trace)
        )

    assert samples is not None

    model = modelcontext(model)

    if model.potentials:
        warnings.warn(
            "The effect of Potentials on other parameters is ignored during posterior predictive sampling. "
            "This is likely to lead to invalid or biased predictive samples.",
            UserWarning,
            stacklevel=2,
        )

    constant_coords = set()
    for dim, coord in trace_coords.items():
        current_coord = model.coords.get(dim, None)
        if (
            current_coord is not None
            and len(coord) == len(current_coord)
            and np.all(coord == current_coord)
        ):
            constant_coords.add(dim)

    if var_names is not None:
        vars_ = [model[x] for x in var_names]
    else:
        vars_ = model.observed_RVs + model.auto_deterministics

    indices = np.arange(samples)
    if progressbar:
        indices = progress_bar(indices, total=samples, display=progressbar)

    vars_to_sample = list(get_default_varnames(vars_, include_transformed=False))

    if not vars_to_sample:
        if return_inferencedata and not extend_inferencedata:
            return InferenceData()
        elif return_inferencedata and extend_inferencedata:
            return trace
        return {}

    vars_in_trace = get_vars_in_point_list(_trace, model)

    if random_seed is not None:
        (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    if compile_kwargs is None:
        compile_kwargs = {}
    compile_kwargs.setdefault("allow_input_downcast", True)
    compile_kwargs.setdefault("accept_inplace", True)

    _sampler_fn, volatile_basic_rvs = compile_forward_sampling_function(
        outputs=vars_to_sample,
        vars_in_trace=vars_in_trace,
        basic_rvs=model.basic_RVs,
        givens_dict=None,
        random_seed=random_seed,
        constant_data=constant_data,
        constant_coords=constant_coords,
        **compile_kwargs,
    )
    sampler_fn = point_wrapper(_sampler_fn)
    # All model variables have a name, but mypy does not know this
    _log.info(f"Sampling: {list(sorted(volatile_basic_rvs, key=lambda var: var.name))}")  # type: ignore
    ppc_trace_t = _DefaultTrace(samples)
    try:
        for idx in indices:
            if nchain > 1:
                # the trace object will either be a MultiTrace (and have _straces)...
                if hasattr(_trace, "_straces"):
                    chain_idx, point_idx = np.divmod(idx, len_trace)
                    chain_idx = chain_idx % nchain
                    param = cast(MultiTrace, _trace)._straces[chain_idx].point(point_idx)
                # ... or a PointList
                else:
                    param = cast(PointList, _trace)[idx % (len_trace * nchain)]
            # there's only a single chain, but the index might hit it multiple times if
            # the number of indices is greater than the length of the trace.
            else:
                param = _trace[idx % len_trace]

            values = sampler_fn(**param)

            for k, v in zip(vars_, values):
                ppc_trace_t.insert(k.name, v, idx)
    except KeyboardInterrupt:
        pass

    ppc_trace = ppc_trace_t.trace_dict

    for k, ary in ppc_trace.items():
        if stacked_dims is not None:
            ppc_trace[k] = ary.reshape(
                (*[len(coord) for coord in stacked_dims.values()], *ary.shape[1:])
            )
        else:
            ppc_trace[k] = ary.reshape((nchain, len_trace, *ary.shape[1:]))

    if not return_inferencedata:
        return ppc_trace
    ikwargs: Dict[str, Any] = dict(model=model, **idata_kwargs)
    ikwargs.setdefault("sample_dims", sample_dims)
    if stacked_dims is not None:
        coords = ikwargs.get("coords", {})
        ikwargs["coords"] = {**stacked_dims, **coords}
    if predictions:
        if extend_inferencedata:
            ikwargs.setdefault("idata_orig", idata)
            ikwargs.setdefault("inplace", True)
        return pm.predictions_to_inference_data(ppc_trace, **ikwargs)
    idata_pp = pm.to_inference_data(posterior_predictive=ppc_trace, **ikwargs)

    if extend_inferencedata and idata is not None:
        idata.extend(idata_pp)
        return idata
    return idata_pp


def sample_posterior_predictive_w(
    traces,
    samples: Optional[int] = None,
    models: Optional[List[Model]] = None,
    weights: Optional[ArrayLike] = None,
    random_seed: RandomState = None,
    progressbar: bool = True,
    return_inferencedata: bool = True,
    idata_kwargs: dict = None,
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
    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.
    progressbar : bool, optional default True
        Whether or not to display a progress bar in the command line. The bar shows the percentage
        of completion, the sampling speed in samples per second (SPS), and the estimated remaining
        time until completion ("expected time of arrival"; ETA).
    return_inferencedata : bool
        Whether to return an :class:`arviz:arviz.InferenceData` (True) object or a dictionary (False).
        Defaults to True.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data`

    Returns
    -------
    arviz.InferenceData or Dict
        An ArviZ ``InferenceData`` object containing the posterior predictive samples from the
        weighted models (default), or a dictionary with variable names as keys, and samples as
        numpy arrays.
    """
    raise NotImplementedError(f"sample_posterior_predictive_w has not yet been ported to PyMC 4.0.")

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

    if random_seed is not None:
        (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    for model in models:
        if model.potentials:
            warnings.warn(
                "The effect of Potentials on other parameters is ignored during posterior predictive sampling. "
                "This is likely to lead to invalid or biased predictive samples.",
                UserWarning,
                stacklevel=2,
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
            for cidx, pidx in zip(chain_idx, point_idx):
                trace.append(tr._straces[cidx].point(pidx))
        else:
            for idx in indices:
                trace.append(tr[idx])

    obs = [x for m in models for x in m.observed_RVs]
    variables = np.repeat(obs, n)

    lengths = list({np.atleast_1d(observed).shape for observed in obs})

    size: List[Optional[Tuple[int, ...]]] = []
    if len(lengths) == 1:
        size = [None] * len(variables)
    elif len(lengths) > 2:
        raise ValueError("Observed variables could not be broadcast together")
    else:
        x = np.zeros(shape=lengths[0])
        y = np.zeros(shape=lengths[1])
        b = np.broadcast(x, y)
        for var in variables:
            # XXX: This needs to be refactored
            shape = None  # np.shape(np.atleast_1d(var.distribution.default()))
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
        ppcl: Dict[str, list] = defaultdict(list)
        for idx in indices:
            param = trace[idx]
            var = variables[idx]
            # TODO sample_posterior_predictive_w is currently only work for model with
            # one observed.
            # XXX: This needs to be refactored
            # ppc[var.name].append(draw_values([var], point=param, size=size[idx])[0])
            raise NotImplementedError()

    except KeyboardInterrupt:
        pass
    else:
        ppcd = {k: np.asarray(v) for k, v in ppcl.items()}
        if not return_inferencedata:
            return ppcd
        ikwargs: Dict[str, Any] = dict(model=models)
        if idata_kwargs:
            ikwargs.update(idata_kwargs)
        return pm.to_inference_data(posterior_predictive=ppcd, **ikwargs)


def sample_prior_predictive(
    samples: int = 500,
    model: Optional[Model] = None,
    var_names: Optional[Iterable[str]] = None,
    random_seed: RandomState = None,
    return_inferencedata: bool = True,
    idata_kwargs: dict = None,
    compile_kwargs: dict = None,
) -> Union[InferenceData, Dict[str, np.ndarray]]:
    """Generate samples from the prior predictive distribution.

    Parameters
    ----------
    samples : int
        Number of samples from the prior predictive to generate. Defaults to 500.
    model : Model (optional if in ``with`` context)
    var_names : Iterable[str]
        A list of names of variables for which to compute the prior predictive
        samples. Defaults to both observed and unobserved RVs. Transformed values
        are not included unless explicitly defined in var_names.
    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.
    return_inferencedata : bool
        Whether to return an :class:`arviz:arviz.InferenceData` (True) object or a dictionary (False).
        Defaults to True.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data`
    compile_kwargs: dict, optional
        Keyword arguments for :func:`pymc.aesaraf.compile_pymc`.

    Returns
    -------
    arviz.InferenceData or Dict
        An ArviZ ``InferenceData`` object containing the prior and prior predictive samples (default),
        or a dictionary with variable names as keys and samples as numpy arrays.
    """
    model = modelcontext(model)

    if model.potentials:
        warnings.warn(
            "The effect of Potentials on other parameters is ignored during prior predictive sampling. "
            "This is likely to lead to invalid or biased predictive samples.",
            UserWarning,
            stacklevel=2,
        )

    if var_names is None:
        prior_pred_vars = model.observed_RVs + model.auto_deterministics
        prior_vars = (
            get_default_varnames(model.unobserved_RVs, include_transformed=True) + model.potentials
        )
        vars_: Set[str] = {var.name for var in prior_vars + prior_pred_vars}
    else:
        vars_ = set(var_names)

    names = sorted(get_default_varnames(vars_, include_transformed=False))
    vars_to_sample = [model[name] for name in names]

    # Any variables from var_names that are missing must be transformed variables.
    # Misspelled variables would have raised a KeyError above.
    missing_names = vars_.difference(names)
    for name in sorted(missing_names):
        transformed_value_var = model[name]
        rv_var = model.values_to_rvs[transformed_value_var]
        transform = transformed_value_var.tag.transform
        transformed_rv_var = transform.forward(rv_var, *rv_var.owner.inputs)

        names.append(name)
        vars_to_sample.append(transformed_rv_var)

        # If the user asked for the transformed variable in var_names, but not the
        # original RV, we add it manually here
        if rv_var.name not in names:
            names.append(rv_var.name)
            vars_to_sample.append(rv_var)

    if random_seed is not None:
        (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    if compile_kwargs is None:
        compile_kwargs = {}
    compile_kwargs.setdefault("allow_input_downcast", True)
    compile_kwargs.setdefault("accept_inplace", True)

    sampler_fn, volatile_basic_rvs = compile_forward_sampling_function(
        vars_to_sample,
        vars_in_trace=[],
        basic_rvs=model.basic_RVs,
        givens_dict=None,
        random_seed=random_seed,
        **compile_kwargs,
    )

    # All model variables have a name, but mypy does not know this
    _log.info(f"Sampling: {list(sorted(volatile_basic_rvs, key=lambda var: var.name))}")  # type: ignore
    values = zip(*(sampler_fn() for i in range(samples)))

    data = {k: np.stack(v) for k, v in zip(names, values)}
    if data is None:
        raise AssertionError("No variables sampled: attempting to sample %s" % names)

    prior: Dict[str, np.ndarray] = {}
    for var_name in vars_:
        if var_name in data:
            prior[var_name] = data[var_name]

    if not return_inferencedata:
        return prior
    ikwargs: Dict[str, Any] = dict(model=model)
    if idata_kwargs:
        ikwargs.update(idata_kwargs)
    return pm.to_inference_data(prior=prior, **ikwargs)


def draw(
    vars: Union[Variable, Sequence[Variable]],
    draws: int = 1,
    random_seed: RandomState = None,
    **kwargs,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Draw samples for one variable or a list of variables

    Parameters
    ----------
    vars : TensorVariable or iterable of TensorVariable
        A variable or a list of variables for which to draw samples.
    draws : int, default 1
        Number of samples needed to draw.
    random_seed : int, RandomState or numpy_Generator, optional
        Seed for the random number generator.
    **kwargs : dict, optional
        Keyword arguments for :func:`pymc.aesaraf.compile_pymc`.

    Returns
    -------
    list of ndarray
        A list of numpy arrays.

    Examples
    --------
        .. code-block:: python

            import pymc as pm

            # Draw samples for one variable
            with pm.Model():
                x = pm.Normal("x")
            x_draws = pm.draw(x, draws=100)
            print(x_draws.shape)

            # Draw 1000 samples for several variables
            with pm.Model():
                x = pm.Normal("x")
                y = pm.Normal("y", shape=10)
                z = pm.Uniform("z", shape=5)
            num_draws = 1000
            # Draw samples of a list variables
            draws = pm.draw([x, y, z], draws=num_draws)
            assert draws[0].shape == (num_draws,)
            assert draws[1].shape == (num_draws, 10)
            assert draws[2].shape == (num_draws, 5)
    """
    if random_seed is not None:
        (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    draw_fn = compile_pymc(inputs=[], outputs=vars, random_seed=random_seed, **kwargs)

    if draws == 1:
        return draw_fn()

    # Single variable output
    if not isinstance(vars, (list, tuple)):
        cast(Callable[[], np.ndarray], draw_fn)
        return np.stack([draw_fn() for _ in range(draws)])

    # Multiple variable output
    cast(Callable[[], List[np.ndarray]], draw_fn)
    drawn_values = zip(*(draw_fn() for _ in range(draws)))
    return [np.stack(v) for v in drawn_values]


def _init_jitter(
    model: Model,
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]],
    seeds: Union[Sequence[int], np.ndarray],
    jitter: bool,
    jitter_max_retries: int,
) -> List[PointType]:
    """Apply a uniform jitter in [-1, 1] to the test value as starting point in each chain.

    ``model.check_start_vals`` is used to test whether the jittered starting
    values produce a finite log probability. Invalid values are resampled
    unless `jitter_max_retries` is achieved, in which case the last sampled
    values are returned.

    Parameters
    ----------
    jitter: bool
        Whether to apply jitter or not.
    jitter_max_retries : int
        Maximum number of repeated attempts at initializing values (per chain).

    Returns
    -------
    start : ``pymc.model.Point``
        Starting point for sampler
    """

    ipfns = make_initial_point_fns_per_chain(
        model=model,
        overrides=initvals,
        jitter_rvs=set(model.free_RVs) if jitter else set(),
        chains=len(seeds),
    )

    if not jitter:
        return [ipfn(seed) for ipfn, seed in zip(ipfns, seeds)]

    initial_points = []
    for ipfn, seed in zip(ipfns, seeds):
        rng = np.random.RandomState(seed)
        for i in range(jitter_max_retries + 1):
            point = ipfn(seed)
            if i < jitter_max_retries:
                try:
                    model.check_start_vals(point)
                except SamplingError:
                    # Retry with a new seed
                    seed = rng.randint(2**30, dtype=np.int64)
                else:
                    break
        initial_points.append(point)
    return initial_points


def init_nuts(
    *,
    init: str = "auto",
    chains: int = 1,
    n_init: int = 500_000,
    model=None,
    random_seed: RandomSeed = None,
    progressbar=True,
    jitter_max_retries: int = 10,
    tune: Optional[int] = None,
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
    **kwargs,
) -> Tuple[Sequence[PointType], NUTS]:
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
        * jitter+adapt_diag_grad:
          An experimental initialization method that uses information from gradients and samples
          during tuning.
        * advi+adapt_diag: Run ADVI and then adapt the resulting diagonal mass matrix based on the
          sample variance of the tuning samples.
        * advi: Run ADVI to estimate posterior mean and diagonal mass matrix.
        * advi_map: Initialize ADVI with MAP and use MAP as starting point.
        * map: Use the MAP as starting point. This is discouraged.
        * adapt_full: Adapt a dense mass matrix using the sample covariances. All chains use the
          test value (usually the prior mean) as starting point.
        * jitter+adapt_full: Same as ``adapt_full``, but use test value plus a uniform jitter in
          [-1, 1] as starting point in each chain.

    chains : int
        Number of jobs to start.
    initvals : optional, dict or list of dicts
        Dict or list of dicts with initial values to use instead of the defaults from `Model.initial_values`.
        The keys should be names of transformed random variables.
    n_init : int
        Number of iterations of initializer. Only works for 'ADVI' init methods.
    model : Model (optional if in ``with`` context)
    random_seed : int, array-like of int, RandomState or Generator, optional
        Seed for the random number generator.
    progressbar : bool
        Whether or not to display a progressbar for advi sampling.
    jitter_max_retries : int
        Maximum number of repeated attempts (per chain) at creating an initial matrix with uniform jitter
        that yields a finite probability. This applies to ``jitter+adapt_diag`` and ``jitter+adapt_full``
        init methods.
    **kwargs : keyword arguments
        Extra keyword arguments are forwarded to pymc.NUTS.

    Returns
    -------
    initial_points : list
        Starting points for each chain.
    nuts_sampler : ``pymc.step_methods.NUTS``
        Instantiated and initialized NUTS sampler object
    """
    model = modelcontext(model)

    vars = kwargs.get("vars", model.value_vars)
    if set(vars) != set(model.value_vars):
        raise ValueError("Must use init_nuts on all variables of a model.")
    if not all_continuous(vars):
        raise ValueError("init_nuts can only be used for models with continuous variables.")

    if not isinstance(init, str):
        raise TypeError("init must be a string.")

    init = init.lower()

    if init == "auto":
        init = "jitter+adapt_diag"

    random_seed_list = _get_seeds_per_chain(random_seed, chains)

    _log.info(f"Initializing NUTS using {init}...")

    cb = [
        pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff="absolute"),
        pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff="relative"),
    ]

    initial_points = _init_jitter(
        model,
        initvals,
        seeds=random_seed_list,
        jitter="jitter" in init,
        jitter_max_retries=jitter_max_retries,
    )

    apoints = [DictToArrayBijection.map(point) for point in initial_points]
    apoints_data = [apoint.data for apoint in apoints]
    potential: quadpotential.QuadPotential

    if init == "adapt_diag":
        mean = np.mean(apoints_data, axis=0)
        var = np.ones_like(mean)
        n = len(var)
        potential = quadpotential.QuadPotentialDiagAdapt(n, mean, var, 10)
    elif init == "jitter+adapt_diag":
        mean = np.mean(apoints_data, axis=0)
        var = np.ones_like(mean)
        n = len(var)
        potential = quadpotential.QuadPotentialDiagAdapt(n, mean, var, 10)
    elif init == "jitter+adapt_diag_grad":
        mean = np.mean(apoints_data, axis=0)
        var = np.ones_like(mean)
        n = len(var)

        if tune is not None and tune > 250:
            stop_adaptation = tune - 50
        else:
            stop_adaptation = None

        potential = quadpotential.QuadPotentialDiagAdaptExp(
            n,
            mean,
            alpha=0.02,
            use_grads=True,
            stop_adaptation=stop_adaptation,
        )
    elif init == "advi+adapt_diag":
        approx = pm.fit(
            random_seed=random_seed_list[0],
            n=n_init,
            method="advi",
            model=model,
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window,
        )
        approx_sample = approx.sample(
            draws=chains, random_seed=random_seed_list[0], return_inferencedata=False
        )
        initial_points = [approx_sample[i] for i in range(chains)]
        std_apoint = approx.std.eval()
        cov = std_apoint**2
        mean = approx.mean.get_value()
        weight = 50
        n = len(cov)
        potential = quadpotential.QuadPotentialDiagAdapt(n, mean, cov, weight)
    elif init == "advi":
        approx = pm.fit(
            random_seed=random_seed_list[0],
            n=n_init,
            method="advi",
            model=model,
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window,
        )
        approx_sample = approx.sample(
            draws=chains, random_seed=random_seed_list[0], return_inferencedata=False
        )
        initial_points = [approx_sample[i] for i in range(chains)]
        cov = approx.std.eval() ** 2
        potential = quadpotential.QuadPotentialDiag(cov)
    elif init == "advi_map":
        start = pm.find_MAP(include_transformed=True, seed=random_seed_list[0])
        approx = pm.MeanField(model=model, start=start)
        pm.fit(
            random_seed=random_seed_list[0],
            n=n_init,
            method=pm.KLqp(approx),
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window,
        )
        approx_sample = approx.sample(
            draws=chains, random_seed=random_seed_list[0], return_inferencedata=False
        )
        initial_points = [approx_sample[i] for i in range(chains)]
        cov = approx.std.eval() ** 2
        potential = quadpotential.QuadPotentialDiag(cov)
    elif init == "map":
        start = pm.find_MAP(include_transformed=True, seed=random_seed_list[0])
        cov = pm.find_hessian(point=start)
        initial_points = [start] * chains
        potential = quadpotential.QuadPotentialFull(cov)
    elif init == "adapt_full":
        mean = np.mean(apoints_data * chains, axis=0)
        initial_point = initial_points[0]
        initial_point_model_size = sum(initial_point[n.name].size for n in model.value_vars)
        cov = np.eye(initial_point_model_size)
        potential = quadpotential.QuadPotentialFullAdapt(initial_point_model_size, mean, cov, 10)
    elif init == "jitter+adapt_full":
        mean = np.mean(apoints_data, axis=0)
        initial_point = initial_points[0]
        initial_point_model_size = sum(initial_point[n.name].size for n in model.value_vars)
        cov = np.eye(initial_point_model_size)
        potential = quadpotential.QuadPotentialFullAdapt(initial_point_model_size, mean, cov, 10)
    else:
        raise ValueError(f"Unknown initializer: {init}.")

    step = pm.NUTS(potential=potential, model=model, **kwargs)

    # Filter deterministics from initial_points
    value_var_names = [var.name for var in model.value_vars]
    initial_points = [
        {k: v for k, v in initial_point.items() if k in value_var_names}
        for initial_point in initial_points
    ]

    return initial_points, step
