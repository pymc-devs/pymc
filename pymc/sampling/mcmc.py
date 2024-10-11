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

"""Functions for MCMC sampling."""

import contextlib
import logging
import pickle
import sys
import time
import warnings

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import (
    Any,
    Literal,
    TypeAlias,
    overload,
)

import numpy as np
import pytensor.gradient as tg

from arviz import InferenceData, dict_to_dataset
from arviz.data.base import make_attrs
from pytensor.graph.basic import Variable
from rich.console import Console
from rich.progress import BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.theme import Theme
from threadpoolctl import threadpool_limits
from typing_extensions import Protocol

import pymc as pm

from pymc.backends import RunType, TraceOrBackend, init_traces
from pymc.backends.arviz import (
    coords_and_dims_for_inferencedata,
    find_constants,
    find_observations,
)
from pymc.backends.base import IBaseTrace, MultiTrace, _choose_chains
from pymc.blocking import DictToArrayBijection
from pymc.exceptions import SamplingError
from pymc.initial_point import PointType, StartDict, make_initial_point_fns_per_chain
from pymc.model import Model, modelcontext
from pymc.sampling.parallel import Draw, _cpu_count
from pymc.sampling.population import _sample_population
from pymc.stats.convergence import (
    log_warning_stats,
    log_warnings,
    run_convergence_checks,
)
from pymc.step_methods import NUTS, CompoundStep
from pymc.step_methods.arraystep import BlockedStep, PopulationArrayStepShared
from pymc.step_methods.hmc import quadpotential
from pymc.util import (
    CustomProgress,
    RandomSeed,
    RandomState,
    _get_seeds_per_chain,
    default_progress_theme,
    drop_warning_stat,
    get_random_generator,
    get_untransformed_name,
    is_transformed_name,
)
from pymc.vartypes import discrete_types

sys.setrecursionlimit(10000)

__all__ = [
    "sample",
    "init_nuts",
]

Step: TypeAlias = BlockedStep | CompoundStep


class SamplingIteratorCallback(Protocol):
    """Signature of the callable that may be passed to `pm.sample(callable=...)`."""

    def __call__(self, trace: IBaseTrace, draw: Draw):
        pass


_log = logging.getLogger(__name__)


def instantiate_steppers(
    model: Model,
    steps: list[Step],
    selected_steps: Mapping[type[BlockedStep], list[Any]],
    step_kwargs: dict[str, dict] | None = None,
) -> Step | list[Step]:
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
            name = getattr(step_class, "name")
            args = step_kwargs.get(name, {})
            used_keys.add(name)
            step = step_class(vars=vars, model=model, **args)
            steps.append(step)

    unused_args = set(step_kwargs).difference(used_keys)
    if unused_args:
        s = "s" if len(unused_args) > 1 else ""
        example_arg = sorted(unused_args)[0]
        example_step = (list(selected_steps.keys()) or pm.STEP_METHODS)[0]
        example_step_name = getattr(example_step, "name")
        raise ValueError(
            f"Invalid key{s} found in step_kwargs: {unused_args}. "
            "Keys must be step names and values valid kwargs for that stepper. "
            f'Did you mean {{"{example_step_name}": {{"{example_arg}": ...}}}}?'
        )

    if len(steps) == 1:
        return steps[0]

    return steps


def assign_step_methods(
    model: Model,
    step: Step | Sequence[Step] | None = None,
    methods: Sequence[type[BlockedStep]] | None = None,
    step_kwargs: dict[str, Any] | None = None,
) -> Step | list[Step]:
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
    steps: list[Step] = []
    assigned_vars: set[Variable] = set()

    if step is not None:
        if isinstance(step, BlockedStep | CompoundStep):
            steps.append(step)
        else:
            steps.extend(step)
        for step in steps:
            for var in step.vars:
                if var not in model.value_vars:
                    raise ValueError(
                        f"{var} assigned to {step} sampler is not a value variable in the model. "
                        "You can use `util.get_value_vars_from_user_vars` to parse user provided variables."
                    )
            assigned_vars = assigned_vars.union(set(step.vars))

    # Use competence classmethods to select step methods for remaining
    # variables
    methods_list: list[type[BlockedStep]] = list(methods or pm.STEP_METHODS)
    selected_steps: dict[type[BlockedStep], list] = {}
    model_logp = model.logp()

    for var in model.value_vars:
        if var not in assigned_vars:
            # determine if a gradient can be computed
            has_gradient = getattr(var, "dtype") not in discrete_types
            if has_gradient:
                try:
                    tg.grad(model_logp, var)  # type: ignore[arg-type]
                except (NotImplementedError, tg.NullTypeGradError):
                    has_gradient = False

            # select the best method
            rv_var = model.values_to_rvs[var]
            selected = max(
                methods_list,
                key=lambda method, var=rv_var, has_gradient=has_gradient: method._competence(  # type: ignore[misc]
                    var, has_gradient
                ),
            )
            selected_steps.setdefault(selected, []).append(var)

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
    """Check that vars not include discrete variables."""
    if any((var.dtype in discrete_types) for var in vars):
        return False
    else:
        return True


def _sample_external_nuts(
    sampler: Literal["nutpie", "numpyro", "blackjax"],
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    random_seed: RandomState | None,
    initvals: StartDict | Sequence[StartDict | None] | None,
    model: Model,
    var_names: Sequence[str] | None,
    progressbar: bool,
    idata_kwargs: dict | None,
    compute_convergence_checks: bool,
    nuts_sampler_kwargs: dict | None,
    **kwargs,
):
    if nuts_sampler_kwargs is None:
        nuts_sampler_kwargs = {}

    if sampler == "nutpie":
        try:
            import nutpie
        except ImportError as err:
            raise ImportError(
                "nutpie not found. Install it with conda install -c conda-forge nutpie"
            ) from err

        if initvals is not None:
            warnings.warn(
                "`initvals` are currently not passed to nutpie sampler. "
                "Use `init_mean` kwarg following nutpie specification instead.",
                UserWarning,
            )

        if idata_kwargs is not None:
            warnings.warn(
                "`idata_kwargs` are currently ignored by the nutpie sampler",
                UserWarning,
            )
        if var_names is not None:
            warnings.warn(
                "`var_names` are currently ignored by the nutpie sampler",
                UserWarning,
            )
        compile_kwargs = {}
        for kwarg in ("backend", "gradient_backend"):
            if kwarg in nuts_sampler_kwargs:
                compile_kwargs[kwarg] = nuts_sampler_kwargs.pop(kwarg)
        compiled_model = nutpie.compile_pymc_model(
            model,
            **compile_kwargs,
        )
        t_start = time.time()
        idata = nutpie.sample(
            compiled_model,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            seed=_get_seeds_per_chain(random_seed, 1)[0],
            progress_bar=progressbar,
            **nuts_sampler_kwargs,
        )
        t_sample = time.time() - t_start
        # Temporary work-around. Revert once https://github.com/pymc-devs/nutpie/issues/74 is fixed
        # gather observed and constant data as nutpie.sample() has no access to the PyMC model
        coords, dims = coords_and_dims_for_inferencedata(model)
        constant_data = dict_to_dataset(
            find_constants(model),
            library=pm,
            coords=coords,
            dims=dims,
            default_dims=[],
        )
        observed_data = dict_to_dataset(
            find_observations(model),
            library=pm,
            coords=coords,
            dims=dims,
            default_dims=[],
        )
        attrs = make_attrs(
            {
                "sampling_time": t_sample,
                "tuning_steps": tune,
            },
            library=nutpie,
        )
        for k, v in attrs.items():
            idata.posterior.attrs[k] = v
        idata.add_groups(
            {"constant_data": constant_data, "observed_data": observed_data},
            coords=coords,
            dims=dims,
        )
        return idata

    elif sampler in ("numpyro", "blackjax"):
        import pymc.sampling.jax as pymc_jax

        idata = pymc_jax.sample_jax_nuts(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            initvals=initvals,
            model=model,
            var_names=var_names,
            progressbar=progressbar,
            nuts_sampler=sampler,
            idata_kwargs=idata_kwargs,
            compute_convergence_checks=compute_convergence_checks,
            **nuts_sampler_kwargs,
        )
        return idata

    else:
        raise ValueError(
            f"Sampler {sampler} not found. Choose one of ['nutpie', 'numpyro', 'blackjax', 'pymc']."
        )


@overload
def sample(
    draws: int = 1000,
    *,
    tune: int = 1000,
    chains: int | None = None,
    cores: int | None = None,
    random_seed: RandomState = None,
    progressbar: bool = True,
    progressbar_theme: Theme | None = default_progress_theme,
    step=None,
    var_names: Sequence[str] | None = None,
    nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] = "pymc",
    initvals: StartDict | Sequence[StartDict | None] | None = None,
    init: str = "auto",
    jitter_max_retries: int = 10,
    n_init: int = 200_000,
    trace: TraceOrBackend | None = None,
    discard_tuned_samples: bool = True,
    compute_convergence_checks: bool = True,
    keep_warning_stat: bool = False,
    return_inferencedata: Literal[True] = True,
    idata_kwargs: dict[str, Any] | None = None,
    nuts_sampler_kwargs: dict[str, Any] | None = None,
    callback=None,
    mp_ctx=None,
    blas_cores: int | None | Literal["auto"] = "auto",
    **kwargs,
) -> InferenceData: ...


@overload
def sample(
    draws: int = 1000,
    *,
    tune: int = 1000,
    chains: int | None = None,
    cores: int | None = None,
    random_seed: RandomState = None,
    progressbar: bool = True,
    progressbar_theme: Theme | None = default_progress_theme,
    step=None,
    var_names: Sequence[str] | None = None,
    nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] = "pymc",
    initvals: StartDict | Sequence[StartDict | None] | None = None,
    init: str = "auto",
    jitter_max_retries: int = 10,
    n_init: int = 200_000,
    trace: TraceOrBackend | None = None,
    discard_tuned_samples: bool = True,
    compute_convergence_checks: bool = True,
    keep_warning_stat: bool = False,
    return_inferencedata: Literal[False],
    idata_kwargs: dict[str, Any] | None = None,
    nuts_sampler_kwargs: dict[str, Any] | None = None,
    callback=None,
    mp_ctx=None,
    model: Model | None = None,
    blas_cores: int | None | Literal["auto"] = "auto",
    **kwargs,
) -> MultiTrace: ...


def sample(
    draws: int = 1000,
    *,
    tune: int = 1000,
    chains: int | None = None,
    cores: int | None = None,
    random_seed: RandomState = None,
    progressbar: bool = True,
    progressbar_theme: Theme | None = default_progress_theme,
    step=None,
    var_names: Sequence[str] | None = None,
    nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] = "pymc",
    initvals: StartDict | Sequence[StartDict | None] | None = None,
    init: str = "auto",
    jitter_max_retries: int = 10,
    n_init: int = 200_000,
    trace: TraceOrBackend | None = None,
    discard_tuned_samples: bool = True,
    compute_convergence_checks: bool = True,
    keep_warning_stat: bool = False,
    return_inferencedata: bool = True,
    idata_kwargs: dict[str, Any] | None = None,
    nuts_sampler_kwargs: dict[str, Any] | None = None,
    callback=None,
    mp_ctx=None,
    blas_cores: int | None | Literal["auto"] = "auto",
    model: Model | None = None,
    **kwargs,
) -> InferenceData | MultiTrace:
    r"""Draw samples from the posterior using the given step methods.

    Multiple step methods are supported via compound step methods.

    Parameters
    ----------
    draws : int
        The number of samples to draw. Defaults to 1000. The number of tuned samples are discarded
        by default. See ``discard_tuned_samples``.
    tune : int
        Number of iterations to tune, defaults to 1000. Samplers adjust the step sizes, scalings or
        similar during tuning. Tuning samples will be drawn in addition to the number specified in
        the ``draws`` argument, and will be discarded unless ``discard_tuned_samples`` is set to
        False.
    chains : int
        The number of chains to sample. Running independent chains is important for some
        convergence statistics and can also reveal multiple modes in the posterior. If ``None``,
        then set to either ``cores`` or 2, whichever is larger.
    cores : int
        The number of chains to run in parallel. If ``None``, set to the number of CPUs in the
        system, but at most 4.
    random_seed : int, array-like of int, or Generator, optional
        Random seed(s) used by the sampling steps. Each step will create its own
        :py:class:`~numpy.random.Generator` object to make its random draws in a way that is
        indepedent from all other steppers and all other chains. If a list, tuple or array of ints
        is passed, each entry will be used to seed the creation of ``Generator`` objects.
        A ``ValueError`` will be raised if the length does not match the number of chains.
        A ``TypeError`` will be raised if a :py:class:`~numpy.random.RandomState` object is passed.
        We no longer support ``RandomState`` objects because their seeding mechanism does not allow
        easy spawning of new independent random streams that are needed by the step methods.
    progressbar : bool, optional default=True
        Whether or not to display a progress bar in the command line. The bar shows the percentage
        of completion, the sampling speed in samples per second (SPS), and the estimated remaining
        time until completion ("expected time of arrival"; ETA).
        Only applicable to the pymc nuts sampler.
    step : function or iterable of functions
        A step function or collection of functions. If there are variables without step methods,
        step methods for those variables will be assigned automatically. By default the NUTS step
        method will be used, if appropriate to the model.
    var_names : list of str, optional
        Names of variables to be stored in the trace. Defaults to all free variables and deterministics.
    nuts_sampler : str
        Which NUTS implementation to run. One of ["pymc", "nutpie", "blackjax", "numpyro"].
        This requires the chosen sampler to be installed.
        All samplers, except "pymc", require the full model to be continuous.
    blas_cores: int or "auto" or None, default = "auto"
        The total number of threads blas and openmp functions should use during sampling.
        Setting it to "auto" will ensure that the total number of active blas threads is the
        same as the `cores` argument. If set to an integer, the sampler will try to use that total
        number of blas threads. If `blas_cores` is not divisible by `cores`, it might get rounded
        down. If set to None, this will keep the default behavior of whatever blas implementation
        is used at runtime.
    initvals : optional, dict, array of dict
        Dict or list of dicts with initial value strategies to use instead of the defaults from
        `Model.initial_values`. The keys should be names of transformed random variables.
        Initialization methods for NUTS (see ``init`` keyword) can overwrite the default.
    init : str
        Initialization method to use for auto-assigned NUTS samplers. See `pm.init_nuts` for a list
        of all options. This argument is ignored when manually passing the NUTS step method.
        Only applicable to the pymc nuts sampler.
    jitter_max_retries : int
        Maximum number of repeated attempts (per chain) at creating an initial matrix with uniform
        jitter that yields a finite probability. This applies to ``jitter+adapt_diag`` and
        ``jitter+adapt_full`` init methods.
    n_init : int
        Number of iterations of initializer. Only works for 'ADVI' init methods.
    trace : backend, optional
        A backend instance or None.
        If None, the NDArray backend is used.
    discard_tuned_samples : bool
        Whether to discard posterior samples of the tune interval.
    compute_convergence_checks : bool, default=True
        Whether to compute sampler statistics like Gelman-Rubin and ``effective_n``.
    keep_warning_stat : bool
        If ``True`` the "warning" stat emitted by, for example, HMC samplers will be kept
        in the returned ``idata.sample_stats`` group.
        This leads to the ``idata`` not supporting ``.to_netcdf()`` or ``.to_zarr()`` and
        should only be set to ``True`` if you intend to use the "warning" objects right away.
        Defaults to ``False`` such that ``pm.drop_warning_stat`` is applied automatically,
        making the ``InferenceData`` compatible with saving.
    return_inferencedata : bool
        Whether to return the trace as an :class:`arviz:arviz.InferenceData` (True) object or a
        `MultiTrace` (False). Defaults to `True`.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data`
    nuts_sampler_kwargs : dict, optional
        Keyword arguments for the sampling library that implements nuts.
        Only used when an external sampler is specified via the `nuts_sampler` kwarg.
    callback : function, default=None
        A function which gets called for every sample from the trace of a chain. The function is
        called with the trace and the current draw and will contain all samples for a single trace.
        the ``draw.chain`` argument can be used to determine which of the active chains the sample
        is drawn from.
        Sampling can be interrupted by throwing a ``KeyboardInterrupt`` in the callback.
    mp_ctx : multiprocessing.context.BaseContent
        A multiprocessing context for parallel sampling.
        See multiprocessing documentation for details.
    model : Model (optional if in ``with`` context)
        Model to sample from. The model needs to have free random variables.

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

        step = [
            pm.NUTS([freeRV1, freeRV2], target_accept=0.9),
            pm.BinaryGibbsMetropolis([freeRV3], transit_p=0.7),
        ]

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
    if nuts_sampler_kwargs is None:
        nuts_sampler_kwargs = {}
    if "target_accept" in kwargs:
        if "nuts" in kwargs and "target_accept" in kwargs["nuts"]:
            raise ValueError(
                "`target_accept` was defined twice. Please specify it either as a direct keyword argument or in the `nuts` kwarg."
            )
        if "nuts" in kwargs:
            kwargs["nuts"]["target_accept"] = kwargs.pop("target_accept")
        else:
            kwargs["nuts"] = {"target_accept": kwargs.pop("target_accept")}
    if isinstance(trace, list):
        raise ValueError("Please use `var_names` keyword argument for partial traces.")

    model = modelcontext(model)
    if not model.free_RVs:
        raise SamplingError(
            "Cannot sample from the model, since the model does not contain any free variables."
        )

    if cores is None:
        cores = min(4, _cpu_count())

    if chains is None:
        chains = max(2, cores)

    if blas_cores == "auto":
        blas_cores = cores

    cores = min(cores, chains)

    num_blas_cores_per_chain: int | None
    joined_blas_limiter: Callable[[], Any]

    if blas_cores is None:
        joined_blas_limiter = contextlib.nullcontext
        num_blas_cores_per_chain = None
    elif isinstance(blas_cores, int):

        def joined_blas_limiter():
            return threadpool_limits(limits=blas_cores)

        num_blas_cores_per_chain = blas_cores // cores
    else:
        raise ValueError(
            f"Invalid argument `blas_cores`, must be int, 'auto' or None: {blas_cores}"
        )

    if random_seed == -1:
        random_seed = None
    rngs = get_random_generator(random_seed).spawn(chains)
    random_seed_list = [rng.integers(2**30) for rng in rngs]

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
    elif draws < 100:
        msg = f"Only {draws} samples per chain. Reliable r-hat and ESS diagnostics require longer chains for accurate estimate."
        _log.warning(msg)

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

    if nuts_sampler != "pymc":
        if not isinstance(step, NUTS):
            raise ValueError(
                "Model can not be sampled with NUTS alone. Your model is probably not continuous."
            )

        with joined_blas_limiter():
            return _sample_external_nuts(
                sampler=nuts_sampler,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=kwargs.pop("nuts", {}).get("target_accept", 0.8),
                random_seed=random_seed,
                initvals=initvals,
                model=model,
                var_names=var_names,
                progressbar=progressbar,
                idata_kwargs=idata_kwargs,
                compute_convergence_checks=compute_convergence_checks,
                nuts_sampler_kwargs=nuts_sampler_kwargs,
                **kwargs,
            )

    if isinstance(step, list):
        step = CompoundStep(step)
    elif isinstance(step, NUTS) and auto_nuts_init:
        if "nuts" in kwargs:
            nuts_kwargs = kwargs.pop("nuts")
            [kwargs.setdefault(k, v) for k, v in nuts_kwargs.items()]
        _log.info("Auto-assigning NUTS sampler...")
        with joined_blas_limiter():
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
            jitter_rvs=set(),
            chains=chains,
        )
        initial_points = [ipfn(seed) for ipfn, seed in zip(ipfns, random_seed_list)]

    # One final check that shapes and logps at the starting points are okay.
    ip: dict[str, np.ndarray]
    for ip in initial_points:
        model.check_start_vals(ip)
        _check_start_shape(model, ip)

    if var_names is not None:
        trace_vars = [v for v in model.unobserved_RVs if v.name in var_names]
        trace_vars = model.replace_rvs_by_values(trace_vars)
        assert len(trace_vars) == len(var_names), "Not all var_names were found in the model"
    else:
        trace_vars = None

    # Create trace backends for each chain
    run, traces = init_traces(
        backend=trace,
        chains=chains,
        expected_length=draws + tune,
        step=step,
        trace_vars=trace_vars,
        initial_point=ip,
        model=model,
    )

    sample_args = {
        # draws is now the total number of draws, including tuning
        "draws": draws + tune,
        "step": step,
        "start": initial_points,
        "traces": traces,
        "chains": chains,
        "tune": tune,
        "var_names": var_names,
        "progressbar": progressbar,
        "progressbar_theme": progressbar_theme,
        "model": model,
        "cores": cores,
        "callback": callback,
        "discard_tuned_samples": discard_tuned_samples,
    }
    parallel_args = {
        "mp_ctx": mp_ctx,
        "blas_cores": num_blas_cores_per_chain,
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
        sample_args["rngs"] = rngs
    else:
        # We pass None if the original random seed was None. The single core sampler
        # methods will only set a global seed when it is not None.
        sample_args["rngs"] = rngs

    t_start = time.time()
    if parallel:
        _log.info(f"Multiprocess sampling ({chains} chains in {cores} jobs)")
        _print_step_hierarchy(step)
        try:
            _mp_sample(**sample_args, **parallel_args)
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
            _log.info(f"Population sampling ({chains} chains)")
            _print_step_hierarchy(step)
            with joined_blas_limiter():
                _sample_population(
                    initial_points=initial_points, parallelize=cores > 1, **sample_args
                )
        else:
            _log.info(f"Sequential sampling ({chains} chains in 1 job)")
            _print_step_hierarchy(step)
            with joined_blas_limiter():
                _sample_many(**sample_args)

    t_sampling = time.time() - t_start

    # Packaging, validating and returning the result was extracted
    # into a function to make it easier to test and refactor.
    return _sample_return(
        run=run,
        traces=traces,
        tune=tune,
        t_sampling=t_sampling,
        discard_tuned_samples=discard_tuned_samples,
        compute_convergence_checks=compute_convergence_checks,
        return_inferencedata=return_inferencedata,
        keep_warning_stat=keep_warning_stat,
        idata_kwargs=idata_kwargs or {},
        model=model,
    )


def _sample_return(
    *,
    run: RunType | None,
    traces: Sequence[IBaseTrace],
    tune: int,
    t_sampling: float,
    discard_tuned_samples: bool,
    compute_convergence_checks: bool,
    return_inferencedata: bool,
    keep_warning_stat: bool,
    idata_kwargs: dict[str, Any],
    model: Model,
) -> InferenceData | MultiTrace:
    """Pick/slice chains, run diagnostics and convert to the desired return type.

    Final step of `pm.sampler`.
    """
    # Pick and slice chains to keep the maximum number of samples
    if discard_tuned_samples:
        traces, length = _choose_chains(traces, tune)
    else:
        traces, length = _choose_chains(traces, 0)
    mtrace = MultiTrace(traces)[:length]

    # count the number of tune/draw iterations that happened
    # ideally via the "tune" statistic, but not all samplers record it!
    if "tune" in mtrace.stat_names:
        # Get the tune stat directly from chain 0, sampler 0
        stat = mtrace._straces[0].get_sampler_stats("tune", sampler_idx=0)
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

    idata = None
    if compute_convergence_checks or return_inferencedata:
        ikwargs: dict[str, Any] = {"model": model, "save_warmup": not discard_tuned_samples}
        ikwargs.update(idata_kwargs)
        idata = pm.to_inference_data(mtrace, **ikwargs)

        if compute_convergence_checks:
            warns = run_convergence_checks(idata, model)
            mtrace.report._add_warnings(warns)
            log_warnings(warns)

        if return_inferencedata:
            # By default we drop the "warning" stat which contains `SamplerWarning`
            # objects that can not be stored with `.to_netcdf()`.
            if not keep_warning_stat:
                return drop_warning_stat(idata)
            return idata
    return mtrace


def _check_start_shape(model, start: PointType):
    """Check that the prior evaluations and initial points have identical shapes.

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
    *,
    draws: int,
    chains: int,
    traces: Sequence[IBaseTrace],
    start: Sequence[PointType],
    rngs: Sequence[np.random.Generator],
    step: Step,
    callback: SamplingIteratorCallback | None = None,
    **kwargs,
):
    """Sample all chains sequentially.

    Parameters
    ----------
    draws: int
        The number of samples to draw
    chains: int
        Total number of chains to sample.
    start: list
        Starting points for each chain
    rngs: list of random Generators
        A list of :py:class:`~numpy.random.Generator` objects, one for each chain
    step: function
        Step function
    """
    for i in range(chains):
        _sample(
            draws=draws,
            chain=i,
            start=start[i],
            step=step,
            trace=traces[i],
            rng=rngs[i],
            callback=callback,
            **kwargs,
        )
    return


def _sample(
    *,
    chain: int,
    progressbar: bool,
    rng: np.random.Generator,
    start: PointType,
    draws: int,
    step: Step,
    trace: IBaseTrace,
    tune: int,
    model: Model | None = None,
    progressbar_theme: Theme | None = default_progress_theme,
    callback=None,
    **kwargs,
) -> None:
    """Sample one chain (singleprocess).

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
    trace
        A chain backend to record draws and stats.
    tune : int
        Number of iterations to tune.
    model : Model (optional if in ``with`` context)
    progressbar_theme : Theme
        Optional custom theme for the progress bar.
    """
    skip_first = kwargs.get("skip_first", 0)

    sampling_gen = _iter_sample(
        draws=draws,
        step=step,
        start=start,
        trace=trace,
        chain=chain,
        tune=tune,
        model=model,
        rng=rng,
        callback=callback,
    )
    _pbar_data = {"chain": chain, "divergences": 0}
    _desc = "Sampling chain {chain:d}, {divergences:,d} divergences"

    progress = CustomProgress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        TextColumn("/"),
        TimeElapsedColumn(),
        console=Console(theme=progressbar_theme),
        disable=not progressbar,
    )

    with progress:
        try:
            task = progress.add_task(_desc.format(**_pbar_data), completed=0, total=draws)
            for it, diverging in enumerate(sampling_gen):
                if it >= skip_first and diverging:
                    _pbar_data["divergences"] += 1
                progress.update(task, description=_desc.format(**_pbar_data), completed=it)
            progress.update(
                task, description=_desc.format(**_pbar_data), completed=draws, refresh=True
            )
        except KeyboardInterrupt:
            pass


def _iter_sample(
    *,
    draws: int,
    step: Step,
    start: PointType,
    trace: IBaseTrace,
    chain: int = 0,
    tune: int = 0,
    rng: np.random.Generator,
    model: Model | None = None,
    callback: SamplingIteratorCallback | None = None,
) -> Iterator[bool]:
    """Sample one chain with a generator (singleprocess).

    Parameters
    ----------
    draws : int
        The number of samples to draw
    step : function
        Step function
    start : dict
        Starting point in parameter space (or partial point).
        Must contain numeric (transformed) initial values for all (transformed) free variables.
    trace
        A chain backend to record draws and stats.
    chain : int, optional
        Chain number used to store sample in backend.
    tune : int, optional
        Number of iterations to tune (defaults to 0).
    model : Model (optional if in ``with`` context)
    random_seed : single random seed, optional

    Yields
    ------
    diverging : bool
        Indicates if the draw is divergent. Only available with some samplers.
    """
    model = modelcontext(model)
    draws = int(draws)

    if draws < 1:
        raise ValueError("Argument `draws` must be greater than 0.")

    step.set_rng(rng)

    point = start

    try:
        step.tune = bool(tune)
        if hasattr(step, "reset_tuning"):
            step.reset_tuning()
        for i in range(draws):
            diverging = False

            if i == 0 and hasattr(step, "iter_count"):
                step.iter_count = 0
            if i == tune:
                step.stop_tuning()
            point, stats = step.step(point)
            trace.record(point, stats)
            log_warning_stats(stats)
            diverging = i > tune and len(stats) > 0 and (stats[0].get("diverging") is True)
            if callback is not None:
                callback(
                    trace=trace,
                    draw=Draw(chain, i == draws, i, i < tune, stats, point),
                )

            yield diverging
    except KeyboardInterrupt:
        trace.close()
        raise
    except BaseException:
        trace.close()
        raise
    else:
        trace.close()


def _mp_sample(
    *,
    draws: int,
    tune: int,
    step,
    chains: int,
    cores: int,
    rngs: Sequence[np.random.Generator],
    start: Sequence[PointType],
    progressbar: bool = True,
    progressbar_theme: Theme | None = default_progress_theme,
    traces: Sequence[IBaseTrace],
    model: Model | None = None,
    callback: SamplingIteratorCallback | None = None,
    blas_cores: int | None = None,
    mp_ctx=None,
    **kwargs,
) -> None:
    """Sample all chains (multiprocess).

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
    rngs: list of random Generators
        A list of :py:class:`~numpy.random.Generator` objects, one for each chain
    start : list
        Starting points for each chain.
        Dicts must contain numeric (transformed) initial values for all (transformed) free variables.
    progressbar : bool
        Whether or not to display a progress bar in the command line.
    progressbar_theme : Theme
        Optional custom theme for the progress bar.
    traces
        Recording backends for each chain.
    model : Model (optional if in ``with`` context)
    callback
        A function which gets called for every sample from the trace of a chain. The function is
        called with the trace and the current draw and will contain all samples for a single trace.
        the ``draw.chain`` argument can be used to determine which of the active chains the sample
        is drawn from.
        Sampling can be interrupted by throwing a ``KeyboardInterrupt`` in the callback.
    """
    import pymc.sampling.parallel as ps

    # We did draws += tune in pm.sample
    draws -= tune

    sampler = ps.ParallelSampler(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        rngs=rngs,
        start_points=start,
        step_method=step,
        progressbar=progressbar,
        progressbar_theme=progressbar_theme,
        blas_cores=blas_cores,
        mp_ctx=mp_ctx,
    )
    try:
        try:
            with sampler:
                for draw in sampler:
                    strace = traces[draw.chain]
                    strace.record(draw.point, draw.stats)
                    log_warning_stats(draw.stats)
                    if draw.is_last:
                        strace.close()

                    if callback is not None:
                        callback(trace=strace, draw=draw)

        except ps.ParallelSamplingError as error:
            strace = traces[error._chain]
            for strace in traces:
                strace.close()
            raise
    except KeyboardInterrupt:
        pass
    finally:
        for strace in traces:
            strace.close()


def _init_jitter(
    model: Model,
    initvals: StartDict | Sequence[StartDict | None] | None,
    seeds: Sequence[int] | np.ndarray,
    jitter: bool,
    jitter_max_retries: int,
) -> list[PointType]:
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
    model: Model | None = None,
    random_seed: RandomSeed = None,
    progressbar=True,
    jitter_max_retries: int = 10,
    tune: int | None = None,
    initvals: StartDict | Sequence[StartDict | None] | None = None,
    **kwargs,
) -> tuple[Sequence[PointType], NUTS]:
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
        potential = quadpotential.QuadPotentialDiagAdapt(n, mean, var, 10, rng=random_seed_list[0])
    elif init == "jitter+adapt_diag":
        mean = np.mean(apoints_data, axis=0)
        var = np.ones_like(mean)
        n = len(var)
        potential = quadpotential.QuadPotentialDiagAdapt(n, mean, var, 10, rng=random_seed_list[0])
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
            rng=random_seed_list[0],
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
        potential = quadpotential.QuadPotentialDiagAdapt(
            n, mean, cov, weight, rng=random_seed_list[0]
        )
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
        potential = quadpotential.QuadPotentialDiag(cov, rng=random_seed_list[0])
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
        potential = quadpotential.QuadPotentialDiag(cov, rng=random_seed_list[0])
    elif init == "map":
        start = pm.find_MAP(include_transformed=True, seed=random_seed_list[0])
        cov = -pm.find_hessian(point=start, negate_output=False)
        initial_points = [start] * chains
        potential = quadpotential.QuadPotentialFull(cov, rng=random_seed_list[0])
    elif init == "adapt_full":
        mean = np.mean(apoints_data * chains, axis=0)
        initial_point = initial_points[0]
        initial_point_model_size = sum(initial_point[n.name].size for n in model.value_vars)
        cov = np.eye(initial_point_model_size)
        potential = quadpotential.QuadPotentialFullAdapt(
            initial_point_model_size, mean, cov, 10, rng=random_seed_list[0]
        )
    elif init == "jitter+adapt_full":
        mean = np.mean(apoints_data, axis=0)
        initial_point = initial_points[0]
        initial_point_model_size = sum(initial_point[n.name].size for n in model.value_vars)
        cov = np.eye(initial_point_model_size)
        potential = quadpotential.QuadPotentialFullAdapt(
            initial_point_model_size, mean, cov, 10, rng=random_seed_list[0]
        )
    else:
        raise ValueError(f"Unknown initializer: {init}.")

    step = pm.NUTS(potential=potential, model=model, rng=random_seed_list[0], **kwargs)

    # Filter deterministics from initial_points
    value_var_names = [var.name for var in model.value_vars]
    initial_points = [
        {k: v for k, v in initial_point.items() if k in value_var_names}
        for initial_point in initial_points
    ]

    return initial_points, step
