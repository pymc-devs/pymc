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

import logging
import warnings

from collections.abc import Callable, Iterable, Sequence
from typing import (
    Any,
    Literal,
    TypeAlias,
    cast,
    overload,
)

import numpy as np
import xarray
import xarray as xr

from pytensor import tensor as pt
from pytensor.graph import vectorize_graph
from pytensor.graph.basic import (
    Constant,
    Variable,
)
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.traversal import ancestors, general_toposort, walk
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.sharedvar import SharedVariable, TensorSharedVariable
from rich.theme import Theme

import pymc as pm

from pymc.backends.arviz import _DefaultTrace, dataset_to_point_list
from pymc.backends.base import MultiTrace
from pymc.blocking import PointType
from pymc.distributions.shape_utils import change_dist_size
from pymc.exceptions import ImplicitFreezeWarning
from pymc.model import Model, modelcontext
from pymc.progress_bar import create_simple_progress, default_progress_theme
from pymc.pytensorf import compile, rvs_in_graph
from pymc.util import (
    RandomState,
    _get_seeds_per_chain,
    get_default_varnames,
    point_wrapper,
)

__all__ = (
    "compile_forward_sampling_function",
    "draw",
    "sample_posterior_predictive",
    "sample_prior_predictive",
    "vectorize_over_posterior",
)

ArrayLike: TypeAlias = np.ndarray | list[float]
PointList: TypeAlias = list[PointType]

_log = logging.getLogger(__name__)


def _build_constant_data(
    trace_constant_data: dict[str, np.ndarray],
    trace_coords: dict[str, np.ndarray],
    model: Model,
) -> dict[Variable, Any]:
    """Collect trace-time values for data-like nodes in the model.

    Returns a mapping from the model's registered data Variables (``pm.Data``
    SharedVariables plus named Constants introduced by ``pm.do`` /
    ``freeze_dims_and_data``) and its shared dim-length vars to the value each
    held at inference time, as recovered from the trace.

    Dim-length shareds are only included when the trace's coord labels still
    match the model's (or when neither has explicit labels), so a same-length
    coord with different labels still counts as volatile.
    """
    constant_data: dict[Variable, Any] = {}
    for dv in model.data_vars:
        if dv.name in trace_constant_data:
            constant_data[dv] = trace_constant_data[dv.name]
    for dim, length_var in model.dim_lengths.items():
        if not isinstance(length_var, SharedVariable) or dim not in trace_coords:
            continue
        trace_coord = trace_coords[dim]
        current_coord = model.coords.get(dim)
        if current_coord is not None:
            unchanged = len(current_coord) == len(trace_coord) and bool(
                np.all(np.asarray(current_coord) == trace_coord)
            )
        else:
            # Coord declared with a length only; match against the current length.
            unchanged = int(length_var.get_value()) == len(trace_coord)
        if unchanged:
            constant_data[length_var] = len(trace_coord)
    return constant_data


def get_vars_in_point_list(trace, model):
    """Get the list of Variable instances in the model that have values stored in the trace."""
    if not isinstance(trace, MultiTrace):
        names_in_trace = list(trace[0])
    else:
        names_in_trace = trace.varnames
    traceable_varnames = {var.name for var in (model.free_RVs + model.deterministics)}
    vars_in_trace = [model[v] for v in names_in_trace if v in traceable_varnames]
    return vars_in_trace


def _data_var_is_volatile(
    var: Variable,
    constant_data: dict[Variable, Any],
) -> bool:
    """Return True if a data-like variable no longer matches its trace-time value.

    Membership in ``constant_data`` is by Variable identity: registered data
    variables (``model.data_vars`` — ``pm.Data`` SharedVariables plus named
    Constants from ``pm.do`` / ``freeze_dims_and_data``) and shared dim-length
    vars get their trace-time value stored there. For a ``SharedVariable`` or
    ``Constant`` in the mapping, current value is compared against the stored
    one. Non-root data variables (e.g. expression Variables introduced by
    ``pm.do``) have no readable current value; their volatility is determined
    by propagation from their inputs in :func:`_compute_volatile_vars`, not
    here.
    """
    if isinstance(var.type, RandomType):
        # RNG variables are not themselves a source of volatility; their consumers are.
        return False
    if isinstance(var, SharedVariable):
        if var in constant_data:
            stored = var.type.filter(constant_data[var])
            return not var.type.values_eq_approx(stored, var.get_value(borrow=True))
        return True  # unknown shared variable — conservatively volatile
    if isinstance(var, Constant):
        if var in constant_data:
            stored = var.type.filter(constant_data[var])
            return not var.type.values_eq_approx(stored, var.data)
        return False  # literal constant not in registry — non-volatile
    # Non-root expression Variable: propagation handles it.
    return False


def _compute_volatile_vars(
    fg: FunctionGraph,
    *,
    vars_in_trace: set[Variable],
    rvs: set[Variable],
    constant_data: dict[Variable, Any],
    volatile_vars: set[Variable],
    freeze_vars: set[Variable],
) -> set[Variable]:
    """Classify every variable reachable from ``fg.outputs`` as volatile or not.

    A variable is volatile when any of these holds (unless it is in
    ``freeze_vars``, which blocks propagation):

    - It is in ``volatile_vars`` (explicit seed).
    - It is a data-like variable whose current value no longer matches the
      trace-time value in ``constant_data`` (see :func:`_data_var_is_volatile`).
    - It is one of ``rvs`` but not present in the trace.
    - Any of its inputs is already flagged volatile.
    """
    volatile_closure: set[Variable] = set()
    variables = general_toposort(fg.outputs, deps=lambda x: x.owner.inputs if x.owner else [])
    for var in variables:
        if var in freeze_vars:
            continue
        if (
            var in volatile_vars
            or (var in rvs and var not in vars_in_trace)
            or (var.owner is None and _data_var_is_volatile(var, constant_data))
            or (var.owner is not None and any(inp in volatile_closure for inp in var.owner.inputs))
        ):
            volatile_closure.add(var)
    return volatile_closure


def _maybe_warn_implicit_freeze(
    *,
    auto_frozen: list[Variable],
    volatile_closure: set[Variable],
    volatile_vars: set[Variable],
    rv_set: set[Variable],
    vars_in_trace_set: set[Variable],
    constant_data: dict[Variable, Any],
    model: Model,
    trace_coords: dict[str, np.ndarray],
) -> None:
    """Warn when an auto-frozen trace RV has a volatile ancestor the user may have wanted resampled."""
    # Partition direct volatile sources once, so the per-RV loop is just intersections.
    # "data": shared Data/coord values that changed since inference time.
    # "non-data": RVs in sample_vars plus RVs missing from the trace (both get a fresh draw).
    data_sources = {
        var
        for var in volatile_closure
        if var.owner is None and _data_var_is_volatile(var, constant_data)
    }
    non_data_sources = {
        var
        for var in volatile_closure
        if var in volatile_vars or (var in rv_set and var not in vars_in_trace_set)
    }

    warn_reasons: dict[str, list[str]] = {}
    for rv in auto_frozen:
        ancs = set(ancestors([rv])) - {rv}
        reasons: list[str] = []
        changed = sorted({anc.name or "<unnamed>" for anc in ancs & data_sources})
        if changed:
            reasons.append(f"upstream Data/coords changed ({', '.join(changed)})")
        resampled = sorted({anc.name for anc in ancs & non_data_sources if anc.name is not None})
        if resampled:
            reasons.append(f"ancestor is resampled ({', '.join(resampled)})")
        if reasons:
            assert rv.name is not None  # trace RVs always have names
            warn_reasons[rv.name] = reasons

    if not warn_reasons:
        return

    # Flag coord-length changes specifically, since they are very likely to cause
    # shape errors with the (unchanged-shape) frozen trace values.
    length_changes = []
    for dim_name, dim_length_var in model.dim_lengths.items():
        if dim_name not in trace_coords or not isinstance(dim_length_var, TensorSharedVariable):
            continue
        new_length = int(dim_length_var.get_value())
        old_length = len(trace_coords[dim_name])
        if new_length != old_length:
            length_changes.append((dim_name, old_length, new_length))

    details = "; ".join(
        f"{name!r} ({'; '.join(reasons)})" for name, reasons in sorted(warn_reasons.items())
    )
    msg = (
        f"The following trace variables were implicitly frozen, but something in their "
        f"upstream suggests you may have wanted them resampled: {details}. To resample "
        f"them, pass them in `sample_vars`; to silence this warning while keeping the "
        f"trace values, pass them in `freeze_vars`."
    )
    if length_changes:
        changes_str = ", ".join(f"{name!r} ({old} -> {new})" for name, old, new in length_changes)
        msg += (
            f" Coordinates changed length ({changes_str}); this function is likely "
            f"to fail with shape errors unless the affected variables are added to "
            f"`sample_vars`."
        )
    warnings.warn(msg, ImplicitFreezeWarning, stacklevel=3)


def compile_forward_sampling_function(
    outputs: list[Variable],
    vars_in_trace: list[Variable],
    basic_rvs: list[Variable] | None = None,
    constant_data: dict[Variable, Any] | None = None,
    volatile_vars: set[Variable] | None = None,
    freeze_vars: set[Variable] | None = None,
    **kwargs,
) -> tuple[Callable[..., np.ndarray | list[np.ndarray]], set[Variable]]:
    """Compile a function to draw samples, conditioned on the values of some variables.

    The goal of this function is to walk the pytensor computational graph from the list
    of output nodes down to the root nodes, and then compile a function that will produce
    values for these output nodes. The compiled function will take as inputs the subset of
    variables in the ``vars_in_trace`` that are deemed to not be **volatile**.

    Volatile variables are variables whose values could change between runs of the
    compiled function or after inference has been run. These variables are:

    - Variables in ``volatile_vars``
    - ``SharedVariable`` or ``Constant`` data nodes whose current value no longer matches the trace-time value stored in ``constant_data`` (see :func:`_data_var_is_volatile`)
    - Variables that are in the ``basic_rvs`` list but not in the ``vars_in_trace`` list
    - Variables that have volatile inputs

    Variables in ``freeze_vars`` are never considered volatile, regardless of the above
    rules. They act as volatility barriers, stopping the propagation of volatility to
    their dependents. Frozen variables are always treated as trace inputs.

    Concretely, this function can be used to compile a function to sample from the
    posterior predictive distribution of a model that has variables that are conditioned
    on ``Data`` instances. The variables that depend on the mutable data that have changed
    will be considered volatile, and as such, they wont be included as inputs into the compiled
    function. This means that if they have values stored in the posterior, these values will be
    ignored and new values will be computed (in the case of deterministics and potentials) or
    sampled (in the case of random variables).

    Parameters
    ----------
    outputs : List[pytensor.graph.basic.Variable]
        The list of variables that will be returned by the compiled function.
        Outputs are not inherently volatile.
    vars_in_trace : List[pytensor.graph.basic.Variable]
        The list of variables that are assumed to have values stored in the trace
    basic_rvs : Optional[List[pytensor.graph.basic.Variable]]
        A list of random variables that are defined in the model. This list (which could be the
        output of ``model.basic_RVs``) should have a reference to the variables that should
        be considered as random variable instances. This includes variables that have
        a ``RandomVariable`` owner op, but also unpure random variables like Mixtures, or
        Censored distributions.
    constant_data : Optional[Dict[Variable, Any]]
        A dictionary that maps data-like ``Variable`` instances (``pm.Data``
        SharedVariables, named ``Constant`` nodes introduced by ``pm.do`` /
        ``freeze_dims_and_data``, or shared dim-length vars) to the value each
        held at inference time. At check time the current value of each listed
        node is compared against the stored one via ``values_eq_approx``; if
        they differ the node is treated as volatile. A ``SharedVariable``
        absent from ``constant_data`` is conservatively treated as volatile;
        an absent ``Constant`` is treated as non-volatile. Setting
        ``constant_data`` to ``None`` is equivalent to passing an empty
        dictionary.
    volatile_vars : Optional[Set[pytensor.graph.basic.Variable]]
        Variables that are unconditionally volatile. Volatility propagates from these
        to their dependents in the graph.
    freeze_vars : Optional[Set[pytensor.graph.basic.Variable]]
        A set of variables that should never be considered volatile, even if they would
        otherwise be due to having volatile inputs or depending on changed data. Frozen
        variables act as volatility barriers: they stop the propagation of volatility to
        their dependents and are always treated as inputs that pull values from the trace.

    Returns
    -------
    function: Callable
        Compiled forward sampling PyTensor function
    volatile_basic_rvs: Set of Variable
        Set of all basic_rvs that were considered volatile and will be resampled when
        the function is evaluated
    """
    if basic_rvs is None:
        basic_rvs = []

    if constant_data is None:
        constant_data = {}
    if volatile_vars is None:
        volatile_vars = set()
    if freeze_vars is None:
        freeze_vars = set()

    fg = FunctionGraph(outputs=outputs, clone=False)
    volatile_closure = _compute_volatile_vars(
        fg,
        vars_in_trace=set(vars_in_trace),
        rvs=set(basic_rvs),
        constant_data=constant_data,
        volatile_vars=volatile_vars,
        freeze_vars=freeze_vars,
    )

    # Collect the function inputs by walking the graph from the outputs. Inputs will be:
    # 1. Random variables that are not volatile
    # 2. Variables that have no owner and are not constant or shared
    inputs = []

    def expand(var):
        if (
            (
                var.owner is None and not isinstance(var, Constant | SharedVariable)
            )  # Variables without owners that are not constant or shared
            or var in vars_in_trace  # Variables in the trace
        ) and var not in volatile_closure:
            # This test will include variables without owners, and that are not constant
            # or shared, because these variables will never be considered volatile
            inputs.append(var)
        if var.owner:
            return var.owner.inputs

    # walk produces a generator, so we have to actually exhaust the generator in a list to walk
    # the entire graph
    list(walk(fg.outputs, expand))

    return (
        compile(inputs, fg.outputs, on_unused_input="ignore", **kwargs),
        set(basic_rvs) & volatile_closure,  # Basic RVs that will be resampled
    )


def draw(
    vars: Variable | Sequence[Variable],
    draws: int = 1,
    random_seed: RandomState = None,
    **kwargs,
) -> np.ndarray | list[np.ndarray]:
    """Draw samples for one variable or a list of variables.

    Parameters
    ----------
    vars : TensorVariable or iterable of TensorVariable
        A variable or a list of variables for which to draw samples.
    draws : int, default 1
        Number of samples needed to draw.
    random_seed : int, RandomState or numpy_Generator, optional
        Seed for the random number generator.
    **kwargs : dict, optional
        Keyword arguments for :func:`pymc.pytensorf.compile`.

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

    draw_fn = compile(inputs=[], outputs=vars, random_seed=random_seed, **kwargs)

    if draws == 1:
        return draw_fn()

    # Single variable output
    if not isinstance(vars, list | tuple):
        cast(Callable[[], np.ndarray], draw_fn)
        return np.stack([draw_fn() for _ in range(draws)])

    # Multiple variable output
    cast(Callable[[], list[np.ndarray]], draw_fn)
    drawn_values = zip(*(draw_fn() for _ in range(draws)))
    return [np.stack(v) for v in drawn_values]


def observed_dependent_deterministics(model: Model, extra_observeds=None):
    """Find deterministics that depend directly on observed variables."""
    if extra_observeds is None:
        extra_observeds = []

    deterministics = model.deterministics
    observed_rvs = set(model.observed_RVs + extra_observeds)
    blockers = model.basic_RVs
    return [
        deterministic
        for deterministic in deterministics
        if observed_rvs & set(ancestors([deterministic], blockers=blockers))
    ]


@overload
def sample_prior_predictive(
    draws: int = 500,
    model: Model | None = None,
    var_names: Iterable[str] | None = None,
    random_seed: RandomState = None,
    return_inferencedata: Literal[True] = True,
    idata_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
) -> xarray.DataTree: ...
@overload
def sample_prior_predictive(
    draws: int = 500,
    model: Model | None = None,
    var_names: Iterable[str] | None = None,
    random_seed: RandomState = None,
    return_inferencedata: Literal[False] = False,
    idata_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
) -> dict[str, np.ndarray]: ...
def sample_prior_predictive(
    draws: int = 500,
    model: Model | None = None,
    var_names: Iterable[str] | None = None,
    random_seed: RandomState = None,
    return_inferencedata: bool = True,
    idata_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
) -> xarray.DataTree | dict[str, np.ndarray]:
    """Generate samples from the prior predictive distribution.

    Parameters
    ----------
    draws : int
        Number of samples from the prior predictive to generate. Defaults to 500.
    model : Model (optional if in ``with`` context)
    var_names : Iterable[str]
        A list of names of variables for which to compute the prior predictive
        samples. Defaults to both observed and unobserved RVs. Transformed values
        are not allowed.
    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.
    return_inferencedata : bool
        Whether to return an :class:`xarray:xarray.DataTree` (True) object or a dictionary (False).
        Defaults to True.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data`
    compile_kwargs: dict, optional
        Keyword arguments for :func:`pymc.pytensorf.compile`.

    Returns
    -------
    xarray.DataTree or dict
        A ``DataTree`` object containing the prior and prior predictive samples (default),
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
        vars_: set[str] = {var.name for var in model.basic_RVs + model.deterministics}
    else:
        vars_ = set(var_names)

    names = sorted(get_default_varnames(vars_, include_transformed=False))
    vars_to_evaluate = [model[name] for name in names]

    # Any variables from var_names still missing are assumed to be transformed variables.
    missing_names = vars_.difference(names)
    if missing_names:
        raise ValueError(f"Unrecognized var_names: {missing_names}")

    if random_seed is not None:
        (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    if compile_kwargs is None:
        compile_kwargs = {}
    compile_kwargs.setdefault("allow_input_downcast", True)
    compile_kwargs.setdefault("accept_inplace", True)

    sampler_fn, volatile_basic_rvs = compile_forward_sampling_function(
        vars_to_evaluate,
        vars_in_trace=[],
        basic_rvs=model.basic_RVs,
        random_seed=random_seed,
        **compile_kwargs,
    )

    # All model variables have a name, but mypy does not know this
    _log.info(f"Sampling: {sorted(volatile_basic_rvs, key=lambda var: var.name)}")  # type: ignore[arg-type, return-value]
    values = zip(*(sampler_fn() for i in range(draws)))

    data = {k: np.stack(v) for k, v in zip(names, values)}
    if data is None:
        raise AssertionError(f"No variables sampled: attempting to sample {names}")

    prior: dict[str, np.ndarray] = {}
    for var_name in vars_:
        if var_name in data:
            prior[var_name] = data[var_name]

    if not return_inferencedata:
        return prior
    ikwargs: dict[str, Any] = {"model": model}
    if idata_kwargs:
        ikwargs.update(idata_kwargs)
    return pm.to_inference_data(prior=prior, **ikwargs)


@overload
def sample_posterior_predictive(
    trace,
    model: Model | None = None,
    *,
    var_names: str | list[str] | None = None,
    sample_vars: str | list[str] | None = None,
    freeze_vars: str | list[str] | None = None,
    sample_dims: list[str] | None = None,
    random_seed: RandomState = None,
    progressbar: bool = True,
    progressbar_theme: Theme | None = default_progress_theme,
    return_inferencedata: Literal[True] = True,
    extend_inferencedata: bool = False,
    predictions: bool = False,
    idata_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
) -> xarray.DataTree: ...
@overload
def sample_posterior_predictive(
    trace,
    model: Model | None = None,
    *,
    var_names: str | list[str] | None = None,
    sample_vars: str | list[str] | None = None,
    freeze_vars: str | list[str] | None = None,
    sample_dims: list[str] | None = None,
    random_seed: RandomState = None,
    progressbar: bool = True,
    progressbar_theme: Theme | None = default_progress_theme,
    return_inferencedata: Literal[False] = False,
    extend_inferencedata: bool = False,
    predictions: bool = False,
    idata_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
) -> dict[str, np.ndarray]: ...
def sample_posterior_predictive(
    trace,
    model: Model | None = None,
    *,
    var_names: str | list[str] | None = None,
    sample_vars: str | list[str] | None = None,
    freeze_vars: str | list[str] | None = None,
    sample_dims: list[str] | None = None,
    random_seed: RandomState = None,
    progressbar: bool = True,
    progressbar_theme: Theme | None = default_progress_theme,
    return_inferencedata: bool = True,
    extend_inferencedata: bool = False,
    predictions: bool = False,
    idata_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
) -> xarray.DataTree | dict[str, np.ndarray]:
    """Generate forward samples for `var_names`, conditioned on the posterior samples of variables found in the `trace`.

    This method can be used to perform different kinds of model predictions, including posterior predictive checks.

    The matching of unobserved model variables, and posterior samples in the `trace` is made based on the variable
    names. Therefore, a different model than the one used for posterior sampling may be used for posterior predictive
    sampling, as long as the variables whose posterior we want to condition on have the same name, and compatible shape
    and coordinates.


    Parameters
    ----------
    trace : backend, list, xarray.Dataset, xarray.DataTree, or MultiTrace
        Trace generated from MCMC sampling, or a list of dicts (eg. points or from :func:`~pymc.find_MAP`),
        or :class:`xarray.Dataset` (eg. DataTree.posterior or DataTree.prior)
    model : Model (optional if in ``with`` context)
        Model to be used to generate the posterior predictive samples. It will
        generally be the model used to generate the `trace`, but it doesn't need to be.
    sample_vars : str or list of str, optional
        Random variables or deterministics to regenerate on each draw rather
        than copy from the trace. Regeneration propagates volatility downstream:
        an RV that is in the trace and not listed here keeps its trace value,
        but if one of its ancestors is volatile (listed here, or a changed
        Data/coord) an :class:`~pymc.exceptions.ImplicitFreezeWarning` flags
        it so the user can opt in by adding it here, or silence the warning via
        ``freeze_vars``. Empty by default — RVs missing from the trace (including
        observed RVs) are always regenerated automatically. Cannot overlap with
        ``freeze_vars``.
    freeze_vars : str or list of str, optional
        Trace variables (RVs or deterministics) to reuse from the trace. Cannot
        overlap with ``sample_vars``. Trace RVs not in ``sample_vars`` are already
        implicitly frozen, so the practical effect of listing an RV here is to
        silence its :class:`~pymc.exceptions.ImplicitFreezeWarning`. Deterministics
        don't trigger that warning at all — a volatile deterministic just
        recomputes with the current upstream values — so listing one only matters
        when you want to keep the trace value instead (see example below).
    var_names : str or list of str, optional
        Controls only which variables appear in the output; does not trigger
        resampling. Each listed name is either computed fresh or copied from the
        input trace, depending on whether it or any of its upstream is volatile
        (see the behavior section below). Defaults to ``sample_vars`` when that is
        specified; otherwise (the classic posterior-predictive default) to the
        observed variables plus any deterministic that depends on these.
    sample_dims : list of str, optional
        Dimensions over which to loop and generate posterior predictive samples.
        When ``sample_dims`` is ``None`` (default) both "chain" and "draw" are considered sample
        dimensions. Only taken into account when `trace` is xarray.DataTree or Dataset.
    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.
    progressbar : bool
        Whether to display a progress bar in the command line. The bar shows the percentage
        of completion, the sampling speed in samples per second (SPS), and the estimated remaining
        time until completion ("expected time of arrival"; ETA).
    return_inferencedata : bool, default True
        Whether to return an :class:`xarray:xarray.DataTree` (True) object or a dictionary (False).
    extend_inferencedata : bool, default False
        Whether to automatically use :meth:`xarray.DataTree.update` to add the posterior predictive samples to
        `trace` or not. If True, `trace` is modified inplace but still returned. If the DataTree
        already contains a group that would be added (e.g. ``posterior_predictive``), a warning
        is issued and the existing group is overwritten.
    predictions : bool, default False
        Flag used to set the location of posterior predictive samples within the returned ``xarray.DataTree`` object.
        If False, assumes samples are generated based on the fitting data to be used for posterior predictive checks,
        and samples are stored in the ``posterior_predictive``. If True, assumes samples are generated based on
        out-of-sample data as predictions, and samples are stored in the ``predictions`` group.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data` if ``predictions=False`` or to
        :func:`pymc.predictions_to_inference_data` otherwise.
    compile_kwargs: dict, optional
        Keyword arguments for :func:`pymc.pytensorf.compile`.

    Returns
    -------
    xarray.DataTree or Dict
        A ``xarray.DataTree`` object containing the posterior predictive samples (default), or
        a dictionary with variable names as keys, and samples as numpy arrays.


    Examples
    --------
    Posterior predictive checks and predictions
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    The most common use of `sample_posterior_predictive` is to perform posterior predictive checks (in-sample predictions)
    and new model predictions (out-of-sample predictions). Deterministics that
    depend on :class:`~pymc.Data` are recomputed automatically when the data
    changes — no extra work needed:

    .. code-block:: python

        import pymc as pm

        with pm.Model(coords={"trial": [0, 1, 2]}) as model:
            x = pm.Data("x", [-1, 0, 1], dims=["trial"])
            beta = pm.Normal("beta")
            noise = pm.HalfNormal("noise")
            linpred = pm.Deterministic("linpred", x * beta, dims=["trial"])
            y = pm.Normal("y", mu=linpred, sigma=noise, observed=[-2, 0, 3], dims=["trial"])

            idata = pm.sample()

            # in-sample posterior predictive
            posterior_predictive = pm.sample_posterior_predictive(idata).posterior_predictive

        with model:
            pm.set_data({"x": [-2, 2]}, coords={"trial": [3, 4]})
            # out-of-sample predictions. `linpred` is recomputed with the new `x`
            # (and the trace's `beta`); `y` is resampled from the new `linpred`.
            pm.sample_posterior_predictive(idata, predictions=True, extend_inferencedata=True)


    Freezing deterministics
    ^^^^^^^^^^^^^^^^^^^^^^^

    A deterministic is normally recomputed whenever its inputs change.
    Occasionally, though, a deterministic captures something
    that should stay anchored to the *training* data — e.g. an HSGP standardization
    computed from ``pm.Data`` that must not be rederived from the prediction data.
    Pass the deterministic in ``freeze_vars`` to keep its trace value:

    .. code-block:: python

        import pymc as pm

        with pm.Model() as model:
            x = pm.Data("x", [1.0, 2.0, 3.0])
            x_mean = pm.Deterministic("x_mean", x.mean())
            centered = pm.Deterministic("centered", x - x_mean)
            mu = pm.Normal("mu")
            obs = pm.Normal("obs", mu + centered, 1, observed=[0, 0, 0])

            idata = pm.sample()

        # New x values. Without freezing, `x_mean` would be recomputed as the new mean.
        with model:
            pm.set_data({"x": [100.0, 200.0, 300.0]})
            pm.sample_posterior_predictive(idata, freeze_vars=["x_mean"])


    Forcing a deterministic to recompute
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    If :func:`~pymc.do` swaps a new expression into a deterministic while
    every RV and Data value stays unchanged, ``sample_posterior_predictive``
    sees nothing volatile and reuses the deterministic from the trace. List
    it in ``sample_vars`` to force recomputation from the current graph:

    .. code-block:: python

        with pm.Model() as model:
            x = pm.Normal("x")
            pm.Deterministic("det", x**2)
            pm.Normal("obs", model["det"], 1, observed=[0.0])
            idata = pm.sample()

        with pm.do(model, {model["det"]: model["x"] ** 3}) as intervened_model:
            # Force recomputation using the new `x**3` graph.
            pm.sample_posterior_predictive(idata, sample_vars=["det", "obs"])


    Using different models
    ^^^^^^^^^^^^^^^^^^^^^^

    It's common to use the same model for posterior and posterior predictive sampling, but this is not required.
    The matching between unobserved model variables and posterior samples is based on the name alone.

    For the last example we could have created a new predictions model.
    Since the new ``y`` has no observations, we request it via ``sample_vars`` argument.

    .. code-block:: python

        import pymc as pm

        with pm.Model(coords={"trial": [0, 1, 2]}) as train_model:
            x = pm.Data("x", [-1, 0, 1], dims=["trial"])
            beta = pm.Normal("beta")
            noise = pm.HalfNormal("noise")
            y = pm.Normal("y", mu=x * beta, sigma=noise, observed=[-2, 0, 3], dims=["trial"])

            idata = pm.sample()

        with pm.Model(coords={"trial": [3, 4]}) as prediction_model:
            x = pm.Data("x", [-2, 2], dims=["trial"])
            beta = pm.Normal("beta")
            noise = pm.HalfNormal("noise")
            y = pm.Normal("y", mu=x * beta, sigma=noise, dims=["trial"])

            predictions = pm.sample_posterior_predictive(
                idata,
                sample_vars=["y"],
                predictions=True,
            )


    The new model may even have a different structure and unobserved variables that don't exist in the trace.
    These variables will be sampled automatically because they have no trace values to fall back on.
    In the following example we added a new ``extra_noise`` variable between the inferred posterior ``noise``
    and the new StudentT observational distribution  ``y``:

    .. code-block:: python

        with pm.Model(coords={"trial": [3, 4]}) as distinct_predictions_model:
            x = pm.Data("x", [-2, 2], dims=["trial"])
            beta = pm.Normal("beta")
            noise = pm.HalfNormal("noise")
            extra_noise = pm.HalfNormal("extra_noise", sigma=noise)
            y = pm.StudentT("y", nu=4, mu=x * beta, sigma=extra_noise, dims=["trial"])

            predictions = pm.sample_posterior_predictive(idata, var_names=["y"], predictions=True)


    For more about out-of-model predictions, see this `blog post <https://www.pymc-labs.com/blog-posts/out-of-model-predictions-with-pymc/>`_.

    The behavior of ``sample_vars``, ``freeze_vars``, and ``var_names``
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Each of these three arguments controls one aspect of the operation:

    - ``sample_vars`` — trace variables to treat as volatile: regenerate them
      (from their distribution or expression) instead of copying from the trace.
      Empty by default.
    - ``freeze_vars`` — which trace variables to reuse explicitly (silences the
      implicit-freeze warning below).
    - ``var_names`` — which variables appear in the output. Does not trigger
      resampling of variables in the trace. Defaults to ``sample_vars``.

    **Volatility.** Volatility originates from three sources — variables listed
    in ``sample_vars``, changed Data/coords, and RVs missing from the trace
    (including observed RVs, which are always regenerated since they have no
    trace value to reuse). It then propagates downstream through deterministics
    and other RVs. An RV that *is* in the trace and not listed in ``sample_vars``
    keeps its trace value — even when one of its ancestors is being resampled.
    This prevents a single ``sample_vars=["x"]`` call, or a ``set_data`` call,
    from silently invalidating the posterior values for every downstream
    variable. When an auto-frozen trace variable has a volatile ancestor, an
    :class:`~pymc.exceptions.ImplicitFreezeWarning` flags it so the user can
    opt in by adding it to ``sample_vars`` (to resample) or opt out by adding
    it to ``freeze_vars`` (to silence the warning while keeping the trace
    value). The log lists all the RVs being resampled in any given call.

    The following examples use this model:

    .. code-block:: python

        from logging import getLogger
        import pymc as pm

        # Some environments like google colab suppress
        # the default logging output of PyMC
        getLogger("pymc").setLevel("INFO")

        kwargs = {"progressbar": False, "random_seed": 0}

        with pm.Model() as model:
            x = pm.Normal("x")
            y = pm.Normal("y")
            z = pm.Normal("z", x + y**2)
            det = pm.Deterministic("det", pm.math.exp(z))
            obs = pm.Normal("obs", det, 1, observed=[20])

            idata = pm.sample(tune=10, draws=10, chains=2, **kwargs)

    Default behavior: Generate samples of ``obs`` conditioned on the posterior
    samples of ``z`` found in the trace. These are often referred to as posterior
    predictive samples in the literature:

    .. code-block:: python

        with model:
            pm.sample_posterior_predictive(idata, **kwargs)
            # Sampling: [obs]

    Copy the trace values for ``z`` and ``det``. Nothing is resampled without explicit `sample_vars`:

    .. code :: python

        with model:
            pm.sample_posterior_predictive(idata, var_names=["z", "det"], **kwargs)
            # Sampling: []

    Generate new samples of z and det, conditioned on the posterior samples of x and y found in the trace.

    .. code :: python

        with model:
            pm.sample_posterior_predictive(idata, var_names=["z", "det"], sample_vars=["z"], **kwargs)
            # Sampling: [z]

    Generate samples of y, z and det, conditioned on the posterior samples of x found in the trace.

    .. warning::

        The samples of ``y`` are equivalent to its prior, since it does not depend on any other variables.

    In contrast, the samples of ``z`` and ``det`` depend on the new samples of ``y`` and the posterior samples of ``x`` found in the trace.

    .. code :: python

        with model:
            pm.sample_posterior_predictive(idata, var_names=["y", "z", "det"], sample_vars=["y", "z"], **kwargs)
            # Sampling: [y, z]

    Note that if ``z`` is not placed in `sample_vars` it *won't* be resampled even though it depends on the freshly drawn
    ``y`` — cascade stops at RVs that are in the trace. A warning flags this behavior for ``z``:

    .. code :: python

        with model:
            pm.sample_posterior_predictive(idata, var_names=["y", "z", "det"], sample_vars=["y"], **kwargs)
            # ImplicitFreezeWarning: 'z' (ancestor is resampled (y))
            # Sampling: [y]

    If this is the intended behavior `z` can be added to freeze_vars explicitly, and the warning is avoided.

    .. code :: python

        with model:
            pm.sample_posterior_predictive(idata, var_names=["y", "z", "det"], sample_vars=["y"], freeze_vars=["z"], **kwargs)
            # Sampling: [y]

    Passing every RV to ``sample_vars`` makes this equivalent to :func:`~pymc.sample_prior_predictive`.
    Including ``obs`` in ``sample_vars`` is redundant — it isn't in the trace so it is always regenerated:

    .. code :: python

        with model:
            pm.sample_posterior_predictive(
                idata,
                var_names=["x", "y", "z", "det", "obs"],
                sample_vars=["x", "y", "z", "obs"],
                **kwargs,
            )
            # Sampling: [obs, x, y, z]

    Controlling the number of samples
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    You can manipulate the DataTree to control the number of samples

    .. code-block:: python

        import pymc as pm

        with pm.Model() as model:
            ...
            idata = pm.sample()

    Generate 1 posterior predictive sample for every 5 posterior samples.

    .. code-block:: python

        thinned_idata = idata.sel(draw=slice(None, None, 5))
        with model:
            idata.update(pm.sample_posterior_predictive(thinned_idata))


    Generate 5 posterior predictive samples for every posterior sample.

    .. code-block:: python

        expanded_idata = idata.copy()
        expanded_idata.posterior = idata.posterior.expand_dims(pred_id=5)
        with model:
            pm.sample_posterior_predictive(
                expanded_idata,
                sample_dims=["chain", "draw", "pred_id"],
                extend_inferencedata=True,
            )


    """
    _trace: MultiTrace | PointList
    nchain: int
    if isinstance(var_names, str):
        var_names = [var_names]
    if isinstance(sample_vars, str):
        sample_vars = [sample_vars]
    if isinstance(freeze_vars, str):
        freeze_vars = [freeze_vars]
    if idata_kwargs is None:
        idata_kwargs = {}
    else:
        idata_kwargs = idata_kwargs.copy()
    if sample_dims is None:
        sample_dims = ["chain", "draw"]
    trace_constant_data: dict[str, np.ndarray] = {}
    trace_coords: dict[str, np.ndarray] = {}
    if "coords" not in idata_kwargs:
        idata_kwargs["coords"] = {}
    idata: xarray.DataTree | None = None
    observed_data = None
    stacked_dims = None
    if isinstance(trace, xarray.DataTree):
        _constant_data = getattr(trace, "constant_data", None)
        if _constant_data is not None:
            trace_coords.update({str(k): v.data for k, v in _constant_data.coords.items()})
            trace_constant_data.update({str(k): v.data for k, v in _constant_data.items()})
        idata = trace
        observed_data = trace.get("observed_data", None)
        if "posterior" in trace.children:
            trace = trace["posterior"].dataset
        else:
            trace = trace.dataset
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
            f"Do not know how to compute number of samples for trace argument of type {type(_trace)}"
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

    constant_data = _build_constant_data(trace_constant_data, trace_coords, model)
    vars_in_trace = get_vars_in_point_list(_trace, model)

    # Resolve output variables (what to return in the dataset).
    # - If var_names is given, use it verbatim.
    # - If sample_vars is given, the output is exactly sample_vars (the user's
    #   explicit set of variables to re-evaluate).
    # - Otherwise (the classic PPC default), the output is observed RVs plus any
    #   deterministic that depends on them — the latter is a hack for partially
    #   observed RVs (represented internally as a set_subtensor deterministic
    #   combining the observed and unobserved parts).
    if var_names is not None:
        output_vars = [model[x] for x in var_names]
    elif sample_vars is not None:
        output_vars = [model[x] for x in sample_vars]
    else:
        extra_observeds: list[Variable] = []
        output_vars = list(model.observed_RVs)
        if observed_data is not None:
            for name in observed_data:
                if name in model and model[name] not in output_vars:
                    output_vars.append(model[name])
                    extra_observeds.append(model[name])
        output_vars_set = set(output_vars)
        output_vars += [
            d
            for d in observed_dependent_deterministics(model, extra_observeds)
            if d not in output_vars_set
        ]

    # Resolve sample_vars to the set of volatile-seed Variable objects. These override
    # trace values on each draw (RVs re-sampled, deterministics re-evaluated). Basic
    # RVs missing from the trace (including observed RVs) are always regenerated via
    # compile_forward_sampling_function's volatility rule, so the default here is empty.
    volatile_vars: set[Variable]
    if sample_vars is None:
        volatile_vars = set()
    else:
        # sample_vars entries must be RVs or deterministics (i.e. things with a
        # trace-level concept). Data/coord containers don't fit.
        allowed = set(model.basic_RVs) | set(model.deterministics)
        if invalid := [name for name in sample_vars if model[name] not in allowed]:
            raise ValueError(
                f"sample_vars {sorted(invalid)} are not random variables or "
                f"deterministics; only those can have trace values to override."
            )
        volatile_vars = {model[name] for name in sample_vars}

    rv_set = set(model.basic_RVs)
    vars_in_trace_set = set(vars_in_trace)
    sample_var_names = set(sample_vars or ())
    trace_rvs = [rv for rv in vars_in_trace if rv in rv_set]

    frozen_vars: set[Variable] = set()
    if freeze_vars is not None:
        frozen_vars = {model[x] for x in freeze_vars}
        vars_in_trace_names = {v.name for v in vars_in_trace}
        missing = {x for x in freeze_vars if x not in vars_in_trace_names}
        if missing:
            raise ValueError(
                f"freeze_vars {sorted(missing)} are not present in the trace. "
                f"Cannot freeze variables without stored values."
            )
        if sample_vars is not None and (overlap := (set(freeze_vars) & set(sample_vars))):
            raise ValueError(
                f"Variables {sorted(overlap)} are in both sample_vars and freeze_vars. "
                f"A variable cannot be both resampled and frozen."
            )

    # Auto-freeze every trace basic RV not already in sample_vars or explicit freeze_vars.
    # A warning below lists cases where this freeze is likely unintended
    # (upstream Data/coords changed, or an ancestor is being resampled).
    auto_frozen_vars = [
        rv for rv in trace_rvs if rv.name not in sample_var_names and rv not in frozen_vars
    ]
    frozen_vars.update(auto_frozen_vars)

    # Run the full volatility analysis once, over the combined graph of
    # output_vars (so we can classify deterministics in var_names)
    # and the auto-frozen trace RVs (so the warning below can check their ancestors).
    volatility_fg = FunctionGraph(outputs=list(output_vars) + auto_frozen_vars, clone=False)
    volatile_closure = _compute_volatile_vars(
        volatility_fg,
        vars_in_trace=vars_in_trace_set,
        rvs=rv_set,
        constant_data=constant_data,
        volatile_vars=volatile_vars,
        freeze_vars=frozen_vars,
    )

    # Dedupe output_vars preserving order, so downstream loops don't emit duplicates.
    output_vars = list(dict.fromkeys(output_vars))

    if not output_vars:
        # Nothing to produce — neither sampled nor copied from the trace.
        if return_inferencedata and not extend_inferencedata:
            return xr.DataTree()
        elif return_inferencedata and extend_inferencedata:
            return trace if idata is None else idata
        return {}

    # Compiled function outputs are strictly the entries of output_vars that need regeneration.
    # An output in the trace and not volatile is copied from the trace below.
    # Upstream volatile_vars and intermediate deterministics are computed by PyTensor as needed
    # — they don't need to be listed as outputs.
    compiled_outputs = [
        var for var in output_vars if not (var in vars_in_trace_set and var not in volatile_closure)
    ]
    vars_to_evaluate = list(get_default_varnames(compiled_outputs, include_transformed=False))

    _maybe_warn_implicit_freeze(
        auto_frozen=auto_frozen_vars,
        volatile_closure=volatile_closure,
        volatile_vars=volatile_vars,
        rv_set=rv_set,
        vars_in_trace_set=vars_in_trace_set,
        constant_data=constant_data,
        model=model,
        trace_coords=trace_coords,
    )

    if random_seed is not None:
        (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    if compile_kwargs is None:
        compile_kwargs = {}
    compile_kwargs.setdefault("allow_input_downcast", True)
    compile_kwargs.setdefault("accept_inplace", True)

    sampler_fn: Callable | None
    if vars_to_evaluate:
        _sampler_fn, volatile_basic_rvs = compile_forward_sampling_function(
            outputs=vars_to_evaluate,
            vars_in_trace=vars_in_trace,
            basic_rvs=model.basic_RVs,
            random_seed=random_seed,
            constant_data=constant_data,
            volatile_vars=volatile_vars,
            freeze_vars=frozen_vars,
            **compile_kwargs,
        )
        sampler_fn = point_wrapper(_sampler_fn)
        # All model variables have a name, but mypy does not know this
        _log.info(f"Sampling: {sorted(volatile_basic_rvs, key=lambda var: var.name)}")  # type: ignore[arg-type, return-value]
    else:
        # Nothing needs to be sampled — output_vars will be fully populated from the
        # trace. Skip compilation entirely.
        sampler_fn = None
        _log.info("Sampling: []")

    # Determine output-only variables that should be copied from trace, not sampled.
    evaluated_names = {v.name for v in vars_to_evaluate}
    copy_from_trace_names = [var.name for var in output_vars if var.name not in evaluated_names]

    ppc_trace_t = _DefaultTrace(samples)

    progress = create_simple_progress(
        progressbar=progressbar,
        progressbar_theme=progressbar_theme,
    )

    try:
        with progress:
            task = progress.add_task("Sampling ...", completed=0, total=samples)
            for idx in range(samples):
                if nchain > 1:
                    # the trace object will either be a MultiTrace (and have _straces)...
                    if hasattr(_trace, "_straces"):
                        chain_idx, point_idx = np.divmod(idx, len_trace)
                        chain_idx = chain_idx % nchain
                        param = _trace._straces[chain_idx].point(point_idx)
                    # ... or a PointList
                    else:
                        param = cast(PointList, _trace)[idx % (len_trace * nchain)]
                # there's only a single chain, but the index might hit it multiple times if
                # the number of indices is greater than the length of the trace.
                else:
                    param = _trace[idx % len_trace]

                if sampler_fn is not None:
                    values = sampler_fn(**param)
                    for k, v in zip(vars_to_evaluate, values):
                        ppc_trace_t.insert(k.name, v, idx)

                # Copy output-only variables from trace
                for name in copy_from_trace_names:
                    ppc_trace_t.insert(name, param[name], idx)

                progress.advance(task)
            progress.update(task, refresh=True, completed=samples)

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
    ikwargs: dict[str, Any] = dict(model=model, **idata_kwargs)
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
        existing_groups = set(idata.children) & set(idata_pp.children)
        conflicting = existing_groups - {"observed_data", "constant_data"}
        if conflicting:
            warnings.warn(
                f"groups {conflicting} already exist in the DataTree and will be overwritten. "
                "To avoid this, set extend_inferencedata=False.",
                UserWarning,
                stacklevel=2,
            )
        idata.update(idata_pp)
        return idata
    return idata_pp


def vectorize_over_posterior(
    outputs: list[Variable],
    posterior: xr.Dataset,
    input_rvs: list[Variable],
    allow_rvs_in_graph: bool = True,
    sample_dims: tuple[str, ...] = ("chain", "draw"),
) -> list[Variable]:
    """Vectorize outputs over posterior samples of subset of input rvs.

    This function creates a new graph for the supplied outputs, where the required
    subset of input rvs are replaced by their posterior samples (chain and draw
    dimensions are flattened). The other input tensors are kept as is.

    Parameters
    ----------
    outputs : list[Variable]
        The list of variables to vectorize over the posterior samples.
    posterior : xr.Dataset
        The posterior samples to use as replacements for the `input_rvs`.
    input_rvs : list[Variable]
        The list of random variables to replace with their posterior samples.
    allow_rvs_in_graph : bool
        Whether to allow random variables to be present in the graph. If False,
        an error will be raised if any random variables are found in the graph. If
        True, the remaining random variables will be resized to match the number of
        draws from the posterior.
    sample_dims : tuple[str, ...]
        The dimensions of the posterior samples to use for vectorizing the `input_rvs`.


    Returns
    -------
    vectorized_outputs : list[Variable]
        The vectorized variables

    Raises
    ------
    RuntimeError
        If random variables are found in the graph and `allow_rvs_in_graph` is False
    """
    # Identify which free RVs are needed to compute `outputs`
    needed_rvs: list[Variable] = [
        rv for rv in ancestors(outputs, blockers=input_rvs) if rv in set(input_rvs)
    ]

    # Replace needed_rvs with actual posterior samples
    batch_shape = tuple([len(posterior.coords[dim]) for dim in sample_dims])
    replace_dict: dict[Variable, Variable] = {}
    for rv in needed_rvs:
        posterior_samples = posterior[rv.name].data

        replace_dict[rv] = pt.constant(posterior_samples.astype(rv.dtype), name=rv.name)  # type: ignore[attr-defined]

    # Replace the rvs that remain in the graph with resized versions
    all_rvs = rvs_in_graph(outputs)

    # Once we give values for the needed_rvs (setting them to their posterior samples),
    # we need to identify the rvs that only depend on these conditioned values, and
    # don't depend on any other rvs or output nodes.
    # These variables need to be resized because they won't be resized implicitly by
    # the replacement of the needed_rvs or other random variables in the graph when we
    # later call vectorize_graph.
    independent_rvs: list[Variable] = []
    for rv in [
        rv
        for rv in general_toposort(  # type: ignore[call-overload]
            all_rvs, lambda x: x.owner.inputs if x.owner is not None else None
        )
        if rv in all_rvs and rv not in needed_rvs
    ]:
        blockers = [*needed_rvs, *independent_rvs, *outputs]
        rv_ancestors = ancestors([rv], blockers=blockers)
        if not (set(blockers) & set(rv_ancestors)):
            independent_rvs.append(rv)
    for rv in independent_rvs:
        replace_dict[rv] = change_dist_size(rv, new_size=batch_shape, expand=True)

    # Vectorize across samples
    vectorized_outputs = list(vectorize_graph(outputs, replace=replace_dict))
    for vectorized_output, output in zip(vectorized_outputs, outputs):
        vectorized_output.name = output.name

    if not allow_rvs_in_graph:
        remaining_rvs = rvs_in_graph(vectorized_outputs)
        if remaining_rvs:
            raise RuntimeError(
                f"The following random variables found in the extracted graph: {remaining_rvs}"
            )
    return vectorized_outputs
