#   Copyright 2023 The PyMC Developers
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

"""Functions for prior and posterior predictive sampling."""

import logging
import warnings

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
import xarray

from arviz import InferenceData
from fastprogress.fastprogress import progress_bar
from pytensor import tensor as pt
from pytensor.graph.basic import (
    Apply,
    Constant,
    Variable,
    ancestors,
    general_toposort,
    walk,
)
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.random.var import (
    RandomGeneratorSharedVariable,
    RandomStateSharedVariable,
)
from pytensor.tensor.sharedvar import SharedVariable
from typing_extensions import TypeAlias

import pymc as pm

from pymc.backends.arviz import _DefaultTrace
from pymc.backends.base import MultiTrace
from pymc.blocking import PointType
from pymc.model import Model, modelcontext
from pymc.pytensorf import compile_pymc
from pymc.util import (
    RandomState,
    _get_seeds_per_chain,
    dataset_to_point_list,
    get_default_varnames,
    point_wrapper,
)

__all__ = (
    "compile_forward_sampling_function",
    "draw",
    "sample_prior_predictive",
    "sample_posterior_predictive",
    "sample_posterior_predictive_w",
)


ArrayLike: TypeAlias = Union[np.ndarray, List[float]]
PointList: TypeAlias = List[PointType]

_log = logging.getLogger(__name__)


def get_vars_in_point_list(trace, model):
    """Get the list of Variable instances in the model that have values stored in the trace."""
    if not isinstance(trace, MultiTrace):
        names_in_trace = list(trace[0])
    else:
        names_in_trace = trace.varnames
    traceable_varnames = {var.name for var in (model.free_RVs + model.deterministics)}
    vars_in_trace = [model[v] for v in names_in_trace if v in traceable_varnames]
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

    The goal of this function is to walk the pytensor computational graph from the list
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
    to set the ``givens`` argument of the pytensor function compilation. This will essentially
    replace a node in the computational graph with any other expression that has the same
    type as the desired node. Passing variables in the givens_dict is considered an intervention
    that might lead to different variable values from those that could have been seen during
    inference, as such, **any variable that is passed in the ``givens_dict`` will be considered
    volatile**.

    Parameters
    ----------
    outputs : List[pytensor.graph.basic.Variable]
        The list of variables that will be returned by the compiled function
    vars_in_trace : List[pytensor.graph.basic.Variable]
        The list of variables that are assumed to have values stored in the trace
    basic_rvs : Optional[List[pytensor.graph.basic.Variable]]
        A list of random variables that are defined in the model. This list (which could be the
        output of ``model.basic_RVs``) should have a reference to the variables that should
        be considered as random variable instances. This includes variables that have
        a ``RandomVariable`` owner op, but also unpure random variables like Mixtures, or
        Censored distributions.
    givens_dict : Optional[Dict[pytensor.graph.basic.Variable, Any]]
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
        Compiled forward sampling PyTensor function
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
            else pt.constant(value, dtype=getattr(node, "dtype", None), name=node.name),
        )
        for node, value in givens_dict.items()
    ]

    return (
        compile_pymc(inputs, fg.outputs, givens=givens, on_unused_input="ignore", **kwargs),
        set(basic_rvs) & (volatile_nodes - set(givens_dict)),  # Basic RVs that will be resampled
    )


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
        Keyword arguments for :func:`pymc.pytensorf.compile_pymc`.

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


def observed_dependent_deterministics(model: Model):
    """Find deterministics that depend directly on observed variables"""
    deterministics = model.deterministics
    observed_rvs = set(model.observed_RVs)
    blockers = model.basic_RVs
    return [
        deterministic
        for deterministic in deterministics
        if observed_rvs & set(ancestors([deterministic], blockers=blockers))
    ]


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
        are not allowed.
    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.
    return_inferencedata : bool
        Whether to return an :class:`arviz:arviz.InferenceData` (True) object or a dictionary (False).
        Defaults to True.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data`
    compile_kwargs: dict, optional
        Keyword arguments for :func:`pymc.pytensorf.compile_pymc`.

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
        vars_: Set[str] = {var.name for var in model.basic_RVs + model.deterministics}
    else:
        vars_ = set(var_names)

    names = sorted(get_default_varnames(vars_, include_transformed=False))
    vars_to_sample = [model[name] for name in names]

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
        Flag used to set the location of posterior predictive samples within the returned ``arviz.InferenceData`` object. If False, assumes samples are generated based on the fitting data to be used for posterior predictive checks, and samples are stored in the ``posterior_predictive``. If True, assumes samples are generated based on out-of-sample data as predictions, and samples are stored in the ``predictions`` group.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data` if ``predictions=False`` or to
        :func:`pymc.predictions_to_inference_data` otherwise.
    compile_kwargs: dict, optional
        Keyword arguments for :func:`pymc.pytensorf.compile_pymc`.

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
        vars_ = model.observed_RVs + observed_dependent_deterministics(model)

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
    raise FutureWarning(
        "The function `sample_posterior_predictive_w` has been removed in PyMC 4.3.0. "
        "Switch to `arviz.stats.weight_predictions`"
    )
