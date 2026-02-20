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

from arviz import InferenceData
from pytensor import tensor as pt
from pytensor.graph import vectorize_graph
from pytensor.graph.basic import (
    Apply,
    Constant,
    Variable,
)
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.traversal import ancestors, general_toposort, walk
from pytensor.tensor.random.var import RandomGeneratorSharedVariable
from pytensor.tensor.sharedvar import SharedVariable, TensorSharedVariable
from pytensor.tensor.variable import TensorConstant
from rich.theme import Theme

import pymc as pm

from pymc.backends.arviz import _DefaultTrace, dataset_to_point_list
from pymc.backends.base import MultiTrace
from pymc.blocking import PointType
from pymc.distributions.shape_utils import change_dist_size
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


def get_constant_coords(trace_coords: dict[str, np.ndarray], model: Model) -> set:
    """Get the set of coords that have remained constant between the trace and model."""
    constant_coords = set()
    for dim, coord in trace_coords.items():
        current_coord = model.coords.get(dim, None)
        current_length = model.dim_lengths.get(dim, None)
        if isinstance(current_length, TensorSharedVariable):
            current_length = current_length.get_value()
        elif isinstance(current_length, TensorConstant):
            current_length = current_length.data
        if (
            current_coord is not None
            and len(coord) == len(current_coord)
            and np.all(coord == current_coord)
        ) or (
            # Coord was defined without values (only length)
            current_coord is None and len(coord) == current_length
        ):
            constant_coords.add(dim)
    return constant_coords


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
    outputs: list[Variable],
    vars_in_trace: list[Variable],
    basic_rvs: list[Variable] | None = None,
    givens_dict: dict[Variable, Any] | None = None,
    constant_data: dict[str, np.ndarray] | None = None,
    constant_coords: set[str] | None = None,
    **kwargs,
) -> tuple[Callable[..., np.ndarray | list[np.ndarray]], set[Variable]]:
    """Compile a function to draw samples, conditioned on the values of some variables.

    The goal of this function is to walk the pytensor computational graph from the list
    of output nodes down to the root nodes, and then compile a function that will produce
    values for these output nodes. The compiled function will take as inputs the subset of
    variables in the ``vars_in_trace`` that are deemed to not be **volatile**.

    Volatile variables are variables whose values could change between runs of the
    compiled function or after inference has been run. These variables are:

    - Variables in the outputs list
    - ``SharedVariable`` instances that are not ``RandomGeneratorSharedVariable``, and whose values changed with respect to what they were at inference time
    - Variables that are in the `basic_rvs` list but not in the ``vars_in_trace`` list
    - Variables that are keys in the ``givens_dict``
    - Variables that have volatile inputs

    Concretely, this function can be used to compile a function to sample from the
    posterior predictive distribution of a model that has variables that are conditioned
    on ``Data`` instances. The variables that depend on the mutable data that have changed
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
        A dictionary that maps the names of ``Data`` instances to their
        corresponding values at inference time. If a model was created with ``Data``, these
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
    nodes: list[Variable] = general_toposort(
        fg.outputs, deps=lambda x: x.owner.inputs if x.owner else []
    )  # type: ignore[call-overload]
    volatile_nodes: set[Any] = set()
    for node in nodes:
        if (
            node in fg.outputs
            or node in givens_dict
            or (  # SharedVariables, except RandomState/Generators
                isinstance(node, SharedVariable)
                and not isinstance(node, RandomGeneratorSharedVariable)
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
                node.owner is None and not isinstance(node, Constant | SharedVariable)
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
            if isinstance(value, Variable | Apply)
            else pt.constant(value, dtype=getattr(node, "dtype", None), name=node.name),
        )
        for node, value in givens_dict.items()
    ]

    return (
        compile(inputs, fg.outputs, givens=givens, on_unused_input="ignore", **kwargs),
        set(basic_rvs) & (volatile_nodes - set(givens_dict)),  # Basic RVs that will be resampled
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
    samples: int | None = None,
) -> InferenceData: ...
@overload
def sample_prior_predictive(
    draws: int = 500,
    model: Model | None = None,
    var_names: Iterable[str] | None = None,
    random_seed: RandomState = None,
    return_inferencedata: Literal[False] = False,
    idata_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
    samples: int | None = None,
) -> dict[str, np.ndarray]: ...
def sample_prior_predictive(
    draws: int = 500,
    model: Model | None = None,
    var_names: Iterable[str] | None = None,
    random_seed: RandomState = None,
    return_inferencedata: bool = True,
    idata_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
    samples: int | None = None,
) -> InferenceData | dict[str, np.ndarray]:
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
        Whether to return an :class:`arviz:arviz.InferenceData` (True) object or a dictionary (False).
        Defaults to True.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data`
    compile_kwargs: dict, optional
        Keyword arguments for :func:`pymc.pytensorf.compile_pymc`.
    samples : int
        Number of samples from the prior predictive to generate. Deprecated in favor of `draws`.

    Returns
    -------
    arviz.InferenceData or Dict
        An ArviZ ``InferenceData`` object containing the prior and prior predictive samples (default),
        or a dictionary with variable names as keys and samples as numpy arrays.
    """
    if samples is not None:
        warnings.warn(
            f"The samples argument has been deprecated in favor of draws. Use draws={samples} going forward.",
            DeprecationWarning,
            stacklevel=1,
        )

        draws = samples

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
    var_names: list[str] | None = None,
    sample_dims: list[str] | None = None,
    random_seed: RandomState = None,
    progressbar: bool = True,
    progressbar_theme: Theme | None = default_progress_theme,
    return_inferencedata: Literal[True] = True,
    extend_inferencedata: bool = False,
    predictions: bool = False,
    idata_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
) -> InferenceData: ...
@overload
def sample_posterior_predictive(
    trace,
    model: Model | None = None,
    var_names: list[str] | None = None,
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
    var_names: list[str] | None = None,
    sample_dims: list[str] | None = None,
    random_seed: RandomState = None,
    progressbar: bool = True,
    progressbar_theme: Theme | None = default_progress_theme,
    return_inferencedata: bool = True,
    extend_inferencedata: bool = False,
    predictions: bool = False,
    idata_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
) -> InferenceData | dict[str, np.ndarray]:
    """Generate forward samples for `var_names`, conditioned on the posterior samples of variables found in the `trace`.

    This method can be used to perform different kinds of model predictions, including posterior predictive checks.

    The matching of unobserved model variables, and posterior samples in the `trace` is made based on the variable
    names. Therefore, a different model than the one used for posterior sampling may be used for posterior predictive
    sampling, as long as the variables whose posterior we want to condition on have the same name, and compatible shape
    and coordinates.


    Parameters
    ----------
    trace : backend, list, xarray.Dataset, arviz.InferenceData, or MultiTrace
        Trace generated from MCMC sampling, or a list of dicts (eg. points or from :func:`~pymc.find_MAP`),
        or :class:`xarray.Dataset` (eg. InferenceData.posterior or InferenceData.prior)
    model : Model (optional if in ``with`` context)
        Model to be used to generate the posterior predictive samples. It will
        generally be the model used to generate the `trace`, but it doesn't need to be.
    var_names : Iterable[str], optional
        Names of variables for which to compute the posterior predictive samples.
        By default, only observed variables are sampled.
        See the example below for what happens when this argument is customized.
    sample_dims : list of str, optional
        Dimensions over which to loop and generate posterior predictive samples.
        When ``sample_dims`` is ``None`` (default) both "chain" and "draw" are considered sample
        dimensions. Only taken into account when `trace` is InferenceData or Dataset.
    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.
    progressbar : bool
        Whether to display a progress bar in the command line. The bar shows the percentage
        of completion, the sampling speed in samples per second (SPS), and the estimated remaining
        time until completion ("expected time of arrival"; ETA).
    return_inferencedata : bool, default True
        Whether to return an :class:`arviz:arviz.InferenceData` (True) object or a dictionary (False).
    extend_inferencedata : bool, default False
        Whether to automatically use :meth:`arviz.InferenceData.extend` to add the posterior predictive samples to
        `trace` or not. If True, `trace` is modified inplace but still returned.
    predictions : bool, default False
        Flag used to set the location of posterior predictive samples within the returned ``arviz.InferenceData`` object.
        If False, assumes samples are generated based on the fitting data to be used for posterior predictive checks,
        and samples are stored in the ``posterior_predictive``. If True, assumes samples are generated based on
        out-of-sample data as predictions, and samples are stored in the ``predictions`` group.
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
    Posterior predictive checks and predictions
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    The most common use of `sample_posterior_predictive` is to perform posterior predictive checks (in-sample predictions)
    and new model predictions (out-of-sample predictions).

    .. code-block:: python

        import pymc as pm

        with pm.Model(coords={"trial": [0, 1, 2]}) as model:
            x = pm.Data("x", [-1, 0, 1], dims=["trial"])
            beta = pm.Normal("beta")
            noise = pm.HalfNormal("noise")
            y = pm.Normal("y", mu=x * beta, sigma=noise, observed=[-2, 0, 3], dims=["trial"])

            idata = pm.sample()
            # in-sample predictions
            posterior_predictive = pm.sample_posterior_predictive(idata).posterior_predictive

        with model:
            pm.set_data({"x": [-2, 2]}, coords={"trial": [3, 4]})
            # out-of-sample predictions
            predictions = pm.sample_posterior_predictive(idata, predictions=True).predictions


    Using different models
    ^^^^^^^^^^^^^^^^^^^^^^

    It's common to use the same model for posterior and posterior predictive sampling, but this is not required.
    The matching between unobserved model variables and posterior samples is based on the name alone.

    For the last example we could have created a new predictions model. Note that we have to specify
    `var_names` explicitly, because the newly defined `y` was not given any observations:

    .. code-block:: python

        with pm.Model(coords={"trial": [3, 4]}) as predictions_model:
            x = pm.Data("x", [-2, 2], dims=["trial"])
            beta = pm.Normal("beta")
            noise = pm.HalfNormal("noise")
            y = pm.Normal("y", mu=x * beta, sigma=noise, dims=["trial"])

            predictions = pm.sample_posterior_predictive(
                idata, var_names=["y"], predictions=True
            ).predictions


    The new model may even have a different structure and unobserved variables that don't exist in the trace.
    These variables will also be forward sampled. In the following example we added a new ``extra_noise``
    variable between the inferred posterior ``noise`` and the new StudentT observational distribution  ``y``:

    .. code-block:: python

        with pm.Model(coords={"trial": [3, 4]}) as distinct_predictions_model:
            x = pm.Data("x", [-2, 2], dims=["trial"])
            beta = pm.Normal("beta")
            noise = pm.HalfNormal("noise")
            extra_noise = pm.HalfNormal("extra_noise", sigma=noise)
            y = pm.StudentT("y", nu=4, mu=x * beta, sigma=extra_noise, dims=["trial"])

            predictions = pm.sample_posterior_predictive(
                idata, var_names=["y"], predictions=True
            ).predictions


    For more about out-of-model predictions, see this `blog post <https://www.pymc-labs.com/blog-posts/out-of-model-predictions-with-pymc/>`_.

    The behavior of `var_names`
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    The function returns forward samples for any variable included in `var_names`,
    conditioned on the values of other random variables found in the trace.

    To ensure the samples are internally consistent, any random variable that depends
    on another random variable that is being sampled is itself sampled, even if
    this variable is present in the trace and was not included in `var_names`.
    The final list of variables being sampled is shown in the log output.

    Note that if a random variable has no dependency on other random variables,
    these forward samples are equivalent to their prior samples.
    Likewise, if all random variables are being sampled, the behavior of this function
    is equivalent to that of :func:`~pymc.sample_prior_predictive`.

    .. warning:: A random variable included in `var_names` will never be copied from the posterior group. It will always be sampled as described above. If you want, you can copy manually via ``idata.posterior_predictive["var_name"] = idata.posterior["var_name"]``.


    The following code block explores how the behavior changes with different `var_names`:

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

    Default behavior. Generate samples of ``obs``, conditioned on the posterior samples of ``z`` found in the trace.
    These are often referred to as posterior predictive samples in the literature:

    .. code-block:: python

        with model:
            pm.sample_posterior_predictive(idata, var_names=["obs"], **kwargs)
            # Sampling: [obs]

    Re-compute the deterministic variable ``det``, conditioned on the posterior samples of ``z`` found in the trace:

    .. code :: python

          pm.sample_posterior_predictive(idata, var_names=["det"], **kwargs)
          # Sampling: []

    Generate samples of ``z`` and ``det``, conditioned on the posterior samples of ``x`` and ``y`` found in the trace.

    .. code :: python

        with model:
            pm.sample_posterior_predictive(idata, var_names=["z", "det"], **kwargs)
            # Sampling: [z]


    Generate samples of ``y``, ``z`` and ``det``, conditioned on the posterior samples of ``x`` found in the trace.

    Note: The samples of ``y`` are equivalent to its prior, since it does not depend on any other variables.
    In contrast, the samples of ``z`` and ``det`` depend on the new samples of ``y`` and the posterior samples of
    ``x`` found in the trace.

    .. code :: python

        with model:
            pm.sample_posterior_predictive(idata, var_names=["y", "z", "det"], **kwargs)
            # Sampling: [y, z]


    Same as before, except ``z`` is not stored in the returned trace.
    For computing ``det`` we still have to sample ``z`` as it depends on ``y``, which is also being sampled.

    .. code :: python

        with model:
            pm.sample_posterior_predictive(idata, var_names=["y", "det"], **kwargs)
            # Sampling: [y, z]

    Every random variable is sampled. This is equivalent to calling :func:`~pymc.sample_prior_predictive`

    .. code :: python

        with model:
            pm.sample_posterior_predictive(idata, var_names=["x", "y", "z", "det", "obs"], **kwargs)
            # Sampling: [x, y, z, obs]


    .. danger:: Including a :func:`~pymc.Deterministic` in `var_names` may incorrectly force a random variable to be resampled, as happens with ``z`` in the following example:


    .. code :: python

        with pm.Model() as model:
          x = pm.Normal("x")
          y = pm.Normal("y")
          det_xy = pm.Deterministic("det_xy", x + y**2)
          z = pm.Normal("z", det_xy)
          det_z = pm.Deterministic("det_z", pm.math.exp(z))
          obs = pm.Normal("obs", det_z, 1, observed=[20])

          idata = pm.sample(tune=10, draws=10, chains=2, **kwargs)

          pm.sample_posterior_predictive(idata, var_names=["det_xy", "det_z"], **kwargs)
          # Sampling: [z]


    Controlling the number of samples
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    You can manipulate the InferenceData to control the number of samples

    .. code-block:: python

        import pymc as pm

        with pm.Model() as model:
            ...
            idata = pm.sample()

    Generate 1 posterior predictive sample for every 5 posterior samples.

    .. code-block:: python

        thinned_idata = idata.sel(draw=slice(None, None, 5))
        with model:
            idata.extend(pm.sample_posterior_predictive(thinned_idata))


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
    if idata_kwargs is None:
        idata_kwargs = {}
    else:
        idata_kwargs = idata_kwargs.copy()
    if sample_dims is None:
        sample_dims = ["chain", "draw"]
    constant_data: dict[str, np.ndarray] = {}
    trace_coords: dict[str, np.ndarray] = {}
    if "coords" not in idata_kwargs:
        idata_kwargs["coords"] = {}
    idata: InferenceData | None = None
    observed_data = None
    stacked_dims = None
    if isinstance(trace, InferenceData):
        _constant_data = getattr(trace, "constant_data", None)
        if _constant_data is not None:
            trace_coords.update({str(k): v.data for k, v in _constant_data.coords.items()})
            constant_data.update({str(k): v.data for k, v in _constant_data.items()})
        idata = trace
        observed_data = trace.get("observed_data", None)
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

    constant_coords = get_constant_coords(trace_coords, model)

    if var_names is not None:
        vars_ = [model[x] for x in var_names]
    else:
        observed_vars = model.observed_RVs
        if observed_data is not None:
            observed_vars += [
                model[x] for x in observed_data if x in model and x not in observed_vars
            ]
        vars_ = observed_vars + observed_dependent_deterministics(model, observed_vars)

    vars_to_sample = list(get_default_varnames(vars_, include_transformed=False))

    if not vars_to_sample:
        if return_inferencedata and not extend_inferencedata:
            return InferenceData()
        elif return_inferencedata and extend_inferencedata:
            return trace if idata is None else idata
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
    _log.info(f"Sampling: {sorted(volatile_basic_rvs, key=lambda var: var.name)}")  # type: ignore[arg-type, return-value]
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

                values = sampler_fn(**param)

                for k, v in zip(vars_, values):
                    ppc_trace_t.insert(k.name, v, idx)

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
        idata.extend(idata_pp)
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
