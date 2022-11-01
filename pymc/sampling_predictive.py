#   Copyright 2022 The PyMC Developers
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

from typing import Any, Dict, Iterable, List, Optional, Set, Union, cast

import numpy as np
import xarray

from arviz import InferenceData
from fastprogress.fastprogress import progress_bar

import pymc as pm

from pymc.backends.arviz import _DefaultTrace
from pymc.backends.base import MultiTrace
from pymc.model import Model, modelcontext
from pymc.sampling_utils import (
    ArrayLike,
    PointList,
    RandomState,
    _get_seeds_per_chain,
    compile_forward_sampling_function,
    get_vars_in_point_list,
)
from pymc.util import dataset_to_point_list, get_default_varnames, point_wrapper

__all__ = (
    "sample_prior_predictive",
    "sample_posterior_predictive",
    "sample_posterior_predictive_w",
)


_log = logging.getLogger("pymc")


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
    raise FutureWarning(
        "The function `sample_posterior_predictive_w` has been removed in PyMC 4.3.0. "
        "Switch to `arviz.stats.weight_predictions`"
    )
