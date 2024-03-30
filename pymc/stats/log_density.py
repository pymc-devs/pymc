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
from collections.abc import Sequence
from typing import Optional, cast

from arviz import InferenceData, dict_to_dataset
from rich.progress import track

import pymc

from pymc.backends.arviz import _DefaultTrace, coords_and_dims_for_inferencedata
from pymc.model import Model, modelcontext
from pymc.pytensorf import PointFunc
from pymc.util import dataset_to_point_list

__all__ = ("compute_log_likelihood", "compute_log_prior")


def compute_log_likelihood(
    idata: InferenceData,
    *,
    var_names: Optional[Sequence[str]] = None,
    extend_inferencedata: bool = True,
    model: Optional[Model] = None,
    sample_dims: Sequence[str] = ("chain", "draw"),
    progressbar=True,
):
    """Compute elemwise log_likelihood of model given InferenceData with posterior group

    Parameters
    ----------
    idata : InferenceData
        InferenceData with posterior group
    var_names : sequence of str, optional
        List of Observed variable names for which to compute log_likelihood.
        Defaults to all observed variables.
    extend_inferencedata : bool, default True
        Whether to extend the original InferenceData or return a new one
    model : Model, optional
    sample_dims : sequence of str, default ("chain", "draw")
    progressbar : bool, default True

    Returns
    -------
    idata : InferenceData
        InferenceData with log_likelihood group
    """
    return compute_log_density(
        idata=idata,
        var_names=var_names,
        extend_inferencedata=extend_inferencedata,
        model=model,
        kind="likelihood",
        sample_dims=sample_dims,
        progressbar=progressbar,
    )


def compute_log_prior(
    idata: InferenceData,
    var_names: Optional[Sequence[str]] = None,
    extend_inferencedata: bool = True,
    model: Optional[Model] = None,
    sample_dims: Sequence[str] = ("chain", "draw"),
    progressbar=True,
):
    """Compute elemwise log_prior of model given InferenceData with posterior group

    Parameters
    ----------
    idata : InferenceData
        InferenceData with posterior group
    var_names : sequence of str, optional
        List of Observed variable names for which to compute log_prior.
        Defaults to all all free variables.
    extend_inferencedata : bool, default True
        Whether to extend the original InferenceData or return a new one
    model : Model, optional
    sample_dims : sequence of str, default ("chain", "draw")
    progressbar : bool, default True

    Returns
    -------
    idata : InferenceData
        InferenceData with log_prior group
    """
    return compute_log_density(
        idata=idata,
        var_names=var_names,
        extend_inferencedata=extend_inferencedata,
        model=model,
        kind="prior",
        sample_dims=sample_dims,
        progressbar=progressbar,
    )


def compute_log_density(
    idata: InferenceData,
    *,
    var_names: Optional[Sequence[str]] = None,
    extend_inferencedata: bool = True,
    model: Optional[Model] = None,
    kind="likelihood",
    sample_dims: Sequence[str] = ("chain", "draw"),
    progressbar=True,
):
    """
    Compute elemwise log_likelihood or log_prior of model given InferenceData with posterior group
    """

    posterior = idata["posterior"]

    model = modelcontext(model)

    if kind not in ("likelihood", "prior"):
        raise ValueError("kind must be either 'likelihood' or 'prior'")

    if kind == "likelihood":
        target_rvs = model.observed_RVs
        target_str = "observed_RVs"
    else:
        target_rvs = model.free_RVs
        target_str = "free_RVs"

    if var_names is None:
        vars = target_rvs
        var_names = tuple(rv.name for rv in vars)
    else:
        vars = [model.named_vars[name] for name in var_names]
        if not set(vars).issubset(target_rvs):
            raise ValueError(f"var_names must refer to {target_str} in the model. Got: {var_names}")

    # We need to temporarily disable transforms, because the InferenceData only keeps the untransformed values
    try:
        original_rvs_to_values = model.rvs_to_values
        original_rvs_to_transforms = model.rvs_to_transforms

        model.rvs_to_values = {
            rv: rv.clone() if rv not in model.observed_RVs else value
            for rv, value in model.rvs_to_values.items()
        }
        model.rvs_to_transforms = {rv: None for rv in model.basic_RVs}

        elemwise_logdens_fn = model.compile_fn(
            inputs=model.value_vars,
            outs=model.logp(vars=vars, sum=False),
            on_unused_input="ignore",
        )
        elemwise_logdens_fn = cast(PointFunc, elemwise_logdens_fn)
    finally:
        model.rvs_to_values = original_rvs_to_values
        model.rvs_to_transforms = original_rvs_to_transforms

    # Ignore Deterministics
    posterior_values = posterior[[rv.name for rv in model.free_RVs]]
    posterior_pts, stacked_dims = dataset_to_point_list(posterior_values, sample_dims)

    n_pts = len(posterior_pts)
    logdens_dict = _DefaultTrace(n_pts)
    if progressbar:
        indices = track(range(n_pts), description="Computing log density")
    else:
        indices = range(n_pts)

    for idx in indices:
        logdenss_pts = elemwise_logdens_fn(posterior_pts[idx])
        for rv_name, rv_logdens in zip(var_names, logdenss_pts):
            logdens_dict.insert(rv_name, rv_logdens, idx)

    logdens_trace = logdens_dict.trace_dict
    for key, array in logdens_trace.items():
        logdens_trace[key] = array.reshape(
            (*[len(coord) for coord in stacked_dims.values()], *array.shape[1:])
        )

    coords, dims = coords_and_dims_for_inferencedata(model)
    logdens_dataset = dict_to_dataset(
        logdens_trace,
        library=pymc,
        dims=dims,
        coords=coords,
        default_dims=list(sample_dims),
        skip_event_dims=True,
    )

    if extend_inferencedata:
        idata.add_groups({f"log_{kind}": logdens_dataset})
        return idata
    else:
        return logdens_dataset
