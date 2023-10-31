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
from typing import Optional, Sequence, cast

from arviz import InferenceData, dict_to_dataset
from fastprogress import progress_bar

import pymc

from pymc.backends.arviz import _DefaultTrace, coords_and_dims_for_inferencedata
from pymc.model import Model, modelcontext
from pymc.pytensorf import PointFunc
from pymc.util import dataset_to_point_list

__all__ = ("compute_log_likelihood",)


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
        List of Observed variable names for which to compute log_likelihood. Defaults to all observed variables
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

    posterior = idata["posterior"]

    model = modelcontext(model)

    if var_names is None:
        observed_vars = model.observed_RVs
        var_names = tuple(rv.name for rv in observed_vars)
    else:
        observed_vars = [model.named_vars[name] for name in var_names]
        if not set(observed_vars).issubset(model.observed_RVs):
            raise ValueError(f"var_names must refer to observed_RVs in the model. Got: {var_names}")

    # We need to temporarily disable transforms, because the InferenceData only keeps the untransformed values
    # pylint: disable=used-before-assignment
    try:
        original_rvs_to_values = model.rvs_to_values
        original_rvs_to_transforms = model.rvs_to_transforms

        model.rvs_to_values = {
            rv: rv.clone() if rv not in model.observed_RVs else value
            for rv, value in model.rvs_to_values.items()
        }
        model.rvs_to_transforms = {rv: None for rv in model.basic_RVs}

        elemwise_loglike_fn = model.compile_fn(
            inputs=model.value_vars,
            outs=model.logp(vars=observed_vars, sum=False),
            on_unused_input="ignore",
        )
        elemwise_loglike_fn = cast(PointFunc, elemwise_loglike_fn)
    finally:
        model.rvs_to_values = original_rvs_to_values
        model.rvs_to_transforms = original_rvs_to_transforms
    # pylint: enable=used-before-assignment

    # Ignore Deterministics
    posterior_values = posterior[[rv.name for rv in model.free_RVs]]
    posterior_pts, stacked_dims = dataset_to_point_list(posterior_values, sample_dims)
    n_pts = len(posterior_pts)
    loglike_dict = _DefaultTrace(n_pts)
    indices = range(n_pts)
    if progressbar:
        indices = progress_bar(indices, total=n_pts, display=progressbar)

    for idx in indices:
        loglikes_pts = elemwise_loglike_fn(posterior_pts[idx])
        for rv_name, rv_loglike in zip(var_names, loglikes_pts):
            loglike_dict.insert(rv_name, rv_loglike, idx)

    loglike_trace = loglike_dict.trace_dict
    for key, array in loglike_trace.items():
        loglike_trace[key] = array.reshape(
            (*[len(coord) for coord in stacked_dims.values()], *array.shape[1:])
        )

    coords, dims = coords_and_dims_for_inferencedata(model)
    loglike_dataset = dict_to_dataset(
        loglike_trace,
        library=pymc,
        dims=dims,
        coords=coords,
        default_dims=list(sample_dims),
        skip_event_dims=True,
    )

    if extend_inferencedata:
        idata.add_groups(dict(log_likelihood=loglike_dataset))
        return idata
    else:
        return loglike_dataset
