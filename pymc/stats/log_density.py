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
from typing import Any, Literal

from arviz import InferenceData
from xarray import Dataset

from pymc.backends.arviz import (
    apply_function_over_dataset,
    coords_and_dims_for_inferencedata,
)
from pymc.model import Model, modelcontext

__all__ = ("compute_log_likelihood", "compute_log_prior")

from pymc.model.transform.conditioning import remove_value_transforms


def compute_log_likelihood(
    idata: InferenceData,
    *,
    var_names: Sequence[str] | None = None,
    extend_inferencedata: bool = True,
    model: Model | None = None,
    sample_dims: Sequence[str] = ("chain", "draw"),
    progressbar=True,
    compile_kwargs: dict[str, Any] | None = None,
):
    """Compute elemwise log_likelihood of model given InferenceData with posterior group.

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
    compile_kwargs : dict[str, Any] | None
        Extra compilation arguments to supply to :py:func:`~pymc.stats.compute_log_density`

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
        compile_kwargs=compile_kwargs,
    )


def compute_log_prior(
    idata: InferenceData,
    var_names: Sequence[str] | None = None,
    extend_inferencedata: bool = True,
    model: Model | None = None,
    sample_dims: Sequence[str] = ("chain", "draw"),
    progressbar=True,
    compile_kwargs=None,
):
    """Compute elemwise log_prior of model given InferenceData with posterior group.

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
    compile_kwargs : dict[str, Any] | None
        Extra compilation arguments to supply to :py:func:`~pymc.stats.compute_log_density`

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
        compile_kwargs=compile_kwargs,
    )


def compute_log_density(
    idata: InferenceData,
    *,
    var_names: Sequence[str] | None = None,
    extend_inferencedata: bool = True,
    model: Model | None = None,
    kind: Literal["likelihood", "prior"] = "likelihood",
    sample_dims: Sequence[str] = ("chain", "draw"),
    progressbar=True,
    compile_kwargs=None,
) -> InferenceData | Dataset:
    """
    Compute elemwise log_likelihood or log_prior of model given InferenceData with posterior group.

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
    kind: Literal["likelihood", "prior"]
        Whether to compute the log density of the observed random variables (likelihood)
        or to compute the log density of the latent random variables (prior). This
        parameter determines the group that gets added to the returned `~arviz.InferenceData` object.
    sample_dims : sequence of str, default ("chain", "draw")
    progressbar : bool, default True
    compile_kwargs : dict[str, Any] | None
        Extra compilation arguments to supply to :py:func:`pymc.model.core.Model.compile_fn`

    Returns
    -------
    idata : InferenceData
        InferenceData with the ``log_likelihood`` group when ``kind == "likelihood"``
        or the ``log_prior`` group when ``kind == "prior"``.
    """
    posterior = idata["posterior"]

    model = modelcontext(model)
    if compile_kwargs is None:
        compile_kwargs = {}

    if kind not in ("likelihood", "prior"):
        raise ValueError("kind must be either 'likelihood' or 'prior'")

    # We need to disable transforms, because the InferenceData only keeps the untransformed values
    umodel = remove_value_transforms(model)

    if kind == "likelihood":
        target_rvs = list(umodel.observed_RVs)
        target_str = "observed_RVs"
    else:
        target_rvs = list(umodel.free_RVs)
        target_str = "free_RVs"

    if var_names is None:
        vars = target_rvs
        var_names = tuple(rv.name for rv in vars)
    else:
        vars = [umodel.named_vars[name] for name in var_names]
        if not set(vars).issubset(target_rvs):
            raise ValueError(f"var_names must refer to {target_str} in the model. Got: {var_names}")

    elemwise_logdens_fn = umodel.compile_fn(
        inputs=umodel.value_vars,
        outs=umodel.logp(vars=vars, sum=False),
        on_unused_input="ignore",
        **compile_kwargs,
    )

    coords, dims = coords_and_dims_for_inferencedata(umodel)

    logdens_dataset = apply_function_over_dataset(
        elemwise_logdens_fn,
        posterior[[rv.name for rv in umodel.free_RVs]],
        output_var_names=var_names,
        sample_dims=sample_dims,
        dims=dims,
        coords=coords,
        progressbar=progressbar,
    )

    if extend_inferencedata:
        idata.add_groups({f"log_{kind}": logdens_dataset})
        return idata
    else:
        return logdens_dataset
