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
from typing import Optional

from arviz import InferenceData

from pymc.model import Model
from pymc.stats.log_density import compute_log_density

__all__ = "compute_log_prior"


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
