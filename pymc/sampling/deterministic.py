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

import xarray

from xarray import Dataset

from pymc.backends.arviz import apply_function_over_dataset, coords_and_dims_for_inferencedata
from pymc.model.core import Model, modelcontext


def compute_deterministics(
    dataset: Dataset,
    *,
    var_names: Sequence[str] | None = None,
    model: Model | None = None,
    sample_dims: Sequence[str] = ("chain", "draw"),
    merge_dataset: bool = False,
    progressbar: bool = True,
    compile_kwargs: dict | None = None,
) -> Dataset:
    """Compute model deterministics given a dataset with values for model variables.

    Parameters
    ----------
    dataset : Dataset
        Dataset with values for model variables. Commonly InferenceData["posterior"].
    var_names : sequence of str, optional
        List of names of deterministic variable to compute.
        If None, compute all deterministics in the model.
    model : Model, optional
        Model to use. If None, use context model.
    sample_dims : sequence of str, default ("chain", "draw")
        Sample (batch) dimensions of the dataset over which to compute the deterministics.
    merge_dataset : bool, default False
        Whether to extend the original dataset or return a new one.
    progressbar : bool, default True
        Whether to display a progress bar in the command line.
    progressbar_theme : Theme, optional
        Custom theme for the progress bar.
    compile_kwargs: dict, optional
        Additional arguments passed to `model.compile_fn`.

    Returns
    -------
    Dataset
        Dataset with values for the deterministics.


    Examples
    --------
    .. code:: python

        import pymc as pm

        with pm.Model(coords={"group": (0, 2, 4)}) as m:
            mu_raw = pm.Normal("mu_raw", 0, 1, dims="group")
            mu = pm.Deterministic("mu", mu_raw.cumsum(), dims="group")

            trace = pm.sample(var_names=["mu_raw"], chains=2, tune=5 draws=5)

        assert "mu" not in trace.posterior

        with m:
            trace.posterior = pm.compute_deterministics(trace.posterior, merge_dataset=True)

        assert "mu" in trace.posterior


    """
    model = modelcontext(model)

    if var_names is None:
        deterministics = list(model.deterministics)
        var_names = [det.name for det in deterministics]
    else:
        deterministics = [model[var_name] for var_name in var_names]
        if not set(deterministics).issubset(set(model.deterministics)):
            raise ValueError("Not all var_names corresponded to model deterministics")

    fn = model.compile_fn(
        inputs=model.free_RVs,
        outs=deterministics,
        on_unused_input="ignore",
        **(compile_kwargs or {}),
    )

    coords, dims = coords_and_dims_for_inferencedata(model)

    new_dataset = apply_function_over_dataset(
        fn,
        dataset[[rv.name for rv in model.free_RVs]],
        output_var_names=var_names,
        dims=dims,
        coords=coords,
        sample_dims=sample_dims,
        progressbar=progressbar,
    )

    if merge_dataset:
        new_dataset = xarray.merge([dataset, new_dataset], compat="override")

    return new_dataset
