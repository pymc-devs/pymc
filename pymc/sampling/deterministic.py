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
import warnings

from collections.abc import Sequence

from xarray import Dataset, DataTree, merge

from pymc.backends.arviz import apply_function_over_dataset, coords_and_dims_for_inferencedata
from pymc.model.core import BaseModel, modelcontext
from pymc.pytensorf import resolve_backend_compile_kwargs


def _select_group(dataset: Dataset | DataTree, group: str | None) -> Dataset | DataTree:
    """Select the relevant group when a whole InferenceData object is passed."""
    if not (isinstance(dataset, DataTree) and dataset.children):
        if group is not None:
            raise ValueError(
                "The `group` argument can only be used when passing a whole InferenceData object, "
                "not a single group."
            )
        return dataset

    if group is None:
        for default_group in ("posterior", "prior"):
            if default_group in dataset.children:
                group = default_group
                break
        else:
            raise ValueError(
                "InferenceData has neither a `posterior` nor a `prior` group. "
                f"Pass `group` explicitly, one of: {sorted(dataset.children)}"
            )
    elif group not in dataset.children:
        raise ValueError(
            f"InferenceData has no group {group!r}. Available groups: {sorted(dataset.children)}"
        )

    return dataset.children[group]


def compute_deterministics(
    dataset: Dataset | DataTree,
    *,
    group: str | None = None,
    var_names: Sequence[str] | None = None,
    model: BaseModel | None = None,
    sample_dims: Sequence[str] = ("chain", "draw"),
    merge_dataset: bool = False,
    extend_dataset: bool = False,
    progressbar: bool = True,
    backend: str | None = None,
    compile_kwargs: dict | None = None,
) -> Dataset | DataTree:
    """Compute model deterministics given a dataset with values for model variables.

    Parameters
    ----------
    dataset : Dataset or DataTree
        Dataset with values for model variables, such as ``idata.posterior``.
        A whole InferenceData object can also be passed, in which case the group
        given by ``group`` is used.
    group : str, optional
        Which group to use when ``dataset`` is a whole InferenceData object.
        If None, "posterior" is used, falling back to "prior" when there is no
        posterior group. Cannot be used when a single group is passed directly.
    var_names : sequence of str, optional
        List of names of deterministic variable to compute.
        If None, compute all deterministics in the model.
    model : BaseModel, optional
        Model to use. If None, use context model.
    sample_dims : sequence of str, default ("chain", "draw")
        Sample (batch) dimensions of the dataset over which to compute the deterministics.
    merge_dataset : bool, default False
        Whether to include the values of the original dataset in the returned one.

        .. deprecated::
            ``merge_dataset`` is deprecated and will be removed in a future release.
            Use ``extend_dataset`` instead.
    extend_dataset : bool, default False
        Whether to add the deterministics to the original dataset in place, instead of
        returning a new one. The mutated input object is returned, so for an InferenceData
        the deterministics end up in the selected group.
        Cannot be combined with ``merge_dataset``.
    progressbar : bool, default True
        Whether to display a progress bar in the command line.
    progressbar_theme : Theme, optional
        Custom theme for the progress bar.
    backend: str, optional
        Which computational backend to use. Recommended to be one of "numba", "c", and "jax".
    compile_kwargs: dict, optional
        Additional arguments passed to `model.compile_fn`.
        ``compile_kwargs["mode"]`` cannot be combined with ``backend``.

    Returns
    -------
    Dataset or DataTree
        Dataset with values for the deterministics. When ``merge_dataset`` is True,
        the values of the input dataset (or of the selected group) are included as well.
        When ``extend_dataset`` is True, the input object is returned instead, with the
        deterministics added to it.


    Examples
    --------
    .. code-block:: python

        import pymc as pm

        with pm.Model(coords={"group": (0, 2, 4)}) as m:
            mu_raw = pm.Normal("mu_raw", 0, 1, dims="group")
            mu = pm.Deterministic("mu", mu_raw.cumsum(), dims="group")

            trace = pm.sample(var_names=["mu_raw"], chains=2, tune=5, draws=5)

        assert "mu" not in trace.posterior

        with m:
            pm.compute_deterministics(trace, extend_dataset=True)

        assert "mu" in trace.posterior


    """
    if merge_dataset and extend_dataset:
        raise ValueError(
            "`merge_dataset` and `extend_dataset` cannot be combined. "
            "`extend_dataset` already keeps the values of the original dataset."
        )

    if merge_dataset:
        warnings.warn(
            "`merge_dataset` is deprecated and will be removed in a future release. "
            "Use `extend_dataset=True` to add the deterministics to the original dataset, "
            "passing a `.copy()` of it if it should not be mutated.",
            FutureWarning,
        )

    original_object = dataset
    dataset = _select_group(dataset, group)

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
        **resolve_backend_compile_kwargs(backend, compile_kwargs),
    )

    coords, dims = coords_and_dims_for_inferencedata(model)

    group_dataset: Dataset = dataset.dataset if isinstance(dataset, DataTree) else dataset

    new_dataset = apply_function_over_dataset(
        fn,
        group_dataset[[rv.name for rv in model.free_RVs]],
        output_var_names=var_names,
        dims=dims,
        coords=coords,
        sample_dims=sample_dims,
        progressbar=progressbar,
    )

    if extend_dataset:
        dataset.update(new_dataset)
        return original_object

    if merge_dataset:
        new_dataset = merge([group_dataset, new_dataset], compat="override")

    return new_dataset
