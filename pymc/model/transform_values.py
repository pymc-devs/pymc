#   Copyright 2026 - present The PyMC Developers
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

import pytensor
import pytensor.tensor as pt
import xarray as xr

from pytensor.graph.basic import Variable
from pytensor.xtensor.vectorization import vectorize_graph as xvectorize_graph

from pymc.model import BaseModel, modelcontext
from pymc.pytensorf import compile, replace_vars_in_graphs


def _build_transform_graph(
    model: BaseModel,
    forward: bool,
) -> tuple[list[Variable], list[Variable]]:
    """Build a per-sample graph that applies transforms to all free RVs.

    Parameters
    ----------
    model : BaseModel
        PyMC model whose free RVs define the transforms.
    forward : bool
        ``True`` for natural → unconstrained, ``False`` for the inverse.
    """
    inputs = []
    outputs = []
    for rv in model.free_RVs:
        value_var = model.rvs_to_values[rv]
        transform = model.rvs_to_transforms.get(rv, None)
        if forward:
            inp = rv.type()
        else:
            inp = value_var.type()
        inp.name = rv.name + "_in"
        inputs.append(inp)
        if transform is None:
            outputs.append(inp)
        elif forward:
            outputs.append(transform.forward(inp, *rv.owner.inputs))
        else:
            outputs.append(transform.backward(inp, *rv.owner.inputs))
    # For forward, dependent RVs are replaced by their natural-scale inputs.
    # For backward, they are replaced by backward(input) = natural-scale outputs,
    # so dependencies that appear as model RVs resolve to natural-scale values.
    rv_replacements = inputs if forward else outputs
    outputs = replace_vars_in_graphs(
        outputs, dict(zip(model.free_RVs, rv_replacements, strict=True))
    )
    return inputs, outputs


def _eval_transform_graph(
    *,
    forward: bool,
    dataset: xr.Dataset,
    model: BaseModel | None = None,
    sample_dims: Sequence[str] = ("chain", "draw"),
    compile_kwargs: dict | None = None,
):
    model = modelcontext(model)

    if missing := (set(sample_dims) - set(dataset.dims)):  # type: ignore[arg-type]
        raise ValueError(f"sample dims {sorted(missing)} missing from dataset")

    constrained_names = [rv.name for rv in model.free_RVs]
    unconstrained_names = [model.rvs_to_values[rv].name for rv in model.free_RVs]
    if forward:
        input_names = constrained_names
        output_names = unconstrained_names
    else:
        input_names = unconstrained_names
        output_names = constrained_names

    if missing := (set(input_names) - set(dataset.data_vars)):
        raise ValueError(f"Variables {sorted(missing)} missing from dataset")

    inputs, outputs = _build_transform_graph(model, forward=forward)
    replacements = {}
    batch_inputs = []
    for inp in inputs:
        batch_inp = pt.tensor(
            inp.name + "_batch",  # type: ignore[operator]
            shape=(None,) * len(sample_dims) + inp.type.shape,
            dtype=inp.type.dtype,
        )
        replacements[inp] = batch_inp
        batch_inputs.append(batch_inp)
    outputs_vec = xvectorize_graph(outputs, replacements, new_tensor_dims=tuple(sample_dims))
    fn = compile(
        [pytensor.In(bi, borrow=True) for bi in batch_inputs],  # type: ignore[misc]
        [pytensor.Out(ov, borrow=True) for ov in outputs_vec],  # type: ignore[misc]
        trust_input=True,
        **(compile_kwargs or {}),
    )
    results = fn(*(dataset[name].transpose(*sample_dims, ...) for name in input_names))

    out_darrays: dict[str, xr.DataArray] = {}
    for inp_name, out_name, result in zip(input_names, output_names, results, strict=True):
        inp_var = dataset[inp_name]
        inp_core_dims = [d for d in inp_var.dims if d not in sample_dims]
        out_core_shape = result.shape[len(sample_dims) :]
        out_dims = list(sample_dims)
        for i, d in enumerate(inp_core_dims):
            if i < len(out_core_shape) and inp_var.sizes[d] == out_core_shape[i]:
                out_dims.append(d)
            else:
                out_dims.append(f"dim_{d}_{i}")
        while len(out_dims) < result.ndim:
            out_dims.append(f"dim_{len(out_dims)}")
        out_darrays[out_name] = xr.DataArray(result, dims=out_dims)

    return xr.Dataset(
        out_darrays, coords={d: dataset[d] for d in sample_dims if d in dataset.coords}
    )


def unconstrain_values(
    values: xr.Dataset,
    *,
    model: BaseModel | None = None,
    sample_dims: Sequence[str] = ("chain", "draw"),
    compile_kwargs: dict | None = None,
) -> xr.Dataset:
    """Transform a dataset of constrained to unconstrained values.

    Example
    -------
    .. code-block:: python

        import pymc as pm
        import xarray as xr

        with pm.Model() as model:
            sigma = pm.HalfNormal("sigma", 1)
            pm.Normal("y", 0, sigma, observed=[1, 2])

        ds = xr.Dataset({"sigma": xr.DataArray([[0.5, 1.0]], dims=("chain", "draw"))})
        with model:
            # {"sigma": [[0.5, 1.0]]} -> {"sigma_log__": [[-0.69, 0.0]]}
            unconstrain_values(ds)
    """
    return _eval_transform_graph(
        forward=True,
        dataset=values,
        model=model,
        sample_dims=sample_dims,
        compile_kwargs=compile_kwargs,
    )


def constrain_values(
    values: xr.Dataset,
    *,
    model: BaseModel | None = None,
    sample_dims: Sequence[str] = ("chain", "draw"),
    compile_kwargs: dict | None = None,
) -> xr.Dataset:
    """Transform a dataset of unconstrained to constrained values.

    Example
    -------
    .. code-block:: python

        import pymc as pm
        import xarray as xr

        with pm.Model() as model:
            sigma = pm.HalfNormal("sigma", 1)
            pm.Normal("y", 0, sigma, observed=[1, 2])

        ds = xr.Dataset({"sigma_log__": xr.DataArray([[0.0, 1.0]], dims=("chain", "draw"))})
        with model:
            # {"sigma_log__": [[0.0, 1.0]]} -> {"sigma": [[1.0, 2.72]]}
            constrain_values(ds)
    """
    return _eval_transform_graph(
        forward=False,
        dataset=values,
        model=model,
        sample_dims=sample_dims,
        compile_kwargs=compile_kwargs,
    )


__all__ = (
    "constrain_values",
    "unconstrain_values",
)
