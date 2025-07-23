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

from collections.abc import Mapping, Sequence
from typing import Any, Union

import pytensor

from pytensor.graph import Constant, ancestors
from pytensor.tensor import TensorVariable

from pymc.logprob.transforms import Transform
from pymc.model.core import Model
from pymc.model.fgraph import (
    ModelDeterministic,
    ModelFreeRV,
    ModelValuedVar,
    extract_dims,
    fgraph_from_model,
    model_deterministic,
    model_free_rv,
    model_from_fgraph,
    model_named,
    model_observed_rv,
)
from pymc.model.transform.basic import (
    ModelVariable,
    parse_vars,
    prune_vars_detached_from_observed,
)
from pymc.pytensorf import replace_vars_in_graphs, rvs_in_graph, toposort_replace
from pymc.util import get_transformed_name, get_untransformed_name


def observe(
    model: Model, vars_to_observations: Mapping[Union["str", TensorVariable], Any]
) -> Model:
    """Convert free RVs or Deterministics to observed RVs.

    Parameters
    ----------
    model: PyMC Model
    vars_to_observations: Dict of variable or name to TensorLike
        Dictionary that maps model variables (or names) to observed values.
        Observed values must have a shape and data type that is compatible
        with the original model variable.

    Returns
    -------
    new_model: PyMC model
        A distinct PyMC model with the relevant variables observed.
        All remaining variables are cloned and can be retrieved via `new_model["var_name"]`.

    Examples
    --------
    .. code-block:: python

        import pymc as pm

        with pm.Model() as m:
            x = pm.Normal("x")
            y = pm.Normal("y", x)
            z = pm.Normal("z", y)

        m_new = pm.observe(m, {y: 0.5})

    Deterministic variables can also be observed. If the variable has already
    been observed, its old value is replaced with the one provided.

    This relies on PyMC ability to infer the logp of the underlying expression

    .. code-block:: python

        import pymc as pm

        with pm.Model() as m:
            x = pm.Normal("x")
            y = pm.Normal.dist(x, shape=(5,))
            y_censored = pm.Deterministic("y_censored", pm.math.clip(y, -1, 1))

        new_m = pm.observe(m, {y_censored: [0.9, 0.5, 0.3, 1, 1]})


    """
    vars_to_observations = {
        model[var] if isinstance(var, str) else var: obs
        for var, obs in vars_to_observations.items()
    }

    valid_model_vars = set(model.basic_RVs + model.deterministics)
    if any(var not in valid_model_vars for var in vars_to_observations):
        raise ValueError("At least one var is not a random variable or deterministic in the model")

    fgraph, memo = fgraph_from_model(model)

    replacements = {}
    for var, obs in vars_to_observations.items():
        model_var = memo[var]

        # Just a sanity check
        assert isinstance(model_var.owner.op, ModelValuedVar | ModelDeterministic)
        assert model_var in fgraph.variables

        var = model_var.owner.inputs[0]
        var.name = model_var.name
        dims = extract_dims(model_var)
        model_obs_rv = model_observed_rv(var, var.type.filter_variable(obs), *dims)
        replacements[model_var] = model_obs_rv

    toposort_replace(fgraph, tuple(replacements.items()))

    return model_from_fgraph(fgraph, mutate_fgraph=True)


def do(
    model: Model,
    vars_to_interventions: Mapping[Union["str", TensorVariable], Any],
    *,
    make_interventions_shared: bool = True,
    prune_vars: bool = False,
) -> Model:
    """Replace model variables by intervention variables.

    Intervention variables will either show up as `Data` or `Deterministics` in the new model,
    depending on whether they depend on other RandomVariables or not.

    Parameters
    ----------
    model: PyMC Model
    vars_to_interventions: Dict of variable or name to TensorLike
        Dictionary that maps model variables (or names) to intervention expressions.
        Intervention expressions must have a shape and data type that is compatible
        with the original model variable.
    make_interventions_shared: bool, defaults to True,
        Whether to make constant interventions shared variables.
    prune_vars: bool, defaults to False
        Whether to prune model variables that are not connected to any observed variables,
        after the interventions.

    Returns
    -------
    new_model: PyMC model
        A distinct PyMC model with the relevant variables replaced by the intervention expressions.
        All remaining variables are cloned and can be retrieved via `new_model["var_name"]`.

    Examples
    --------
    .. code-block:: python

        import pymc as pm

        with pm.Model() as m:
            x = pm.Normal("x", 0, 1)
            y = pm.Normal("y", x, 1)
            z = pm.Normal("z", y + x, 1)

        # Dummy posterior, same as calling `pm.sample`
        idata_m = az.from_dict({rv.name: [pm.draw(rv, draws=500)] for rv in [x, y, z]})

        # Replace `y` by a constant `100.0`
        with pm.do(m, {y: 100.0}) as m_do:
            idata_do = pm.sample_posterior_predictive(idata_m, var_names="z")

    """
    do_mapping = {}
    for var, intervention in vars_to_interventions.items():
        if isinstance(var, str):
            var = model[var]
        try:
            intervention = var.type.filter_variable(intervention)
            if make_interventions_shared and isinstance(intervention, Constant):
                intervention = pytensor.shared(intervention.data, name=var.name)
            do_mapping[var] = intervention
        except TypeError as err:
            raise TypeError(
                "Incompatible replacement type. Make sure the shape and datatype of the interventions match the original variables"
            ) from err

    if any(var not in model.named_vars.values() for var in do_mapping):
        raise ValueError("At least one var is not a named variable in the model")

    fgraph, memo = fgraph_from_model(model, inlined_views=True)

    # We need the interventions defined in terms of the IR fgraph representation,
    # In case they reference other variables in the model
    ir_interventions = replace_vars_in_graphs(list(do_mapping.values()), replacements=memo)

    replacements = {}
    for var, intervention in zip(do_mapping, ir_interventions):
        model_var = memo[var]

        # Just a sanity check
        assert model_var in fgraph.variables

        # If the intervention references the original variable we must give it a different name
        if model_var in ancestors([intervention]):
            intervention.name = f"do_{model_var.name}"
            warnings.warn(
                f"Intervention expression references the variable that is being intervened: {model_var.name}. "
                f"Intervention will be given the name: {intervention.name}"
            )
        else:
            intervention.name = model_var.name
        dims = extract_dims(model_var)
        # If there are any RVs in the graph we introduce the intervention as a deterministic
        if rvs_in_graph([intervention]):
            new_var = model_deterministic(intervention.copy(name=intervention.name), *dims)
        # Otherwise as a named variable (Constant or Shared data)
        else:
            new_var = model_named(intervention, *dims)

        replacements[model_var] = new_var

    # Replace variables by interventions
    toposort_replace(fgraph, tuple(replacements.items()))

    model = model_from_fgraph(fgraph, mutate_fgraph=True)
    if prune_vars:
        return prune_vars_detached_from_observed(model)
    return model


def change_value_transforms(
    model: Model,
    vars_to_transforms: Mapping[ModelVariable, Transform | None],
) -> Model:
    r"""Change the value variables transforms in the model.

    Parameters
    ----------
    model : Model
    vars_to_transforms : Dict
        Dictionary that maps RVs to new transforms to be applied to the respective value variables

    Returns
    -------
    new_model : Model
        Model with the updated transformed value variables

    Examples
    --------
    Extract untransformed space Hessian after finding transformed space MAP

    .. code-block:: python

        import pymc as pm
        from pymc.distributions.transforms import logodds
        from pymc.model.transform.conditioning import change_value_transforms

        with pm.Model() as base_m:
            p = pm.Uniform("p", 0, 1, default_transform=None)
            w = pm.Binomial("w", n=9, p=p, observed=6)

        with change_value_transforms(base_m, {"p": logodds}) as transformed_p:
            mean_q = pm.find_MAP()

        with change_value_transforms(transformed_p, {"p": None}) as untransformed_p:
            new_p = untransformed_p["p"]
            std_q = ((1 / pm.find_hessian(mean_q, vars=[new_p])) ** 0.5)[0]

        print(f"  Mean, Standard deviation\\np {mean_q['p']:.2}, {std_q[0]:.2}")
        #   Mean, Standard deviation
        # p 0.67, 0.16

    """
    vars_to_transforms = {
        parse_vars(model, var)[0]: transform for var, transform in vars_to_transforms.items()
    }

    if set(vars_to_transforms.keys()) - set(model.free_RVs):
        raise ValueError(f"All keys must be free variables in the model: {model.free_RVs}")

    fgraph, memo = fgraph_from_model(model)

    vars_to_transforms = {memo[var]: transform for var, transform in vars_to_transforms.items()}
    replacements = {}
    for node in fgraph.apply_nodes:
        if not isinstance(node.op, ModelFreeRV):
            continue

        [dummy_rv] = node.outputs
        if dummy_rv not in vars_to_transforms:
            continue

        transform = vars_to_transforms[dummy_rv]

        rv, value, *dims = node.inputs

        new_value = rv.type()
        try:
            untransformed_name = get_untransformed_name(value.name)
        except ValueError:
            untransformed_name = value.name
        if transform:
            new_name = get_transformed_name(untransformed_name, transform)
        else:
            new_name = untransformed_name
        new_value.name = new_name

        new_dummy_rv = model_free_rv(rv, new_value, transform, *dims)
        replacements[dummy_rv] = new_dummy_rv

    toposort_replace(fgraph, tuple(replacements.items()))
    return model_from_fgraph(fgraph, mutate_fgraph=True)


def remove_value_transforms(
    model: Model,
    vars: Sequence[ModelVariable] | None = None,
) -> Model:
    r"""Remove the value variables transforms in the model.

    Parameters
    ----------
    model : Model
    vars : Model variables, optional
        Model variables for which to remove transforms. Defaults to all transformed variables

    Returns
    -------
    new_model : Model
        Model with the removed transformed value variables

    Examples
    --------
    Extract untransformed space Hessian after finding transformed space MAP

    .. code-block:: python

        import pymc as pm
        from pymc.model.transform.conditioning import remove_value_transforms

        with pm.Model() as transformed_m:
            p = pm.Uniform("p", 0, 1)
            w = pm.Binomial("w", n=9, p=p, observed=6)
            mean_q = pm.find_MAP()

        with remove_value_transforms(transformed_m) as untransformed_m:
            new_p = untransformed_m["p"]
            std_q = ((1 / pm.find_hessian(mean_q, vars=[new_p])) ** 0.5)[0]
            print(f"  Mean, Standard deviation\\np {mean_q['p']:.2}, {std_q[0]:.2}")

        #   Mean, Standard deviation
        # p 0.67, 0.16

    """
    if vars is None:
        vars = model.free_RVs
    return change_value_transforms(model, dict.fromkeys(vars))


__all__ = (
    "change_value_transforms",
    "do",
    "observe",
    "remove_value_transforms",
)
