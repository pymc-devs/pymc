#   Copyright 2025 - present The PyMC Developers
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

from pytensor import Variable
from pytensor.graph import FunctionGraph, ancestors

from build.lib.pymc.variational.minibatch_rv import MinibatchRandomVariable
from pymc import Minibatch, Model
from pymc.data import MinibatchOp
from pymc.model.fgraph import ModelObservedRV, fgraph_from_model, model_from_fgraph
from pymc.model.transform.basic import parse_vars
from pymc.pytensorf import toposort_replace


def minibatch_model(
    model: Model,
    *,
    batch_size: int,
    minibatch_vars: Sequence[str | Variable] | None = None,
) -> Model:
    """Create a minibatch version of the given Model.

    Replaces minibatch_vars data containers with Minibatch views and rescales the logp of dependent observed variables.

    .. warning:: This transformation acts on the leading dimension of the specified data variables and dependent observed RVs. If a dimension other than the first is linked to the minibatched data variables, the resulting model will be invalid.

    Parameters
    ----------
    model : Model
        The original model to transform.
    batch_size : int
        The minibatch size to use.
    minibatch_vars : Sequence of Variable or string, optional
        Data variables to convert to minibatch. If None, all data variables with a leading dimension of size None will be minibatched.

    Returns
    -------
    Model
        A new Model with the specified data variables replaced by Minibatch views and dependent observed RVs adjusted accordingly.

    Raises
    ------
    ValueError
        If any of the specified variables cannot be minibatched (e.g., scalar variables or variables with static leading dimensions), or if dependent variables are Potentials / Unobserved RVs.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import pymc as pm
        from pymc.model.transform.minibatch import minibatch_model

        with pm.Model() as m:
            obs_data = pm.Data("obs_data", np.random.normal(size=(100,)))
            X_data = pm.Data("X_data", np.random.normal(size=(100, 4)))
            beta = pm.Normal("beta", mu=np.pi, dims="feature")

            mu = X_data @ beta
            y = pm.Normal("y", mu=mu, sigma=1, observed=obs_data)

        with minibatch_model(m, batch_size=10) as mb:
            pm.fit()
    """
    from pymc.variational.minibatch_rv import create_minibatch_rv

    if minibatch_vars is None:
        original_minibatch_vars = [
            variable
            for variable in model.data_vars
            if (variable.type.ndim > 0) and (variable.type.shape[0] is None)
        ]
    else:
        original_minibatch_vars = parse_vars(model, minibatch_vars)
        for variable in original_minibatch_vars:
            if variable.type.ndim == 0:
                raise ValueError(
                    f"Cannot minibatch {variable.name} because it is a scalar variable."
                )
            if variable.type.shape[0] is not None:
                raise ValueError(
                    f"Cannot minibatch {variable.name} because its first dimension is static "
                    f"(size={variable.type.shape[0]})."
                )

    # TODO: Validate that this graph is actually valid to minibatch. Example: linear regression with sigma fixed
    #  shape, but mu from data --> y cannot be minibatched because of sigma.

    fgraph, memo = fgraph_from_model(model, inlined_views=True)

    pre_minibatch_vars = [memo[var] for var in original_minibatch_vars]
    minibatch_vars = Minibatch(*pre_minibatch_vars, batch_size=batch_size)

    # Replace uses of the specified data variables with Minibatch variables
    # We need a two-step clone because FunctionGraph can only mutate one variable at a time
    # and when there are multiple vars to minibatch you end up replacing the same variable twice recursively
    # exampre: out = x + y
    # goal: replace (x, y) by (Minibatch(x, y).0, Minibatch(x, y).1)]
    # replace x first we get: out = Minibatch(x, y).0 + y
    # then replace y we get: out = Minibatch(x, Minibatch(...).1).0 + Minibatch(x, y).1
    # The second replacement of y ends up creating a circular dependency
    pre_minibatch_var_to_dummy = tuple((var, var.type()) for var in pre_minibatch_vars)
    dummy_to_minibatch_var = tuple(
        (dummy, minibatch_var)
        for (_, dummy), minibatch_var in zip(pre_minibatch_var_to_dummy, minibatch_vars)
    )

    # Furthermore, we only want to replace uses of the data variables (x, y), but not the data variables themselves,
    # So we use an intermediate FunctionGraph that doesn't contain the data variables as outputs
    other_model_vars = [out for out in fgraph.outputs if out not in pre_minibatch_vars]
    minibatch_fgraph = FunctionGraph(outputs=other_model_vars, clone=False)
    minibatch_fgraph._coords = fgraph._coords  # type: ignore[attr-defined]
    minibatch_fgraph._dim_lengths = fgraph._dim_lengths  # type: ignore[attr-defined]
    toposort_replace(minibatch_fgraph, pre_minibatch_var_to_dummy)
    toposort_replace(minibatch_fgraph, dummy_to_minibatch_var)

    # Then replace all observed RVs that depend on the minibatch variables with MinibatchRVs
    dependent_replacements = {}
    total_size = (pre_minibatch_vars[0].owner.inputs[0].shape[0], ...)
    vars_to_minibatch_set = set(pre_minibatch_vars)
    for model_var in minibatch_fgraph.outputs:
        if not (set(ancestors([model_var])) & vars_to_minibatch_set):
            continue
        if not isinstance(model_var.owner.op, ModelObservedRV):
            raise ValueError(
                "Minibatching only supports observed RVs depending on minibatched variables. "
                f"Found dependent unobserved variable: {model_var.name}."
            )
        # TODO: If vars_to_minibatch had a leading dim, we should check that the dependent RVs also has that same dim
        # And conversely other variables do not have that dim
        observed_rv = model_var.owner.inputs[0]
        dependent_replacements[observed_rv] = create_minibatch_rv(
            observed_rv, total_size=total_size
        )

    toposort_replace(minibatch_fgraph, tuple(dependent_replacements.items()))

    # Finally reintroduce the original data variable outputs
    for pre_minibatch_var in pre_minibatch_vars:
        minibatch_fgraph.add_output(pre_minibatch_var)

    return model_from_fgraph(minibatch_fgraph, mutate_fgraph=True)


def remove_minibatch(model: Model) -> Model:
    """Remove all uses of Minibatch data and random variables from the Model.

    Parameters
    ----------
    model : Model
        The original model to transform.

    Returns
    -------
    Model
        A new Model with all Minibatch data variables and MinibatchRVs replaced by their original counterparts.

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc.model.transform.minibatch import undo_minibatch

        with pm.Model() as mb:
            X_data = pm.Data("X_data", [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
            obs_data = pm.Data("obs_data", [1, 2, 3, 4, 5])
            minibatch_X_data, minibatch_obs_data = pm.Minibatch(X_data, obs_data, batch_size=3)

            beta = pm.Normal("beta", shape=(2,))
            mu = minibatch_X_data @ beta
            y = pm.Normal("y", mu=mu, sigma=1, observed=minibatch_obs_data, total_size=(5,))

        with undo_minibatch(mb) as m:
            idata = pm.sample_prior_predictive()
            assert idata.prior["y"].shape[-1] == 5  # Original data size restored

    """
    fgraph, _ = fgraph_from_model(model)

    replacements = []
    for var in fgraph.apply_nodes:
        if isinstance(var.op, MinibatchOp):
            replacements.extend(zip(var.inputs, var.outputs))
        elif isinstance(var.op, MinibatchRandomVariable):
            replacements.append((var.outputs[0], var.inputs[0]))

    toposort_replace(fgraph, replacements)
    return model_from_fgraph(fgraph, mutate_fgraph=True)
