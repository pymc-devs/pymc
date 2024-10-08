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
import warnings

from copy import copy, deepcopy

import pytensor

from pytensor import Variable
from pytensor.compile import SharedVariable
from pytensor.graph import Apply, FunctionGraph, Op, node_rewriter
from pytensor.graph.rewriting.basic import out2in
from pytensor.scalar import Identity
from pytensor.tensor.elemwise import Elemwise

from pymc.logprob.transforms import Transform
from pymc.model.core import Model
from pymc.pytensorf import StringType, find_rng_nodes, toposort_replace


class ModelVar(Op):
    """A dummy Op that describes the purpose of a Model variable and contains meta-information as additional inputs (value and dims)."""

    def make_node(self, rv, *dims):
        assert isinstance(rv, Variable)
        dims = self._parse_dims(rv, *dims)
        return Apply(self, [rv, *dims], [rv.type(name=rv.name)])

    def _parse_dims(self, rv, *dims):
        if dims:
            dims = [pytensor.as_symbolic(dim) for dim in dims]
            assert all(isinstance(dim.type, StringType) for dim in dims)
            assert len(dims) == rv.type.ndim
        return dims

    def infer_shape(self, fgraph, node, inputs_shape):
        return [inputs_shape[0]]

    def do_constant_folding(self, fgraph, node):
        return False

    def perform(self, *args, **kwargs):
        raise RuntimeError("ModelVars should never be in a final graph!")


class ModelValuedVar(ModelVar):
    __props__ = ("transform",)

    def __init__(self, transform: Transform | None = None):
        if transform is not None and not isinstance(transform, Transform):
            raise TypeError(f"transform must be None or RVTransform type, got {type(transform)}")
        self.transform = transform
        super().__init__()

    def make_node(self, rv, value, *dims):
        assert isinstance(rv, Variable)
        dims = self._parse_dims(rv, *dims)
        if value is not None:
            assert isinstance(value, Variable)
            assert rv.type.dtype == value.type.dtype
            return Apply(self, [rv, value, *dims], [rv.type(name=rv.name)])


class ModelFreeRV(ModelValuedVar):
    pass


class ModelObservedRV(ModelValuedVar):
    pass


class ModelPotential(ModelVar):
    pass


class ModelDeterministic(ModelVar):
    pass


class ModelNamed(ModelVar):
    pass


def model_free_rv(rv, value, transform, *dims):
    return ModelFreeRV(transform=transform)(rv, value, *dims)


model_observed_rv = ModelObservedRV()
model_potential = ModelPotential()
model_deterministic = ModelDeterministic()
model_named = ModelNamed()


@node_rewriter([Elemwise])
def local_remove_identity(fgraph, node):
    if isinstance(node.op.scalar_op, Identity):
        return [node.inputs[0]]


remove_identity_rewrite = out2in(local_remove_identity)


def deepcopy_shared_variable(var: SharedVariable) -> SharedVariable:
    # Shared variables don't have a deepcopy method (SharedVariable.clone reuses the old container and contents).
    # We recreate Shared Variables manually after deepcopying their container.
    new_var = type(var)(
        type=var.type,
        value=None,
        strict=None,
        container=deepcopy(var.container),
        name=var.name,
    )
    assert new_var.type == var.type
    new_var.tag = copy(var.tag)
    return new_var


def fgraph_from_model(
    model: Model, inlined_views=False
) -> tuple[FunctionGraph, dict[Variable, Variable]]:
    """Convert Model to FunctionGraph.

    See: model_from_fgraph

    Parameters
    ----------
    model: PyMC model
    inlined_views: bool, default False
        Whether "view" variables (Deterministics and Data) should be inlined among RVs in the fgraph,
        or show up as separate branches.

    Returns
    -------
    fgraph: FunctionGraph
        FunctionGraph that includes a copy of model variables, wrapped in dummy `ModelVar` Ops.
        It should be possible to reconstruct a valid PyMC model using `model_from_fgraph`.

    memo: Dict
        A dictionary mapping original model variables to the equivalent nodes in the fgraph.
    """
    if any(v is not None for v in model.rvs_to_initial_values.values()):
        raise NotImplementedError("Cannot convert models with non-default initial_values")

    if model.parent is not None:
        raise ValueError(
            "Nested sub-models cannot be converted to fgraph. Convert the parent model instead"
        )

    if any(
        ("_rotated_" in var_name or "_hsgp_coeffs_" in var_name) for var_name in model.named_vars
    ):
        warnings.warn(
            "Detected variables likely created by GP objects. Further use of these old GP objects should be avoided as it may reintroduce variables from the old model. See issue: https://github.com/pymc-devs/pymc/issues/6883",
            UserWarning,
        )

    # Collect PyTensor variables
    rvs_to_values = model.rvs_to_values
    rvs = list(rvs_to_values.keys())
    free_rvs = model.free_RVs
    observed_rvs = model.observed_RVs
    potentials = model.potentials
    # We copy Deterministics (Identity Op) so that they don't show in between "main" variables
    # We later remove these Identity Ops when we have a Deterministic ModelVar Op as a separator
    old_deterministics = model.deterministics
    deterministics = [det if inlined_views else det.copy(det.name) for det in old_deterministics]
    # Value variables (we also have to decide whether to inline named ones)
    old_value_vars = list(rvs_to_values.values())
    data_vars = model.data_vars
    unnamed_value_vars = [val for val in old_value_vars if val not in data_vars]
    named_value_vars = [
        val if inlined_views else val.copy(name=val.name)
        for val in old_value_vars
        if val in data_vars
    ]
    value_vars = old_value_vars.copy()
    if inlined_views:
        # In this case we want to use the named_value_vars as the value_vars in RVs
        for named_val in named_value_vars:
            idx = value_vars.index(named_val)
            value_vars[idx] = named_val
    # Data vars that are not value vars
    other_named_vars = [
        var if inlined_views else var.copy(var.name)
        for var in data_vars
        if var not in old_value_vars
    ]

    model_vars = (
        rvs + potentials + deterministics + other_named_vars + named_value_vars + unnamed_value_vars
    )

    memo = {}

    # Replace the following shared variables in the model:
    # 1. RNGs
    # 2. Data (could increase memory usage significantly)
    # 3. Symbolic coords dim lengths
    shared_vars_to_copy = find_rng_nodes(model_vars)
    shared_vars_to_copy += [v for v in model.dim_lengths.values() if isinstance(v, SharedVariable)]
    shared_vars_to_copy += [v for v in model.named_vars.values() if isinstance(v, SharedVariable)]
    for var in shared_vars_to_copy:
        new_var = deepcopy_shared_variable(var)
        # We can replace input variables by placing them in the memo
        memo[var] = new_var

    fgraph = FunctionGraph(
        outputs=model_vars,
        clone=True,
        memo=memo,
        copy_orphans=True,
        copy_inputs=True,
    )
    # Copy model meta-info to fgraph
    fgraph._coords = model._coords.copy()
    fgraph._dim_lengths = {k: memo.get(v, v) for k, v in model._dim_lengths.items()}

    rvs_to_transforms = model.rvs_to_transforms
    named_vars_to_dims = model.named_vars_to_dims

    # Introduce dummy `ModelVar` Ops
    free_rvs_to_transforms = {memo[k]: tr for k, tr in rvs_to_transforms.items()}
    free_rvs_to_values = {memo[k]: memo[v] for k, v in zip(rvs, value_vars) if k in free_rvs}
    observed_rvs_to_values = {
        memo[k]: memo[v] for k, v in zip(rvs, value_vars) if k in observed_rvs
    }
    potentials = [memo[k] for k in potentials]
    deterministics = [memo[k] for k in deterministics]
    named_vars = [memo[k] for k in other_named_vars + named_value_vars]

    vars = fgraph.outputs
    new_vars = []
    for var in vars:
        dims = named_vars_to_dims.get(var.name, ())
        if var in free_rvs_to_values:
            new_var = model_free_rv(
                var, free_rvs_to_values[var], free_rvs_to_transforms[var], *dims
            )
        elif var in observed_rvs_to_values:
            new_var = model_observed_rv(var, observed_rvs_to_values[var], *dims)
        elif var in potentials:
            new_var = model_potential(var, *dims)
        elif var in deterministics:
            new_var = model_deterministic(var, *dims)
        elif var in named_vars:
            new_var = model_named(var, *dims)
        else:
            # Unnamed value variables
            new_var = var
        new_vars.append(new_var)

    replacements = tuple(zip(vars, new_vars))
    toposort_replace(fgraph, replacements, reverse=True)

    # Reference model vars in memo
    inverse_memo = {v: k for k, v in memo.items()}
    for var, model_var in replacements:
        if not inlined_views and (
            model_var.owner and isinstance(model_var.owner.op, ModelDeterministic | ModelNamed)
        ):
            # Ignore extra identity that will be removed at the end
            var = var.owner.inputs[0]
        original_var = inverse_memo[var]
        memo[original_var] = model_var

    # Remove the last outputs corresponding to unnamed value variables, now that they are graph inputs
    first_idx_to_remove = len(fgraph.outputs) - len(unnamed_value_vars)
    for _ in unnamed_value_vars:
        fgraph.remove_output(first_idx_to_remove)

    # Now that we have Deterministic dummy Ops, we remove the noisy `Identity`s from the graph
    remove_identity_rewrite.apply(fgraph)

    return fgraph, memo


def model_from_fgraph(fgraph: FunctionGraph, mutate_fgraph: bool = False) -> Model:
    """Convert FunctionGraph to PyMC model.

    Parameters
    ----------
    fgraph: FunctionGraph
        fgraph representation of a PyMC model, with dummy `ModelVar` Ops.
        See `fgraph_from_model` for more details.

    mutate_fgraph: bool, default False
        Whether the function is allowed to modify the fgraph (and it's variables) in place.
         This is useful if these are not needed anymore after the model is created.
    """

    def first_non_model_var(var):
        if var.owner and isinstance(var.owner.op, ModelVar):
            new_var = var.owner.inputs[0]
            return first_non_model_var(new_var)
        else:
            return var

    model = Model(model=None)  # Do not inherit from any model in the context manager

    _coords = getattr(fgraph, "_coords", {})
    _dim_lengths = getattr(fgraph, "_dim_lengths", {})

    if not mutate_fgraph:
        fgraph, memo = fgraph.clone_get_equiv(check_integrity=False, attach_feature=False)
        # Shared dim lengths are not extracted from the fgraph representation,
        # so we need to update after we clone the fgraph
        # TODO: Consider representing/extracting them from the fgraph!
        _dim_lengths = {k: memo.get(v, v) for k, v in _dim_lengths.items()}

    model._coords = _coords
    model._dim_lengths = _dim_lengths

    # Replace dummy `ModelVar` Ops by the underlying variables,
    model_dummy_vars = [
        model_node.outputs[0]
        for model_node in fgraph.toposort()
        if isinstance(model_node.op, ModelVar)
    ]
    model_dummy_vars_to_vars = {
        # Deterministics could refer to other model variables directly,
        # We make sure to replace them by the first non-model variable
        dummy_var: first_non_model_var(dummy_var.owner.inputs[0])
        for dummy_var in model_dummy_vars
    }
    toposort_replace(fgraph, tuple(model_dummy_vars_to_vars.items()))

    # Populate new PyMC model mappings
    for model_var in model_dummy_vars:
        if isinstance(model_var.owner.op, ModelFreeRV):
            var, value, *dims = model_var.owner.inputs
            transform = model_var.owner.op.transform
            model.free_RVs.append(var)
            model.create_value_var(
                var, transform=transform, default_transform=None, value_var=value
            )
            model.set_initval(var, initval=None)
        elif isinstance(model_var.owner.op, ModelObservedRV):
            var, value, *dims = model_var.owner.inputs
            model.observed_RVs.append(var)
            model.create_value_var(var, transform=None, default_transform=None, value_var=value)
        elif isinstance(model_var.owner.op, ModelPotential):
            var, *dims = model_var.owner.inputs
            model.potentials.append(var)
        elif isinstance(model_var.owner.op, ModelDeterministic):
            var, *dims = model_var.owner.inputs
            # If a Deterministic is a direct view on an RV, copy it
            if var in model.basic_RVs:
                var = var.copy()
            model.deterministics.append(var)
        elif isinstance(model_var.owner.op, ModelNamed):
            var, *dims = model_var.owner.inputs
            model.data_vars.append(var)
        else:
            raise TypeError(f"Unexpected ModelVar type {type(model_var)}")

        var.name = model_var.name
        dims = [dim.data for dim in dims] if dims else None
        model.add_named_variable(var, dims=dims)

    return model


def clone_model(model: Model) -> Model:
    """Clone a PyMC model.

    Recreates a PyMC model with clones of the original variables.
    Shared variables will point to the same container but be otherwise different objects.
    Constants are not cloned.


    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc.model.fgraph import clone_model

        with pm.Model() as m:
            p = pm.Beta("p", 1, 1)
            x = pm.Bernoulli("x", p=p, shape=(3,))

        with clone_model(m) as clone_m:
            # Access cloned variables by name
            clone_x = clone_m["x"]

            # z will be part of clone_m but not m
            z = pm.Deterministic("z", clone_x + 1)

    """
    return model_from_fgraph(fgraph_from_model(model)[0], mutate_fgraph=True)


def extract_dims(var) -> tuple:
    dims = ()
    node = var.owner
    if node and isinstance(node.op, ModelVar):
        if isinstance(node.op, ModelValuedVar):
            dims = node.inputs[2:]
        else:
            dims = node.inputs[1:]
    return dims


__all__ = (
    "fgraph_from_model",
    "model_from_fgraph",
    "clone_model",
)
