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
from copy import copy
from typing import Dict, Optional, Tuple

import pytensor

from pytensor import Variable, shared
from pytensor.compile import SharedVariable
from pytensor.graph import Apply, FunctionGraph, Op, node_rewriter
from pytensor.graph.rewriting.basic import out2in
from pytensor.scalar import Identity
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.sharedvar import ScalarSharedVariable

from pymc.logprob.transforms import RVTransform
from pymc.model.core import Model
from pymc.pytensorf import StringType, find_rng_nodes, toposort_replace


class ModelVar(Op):
    """A dummy Op that describes the purpose of a Model variable and contains
    meta-information as additional inputs (value and dims).
    """

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

    def __init__(self, transform: Optional[RVTransform] = None):
        if transform is not None and not isinstance(transform, RVTransform):
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


def fgraph_from_model(
    model: Model, inlined_views=False
) -> Tuple[FunctionGraph, Dict[Variable, Variable]]:
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

    # Collect PyTensor variables
    rvs_to_values = model.rvs_to_values
    rvs = list(rvs_to_values.keys())
    free_rvs = model.free_RVs
    observed_rvs = model.observed_RVs
    potentials = model.potentials
    named_vars = model.named_vars.values()
    # We copy Deterministics (Identity Op) so that they don't show in between "main" variables
    # We later remove these Identity Ops when we have a Deterministic ModelVar Op as a separator
    old_deterministics = model.deterministics
    deterministics = [det if inlined_views else det.copy(det.name) for det in old_deterministics]
    # Value variables (we also have to decide whether to inline named ones)
    old_value_vars = list(rvs_to_values.values())
    unnamed_value_vars = [val for val in old_value_vars if val not in named_vars]
    named_value_vars = [
        val if inlined_views else val.copy(val.name) for val in old_value_vars if val in named_vars
    ]
    value_vars = old_value_vars.copy()
    if inlined_views:
        # In this case we want to use the named_value_vars as the value_vars in RVs
        for named_val in named_value_vars:
            idx = value_vars.index(named_val)
            value_vars[idx] = named_val
    # Other variables that are in named_vars but are not any of the categories above
    # E.g., MutableData, ConstantData, _dim_lengths
    # We use the same trick as deterministics!
    accounted_for = set(free_rvs + observed_rvs + potentials + old_deterministics + old_value_vars)
    other_named_vars = [
        var if inlined_views else var.copy(var.name)
        for var in named_vars
        if var not in accounted_for
    ]

    model_vars = (
        rvs + potentials + deterministics + other_named_vars + named_value_vars + unnamed_value_vars
    )

    memo = {}

    # Replace the following shared variables in the model:
    # 1. RNGs
    # 2. MutableData (could increase memory usage significantly)
    # 3. Mutable coords dim lengths
    shared_vars_to_copy = find_rng_nodes(model_vars)
    shared_vars_to_copy += [v for v in model.dim_lengths.values() if isinstance(v, SharedVariable)]
    shared_vars_to_copy += [v for v in model.named_vars.values() if isinstance(v, SharedVariable)]
    for var in shared_vars_to_copy:
        # FIXME: ScalarSharedVariables are converted to 0d numpy arrays internally,
        #  so calling shared(shared(5).get_value()) returns a different type: TensorSharedVariables!
        #  Furthermore, PyMC silently ignores mutable dim changes that are SharedTensorVariables...
        #  https://github.com/pymc-devs/pytensor/issues/396
        if isinstance(var, ScalarSharedVariable):
            new_var = shared(var.get_value(borrow=False).item())
        else:
            new_var = shared(var.get_value(borrow=False))

        assert new_var.type == var.type
        new_var.name = var.name
        new_var.tag = copy(var.tag)
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
            model_var.owner and isinstance(model_var.owner.op, (ModelDeterministic, ModelNamed))
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


def model_from_fgraph(fgraph: FunctionGraph) -> Model:
    """Convert FunctionGraph to PyMC model.

    This requires nodes to be properly tagged with `ModelVar` dummy Ops.

    See: fgraph_from_model
    """

    def first_non_model_var(var):
        if var.owner and isinstance(var.owner.op, ModelVar):
            new_var = var.owner.inputs[0]
            return first_non_model_var(new_var)
        else:
            return var

    model = Model()
    if model.parent is not None:
        raise RuntimeError("model_to_fgraph cannot be called inside a PyMC model context")
    model._coords = getattr(fgraph, "_coords", {})
    model._dim_lengths = getattr(fgraph, "_dim_lengths", {})

    # Replace dummy `ModelVar` Ops by the underlying variables,
    fgraph = fgraph.clone()
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
            # PyMC does not allow setting transform when we pass a value_var. Why?
            model.create_value_var(var, transform=None, value_var=value)
            model.rvs_to_transforms[var] = transform
            model.set_initval(var, initval=None)
        elif isinstance(model_var.owner.op, ModelObservedRV):
            var, value, *dims = model_var.owner.inputs
            model.observed_RVs.append(var)
            model.create_value_var(var, transform=None, value_var=value)
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
    return model_from_fgraph(fgraph_from_model(model)[0])


def extract_dims(var) -> Tuple:
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
