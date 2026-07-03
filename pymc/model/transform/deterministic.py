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
from collections.abc import Sequence

from pytensor.graph import Apply, Op, ancestors
from pytensor.graph.basic import Variable
from pytensor.graph.fg import FrozenFunctionGraph, FunctionGraph
from pytensor.graph.rewriting.basic import in2out, node_rewriter

from pymc.model.core import Model
from pymc.model.fgraph import (
    ModelDeterministic,
    ModelVar,
    fgraph_from_model,
    model_from_fgraph,
)


class ModelAnchor(Op):
    """Placeholder that tags a variable by name in a detached Deterministic subgraph.

    It marks the surrounding Model variables a Deterministic depends on, so that
    :func:`insert_deterministics` can splice the subgraph back into a Model by matching
    these names against the target Model variables. Anchors are always removed before the
    Deterministic is reinserted, so they never appear in a final Model graph.
    """

    __props__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def make_node(self, var):
        assert isinstance(var, Variable)
        return Apply(self, [var], [var.type()])

    def perform(self, *args, **kwargs):
        raise RuntimeError("ModelAnchors should never be in a final graph!")


@node_rewriter([ModelAnchor])
def local_remove_anchor(fgraph, node):
    [inp] = node.inputs
    inp.name = node.op.name
    return [inp]


remove_anchors = in2out(local_remove_anchor, ignore_newtrees=True)


def extract_deterministics(
    model: Model, var_names: str | Sequence[str] | None = None
) -> tuple[Model, list[FrozenFunctionGraph]]:
    """Remove Deterministics from a Model, returning them as detached subgraphs.

    The Deterministic computations are inlined into the variables that depend on them,
    so the returned Model is equivalent to the original one but without the Deterministic
    labels. The removed Deterministics are returned as standalone graphs that can later be
    spliced back into a (possibly different) Model with :func:`insert_deterministics`.

    Parameters
    ----------
    model : Model
        The model to extract Deterministics from.
    var_names : str or sequence of str, optional
        The names of the Deterministics to extract. Defaults to all the Deterministics
        in the model.

    Returns
    -------
    new_model : Model
        A copy of the model without the extracted Deterministics.
    deterministics : list of FrozenFunctionGraph
        The extracted Deterministics, as standalone graphs. The order is topological,
        so that Deterministics that depend on other extracted Deterministics come later.

    See Also
    --------
    insert_deterministics : Splice Deterministics back into a Model.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import pymc as pm
        from pymc.model.transform import (
            extract_deterministics,
            insert_deterministics,
        )

        with pm.Model() as model:
            x = pm.Data("x", np.ones((10, 3)))
            beta = pm.Normal("beta", shape=(3,))
            mu = pm.Deterministic("mu", x @ beta)
            pm.Normal("y", mu=mu, sigma=1, observed=np.ones(10))

        # Drop the ``mu`` Deterministic (it gets inlined into ``y``)
        no_det_model, deterministics = extract_deterministics(model)

        # Put it back
        model_again = insert_deterministics(no_det_model, deterministics)

    """
    dets: Sequence[Variable]
    if var_names is None:
        dets = model.deterministics
    else:
        if isinstance(var_names, str):
            var_names = (var_names,)
        dets = [model[name] for name in var_names]
        if any(det not in model.deterministics for det in dets):
            raise ValueError("At least one var is not a Deterministic in the model")

    if not dets:
        return model.copy(), []

    fgraph, memo = fgraph_from_model(model, inlined_views=True)
    memo_dets = [memo[d] for d in dets]

    replacements = []
    deterministics = []
    model_vars: list = []
    for node in fgraph.toposort():
        if not isinstance(node.op, ModelVar):
            continue
        [model_var] = node.outputs
        if isinstance(node.op, ModelDeterministic) and model_var in memo_dets:
            # Inline the Deterministic into its dependents
            replacements.append((model_var, model_var.owner.inputs[0]))
            # Capture the Deterministic subgraph up to the surrounding Model variables
            det_inputs = [a for a in ancestors([model_var], blockers=model_vars) if a in model_vars]
            det_memo: dict = {}
            det_fgraph = FunctionGraph(det_inputs, [model_var], memo=det_memo)
            # Tag the surrounding Model variables by name so the subgraph can be re-attached
            det_fgraph.replace_all(
                # Model variables always have a name, and the single-output Op call
                # returns a Variable (mypy infers the broad Op.__call__ return type).
                [(det_memo[i], ModelAnchor(i.name)(det_memo[i])) for i in det_inputs]  # type: ignore[arg-type, misc]
            )
            deterministics.append(
                FrozenFunctionGraph(inputs=det_fgraph.inputs, outputs=det_fgraph.outputs)
            )
        model_vars.append(model_var)

    fgraph.replace_all(replacements, reason="extract_deterministics")
    return model_from_fgraph(fgraph, mutate_fgraph=True), deterministics


def insert_deterministics(model: Model, deterministics: Sequence[FrozenFunctionGraph]) -> Model:
    """Splice detached Deterministics into a Model.

    This is the inverse of :func:`extract_deterministics`. The Deterministics are attached
    by matching the names of the Model variables they depend on against the variables in
    the target Model.

    Parameters
    ----------
    model : Model
        The model to insert the Deterministics into.
    deterministics : sequence of FrozenFunctionGraph
        The Deterministics to insert, as returned by :func:`extract_deterministics`. They
        must be provided in topological order (Deterministics that depend on other inserted
        Deterministics come later), which is how ``extract_deterministics`` returns them.

    Returns
    -------
    new_model : Model
        A copy of the model with the Deterministics inserted.

    See Also
    --------
    extract_deterministics : Remove Deterministics from a Model as detached subgraphs.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import pymc as pm
        from pymc.model.transform import (
            extract_deterministics,
            insert_deterministics,
        )

        with pm.Model() as model:
            x = pm.Data("x", np.ones((10, 3)))
            beta = pm.Normal("beta", shape=(3,))
            mu = pm.Deterministic("mu", x @ beta)
            pm.Normal("y", mu=mu, sigma=1, observed=np.ones(10))

        # Drop the ``mu`` Deterministic (it gets inlined into ``y``)
        no_det_model, deterministics = extract_deterministics(model)

        # Put it back
        model_again = insert_deterministics(no_det_model, deterministics)

    """
    fgraph, _ = fgraph_from_model(model)
    named_vars = {
        node.outputs[0].name: node.outputs[0]
        for node in fgraph.toposort()
        if isinstance(node.op, ModelVar)
    }
    for det in deterministics:
        anchors = [node for node in det.toposort() if isinstance(node.op, ModelAnchor)]
        [det_live] = det.bind({a.inputs[0]: named_vars[a.op.name] for a in anchors})
        fgraph.add_output(det_live)
        # Expose the inserted Deterministic by name so dependent Deterministics can attach to it
        named_vars[det_live.owner.op.name] = det_live
    remove_anchors.apply(fgraph)
    return model_from_fgraph(fgraph, mutate_fgraph=True)
