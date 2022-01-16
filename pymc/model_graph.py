#   Copyright 2020 The PyMC Developers
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

from collections import defaultdict, deque
from typing import Dict, Iterator, NewType, Optional, Set

from aesara import function
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import walk
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorConstant, TensorVariable

import pymc as pm

from pymc.util import get_default_varnames, get_var_name

VarName = NewType("VarName", str)


class ModelGraph:
    def __init__(self, model):
        self.model = model
        self.var_names = get_default_varnames(self.model.named_vars, include_transformed=False)
        self.var_list = self.model.named_vars.values()
        self.transform_map = {
            v.transformed: v.name for v in self.var_list if hasattr(v, "transformed")
        }
        self._deterministics = None

    def get_deterministics(self, var):
        """Compute the deterministic nodes of the graph, **not** including var itself."""
        deterministics = []
        attrs = ("transformed", "logpt")
        for v in self.var_list:
            if v != var and all(not hasattr(v, attr) for attr in attrs):
                deterministics.append(v)
        return deterministics

    def _get_ancestors(self, var: TensorVariable, func) -> Set[TensorVariable]:
        """Get all ancestors of a function, doing some accounting for deterministics."""

        # this contains all of the variables in the model EXCEPT var...
        vars = set(self.var_list)
        vars.remove(var)

        blockers = set()  # type: Set[TensorVariable]
        retval = set()  # type: Set[TensorVariable]

        def _expand(node) -> Optional[Iterator[TensorVariable]]:
            if node in blockers:
                return None
            elif node in vars:
                blockers.add(node)
                retval.add(node)
                return None
            elif node.owner:
                blockers.add(node)
                return reversed(node.owner.inputs)
            else:
                return None

        list(walk(deque([func]), _expand, bfs=True))
        return retval

    def _filter_parents(self, var, parents) -> Set[VarName]:
        """Get direct parents of a var, as strings"""
        keep = set()  # type: Set[VarName]
        for p in parents:
            if p == var:
                continue
            elif p.name in self.var_names:
                keep.add(p.name)
            elif p in self.transform_map:
                if self.transform_map[p] != var.name:
                    keep.add(self.transform_map[p])
            else:
                raise AssertionError(f"Do not know what to do with {get_var_name(p)}")
        return keep

    def get_parents(self, var: TensorVariable) -> Set[VarName]:
        """Get the named nodes that are direct inputs to the var"""
        # TODO: Update these lines, variables no longer have a `logpt` attribute
        if hasattr(var, "transformed"):
            func = var.transformed.logpt
        elif hasattr(var, "logpt"):
            func = var.logpt
        else:
            func = var

        parents = self._get_ancestors(var, func)
        return self._filter_parents(var, parents)

    def make_compute_graph(self) -> Dict[str, Set[VarName]]:
        """Get map of var_name -> set(input var names) for the model"""
        input_map = defaultdict(set)  # type: Dict[str, Set[VarName]]

        for var_name in self.var_names:
            var = self.model[var_name]
            key = var_name
            val = self.get_parents(var)
            input_map[key] = input_map[key].union(val)

            if hasattr(var.tag, "observations"):
                try:
                    obs_name = var.tag.observations.name
                    if obs_name:
                        input_map[var_name] = input_map[var_name].difference({obs_name})
                        input_map[obs_name] = input_map[obs_name].union({var_name})
                except AttributeError:
                    pass
        return input_map

    def _make_node(self, var_name, graph, *, formatting: str = "plain"):
        """Attaches the given variable to a graphviz Digraph"""
        v = self.model[var_name]

        shape = None
        style = None
        label = str(v)

        if v in self.model.potentials:
            shape = "octagon"
            style = "filled"
            label = f"{var_name}\n~\nPotential"
        elif isinstance(v, TensorConstant):
            shape = "box"
            style = "rounded, filled"
            label = f"{var_name}\n~\nConstantData"
        elif isinstance(v, SharedVariable):
            shape = "box"
            style = "rounded, filled"
            label = f"{var_name}\n~\nMutableData"
        elif v.owner and isinstance(v.owner.op, RandomVariable):
            shape = "ellipse"
            if hasattr(v.tag, "observations"):
                # observed RV
                style = "filled"
            else:
                shape = "ellipse"
                syle = None
            symbol = v.owner.op.__class__.__name__.strip("RV")
            label = f"{var_name}\n~\n{symbol}"
        else:
            shape = "box"
            style = None
            label = f"{var_name}\n~\nDeterministic"

        kwargs = {
            "shape": shape,
            "style": style,
            "label": label,
        }

        graph.node(var_name.replace(":", "&"), **kwargs)

    def _eval(self, var):
        return function([], var, mode="FAST_COMPILE")()

    def get_plates(self):
        """Rough but surprisingly accurate plate detection.

        Just groups by the shape of the underlying distribution.  Will be wrong
        if there are two plates with the same shape.

        Returns
        -------
        dict: str -> set[str]
        """
        plates = defaultdict(set)
        for var_name in self.var_names:
            v = self.model[var_name]
            if var_name in self.model.RV_dims:
                plate_label = " x ".join(
                    f"{d} ({self._eval(self.model.dim_lengths[d])})"
                    for d in self.model.RV_dims[var_name]
                )
            else:
                plate_label = " x ".join(map(str, self._eval(v.shape)))
            plates[plate_label].add(var_name)
        return plates

    def make_graph(self, formatting: str = "plain"):
        """Make graphviz Digraph of PyMC model

        Returns
        -------
        graphviz.Digraph
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "This function requires the python library graphviz, along with binaries. "
                "The easiest way to install all of this is by running\n\n"
                "\tconda install -c conda-forge python-graphviz"
            )
        graph = graphviz.Digraph(self.model.name)
        for plate_label, var_names in self.get_plates().items():
            if plate_label:
                # must be preceded by 'cluster' to get a box around it
                with graph.subgraph(name="cluster" + plate_label) as sub:
                    for var_name in var_names:
                        self._make_node(var_name, sub, formatting=formatting)
                    # plate label goes bottom right
                    sub.attr(label=plate_label, labeljust="r", labelloc="b", style="rounded")
            else:
                for var_name in var_names:
                    self._make_node(var_name, graph, formatting=formatting)

        for key, values in self.make_compute_graph().items():
            for value in values:
                graph.edge(value.replace(":", "&"), key.replace(":", "&"))
        return graph


def model_to_graphviz(model=None, *, formatting: str = "plain"):
    """Produce a graphviz Digraph from a PyMC model.

    Requires graphviz, which may be installed most easily with
        conda install -c conda-forge python-graphviz

    Alternatively, you may install the `graphviz` binaries yourself,
    and then `pip install graphviz` to get the python bindings.  See
    http://graphviz.readthedocs.io/en/stable/manual.html
    for more information.

    Parameters
    ----------
    model : pm.Model
        The model to plot. Not required when called from inside a modelcontext.
    formatting : str
        one of { "plain" }

    Examples
    --------
    How to plot the graph of the model.

    .. code-block:: python

        import numpy as np
        from pymc import HalfCauchy, Model, Normal, model_to_graphviz

        J = 8
        y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
        sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

        with Model() as schools:

            eta = Normal("eta", 0, 1, shape=J)
            mu = Normal("mu", 0, sigma=1e6)
            tau = HalfCauchy("tau", 25)

            theta = mu + tau * eta

            obs = Normal("obs", theta, sigma=sigma, observed=y)

        model_to_graphviz(schools)
    """
    if not "plain" in formatting:
        raise ValueError(f"Unsupported formatting for graph nodes: '{formatting}'. See docstring.")
    if formatting != "plain":
        warnings.warn(
            "Formattings other than 'plain' are currently not supported.", UserWarning, stacklevel=2
        )
    model = pm.modelcontext(model)
    return ModelGraph(model).make_graph(formatting=formatting)
