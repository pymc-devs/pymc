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

from collections import deque
from typing import Dict, Iterator, Optional, Set

VarName = str

from theano.compile import SharedVariable
from theano.graph.basic import walk
from theano.tensor import Tensor

import pymc3 as pm

from pymc3.model import ObservedRV
from pymc3.util import get_default_varnames, get_var_name


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

    def _get_ancestors(self, var: Tensor, func) -> Set[Tensor]:
        """Get all ancestors of a function, doing some accounting for deterministics."""

        # this contains all of the variables in the model EXCEPT var...
        vars = set(self.var_list)
        vars.remove(var)

        blockers = set()  # type: Set[Tensor]
        retval = set()  # type: Set[Tensor]

        def _expand(node) -> Optional[Iterator[Tensor]]:
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
                raise AssertionError("Do not know what to do with {}".format(get_var_name(p)))
        return keep

    def get_parents(self, var: Tensor) -> Set[VarName]:
        """Get the named nodes that are direct inputs to the var"""
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
        input_map = {}  # type: Dict[str, Set[VarName]]

        def update_input_map(key: str, val: Set[VarName]):
            if key in input_map:
                input_map[key] = input_map[key].union(val)
            else:
                input_map[key] = val

        for var_name in self.var_names:
            var = self.model[var_name]
            update_input_map(var_name, self.get_parents(var))
            if isinstance(var, ObservedRV):
                try:
                    obs_name = var.observations.name
                    if obs_name:
                        input_map[var_name] = input_map[var_name].difference({obs_name})
                        update_input_map(obs_name, {var_name})
                except AttributeError:
                    pass
        return input_map

    def _make_node(self, var_name, graph, *, formatting: str = "plain"):
        """Attaches the given variable to a graphviz Digraph"""
        v = self.model[var_name]

        # styling for node
        attrs = {}
        if isinstance(v, pm.model.ObservedRV):
            attrs["style"] = "filled"

        # make Data be roundtangle, instead of rectangle
        if isinstance(v, SharedVariable):
            attrs["style"] = "rounded, filled"

        # determine the shape for this node (default (Distribution) is ellipse)
        if v in self.model.potentials:
            attrs["shape"] = "octagon"
        elif isinstance(v, SharedVariable) or not hasattr(v, "distribution"):
            # shared variables and Deterministic represented by a box
            attrs["shape"] = "box"

        if v in self.model.potentials:
            label = f"{var_name}\n~\nPotential"
        elif isinstance(v, SharedVariable):
            label = f"{var_name}\n~\nData"
        else:
            label = v._str_repr(formatting=formatting).replace(" ~ ", "\n~\n")

        graph.node(var_name.replace(":", "&"), label, **attrs)

    def get_plates(self):
        """Rough but surprisingly accurate plate detection.

        Just groups by the shape of the underlying distribution.  Will be wrong
        if there are two plates with the same shape.

        Returns
        -------
        dict: str -> set[str]
        """
        plates = {}
        for var_name in self.var_names:
            v = self.model[var_name]
            if hasattr(v, "observations"):
                try:
                    # To get shape of _observed_ data container `pm.Data`
                    # (wrapper for theano.SharedVariable) we evaluate it.
                    shape = tuple(v.observations.shape.eval())
                except AttributeError:
                    shape = v.observations.shape
            elif hasattr(v, "dshape"):
                shape = v.dshape
            else:
                shape = v.tag.test_value.shape
            if shape == (1,):
                shape = tuple()
            if shape not in plates:
                plates[shape] = set()
            plates[shape].add(var_name)
        return plates

    def make_graph(self, formatting: str = "plain"):
        """Make graphviz Digraph of PyMC3 model

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
        for shape, var_names in self.get_plates().items():
            if isinstance(shape, SharedVariable):
                shape = shape.eval()
            label = " x ".join(map("{:,d}".format, shape))
            if label:
                # must be preceded by 'cluster' to get a box around it
                with graph.subgraph(name="cluster" + label) as sub:
                    for var_name in var_names:
                        self._make_node(var_name, sub, formatting=formatting)
                    # plate label goes bottom right
                    sub.attr(label=label, labeljust="r", labelloc="b", style="rounded")
            else:
                for var_name in var_names:
                    self._make_node(var_name, graph, formatting=formatting)

        for key, values in self.make_compute_graph().items():
            for value in values:
                graph.edge(value.replace(":", "&"), key.replace(":", "&"))
        return graph


def model_to_graphviz(model=None, *, formatting: str = "plain"):
    """Produce a graphviz Digraph from a PyMC3 model.

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
        one of { "plain", "plain_with_params" }
    """
    if not "plain" in formatting:
        raise ValueError(f"Unsupported formatting for graph nodes: '{formatting}'. See docstring.")
    model = pm.modelcontext(model)
    return ModelGraph(model).make_graph(formatting=formatting)
