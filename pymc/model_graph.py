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
import warnings

from collections import defaultdict
from typing import Dict, Iterable, List, NewType, Optional, Sequence, Set

from pytensor import function
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph import Apply
from pytensor.graph.basic import ancestors, walk
from pytensor.scalar.basic import Cast
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.shape import Shape
from pytensor.tensor.variable import TensorConstant, TensorVariable

import pymc as pm

from pymc.util import get_default_varnames, get_var_name

VarName = NewType("VarName", str)


__all__ = (
    "ModelGraph",
    "model_to_graphviz",
    "model_to_networkx",
)


def fast_eval(var):
    return function([], var, mode="FAST_COMPILE")()


class ModelGraph:
    def __init__(self, model):
        self.model = model
        self._all_var_names = get_default_varnames(self.model.named_vars, include_transformed=False)
        self.var_list = self.model.named_vars.values()

    def get_parent_names(self, var: TensorVariable) -> Set[VarName]:
        if var.owner is None or var.owner.inputs is None:
            return set()

        def _filter_non_parameter_inputs(var):
            node = var.owner
            if isinstance(node.op, Shape):
                # Don't show shape-related dependencies
                return []
            if isinstance(node.op, RandomVariable):
                # Filter out rng, dtype and size parameters or RandomVariable nodes
                return node.inputs[3:]
            else:
                # Otherwise return all inputs
                return node.inputs

        blockers = set(self.model.named_vars)

        def _expand(x):
            nonlocal blockers
            if x.name in blockers:
                return [x]
            if isinstance(x.owner, Apply):
                return reversed(_filter_non_parameter_inputs(x))
            return []

        parents = {
            VarName(get_var_name(x))
            for x in walk(nodes=_filter_non_parameter_inputs(var), expand=_expand)
            # Only consider nodes that are in the named model variables.
            if x.name and x.name in self._all_var_names
        }

        return parents

    def vars_to_plot(self, var_names: Optional[Iterable[VarName]] = None) -> List[VarName]:
        if var_names is None:
            return self._all_var_names

        selected_names = set(var_names)

        # .copy() because sets cannot change in size during iteration
        for var_name in selected_names.copy():
            if var_name not in self._all_var_names:
                raise ValueError(f"{var_name} is not in this model.")

            for model_var in self.var_list:
                if model_var in self.model.observed_RVs:
                    if self.model.rvs_to_values[model_var] == self.model[var_name]:
                        selected_names.add(model_var.name)

        selected_ancestors = set(
            filter(
                lambda rv: rv.name in self._all_var_names,
                list(ancestors([self.model[var_name] for var_name in selected_names])),
            )
        )

        for var in selected_ancestors.copy():
            if var in self.model.observed_RVs:
                selected_ancestors.add(self.model.rvs_to_values[var])

        # ordering of self._all_var_names is important
        return [VarName(var.name) for var in selected_ancestors]

    def make_compute_graph(
        self, var_names: Optional[Iterable[VarName]] = None
    ) -> Dict[VarName, Set[VarName]]:
        """Get map of var_name -> set(input var names) for the model"""
        input_map: Dict[VarName, Set[VarName]] = defaultdict(set)

        for var_name in self.vars_to_plot(var_names):
            var = self.model[var_name]
            parent_name = self.get_parent_names(var)
            input_map[var_name] = input_map[var_name].union(parent_name)

            if var in self.model.observed_RVs:
                obs_node = self.model.rvs_to_values[var]

                # loop created so that the elif block can go through this again
                # and remove any intermediate ops, notably dtype casting, to observations
                while True:
                    obs_name = obs_node.name
                    if obs_name and obs_name != var_name:
                        input_map[var_name] = input_map[var_name].difference({obs_name})
                        input_map[obs_name] = input_map[obs_name].union({var_name})
                        break
                    elif (
                        # for cases where observations are cast to a certain dtype
                        # see issue 5795: https://github.com/pymc-devs/pymc/issues/5795
                        obs_node.owner
                        and isinstance(obs_node.owner.op, Elemwise)
                        and isinstance(obs_node.owner.op.scalar_op, Cast)
                    ):
                        # we can retrieve the observation node by going up the graph
                        obs_node = obs_node.owner.inputs[0]
                    else:
                        break

        return input_map

    def _make_node(self, var_name, graph, *, nx=False, cluster=False, formatting: str = "plain"):
        """Attaches the given variable to a graphviz or networkx Digraph"""
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
        elif v in self.model.basic_RVs:
            shape = "ellipse"
            if v in self.model.observed_RVs:
                style = "filled"
            else:
                style = None
            symbol = v.owner.op.__class__.__name__
            if symbol.endswith("RV"):
                symbol = symbol[:-2]
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

        if cluster:
            kwargs["cluster"] = cluster

        if nx:
            graph.add_node(var_name.replace(":", "&"), **kwargs)
        else:
            graph.node(var_name.replace(":", "&"), **kwargs)

    def get_plates(self, var_names: Optional[Iterable[VarName]] = None) -> Dict[str, Set[VarName]]:
        """Rough but surprisingly accurate plate detection.

        Just groups by the shape of the underlying distribution.  Will be wrong
        if there are two plates with the same shape.

        Returns
        -------
        dict
            Maps plate labels to the set of ``VarName``s inside the plate.
        """
        plates = defaultdict(set)

        # TODO: Evaluate all RV shapes and dim_length at once.
        #       This should help to find discrepancies, and
        #       avoids unnecessary function compiles for deetermining labels.

        for var_name in self.vars_to_plot(var_names):
            v = self.model[var_name]
            shape: Sequence[int] = fast_eval(v.shape)
            dim_labels = []
            if var_name in self.model.named_vars_to_dims:
                # The RV is associated with `dims` information.
                for d, dname in enumerate(self.model.named_vars_to_dims[var_name]):
                    if dname is None:
                        # Unnamed dimension in a `dims` tuple!
                        dlen = shape[d]
                        dname = f"{var_name}_dim{d}"
                    else:
                        dlen = fast_eval(self.model.dim_lengths[dname])
                    dim_labels.append(f"{dname} ({dlen})")
                plate_label = " x ".join(dim_labels)
            else:
                # The RV has no `dims` information.
                dim_labels = [str(x) for x in shape]
                plate_label = " x ".join(map(str, shape))
            plates[plate_label].add(var_name)

        return dict(plates)

    def make_graph(self, var_names: Optional[Iterable[VarName]] = None, formatting: str = "plain"):
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
        for plate_label, all_var_names in self.get_plates(var_names).items():
            if plate_label:
                # must be preceded by 'cluster' to get a box around it
                with graph.subgraph(name="cluster" + plate_label) as sub:
                    for var_name in all_var_names:
                        self._make_node(var_name, sub, formatting=formatting)
                    # plate label goes bottom right
                    sub.attr(label=plate_label, labeljust="r", labelloc="b", style="rounded")
            else:
                for var_name in all_var_names:
                    self._make_node(var_name, graph, formatting=formatting)

        for child, parents in self.make_compute_graph(var_names=var_names).items():
            # parents is a set of rv names that precede child rv nodes
            for parent in parents:
                graph.edge(parent.replace(":", "&"), child.replace(":", "&"))

        return graph

    def make_networkx(
        self, var_names: Optional[Iterable[VarName]] = None, formatting: str = "plain"
    ):
        """Make networkx Digraph of PyMC model

        Returns
        -------
        networkx.Digraph
        """
        try:
            import networkx
        except ImportError:
            raise ImportError(
                "This function requires the python library networkx, along with binaries. "
                "The easiest way to install all of this is by running\n\n"
                "\tconda install networkx"
            )
        graphnetwork = networkx.DiGraph(name=self.model.name)
        for plate_label, all_var_names in self.get_plates(var_names).items():
            if plate_label:
                # # must be preceded by 'cluster' to get a box around it

                subgraphnetwork = networkx.DiGraph(name="cluster" + plate_label, label=plate_label)

                for var_name in all_var_names:
                    self._make_node(
                        var_name,
                        subgraphnetwork,
                        nx=True,
                        cluster="cluster" + plate_label,
                        formatting=formatting,
                    )
                for sgn in subgraphnetwork.nodes:
                    networkx.set_node_attributes(
                        subgraphnetwork,
                        {sgn: {"labeljust": "r", "labelloc": "b", "style": "rounded"}},
                    )
                node_data = {
                    e[0]: e[1]
                    for e in graphnetwork.nodes(data=True) & subgraphnetwork.nodes(data=True)
                }

                graphnetwork = networkx.compose(graphnetwork, subgraphnetwork)
                networkx.set_node_attributes(graphnetwork, node_data)
                graphnetwork.graph["name"] = self.model.name
            else:
                for var_name in all_var_names:
                    self._make_node(var_name, graphnetwork, nx=True, formatting=formatting)

        for child, parents in self.make_compute_graph(var_names=var_names).items():
            # parents is a set of rv names that precede child rv nodes
            for parent in parents:
                graphnetwork.add_edge(parent.replace(":", "&"), child.replace(":", "&"))
        return graphnetwork


def model_to_networkx(
    model=None,
    *,
    var_names: Optional[Iterable[VarName]] = None,
    formatting: str = "plain",
):
    """Produce a networkx Digraph from a PyMC model.

    Requires networkx, which may be installed most easily with::

        conda install networkx

    Alternatively, you may install using pip with::

        pip install networkx

    See https://networkx.org/documentation/stable/ for more information.

    Parameters
    ----------
    model : Model
        The model to plot. Not required when called from inside a modelcontext.
    var_names : iterable of str, optional
        Subset of variables to be plotted that identify a subgraph with respect to the entire model graph
    formatting : str, optional
        one of { "plain" }

    Examples
    --------
    How to plot the graph of the model.

    .. code-block:: python

        import numpy as np
        from pymc import HalfCauchy, Model, Normal, model_to_networkx

        J = 8
        y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
        sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

        with Model() as schools:

            eta = Normal("eta", 0, 1, shape=J)
            mu = Normal("mu", 0, sigma=1e6)
            tau = HalfCauchy("tau", 25)

            theta = mu + tau * eta

            obs = Normal("obs", theta, sigma=sigma, observed=y)

        model_to_networkx(schools)
    """
    if "plain" not in formatting:
        raise ValueError(f"Unsupported formatting for graph nodes: '{formatting}'. See docstring.")

    if formatting != "plain":
        warnings.warn(
            "Formattings other than 'plain' are currently not supported.",
            UserWarning,
            stacklevel=2,
        )
    model = pm.modelcontext(model)
    return ModelGraph(model).make_networkx(var_names=var_names, formatting=formatting)


def model_to_graphviz(
    model=None,
    *,
    var_names: Optional[Iterable[VarName]] = None,
    formatting: str = "plain",
):
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
    var_names : iterable of variable names, optional
        Subset of variables to be plotted that identify a subgraph with respect to the entire model graph
    formatting : str, optional
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
    if "plain" not in formatting:
        raise ValueError(f"Unsupported formatting for graph nodes: '{formatting}'. See docstring.")
    if formatting != "plain":
        warnings.warn(
            "Formattings other than 'plain' are currently not supported.",
            UserWarning,
            stacklevel=2,
        )
    model = pm.modelcontext(model)
    return ModelGraph(model).make_graph(var_names=var_names, formatting=formatting)
